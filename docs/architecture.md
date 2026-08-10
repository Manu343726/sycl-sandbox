# Architecture

## Project overview

`sycl-sandbox` is an interactive GPU/CPU rendering sandbox with hot-reloadable
SYCL kernels.  It renders scenes using procedurally generated geometry, accumulates
samples progressively, and provides a Dear ImGui UI for scene selection, parameter
tweaking, and camera control.

The application is split into two runtime components:

1. **Host** — the `sycl-sandbox` executable (OpenGL + ImGui + SYCL runtime).
2. **Kernels** — shared libraries (`.so`) loaded at runtime via `dlopen`.
   Each kernel defines a scene and is hot-reloaded on source change.

## Host side (`src/`)

### Main loop (`src/main.cpp`)

Each frame:
1. **Poll events** — GLFW input, inotify source watcher.
2. **Detect window resize** — recreate render texture + SYCL buffers.
3. **Check source changes** — hot-reload any modified kernel `.cpp`/`.h`.
4. **ImGui frame** — dockspace, background render texture, control panels.
5. **Camera controls** — 3D orbit/WASD (raytracer).
   Modifier scroll: Ctrl = aperture, Ctrl+Shift = FOV, Ctrl+Alt = roll.
6. **Render** — fill an `rt::Context` (params lookup, scene view, profiler
   ring, trace counters, framebuffer) and call `call_kernel_entry(handle, ctx)`.
   The kernel's single entry point is called once per frame; the kernel
   enqueues its device work and returns without waiting (enqueue-only
   contract — the native/software backend executes synchronously).
7. **Tonemap + display** — wait for the frame, then copy the accumulation
   buffer from device, tonemap (Reinhard + gamma), upload to OpenGL texture.
8. **Render ImGui** — draw UI over the scene.

### Kernel library (`src/kernel/library.cpp`)

Each kernel is a shared library built by `acpp` (AdaptiveCpp).  On every load:
1. Copy the `.so` to a versioned path (e.g. `libraytracer.v3.so`).
2. `dlopen` the copy (guarantees a fresh handle from the dynamic linker).
3. `dlsym` `kernel_entry` — the single kernel ABI entry point
   (`extern "C" void kernel_entry(const rt::Context *)`).  Kernels that
   don't export it are refused.
4. Scan the kernel source directory + shared headers for profiler-macro
   string literals (hot-reload: a rebuilt .so can contain newly named
   zones).
5. Store the active handle; keep the old handle alive (never `dlclose`).

There is no metadata query, no setup phase, no op dispatch: kernel
metadata (name, max_spp, whether it consumes a scene) lives host-side in
the scene data, and every resource the kernel needs arrives through the
`rt::Context` on each call.  The kernel keeps no state between calls.

Hot-reload is driven by `inotify` watching kernel source directories.
When a source file changes, `cmake --build --target <kernel>` is run,
then the `.so` is reloaded and the kernel state is re-initialised with
the current params (ordered reload: shutdown → rebuild → load → init).

### Scene registry (`src/scene/registry.cpp`)

YAML files in `scenes/` define named scenes:

```yaml
name: "One Weekend — Random Spheres"
kernel: raytracer
max_spp: 1024
scene_type: 3d

params:
  num_spheres:
    type: int
    description: "Number of random small spheres scattered on the ground"
    default: 22
    range: [0, 500]
```

Objects can load triangle meshes from STL files (ASCII or binary,
auto-detected) — `file` is resolved relative to the scene YAML, with
optional `position` / `rotation` (Euler degrees, Z → Y → X) / `scale`
placement (see `scenes/mesh_demo.yaml`).  The mesh's triangles land in
the scene's per-type triangle array as one `Mesh` handle
(`include/sycl-sandbox/rt/hittables/mesh.h`, `SceneBuilder::add_mesh()`),
loaded by `include/sycl-sandbox/stl_loader.h` (host-only).  Zero-area
(degenerate) triangles are filtered out at load time and the `Triangle`
constructor yields a clean zero normal for them — a NaN normal from a
degenerate triangle would otherwise leak through on GPU fast-math builds
and progressively blacken accumulated pixels (see `docs/raytracing.md`).
Large meshes also get a **per-mesh BVH**: `SceneBuilder::build_mesh_bvhs()`
builds one flat binary BVH per mesh (leaves reference absolute indices
into `SceneView::triangles`) appended into a single
`SceneView::mesh_bvh_nodes` buffer, and each `Mesh` stores its tree's
`bvh_root`; `Mesh::hit()` traverses it instead of linearly scanning
(see `docs/raytracing.md` → "Per-mesh BVH").

`SceneDescriptor::build_layout()` fills the param buffer from YAML-declared
defaults.  Camera parameters are auto-generated from `scene_type: 3d|2d`
and are NOT declared in YAML.  Values are always stored as `float` (ints and
bools are cast to `float` when writing) so the kernel reads them uniformly.

Standard params auto-injected into every scene: render (`spp_frame`,
`max_bounces`, `tick`, `time`, `transparent_backfaces` [X-ray mode: rays
pass through back faces, off by default]), display pipeline
(`tonemap_enabled` [default off], `tonemap_operator`, `tonemap_exposure`,
`tonemap_gamma`), and debug switches (`debug_colorchecker`,
`debug_colorchecker_raw`).  Scenes can set a
default for any standard param by declaring the same name in YAML — the
loader merges the override into the auto-generated descriptor instead of
duplicating or skipping it (see `src/scene/loader.cpp`).

### Parameter UI (`src/ui/params/controls.cpp`)

Dynamically generates ImGui controls from `ParamDescriptor[]`.  Supported types:
`FLOAT`, `INT`, `BOOL`, `COLOR_RGB`, `COLOR_RGBA`, `VEC3`, `ENUM`.  Int and
bool values are read/written as `float` in the buffer (cast on access) so
that the kernel's `(int)p[idx]` pattern works correctly.

### System metrics (`src/ui/metrics/`)

"System Metrics" window with rolling graphs of **CPU**, **system RAM**, and
**GPU** usage (120 samples ≈ 60 s at the internal 2 Hz sample rate; sampling
is throttled to `SAMPLE_PERIOD = 500 ms` and called every frame from
`frame_loop.h`).

Platform backends (see `system_metrics.cpp`):

| Metric | Linux | Windows |
|--------|-------|---------|
| CPU    | `/proc/stat` deltas | `GetSystemTimes` deltas |
| RAM    | `/proc/meminfo` (MemTotal / MemAvailable) | `GlobalMemoryStatusEx` |
| GPU    | NVIDIA NVML (`libnvidia-ml.so.1`, dlopen) or amdgpu sysfs (`gpu_busy_percent`, `mem_info_vram_*`) | NVIDIA NVML (`nvml.dll`, LoadLibrary) |

NVML is resolved at runtime, so the app links nothing GPU-specific and runs
without an NVIDIA GPU — the GPU section then shows "unavailable" (and falls
back to amdgpu sysfs on Linux). The panel is toggled from the Controls
window ("Show System Metrics").

### Statistics

Every scene gets a set of auto-generated "standard" statistics
(`auto_standard_stats()` in `src/scene/loader.cpp`) plus any scene-specific
ones declared in YAML under `statistics:`.  Standard stats:

| Stat | Meaning | Written by |
|------|---------|------------|
| `fps`, `frame_time_ms`, `spp`, `pixel_count`, `device_memory_mb`, `host_memory_mb` | UI/host metrics | UI thread (`update_standard_stats()`) |
| `num_objects`, `num_bvh_nodes`, `num_lights` | Scene composition | Kernel `kernel_entry()` (host-side, via `ctx.stats`) |
| `num_hits` | Closest hits found this frame (all bounces) | Device-side `TraceCounters` (`ctx.trace_counters`) |
| `num_bvh_hits` | Of those, found via BVH traversal (0 when the scene has no BVH) | same |

Data flow: the render thread writes the runtime-owned stat block
(`KernelRuntime::stat_buffer_`), publishes it through the
`PublishedStats` seqlock after each frame; the UI thread copies it into the
scene descriptor's buffer for the stats panel.  The scene descriptor's own
buffer stays UI-thread-private.

Scene-composition stats (`num_objects`, `num_bvh_nodes`, `num_lights`)
are published by the kernel host-side via `ctx.stats` (an `rt::StatWriter`
over the runtime-owned stat block) right after enqueueing — they are
scene properties, not device outputs.

`num_hits` / `num_bvh_hits` are per-frame values counted BY the device:
`rt::trace()` increments the `rt::TraceCounters` scratch buffer
(`ctx.trace_counters` — a device buffer the host zeroes each frame before
enqueueing) atomically (`sycl::atomic_ref` on device; plain increments in
native software mode) once per closest hit — including all bounces.  With
the BVH active every hit is a BVH hit, so `num_hits == num_bvh_hits`; the
linear-scan fallback counts `num_bvh_hits == 0`.

### Scene debug window (`src/ui/scene_debug/`)

The "Scene Debug" window renders a 3D view of the current YAML scene
completely independently of the raytracing kernel.  It has its own orbit
camera whose controls mirror the main Viewport's 3D orbit bindings
(LMB pan, MMB orbit, wheel zoom, arrows orbit, Shift+arrows pan,
WASD/QE move the target; defaults to a view from above so
the scene is right-side up — the world is Y-up, same as the raytracer)
and shows floor/grid/objects/wireframe/AABB/frustum/camera-indicator
overlays.  RMB is reserved for the ray/pixel pick.

**Architecture** — a background render thread owns a hidden GLFW window that
shares the main window's GL context:

- `renderer.h` / `renderer.cpp` — `SceneDebugRenderer`.  `init()` releases
  the main context (`glfwMakeContextCurrent(nullptr)` — GLFW requires the
  share context to not be current anywhere during creation), creates the
  hidden 3.3-core window, restores the main context, and spawns the render
  thread.  The thread renders the scene into one of three FBO-backed
  textures (triple buffering) with plain GL (VAO/VBO + a tiny shader, all
  functions via `src/gl_loader.h`); after `glFinish()` the slot is marked
  READY.  The UI thread calls `present()` each frame, which takes the
  newest READY slot and hands the texture to `ImGui::Image`.
- `panel.h` / `panel.cpp` — ImGui side: toolbar checkboxes (Floor, Grid,
  Objects, Wireframe, AABBs, Frustum, Camera, Visibility), stats line,
  camera input, and the `ImGui::Image` present.  Shared state between UI
  and render thread lives in `SceneDebugRenderer::State` (mutex-protected);
  the window collapsed state pauses the render thread.
- `src/scene/host_scene.h` — `HostScene::debug_scene` holds a
  `std::shared_ptr<SceneDebugScene>`: a host copy of the generated scene
  (`rt::SceneData` + `rt::SceneView` + `version`).  `rebuild_yaml_scene()`
  builds it with `SceneBuilder::build()` against a null-queue `rt::Runtime`
  (heap allocs, `memcpy` copies — the raytracing runtime is fully
  host-compilable).  The `shared_ptr` handoff keeps the scene alive while
  the render thread is mid-frame with it; the old instance is freed when
  the last reference drops.
- The visibility overlay traces real rays against the host copy
  (`rt::bvh_hit()` when a BVH exists, linear `handle_hit` fallback), so
  occlusion is exact — the same code the kernel runs on device.
- **Single-ray inspector — the debugger runs the REAL path tracer.**
  The whole pipeline — `rt::trace()` (include/sycl-sandbox/rt/trace.h),
  `rt::bvh_hit()`, every primitive's `hit()`, every material's
  `scatter()`/`emit()` and every texture `sample()` — takes a
  `const rt::Context &ctx` (include/sycl-sandbox/context.h): the same
  per-call kernel context carries the device profiler ring, the
  per-work-item id, the per-frame hit counters and the debug
  *collector*, so the pipeline's profiler zones, hit counters and debug
  hooks all read from one object instead of a growing list of
  parameters.  The collector (`rt::TraceCollector`, rt/collector.h) is
  an optional step/event ring: an inactive collector (the default
  `rt::Context{}`) makes every hook a no-op the compiler eliminates —
  zero cost on the kernel path.  The scene-debug view
  (`trace_ray_debug()`, src/ui/scene_debug/ray_trace.h) arms a
  `TraceCollector` over host stack rings and runs the real `trace()`,
  which records every bounce (ray, hit record, emit, scatter,
  attenuation, running throughput) and — via the same collector
  threaded through `rt::bvh_hit()`, every primitive's `hit()`, every
  material's `scatter()`/`emit()` and every texture `sample()` —
  appends deep metadata (BVH nodes entered — both the scene BVH and
  per-mesh BVHs —, hit tests performed,
  scatter/emit evaluations, texture samples) to each step.  There is
  exactly ONE path-tracing implementation — the debugger cannot diverge
  from the renderer.  Per-hit output colours are recovered after the
  trace by back-propagation (emit + attenuation × continuation, with
  the kernel's exact sky fold on escape/bounce-limit), so
  `steps[0].color` is bit-identical to `rt::trace()`'s return value
  (verified by a headless harness).
- **The debug trace mirrors the kernel's render parameters.**  The UI
  thread reads `background`, `max_bounces`, `transparent_backfaces` and
  the tone-map stage params from the scene descriptor and passes them
  as `SceneRenderParams` to the renderer (frame_loop.h →
  panel.h → renderer State).  The debug ray is traced with the same
  bounce depth, backface mode and background, and the overlay + swatch
  colours are displayed through the same tone-map operator + exposure +
  gamma as the framebuffer (`scene_debug_display_color`, using the
  shared `tonemap::apply_operator` from include/sycl-sandbox/tonemap_ops.h
  that tonemap.h also uses) — the traced ray shows exactly what the
  rendered image shows, not raw linear radiance.
- The frustum overlay's near plane is the scene camera's actual
  framebuffer: the raytracer projects onto an image plane 1 unit in front
  of the eye (`rt::lookat()`'s default `focus_dist`), so the rectangle
  (`lower_left`/`horizontal`/`vertical` from `rt::lookat`, using the
  render target's aspect) shows exactly where the rendered image lives in
  the scene — drawn as a translucent fill + bright outline.

**Image orientation conventions** — the accum buffer stores row 0 at the
BOTTOM of the image plane (the raytracer maps `v = (y+.5)/h` with `v`
pointing world-up; 2D kernels map row 0 to the bottom of their plane).
The tone-mapper Y-flips (`dy = h-1-y`) so the display texture is
top-down for ImGui.  Kernels must therefore treat row 0 as the image
bottom; the 3D raymarching kernels got this wrong historically
(`py = 1-2(y+.5)/h`) and were displayed upside down — the convention is
`py = 2(y+.5)/h-1`.

## Kernel ABI — single entry point

Every kernel is a shared library (`kernels/<name>/kernel.cpp`) that exposes
exactly ONE `extern "C"` function (include/sycl-sandbox/context.h):

```cpp
extern "C" void kernel_entry(const rt::Context *ctx);
```

There is no op dispatch, no setup phase, no metadata query, no teardown:
the caller owns every resource and passes it in the `rt::Context` on every
call.  The kernel reads the context, enqueues its device work, publishes
host-side stats, and returns.  It keeps NO state between calls — local
variables only (profiler zones record through `ctx.prof`), so unloading
the .so needs no teardown.  Kernel metadata (name, max_spp, whether it
consumes a scene) lives host-side in the scene data, not in the kernel
binary.

### `rt::Context` — per-call data

```cpp
struct Context {
    Runtime *runtime = nullptr;          // queue + USM allocators (never null)
    const ParamLookup *params = nullptr; // host-visible param snapshot
    const StatWriter *stats = nullptr;   // host-side stat block writer
    const SceneView *scene = nullptr;    // device scene view for this frame
    profiler::DeviceRing prof = {};      // device profiler ring (POD handle)
    TraceCounters *trace_counters = nullptr;  // per-frame hit counters
    TraceCollector collector = {};       // debug step/event ring (inactive by default)
    uint32_t linear_id = 0;              // work-item id for profiler decimation
    int32_t width = 0, height = 0;       // framebuffer dimensions
    float *accum = nullptr;              // persistent float4[width*height]
    uint32_t spp_frame = 0;              // samples to add in this call
    uint32_t spp_total = 0;              // samples already accumulated
    uint64_t frame_index = 0;            // monotone frame counter
};
```

The context is POD and SYCL-free — the queue lives in `rt::Runtime`, not
here, so the .so ABI is a plain pointer to a POD struct.  The render loop
fills one `rt::Context` per frame (src/render_loop.h) and forwards it to
`call_kernel_entry()`.

**Enqueue-only contract.**  The kernel must submit its device work to the
queue (via `ctx.runtime`) and return WITHOUT waiting for completion.  The
host chains the tone-map and display publication after it on the same
in-order queue.  (Native/software mode executes synchronously — the
contract is trivially met.)

### Host-side dispatch (`src/kernel/dispatch.h`)

- `resolve_kernel_entry(handle)` — `dlsym("kernel_entry")`
- `call_kernel_entry(handle, ctx)` — forward the fully-populated
  `rt::Context` to the kernel (no-op when the .so doesn't export the
  entry — the loader refuses such kernels anyway)

### Profiler buffer wiring

The device profiler ring is a HOST-owned device buffer pair
(`AppState::d_ring_header_` / `d_ring_records_`, capacity 256 K records).
Each frame the render thread builds a `profiler::DeviceRing` POD handle
over them (`AppState::device_ring(pixels)`, include/sycl-sandbox/profiler.h)
and passes it to the kernel in `ctx.prof`.  Kernel-side `PROFILER_ZONE("name")`
and `PROFILER_FUNCTION()` records sit in that ring; records carry only a
32-bit FNV-1a zone hash + lane id (no strings on device), and an inactive
ring (null header) makes every zone a no-op.  After the frame the host
calls `Bridge::submit_device_ring()` to drain the ring into the Tracy
client.

There is no per-frame reset: the ring is a circular buffer — the host
reads the window `[write_pos - capacity, write_pos)` after each frame,
and the kernel just keeps pushing into it.

### Profiler compile-time toggle

The whole profiler stack (device ring, host zones, Tracy bridge
submission) is gated on the `SANDBOX_ENABLE_PROFILER` CMake option
(default ON).  When OFF, `profiler.h`'s master gate
(`#if !defined(SANDBOX_ENABLE_PROFILER)`) expands every `PROFILER_*`
macro to nothing and removes the zone-name extractor from the build, so
both the app and the on-the-fly-built kernels get ZERO profiler
overhead.

The **Controls panel → Profiler checkbox** toggles this at runtime: it
reconfigures the kernel build directory via
`cmake -DSANDBOX_ENABLE_PROFILER=ON|OFF <build_dir>` and triggers a
hot-reload of the active kernel.  The app itself keeps its own compiled
flag (`Runtime::profiler_enabled()`, cached into
`AppState::profiler_enabled`) so it knows which build the checkbox
requested.  `SANDBOX_ENABLE_TRACY` is independent — it only controls the
Tracy client/GUI; with the profiler disabled there is simply nothing to
forward.

### Available kernels

There is one kernel library today — `raytracer` (kernels/raytracer/), the
SYCL path tracer built from the shared `include/sycl-sandbox/rt/` library.
Each YAML scene in `scenes/` binds to it and declares its own `max_spp`
and params (`src/scene/registry.cpp`), so switching scenes never reloads
the .so — the same kernel binary renders every scene.

Each kernel is built twice from the same sources (kernels/CMakeLists.txt):

| Target            | Built with | Backend |
|-------------------|------------|---------|
| `<name>.so`       | `add_sycl_to_target()` (SYCL) | GPU / SYCL CPU backends |
| `<name>_native.so`| plain C++ (`KERNEL_NATIVE`) | Software (CPU, no SYCL runtime) |

## Build system

Dependencies are managed by Conan (`conanfile.py`).  Two build directories
are maintained via separate CMake presets:

```bash
build/        # cmake --preset conan-release    (Release, -O3)
build_debug/  # cmake --preset conan-debug      (Debug, -O0 -g)
```

The `cmake --preset conan-<config>` load the Conan-generated toolchain, which
sets up `find_package` paths for all dependencies (GLFW, ImGui, yaml-cpp, spdlog,
glm, AdaptiveCpp, fmt).

Kernels are built via `add_sycl_to_target()` (AdaptiveCpp's CMake integration).
The `generic` backend target is used, which compiles to OpenMP CPU code —
fully portable, no GPU-specific flags needed.

### CUDA libdevice note

AdaptiveCpp's SSCP JIT needs `libdevice.10.bc` to compile `generic` kernels to
PTX for GPU execution.  The file ships with CUDA at
`/opt/cuda/nvvm/libdevice/libdevice.10.bc` and must be symlinked to
`/opt/cuda/lib64/` if the runtime can't find it:

```bash
sudo ln -sf /opt/cuda/nvvm/libdevice/libdevice.10.bc /opt/cuda/lib64/libdevice.10.bc
```

### Tracy profiler integration

GPU kernel timeline tracing and the full Tracy profiler UI are available
behind the `SANDBOX_ENABLE_TRACY` CMake option (on by default).  In VS Code
the option is passed via `.vscode/settings.json` → `cmake.configureArgs`
(CMake Tools reconfigures automatically), or explicitly on the command line:

```bash
cmake --preset conan-debug -B build_debug -DSANDBOX_ENABLE_TRACY=ON
```

When enabled, Tracy (master) is fetched via `FetchContent` and provides:

- **`TracyClient`** — the on-demand client, linked into `sycl-sandbox`.
  It listens on `127.0.0.1:8086` and records nothing until a profiler
  connects (on-demand mode).
- **`tracy-profiler`** — the standalone profiler GUI, built by Tracy's
  own CMake (`add_subdirectory(${tracy_SOURCE_DIR}/profiler ...)` from
  the top-level `CMakeLists.txt`) to
  `build_debug/profiler/bin/tracy-profiler` (VS Code tasks
  "build tracy-server" / "launch tracy-server").  Nothing is embedded
  in the sandbox: the app only ships the client, and the profiler is a
  separate process that connects to it.

**Bridge** (`src/tracy/tracy_bridge.{h,cpp}`).  The kernel profiler's
device records (read back from the `ctx.prof` ring after each frame,
timestamps already in the host rdtsc cycle domain) are forwarded to the
client with the serial GPU C API (`TracyC.h`
`___tracy_emit_gpu_*_serial`) on a "Custom" GPU context (type 7).  Like
the official Rocprof client, the context period is `1.0f` and a
`GpuCalibration` event is emitted every frame — the server maps
gpu→cpu time from `cpuDelta` (cpu time elapsed since the previous
calibration).  Zone names are cached as source locations with stable
FNV-1a colors.  All emissions are connection-gated, so `submit()` runs
from the render thread every frame at no cost while nothing is
connected.

**Device zones & profiler name extraction.**  Every raytracing stage is
instrumented with a device zone (`PROFILER_ZONE` in
`include/sycl-sandbox/profiler.h`, records written into the
`ctx.prof` ring with per-pixel decimation via `ctx.linear_id`):
`trace_px`/`sample_px` per pixel, `trace`/`bounce`/`linear_scan` per path,
`bvh_query` per traversal, one zone per primitive `hit()`, material
`scatter()`/`emit()`, and texture `sample()`.  The ring rides into the
kernel inside `rt::Context` — the same per-call context as the trace
counters and the debug collector.

Device records carry only a 32-bit FNV-1a hash (no strings on device).
The host resolves hashes to display names through a two-tier lookup:

1. **Constexpr perfect-hash table** — `tools/zone_names_extractor` (libclang
   C API) parses all kernel + host source TUs at build time, extracts every
   profiler-macro string literal (`PROFILER_ZONE`, `PROFILER_PLOT`,
   `PROFILER_MSG`, `PROFILER_DEVICE_ZONE`), finds a collision-free hash
   table (tries power-of-2 with XOR-fold mixing and prime-modulo
   strategies, picks the smallest), and emits `zone_names.generated.h`
   with `constexpr std::array<std::string_view, N>` and an O(1)
   `lookup_profiler_name(hash)` function.  Built as a CMake custom-command
   dependency of the app target (`SANDBOX_HAVE_GENERATED_ZONE_NAMES=1`).

2. **Runtime fallback** — `zone_names.h` scans source directories with
   regex at startup (when libclang is unavailable) and on every kernel
   load (hot-reload: a rebuilt .so can introduce new names without
   rebuilding the app binary).  These go into a mutex-guarded
   `unordered_map`.

`lookup_zone_name(hash)` checks the constexpr table first (lock-free),
then the runtime map.  Unknown hashes get `"zone_<hash>"` as a fallback.

**Host zones** (`PROFILER_ZONE` / `PROFILER_FUNCTION`) embed their name
directly in the 64-byte `Record` (no hash lookup needed).
`PROFILER_FUNCTION()` uses `__PRETTY_FUNCTION__` for fully qualified
signatures (e.g. `KernelLibrary::load(std::string const&)`).

**Launcher** (`src/tracy/tracy_launcher.{h,cpp}`).  Controls →
"Launch Tracy Profiler" forks the standalone `tracy-profiler`
(`<build_dir>/profiler/bin/tracy-profiler`, found next to the app's own
build directory), detached from the terminal.  The profiler then
connects to the in-process client and the capture runs while it is
connected; closing the profiler (or exiting the sandbox) ends the
capture.  The same flow is available from the VS Code task "launch
tracy-server".

**Build details.**  The profiler app's native file dialog (nfd) and
Wayland backend are disabled (`NO_FILESELECTOR`, `LEGACY` — X11);
`TRACY_NO_FILESELECTOR` is also defined on `tracy-profiler` because
Tracy's CMake never translates the option into a compile definition.
Fonts / manual / achievements data are generated by Tracy's own `embed`
rules for the `tracy-profiler` target — no app-side CMake duplicates
them.

Without `SANDBOX_ENABLE_TRACY`, `src/tracy/tracy_bridge.h` and
`src/tracy/tracy_launcher.h` provide compile-time no-op stubs, so no
Tracy code is built.
