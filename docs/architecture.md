# Architecture

## Project overview

`sycl-sandbox` is an interactive GPU/CPU rendering sandbox with hot-reloadable
SYCL kernels.  It renders scenes using procedurally generated geometry, accumulates
samples progressively, and provides a Dear ImGui UI for scene selection, parameter
tweaking, and camera control.

The application is split into two runtime components:

1. **Host** — the `sandbox` executable (OpenGL + ImGui + SYCL runtime).
2. **Kernels** — shared libraries (`.so`) loaded at runtime via `dlopen`.
   Each kernel defines a scene and is hot-reloaded on source change.

## Host side (`src/`)

### Main loop (`src/main.cpp`)

Each frame:
1. **Poll events** — GLFW input, inotify source watcher.
2. **Detect window resize** — recreate render texture + SYCL buffers.
3. **Check source changes** — hot-reload any modified kernel `.cpp`/`.h`.
4. **ImGui frame** — dockspace, background render texture, control panels.
5. **Camera controls** — 2D pan/zoom (Mandelbrot) or 3D orbit/WASD (raytracers).
   Modifier scroll: Ctrl = aperture, Ctrl+Shift = FOV, Ctrl+Alt = roll.
6. **Render** — call kernel's `render_kernel()` via `dlsym`.
7. **Tonemap + display** — copy accumulation buffer from device, tonemap
   (Reinhard + gamma), upload to OpenGL texture.
8. **Render ImGui** — draw UI over the scene.

### Kernel library (`src/kernel_library.cpp`)

Each kernel is a shared library built by `acpp` (AdaptiveCpp).  On every load:
1. Copy the `.so` to a versioned path (e.g. `libone_weekend.v3.so`).
2. `dlopen` the copy (guarantees a fresh handle from the dynamic linker).
3. `dlsym` `get_kernel_desc()` to read parameter metadata.
4. Compute per-param buffer offsets from the metadata.
5. Store the active handle; keep the old handle alive (never `dlclose`).

Hot-reload is driven by `inotify` watching kernel source directories.
When a source file changes, `cmake --build --target <kernel>` is run,
then the `.so` is reloaded and `init_kernel()` is called with current params.

### Scene registry (`src/scene_registry.cpp`)

YAML files in `scenes/` define named scenes:

```yaml
name: "Cornell Box"
kernel: cornell_box
params:
  spp_frame: 1
  light_color: [1.0, 1.0, 1.0]
```

`apply_params()` fills a kernel's params buffer from `ParamMeta` defaults,
then overlays YAML values.  Values are always stored as `float` (ints and
bools are cast to `float` when writing) so the kernel reads them uniformly.

### Parameter UI (`src/param_ui.cpp`)

Dynamically generates ImGui controls from `ParamMeta[]`.  Supported types:
`float`, `int`, `bool`, `COLOR_RGB`, `COLOR_RGBA`, `VEC3`, `ENUM`.  Int and
bool values are read/written as `float` in the buffer (cast on access) so
that the kernel's `(int)p[idx]` pattern works correctly.

## Kernel API

Every kernel is a shared library (`kernels/<name>/kernel.cpp`) that exposes
four `extern "C"` functions.  The host calls them via `dlsym`:

```cpp
extern "C" KernelDesc* get_kernel_desc();
extern "C" void        init_kernel(sycl::queue*, int w, int h,
                                   const void* params, size_t params_size);
extern "C" void        render_kernel(sycl::queue*, int w, int h,
                                     const void* params,
                                     void* accum_buffer, int sample_index);
extern "C" void        shutdown_kernel(sycl::queue*);
```

### `get_kernel_desc()`

Return a `KernelDesc` containing:
- `name`, `description` — human-readable identifiers.
- `param_count`, `params` — the `ParamMeta[]` array describing every parameter.
- `params_buffer_size` — computed by summing `param_buffer_size()` for each param.
- `max_spp` — how many samples per pixel this kernel benefits from.
  `1` means single-frame (Mandelbrot), `4096` means progressive (raytracers).
- `source_count`, `sources` — source files to watch for hot-reload.

The host uses this metadata to build the ImGui parameter controls and validate
the params buffer size.  This function is the only way the host knows what
parameters a kernel expects.

### `init_kernel()`

Called when the scene is selected, after params have been uploaded to device
memory.  Typical actions:
1. Read kernel-specific params from the float buffer.
2. Build scene geometry on the host (allocate with `new`).
3. Upload geometry to device via `sycl::malloc_device` + `memcpy`.
4. Store device pointers in static globals for later use in `render_kernel()`.

The params buffer (`d_params`) is allocated with `sycl::malloc_host` —
accessible from both host and device — so `init_kernel()` reads params
directly on the host without a device transfer.

### `render_kernel()`

Called every frame.  The kernel reads its params (SPP, bounces, camera, etc.),
computes one frame of samples, and accumulates into `accum_buffer` (a
`float4` array: RGBA, one element per pixel).

For raytracers this is typically a one-liner delegating to the shared library:

```cpp
extern "C" void render_kernel(sycl::queue* queue, int w, int h,
                               const void* params, void* accum, int si) {
    rt::render_main(queue, w, h, (const float*)params, (float*)accum, si,
                    g_scene_objects, g_num_objects,
                    [](const rt::Ray& ray) -> rt::float3 {
                        return background_colour(ray);
                    });
}
```

The kernel only provides its scene (`Object[]`) and a background function.
The shared library handles camera setup, ray generation, path tracing,
and accumulation (see [docs/raytracing.md](raytracing.md)).

For non-raytracing kernels (e.g. Mandelbrot), `render_kernel()` implements
its own computation directly.

### `shutdown_kernel()`

Free device memory allocated in `init_kernel()`.

### Available kernels

| Kernel       | Type        | `max_spp` | Description                              |
|--------------|-------------|-----------|------------------------------------------|
| mandelbrot   | Fractal     | 1         | Coloured Mandelbrot set, single-frame    |
| one_weekend  | Raytracing  | 4096      | Random spheres with realistic materials  |
| cornell_box  | Raytracing  | 4096      | Cornell box with emissive light          |

Raytracing kernels use the shared `include/rt/` library; the Mandelbrot
kernel is self-contained.

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
