# SYCL Sandbox

Interactive GPU/CPU rendering sandbox with hot-reloadable SYCL kernels. Built with
[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp), Dear ImGui, and OpenGL.

## What it does

- Renders procedurally generated YAML scenes using SYCL kernels (path tracing)
- Accumulates samples progressively across frames (SPP)
- Hot-reloads kernel shared libraries on source change — edit a kernel and see the result live
- Interactive camera controls (pan, orbit, zoom) via mouse and keyboard
- GPU or CPU backend selectable at runtime

## Built-in scenes

Scenes are YAML files in `scenes/` (`name`, `kernel`, `max_spp`, params,
procedural objects — see `docs/raytracing.md`).  The bundled ones:

| Scene YAML | Description |
|------------|-------------|
| `one_weekend_final.yaml` | Random spheres with realistic materials (lambertian, metal, dielectric), ColorChecker ground |
| `mesh_demo.yaml` | Triangle-mesh objects loaded from STL files |
| `portal_rooms.yaml` | Portals linking two rooms |

All scenes render with the single `raytracer` kernel (the shared
`include/sycl-sandbox/rt/` path tracer).

## Dependencies

- [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) — SYCL implementation
- [Dear ImGui](https://github.com/ocornut/imgui) — UI
- [GLFW](https://www.glfw.org/) — window/input
- [glm](https://github.com/g-truc/glm) — math
- [yaml-cpp](https://github.com/jbeder/yaml-cpp) — scene config files
- [spdlog](https://github.com/gabime/spdlog) — logging
- [args](https://github.com/Taywee/args) — CLI argument parsing
- [Tracy](https://github.com/wolfpld/tracy) — profiler (on by default, `-DSANDBOX_ENABLE_TRACY=OFF` to disable)
- OpenGL 3.3+

## Building

### 1. Register the local recipes index

`conan/` is a [local-recipes-index](https://docs.conan.io/2/devops/devops_local_recipes_index.html)
repository: a local Conan remote (mirroring the conan-center-index layout) that hosts the
project's patched recipe for [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp),
which is not available on ConanCenter. The patch adds an `output_stream::set_stream()`
API for redirecting AdaptiveCpp's internal logging through spdlog. Register it as a
local Conan remote (once per machine, it points to this checkout):

```bash
conan remote add sycl-sandbox-dependencies ./conan --type=local-recipes-index --allowed-packages="adaptivecpp/*"
```

- `--type=local-recipes-index` makes Conan read recipes from a local folder
- `--allowed-packages=adaptivecpp/*` ensures Conan only ever resolves `adaptivecpp` from it

This must be done once after cloning — the remote is local and stays in your
Conan configuration. It is a recipes-only remote (no binaries), so `--build=missing`
compiles AdaptiveCpp from source on first install.

### 2. Build the patched AdaptiveCpp package

Build the patched AdaptiveCpp once so it's cached locally:

```bash
# Release build (used by ./build/src/sycl-sandbox)
conan create conan/recipes/adaptivecpp/all \
    -o '&:with_cuda=True' -o '&:with_openmp=True' -o '&:experimental_llvm=True'

# Debug build (used by ./build_debug/src/sycl-sandbox, for gdb)
conan create conan/recipes/adaptivecpp/all \
    -s build_type=Debug \
    -o '&:with_cuda=True' -o '&:with_openmp=True' -o '&:experimental_llvm=True'
```

### 3. Install dependencies and build

```bash
conan install . -of build --build missing
cmake --preset conan-release -B build
make -C build -j$(nproc)

# Build Debug (for debugging with gdb)
conan install . -of build_debug -s build_type=Debug --build missing
cmake --preset conan-debug -B build_debug
make -C build_debug -j$(nproc)
```

> **Note:** AdaptiveCpp is compiled from source by Conan and may take several
> minutes.  The result is cached in your local Conan cache.

### 4. Tracy (profiler support, on by default)

```bash
# Build with Tracy profiler support (on by default; pass -DSANDBOX_ENABLE_TRACY=OFF to disable)
cmake --preset conan-release -B build
make -C build -j$(nproc)

# Standalone profiler UI (built from Tracy's own CMake; the app only
# ships the client — nothing is embedded)
cmake --build build --target tracy-profiler
./build/profiler/bin/tracy-profiler
```

In VS Code the option is set in `.vscode/settings.json` →
`cmake.configureArgs` so CMake Tools picks it up on configure.  With it
enabled, the app runs a Tracy client (on-demand, port 8086) and the
profiler UI is the standalone `tracy-profiler` executable: launch it
from **Controls → "Launch Tracy Profiler"** (or the "launch tracy-server"
VS Code task) and it connects to the app and captures the kernel GPU
timeline.  See `docs/architecture.md` for the integration design.

### Recipe changes

Once a recipe revision is cached locally, Conan keeps using it and won't re-export from the
local repository. After editing a recipe under `conan/recipes/`, purge the cached revision so
it's re-exported and rebuilt on next install:

```bash
conan remove adaptivecpp/25.10.0 -c
conan install . -of build --build missing
```

This must be done once after cloning — the remote is local and stays in your
Conan configuration.

### CUDA toolkit detection

The AdaptiveCpp recipe auto-detects the CUDA toolkit among `$CUDA_HOME`, `$CUDA_PATH`,
`/usr/local/cuda`, and `/opt/cuda` (set `CUDA_HOME` to override). AdaptiveCpp's JIT compiler
also needs `libdevice.10.bc` to compile kernels to PTX for GPU execution. If you get
`Could not open file ... libdevice.10.bc`, the CUDA toolkit stores it in `nvvm/libdevice/`
— symlink it next to the toolkit's other libs:

```bash
sudo ln -sf /usr/local/cuda/nvvm/libdevice/libdevice.10.bc /usr/local/cuda/lib64/libdevice.10.bc
```

On systems without a CUDA toolkit, build without the CUDA backend instead:

```bash
conan install . -of build --build missing -o adaptivecpp/*:with_cuda=False
```

## Usage

```bash
./build/src/sycl-sandbox                  # default: GPU backend
./build/src/sycl-sandbox -b cpu           # force CPU (OpenMP) backend
./build/src/sycl-sandbox --help           # show help
```

### VS Code

`.vscode/launch.json` provides four configurations (Release/Debug × CPU/GPU),
each auto-building via `make -C build -j$(nproc)` before launch.

### Controls

| Input | 3D camera (raytracing) |
|-------|------------------------|
| LMB drag | Orbit around target |
| Scroll | Zoom in/out |
| Arrow keys | Orbit |
| Shift + Arrows | Pan target point |

The camera controls only appear when the active kernel exposes the relevant
parameters (`cam_eye`/`cam_at`/`cam_up`/`cam_fov` for 3D).

### UI

- **Controls** panel: scene selector, kernel parameters, target SPP slider, accumulation reset
- Rendered view fills the full window background
- Camera info shows current position/zoom

## Project structure

```
├── CMakeLists.txt           # Root build file
├── conanfile.py             # Conan dependency recipe
├── conan/                   # Local Conan repository (local-recipes-index remote)
│   └── recipes/
│       └── adaptivecpp/     # Custom AdaptiveCpp recipe (not on ConanCenter)
│           ├── config.yml   # Maps versions to recipe folders
│           └── all/
│               ├── conanfile.py
│               ├── conandata.yml
│               └── patches/ # set_stream_logging + LLVM 22 compat patches
├── scenes/*.yaml            # Scene definitions (kernel + parameter overrides)
├── kernels/                 # SYCL kernel shared libraries
│   ├── raytracer/           # Path-tracing kernel (single kernel_entry)
│   └── CMakeLists.txt
├── src/                     # Sandbox host code
│   ├── main.cpp             # Main loop, camera controls
│   ├── render_loop.h        # Render thread: Context fill + dispatch + tonemap
│   ├── kernel/              # Kernel .so loading, build system, runtime
│   ├── scene/               # YAML registry, scene descriptors, host scene
│   ├── ui/                  # ImGui panels (controls, params, scene debug)
│   ├── io/                  # Source watcher (inotify + .d tracking)
│   └── tracy/               # Tracy bridge + profiler launcher
├── include/sycl-sandbox/    # Shared headers (kernel + host)
│   ├── context.h            # rt::Context — the kernel ABI (single entry)
│   ├── sandbox_api.h        # Kernel API definition
│   ├── profiler.h           # Profiler macros (device ring / Tracy / no-op)
│   └── rt/                  # Raytracing library
├── docs/                    # Architecture and coding guidelines
└── build/                   # Build output (Release)
```

## Adding a new kernel

1. Create `kernels/mykernel/kernel.cpp` with a single
   `extern "C" void kernel_entry(const rt::Context *ctx)` that calls
   `rt::render<MyKernel>(ctx, background_fn)` — see
   `kernels/raytracer/kernel.cpp` (no ops, no globals, no init/shutdown)
2. Add `mykernel` to `KERNEL_DIRS` in `kernels/CMakeLists.txt` (this also
   builds the `mykernel_native` variant for the software backend)
3. Create `scenes/mykernel.yaml` referencing the kernel
4. Build and run — the scene appears in the dropdown
