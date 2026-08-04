# SYCL Sandbox

Interactive GPU/CPU rendering sandbox with hot-reloadable SYCL kernels. Built with
[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp), Dear ImGui, and OpenGL.

## What it does

- Renders procedurally generated scenes using SYCL kernels (Mandelbrot fractal, ray tracing)
- Accumulates samples progressively across frames (SPP)
- Hot-reloads kernel shared libraries on source change — edit a kernel and see the result live
- Interactive camera controls (pan, orbit, zoom) via mouse and keyboard
- GPU or CPU backend selectable at runtime

## Built-in scenes

| Scene | Kernel | Description | Requires accumulation |
|-------|--------|-------------|----------------------|
| Mandelbrot Fractal | `mandelbrot` | Colored Mandelbrot set with HSV coloring | No (single frame) |
| Random Spheres | `one_weekend` | Raytraced spheres with materials (lambertian, metal, dielectric) | Yes |

## Dependencies

- [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) — SYCL implementation
- [Dear ImGui](https://github.com/ocornut/imgui) — UI
- [GLFW](https://www.glfw.org/) — window/input
- [glm](https://github.com/g-truc/glm) — math
- [yaml-cpp](https://github.com/jbeder/yaml-cpp) — scene config files
- [spdlog](https://github.com/gabime/spdlog) — logging
- [args](https://github.com/Taywee/args) — CLI argument parsing
- [Tracy](https://github.com/wolfpld/tracy) — profiler (optional, `-DTRACY_PROFILER=ON`)
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

### 4. Tracy (optional profiler support)

```bash
# Build with Tracy profiler support
cmake --preset conan-release -B build -DTRACY_PROFILER=ON
make -C build -j$(nproc)

# Build the Tracy server UI (standalone profiler application)
cmake --build build --target tracy-server
```

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

### 4. Tracy (optional profiler support)

```bash
# Build with Tracy profiler support
cmake --preset conan-release -B build -DTRACY_PROFILER=ON
make -C build -j$(nproc)

# Build the Tracy server UI (standalone profiler application)
cmake --build build --target tracy-server
```

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

| Input | 2D camera (Mandelbrot) | 3D camera (raytracing) |
|-------|------------------------|------------------------|
| LMB drag | Pan | Orbit around target |
| Scroll | Zoom | Zoom in/out |
| Arrow keys | — | Orbit |
| Shift + Arrows | — | Pan target point |

The camera controls only appear when the active kernel exposes the relevant parameters
(`center_x`/`center_y`/`zoom` for 2D, `cam_eye`/`cam_at`/`cam_fov` for 3D).

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
│   ├── mandelbrot/          # Mandelbrot fractal
│   ├── one_weekend/         # Raytracing in One Weekend
│   └── CMakeLists.txt
├── src/                     # Sandbox host code
│   ├── main.cpp             # Main loop, rendering pipeline, camera controls
│   ├── kernel_library.cpp   # Hot-reloadable .so loader
│   ├── scene_registry.cpp   # YAML scene parser
│   ├── param_ui.cpp         # ImGui parameter controls
│   └── watcher.cpp          # Inotify source file watcher
├── include/sycl-sandbox/    # Shared headers (kernel + host)
│   ├── sandbox_api.h        # Kernel API definition
│   ├── param_types.h        # Parameter metadata types
│   ├── profiling.h          # Tracy profiling wrapper macros
│   └── rt/                  # Raytracing library
├── profiler/                # Tracy server build (ExternalProject)
│   └── CMakeLists.txt
├── docs/                    # Architecture and coding guidelines
└── build/                   # Build output (Release)
```

## Adding a new kernel

1. Create `kernels/mykernel/kernel.cpp` implementing `get_kernel_desc()`, `init_kernel()`,
   `render_kernel()`, and `shutdown_kernel()`
2. Add `mykernel` to `KERNEL_DIRS` in `kernels/CMakeLists.txt`
3. Create `scenes/mykernel.yaml` referencing the kernel
4. Build and run — the kernel appears in the scene dropdown
