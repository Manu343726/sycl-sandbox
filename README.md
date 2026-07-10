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

## Build

```bash
# Install dependencies
conan install . -of build --build missing

# Build Release
cmake --preset conan-release -B build
make -C build -j$(nproc)

# Build with Tracy profiler support
cmake --preset conan-release -B build -DTRACY_PROFILER=ON
make -C build -j$(nproc)

# Build the Tracy server UI (standalone profiler application)
cmake --build build --target tracy-server

# Build Debug (for debugging with gdb)
conan install . -of build_debug -s build_type=Debug --build missing
cmake --preset conan-debug -B build_debug
make -C build_debug -j$(nproc)
```

### CUDA libdevice note

AdaptiveCpp's JIT compiler needs `libdevice.10.bc` to compile kernels to PTX for GPU execution.
If you get `Could not open file /opt/cuda/lib64/libdevice.10.bc`:

```bash
sudo ln -sf /opt/cuda/nvvm/libdevice/libdevice.10.bc /opt/cuda/lib64/libdevice.10.bc
```

## Usage

```bash
./build/src/sandbox                  # default: GPU backend
./build/src/sandbox -b cpu           # force CPU (OpenMP) backend
./build/src/sandbox --help           # show help
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
