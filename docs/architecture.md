# Architecture

## Project overview

`sycl-sandbox` is an interactive GPU/CPU rendering sandbox with hot-reloadable
SYCL kernels.  It renders scenes using procedurally generated geometry (spheres,
quads, boxes), accumulates samples progressively, and provides a Dear ImGui UI
for scene selection, parameter tweaking, and camera control.

The application is split into two runtime components:

1. **Host** — the `sandbox` executable (OpenGL + ImGui + SYCL runtime).
2. **Kernels** — shared libraries (`.so`) loaded at runtime via `dlopen`.
   Each kernel defines a scene and is hot-reloaded on source change.

```
                          ┌──────────────────┐
     GLFW + OpenGL        │   sandbox (host)  │  dlopen → kernel.so
      ←→ ImGui window     │   main.cpp        │──────────→ one_weekend.so
       ↑                  │   kernel_library  │──────────→ cornell_box.so
       │                  │   scene_registry  │──────────→ mandelbrot.so
       │                  │   param_ui        │
       │                  │   watcher         │
       │                  │   spdlog          │
       │                  └──────────────────┘
     Framebuffer               │ SYCL queue
     (OpenGL texture)          │ (GPU or CPU)
                               ↓
                         parallel_for
                         (per-pixel tracing)
```

## Host side (`src/`)

### Main loop (`src/main.cpp`)

Each frame:
1. **Poll events** — GLFW input, inotify source watcher.
2. **Detect window resize** — recreate render texture + SYCL buffers.
3. **Check source changes** — hot-reload any modified kernel `.cpp`/`.h`.
4. **ImGui frame** — dockspace, background render texture, control panels.
5. **Camera controls** — 2D pan/zoom (Mandelbrot) or 3D orbit/WASD (raytracers).
   Ctrl+scroll = aperture, Ctrl+Shift+scroll = FOV, Ctrl+Alt+scroll = roll.
6. **Render** — call kernel's `render_kernel()` via `dlsym`.
7. **Tonemap + display** — copy accumulation buffer from device, tonemap
   (Reinhard + gamma), upload to OpenGL texture.
8. **Render ImGui** — draw UI over the scene.

### Kernel library (`src/kernel_library.cpp`)

Each kernel is a shared library built by `acpp` (AdaptiveCpp).  On load:
1. Copy the `.so` to a versioned path (e.g. `libone_weekend.v3.so`).
2. `dlopen` the copy (so a fresh handle is guaranteed).
3. `dlsym` `get_kernel_desc()` to read the parameter metadata.
4. Compute per-param buffer offsets.

Hot-reload: `inotify` watches the kernel source directory.  On file change,
`cmake --build` is run for that target, then the `.so` is reloaded.

### Scene registry (`src/scene_registry.cpp`)

YAML files in `scenes/` define named scenes referencing a kernel:

```yaml
name: "Cornell Box"
kernel: cornell_box
params:
  spp_frame: 1
  light_color: [1.0, 1.0, 1.0]
```

`apply_params()` fills a kernel's params buffer with defaults from
`ParamMeta`, then overlays YAML values.

### Parameter UI (`src/param_ui.cpp`)

Dynamically generates ImGui controls from `ParamMeta[]`:
- `float` → `SliderFloat` / `InputFloat`
- `int` → `SliderInt` / `InputInt`
- `bool` → `Checkbox`
- `COLOR_RGB` → `ColorEdit3`
- `VEC3` → `InputFloat3`
- `ENUM` → `Combo`

All values are stored as `float` in the params buffer (including ints and
bools) so the kernel reads them uniformly as `(int)p[idx]`.

## Raytracing library (`include/rt/`)

### Variant-based polymorphism

Geometry and material types are stored as `std::variant` with no base
classes and no virtual methods.  Dispatch is done at compile time via
`visit_rt()` — a recursive template that expands to a chain of
`if (index == 0) … else if …`, producing no function pointers or vtables.

```
Hittable = std::variant<Sphere, Quad>
Material = std::variant<Lambertian, Metal, Dielectric, DiffuseLight>
Object   { Hittable hittable; Material material; }
```

- `Object::hit()` → `visit_rt(hittable, [](auto& h) { h.hit(…); })`
- `Object::scatter()` → `visit_rt(material, [](auto& m) { m.scatter(…); })`
- `Object::emit()` → `visit_rt(material, [](auto& m) { m.emit(…); })`

This works on any SYCL backend (CPU, CUDA) because no vtables or
function pointers are used.

### std::optional return values

Instead of output-reference parameters + `bool`:

```cpp
std::optional<HitRecord>      hit(const Ray&, float, float) const;
std::optional<ScatterRecord>  scatter(const Ray&, const HitRecord&, RNG&) const;
```

`ScatterRecord` bundles `attenuation` + `scattered` ray.

### Scene building helpers

`rt/scene.h` provides host-side helpers for axis-aligned geometry:

```cpp
quad_corner(Axis::Y, 3.0f, -2, 2, -2, 2, 0);
add_quad(objects, count, Axis::X, -2.0f, -2, 2, 0, 3, material);
add_box(objects, count, cx, cy, cz, sx, sy, sz, material);
```

### Standard parameter layout

Every raytracer kernel's params buffer starts with the same layout
(`enum rt_std_param` in `params.h`), then kernel-specific params from
index 13 onward (`enum { … }`).

### File structure

```
include/rt/
  math.h              — float3, operators, RNG
  types_fwd.h         — Ray, HitRecord, ScatterRecord
  types.h             — Hittable/Material variants, Object class
  helpers.h           — random_in_unit_sphere, reflect, refract, schlick
  variant.h           — visit_rt<>() compile-time variant dispatch
  camera.h            — Camera struct, lookat()
  params.h            — rt_std_param enum
  trace.h             — Object dispatch, trace(), render_main<>()
  scene.h             — Axis enum, quad_corner, add_quad, add_box
  hittables/          — Sphere, Quad (each in own file, with factory)
  materials/          — Lambertian, Metal, Dielectric, DiffuseLight
```

## Kernels (`kernels/`)

Each kernel is a shared library exposing four `extern "C"` functions:

| Function            | Called when      | Purpose                                  |
|---------------------|------------------|------------------------------------------|
| `get_kernel_desc()` | `.so` loaded     | Return parameter metadata + limits       |
| `init_kernel()`     | Scene selected   | Build geometry, upload to device         |
| `render_kernel()`   | Every frame      | Read params, call `rt::render_main()`   |
| `shutdown_kernel()` | Scene changed    | Free device memory                       |

Three kernels:
- **mandelbrot** — Mandelbrot fractal (no ray tracing, single-frame).
- **one_weekend** — Raytraced spheres (lambertian, metal, dielectric).
- **cornell_box** — Raytraced quads with emissive light source.

## Build system

Dependencies are managed by Conan (`conanfile.py`).  Two build directories
are maintained:

```bash
build/        # cmake --preset conan-release    (Release, -O3)
build_debug/  # cmake --preset conan-debug      (Debug, -O0 -g)
```

Kernels are built via `add_sycl_to_target()` (AdaptiveCpp).  The `generic`
backend is used (OpenMP CPU), portable across any system.
