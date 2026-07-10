# Architecture

## Overview

```
kernel.cpp (extern "C")  →  rt::render_main()  →  rt::trace()  →  Object::hit/scatter/emit
       │                        │                       │              │
       │  init_kernel()         │  reads std params     │  bounces    visit_rt() dispatch
       │  builds Object[]       │  sets up camera       │  loop        to variant
       │  uploads to device     │  launches parallel_for│              types
```

Each raytracer kernel (`.so` loaded at runtime) exposes four `extern "C"` functions:
`get_kernel_desc()`, `init_kernel()`, `render_kernel()`, `shutdown_kernel()`.
The heavy lifting is done by the shared `rt/` library.

## Variant-based polymorphism (no vtables, no function pointers)

Geometry and material types are stored as `std::variant` with no base classes
and no virtual methods.  Dispatch is done at compile time via `visit_rt()` —
a recursive template that expands to a chain of `if (index == 0) … else if …`
at compile time, producing no function pointers or vtable lookups.

```
Hittable = std::variant<Sphere, Quad>
Material = std::variant<Lambertian, Metal, Dielectric, DiffuseLight>

Object { Hittable hittable; Material material; }
```

- `Object::hit()` → `visit_rt(hittable, [](auto& h) { h.hit(…); })`
- `Object::scatter()` → `visit_rt(material, [](auto& m) { m.scatter(…); })`
- `Object::emit()` → `visit_rt(material, [](auto& m) { m.emit(…); })`

This works on any SYCL backend (CPU, CUDA, ROCm) because no virtual tables
or function pointers are used.

## std::optional return values

Instead of output-reference parameters + `bool`:

```cpp
// Before (bad)
bool hit(const Ray&, float, float, HitRecord&) const;
bool scatter(const Ray&, const HitRecord&, float3&, Ray&, RNG&) const;

// After (good)
std::optional<HitRecord> hit(const Ray&, float, float) const;
std::optional<ScatterRecord> scatter(const Ray&, const HitRecord&, RNG&) const;
```

`ScatterRecord` bundles `attenuation` + `scattered` ray into a single struct.

## Scene building helpers

`rt/scene.h` provides host-side helpers for constructing axis-aligned geometry:

```cpp
quad_corner(Axis::Y, 3.0f, -2, 2, -2, 2, 0);
add_quad(objects, count, Axis::X, -2.0f, -2, 2, 0, 3, material);
add_box(objects, count, corner_x, corner_y, corner_z, size_x, size_y, size_z, material);
```

The `Axis` enum class (`Axis::X`, `Axis::Y`, `Axis::Z`) replaces bare `0`/`1`/`2`.

## Standard parameter layout

Every raytracer kernel's params buffer starts with these seven values
(in this exact order, as a plain `enum rt_std_param`):

| Index | Name            | Type   | Description                     |
|-------|-----------------|--------|---------------------------------|
| 0     | RT_SPP_FRAME    | int    | Samples per frame               |
| 1     | RT_MAX_BOUNCES  | int    | Maximum ray path depth          |
| 2–4   | RT_CAM_EYE      | VEC3   | Camera position                 |
| 5–7   | RT_CAM_AT       | VEC3   | Look-at target                  |
| 8     | RT_CAM_FOV      | float  | Vertical field of view (deg)    |
| 9     | RT_CAM_APERTURE | float  | Depth-of-field aperture         |
| 10–12 | RT_CAM_UP       | VEC3   | Camera up vector                |
| 13+   | —               | —      | Kernel-specific parameters      |

Kernel-specific params start at index 13 with an anonymous `enum { … }`.

## File structure

```
include/rt/
  math.h              — float3, operators, RNG
  types_fwd.h         — Ray, HitRecord, ScatterRecord
  types.h             — Hittable variant, Material variant, Object class
  helpers.h           — random_in_unit_sphere, reflect, refract, schlick
  variant.h           — visit_rt<>() — compile-time variant dispatch
  camera.h            — Camera, lookat()
  params.h            — rt_std_param enum
  trace.h             — Object::hit/scatter/emit dispatch, trace(), render_main<>()
  scene.h             — Axis enum, quad_corner, add_quad, add_box
  hittables/
    sphere.h          — Sphere class + sphere() factory
    quad.h            — Quad   class + quad()   factory
  materials/
    lambertian.h      — Lambertian  + lambertian()
    metal.h           — Metal       + metal()
    dielectric.h      — Dielectric  + dielectric()
    diffuse_light.h   — DiffuseLight + diffuse_light()
```
