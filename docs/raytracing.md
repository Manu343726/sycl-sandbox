# Raytracing Library

## Overview

The shared `include/rt/` library provides a complete raytracing pipeline that
any kernel can use.  A raytracer kernel is just a normal kernel that happens
to include `rt/` headers and call `rt::render_main()` from its
`render_kernel()`.  The kernel owns only its scene geometry and material
parameters; everything else (camera, path tracing, accumulation) is handled
by the library.

```
┌──────────────────────────────────────────────────┐
│  kernel.cpp (your code)                          │
│                                                  │
│  init_kernel() → build scene, upload to device   │
│  render_kernel() → rt::render_main(queue, …,     │
│                      scene_objects, num_objects, │
│                      background_fn)              │
│  shutdown_kernel() → free scene memory           │
│                                                  │
│  ┌──────────────────────────────────────────────┐│
│  │  include/rt/ (shared library)                 ││
│  │                                               ││
│  │  render_main() → reads standard params,       ││
│  │                   sets up camera, launches    ││
│  │                   parallel_for → trace()      ││
│  │                    → Object::hit/scatter/emit ││
│  │                     → visit_rt() dispatch     ││
│  └──────────────────────────────────────────────┘│
└──────────────────────────────────────────────────┘
```

## Anatomy of a minimal raytracer kernel

Standard params (SPP, bounces, camera) are **implicit** — kernels don't
declare them in `params_meta[]`.  The host allocates space for them at
fixed `rt_std_param` indices and fills defaults via `init_std_params()`.
Kernels only declare their own kernel-specific params.

```cpp
#include "rt/types.h"        // Object, Hittable, Material
#include "rt/trace.h"        // rt::render_main()
#include "rt/params.h"       // rt_std_param enum (implicit)
#include "rt/scene.h"        // Axis, quad_corner, DeviceBuffer
#include "rt/hittables/quad.h"
#include "rt/materials/lambertian.h"
#include "rt/materials/diffuse_light.h"

using namespace rt;
using rt::materials::lambertian;
using rt::materials::diffuse_light;

// ── Params (kernel-specific only; standard 0–12 are implicit) ──────────
static ParamMeta params_meta[] = {
    {"light_color",…}, {"light_strength",…},
};
// The standard param area (13 floats) is at the start of the buffer.
// Kernel-specific params start at index RT_NUM_STD_PARAMS.
enum { PARAM_LIGHT_COLOR = 0, PARAM_LIGHT_STRENGTH = PARAM_LIGHT_COLOR + 3 };

// ── Scene state ───────────────────────────────────────────────────────
static Object* g_scene_objects = nullptr;
static int     g_num_objects   = 0;

// ── Scene builder (host, called by init_kernel) ────────────────────────
extern "C" void init_kernel(sycl::queue* queue, int, int,
                            const void* params, size_t) {
    const float* p = (const float*)params;
    // read kernel-specific params, build geometry with add_quad/add_box/…
    // upload Object[] to device
    g_scene_objects = sycl::malloc_device<Object>(count, *queue);
    queue->memcpy(g_scene_objects, objects, count * sizeof(Object)).wait();
    g_num_objects = count;
}

// ── Render (called every frame) ───────────────────────────────────────
extern "C" void render_kernel(sycl::queue* queue, int w, int h,
                               const void* params, void* accum, int si) {
    rt::render_main(queue, w, h, (const float*)params, (float*)accum, si,
                    g_scene_objects, g_num_objects,
                    [](const Ray&) -> float3 { return {0,0,0}; });
}

extern "C" void shutdown_kernel(sycl::queue* queue) { … }
```

## Primitive composition

Primitives can be composed from other primitives.  For example, `Box` is
implemented as six `Quad` faces internally — the `Box::hit()` method
iterates over all six faces and returns the closest intersection.  This
avoids proliferating custom geometry helpers (like `add_quad`/`add_box`)
— just construct the primitive directly and add it as a single `Object`:

```cpp
g_scene.push_back({hittables::box(cx, cy, cz, sx, sy, sz), material});
```

The same composition principle can be extended to other compound primitives
in the future.

## Variant-based polymorphism

Geometry and material types are stored as `std::variant` with no base classes
and no virtual methods.  Dispatch is done at compile time via `visit_rt()`
— a recursive template that expands to a chain of `if (index == 0) … else if …`
at compile time, producing no function pointers or vtable lookups.

```cpp
Hittable = std::variant<Sphere, Quad>
Material = std::variant<Lambertian, Metal, Dielectric, DiffuseLight>
Object   { Hittable hittable; Material material; }
```

- `Object::hit()` → `visit_rt(hittable, …)` → calls `Sphere::hit()` or `Quad::hit()`
- `Object::scatter()` → `visit_rt(material, …)` → calls `Lambertian::scatter()` etc.
- `Object::emit()` → `visit_rt(material, …)` → calls `DiffuseLight::emit()` etc.

This works on any SYCL backend (CPU OpenMP, CUDA) because no vtables or
function pointers are used.

## std::optional return values

Instead of output-reference parameters + `bool`:

```cpp
std::optional<HitRecord>      hit(const Ray&, float, float) const;
std::optional<ScatterRecord>  scatter(const Ray&, const HitRecord&, RNG&) const;
```

`ScatterRecord` bundles `attenuation` + `scattered` ray into a single struct.

## Standard parameter layout

Every raytracer kernel's params buffer starts with these seven values
(`enum rt_std_param` in `params.h`), in this exact order:

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

`rt::render_main()` reads indices 0–12 directly from the params buffer and
ignores everything beyond.  Kernel-specific params use a plain anonymous
`enum { PARAM_X = RT_NUM_STD_PARAMS, … }` for implicit int conversion.

## Scene building helpers

`rt/scene.h` provides `DeviceBuffer<T>`, a host-resident buffer that grows
on demand (`push_back`) and uploads to device (`transfer_to_device()`):

```cpp
static DeviceBuffer<Object> g_scene;

void init_kernel(sycl::queue *queue, …) {
    g_scene = DeviceBuffer<Object>(queue, 64);           // initial capacity

    g_scene.push_back({hittables::quad(axis, value, …), material});  // one face
    g_scene.push_back({hittables::box(cx, cy, cz, sx, sy, sz), material});  // box

    g_scene.transfer_to_device();   // malloc_device + memcpy
}

void render_kernel(…) {
    render_main(queue, …, g_scene.device_ptr(), g_scene.size(), …);
}

void shutdown_kernel(…) {
    g_scene = DeviceBuffer<Object>();  // move-assign empty → frees old
}
```

`quad()` takes the primary axis as a bare int (`0` = X, `1` = Y, `2` = Z).
`Box` is a first-class hittable — it is implemented as six `Quad` faces
internally but hits as a single `Object`.

## File structure

```
include/rt/
  math.h              — float3, operators, RNG
  types_fwd.h         — Ray, HitRecord, ScatterRecord
  types.h             — Hittable variant, Material variant, Object class
  helpers.h           — random_in_unit_sphere, reflect, refract, schlick
  variant.h           — visit_rt<>() — compile-time variant dispatch
  camera.h            — Camera struct, lookat()
  params.h            — rt_std_param enum
  trace.h             — Object::hit/scatter/emit dispatch, trace(), render_main<>()
    scene.h             — Axis enum, quad_corner, DeviceBuffer
  hittables/
    sphere.h          — Sphere class + sphere() factory
    quad.h            — Quad   class + quad()   factory
    box.h             — Box    class + box()    factory
  materials/
    lambertian.h      — Lambertian  + lambertian()
    metal.h           — Metal       + metal()
    dielectric.h      — Dielectric  + dielectric()
    diffuse_light.h   — DiffuseLight + diffuse_light()
```
