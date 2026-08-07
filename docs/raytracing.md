# Raytracing Library

## Overview

The shared `include/sycl-sandbox/rt/` library provides a complete raytracing pipeline that
any kernel can use.  A raytracer kernel is just a normal kernel that happens
to include `rt/` headers and call `rt::render_main()` from its
`render_kernel()`.  The kernel owns only its scene geometry and material
parameters; everything else (camera, path tracing, accumulation) is handled
by the library.

```
┌──────────────────────────────────────────────────────┐
│  kernel.cpp (your code)                              │
│                                                      │
│  init_kernel() → build scene via SceneBuilder,       │
│                  upload per-type arrays to device     │
│  render_kernel() → rt::render_main(queue, …,         │
│                      scene_view, background_fn)      │
│  shutdown_kernel() → free device memory              │
│                                                      │
│  ┌──────────────────────────────────────────────────┐│
│  │  include/sycl-sandbox/rt/ (shared library)       ││
│  │                                                  ││
│  │  render_main() → reads standard params,          ││
│  │                   sets up camera, launches       ││
│  │                   parallel_for → trace()         ││
│  │                    → handle_hit/scatter/emit     ││
│  │                     → switch dispatch per-type   ││
│  └──────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
```

## Data-oriented scene layout

Scene data is stored in separate per-type arrays rather than interleaved
`Object` structs.  This improves cache performance: the hot intersection
loop reads only compact 8-byte handles and dispatches to type-specific
arrays, while material data is only accessed for the one winning hit.

### Handle

An 8-byte reference to a hittable and material in separate per-type arrays:

```cpp
struct Handle {
    uint32_t hittable;  // (type_id << 24) | index_in_type_array
    uint32_t material;  // (type_id << 24) | index_in_type_array
};
```

Type IDs: 0 = Sphere, 1 = Triangle, 2 = Quad, 3 = Box, 4 = Portal (hittable);
0 = Lambertian, 1 = Metal, 2 = Dielectric, 3 = DiffuseLight,
4 = TexturedLambertian (material).

### SceneView (device-side)

Trivially-copyable struct of device pointers, captured by value in SYCL
lambdas.  All pointers are `sycl::malloc_device` allocations.

```cpp
struct SceneView {
    Handle *handles; int num_handles;
    Sphere *spheres; int num_spheres;
    Triangle *triangles; int num_triangles;
    Quad *quads; int num_quads;
    Box *boxes; int num_boxes;
    Lambertian *lambertians; int num_lambertians;
    Metal *metals; int num_metals;
    Dielectric *dielectrics; int num_dielectrics;
    DiffuseLight *diffuse_lights; int num_diffuse_lights;
    TexturedLambertian *textured_lambertians; int num_textured_lambertians;
    Portal *portals; int num_portals;
    Aabb *aabbs;                          // per-handle AABBs
    BvhNode *bvh_nodes; int bvh_root;     // BVH (future)
    LightInfo *lights; int num_lights;    // light list (future)
};
```

### SceneBuilder (host-side)

Accumulates hittable-material pairs via `add()`, then uploads per-type
arrays to device memory via `build()`:

```cpp
SceneBuilder scene;
scene.add({hittables::sphere({0, 0, 0}, 1), materials::lambertian({1, 1, 1})});
scene.add({hittables::box(-1, -1, -1, 2, 2, 2), materials::metal({0.8f, 0.8f, 0.8f}, 0)});
scene_view = scene.build(*queue);  // upload to device
```

### Dispatch

The `trace()` function iterates handles and calls `handle_hit()`,
`handle_scatter()`, `handle_emit()` — inline functions that switch on
the packed type tag and index into the correct per-type array.

```cpp
for (int i = 0; i < scene.num_handles; i++) {
    auto hit = handle_hit(scene.handles[i], ray, t_min, t_max, scene);
    if (hit) { closest_hit = hit; hit_index = i; }
}
// After loop: dispatch emit/scatter on the winning handle
float3 emitted = handle_emit(winning_handle, *closest_hit, scene);
auto scattered = handle_scatter(winning_handle, ray, *closest_hit, rng, scene);
```

## Anatomy of a minimal raytracer kernel

All parameters (including standard ones like SPP, bounces, camera) are
declared in the YAML scene file and managed by `SceneDescriptor`.  Kernels
read parameters by name via `ParamLookup::read<T>("name")`.

```cpp
#include <sycl-sandbox/rt/types.h>      // Object, Hittable, Material
#include <sycl-sandbox/rt/trace.h>      // rt::render_main()
#include <sycl-sandbox/rt/scene_data.h> // SceneBuilder, SceneView
#include <sycl-sandbox/rt/hittables/quad.h>
#include <sycl-sandbox/rt/materials/lambertian.h>
#include <sycl-sandbox/rt/materials/diffuse_light.h>

using namespace rt;
using rt::materials::lambertian;
using rt::materials::diffuse_light;

// ── Kernel-specific params are read by name from the buffer ───────────

// ── Scene state ───────────────────────────────────────────────────────
static SceneBuilder scene;
static SceneView scene_view = {};

// ── Scene builder (host, called by init_kernel) ────────────────────────
extern "C" void init_kernel(sycl::queue* queue, int, int,
                            const void* params, size_t) {
    rt::ParamLookup lookup(/* ... */);
    // Read kernel-specific params by name — type-safe
    // float3 light_color = lookup.read<float3>("light_color");
    // Build geometry with scene.add(…)
    scene = SceneBuilder();
    scene.add({hittables::quad(axis, value, …), lambertian(color)});
    scene.add({hittables::box(cx, cy, cz, sx, sy, sz), lambertian(color)});
    scene_view = scene.build(*queue);   // upload per-type arrays to device
}

// ── Render (called every frame) ───────────────────────────────────────
extern "C" void render_kernel(sycl::queue* queue, int w, int h,
                               const void* params, void* accum, int si) {
    rt::render_main(queue, w, h, (const float*)params, (float*)accum, si,
                    scene_view,
                    [](const Ray&) -> float3 { return {0,0,0}; });
}

extern "C" void shutdown_kernel(sycl::queue* queue) {
    scene_view.free(*queue);
}
```

## Primitive composition

Primitives can be composed from other primitives.  For example, `Box` is
implemented as six `Quad` faces internally — the `Box::hit()` method
iterates over all six faces and returns the closest intersection.  This
avoids proliferating custom geometry helpers (like `add_quad`/`add_box`)
— just construct the primitive directly and add it as a single handle:

```cpp
scene.add({hittables::box(cx, cy, cz, sx, sy, sz), material});
```

The same composition principle can be extended to other compound primitives
in the future.

## AABB and acceleration structures

Every hittable type provides an `aabb()` method returning the enclosing
axis-aligned bounding box.  The `SceneBuilder::build()` computes per-handle
AABBs and stores them in `SceneView::aabbs`.

The `aabb_hit()` utility performs a fast slab-method ray-AABB test, used
for broad-phase rejection in BVH traversal and available for future
importance sampling.

BVH construction (`SceneBuilder::build_bvh()`) and light list computation
are stubbed but the data slots are in place in `SceneView`.

## Textures

A texture maps surface parametric coordinates plus time to a colour:

```cpp
float3 sample(float u, float v, float time, RNG &rng) const;
```

`u`/`v` come from the hit point's parametric coordinates, filled into
`HitRecord::u/v` by every hittable:

| primitive | UV meaning |
|-----------|-----------|
| Quad      | affine (α, β) coordinates, both in [0,1] |
| Triangle  | barycentric coordinates of vertices b, c |
| Sphere    | spherical mapping from the outward direction |
| Box       | the hit face's quad coordinates |
| Portal    | the entry hit's UVs, forwarded to the exit shape (see Portals) |

Out-of-range UVs are **not** clamped by the pipeline — every texture
implements its own clamp / round / wrap behavior.  This is what allows
both procedural textures (infinite tiling, noise, ...) and file-loaded
image textures (clamped or wrapped sampling) behind one interface.

The sampler also receives the path's `RNG`, so stochastic textures
(noise, jittered filtering, ...) can draw from it without owning their
own state.  Deterministic textures simply ignore it.

### Fully procedural, no base class

Textures are plain procedural structs — no virtual base class (SYCL
device code cannot dispatch through vtables) and no snapshot machinery.
Polymorphism comes from the same variant + compile-time `visit()`
dispatch used by `Hittable`/`Material`:

```cpp
using Texture = std::variant<SolidColor, ColorChecker>;

// Free function dispatching to the concrete texture's sample():
float3 sample(const Texture&, float u, float v, float time, RNG &rng);
```

Each texture struct owns its data and implements `sample()` however it
likes — the interface is simply the `sample(u, v, time, rng)` convention.

### Built-in textures

| texture | behavior |
|---------|----------|
| `SolidColor` | ignores (u, v, time), returns one constant colour |
| `ColorChecker` | 24-patch ColorChecker chart (reference sRGB from `colorchecker.h`); wraps on UV overflow so the chart tiles infinitely; `scale_u`/`scale_v` repeat it per UV unit |

### TexturedLambertian

Diffuse material whose albedo is sampled from a texture:

```cpp
class TexturedLambertian {
    textures::Texture texture;   // variant of procedural textures
    optional<ScatterRecord>
    scatter(const Ray&, const HitRecord&, RNG& rng) const {
        float3 albedo = textures::sample(texture, rec.u, rec.v, incoming_ray.time, rng);
        // …same diffuse math as Lambertian…
    }
};
```

The ray's `time` field (from the scene's `time` param, read once per frame
in `render_main`) is passed through to the sampler, enabling time-varying
procedural textures later.

### YAML usage

```yaml
objects:
  - type: quad
    center: [0, 0, 0]
    u: [4, 0, 0]
    v: [0, 0, 4]
    material:
      type: textured_lambertian
      texture: colorchecker   # or "solid"
      scale_u: 1.0            # chart repetitions per UV unit
      scale_v: 1.0
```

See `scenes/colorchecker_floor.yaml` for a complete demo scene.

## Portals

A portal teleports a ray from one hittable's surface to another's.  It is
implemented as a **pair of hittables** (entry → exit): the parametric
coordinates (`u`/`v`) of the entry hit are used as input for the second
hittable, so the ray reappears at the corresponding point on the exit
surface.  The mapping is exactly inverse to `hit()`: every hittable
provides `point_at_uv(u, v)` that reconstructs a surface point from the
same UV convention its `hit()` fills into `HitRecord::u/v`.

```cpp
class Portal {
    PortalShape entry;   // std::variant<Sphere, Triangle, Quad>
    PortalShape exit;
    ...
};

scene.add_portal(hittables::quad(/* entry */), hittables::quad(/* exit */));
// Optional material — e.g. diffuse_light for an emissive portal:
scene.add_portal(hittables::quad(/* entry */), hittables::quad(/* exit */),
                 materials::diffuse_light({15, 15, 15}));
```

Semantics:

- The ray keeps its **direction** (position-only teleport):
  `portal_origin` = exit point nudged by an epsilon along the exit
  normal in the travel direction (avoids re-hitting the exit surface),
  `portal_dir` = incoming ray direction.
- `HitRecord::t` keeps the **entry distance**, so closest-hit ordering
  works without special-casing.
- Portals are instanced as objects with a **dummy material** (a white
  Lambertian by default).  The teleport lives in the material: every
  material's `scatter()` checks `HitRecord::is_portal` and returns the
  continuation ray (`portal_origin`/`portal_dir`) with unit attenuation
  (see `rt::portal_scatter()`), so the trace loop stays generic.
- Emission is handled **before** scatter, so a portal instanced with a
  real material behaves accordingly — e.g. `diffuse_light` makes an
  **emissive portal** (a glowing surface; the path terminates with its
  emission instead of teleporting).  Emissive portals are excluded from
  the light list (their surface area is undefined).
- Portals are **bidirectional**: the exit shape is a plain hittable too,
  and the exit side works through the same pair.
- The portal's AABB is the **entry** shape's AABB, so BVH culling is
  correct without recursion.

Limitations:

- `Box` is **not** supported as a portal shape: its `hit()` does not
  record which face was hit, so `point_at_uv` could not reconstruct the
  entry point (see `hittables/portal.h`).
- UVs outside [0,1] (hitting the entry's edge is impossible — the
  intersection is clipped — but mapped points are clamped to the exit
  shape's surface) project onto the exit surface without wrapping.

### YAML usage

```yaml
objects:
  - type: portal
    entry: {type: quad, center: [-1, 0, -1.99], u: [2, 0, 0], v: [0, 2, 0]}
    exit:  {type: quad, center: [-1, 0, -2.1],  u: [2, 0, 0], v: [0, 2, 0]}
    # optional material: omit for a pure window (dummy Lambertian),
    # or use diffuse_light for an emissive portal:
    # material: {type: diffuse_light, color: [1, 1, 1], intensity: 5}
```

Entry/exit shapes support `type: quad` (center/u/v), `type: sphere`
(center/radius) and `type: triangle` (v0/v1/v2).  See
`scenes/portal_rooms.yaml` for a complete two-room demo scene and
`kernels/rt_portal_test/kernel.cpp` for a from-scratch reference
implementation.

## optional return values

Instead of output-reference parameters + `bool`:

```cpp
optional<HitRecord>      hit(const Ray&, float, float) const;
optional<ScatterRecord>  scatter(const Ray&, const HitRecord&, RNG&) const;
```

`ScatterRecord` bundles `attenuation` + `scattered` ray into a single struct.

## Standard parameter layout

All parameters are declared in the YAML scene file.  Camera parameters
(`cam_eye`, `cam_at`, `cam_up`, `cam_fov`, `cam_aperture` for 3D; 
`center_x`, `center_y`, `zoom` for 2D) are auto-generated from `scene_type`.
Standard render params (`spp_frame`, `max_bounces`, `tick`, `time`) are
also auto-injected, together with the display-pipeline params:

| Param | Type | Default | Purpose |
|---|---|---|---|
| `tonemap_enabled` | bool | `false` | Master switch for the tone-mapping stage of the display pipeline. When off, accumulated linear values are normalized and hard-clamped to [0,1] (no operator, no gamma). |
| `tonemap_operator` | enum | `0` | Tone-map operator: `0` = Reinhard (`x/(1+x)`), `1` = ACES fitted (Narkowicz 2015), `2` = Filmic (Hable / Uncharted 2). Rendered as a combo box. |
| `tonemap_exposure` | float | `1.0` | Exposure multiplier applied to the linear HDR value before the operator. |
| `tonemap_gamma` | float | `2.2` | Display gamma for the final correction (`pow(clamp(c), 1/gamma)`). |

Scenes may override the default of any standard param by declaring the
same name in the YAML `params:` section — the loader merges the YAML
default/range/description into the auto-generated descriptor instead of
skipping it (keep the auto-generated type):

```yaml
params:
  tonemap_enabled:
    type: bool
    description: "Enable the tone-mapping stage"
    default: true
  tonemap_operator:
    type: enum
    default: 1          # ACES fitted
    options: ["Reinhard", "ACES (fitted)", "Filmic (Hable)"]
```

The host builds the buffer layout via `SceneDescriptor::build_layout()` and
provides a `ParamLookup` for kernel-side name-based reads:

```cpp
// Kernel-side: read by name, type-checked at compile time
rt::ParamLookup lookup(/* entries from SceneDescriptor */);
int spp = lookup.read<int>("spp_frame");
float3 eye = lookup.read<float3>("cam_eye");
```

Kernel-specific params are declared in the YAML under `params:` and read
by name — no enum indexing needed.

## File structure

```
include/sycl-sandbox/
  variant.h           — visit<>() — generic compile-time variant dispatch
  rt/
    math.h            — float3, operators, RNG
    types_fwd.h       — Ray, HitRecord, ScatterRecord, Aabb
    types.h           — Hittable variant, Material variant, Object class
    helpers.h         — random_in_unit_sphere, reflect, refract, schlick
    variant.h         — forward to sycl-sandbox/variant.h
    camera.h          — Camera struct, lookat()
    scene_data.h      — Handle, SceneView, SceneBuilder, handle dispatch
    trace.h           — trace(), render_main<>()
    scene.h           — Axis enum, quad_corner
    hittables/
      sphere.h        — Sphere class + sphere() factory + aabb()
      triangle.h      — Triangle class + triangle() factory + aabb()
      quad.h          — Quad class + quad() factory + aabb()
      box.h           — Box class + box() factory + aabb()
    materials/
      lambertian.h    — Lambertian + lambertian()
      metal.h         — Metal + metal()
      dielectric.h    — Dielectric + dielectric()
      diffuse_light.h — DiffuseLight + diffuse_light()
      textured_lambertian.h — TexturedLambertian + textured_lambertian()
    textures/
      texture.h       — Texture variant + sample() dispatch
      solid_color.h   — SolidColor texture (constant colour)
      colorchecker.h  — ColorChecker texture (infinite tiling chart)
```
