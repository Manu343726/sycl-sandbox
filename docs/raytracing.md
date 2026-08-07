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

### SceneView (device-side, non-owning)

Read-only, NON-OWNING scene data passed to device kernels by value.
Never allocates or frees — the buffers it points at are owned by the
host-side `SceneData` (the upper layer), which is NEVER passed to the
render function.  Because it is just pointers + counts, a `SceneView`
can also point at stack arrays or device memory (`sycl::malloc_device`
allocations from `SceneBuilder::build()`).

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

// Owning layer (host-side only, never passed to the render function)
struct SceneData : SceneView {
    bool empty() const;            // handles == nullptr
    SceneView view() const;        // non-owning snapshot for kernels
    void free(rt::Runtime *rt);    // deallocates every buffer
};
```

### SceneBuilder (host-side)

Accumulates hittable-material pairs via `add()`, then uploads per-type
arrays via `build()` — which returns an OWNING `SceneData`:

```cpp
SceneBuilder scene;
scene.add({hittables::sphere({0, 0, 0}, 1), materials::lambertian({1, 1, 1})});
scene.add({hittables::box(-1, -1, -1, 2, 2, 2), materials::metal({0.8f, 0.8f, 0.8f}, 0)});
SceneData scene_data = scene.build(queue);   // upload to device, owns buffers
// Kernels render through the non-owning view:
render_main(..., scene_data.view(), ...);
// Host frees when done:
scene_data.free(queue);
```

### Dispatch

The `trace()` function finds the closest hit via the flat BVH
(`bvh_hit()`, a stack-based traversal over the contiguous `BvhNode[]`
array with AABB culling) and then calls `handle_hit()`, `handle_scatter()`,
`handle_emit()` — inline functions that switch on the packed type tag
and index into the correct per-type array.  If no BVH was built
(`bvh_nodes == null`), it falls back to the linear per-handle scan:

```cpp
optional<BvhHitResult> bvh = bvh_hit(ray, t_min, t_max, scene);
// Fallback (no BVH):
for (int i = 0; i < scene.num_handles; i++) {
    auto hit = handle_hit(scene.handles[i], ray, t_min, t_max, scene);
    if (hit) { closest_hit = hit; hit_index = i; }
}
// After loop: dispatch emit/scatter on the winning handle
float3 emitted = handle_emit(winning_handle, *closest_hit, scene);
auto scattered = handle_scatter(winning_handle, ray, *closest_hit, rng, scene);
```

X-ray mode (`transparent_backfaces`) is handled inside `bvh_hit()` via the
`skip_backfaces` flag: back-face hits are ignored **without** tightening
`t_max`, so the ray continues past them to objects behind the surface.

### Sky lighting

`trace()` takes the renderer's background function (`BgFn`) as its last
argument.  When a ray escapes the scene (no handle hit) — or reaches the
bounce limit — the background function is evaluated at the *current* ray
and folded into the path, attenuated by the throughput so far:

```cpp
if (!closest_hit) {
    return mul(attenuation, background_fn(ray_in_out));  // sky colour
}
```

This is what makes sky-lit scenes (no emissive lights, e.g. the "one
weekend" style) render at all: without it every path ends black and the
whole image collapses to a flat background gradient (the `render_main`
black-substitution hack that previously masked this was removed).

## Anatomy of a minimal raytracer kernel

All parameters (including standard ones like SPP, bounces, camera) are
declared in the YAML scene file and managed by `SceneDescriptor`.  Kernels
read parameters by name via `ParamLookup::read<T>("name")`.

```cpp
#include <sycl-sandbox/rt/types.h>      // Object, Hittable, Material
#include <sycl-sandbox/rt/trace.h>      // rt::render_main()
#include <sycl-sandbox/rt/scene_data.h> // SceneBuilder, SceneData, SceneView
#include <sycl-sandbox/rt/hittables/quad.h>
#include <sycl-sandbox/rt/materials/lambertian.h>
#include <sycl-sandbox/rt/materials/diffuse_light.h>

using namespace rt;
using rt::materials::lambertian;
using rt::materials::diffuse_light;

// ── Kernel-specific params are read by name from the buffer ───────────

// ── Scene state ───────────────────────────────────────────────────────
static SceneBuilder scene;
static SceneData scene_data = {};   // owning; kernel never sees this
static SceneView scene_view = {};   // non-owning snapshot for render_kernel

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
    scene_data = scene.build(queue);   // upload per-type arrays to device
    scene_view = scene_data.view();    // kernel renders through this
}

// ── Render (called every frame) ───────────────────────────────────────
extern "C" void render_kernel(sycl::queue* queue, int w, int h,
                               const void* params, void* accum, int si) {
    rt::render_main(queue, w, h, (const float*)params, (float*)accum, si,
                    scene_view,
                    [](const Ray&) -> float3 { return {0,0,0}; });
}

extern "C" void shutdown_kernel(sycl::queue* queue) {
    scene_data.free(queue);
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

BVH construction (`SceneBuilder::build_bvh()`) builds a binary BVH over
the scene's per-handle AABBs (median split along the longest axis) and
flattens it to a contiguous pre-order array of `BvhNode`
(`{Aabb bounds, uint32 left, uint32 right}`, `left == BVH_LEAF` marks a
leaf whose `right` is the handle index).  The array is uploaded to
device memory as a single buffer (`SceneView::bvh_nodes`) — no pointers,
all navigation by index, in the same spirit as a `boost::flat_map`.  The
renderer's closest-hit query is `bvh_hit()`, an iterative stack-based
traversal over that flat array.  Light list computation is stubbed but
the data slot is in place in `SceneView`.

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
| Portal    | the HIT shape's UVs, forwarded to the other shape (see Portals) |

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
using Texture = std::variant<SolidColor, ColorChecker, Text, Blend>;

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
| `Text` | renders a fixed string in an embedded 8×8 monospace bitmap font (public-domain `font8x8`, ASCII 32–126); initialized with the string, configurable glyph/background colour, cell size and origin in UV space; UVs outside the text return the background |
| `Blend` | alpha-blends up to 4 textures in order (source-over: `result = lerp(result, layer, alpha)`); layers are the *leaf* texture types (no recursion into the `Texture` variant) |

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
implemented as a **pair of hittables** (entry ↔ exit) and is
**bidirectional**: a ray hitting *either* shape reappears at the
corresponding point on the other one.  The parametric coordinates
(`u`/`v`) of the hit are used as input for the other hittable, so the
ray reappears at the matching point on the counterpart surface.  The
mapping is exactly inverse to `hit()`: every hittable provides
`point_at_uv(u, v)` that reconstructs a surface point from the same UV
convention its `hit()` fills into `HitRecord::u/v`.

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
  `portal_origin` = other-surface point nudged by an epsilon along that
  surface's normal in the travel direction (avoids re-hitting it),
  `portal_dir` = incoming ray direction.
- `HitRecord::t` keeps the **hit shape's distance** (entry *or* exit),
  so closest-hit ordering works without special-casing.
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
- Portals are **bidirectional**: `hit()` tests both shapes with the same
  t-range and the closest hit wins; hitting the exit teleports back to
  the entry.  Both surfaces work through the same pair.
- The portal's AABB merges **both** shapes, so BVH culling never skips
  an exit-side hit.

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
`scenes/portal_rooms.yaml` for the two-room demo.

## Data sources & procedural objects

Scene geometry can be generated procedurally from **data sources** —
array-valued generators whose values can be referenced per-instance by
objects.  Data sources may reference scene params (`$name`) and
previously generated sources.

```yaml
data_sources:
  - id: positions          # generator outputs an array of values
    type: vec3[]
    generator: random_range
    inputs:
      count: $num_spheres  # references a param
      min: [-10, 0.2, -10]
      max: [10, 0.2, 10]

  - id: material_choice
    type: string[]
    generator: weighted_choice
    inputs:
      count: $num_spheres
      choices:
        - value: lambertian   # 60% of instances
          weight: 0.6
        - value: metal
          weight: 0.25
        - value: dielectric
          weight: 0.15
```

Generators: `random_range` (uniform between `min`/`max`, scalar or
vec3) and `weighted_choice` (string or int choices with weights).  The
legacy map form (`data: {name: {random_range: {...}}}`) is also
accepted.  A `data_sources` value is referenced like a param: `$id`.

Objects can then be **expanded** into `count` instances, with array
references yielding one value per instance:

```yaml
objects:
  - count: $num_spheres
    hittable:
      type: sphere
      radius: 0.2
      center: $positions       # positions[i] for instance i
    material:
      source: $material_choice # string per instance
      mapping:                 # per-choice material template
        lambertian:
          type: lambertian
          albedo: $colors      # colors[i]
        metal:
          type: metal
          albedo: $colors
          fuzz: $fuzz          # fuzz[i]
        dielectric:
          type: dielectric
          ir: 1.5
```

Data sources are re-evaluated after param changes, so live UI edits
(e.g. `num_spheres`) regenerate the arrays.  See
`scenes/one_weekend_final.yaml` for a complete example.

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
| `transparent_backfaces` | bool | `false` | X-ray mode: make back faces transparent — rays pass through surfaces hit from behind (`front_face == false`), so a camera outside an enclosed scene (e.g. the Cornell box) can see inside through the walls. |
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
