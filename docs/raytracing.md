# Raytracing Library

## Overview

The shared `include/sycl-sandbox/rt/` library provides a complete raytracing pipeline that
any kernel can use.  A raytracer kernel is just a normal kernel that happens
to include `rt/` headers and call `rt::render()` from its single
`kernel_entry(const rt::Context*)`.  The kernel owns only its scene geometry
(built host-side and handed in via `ctx.scene`) and material parameters;
everything else (camera, path tracing, accumulation) is handled by the
library.

```
┌──────────────────────────────────────────────────────┐
│  kernel.cpp (your code)                              │
│                                                      │
│  kernel_entry(ctx) → read params from ctx,          │
│                      build background_fn             │
│  rt::render<KernelName>(ctx, background_fn)    │
│                                                      │
│  ┌──────────────────────────────────────────────────┐│
│  │  include/sycl-sandbox/rt/ (shared library)       ││
│  │                                                  ││
│  │  render() → reads standard params,               ││
│  │              sets up camera, launches            ││
│  │              foreach_pixel → trace()             ││
│  │                → handle_hit/scatter/emit         ││
│  │                 → switch dispatch per-type       ││
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

Type IDs: 0 = Sphere, 1 = Triangle, 2 = Quad, 3 = Box, 4 = Portal,
5 = Mesh (hittable);
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
// The host hands the non-owning view to the kernel each frame via ctx.scene:
ctx.scene = &scene_data.view();
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

`trace()` takes the renderer's background function (`BgFn`) and an
optional `rt::Context` (default empty — see "Tracing and the debug
collector" below).  When a ray escapes the scene (no handle hit) — or
reaches the bounce limit — the background function is evaluated at the
*current* ray and folded into the path, attenuated by the throughput so
far:

```cpp
if (!closest_hit) {
    return mul(attenuation, background_fn(ray_in_out));  // sky colour
}
```

This is what makes sky-lit scenes (no emissive lights, e.g. the "one
weekend" style) render at all: without it every path ends black and the
whole image collapses to a flat background gradient (the `render_main`
black-substitution hack that previously masked this was removed).

### Tracing and the debug collector

The whole pipeline — `trace()`, `bvh_hit()`, every primitive's `hit()`,
every material's `scatter()`/`emit()` and every texture `sample()` —
takes a `const rt::Context &ctx` (include/sycl-sandbox/context.h) that
rides the *device profiler ring* (`ctx.prof`), the per-work-item
`linear_id`, the per-frame `TraceCounters` and the *debug collector*
(`ctx.collector`).  The collector is an optional step/event ring
(`rt::TraceCollector`, rt/collector.h): when the context carries an
active collector, every pipeline stage appends deep metadata to it — BVH
nodes entered, hit tests performed, scatter/emit evaluations, texture
samples — plus one `TraceStepRecord` per bounce.  An inactive collector
(the default `rt::Context{}`) makes every hook a no-op the compiler
eliminates, so the render path pays zero cost.  Both BVH traversals
report: the scene BVH (`bvh_hit()`) and each mesh's per-mesh BVH
(`Mesh::bvh_hit()`) emit a `HittableBvhNode` event per node entered,
which the debug view reconstructs into `visited_bvh_nodes()` /
`visited_hittable_nodes()`.

The scene-debug view (`trace_ray_debug()`, src/ui/scene_debug/ray_trace.h)
arms a `TraceCollector` over host stack rings, runs the *real* `trace()`,
and reconstructs the per-bounce steps (with their events) from the rings
afterwards.  Per-hit colours are recovered by back-propagation (emit +
attenuation × continuation, with the kernel's exact sky fold on
escape/bounce-limit), so `steps[0].color` is bit-identical to
`rt::trace()`'s return value — see docs/architecture.md → "Scene debug
window".

## Anatomy of a raytracer kernel

The kernel is a single `extern "C"` entry point, called once per frame.
The host builds the scene (`SceneBuilder`/`SceneData`, see above) and hands
it in `ctx.scene`; the kernel reads the standard params (camera, SPP,
bounces) through `ctx.params` and renders every pixel via the shared
`rt::render()` path.  Kernels keep NO state — no globals, no
init/shutdown (kernels/raytracer/kernel.cpp):

```cpp
#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/rt/trace.h>      // rt::trace(), rt::render()

using namespace rt;

// ── Single entry point — one function, no ops, no globals ─────────────
extern "C" void kernel_entry(const rt::Context *ctxp) {
    const rt::Context &ctx = *ctxp;
    PROFILER_ZONE("render");

    // Background colour from the scene params — sky gradient from white
    // at the horizon to the configured background colour.
    const auto bg = ctx.params->read<float3>("background");
    const auto &scene = *ctx.scene;
    const auto background_fn = [bg](const Ray &ray) -> float3 {
        const float t = 0.5f * (ray.dir.y + 1.0f);
        return lerp({1.f, 1.f, 1.f}, bg, t);
    };

    // Enqueue one frame of samples for every pixel (enqueue-only
    // contract — the host chains tone-map + display on the in-order
    // queue after this returns).  KernelName is an explicit SYCL kernel
    // name tag, unique per kernel .so.
    rt::render<class RaytracerPixelKernel>(ctx, background_fn);

    // Scene-derived stats are host-side — publish right after enqueueing.
    if (ctx.stats) {
        ctx.stats->write<int>("num_objects", scene.num_handles);
        ctx.stats->write<int>("num_bvh_nodes", scene.num_bvh_nodes);
        ctx.stats->write<int>("num_lights", scene.num_lights);
    }
}
```

`rt::render<KernelName>()` reads the standard params from `ctx.params`
(`spp_frame`, `max_bounces`, `transparent_backfaces`, camera — see
"Standard parameter layout" below), walks every pixel through
`rt::Runtime::foreach_pixel<KernelName>()` (a SYCL `parallel_for` on the
GPU/CPU backends, plain loops in native mode), and accumulates samples
into `ctx.accum`.

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

## Triangle meshes

A `Mesh` hittable is a window into the scene's global triangle array
(`hittables::Mesh {first_triangle, num_triangles}` — indices into
`SceneView::triangles`).  No per-mesh device allocations: the mesh's
triangles are pushed into the same per-type triangle array as standalone
`Triangle` hittables and uploaded as one buffer.  Texture UVs come from
the winning triangle's barycentric coordinates (u weights vertex b, v
weights vertex c), like a standalone `Triangle`.

```cpp
// Host side (SceneBuilder):
scene.add_mesh(triangles, materials::metal({0.9f, 0.85f, 0.7f}, 0.15f));
```

Meshes are dispatched by `HittableType::Mesh` in `handle_hit()`; their
surface area (sum of triangle areas) feeds the light list for emissive
mesh materials.

### Per-mesh BVH

The scene BVH culls a mesh as a single box (the union of its triangle
AABBs), which is wasted work for a large mesh: a ray enters the box and
then linearly scans every triangle.  To fix that, `SceneBuilder::build_mesh_bvhs()`
builds a **second BVH per mesh** during scene loading, replacing the
linear scan with a proper acceleration structure.

Construction reuses the same pre-order flat builder as the scene BVH:
`build_flat_bvh()` splits on the longest AABB axis at the median
(over the triangles' centroids) and emits a compact `BvhNode` array with
`left == BVH_LEAF` marking leaves.  The two BVHs live in the same
`BvhNode` struct but mean different things in the leaf:

- **Scene BVH** leaves: `right` = **handle index** (which hittable).
- **Mesh BVH** leaves: `right` = **absolute triangle index** into
  `SceneView::triangles` (direct, no per-mesh offset math at traversal
  time).

Per-mesh trees are appended into one `SceneView::mesh_bvh_nodes` buffer
(one device upload), and each `Mesh` stores `bvh_root` — the index of
its tree's root node in that buffer.  Meshes with fewer than two
triangles get `bvh_root = -1` and keep the linear scan.

Because every tree shares one flat array, `build_mesh_bvhs()` rewrites
the builder's *local* child links (0-based within a tree) into
*absolute* indices by adding the tree's start offset before appending —
leaf nodes are untouched (`right` already holds an absolute triangle
index).  Without that offset, only the first mesh (tree at offset 0)
traverses correctly and every other mesh silently misses, which the
`diag mesh` structural validation guards against (`node < bvh_root` is
an out-of-tree child link).

`Mesh::hit()` traverses the mesh BVH when `mesh_bvh_nodes` is present
and `bvh_root >= 0`: an iterative 64-entry stack with AABB pruning and
a `t_max` that tightens after each triangle hit.  The same traversal
emits a `TraceEventKind::HittableBvhNode` per node entered when a trace
collector is armed, so the scene debugger can highlight the visited
mesh-BVH boxes and count `mesh bvh nodes` per bounce (see
"Scene debugger" below).  Mesh-BVH node events are only emitted when the
kernels' trace collection is enabled, keeping the default render path
free of the per-node overhead.

Zero-area (degenerate) triangles are dropped by `SceneBuilder::add_mesh()`
before upload, and the `Triangle` constructor normalizes safely (a
degenerate triangle gets a clean zero normal instead of `NaN`).  This
matters on GPU fast-math builds: a degenerate triangle's `norm(0) = NaN`
normal was observed to leak through `Triangle::hit()`'s denominator check
and poison the pixel accumulation buffer — the whole image progressively
blackened as samples accumulated.  Degenerate triangles can never be hit,
so dropping them is free.

Zero-area (degenerate) triangles are dropped by `SceneBuilder::add_mesh()`
before upload, and the `Triangle` constructor normalizes safely (a
degenerate triangle gets a clean zero normal instead of `NaN`).  This
matters on GPU fast-math builds: a degenerate triangle's `norm(0) = NaN`
normal was observed to leak through `Triangle::hit()`'s denominator check
and poison the pixel accumulation buffer — the whole image progressively
blackened as samples accumulated.  Degenerate triangles can never be hit,
so dropping them is free.

### Loading meshes from STL files

`include/sycl-sandbox/stl_loader.h` (HOST-ONLY — never included by
kernels/device code) loads ASCII and binary STL files (auto-detected via
the binary size invariant `84 + 50·count == file size`, falling back to
token-based ASCII parsing for `solid`-prefixed files).  STL normals are
discarded and recomputed from the vertices.  `apply_transform()` places
the model: scale → rotate (Euler degrees, applied Z → Y → X) → translate.

YAML scenes load meshes with a `type: mesh` object:

```yaml
- type: mesh
  file: models/uv_sphere.stl     # relative to the scene YAML's directory
  position: [-0.9, 1.0, 0.4]     # optional translation
  rotation: [0, 45, 0]           # optional Euler degrees (Z → Y → X)
  scale: 1.4                     # optional float or vec3
  material:
    type: metal
    albedo: [0.9, 0.85, 0.7]
    roughness: 0.15
```

`file` may be a `$param` / `$data_source` reference.  See
`scenes/mesh_demo.yaml` for a full example.

> **STL coordinate convention.**  STL files carry no up-axis metadata —
> many tools export Z-up (e.g. `scenes/models/duck.stl`, whose +Z is the
> vertical).  A Z-up model loaded unrotated lies flat in this Y-up scene.
> Verify any model's orientation headlessly before committing a rotation:
>
> ```
> sycl-sandbox diag stl-viz <file.stl> <outprefix> [--rx --ry --rz]
> ```
>
> This renders shaded front/side/top/three-quarter PNGs through the
> project's own loader and `apply_transform`, so a candidate Euler fix can
> be confirmed end-to-end.  Prefer a single-axis rotation: the fixed-axis
> Z → Y → X order makes combined rotations non-intuitive (applying `ry`
> after `rx` tumbles rather than yawing).

## AABB and acceleration structures

Every hittable type provides an `aabb()` method returning the enclosing
axis-aligned bounding box.  The `SceneBuilder::build()` computes per-handle
AABBs and stores them in `SceneView::aabbs`.

The `aabb_hit()` utility performs a fast slab-method ray-AABB test, used
for broad-phase rejection in BVH traversal and available for future
importance sampling.

BVH construction (`SceneBuilder::build_bvh()`) builds a binary BVH over
the scene's per-handle AABBs and flattens it to a contiguous pre-order
array of `BvhNode` (`{Aabb bounds, uint32 left, uint32 right}`,
`left == BVH_LEAF` marks a leaf whose `right` is the handle index).
Both the scene BVH and every per-mesh BVH come from the same
`build_flat_bvh()` helper (median split along the longest axis over
centroids; root = node 0; pre-order layout).  Each array is uploaded to
device memory as a single buffer (`SceneView::bvh_nodes` and
`SceneView::mesh_bvh_nodes`) — no pointers, all navigation by index, in
the same spirit as a `boost::flat_map`.  The renderer's closest-hit
query is `bvh_hit()`, an iterative stack-based traversal over that flat
array; `Mesh::bvh_hit()` is the analogous per-mesh traversal (see
"Per-mesh BVH" above).  Light list computation is stubbed but the data
slot is in place in `SceneView`.

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
in `rt::render()`/`Params::from_lookup`) is passed through to the sampler,
enabling time-varying procedural textures later.

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
  context.h           — rt::Context — the single kernel ABI (prof ring,
                        counters, collector, params, scene, framebuffer)
  rt/
    math.h            — float3, operators, RNG
    types_fwd.h       — Ray, HitRecord, ScatterRecord, HittableType,
                        MaterialType
    aabb.h            — Aabb, aabb_from_points/merge/hit, BvhNode, BVH_LEAF
    types.h           — Hittable variant, Material variant, Object class
    helpers.h         — random_in_unit_sphere, reflect, refract, schlick
    variant.h         — forward to sycl-sandbox/variant.h
    collector.h       — TraceCollector ring + TraceCounters (no-ops when
                        inactive)
    trace.h           — trace(), render<>()
    hittables/
      sphere.h        — Sphere class + sphere() factory + aabb()
      triangle.h      — Triangle class + triangle() factory + aabb()
      quad.h          — Quad class + quad() factory + aabb()
      box.h           — Box class + box() factory + aabb()
      portal.h        — Portal class + portal() factory (bidirectional)
      mesh.h          — Mesh window into the scene triangle array
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
      text.h          — Text texture (debug)
      blend.h         — Blend texture (mix of two textures)
  scene/
    data.h            — Handle, SceneView, SceneBuilder, handle dispatch
    camera.h          — Camera struct, lookat()
```
