#pragma once
#include <sycl-sandbox/context.h>
#include <sycl-sandbox/rt/collector.h>
#include <sycl-sandbox/rt/aabb.h>
#include <sycl-sandbox/rt/types.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/variant.h>
#ifndef KERNEL_NATIVE
#include <sycl/sycl.hpp>
#endif
#include <algorithm>
#ifndef KERNEL_BUILD
#include <unordered_map>
#endif
#include <vector>

namespace rt {

/// Data-oriented scene representation: per-type arrays, handle-based dispatch,
/// optional BVH acceleration and light list for importance sampling.
///
/// Kernel init code builds the scene via SceneBuilder (host only).
/// The render loop reads SceneView (device only, trivially copyable).

// ── Type tags ──────────────────────────────────────────────────────────
// HittableType / MaterialType live in rt/types_fwd.h so deep pipeline
// code (primitive hit(), material scatter()/emit(), textures) can name
// their kind when reporting through the trace collector.

// ── Handle ─────────────────────────────────────────────────────────────

/// Lightweight reference to a hittable and material stored in separate
/// per-type arrays.  Each half packs a type tag (top 8 bits) and an index
/// within that type's array (lower 24 bits), giving 16 M entries max.
struct Handle {
    uint32_t hittable; ///< (HittableType << 24) | index_in_type_array
    uint32_t material; ///< (MaterialType << 24) | index_in_type_array
};

/// Pack a type tag and index into a 32-bit handle field.
inline uint32_t pack_handle(uint32_t tag, uint32_t index) {
    return (tag << 24) | index;
}

/// Extract the type tag from a packed handle field.
inline uint32_t handle_tag(uint32_t packed) {
    return packed >> 24;
}

/// Extract the index from a packed handle field.
inline uint32_t handle_index(uint32_t packed) {
    return packed & 0x00FFFFFF;
}

// ── Light info (future) ───────────────────────────────────────────────

/// Precomputed information for a light-emitting object, used by importance
/// sampling to select light sources proportional to their contribution.
struct LightInfo {
    Handle handle;   ///< Which object is the light.
    float area;      ///< Surface area of the light geometry.
    float3 emission; ///< Emission colour × intensity.
};

// ── Type traits ────────────────────────────────────────────────────────

/// Compile-time mapping from a C++ hittable type to its HittableType tag.
template <typename T>
struct HittableTag;
template <>
struct HittableTag<hittables::Sphere> {
    static constexpr HittableType value = HittableType::Sphere;
};
template <>
struct HittableTag<hittables::Triangle> {
    static constexpr HittableType value = HittableType::Triangle;
};
template <>
struct HittableTag<hittables::Quad> {
    static constexpr HittableType value = HittableType::Quad;
};
template <>
struct HittableTag<hittables::Box> {
    static constexpr HittableType value = HittableType::Box;
};
template <>
struct HittableTag<hittables::Mesh> {
    static constexpr HittableType value = HittableType::Mesh;
};

/// Compile-time mapping from a C++ material type to its MaterialType tag.
template <typename T>
struct MaterialTag;
template <>
struct MaterialTag<materials::Lambertian> {
    static constexpr MaterialType value = MaterialType::Lambertian;
};
template <>
struct MaterialTag<materials::Metal> {
    static constexpr MaterialType value = MaterialType::Metal;
};
template <>
struct MaterialTag<materials::Dielectric> {
    static constexpr MaterialType value = MaterialType::Dielectric;
};
template <>
struct MaterialTag<materials::DiffuseLight> {
    static constexpr MaterialType value = MaterialType::DiffuseLight;
};
template <>
struct MaterialTag<materials::TexturedLambertian> {
    static constexpr MaterialType value = MaterialType::TexturedLambertian;
};

// ── SceneView (device-side, non-owning) ───────────────────────────────

/// Read-only, NON-OWNING scene data passed to device kernels by value
/// (captured in SYCL lambdas).  Never allocates or frees — the buffers
/// it points at are owned by the host-side SceneData (the upper layer),
/// which is NOT passed to the render function.
///
/// Nullable arrays with counts — unset features have null + zero.
/// Because it is a pure view it can also point at stack arrays.
struct SceneView {
    // ── Handles (always present) ──────────────────────────────────────
    Handle *handles;
    int num_handles;

    // ── Hittable arrays (per-type) ────────────────────────────────────
    hittables::Sphere *spheres;
    int num_spheres;
    hittables::Triangle *triangles;
    int num_triangles;
    hittables::Quad *quads;
    int num_quads;
    hittables::Box *boxes;
    int num_boxes;
    hittables::Portal *portals;
    int num_portals;
    hittables::Mesh *meshes;
    int num_meshes;

    // ── Material arrays (per-type) ────────────────────────────────────
    materials::Lambertian *lambertians;
    int num_lambertians;
    materials::Metal *metals;
    int num_metals;
    materials::Dielectric *dielectrics;
    int num_dielectrics;
    materials::DiffuseLight *diffuse_lights;
    int num_diffuse_lights;
    materials::TexturedLambertian *textured_lambertians;
    int num_textured_lambertians;

    // ── Optional: per-handle AABBs (computed during build) ────────────
    Aabb *aabbs;

    // ── Optional: BVH (built via SceneBuilder::build_bvh()) ──────────
    BvhNode *bvh_nodes;
    int num_bvh_nodes;
    int bvh_root;

    // ── Optional: per-mesh BVHs (built via SceneBuilder::build_mesh_bvhs()) ──
    // Flat concatenation of one BVH per Mesh hittable; each mesh's root is
    // stored in its hittables::Mesh::bvh_root (index into this array).
    // Mesh::hit() traverses it; without it meshes fall back to a linear
    // scan of their triangle window.  Leaf nodes reference ABSOLUTE
    // triangle indices into the triangles array.
    BvhNode *mesh_bvh_nodes;
    int num_mesh_bvh_nodes;

    // ── Optional: light list (built during build if lights exist) ─────
    LightInfo *lights;
    int num_lights;
};

// ── SceneData (host-side owner) ───────────────────────────────────────

/// Owning scene buffers — the upper layer that manages the memory a
/// SceneView points at.  Allocated through the Runtime abstraction
/// (SYCL device memory, or plain heap in software mode).
///
/// Lives on the host (e.g. inside HostScene) and is NEVER passed to the
/// raytracing render function — kernels only ever receive the non-owning
/// SceneView (via set_scene_view), so the render path cannot free or
/// mutate the buffers.
struct SceneData : SceneView {
    /// True while no buffers have been built (or after free()).
    bool empty() const {
        return handles == nullptr;
    }

    /// Non-owning snapshot of the buffers, safe to pass to the render
    /// function or capture by value into kernel lambdas.
    SceneView view() const {
        return *this;
    }

    /// Free all buffers through the Runtime abstraction (SYCL device
    /// or plain delete[]), resetting the scene to empty.
    void free(rt::Runtime *rt);
};

// ── Dispatch functions (device-capable) ────────────────────────────────

/// Test a handle's hittable against a ray.
///
/// \param ctx per-call kernel context: forwarded to the primitive's hit()
///        so it can record profiler zones and report hit tests through
///        the trace collector.
inline optional<HitRecord>
handle_hit(Handle h, const Ray &ray, float t_min, float t_max, const SceneView &scene,
           const Context &ctx = Context{}) {
            PROFILER_FUNCTION();

    auto type = static_cast<HittableType>(handle_tag(h.hittable));
    uint32_t idx = handle_index(h.hittable);
    switch ( type ) {
        case HittableType::Sphere:
            return scene.spheres[idx].hit(ray, t_min, t_max, ctx);
        case HittableType::Triangle:
            return scene.triangles[idx].hit(ray, t_min, t_max, ctx);
        case HittableType::Quad:
            return scene.quads[idx].hit(ray, t_min, t_max, ctx);
        case HittableType::Box:
            return scene.boxes[idx].hit(ray, t_min, t_max, ctx);
        case HittableType::Portal:
            return scene.portals[idx].hit(ray, t_min, t_max, ctx);
        case HittableType::Mesh:
            return scene.meshes[idx].hit(ray, t_min, t_max, scene.triangles,
                                         scene.mesh_bvh_nodes, ctx);
    }
    return nullopt;
}

/// Scatter a ray off a handle's material.
inline optional<ScatterRecord> handle_scatter(Handle h,
                                              const Ray &incoming_ray,
                                              const HitRecord &hit,
                                              RNG &rng,
                                              const SceneView &scene,
                                              const Context &ctx = Context{}) {
    PROFILER_FUNCTION();
    auto type = static_cast<MaterialType>(handle_tag(h.material));
    uint32_t idx = handle_index(h.material);
    switch ( type ) {
        case MaterialType::Lambertian:
            return scene.lambertians[idx].scatter(incoming_ray, hit, rng, ctx);
        case MaterialType::Metal:
            return scene.metals[idx].scatter(incoming_ray, hit, rng, ctx);
        case MaterialType::Dielectric:
            return scene.dielectrics[idx].scatter(incoming_ray, hit, rng, ctx);
        case MaterialType::DiffuseLight:
            return scene.diffuse_lights[idx].scatter(incoming_ray, hit, rng, ctx);
        case MaterialType::TexturedLambertian:
            return scene.textured_lambertians[idx].scatter(incoming_ray, hit, rng, ctx);
    }
    return nullopt;
}

/// Emit light from a handle's material.
inline float3 handle_emit(Handle h, const HitRecord &hit, const SceneView &scene,
                          const Context &ctx = Context{}) {
    auto type = static_cast<MaterialType>(handle_tag(h.material));
    uint32_t idx = handle_index(h.material);
    switch ( type ) {
        case MaterialType::Lambertian:
            return scene.lambertians[idx].emit(hit, ctx);
        case MaterialType::Metal:
            return scene.metals[idx].emit(hit, ctx);
        case MaterialType::Dielectric:
            return scene.dielectrics[idx].emit(hit, ctx);
        case MaterialType::TexturedLambertian:
            return scene.textured_lambertians[idx].emit(hit, ctx);
        case MaterialType::DiffuseLight:
            return scene.diffuse_lights[idx].emit(hit, ctx);
    }
    return {0, 0, 0};
}

// ── Trace debug collection ────────────────────────────────────────────

/// Per-bounce snapshot of a path traced by rt::trace(): everything the
/// scene-debug view needs to draw and explain one step (see
/// src/ui/scene_debug/ray_trace.h).  Kept a plain POD so it is
/// device-compilable — the real kernel path leaves the collector
/// inactive (rt/collector.h), whose hooks are empty and eliminated by
/// the compiler, so the instrumentation costs nothing on the rendering
/// path.
struct TraceStepRecord {
    Ray ray = {};                          ///< ray entering this step
    bool hit = false;                      ///< did the ray hit an object
    Handle handle = {};                    ///< hit handle (valid when hit)
    HitRecord record = {};                 ///< closest hit record
    float3 emit = {0, 0, 0};               ///< emission of the hit material
    bool scattered = false;                ///< did the material scatter
    Ray scattered_ray = {};                ///< continuation ray (when scattered)
    float3 scatter_attenuation = {1, 1, 1}; ///< this scatter's attenuation
    float3 throughput = {1, 1, 1};         ///< running attenuation entering
                                           ///< the step (product so far)
    bool escaped = false;                  ///< ray left the scene (no hit)
    bool absorbed = false;                 ///< material absorbed the ray
    bool bounce_limit = false;             ///< path cut short by max_bounces
    uint32_t event_start = 0;              ///< event-ring position captured
                                           ///< when the step finished; the
                                           ///< events that fired while this
                                           ///< step was evaluated are
                                           ///< [prev.event_start,
                                           ///<  this.event_start) (or
                                           ///< [0, this.event_start) for the
                                           ///< first step).  Set by
                                           ///< TraceCollector::record_step.
};

/// Append a completed path step to the collector's step ring.  Defined
/// here (not in rt/collector.h) because TraceStepRecord is complete in
/// this header.  Stamps the step's event_start with the event-ring
/// position at record time, so the reader can attach this step's events.
inline void TraceCollector::record_step(const TraceStepRecord &r) const {
    if ( !active() ) {
        return;
    }
    const uint32_t pos = bump(&header->step_pos) & (step_capacity - 1);
    steps[pos] = r;
    steps[pos].event_start = header->event_pos;
}

// ── BVH traversal ───────────────────────────────────────────────────

/// Result of a BVH closest-hit query: the hit record plus the handle
/// of the object that was hit (needed for material dispatch).
struct BvhHitResult {
    HitRecord record;
    Handle handle;
};

/// Iterative stack-based BVH traversal.  Returns the closest hit within
/// [t_min, t_max], or nullopt if the ray misses the entire BVH.
///
/// \param skip_backfaces X-ray mode: when true, hits from behind
///        (front_face == false) are ignored WITHOUT tightening t_max,
///        so the ray continues past them and can hit objects behind
///        the surface (e.g. seeing inside an enclosed scene).
/// \param ctx per-call kernel context: carries the device profiler
///        ring, the work-item id, and the trace collector — every node
///        whose AABB the ray enters is reported through
///        ctx.collector.on_bvh_node() (the scene-debug view highlights
///        them).  The kernel passes an inactive context: empty hooks,
///        zero cost.
inline optional<BvhHitResult>
bvh_hit(const Ray &ray, float t_min, float t_max, const SceneView &scene,
        bool skip_backfaces = false, const Context &ctx = Context{}) {
    PROFILER_FUNCTION();

    if ( !scene.bvh_nodes || scene.bvh_root < 0 ) {
        return nullopt;
    }

    optional<BvhHitResult> closest_hit;
    uint32_t stack[32];
    int stack_size = 0;
    stack[stack_size++] = (uint32_t)scene.bvh_root;

    while ( stack_size > 0 ) {
        PROFILER_ZONE("bvh_traverse");

        uint32_t node_index = stack[--stack_size];
        const BvhNode &node = scene.bvh_nodes[node_index];

        // Test ray against this node's bounding box
        if ( !aabb_hit(node.bounds, ray, t_min, t_max, ctx) ) {
            continue;
        }
        ctx.collector.on_bvh_node(node_index);

        // Leaf node: test the single object
        if ( node.left == BVH_LEAF ) {
            auto hit = handle_hit(scene.handles[node.right], ray, t_min, t_max, scene,
                                  ctx);
            if ( hit ) {
                // X-ray mode: pass through hits from behind — do not
                // tighten t_max, the ray must continue past them.
                if ( skip_backfaces && !hit->front_face ) {
                    continue;
                }
                closest_hit = BvhHitResult {*hit, scene.handles[node.right]};
                t_max = hit->t;
            }
            continue;
        }

        // Internal node: push children (right first so left is processed first)
        stack[stack_size++] = node.right;
        stack[stack_size++] = node.left;
    }

    return closest_hit;
}

// ── SceneBuilder (host-side) ──────────────────────────────────────────

#ifndef KERNEL_BUILD

/// Host-side scene builder.  Accumulates hittable/material pairs via add(),
/// then uploads per-type arrays to device memory via build().
///
/// Usage in init_kernel():
/// @code
///   SceneBuilder scene;
///   scene.add(hittables::sphere({0, 0, 0}, 1), materials::lambertian({1, 1, 1}));
///   scene.add(hittables::box(-1, -1, -1, 2, 2, 2), materials::metal({0.8f, 0.8f, 0.8f}, 0));
///   scene_data = scene.build(queue);      // owns the buffers
///   render_main(..., scene_data.view());  // kernels see the non-owning view
/// @endcode
class SceneBuilder {
public:
    SceneBuilder() = default;

    /// Add a hittable-material pair to the scene.
    /// The hittable and material are dispatched to their per-type arrays.
    void add(Hittable h, Material m);

    /// Add a portal (entry <-> exit) to the scene.    /// Portals are BIDIRECTIONAL: a ray hitting either shape teleports
    /// to the other.  They are instanced as regular objects with a
    /// material.  The default is a dummy white Lambertian: every
    /// material's scatter() teleports portal records (see
    /// rt::portal_scatter()), so the ray continues from the other
    /// surface.  Pass a material to instance a portal with real
    /// behavior — e.g. DiffuseLight for an emissive portal (glows; the
    /// path terminates with its emission).  Emissive portals are NOT
    /// added to the light list (surface area undefined).
    void add_portal(hittables::PortalShape entry, hittables::PortalShape exit);
    void add_portal(hittables::PortalShape entry, hittables::PortalShape exit,
                    Material material);

    /// Add an Object (hittable + material pair) to the scene.
    void add(const Object &obj) {
        add(obj.hittable, obj.material);
    }

    /// Add a triangle mesh: copies the triangles into the scene's
    /// per-type triangle array and adds ONE handle whose hittable is a
    /// Mesh window into that range (see hittables::Mesh).  The mesh's
    /// AABB is the union of its triangle AABBs, so the BVH culls the
    /// whole mesh as a single object.  Empty meshes are skipped.
    void add_mesh(const std::vector<hittables::Triangle> &triangles,
                  const Material &material, bool smooth = false);

    /// Optional: build a BVH over the scene for acceleration.
    /// Must be called before build().
    void build_bvh();

    /// Optional: build one BVH per Mesh hittable (over its triangle
    /// window), so hit() culls the mesh interior instead of scanning it
    /// linearly.  Must be called after all add()/add_mesh() calls and
    /// before build().  Meshes without a per-mesh BVH (none built, or
    /// fewer than 2 triangles) fall back to a linear scan.
    void build_mesh_bvhs();

    /// Upload all accumulated data to device/host memory using the Runtime
    /// abstraction and return an owning SceneData.  When rt->queue is
    /// non-null the arrays are allocated via SYCL device memory; when null
    /// they are plain heap allocations (software mode).
    /// Computes per-handle AABBs and builds a light list if lights exist.
    /// The render function receives the non-owning data.view().
    SceneData build(rt::Runtime *rt);

    /// Return a pointer to the host-side AABB array (for debug visualization).
    const float *debug_aabbs() const {
        return reinterpret_cast<const float *>(aabbs_.data());
    }

    /// Return the number of AABBs (== number of objects added via add()).
    int debug_aabb_count() const {
        return (int)aabbs_.size();
    }

    /// Build debug geometry buffers from the current staging data.
    /// Populates debug_spheres_, debug_quads_, debug_boxes_ from the
    /// per-type arrays, matching each hittable with its material colour.
    /// Must be called after all add() calls, typically before or after build().
    void build_debug_geometry();

    const DebugSphere *debug_spheres() const {
        return debug_spheres_.data();
    }
    int debug_sphere_count() const {
        return (int)debug_spheres_.size();
    }
    const DebugQuad *debug_quads() const {
        return debug_quads_.data();
    }
    int debug_quad_count() const {
        return (int)debug_quads_.size();
    }
    const DebugBox *debug_boxes() const {
        return debug_boxes_.data();
    }
    int debug_box_count() const {
        return (int)debug_boxes_.size();
    }

private:
    /// Hittable staging vectors (one per type).
    std::vector<hittables::Sphere> spheres_;
    std::vector<hittables::Triangle> triangles_;
    std::vector<hittables::Quad> quads_;
    std::vector<hittables::Box> boxes_;
    std::vector<hittables::Portal> portal_pairs_;
    std::vector<hittables::Mesh> meshes_;

    /// Material staging vectors (one per type).
    std::vector<materials::Lambertian> lambertians_;
    std::vector<materials::Metal> metals_;
    std::vector<materials::Dielectric> dielectrics_;
    std::vector<materials::DiffuseLight> diffuse_lights_;
    std::vector<materials::TexturedLambertian> textured_lambertians_;

    /// Handle and AABB arrays (parallel, one entry per add() call).
    std::vector<Handle> handles_;
    std::vector<Aabb> aabbs_;

    /// Indices into handles_ for DiffuseLight materials (for light list).
    std::vector<uint32_t> light_handle_indices_;

    /// Debug geometry buffers (populated by build_debug_geometry()).
    std::vector<DebugSphere> debug_spheres_;
    std::vector<DebugQuad> debug_quads_;
    std::vector<DebugBox> debug_boxes_;

    /// BVH nodes (built via build_bvh(), before build()).
    std::vector<BvhNode> bvh_nodes_;
    int num_bvh_nodes_ = 0;
    int bvh_root_ = -1;

    /// Per-mesh BVH nodes (built via build_mesh_bvhs(), before build()).
    /// Flat concatenation of one BVH per Mesh hittable; each mesh's root
    /// is stored in its hittables::Mesh::bvh_root (index into this
    /// array), uploaded with the meshes_.  Leaves reference ABSOLUTE
    /// triangle indices into triangles_.
    std::vector<BvhNode> mesh_bvh_nodes_;
    int num_mesh_bvh_nodes_ = 0;

    /// Compute the surface area of a hittable referenced by a handle.
    float compute_area(Handle h) const;

    /// Push a material into its per-type staging array, returning the
    /// packed material half of the handle.  DiffuseLight materials are
    /// registered in the light list only when register_light is true
    /// (portals are excluded: their surface area is undefined for
    /// importance sampling).
    uint32_t push_material(const Material &m, bool register_light);

    /// Allocate a device/host array and copy a host vector into it (via Runtime).
    template <typename T>
    static T *upload_array(rt::Runtime *rt, const std::vector<T> &host_vec);
};

// ── SceneData::free implementation ─────────────────────────────────────

/// Free all scene buffers using the Runtime abstraction.
/// Uses rt->dealloc (which handles SYCL device or plain delete[]).
inline void SceneData::free(rt::Runtime *rt) {
    auto de = [&](auto *&ptr) { if (ptr) { rt->dealloc(ptr); ptr = nullptr; } };
    de(handles);             num_handles = 0;
    de(spheres);             num_spheres = 0;
    de(triangles);           num_triangles = 0;
    de(quads);               num_quads = 0;
    de(boxes);               num_boxes = 0;
    de(portals);             num_portals = 0;
    de(meshes);              num_meshes = 0;
    de(lambertians);         num_lambertians = 0;
    de(metals);              num_metals = 0;
    de(dielectrics);         num_dielectrics = 0;
    de(diffuse_lights);      num_diffuse_lights = 0;
    de(textured_lambertians); num_textured_lambertians = 0;
    de(aabbs);
    de(bvh_nodes);           num_bvh_nodes = 0; bvh_root = -1;
    de(mesh_bvh_nodes);      num_mesh_bvh_nodes = 0;
    de(lights);              num_lights = 0;
}

// ── SceneBuilder::add implementation ───────────────────────────────────

inline void SceneBuilder::add(Hittable h, Material m) {
    Handle handle;
    Aabb box;

    visit(h, [&](const auto &hittable) {
        using H = std::decay_t<decltype(hittable)>;
        constexpr auto tag = static_cast<uint32_t>(HittableTag<H>::value);
        if constexpr ( std::is_same_v<H, hittables::Sphere> ) {
            handle.hittable = pack_handle(tag, (uint32_t)spheres_.size());
            spheres_.push_back(hittable);
        } else if constexpr ( std::is_same_v<H, hittables::Triangle> ) {
            handle.hittable = pack_handle(tag, (uint32_t)triangles_.size());
            triangles_.push_back(hittable);
        } else if constexpr ( std::is_same_v<H, hittables::Quad> ) {
            handle.hittable = pack_handle(tag, (uint32_t)quads_.size());
            quads_.push_back(hittable);
        } else if constexpr ( std::is_same_v<H, hittables::Box> ) {
            handle.hittable = pack_handle(tag, (uint32_t)boxes_.size());
            boxes_.push_back(hittable);
        } else if constexpr ( std::is_same_v<H, hittables::Mesh> ) {
            handle.hittable = pack_handle(tag, (uint32_t)meshes_.size());
            meshes_.push_back(hittable);
        }
        // Mesh AABBs need the triangle array (the mesh's triangles were
        // pushed into triangles_ before add() — see add_mesh()).
        if constexpr ( std::is_same_v<H, hittables::Mesh> ) {
            box = hittable.aabb(triangles_.data());
        } else {
            box = hittable.aabb();
        }
    });

    visit(m, [&](const auto &material) {
        handle.material = push_material(m, true);
    });

    handles_.push_back(handle);
    aabbs_.push_back(box);
}

// ── SceneBuilder::push_material ────────────────────────────────────────

inline uint32_t SceneBuilder::push_material(const Material &m, bool register_light) {
    uint32_t mat_handle = 0;
    visit(m, [&](const auto &material) {
        using M = std::decay_t<decltype(material)>;
        constexpr auto tag = static_cast<uint32_t>(MaterialTag<M>::value);
        if constexpr ( std::is_same_v<M, materials::Lambertian> ) {
            mat_handle = pack_handle(tag, (uint32_t)lambertians_.size());
            lambertians_.push_back(material);
        } else if constexpr ( std::is_same_v<M, materials::Metal> ) {
            mat_handle = pack_handle(tag, (uint32_t)metals_.size());
            metals_.push_back(material);
        } else if constexpr ( std::is_same_v<M, materials::Dielectric> ) {
            mat_handle = pack_handle(tag, (uint32_t)dielectrics_.size());
            dielectrics_.push_back(material);
        } else if constexpr ( std::is_same_v<M, materials::DiffuseLight> ) {
            mat_handle = pack_handle(tag, (uint32_t)diffuse_lights_.size());
            if ( register_light ) {
                light_handle_indices_.push_back((uint32_t)handles_.size());
            }
            diffuse_lights_.push_back(material);
        } else if constexpr ( std::is_same_v<M, materials::TexturedLambertian> ) {
            mat_handle = pack_handle(tag, (uint32_t)textured_lambertians_.size());
            textured_lambertians_.push_back(material);
        }
    });
    return mat_handle;
}

// ── SceneBuilder helper methods ────────────────────────────────────────

template <typename T>
T *SceneBuilder::upload_array(rt::Runtime *rt, const std::vector<T> &host_vec) {
    if ( host_vec.empty() ) {
        return nullptr;
    }
    T *ptr = rt->template alloc_device<T>((int)host_vec.size());
    rt->copy_to_device(ptr, host_vec.data(), host_vec.size() * sizeof(T));
    return ptr;
}

inline float SceneBuilder::compute_area(Handle h) const {
    auto type = static_cast<HittableType>(handle_tag(h.hittable));
    uint32_t idx = handle_index(h.hittable);
    constexpr float PI = 3.14159265f;
    switch ( type ) {
        case HittableType::Sphere: {
            float r = spheres_[idx].radius;
            return 4.f * PI * r * r;
        }
        case HittableType::Triangle: {
            float3 ab = sub(triangles_[idx].b, triangles_[idx].a);
            float3 ac = sub(triangles_[idx].c, triangles_[idx].a);
            return 0.5f * len(cross(ab, ac));
        }
        case HittableType::Quad: {
            return len(cross(quads_[idx].edge_u, quads_[idx].edge_v));
        }
        case HittableType::Box: {
            float3 ext = sub(boxes_[idx].box_max, boxes_[idx].box_min);
            return 2.f * (ext.x * ext.y + ext.y * ext.z + ext.x * ext.z);
        }
        case HittableType::Mesh: {
            // Sum of the mesh's triangle areas (surface area of the
            // whole mesh, for light-list importance sampling).
            const auto &m = meshes_[idx];
            float area = 0.f;
            for ( uint32_t i = 0; i < m.num_triangles; i++ ) {
                const auto &t = triangles_[m.first_triangle + i];
                area += 0.5f * len(cross(sub(t.b, t.a), sub(t.c, t.a)));
            }
            return area;
        }
        case HittableType::Portal:
            // A portal's surface area is undefined (rays teleport);
            // emissive portals are excluded from the light list for
            // this reason (see add_portal).
            return 0.f;
    }
    return 0.f;
}

// ── Helper: get debug colour for a handle's material ───────────────────

/// Extract the colour to use for debug rendering of a handle's material.
static float3 debug_material_color(const Handle &h,
                                   const std::vector<materials::Lambertian> &lamberts,
                                   const std::vector<materials::Metal> &metals,
                                   const std::vector<materials::Dielectric> &dielectrics,
                                   const std::vector<materials::DiffuseLight> &lights,
                                   const std::vector<materials::TexturedLambertian> &textured) {
    auto mtype = static_cast<MaterialType>(handle_tag(h.material));
    uint32_t midx = handle_index(h.material);
    switch ( mtype ) {
        case MaterialType::Lambertian:
            return lamberts[midx].albedo;
        case MaterialType::Metal:
            return metals[midx].albedo;
        case MaterialType::Dielectric:
            return float3{0.7f, 0.8f, 1.0f};
        case MaterialType::DiffuseLight: {
            // Clamp emission to [0, 1] range for display
            float3 e = lights[midx].emit_color;
            float max_c = math::fmax(math::fmax(e.x, e.y), e.z);
            if ( max_c > 1.f ) {
                e = scale(e, 1.f / max_c);
            }
            return e;
        }
        case MaterialType::TexturedLambertian:
            // Sample the texture at the chart centre for a representative
            // colour (deterministic seed; the sampler gets an RNG anyway).
            {
                RNG rng {0xdeadbeefu};
                return textures::sample(textured[midx].texture, 0.5f, 0.5f, 0.f, rng);
            }
    }
    return float3{1, 1, 1};
}

// ── SceneBuilder::add_portal ─────────────────────────────────────────

inline void SceneBuilder::add_portal(hittables::PortalShape entry,
                                     hittables::PortalShape exit) {
    // Dummy white Lambertian: its scatter() teleports portal records
    // (see rt::portal_scatter()), making the portal a pure window.
    add_portal(std::move(entry), std::move(exit),
               Material {materials::lambertian({1.f, 1.f, 1.f})});
}

inline void SceneBuilder::add_portal(hittables::PortalShape entry,
                                     hittables::PortalShape exit,
                                     Material material) {
    Handle handle;
    handle.hittable = pack_handle((uint32_t)HittableType::Portal,
                                  (uint32_t)portal_pairs_.size());
    // Emissive portals are excluded from the light list (their surface
    // area is undefined for importance sampling) — push_material handles
    // that with register_light = false.
    handle.material = push_material(material, /*register_light=*/false);
    portal_pairs_.push_back(
        hittables::portal(std::move(entry), std::move(exit)));
    handles_.push_back(handle);
    aabbs_.push_back(portal_pairs_.back().aabb());
}

// ── SceneBuilder::add_mesh ────────────────────────────────────────────

inline void SceneBuilder::add_mesh(const std::vector<hittables::Triangle> &triangles,
                                    const Material &material, bool smooth) {
    if ( triangles.empty() ) {
        return;
    }
    // The mesh's triangles go into the SAME per-type triangle array as
    // standalone Triangle hittables; the Mesh handle records the range.
    //
    // Zero-area (degenerate) triangles are dropped here: they can never
    // produce a hit, and carrying them is a NaN risk on GPU fast-math
    // builds (their normals are zeroed, see hittables::Triangle).
    uint32_t first = (uint32_t)triangles_.size();

    if (smooth) {
        // Compute per-vertex normals by averaging adjacent face normals.
        struct VKey {
            float x, y, z;
            bool operator==(const VKey &o) const {
                return x == o.x && y == o.y && z == o.z;
            }
        };
        struct VKeyHash {
            size_t operator()(const VKey &k) const {
                return (size_t)((unsigned)k.x * 73856093u +
                                (unsigned)k.y * 19349663u +
                                (unsigned)k.z * 83492791u);
            }
        };
        std::unordered_map<VKey, float3, VKeyHash> normal_sums;
        size_t nt = triangles.size();
        std::vector<size_t> valid_idx;
        for (size_t i = 0; i < nt; ++i) {
            float3 ab = sub(triangles[i].b, triangles[i].a);
            float3 ac = sub(triangles[i].c, triangles[i].a);
            if (len2(cross(ab, ac)) < 1e-12f) continue;
            float3 fn = rt::norm(triangles[i].normal);
            VKey ka{triangles[i].a.x, triangles[i].a.y, triangles[i].a.z};
            VKey kb{triangles[i].b.x, triangles[i].b.y, triangles[i].b.z};
            VKey kc{triangles[i].c.x, triangles[i].c.y, triangles[i].c.z};
            auto it = normal_sums.find(ka);
            if (it == normal_sums.end()) normal_sums[ka] = fn;
            else it->second = rt::add(it->second, fn);
            it = normal_sums.find(kb);
            if (it == normal_sums.end()) normal_sums[kb] = fn;
            else it->second = rt::add(it->second, fn);
            it = normal_sums.find(kc);
            if (it == normal_sums.end()) normal_sums[kc] = fn;
            else it->second = rt::add(it->second, fn);
            valid_idx.push_back(i);
        }
        for (size_t vi : valid_idx) {
            const auto &t = triangles[vi];
            float3 na = rt::norm(normal_sums[VKey{t.a.x, t.a.y, t.a.z}]);
            float3 nb = rt::norm(normal_sums[VKey{t.b.x, t.b.y, t.b.z}]);
            float3 nc = rt::norm(normal_sums[VKey{t.c.x, t.c.y, t.c.z}]);
            triangles_.push_back(hittables::Triangle(t.a, t.b, t.c, na, nb, nc));
        }
    } else {
        for ( const auto &t : triangles ) {
            float3 ab = sub(t.b, t.a), ac = sub(t.c, t.a);
            if ( len2(cross(ab, ac)) < 1e-12f ) {
                continue;
            }
            triangles_.push_back(t);
        }
    }

    uint32_t count = (uint32_t)triangles_.size() - first;
    if ( count == 0 ) {
        return;
    }
    add(hittables::Mesh {first, count}, material);
}

/// Copy a float3 into a float[3] array.
static void copy_float3(float dst[3], float3 src) {
    dst[0] = src.x;
    dst[1] = src.y;
    dst[2] = src.z;
}

// ── SceneBuilder::build_debug_geometry ─────────────────────────────────

inline void SceneBuilder::build_debug_geometry() {
    debug_spheres_.clear();
    debug_quads_.clear();
    debug_boxes_.clear();

    int num_handles = (int)handles_.size();
    for ( int i = 0; i < num_handles; i++ ) {
        const Handle &h = handles_[i];
        auto htype = static_cast<HittableType>(handle_tag(h.hittable));
        uint32_t hidx = handle_index(h.hittable);
        float3 color = debug_material_color(
            h, lambertians_, metals_, dielectrics_, diffuse_lights_, textured_lambertians_);

        switch ( htype ) {
            case HittableType::Sphere: {
                const auto &s = spheres_[hidx];
                DebugSphere ds;
                copy_float3(ds.center, s.center);
                ds.radius = s.radius;
                copy_float3(ds.color, color);
                debug_spheres_.push_back(ds);
                break;
            }
            case HittableType::Quad: {
                const auto &q = quads_[hidx];
                DebugQuad dq;
                copy_float3(dq.base, q.base);
                copy_float3(dq.edge_u, q.edge_u);
                copy_float3(dq.edge_v, q.edge_v);
                copy_float3(dq.color, color);
                debug_quads_.push_back(dq);
                break;
            }
            case HittableType::Triangle:
                // Triangles not used in current scenes, skip
                break;
            case HittableType::Mesh:
                // Meshes are drawn from the triangle array by the
                // scene-debug renderer (no legacy debug buffer for
                // them), skip
                break;
            case HittableType::Box: {
                const auto &b = boxes_[hidx];
                DebugBox db;
                copy_float3(db.box_min, b.box_min);
                copy_float3(db.box_max, b.box_max);
                copy_float3(db.color, color);
                debug_boxes_.push_back(db);
                break;
            }
            case HittableType::Portal: {
                // Show BOTH portal shapes with a fixed "portal" colour
                // (no material to sample; portals are bidirectional, so
                // both surfaces matter).  Triangle shapes are skipped
                // (no debug buffer for them).
                const auto &p = portal_pairs_[hidx];
                float3 portal_color = {0.2f, 0.8f, 0.9f};
                auto push_shape = [&](const auto &shape) {
                    using S = std::decay_t<decltype(shape)>;
                    if constexpr ( std::is_same_v<S, hittables::Quad> ) {
                        DebugQuad dq;
                        copy_float3(dq.base, shape.base);
                        copy_float3(dq.edge_u, shape.edge_u);
                        copy_float3(dq.edge_v, shape.edge_v);
                        copy_float3(dq.color, portal_color);
                        debug_quads_.push_back(dq);
                    } else if constexpr ( std::is_same_v<S, hittables::Sphere> ) {
                        DebugSphere ds;
                        copy_float3(ds.center, shape.center);
                        ds.radius = shape.radius;
                        copy_float3(ds.color, portal_color);
                        debug_spheres_.push_back(ds);
                    }
                };
                visit(p.entry, push_shape);
                visit(p.exit, push_shape);
                break;
            }
        }
    }
}

// ── SceneBuilder BVH construction ──────────────────────────────────────

/// Build a flat binary BVH over `leaf_values.size()` leaves, using the
/// per-leaf AABBs in `aabbs` (aabbs[i] and leaf_values[i] describe the
/// same leaf i).  Leaves partition by centroid along the longest axis
/// (median split); nodes are flattened to a pre-order array where a leaf
/// stores its object index (scene handle index or absolute triangle
/// index) in `right`.  Returns an empty vector for zero leaves.
///
/// This is the shared core of the scene BVH (SceneBuilder::build_bvh)
/// and the per-mesh BVHs (SceneBuilder::build_mesh_bvhs): identical
/// structure, different leaf payloads.
static std::vector<BvhNode> build_flat_bvh(const std::vector<Aabb> &aabbs,
                                           const std::vector<uint32_t> &leaf_values) {
    int num_objects = (int)leaf_values.size();
    if ( num_objects == 0 ) {
        return {};
    }

    // ── Compute the centroid of each leaf's AABB ──────────────────────
    std::vector<float3> centroids(num_objects);
    for ( int i = 0; i < num_objects; i++ ) {
        centroids[i] = {(aabbs[i].min.x + aabbs[i].max.x) * 0.5f,
                        (aabbs[i].min.y + aabbs[i].max.y) * 0.5f,
                        (aabbs[i].min.z + aabbs[i].max.z) * 0.5f};
    }

    // ── Working index list (sorted into subtrees via nth_element) ─────
    std::vector<uint32_t> indices(num_objects);
    for ( int i = 0; i < num_objects; i++ ) {
        indices[i] = (uint32_t)i;
    }

    // ── Recursively build a binary BVH and flatten to a pre-order array ─
    std::vector<BvhNode> nodes;
    nodes.reserve(num_objects * 2 - 1);

    // Recursive construction using an explicit stack to avoid deep
    // recursion on GPU
    struct StackFrame {
        int start;
        int end;
        uint32_t parent_index;
        bool is_right;
    };
    // Sentinel: the frame covering the whole range reuses the pre-pushed
    // node 0 as its node (it has no parent to link into).
    constexpr uint32_t NO_PARENT = 0xFFFFFFFFu;

    // Root placeholder — node 0 is filled in by the first frame below.
    nodes.push_back({Aabb {{0, 0, 0}, {0, 0, 0}}, BVH_LEAF, BVH_LEAF});
    uint32_t root_index = 0;

    std::vector<StackFrame> stack;
    stack.reserve(64);
    stack.push_back({0, num_objects, NO_PARENT, false});

    while ( !stack.empty() ) {
        StackFrame frame = stack.back();
        stack.pop_back();

        int start = frame.start;
        int end = frame.end;
        int count = end - start;

        // The root frame reuses node 0; every other frame allocates a
        // fresh node and links it into its parent's left/right slot.
        bool is_root = frame.parent_index == NO_PARENT;
        uint32_t node_index = is_root ? root_index : (uint32_t)nodes.size();

        if ( !is_root ) {
            if ( frame.is_right ) {
                nodes[frame.parent_index].right = node_index;
            } else {
                nodes[frame.parent_index].left = node_index;
            }
        }

        // ── Compute bounding box for this range ────────────────────────
        Aabb bounds = aabbs[indices[start]];
        for ( int i = start + 1; i < end; i++ ) {
            bounds = aabb_merge(bounds, aabbs[indices[i]]);
        }

        // ── Leaf node: single object ───────────────────────────────────
        if ( count == 1 ) {
            uint32_t leaf = leaf_values[indices[start]];
            if ( is_root ) {
                nodes[root_index] = {bounds, BVH_LEAF, leaf};
            } else {
                nodes.push_back({bounds, BVH_LEAF, leaf});
            }
            continue;
        }

        // ── Find the longest axis for the split ────────────────────────
        float3 extent = {bounds.max.x - bounds.min.x,
                         bounds.max.y - bounds.min.y,
                         bounds.max.z - bounds.min.z};
        int axis = 0;
        if ( extent.y > extent.x ) {
            axis = 1;
        }
        if ( extent.z > ((axis == 0) ? extent.x : extent.y) ) {
            axis = 2;
        }

        // ── Partition around the median along the longest axis ──────────
        int mid = start + count / 2;
        std::nth_element(indices.begin() + start,
                         indices.begin() + mid,
                         indices.begin() + end,
                         [&](uint32_t a, uint32_t b) {
                             float ca = (axis == 0)   ? centroids[a].x
                                        : (axis == 1) ? centroids[a].y
                                                      : centroids[a].z;
                             float cb = (axis == 0)   ? centroids[b].x
                                        : (axis == 1) ? centroids[b].y
                                                      : centroids[b].z;
                             return ca < cb;
                         });

        // ── Internal node: fill the placeholder, schedule children ──────
        if ( is_root ) {
            nodes[root_index] = {bounds, BVH_LEAF, BVH_LEAF};
        } else {
            nodes.push_back({bounds, BVH_LEAF, BVH_LEAF}); // placeholder
        }

        // Process left child first (pushed second so it's popped first)
        stack.push_back({start, mid, node_index, false});
        // Process right child second
        stack.push_back({mid, end, node_index, true});
    }

    return nodes;
}

inline void SceneBuilder::build_bvh() {
    if ( handles_.empty() ) {
        return;
    }

    int num_objects = (int)handles_.size();
    // The scene BVH's leaves are handle indices (identity).
    std::vector<uint32_t> leaf_values(num_objects);
    for ( int i = 0; i < num_objects; i++ ) {
        leaf_values[i] = (uint32_t)i;
    }
    bvh_nodes_ = build_flat_bvh(aabbs_, leaf_values);
    num_bvh_nodes_ = (int)bvh_nodes_.size();
    bvh_root_ = bvh_nodes_.empty() ? -1 : 0;
}

// ── SceneBuilder::build_mesh_bvhs ──────────────────────────────────────

inline void SceneBuilder::build_mesh_bvhs() {
    mesh_bvh_nodes_.clear();
    num_mesh_bvh_nodes_ = 0;
    if ( meshes_.empty() ) {
        return;
    }

    // Reserve a generous upper bound: one node per triangle (median-split
    // BVH over n triangles has ~2n nodes).
    size_t total_tris = 0;
    for ( const auto &m : meshes_ ) {
        total_tris += m.num_triangles;
    }
    mesh_bvh_nodes_.reserve(total_tris * 2);

    for ( auto &m : meshes_ ) {
        // One or zero triangles — a BVH adds nothing over the linear scan.
        if ( m.num_triangles < 2 ) {
            m.bvh_root = -1;
            continue;
        }
        // Build the mesh's local AABB list; leaves reference ABSOLUTE
        // triangle indices so traversal can index triangles_ directly.
        std::vector<Aabb> aabbs;
        aabbs.reserve(m.num_triangles);
        std::vector<uint32_t> leaf_values;
        leaf_values.reserve(m.num_triangles);
        for ( uint32_t i = 0; i < m.num_triangles; i++ ) {
            aabbs.push_back(triangles_[m.first_triangle + i].aabb());
            leaf_values.push_back(m.first_triangle + i);
        }
        auto nodes = build_flat_bvh(aabbs, leaf_values);
        if ( nodes.empty() ) {
            m.bvh_root = -1;
            continue;
        }
        // Rewrite internal child pointers from LOCAL (0-based within this
        // tree) to ABSOLUTE indices into the shared mesh_bvh_nodes_ array:
        // build_flat_bvh() returns a self-contained tree starting at node
        // 0, but every mesh's tree is concatenated into one flat array, so
        // a non-zero tree offset must be added to each child link.  Leaf
        // nodes are untouched (left == BVH_LEAF, right == absolute
        // triangle index — no offset applies).
        int base = (int)mesh_bvh_nodes_.size();
        for ( auto &n : nodes ) {
            if ( n.left != BVH_LEAF ) {
                n.left += (uint32_t)base;
                n.right += (uint32_t)base;
            }
        }
        m.bvh_root = base;
        mesh_bvh_nodes_.insert(mesh_bvh_nodes_.end(), nodes.begin(),
                               nodes.end());
    }
    num_mesh_bvh_nodes_ = (int)mesh_bvh_nodes_.size();
}

// ── SceneBuilder::build ────────────────────────────────────────────────

inline SceneData SceneBuilder::build(rt::Runtime *rt) {
    SceneData data = {};

    // Upload per-type hittable arrays
    auto upload = [&](const auto &vec, auto *&ptr, int &count) {
        ptr = upload_array(rt, vec);
        count = (int)vec.size();
    };

    upload(spheres_,        data.spheres,        data.num_spheres);
    upload(triangles_,      data.triangles,      data.num_triangles);
    upload(quads_,          data.quads,          data.num_quads);
    upload(boxes_,          data.boxes,          data.num_boxes);
    upload(portal_pairs_,   data.portals,        data.num_portals);
    upload(meshes_,         data.meshes,         data.num_meshes);
    upload(lambertians_,    data.lambertians,    data.num_lambertians);
    upload(metals_,         data.metals,         data.num_metals);
    upload(dielectrics_,    data.dielectrics,    data.num_dielectrics);
    upload(diffuse_lights_, data.diffuse_lights, data.num_diffuse_lights);
    upload(textured_lambertians_, data.textured_lambertians,
           data.num_textured_lambertians);
    upload(handles_,        data.handles,        data.num_handles);

    // AABBs
    if ( !aabbs_.empty() ) {
        data.aabbs = upload_array(rt, aabbs_);
    } else {
        data.aabbs = nullptr;
    }

    // BVH (upload if built via build_bvh())
    if ( !bvh_nodes_.empty() ) {
        data.bvh_nodes = upload_array(rt, bvh_nodes_);
        data.num_bvh_nodes = num_bvh_nodes_;
        data.bvh_root = bvh_root_;
    } else {
        data.bvh_nodes = nullptr;
        data.num_bvh_nodes = 0;
        data.bvh_root = -1;
    }

    // Per-mesh BVHs (upload if built via build_mesh_bvhs()); the meshes_
    // uploaded above carry each mesh's bvh_root into the array.
    if ( !mesh_bvh_nodes_.empty() ) {
        data.mesh_bvh_nodes = upload_array(rt, mesh_bvh_nodes_);
        data.num_mesh_bvh_nodes = num_mesh_bvh_nodes_;
    } else {
        data.mesh_bvh_nodes = nullptr;
        data.num_mesh_bvh_nodes = 0;
    }

    // Build light list from DiffuseLight materials
    if ( !light_handle_indices_.empty() ) {
        std::vector<LightInfo> light_infos;
        light_infos.reserve(light_handle_indices_.size());
        for ( uint32_t handle_idx : light_handle_indices_ ) {
            LightInfo info;
            info.handle = handles_[handle_idx];
            info.area = compute_area(handles_[handle_idx]);
            uint32_t material_idx = handle_index(info.handle.material);
            info.emission = diffuse_lights_[material_idx].emit_color;
            light_infos.push_back(info);
        }
        data.lights = upload_array(rt, light_infos);
        data.num_lights = (int)light_infos.size();
    } else {
        data.lights = nullptr;
        data.num_lights = 0;
    }

    return data;
}

#endif // KERNEL_BUILD

} // namespace rt
