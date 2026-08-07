#pragma once
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/rt/types.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/variant.h>
#ifndef KERNEL_NATIVE
#include <sycl/sycl.hpp>
#endif
#include <sycl-sandbox/profiler.h>
#include <algorithm>
#include <vector>

namespace rt {

/// Data-oriented scene representation: per-type arrays, handle-based dispatch,
/// optional BVH acceleration and light list for importance sampling.
///
/// Kernel init code builds the scene via SceneBuilder (host only).
/// The render loop reads SceneView (device only, trivially copyable).

// ── Type tags ──────────────────────────────────────────────────────────

/// Identifies a hittable type packed into the upper bits of Handle::hittable.
enum class HittableType : uint32_t {
    Sphere = 0,
    Triangle = 1,
    Quad = 2,
    Box = 3,
    Portal = 4,   ///< Portal: teleports the ray (no material)
};

/// Identifies a material type packed into the upper bits of Handle::material.
enum class MaterialType : uint32_t {
    Lambertian = 0,
    Metal = 1,
    Dielectric = 2,
    DiffuseLight = 3,
    TexturedLambertian = 4,
};

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

/// Sentinel value indicating a BVH leaf node (left child = invalid).
static constexpr uint32_t BVH_LEAF = 0xFFFFFFFF;

// ── BVH node (future) ─────────────────────────────────────────────────

/// Flat BVH node stored in a contiguous array.
/// Leaf nodes: left == BVH_LEAF, right == handle index.
/// Internal nodes: left and right are indices into the bvh_nodes array.
struct BvhNode {
    Aabb bounds;    ///< Bounding box enclosing both children.
    uint32_t left;  ///< Left child index (BVH_LEAF for leaf).
    uint32_t right; ///< Right child index or handle index (leaf).
};

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
inline optional<HitRecord>
handle_hit(Handle h, const Ray &ray, float t_min, float t_max, const SceneView &scene) {
    auto type = static_cast<HittableType>(handle_tag(h.hittable));
    uint32_t idx = handle_index(h.hittable);
    switch ( type ) {
        case HittableType::Sphere:
            return scene.spheres[idx].hit(ray, t_min, t_max);
        case HittableType::Triangle:
            return scene.triangles[idx].hit(ray, t_min, t_max);
        case HittableType::Quad:
            return scene.quads[idx].hit(ray, t_min, t_max);
        case HittableType::Box:
            return scene.boxes[idx].hit(ray, t_min, t_max);
        case HittableType::Portal:
            return scene.portals[idx].hit(ray, t_min, t_max);
    }
    return nullopt;
}

/// Scatter a ray off a handle's material.
inline optional<ScatterRecord> handle_scatter(Handle h,
                                              const Ray &incoming_ray,
                                              const HitRecord &hit,
                                              RNG &rng,
                                              const SceneView &scene) {
    auto type = static_cast<MaterialType>(handle_tag(h.material));
    uint32_t idx = handle_index(h.material);
    switch ( type ) {
        case MaterialType::Lambertian:
            return scene.lambertians[idx].scatter(incoming_ray, hit, rng);
        case MaterialType::Metal:
            return scene.metals[idx].scatter(incoming_ray, hit, rng);
        case MaterialType::Dielectric:
            return scene.dielectrics[idx].scatter(incoming_ray, hit, rng);
        case MaterialType::DiffuseLight:
            return scene.diffuse_lights[idx].scatter(incoming_ray, hit, rng);
        case MaterialType::TexturedLambertian:
            return scene.textured_lambertians[idx].scatter(incoming_ray, hit, rng);
    }
    return nullopt;
}

/// Emit light from a handle's material.
inline float3 handle_emit(Handle h, const HitRecord &hit, const SceneView &scene) {
    auto type = static_cast<MaterialType>(handle_tag(h.material));
    uint32_t idx = handle_index(h.material);
    switch ( type ) {
        case MaterialType::Lambertian:
            return scene.lambertians[idx].emit(hit);
        case MaterialType::Metal:
            return scene.metals[idx].emit(hit);
        case MaterialType::Dielectric:
            return scene.dielectrics[idx].emit(hit);
        case MaterialType::TexturedLambertian:
            return scene.textured_lambertians[idx].emit(hit);
        case MaterialType::DiffuseLight:
            return scene.diffuse_lights[idx].emit(hit);
    }
    return {0, 0, 0};
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
inline optional<BvhHitResult>
bvh_hit(const Ray &ray, float t_min, float t_max, const SceneView &scene,
        bool skip_backfaces = false) {
    if ( !scene.bvh_nodes || scene.bvh_root < 0 ) {
        return nullopt;
    }

    optional<BvhHitResult> closest_hit;
    uint32_t stack[32];
    int stack_size = 0;
    stack[stack_size++] = (uint32_t)scene.bvh_root;

    while ( stack_size > 0 ) {
        uint32_t node_index = stack[--stack_size];
        const BvhNode &node = scene.bvh_nodes[node_index];

        // Test ray against this node's bounding box
        if ( !aabb_hit(node.bounds, ray, t_min, t_max) ) {
            continue;
        }

        // Leaf node: test the single object
        if ( node.left == BVH_LEAF ) {
            auto hit = handle_hit(scene.handles[node.right], ray, t_min, t_max, scene);
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

    /// Add a portal (entry <-> exit) to the scene.
    /// Portals are BIDIRECTIONAL: a ray hitting either shape teleports
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

    /// Optional: build a BVH over the scene for acceleration.
    /// Must be called before build().
    void build_bvh();

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
    de(lambertians);         num_lambertians = 0;
    de(metals);              num_metals = 0;
    de(dielectrics);         num_dielectrics = 0;
    de(diffuse_lights);      num_diffuse_lights = 0;
    de(textured_lambertians); num_textured_lambertians = 0;
    de(aabbs);
    de(bvh_nodes);           num_bvh_nodes = 0; bvh_root = -1;
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
        }
        box = hittable.aabb();
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

// ── SceneBuilder::build_bvh ────────────────────────────────────────────

inline void SceneBuilder::build_bvh() {
    if ( handles_.empty() ) {
        return;
    }

    int num_objects = (int)handles_.size();

    // ── Compute the centroid of each handle's AABB ─────────────────────
    std::vector<float3> centroids(num_objects);
    for ( int i = 0; i < num_objects; i++ ) {
        centroids[i] = {(aabbs_[i].min.x + aabbs_[i].max.x) * 0.5f,
                        (aabbs_[i].min.y + aabbs_[i].max.y) * 0.5f,
                        (aabbs_[i].min.z + aabbs_[i].max.z) * 0.5f};
    }

    // ── Working index list (sorted into subtrees via nth_element) ─────
    std::vector<uint32_t> indices(num_objects);
    for ( int i = 0; i < num_objects; i++ ) {
        indices[i] = (uint32_t)i;
    }

    // ── Recursively build a binary BVH and flatten to a pre-order array ─
    std::vector<BvhNode> nodes;
    nodes.reserve(num_objects * 2 - 1);

    // Recursive lambda using an explicit stack to avoid deep recursion on GPU
    struct StackFrame {
        int start;
        int end;
        uint32_t parent_index;
        bool is_right;
    };
    // Sentinel: the frame covering the whole scene reuses the pre-pushed
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
        Aabb bounds = aabbs_[indices[start]];
        for ( int i = start + 1; i < end; i++ ) {
            bounds = aabb_merge(bounds, aabbs_[indices[i]]);
        }

        // ── Leaf node: single object ───────────────────────────────────
        if ( count == 1 ) {
            if ( is_root ) {
                nodes[root_index] = {bounds, BVH_LEAF, indices[start]};
            } else {
                nodes.push_back({bounds, BVH_LEAF, indices[start]});
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

    // ── Upload to device memory ─────────────────────────────────────────
    bvh_nodes_ = std::move(nodes);
    num_bvh_nodes_ = (int)bvh_nodes_.size();
    bvh_root_ = 0;
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

} // namespace rt
