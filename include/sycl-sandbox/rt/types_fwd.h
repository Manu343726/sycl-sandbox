#pragma once
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/optional.h>
#include <cstdint>

/// Forward declarations used by individual hittable/material headers.
namespace rt {

/// Identifies a hittable type packed into the upper bits of Handle::hittable.
/// Defined here (not in scene/data.h) so deep pipeline code — primitive
/// hit() bodies, materials, textures — can name their kind when reporting
/// debug metadata through the trace collector.
enum class HittableType : uint32_t {
    Sphere = 0,
    Triangle = 1,
    Quad = 2,
    Box = 3,
    Portal = 4,   ///< Portal: teleports the ray (no material)
    Mesh = 5,     ///< Triangle mesh: window into the scene's triangle array
};

/// Identifies a material type packed into the upper bits of Handle::material.
enum class MaterialType : uint32_t {
    Lambertian = 0,
    Metal = 1,
    Dielectric = 2,
    DiffuseLight = 3,
    TexturedLambertian = 4,
};

/// A ray in 3D space: origin point + direction vector.
struct Ray {
    float3 orig, dir;
    /// Animation time of the ray (from the scene's `time` param).  Used
    /// by time-varying materials/textures (e.g. animated procedural
    /// textures).  Defaults to 0 for rays built without a time.
    float time = 0.f;
};

/// Record filled by Hittable::hit() when a ray-geometry intersection is found.
struct HitRecord {
    float3 p;        ///< World-space intersection point.
    float3 normal;   ///< Surface normal at intersection (pointing outward).
    float t;         ///< Distance along the ray where the hit occurred.
    bool front_face; ///< True if the ray hit from outside the surface.
    /// Parametric surface coordinates of the hit point, in [0,1] within
    /// the primitive's own UV space (quad/triangle: affine/barycentric
    /// coordinates, sphere: spherical mapping, box: the hit face's quad
    /// coordinates).  Input to texture samplers; textures decide their
    /// own behavior for out-of-range values.
    float u = 0.f;
    float v = 0.f;
    /// Portal teleport marker.  When true, p/normal are the OTHER surface
    /// point and normal: whichever portal shape the ray hit (entry or
    /// exit — portals are bidirectional), its parametric coordinates were
    /// mapped onto the counterpart hittable.  Materials scatter portal
    /// records as a pure teleport: the ray continues from `portal_origin`
    /// along `portal_dir` with unit attenuation (see portal_scatter()).
    /// `t` is the HIT shape's distance (entry or exit), so closest-hit
    /// ordering sees the portal at the surface the ray actually crossed.
    bool is_portal = false;
    float3 portal_origin = {0, 0, 0}; ///< where the continuation ray starts
    float3 portal_dir = {0, 0, 0};    ///< direction of the continuation ray
};

/// Record returned by Material::scatter() when scattering occurs.
struct ScatterRecord {
    float3 attenuation; ///< Colour attenuation (albedo × path throughput).
    Ray scattered;      ///< New ray direction after scattering.
};

/// Build the ScatterRecord for a portal hit: the ray continues from the
/// exit surface (`portal_origin`/`portal_dir`) with unit attenuation —
/// the path passes through the portal unchanged.  Called by every
/// material's scatter() on is_portal records, so the trace loop stays
/// generic.  Emission is handled before scatter, so a portal instanced
/// with an emissive material (e.g. DiffuseLight) glows instead of
/// teleporting.
inline optional<ScatterRecord> portal_scatter(const Ray &incoming_ray,
                                              const HitRecord &rec) {
    return ScatterRecord {{1, 1, 1},
                          Ray {rec.portal_origin, rec.portal_dir, incoming_ray.time}};
}

} // namespace rt
