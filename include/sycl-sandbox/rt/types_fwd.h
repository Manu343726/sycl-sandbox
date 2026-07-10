#pragma once
#include <sycl-sandbox/rt/math.h>
#include <optional>

/// Forward declarations used by individual hittable/material headers.
namespace rt {

/// A ray in 3D space: origin point + direction vector.
struct Ray {
    float3 orig, dir;
};

/// Record filled by Hittable::hit() when a ray-geometry intersection is found.
struct HitRecord {
    float3 p;        ///< World-space intersection point.
    float3 normal;   ///< Surface normal at intersection (pointing outward).
    float t;         ///< Distance along the ray where the hit occurred.
    bool front_face; ///< True if the ray hit from outside the surface.
};

/// Record returned by Material::scatter() when scattering occurs.
struct ScatterRecord {
    float3 attenuation; ///< Colour attenuation (albedo × path throughput).
    Ray scattered;      ///< New ray direction after scattering.
};

} // namespace rt
