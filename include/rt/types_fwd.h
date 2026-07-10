#pragma once
#include "math.h"

/// Forward declarations used by individual hittable/material headers.
namespace rt {

/// A ray in 3D space: origin point + direction vector.
struct Ray { float3 orig, dir; };

/// Record filled by Hittable::hit() when a ray-geometry intersection is found.
struct HitRecord {
    float3 p;          ///< World-space intersection point.
    float3 normal;     ///< Surface normal at intersection (pointing outward).
    float  t;          ///< Distance along the ray where the hit occurred.
    bool   front_face; ///< True if the ray hit from outside the surface.
};

} // namespace rt
