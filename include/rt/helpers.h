#pragma once
#include <rt/math.h>

/// Shared math helpers for raytracing.
namespace rt {

/// Returns a random point uniformly distributed inside the unit sphere
/// using rejection sampling (expected ~52% acceptance rate).
inline float3 random_in_unit_sphere(RNG &rng) {
    for ( int i = 0; i < 100; i++ ) {
        float3 candidate = {2 * rng.next() - 1, 2 * rng.next() - 1, 2 * rng.next() - 1};
        if ( len2(candidate) < 1 ) {
            return candidate;
        }
    }
    return {0, 0, 1};
}

/// Reflect vector `incident` about the surface normal:
///   reflected = incident − 2·(incident·normal)·normal
inline float3 reflect(float3 incident, float3 normal) {
    return sub(incident, scale(normal, 2 * dot(incident, normal)));
}

/// Refract vector `incident` across a dielectric interface using Snell's law.
///
/// Formula:  η·sin(θ₁) = sin(θ₂)
///
/// @returns false if total internal reflection occurs (discriminant ≤ 0).
/// @param[out] out_direction  The refracted direction (valid only when true).
inline bool refract(float3 incident, float3 surface_normal, float eta, float3 &out_direction) {
    // Compute the unit incident direction and the cosine of the incident angle
    float3 unit_incident = norm(incident);
    float cos_theta_i = dot(unit_incident, surface_normal);

    // Compute the discriminant from Snell's law: 1 − η²·(1 − cos²(θ₁))
    float discriminant = 1 - eta * eta * (1 - cos_theta_i * cos_theta_i);
    if ( discriminant <= 0 ) {
        return false; // total internal reflection
    }

    // Compute the refracted direction:  η·(v − n·(v·n)) − n·√(discriminant)
    out_direction = sub(scale(sub(unit_incident, scale(surface_normal, cos_theta_i)), eta),
                        scale(surface_normal, sycl::sqrt(discriminant)));
    return true;
}

/// Schlick's Fresnel reflectance approximation for dielectrics.
inline float schlick(float cosine, float refractive_index) {
    float r0 = (1 - refractive_index) / (1 + refractive_index);
    r0 *= r0;
    return r0 + (1 - r0) * sycl::pow(1 - cosine, 5);
}

} // namespace rt
