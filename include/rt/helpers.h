#pragma once
#include "math.h"

/// Shared math helpers for raytracing (used by individual hittables/materials).
namespace rt {

/// Returns a random point uniformly distributed inside the unit sphere.
///
/// Uses rejection sampling: generate points in the bounding cube (-1..1)³
/// and keep those with length² < 1.  Expected ~52% acceptance rate.
inline float3 random_in_unit_sphere(RNG& rng) {
    for (int i = 0; i < 100; i++) {
        float3 candidate = {2*rng.next()-1, 2*rng.next()-1, 2*rng.next()-1};
        if (len2(candidate) < 1) return candidate;
    }
    return {0,0,1};
}

/// Reflect vector `incident` about the surface normal `normal`.
///
/// Formula:  reflected = incident − 2·(incident·normal)·normal
/// This is the standard reflection from a perfectly smooth surface.
inline float3 reflect(float3 incident, float3 normal) {
    return sub(incident, scale(normal, 2*dot(incident, normal)));
}

/// Refract vector `incident` across the interface with normal `surface_normal`
/// and relative index of refraction `eta` (η = n_outgoing / n_incident).
///
/// Uses Snell's law:  η·sin(θ₁) = sin(θ₂)
/// Returns false if total internal reflection occurs (cos(θ₂)² < 0).
/// The output `out_direction` is only valid when true is returned.
inline bool refract(float3 incident, float3 surface_normal, float eta,
                     float3& out_direction) {
    float3 unit_incident = norm(incident);
    float cos_theta_i = dot(unit_incident, surface_normal);
    float discriminant = 1 - eta*eta*(1 - cos_theta_i*cos_theta_i);
    if (discriminant <= 0) return false;  // total internal reflection
    out_direction = sub(scale(sub(unit_incident, scale(surface_normal, cos_theta_i)), eta),
                        scale(surface_normal, sycl::sqrt(discriminant)));
    return true;
}

/// Schlick's approximation for Fresnel reflectance of a dielectric.
///
/// Approximates the fraction of light reflected at a dielectric interface
/// as a function of the cosine of the incident angle and the refractive
/// index ratio.  Much cheaper than computing the full Fresnel equations.
///
/// @param cosine    Cosine of the angle between the incident direction and
///                  the surface normal.
/// @param refractive_index  Relative index of refraction (η₂ / η₁).
inline float schlick(float cosine, float refractive_index) {
    float r0 = (1-refractive_index)/(1+refractive_index);
    r0 *= r0;
    return r0 + (1-r0)*sycl::pow(1-cosine, 5);
}

} // namespace rt
