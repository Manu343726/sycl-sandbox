#pragma once
#include "math.h"

/// Shared math helpers for raytracing (used by individual hittables/materials).
namespace rt {

/// Returns a random point uniformly distributed inside the unit sphere.
///
/// Uses rejection sampling: generate points in the bounding cube (-1..1)³
/// and keep those with length² < 1.  Expected ~52% acceptance rate.
inline float3 random_in_unit_sphere(RNG& r) {
    for (int i = 0; i < 100; i++) {
        float3 p = {2*r.next()-1, 2*r.next()-1, 2*r.next()-1};
        if (len2(p) < 1) return p;
    }
    return {0,0,1};
}

/// Reflect vector `v` about the surface normal `n`.
///
/// Uses the formula:  reflected = v - 2*(v·n)*n
/// This is the standard reflection from a perfectly smooth surface.
inline float3 reflect(float3 v, float3 n) { return sub(v, scale(n, 2*dot(v,n))); }

/// Refract vector `v` across the interface with normal `n` and relative
/// index of refraction `eta` (η = n_outgoing / n_incident).
///
/// Uses Snell's law:  η·sin(θ₁) = sin(θ₂)
/// Returns false if total internal reflection occurs (cos(θ₂)² < 0).
/// The output `out` is only valid when true is returned.
inline bool refract(float3 v, float3 n, float eta, float3& out) {
    float3 uv = norm(v);
    float dt = dot(uv, n);
    float discriminant = 1 - eta*eta*(1 - dt*dt);
    if (discriminant <= 0) return false;  // total internal reflection
    out = sub(scale(sub(uv, scale(n, dt)), eta), scale(n, sycl::sqrt(discriminant)));
    return true;
}

/// Schlick's approximation for Fresnel reflectance of a dielectric.
///
/// Approximates the fraction of light reflected at a dielectric interface
/// as a function of the cosine of the incident angle and the refractive
/// index ratio.  Much cheaper than computing the full Fresnel equations.
///
/// @param c   Cosine of the angle between the incident direction and the normal.
/// @param ir  Relative index of refraction (η₂ / η₁).
inline float schlick(float c, float ir) {
    float r0 = (1-ir)/(1+ir);
    r0 *= r0;
    return r0 + (1-r0)*sycl::pow(1-c, 5);
}

} // namespace rt
