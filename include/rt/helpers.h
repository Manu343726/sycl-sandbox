#pragma once
#include "math.h"

/// Shared math helpers for raytracing (used by individual hittables/materials).
namespace rt {

/// Returns a random point uniformly distributed inside the unit sphere.
inline float3 random_in_unit_sphere(RNG& r) {
    for (int i = 0; i < 100; i++) {
        float3 p = {2*r.next()-1, 2*r.next()-1, 2*r.next()-1};
        if (len2(p) < 1) return p;
    }
    return {0,0,1};
}

/// Reflects vector `v` about the surface normal `n`.
inline float3 reflect(float3 v, float3 n) { return sub(v, scale(n, 2*dot(v,n))); }

/// Refracts vector `v` across the interface with normal `n` and
/// relative index of refraction `eta`.  Returns true if refraction occurs.
inline bool refract(float3 v, float3 n, float eta, float3& out) {
    float3 uv = norm(v);
    float dt = dot(uv, n);
    float d = 1 - eta*eta*(1 - dt*dt);
    if (d <= 0) return false;
    out = sub(scale(sub(uv, scale(n, dt)), eta), scale(n, sycl::sqrt(d)));
    return true;
}

/// Schlick's approximation for the Fresnel reflectance of a dielectric.
inline float schlick(float c, float ir) {
    float r0 = (1-ir)/(1+ir); r0*=r0;
    return r0 + (1-r0)*sycl::pow(1-c, 5);
}

} // namespace rt
