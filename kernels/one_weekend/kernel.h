#pragma once
#include <sycl/sycl.hpp>

// ── device math types ──────────────────────────────────────────────────
using float3 = sycl::float3;

inline SYCL_EXTERNAL float3 vadd(float3 a, float3 b) {
    return {a.x() + b.x(), a.y() + b.y(), a.z() + b.z()};
}
inline SYCL_EXTERNAL float3 vsub(float3 a, float3 b) {
    return {a.x() - b.x(), a.y() - b.y(), a.z() - b.z()};
}
inline SYCL_EXTERNAL float3 vmul(float3 a, float3 b) {
    return {a.x() * b.x(), a.y() * b.y(), a.z() * b.z()};
}
inline SYCL_EXTERNAL float3 vscale(float3 a, float t) {
    return {a.x() * t, a.y() * t, a.z() * t};
}
inline SYCL_EXTERNAL float  vdot(float3 a, float3 b) {
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}
inline SYCL_EXTERNAL float3 vcross(float3 a, float3 b) {
    return {a.y()*b.z() - a.z()*b.y(),
            a.z()*b.x() - a.x()*b.z(),
            a.x()*b.y() - a.y()*b.x()};
}
inline SYCL_EXTERNAL float  vlen2(float3 a) { return vdot(a, a); }
inline SYCL_EXTERNAL float  vlen(float3 a)  { return sycl::sqrt(vlen2(a)); }
inline SYCL_EXTERNAL float3 vnorm(float3 a) { return vscale(a, 1.f / vlen(a)); }
inline SYCL_EXTERNAL float3 vlerp(float3 a, float3 b, float t) {
    return vadd(vscale(a, 1.f - t), vscale(b, t));
}

// ── RNG (xorshift32 — simple, good enough) ────────────────────────────
struct RNG {
    uint32_t state;
    SYCL_EXTERNAL float next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (float)state / 4294967296.f;
    }
};

// ── ray ────────────────────────────────────────────────────────────────
struct Ray {
    float3 orig;
    float3 dir;
};

// ── material types ─────────────────────────────────────────────────────
enum class MatType : uint8_t { LAMBERTIAN, METAL, DIELECTRIC, DIFFUSE_LIGHT };

// ── sphere (the only primitive for this kernel) ──────────────────────
struct Sphere {
    float3  center;
    float   radius;
    MatType mat_type;
    float3  albedo;
    float   fuzz;       // metal only
    float   ir;         // dielectric only (index of refraction)
    float3  emit;       // light only
};

// ── hit record ─────────────────────────────────────────────────────────
struct HitRecord {
    float3  p;
    float3  normal;
    float   t;
    float3  albedo;
    float3  emit;
    MatType mat_type;
    float   fuzz;
    float   ir;
    bool    front_face;
};

// ── device scene (flat arrays, uploaded at init) ─────────────────────
struct DeviceScene {
    const Sphere* spheres;
    int           num_spheres;
};

// ── camera ─────────────────────────────────────────────────────────────
struct CameraData {
    float3 origin;
    float3 lower_left;
    float3 horizontal;
    float3 vertical;
    float3 u, v, w;
    float  lens_radius;
};
