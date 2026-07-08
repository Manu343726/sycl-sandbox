#pragma once
#include <sycl/sycl.hpp>

// ── device math types ──────────────────────────────────────────────────
// Use sycl::float3 (vec<float,3>) directly — portable across all backends.

inline sycl::float3 vadd(sycl::float3 a, sycl::float3 b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
inline sycl::float3 vsub(sycl::float3 a, sycl::float3 b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
inline sycl::float3 vmul(sycl::float3 a, sycl::float3 b) {
    return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
}
inline sycl::float3 vscale(sycl::float3 a, float t) {
    return {a[0] * t, a[1] * t, a[2] * t};
}
inline float  vdot(sycl::float3 a, sycl::float3 b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline sycl::float3 vcross(sycl::float3 a, sycl::float3 b) {
    return { a[1]*b[2] - a[2]*b[1],
             a[2]*b[0] - a[0]*b[2],
             a[0]*b[1] - a[1]*b[0] };
}
inline float  vlen2(sycl::float3 a) { return vdot(a, a); }
inline float  vlen(sycl::float3 a)  { return sycl::sqrt(vlen2(a)); }
inline sycl::float3 vnorm(sycl::float3 a) { return vscale(a, 1.f / vlen(a)); }
inline sycl::float3 vlerp(sycl::float3 a, sycl::float3 b, float t) {
    return vadd(vscale(a, 1.f - t), vscale(b, t));
}

// ── RNG (xorshift32 — simple, good enough) ────────────────────────────
struct RNG {
    uint32_t state;
    float next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (float)state / 4294967296.f;
    }
};

// ── ray ────────────────────────────────────────────────────────────────
struct Ray {
    sycl::float3 orig;
    sycl::float3 dir;
};

// ── material types ─────────────────────────────────────────────────────
enum class MatType : uint8_t { LAMBERTIAN, METAL, DIELECTRIC, DIFFUSE_LIGHT };

// ── sphere (the only primitive for this kernel) ──────────────────────
struct Sphere {
    sycl::float3  center;
    float         radius;
    MatType       mat_type;
    sycl::float3  albedo;
    float         fuzz;
    float         ir;
    sycl::float3  emit;
};

// ── hit record ─────────────────────────────────────────────────────────
struct HitRecord {
    sycl::float3  p;
    sycl::float3  normal;
    float         t;
    sycl::float3  albedo;
    sycl::float3  emit;
    MatType       mat_type;
    float         fuzz;
    float         ir;
    bool          front_face;
};

// ── device scene (flat arrays, uploaded at init) ─────────────────────
struct DeviceScene {
    const Sphere* spheres;
    int           num_spheres;
};

// ── camera ─────────────────────────────────────────────────────────────
struct CameraData {
    sycl::float3 origin;
    sycl::float3 lower_left;
    sycl::float3 horizontal;
    sycl::float3 vertical;
    sycl::float3 u, v, w;
    float        lens_radius;
};
