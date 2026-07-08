#pragma once
#include <sycl/sycl.hpp>

// ── device math types ──────────────────────────────────────────────────
struct f3 { float x, y, z; };
inline f3 make_f3(float x, float y, float z) { return {x, y, z}; }

inline f3 vadd(f3 a, f3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline f3 vsub(f3 a, f3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline f3 vmul(f3 a, f3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}
inline f3 vscale(f3 a, float t) {
    return {a.x * t, a.y * t, a.z * t};
}
inline float  vdot(f3 a, f3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline f3 vcross(f3 a, f3 b) {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}
inline float  vlen2(f3 a) { return vdot(a, a); }
inline float  vlen(f3 a)  { return sycl::sqrt(vlen2(a)); }
inline f3 vnorm(f3 a) { return vscale(a, 1.f / vlen(a)); }
inline f3 vlerp(f3 a, f3 b, float t) {
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
    f3 orig;
    f3 dir;
};

// ── material types ─────────────────────────────────────────────────────
enum class MatType : uint8_t { LAMBERTIAN, METAL, DIELECTRIC, DIFFUSE_LIGHT };

// ── sphere (the only primitive for this kernel) ──────────────────────
struct Sphere {
    f3  center;
    float   radius;
    MatType mat_type;
    f3  albedo;
    float   fuzz;       // metal only
    float   ir;         // dielectric only (index of refraction)
    f3  emit;       // light only
};

// ── hit record ─────────────────────────────────────────────────────────
struct HitRecord {
    f3  p;
    f3  normal;
    float   t;
    f3  albedo;
    f3  emit;
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
    f3 origin;
    f3 lower_left;
    f3 horizontal;
    f3 vertical;
    f3 u, v, w;
    float  lens_radius;
};
