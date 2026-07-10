#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

/// 3D vector math and RNG for SYCL device code.
namespace rt {

/// 3-component floating-point vector.
struct float3 { float x, y, z; };

inline float3 add(float3 a, float3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline float3 sub(float3 a, float3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline float3 mul(float3 a, float3 b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; }
inline float3 scale(float3 a, float t)   { return {a.x*t, a.y*t, a.z*t}; }
inline float  dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float  len2(float3 a) { return dot(a, a); }
inline float  len(float3 a)  { return sycl::sqrt(len2(a)); }
inline float3 norm(float3 a) { float l = len(a); return {a.x/l, a.y/l, a.z/l}; }
inline float3 cross(float3 a, float3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline float3 lerp(float3 a, float3 b, float t) {
    return add(scale(a, 1-t), scale(b, t));
}

/// Xorshift32 pseudo-random number generator.
struct RNG {
    uint32_t state;
    float next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (float)state / 4294967296.f;
    }
};

} // namespace rt
