#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

/// 3D vector math and RNG for SYCL device code.
namespace rt {

/// 3-component floating-point vector (exactly 12 bytes, no padding).
struct float3 { float x, y, z; };

/// Component-wise addition:  result = a + b.
inline float3 add(float3 a, float3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }

/// Component-wise subtraction:  result = a - b.
inline float3 sub(float3 a, float3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }

/// Component-wise multiplication (Hadamard product):  result = a * b.
inline float3 mul(float3 a, float3 b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; }

/// Scalar multiplication:  result = a * t  (each component scaled by t).
inline float3 scale(float3 a, float t) { return {a.x*t, a.y*t, a.z*t}; }

/// Dot (scalar) product:  result = a·b = a.x*b.x + a.y*b.y + a.z*b.z.
inline float dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

/// Squared Euclidean length:  ‖a‖² = a·a.
inline float len2(float3 a) { return dot(a, a); }

/// Euclidean length:  ‖a‖ = √(a·a).
inline float len(float3 a) { return sycl::sqrt(len2(a)); }

/// Unit vector in the same direction as a:  â = a / ‖a‖.
/// If a is the zero vector the result is undefined (division by zero).
inline float3 norm(float3 a) { float l = len(a); return {a.x/l, a.y/l, a.z/l}; }

/// Cross product:  a × b  (perpendicular to both a and b).
/// The magnitude equals the area of the parallelogram spanned by a and b.
inline float3 cross(float3 a, float3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

/// Linear interpolation:  result = (1-t)·a + t·b  for t ∈ [0,1].
inline float3 lerp(float3 a, float3 b, float t) {
    return add(scale(a, 1-t), scale(b, t));
}

/// Xorshift32 pseudo-random number generator.
///
/// Produces uniformly distributed floats in (0, 1] from a 32-bit integer
/// state.  The state must not be zero (a zero state locks the generator).
struct RNG {
    uint32_t state;  ///< Current RNG state (must be non-zero).

    /// Advance the generator and return the next value in (0, 1].
    float next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (float)state / 4294967296.f;
    }
};

} // namespace rt
