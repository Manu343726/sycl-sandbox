#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>

/// Cross-backend math utilities for SYCL and native C++ compilation.
///
/// These functions dispatch to SYCL math builtins when compiled with
/// a SYCL compiler (`#ifdef __SYCL_DEVICE_ONLY__` or similar), and
/// fall back to `std::` functions otherwise.
///
/// SYCL backends often provide hardware-accelerated math for GPU
/// (e.g. `sycl::sqrt` maps to `__nv_sqrt` on CUDA), so using the
/// native SYCL function is preferred over `std::sqrt` in SYCL mode.
///
/// When KERNEL_NATIVE is defined (CPU software backend), all functions
/// use the standard C++ <cmath> implementations.

namespace math {

// ── sqrt ──────────────────────────────────────────────────────────────

/// Compute the square root of x.
inline float sqrt(float x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::sqrt(x);
#else
    return std::sqrt(x);
#endif
}

inline double sqrt(double x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::sqrt(x);
#else
    return std::sqrt(x);
#endif
}

// ── fabs ─────────────────────────────────────────────────────────────

/// Compute the absolute value of a floating-point number.
inline float fabs(float x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::fabs(x);
#else
    return std::fabs(x);
#endif
}

inline double fabs(double x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::fabs(x);
#else
    return std::fabs(x);
#endif
}

// ── fmin ─────────────────────────────────────────────────────────────

/// Return the smaller of two floating-point values.
inline float fmin(float a, float b) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::fmin(a, b);
#else
    return std::fmin(a, b);
#endif
}

inline double fmin(double a, double b) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::fmin(a, b);
#else
    return std::fmin(a, b);
#endif
}

// ── fmax ─────────────────────────────────────────────────────────────

/// Return the larger of two floating-point values.
inline float fmax(float a, float b) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::fmax(a, b);
#else
    return std::fmax(a, b);
#endif
}

inline double fmax(double a, double b) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::fmax(a, b);
#else
    return std::fmax(a, b);
#endif
}

// ── tan ──────────────────────────────────────────────────────────────

/// Compute the tangent of an angle (in radians).
inline float tan(float x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::tan(x);
#else
    return std::tan(x);
#endif
}

inline double tan(double x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::tan(x);
#else
    return std::tan(x);
#endif
}

// ── sin ──────────────────────────────────────────────────────────────

/// Compute the sine of an angle (in radians).
inline float sin(float x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::sin(x);
#else
    return std::sin(x);
#endif
}

inline double sin(double x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::sin(x);
#else
    return std::sin(x);
#endif
}

// ── cos ──────────────────────────────────────────────────────────────

/// Compute the cosine of an angle (in radians).
inline float cos(float x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::cos(x);
#else
    return std::cos(x);
#endif
}

inline double cos(double x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::cos(x);
#else
    return std::cos(x);
#endif
}

// ── pow ──────────────────────────────────────────────────────────────

/// Raise a number to a power: x^y.
inline float pow(float x, float y) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::pow(x, y);
#else
    return std::pow(x, y);
#endif
}

inline double pow(double x, double y) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::pow(x, y);
#else
    return std::pow(x, y);
#endif
}

// ── floor ─────────────────────────────────────────────────────────────

/// Round down to the nearest integer (toward negative infinity).
inline float floor(float x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::floor(x);
#else
    return std::floor(x);
#endif
}

inline double floor(double x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::floor(x);
#else
    return std::floor(x);
#endif
}

// ── atan2 ─────────────────────────────────────────────────────────────

/// Two-argument arctangent: angle of the point (y, x) in radians.
inline float atan2(float y, float x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::atan2(y, x);
#else
    return std::atan2(y, x);
#endif
}

inline double atan2(double y, double x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::atan2(y, x);
#else
    return std::atan2(y, x);
#endif
}

// ── asin ──────────────────────────────────────────────────────────────

/// Arc sine: inverse of sine, in radians.
inline float asin(float x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::asin(x);
#else
    return std::asin(x);
#endif
}

inline double asin(double x) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::asin(x);
#else
    return std::asin(x);
#endif
}

// ── clamp ─────────────────────────────────────────────────────────────

/// Clamp x to the inclusive range [lo, hi].
inline float clamp(float x, float lo, float hi) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::clamp(x, lo, hi);
#else
    return std::clamp(x, lo, hi);
#endif
}

inline double clamp(double x, double lo, double hi) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::clamp(x, lo, hi);
#else
    return std::clamp(x, lo, hi);
#endif
}

} // namespace math
