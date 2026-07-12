// Shared Shadertoy-style math utilities for SYCL raymarching kernels.
// Uses plain C++ structs for float2/float3 that work in both SYCL device
// code and native software mode.

#pragma once

#include <cmath>
#include <cstdint>
#include <algorithm>

// ── Simple float2/float3 structs (works everywhere) ──────────────────
struct st_float2 { float x = 0, y = 0; };
struct st_float3 { float x = 0, y = 0, z = 0; };

inline st_float2 make_float2(float v) { return {v, v}; }
inline st_float2 make_float2(float x, float y) { return {x, y}; }
inline st_float3 make_float3(float v) { return {v, v, v}; }
inline st_float3 make_float3(float x, float y, float z) { return {x, y, z}; }

// Arithmetic
inline st_float2 operator+(const st_float2 &a, const st_float2 &b) { return {a.x + b.x, a.y + b.y}; }
inline st_float2 operator-(const st_float2 &a, const st_float2 &b) { return {a.x - b.x, a.y - b.y}; }
inline st_float2 operator*(const st_float2 &a, float s) { return {a.x * s, a.y * s}; }
inline st_float2 operator*(float s, const st_float2 &a) { return {a.x * s, a.y * s}; }
inline st_float2 operator/(const st_float2 &a, float s) { return {a.x / s, a.y / s}; }

inline st_float3 operator+(const st_float3 &a, const st_float3 &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline st_float3 operator-(const st_float3 &a, const st_float3 &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline st_float3 operator*(const st_float3 &a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline st_float3 operator*(float s, const st_float3 &a) { return {a.x * s, a.y * s, a.z * s}; }
inline st_float3 operator/(const st_float3 &a, float s) { return {a.x / s, a.y / s, a.z / s}; }
inline st_float3 operator*(const st_float3 &a, const st_float3 &b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
inline st_float3 operator/(const st_float3 &a, const st_float3 &b) { return {a.x / b.x, a.y / b.y, a.z / b.z}; }

inline float dot(const st_float3 &a, const st_float3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float length(const st_float3 &v) { return sqrtf(dot(v, v)); }
inline st_float3 normalize(const st_float3 &v) { float l = length(v); return l > 0 ? v / l : v; }
inline st_float3 lerp(const st_float3 &a, const st_float3 &b, float t) { return a + (b - a) * t; }
inline float lerp_f(float a, float b, float t) { return a + (b - a) * t; }

inline float fract_f(float x) { return x - floorf(x); }
inline float clamp_f(float v, float lo, float hi) { return fmaxf(lo, fminf(hi, v)); }
inline float smoothstep_f(float a, float b, float t) {
    float x = clamp_f((t - a) / (b - a), 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

// ── Cross product ───────────────────────────────────────────────────
inline st_float3 cross(const st_float3 &a, const st_float3 &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// ── Constants ────────────────────────────────────────────────────────
constexpr float ST_PI = 3.14159265359f;
constexpr float ST_TWO_PI = 6.28318530718f;
constexpr float ST_INV_PI = 0.31830988618f;

// ── 2D rotation ──────────────────────────────────────────────────────
inline st_float2 rot2(float a) {
    float c = cosf(a), s = sinf(a);
    return {c, s};
}

// ── 3D helpers ───────────────────────────────────────────────────────
inline st_float3 rotate_y(st_float3 p, float a) {
    float c = cosf(a), s = sinf(a);
    return {p.x * c + p.z * s, p.y, -p.x * s + p.z * c};
}

inline st_float3 rotate_x(st_float3 p, float a) {
    float c = cosf(a), s = sinf(a);
    return {p.x, p.y * c - p.z * s, p.y * s + p.z * c};
}

// ── Smooth min ───────────────────────────────────────────────────────
inline float smin(float a, float b, float k) {
    float h = fmaxf(k - fabsf(a - b), 0.0f) / k;
    return fminf(a, b) - h * h * h * k * (1.0f / 6.0f);
}

// ── Hash / noise ─────────────────────────────────────────────────────
inline float hash1(float x, float y) {
    float n = sinf(x * 127.1f + y * 311.7f) * 43758.5453123f;
    return fract_f(n);
}

inline float hash1(float p) {
    return fract_f(sinf(p * 127.1f + 311.7f) * 43758.5453123f);
}

inline float noise(st_float2 p) {
    st_float2 i = {floorf(p.x), floorf(p.y)};
    st_float2 f = {fract_f(p.x), fract_f(p.y)};
    st_float2 u = {f.x * f.x * (3.0f - 2.0f * f.x), f.y * f.y * (3.0f - 2.0f * f.y)};
    float a = hash1(i.x, i.y);
    float b = hash1(i.x + 1.0f, i.y);
    float c = hash1(i.x, i.y + 1.0f);
    float d = hash1(i.x + 1.0f, i.y + 1.0f);
    return lerp_f(lerp_f(a, b, u.x), lerp_f(c, d, u.x), u.y);
}

// ── fBM (fractal Brownian motion) ────────────────────────────────────
inline float fbm(st_float2 p) {
    float value = 0.0f, amplitude = 0.5f, frequency = 1.0f;
    for (int i = 0; i < 6; ++i) {
        value += amplitude * noise({p.x * frequency, p.y * frequency});
        frequency *= 2.0f;
        amplitude *= 0.5f;
    }
    return value;
}

// ── Domain warping fBM ──────────────────────────────────────────────
inline float fbm_warped(st_float2 p) {
    float f1 = fbm(p);
    float f2 = fbm({p.x + 5.2f, p.y + 1.3f});
    st_float2 q = {f1, f2};
    float r1 = fbm({p.x + 4.0f * q.x + 1.7f, p.y + 4.0f * q.y + 9.2f});
    float r2 = fbm({p.x + 4.0f * q.x + 8.3f, p.y + 4.0f * q.y + 2.8f});
    st_float2 r = {r1, r2};
    return fbm({p.x + 4.0f * r.x, p.y + 4.0f * r.y});
}

// ── Soft shadow ──────────────────────────────────────────────────────
template <typename Sdf>
inline float soft_shadow(Sdf &&sdf, st_float3 ro, st_float3 rd,
                         float t_min, float t_max, float k) {
    float res = 1.0f;
    for (float t = t_min; t < t_max;) {
        float d = sdf(ro + rd * t);
        if (d < 0.001f) return 0.0f;
        res = fminf(res, k * d / t);
        t += d;
    }
    return res;
}

// ── Ambient occlusion ────────────────────────────────────────────────
template <typename Sdf>
inline float ambient_occlusion(Sdf &&sdf, st_float3 p, st_float3 n) {
    float ao = 0.0f, scale = 1.0f;
    for (int i = 0; i < 5; ++i) {
        float d = sdf(p + n * (0.01f + 0.12f * (float)i));
        ao += (d - 0.01f - 0.12f * (float)i) * scale;
        scale *= 0.5f;
    }
    return 1.0f - fmaxf(ao * 0.2f, 0.0f);
}

// ── Normal from SDF ─────────────────────────────────────────────────
template <typename Sdf>
inline st_float3 calc_normal(Sdf &&sdf, st_float3 p) {
    const float e = 0.001f;
    st_float3 eps_x = {1.0f, -1.0f, -1.0f};
    st_float3 eps_y = {-1.0f, 1.0f, -1.0f};
    st_float3 eps_z = {-1.0f, -1.0f, 1.0f};
    return normalize(
        eps_x * sdf(p + make_float3(e, -e, -e) * 0.5773f) +
        eps_y * sdf(p + make_float3(-e, e, -e) * 0.5773f) +
        eps_z * sdf(p + make_float3(-e, -e, e) * 0.5773f) +
        make_float3(1, 1, 1) * sdf(p + make_float3(e, e, e) * 0.5773f));
}

// ── Tonemap (Reinhard + gamma) ──────────────────────────────────────
inline st_float3 tonemap(st_float3 col) {
    col = col / (col + make_float3(1.0f)); // Reinhard
    col = make_float3(powf(col.x, 1.0f / 2.2f),
                      powf(col.y, 1.0f / 2.2f),
                      powf(col.z, 1.0f / 2.2f));
    return col;
}
