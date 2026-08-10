#pragma once

/// @file
/// Tone-mapping stage of the display pipeline.
///
/// Reads the persistent float4 accumulation buffer, normalizes each pixel by
/// its true per-pixel sample count (accum.w — NOT a host frame counter, which
/// is what made the old path ~2× too dark), applies the configured tone-map
/// operator + sRGB gamma, packs to RGBA8, and writes into the display slot
/// Y-flipped so the OpenGL texture is oriented for a standard (0,0)-(1,1)
/// UV draw.
///
/// Behavior is controlled by the standard scene params (`tonemap_enabled`,
/// `tonemap_operator`, `tonemap_exposure`, `tonemap_gamma`) which the render
/// thread reads and packs into tonemap::Params — see render_loop.h.  When
/// disabled, accumulated linear values are normalized and hard-clamped to
/// [0,1] with no operator or gamma.
///
/// enqueue() submits a SYCL kernel and returns its event (no wait — the
/// in-order queue chains it after the render kernel, and the display target
/// promotes the slot to READY when the event completes).  Its implementation
/// lives in tonemap_kernel.cpp, compiled with add_sycl_to_target into the
/// sandbox_tonemap library — the main app is a plain GCC build and cannot
/// carry SYCL device code itself.  run_cpu() is the synchronous host path
/// for the software/native backend.

#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/colorchecker.h>
#include <sycl-sandbox/math.h>
#include <sycl-sandbox/tonemap_ops.h>
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cmath>

namespace tonemap {

// ── Config packed from the standard tone-map scene params ─────────────
struct Params {
    bool  enabled    = false;   ///< tonemap_enabled (default off)
    int   operator_id = Operator::Reinhard;
    float exposure   = 1.0f;    ///< linear HDR multiplier before the operator
    float gamma      = 2.2f;    ///< display gamma for pow(c, 1/gamma)
};

inline void map_pixel(const float *accum, uint8_t *out, int w, int h,
                      int x, int y, const Params &p = {}) {
    size_t src = ((size_t)y * w + x) * 4;
    float n = accum[src + 3];
    float inv = n > 1.f ? 1.f / n : 1.f;
    float r = accum[src + 0] * inv * p.exposure;
    float g = accum[src + 1] * inv * p.exposure;
    float b = accum[src + 2] * inv * p.exposure;
    if ( p.enabled ) {
        r = apply_operator(r, p.operator_id);
        g = apply_operator(g, p.operator_id);
        b = apply_operator(b, p.operator_id);
    }
    if ( p.enabled ) {
        float ginv = 1.0f / p.gamma;
        r = math::pow(math::clamp(r, 0.f, 1.f), ginv);
        g = math::pow(math::clamp(g, 0.f, 1.f), ginv);
        b = math::pow(math::clamp(b, 0.f, 1.f), ginv);
    } else {
        r = math::clamp(r, 0.f, 1.f);
        g = math::clamp(g, 0.f, 1.f);
        b = math::clamp(b, 0.f, 1.f);
    }
    size_t dy = (size_t)(h - 1 - y);      // Y-flip for OpenGL origin
    size_t dst = (dy * w + x) * 4;
    out[dst + 0] = (uint8_t)(r * 255.f);
    out[dst + 1] = (uint8_t)(g * 255.f);
    out[dst + 2] = (uint8_t)(b * 255.f);
    out[dst + 3] = 255;
}

/// Enqueue tone mapping of `accum` (device float4[w*h]) into `out` (RGBA8,
/// w*h*4 bytes, host- or device-visible USM).  Enqueue-only; returns the
/// kernel's event so the caller can gate slot publication on completion.
/// `prof` is the device profiler ring (inactive ring → no-op zones).
sycl::event enqueue(sycl::queue &q, const float *accum, uint8_t *out,
                    int w, int h, const Params &p,
                    profiler::DeviceRing prof = {});

/// Synchronous CPU tone map for the native/software backend (accum and out
/// are plain host pointers, no queue).
inline void run_cpu(const float *accum, uint8_t *out, int w, int h,
                    const Params &p = {}) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            map_pixel(accum, out, w, h, x, y, p);
}

// ── ColorChecker debug modes ──────────────────────────────────────────
//
// Debug-only fill paths that replace the render kernel (and optionally the
// tone-map) with the standard 24-patch ColorChecker chart, see
// <sycl-sandbox/colorchecker.h>.  `prof` is the device profiler ring
// (inactive ring → no-op zones).

/// Enqueue a fill of `accum` (device float4[w*h]) with the ColorChecker
/// chart encoded as *inverse tone-mapped* linear values (alpha = 1), so
/// the standard tone-map step reproduces the reference sRGB colors
/// exactly.  Enqueue-only; chain tonemap::enqueue() after it on the same
/// in-order queue.  Implemented in tonemap_kernel.cpp.
sycl::event enqueue_colorchecker_fill(sycl::queue &q, float *accum,
                                      int w, int h,
                                      profiler::DeviceRing prof = {});

/// Enqueue a write of the ColorChecker chart (reference sRGB, RGBA8,
/// alpha 255) directly into `out`, bypassing the tone-map.  Output is
/// Y-flipped like the tone-map's so the display texture stays oriented
/// for the viewport.  Implemented in tonemap_kernel.cpp.
sycl::event enqueue_colorchecker_raw(sycl::queue &q, uint8_t *out,
                                     int w, int h);

/// Synchronous CPU variants for the native/software backend.
inline void run_cpu_colorchecker_fill(float *accum, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float *s =
                colorchecker::SRGB_REF[colorchecker::patch_index(x, y, w, h)];
            size_t dst = ((size_t)y * w + x) * 4;
            accum[dst + 0] = colorchecker::srgb_to_accum_linear(s[0]);
            accum[dst + 1] = colorchecker::srgb_to_accum_linear(s[1]);
            accum[dst + 2] = colorchecker::srgb_to_accum_linear(s[2]);
            accum[dst + 3] = 1.f;
        }
    }
}

inline void run_cpu_colorchecker_raw(uint8_t *out, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float *s =
                colorchecker::SRGB_REF[colorchecker::patch_index(x, y, w, h)];
            size_t dy = (size_t)(h - 1 - y);      // Y-flip for OpenGL origin
            size_t dst = (dy * w + x) * 4;
            out[dst + 0] = (uint8_t)s[0];
            out[dst + 1] = (uint8_t)s[1];
            out[dst + 2] = (uint8_t)s[2];
            out[dst + 3] = 255;
        }
    }
}

} // namespace tonemap
