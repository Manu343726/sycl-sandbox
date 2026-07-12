#pragma once

/// @file
/// Tone-mapping stage of the display pipeline.
///
/// Reads the persistent float4 accumulation buffer, normalizes each pixel by
/// its true per-pixel sample count (accum.w — NOT a host frame counter, which
/// is what made the old path ~2× too dark), applies Reinhard tone mapping and
/// sRGB gamma, packs to RGBA8, and writes into the display slot Y-flipped so
/// the OpenGL texture is oriented for a standard (0,0)-(1,1) UV draw.
///
/// enqueue() submits a SYCL kernel and returns its event (no wait — the
/// in-order queue chains it after the render kernel, and the display target
/// promotes the slot to READY when the event completes).  Its implementation
/// lives in tonemap_kernel.cpp, compiled with add_sycl_to_target into the
/// sandbox_tonemap library — the main app is a plain GCC build and cannot
/// carry SYCL device code itself.  run_cpu() is the synchronous host path
/// for the software/native backend.

#include <sycl-sandbox/profiler_device.h>
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace tonemap {

inline void map_pixel(const float *accum, uint8_t *out, int w, int h,
                      int x, int y) {
    size_t src = ((size_t)y * w + x) * 4;
    float n = accum[src + 3];
    float inv = n > 1.f ? 1.f / n : 1.f;
    float r = accum[src + 0] * inv;
    float g = accum[src + 1] * inv;
    float b = accum[src + 2] * inv;
    r = r / (1.f + r);
    g = g / (1.f + g);
    b = b / (1.f + b);
    r = std::pow(std::clamp(r, 0.f, 1.f), 1.f / 2.2f);
    g = std::pow(std::clamp(g, 0.f, 1.f), 1.f / 2.2f);
    b = std::pow(std::clamp(b, 0.f, 1.f), 1.f / 2.2f);
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
                    int w, int h, profiler::DeviceRing prof = {});

/// Synchronous CPU tone map for the native/software backend (accum and out
/// are plain host pointers, no queue).
inline void run_cpu(const float *accum, uint8_t *out, int w, int h) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            map_pixel(accum, out, w, h, x, y);
}

} // namespace tonemap
