/// @file
/// SYCL implementation of the display tone-map.  This translation unit is
/// the only device code in the main application, so it is compiled with
/// add_sycl_to_target (acpp) into its own shared library and linked into
/// the GCC-built executable — the same split the kernel .so's use.

#include "tonemap.h"

namespace tonemap {

sycl::event enqueue(sycl::queue &q, const float *accum, uint8_t *out,
                    int w, int h, const Params &p,
                    profiler::DeviceRing prof) {
    return q.parallel_for(sycl::range<2>{(size_t)h, (size_t)w},
                          [=](sycl::item<2> it) {
        int x = it[1], y = it[0];
        uint32_t lid = (uint32_t)it.get_linear_id();
        PROFILER_DEVICE_ZONE(prof, "tonemap_px", lid);
        size_t src = ((size_t)y * w + x) * 4;
        float n = accum[src + 3];
        float inv = n > 1.f ? 1.f / n : 1.f;
        float r = accum[src + 0] * inv * p.exposure;
        float g = accum[src + 1] * inv * p.exposure;
        float b = accum[src + 2] * inv * p.exposure;
        if ( p.enabled ) {
            switch ( p.operator_id ) {
                case Operator::Reinhard:
                    r = r / (1.f + r);
                    g = g / (1.f + g);
                    b = b / (1.f + b);
                    break;
                case Operator::ACES: {
                    const float a = 2.51f, c = 2.43f, d = 0.59f, e = 0.14f;
                    r = (r * (a * r + 0.03f)) / (r * (c * r + d) + e);
                    g = (g * (a * g + 0.03f)) / (g * (c * g + d) + e);
                    b = (b * (a * b + 0.03f)) / (b * (c * b + d) + e);
                    break;
                }
                case Operator::Filmic: {
                    const float A = 0.15f, B = 0.50f, C = 0.10f;
                    const float D = 0.20f, E = 0.02f, F = 0.30f;
                    r = ((r * (A * r + C * B) + D * E) / (r * (A * r + B) + D * F)) - E / F;
                    g = ((g * (A * g + C * B) + D * E) / (g * (A * g + B) + D * F)) - E / F;
                    b = ((b * (A * b + C * B) + D * E) / (b * (A * b + B) + D * F)) - E / F;
                    break;
                }
                default:
                    r = r / (1.f + r);
                    g = g / (1.f + g);
                    b = b / (1.f + b);
                    break;
            }
            float ginv = 1.f / p.gamma;
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
    });
}

sycl::event enqueue_colorchecker_fill(sycl::queue &q, float *accum,
                                      int w, int h,
                                      profiler::DeviceRing prof) {
    return q.parallel_for(sycl::range<2>{(size_t)h, (size_t)w},
                          [=](sycl::item<2> it) {
        int x = it[1], y = it[0];
        uint32_t lid = (uint32_t)it.get_linear_id();
        PROFILER_DEVICE_ZONE(prof, "colorchecker_px", lid);
        const float *s =
            colorchecker::SRGB_REF[colorchecker::patch_index(x, y, w, h)];
        // Inverse of the tone-map (Reinhard + gamma 1/2.2): after it the
        // patch reproduces exactly its reference sRGB value.
        auto inv = [](float c) {
            float y = math::pow(c / 255.f, 2.2f);
            return y / (1.f - y);
        };
        size_t dst = ((size_t)y * w + x) * 4;
        accum[dst + 0] = inv(s[0]);
        accum[dst + 1] = inv(s[1]);
        accum[dst + 2] = inv(s[2]);
        accum[dst + 3] = 1.f;
    });
}

sycl::event enqueue_colorchecker_raw(sycl::queue &q, uint8_t *out,
                                     int w, int h) {
    return q.parallel_for(sycl::range<2>{(size_t)h, (size_t)w},
                          [=](sycl::item<2> it) {
        int x = it[1], y = it[0];
        const float *s =
            colorchecker::SRGB_REF[colorchecker::patch_index(x, y, w, h)];
        size_t dy = (size_t)(h - 1 - y);      // Y-flip for OpenGL origin
        size_t dst = (dy * w + x) * 4;
        out[dst + 0] = (uint8_t)s[0];
        out[dst + 1] = (uint8_t)s[1];
        out[dst + 2] = (uint8_t)s[2];
        out[dst + 3] = 255;
    });
}

} // namespace tonemap
