/// @file
/// SYCL implementation of the display tone-map.  This translation unit is
/// the only device code in the main application, so it is compiled with
/// add_sycl_to_target (acpp) into its own shared library and linked into
/// the GCC-built executable — the same split the kernel .so's use.

#include "tonemap.h"

namespace tonemap {

sycl::event enqueue(sycl::queue &q, const float *accum, uint8_t *out,
                    int w, int h, profiler::DeviceRing prof) {
    return q.parallel_for(sycl::range<2>{(size_t)h, (size_t)w},
                          [=](sycl::item<2> it) {
        int x = it[1], y = it[0];
        uint32_t lid = (uint32_t)it.get_linear_id();
        PROFILER_DEVICE_ZONE(prof, "tonemap_px", lid);
        size_t src = ((size_t)y * w + x) * 4;
        float n = accum[src + 3];
        float inv = n > 1.f ? 1.f / n : 1.f;
        float r = accum[src + 0] * inv;
        float g = accum[src + 1] * inv;
        float b = accum[src + 2] * inv;
        r = r / (1.f + r);
        g = g / (1.f + g);
        b = b / (1.f + b);
        r = sycl::pow(sycl::clamp(r, 0.f, 1.f), 1.f / 2.2f);
        g = sycl::pow(sycl::clamp(g, 0.f, 1.f), 1.f / 2.2f);
        b = sycl::pow(sycl::clamp(b, 0.f, 1.f), 1.f / 2.2f);
        size_t dy = (size_t)(h - 1 - y);      // Y-flip for OpenGL origin
        size_t dst = (dy * w + x) * 4;
        out[dst + 0] = (uint8_t)(r * 255.f);
        out[dst + 1] = (uint8_t)(g * 255.f);
        out[dst + 2] = (uint8_t)(b * 255.f);
        out[dst + 3] = 255;
    });
}

} // namespace tonemap
