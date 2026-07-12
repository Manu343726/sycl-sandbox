#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/kernel/params.h>
#include <sycl-sandbox/kernel/stats.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include <cstring>

// ── Runtime (set by host to dispatch SYCL vs software) ───────────────
static rt::Runtime *g_rt = nullptr;

extern "C" void set_runtime(rt::Runtime *rt) {
    g_rt = rt;
}

// ── Param lookup (host-populated from YAML scene descriptor) ────────
static rt::ParamLookup s_lookup;

extern "C" void set_param_lookup(const rt::ParamLookup *lookup) {
    if (lookup) {
        s_lookup = *lookup;
    } else {
        s_lookup = rt::ParamLookup{};
    }
}

// ── Stat writer (kernel-populated, host reads after each frame) ───────
static rt::StatWriter s_stat_writer;

extern "C" void set_stat_lookup(const rt::StatWriter *writer) {
    if (writer) {
        s_stat_writer = *writer;
    } else {
        s_stat_writer = rt::StatWriter{};
    }
}

static const char *source_files[] = {"kernel.cpp", nullptr};
static KernelDesc desc =
    {"mandelbrot", "Colored Mandelbrot fractal", 1, 2, source_files};

extern "C" KernelDesc *get_kernel_desc() {
    return &desc;
}

extern "C" void init_kernel(KERNEL_QUEUE_PARAM, int, int, const void *buf, size_t) {
    PROFILER_ZONE("init_kernel");
    s_lookup.set_buffer(buf);
}

// ── render_kernel ───────────────────────────────────────────────────
// Single implementation — Runtime handles SYCL vs software dispatch.

extern "C" void render_kernel(KERNEL_QUEUE_PARAM, const RenderContext *ctx) {
    PROFILER_ZONE("render_kernel");
    if (!ctx || !g_rt) return;

    s_lookup.set_buffer(ctx->params);
    float center_x = s_lookup.read<float>("center_x");
    float center_y = s_lookup.read<float>("center_y");
    float zoom_level = s_lookup.read<float>("zoom");
    int max_iterations = s_lookup.read<int>("max_iterations");
    float aspect_ratio = (float)ctx->width / (float)ctx->height;

    auto *accum = ctx->accum;

    g_rt->foreach_pixel<class MandelbrotPixelKernel>(ctx->width, ctx->height, [=](int x, int y, int flat_index) {
        PROFILER_ZONE("pixel");

        // Map the pixel to a point in the complex plane
        float px = (x + 0.5f) / (float)ctx->width;
        float py = (y + 0.5f) / (float)ctx->height;
        float map_x = center_x + (px - 0.5f) * zoom_level * aspect_ratio;
        float map_y = center_y + (py - 0.5f) * zoom_level;

        // Iterate the Mandelbrot recurrence:  z ← z² + c
        float zx = 0.0f, zy = 0.0f;
        float zx2 = 0.0f, zy2 = 0.0f;
        int iteration = 0;

        while ( iteration < max_iterations && zx2 + zy2 < 4.0f ) {
            PROFILER_ZONE("mandelbrot iteration");

            zy = 2.0f * zx * zy + map_y;
            zx = zx2 - zy2 + map_x;
            zx2 = zx * zx;
            zy2 = zy * zy;
            iteration++;
        }

        // Convert the iteration count to an HSV colour, then to RGB
        float red, green, blue;
        if ( iteration == max_iterations ) {
            // Interior of the set — render as black
            red = 0.0f; green = 0.0f; blue = 0.0f;
        } else {
            // Map iteration count to a hue, then apply HSV-to-RGB conversion
            float t = (float)iteration / 50.0f;
            if ( t > 1.0f ) t = 1.0f;

            float hue = t;
            float saturation = 0.9f;
            float value = 0.8f + t * 0.2f;

            float hue_sector = hue * 6.0f;
            int sector_index = (int)hue_sector;
            float fractional_part = hue_sector - (float)sector_index;

            float p = value * (1.0f - saturation);
            float q = value * (1.0f - saturation * fractional_part);
            float t_component = value * (1.0f - saturation * (1.0f - fractional_part));

            switch ( sector_index % 6 ) {
                case 0: red = value; green = t_component; blue = p; break;
                case 1: red = q;     green = value;       blue = p; break;
                case 2: red = p;     green = value;       blue = t_component; break;
                case 3: red = p;     green = q;           blue = value; break;
                case 4: red = t_component; green = p;     blue = value; break;
                default: red = value; green = p;          blue = q; break;
            }
        }

        // Accumulate the colour into the persistent buffer
        int base = flat_index * 4;
        accum[base + 0] += red;
        accum[base + 1] += green;
        accum[base + 2] += blue;
        accum[base + 3] += 1;
    });

    // Write statistics to the per-frame block
    if (ctx->stats) {
        ctx->stats->write<int>("max_iterations", max_iterations);
        ctx->stats->write<float>("zoom_level", zoom_level);
    }
}

extern "C" void shutdown_kernel(KERNEL_QUEUE_PARAM) {
    PROFILER_ZONE("shutdown_kernel");
}
