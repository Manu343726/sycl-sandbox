#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/kernel/params.h>
#include <sycl-sandbox/kernel/stats.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include "../shadertoy_math.h"

static rt::Runtime *g_rt = nullptr;
extern "C" void set_runtime(rt::Runtime *rt) { g_rt = rt; }

static rt::ParamLookup s_lookup;
extern "C" void set_param_lookup(const rt::ParamLookup *l) {
    s_lookup = l ? *l : rt::ParamLookup{};
}
static rt::StatWriter s_stat_writer;
extern "C" void set_stat_lookup(const rt::StatWriter *w) {
    s_stat_writer = w ? *w : rt::StatWriter{};
}
static const char *src[] = {"kernel.cpp", nullptr};
static KernelDesc desc = {"creation", "Animated particle-like abstract creation", 1, 2, src};
extern "C" KernelDesc *get_kernel_desc() { return &desc; }
extern "C" void init_kernel(KERNEL_QUEUE_PARAM, int, int, const void *b, size_t) {
    s_lookup.set_buffer(b);
}

extern "C" void render_kernel(KERNEL_QUEUE_PARAM, const RenderContext *ctx) {
    if (!ctx || !g_rt) return;
    s_lookup.set_buffer(ctx->params);
    float time = s_lookup.read<float>("time");
    auto *ac = ctx->accum;
    int w = ctx->width;
    int h = ctx->height;
    g_rt->foreach_pixel<class CreationPixelKernel>(w, h, [=](int x, int y, int fi) {
        float as = (float)w/h;
        float px = (2.0f*(x+0.5f)/w-1)*as, py = 1-2.0f*(y+0.5f)/h;
        st_float3 col = {0,0,0};
        for (int i = 0; i < 30; ++i) {
            float fi2 = i + time*0.5f;
            st_float3 pos = {sinf(fi2*1.3f+time*0.7f)*cosf(fi2*0.7f+time*0.3f)*1.5f,
                             sinf(fi2*0.9f+time*0.5f)*1.5f,
                             cosf(fi2*1.1f+time*0.4f)*cosf(fi2*0.7f+time*0.3f)*1.5f};
            float sz = 0.15f+0.1f*sinf(fi2*2+time);
            float px2 = px - pos.x, py2 = py - pos.y;
            float d = sqrtf(px2*px2 + py2*py2 + pos.z*pos.z) - sz*0.5f;
            float glow = fmaxf(1-d*4,0);
            st_float3 ci = {0.5f+0.5f*sinf(fi2),0.3f+0.3f*cosf(fi2*1.7f),0.8f+0.2f*sinf(fi2*0.3f)};
            col = col + ci*glow*0.3f;
        }
        col = tonemap(col);
        int b = fi*4; ac[b+0] += col.x; ac[b+1] += col.y; ac[b+2] += col.z; ac[b+3] += 1;
    });
}

extern "C" void shutdown_kernel(KERNEL_QUEUE_PARAM) {}
