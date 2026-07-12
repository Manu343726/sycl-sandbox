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
static KernelDesc desc = {"lover", "Heart patterns and procedural love-themed art", 1, 2, src};
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
    g_rt->foreach_pixel<class LoverPixelKernel>(w, h, [=](int x, int y, int fi) {
        float as = (float)w/h;
        float px = (2.0f*(x+0.5f)/w-1)*as, py = 1-2.0f*(y+0.5f)/h;
        st_float2 uv = {px*2, py*2};
        float heart = uv.y - 0.7f*sqrtf(fabsf(uv.x)) + 0.3f*sqrtf(fabsf(uv.x+0.3f*sinf(time+uv.x)));
        float d = 1-heart*0.4f;
        st_float3 col = {d*0.8f+0.2f*sinf(time+uv.x*3), d*0.2f+0.1f*sinf(time*0.7f+uv.y*2), d*0.2f};
        float nz = noise(make_float2(uv.x*3+time*0.2f,uv.y*3+time*0.3f))*0.15f;
        col = col + make_float3(nz,nz,nz);
        col = tonemap(col);
        int b = fi*4; ac[b+0] += col.x; ac[b+1] += col.y; ac[b+2] += col.z; ac[b+3] += 1;
    });
}

extern "C" void shutdown_kernel(KERNEL_QUEUE_PARAM) {}
