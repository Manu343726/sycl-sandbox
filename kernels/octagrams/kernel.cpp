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
static KernelDesc desc = {"octagrams", "Geometric octagram patterns", 1, 2, src};
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
    g_rt->foreach_pixel<class OctagramsPixelKernel>(w, h, [=](int x, int y, int fi) {
        float as = (float)w/h;
        float px = (2.0f*(x+0.5f)/w-1)*as, py = 1-2.0f*(y+0.5f)/h;
        st_float2 uv = {px*3, py*3};
        float a = atan2f(uv.y, uv.x);
        float r = sqrtf(uv.x*uv.x+uv.y*uv.y);
        int n = 8;
        float sa = sinf(a*n)/coshf(r*2);
        float pat = fabsf(sa)*0.8f+0.2f*noise(make_float2(uv.x*4+time*0.5f,uv.y*4+time*0.5f));
        float rc = cosf(time*0.15f), rs = sinf(time*0.15f);
        float ux2 = uv.x*rc-uv.y*rs, uy2 = uv.x*rs+uv.y*rc;
        float sa2 = sinf(atan2f(uy2,ux2)*n)/coshf(r*3);
        pat = pat*0.5f+0.5f*fabsf(sa2);
        st_float3 col = lerp(make_float3(0.1f,0.0f,0.2f),make_float3(0.8f,0.2f,0.5f),pat);
        col = col + make_float3(0.3f,0.1f,0.5f)*(1-fminf(r*0.3f,1))*0.3f;
        col = tonemap(col);
        int b = fi*4; ac[b+0] += col.x; ac[b+1] += col.y; ac[b+2] += col.z; ac[b+3] += 1;
    });
}

extern "C" void shutdown_kernel(KERNEL_QUEUE_PARAM) {}
