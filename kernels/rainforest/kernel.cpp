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
static KernelDesc desc = {"rainforest", "Procedural rainforest canopy", 1, 2, src};
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
    g_rt->foreach_pixel<class RainforestPixelKernel>(w, h, [=](int x, int y, int fi) {
        float as = (float)w/h;
        float px = (2.0f*(x+0.5f)/w-1)*as, py = 1-2.0f*(y+0.5f)/h;
        st_float3 ro = {0,1.5f,0}, ta = {sinf(time*0.1f)*2,1,cosf(time*0.1f)*2};
        st_float3 fw = normalize(ta-ro), ri = normalize(cross(fw,{0,1,0})), up = cross(ri,fw);
        st_float3 rd = normalize(px*ri+py*up+1.2f*fw);
        float tr = 0; st_float3 col = {0,0,0};
        for (int i = 0; i < 80; ++i) {
            st_float3 p = ro+rd*tr;
            float tree = fbm(make_float2(p.x*0.5f+time*0.02f,p.z*0.5f+time*0.02f));
            float dist = p.y - (tree*0.5f-0.3f);
            if (dist < 0.01f && tr > 0.5f) {
                float fog = fbm(make_float2(p.x*0.1f+time*0.01f,p.z*0.1f+time*0.01f));
                float shade = 0.3f+0.7f*(1-fog);
                col = lerp(make_float3(0.0f,0.3f,0.0f),make_float3(0.1f,0.5f,0.1f),tree)*shade;
                break;
            }
            tr += fmaxf(dist*0.3f,0.005f); if(tr>12){col=make_float3(0.1f,0.15f,0.1f);break;}
        }
        col = col + make_float3(0.02f,0.05f,0.01f);
        col = tonemap(col);
        int b = fi*4; ac[b+0] += col.x; ac[b+1] += col.y; ac[b+2] += col.z; ac[b+3] += 1;
    });
}

extern "C" void shutdown_kernel(KERNEL_QUEUE_PARAM) {}
