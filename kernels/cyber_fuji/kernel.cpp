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
static KernelDesc desc = {"cyber_fuji", "Cyberpunk procedural terrain", 1, 2, src};
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
    g_rt->foreach_pixel<class CyberFujiPixelKernel>(w, h, [=](int x, int y, int fi) {
        float as = (float)w/h;
        float px = (2.0f*(x+0.5f)/w-1)*as, py = 1-2.0f*(y+0.5f)/h;
        float ca = time*0.08f;
        st_float3 ro = {4*sinf(ca),1.5f,4*cosf(ca)}, ta = {0,0.5f,0};
        st_float3 fw = normalize(ta-ro), ri = normalize(cross(fw,{0,1,0})), up = cross(ri,fw);
        st_float3 rd = normalize(px*ri+py*up+1.5f*fw);
        float tr = 0; st_float3 col = {0,0,0};
        for (int i = 0; i < 80; ++i) {
            st_float3 p = ro+rd*tr;
            float h1 = fbm(make_float2(p.x*0.5f+time*0.02f,p.z*0.5f+time*0.02f));
            float terrain = p.y - (h1*1.5f-0.5f);
            if (terrain < 0.005f) {
                float nx = (fbm(make_float2((p.x+0.01f)*0.5f+time*0.02f,p.z*0.5f+time*0.02f))*1.5f-0.5f - h1*1.5f+0.5f) / 0.01f;
                float nz = (fbm(make_float2(p.x*0.5f+time*0.02f,(p.z+0.01f)*0.5f+time*0.02f))*1.5f-0.5f - h1*1.5f+0.5f) / 0.01f;
                st_float3 tn = normalize(make_float3(nx, 1.0f, nz));
                float slope = fmaxf(1-dot(tn,{0,1,0})*3,0);
                col = lerp(make_float3(0.1f,0.2f,0.3f),make_float3(0.8f,0.3f,0.1f),slope);
                col = col * (0.5f+0.5f*fmaxf(dot(tn,normalize(make_float3(1,2,1))),0));
                break;
            }
            tr += fmaxf(terrain*0.5f, 0.005f); if(tr>15){col=lerp(make_float3(0,0.05f,0.1f),make_float3(0.1f,0.2f,0.4f),0.5f+0.5f*rd.y);break;}
        }
        col = tonemap(col);
        int b = fi*4; ac[b+0] += col.x; ac[b+1] += col.y; ac[b+2] += col.z; ac[b+3] += 1;
    });
}

extern "C" void shutdown_kernel(KERNEL_QUEUE_PARAM) {}
