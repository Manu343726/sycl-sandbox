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
static KernelDesc desc = {"seascape", "Procedural ocean scene with animated waves", 1, 2, src};
extern "C" KernelDesc *get_kernel_desc() { return &desc; }

struct MapCtx { float time; };
static float map_sdf(st_float3 p, const MapCtx &ctx) {
    float t = ctx.time;
    float f = fbm(make_float2(p.x*2 + t*0.05f, p.z*2 + t*0.05f));
    float wave = sinf(p.x*0.5f + t*0.3f)*0.3f + sinf(p.z*0.3f + t*0.2f)*0.3f;
    return p.y + 2.0f + f*2.5f - 0.3f*wave;
}
static st_float3 calc_norm(st_float3 p, const MapCtx &ctx) {
    float e = 0.01f;
    float dx = map_sdf(make_float3(p.x+e,p.y,p.z),ctx) - map_sdf(make_float3(p.x-e,p.y,p.z),ctx);
    float dy = map_sdf(make_float3(p.x,p.y+e,p.z),ctx) - map_sdf(make_float3(p.x,p.y-e,p.z),ctx);
    float dz = map_sdf(make_float3(p.x,p.y,p.z+e),ctx) - map_sdf(make_float3(p.x,p.y,p.z-e),ctx);
    return normalize(make_float3(dx,dy,dz));
}
extern "C" void init_kernel(KERNEL_QUEUE_PARAM, int, int, const void *b, size_t) {
    s_lookup.set_buffer(b);
}

extern "C" void render_kernel(KERNEL_QUEUE_PARAM, const RenderContext *render_ctx) {
    if (!render_ctx || !g_rt) return;
    s_lookup.set_buffer(render_ctx->params);
    float time = s_lookup.read<float>("time");
    auto *ac = render_ctx->accum;
    int w = render_ctx->width;
    int h = render_ctx->height;
    MapCtx ctx = {time};
    g_rt->foreach_pixel<class SeascapePixelKernel>(w, h, [=](int x, int y, int fi) {
        float as = (float)w/h;
        float px = (2.0f*(x+0.5f)/w-1)*as, py = 1-2.0f*(y+0.5f)/h;
        float ca = time*0.1f;
        st_float3 ro = {2*sinf(ca),1.5f,2*cosf(ca)}, ta = {0,0,0};
        st_float3 fw = normalize(ta-ro), ri = normalize(cross(fw,{0,1,0})), up = cross(ri,fw);
        st_float3 rd = normalize(px*ri+py*up+1.5f*fw);
        float t = 0; st_float3 col = {0,0,0};
        for (int i = 0; i < 60; ++i) {
            st_float3 pos = ro+rd*t;
            float d = map_sdf(pos, ctx);
            if (d < 0.01f) {
                st_float3 n = calc_norm(pos, ctx);
                float foam = fabsf(d)/0.01f;
                float diff = fmaxf(dot(n,normalize(make_float3(1,2,1))),0)*0.5f+0.5f;
                st_float3 base = lerp(make_float3(0.0f,0.2f,0.4f),make_float3(0.1f,0.6f,0.7f),0.5f+0.5f*noise(make_float2(pos.x*2+time,pos.z*2+time)));
                col = lerp(base,make_float3(1,1,1),1-foam)*diff;
                break;
            }
            t += d; if (t > 15) { col = lerp(make_float3(0.0f,0.05f,0.1f),make_float3(0.2f,0.3f,0.5f),0.5f+0.5f*rd.y); break; }
        }
        int b = fi*4; ac[b+0] += col.x; ac[b+1] += col.y; ac[b+2] += col.z; ac[b+3] += 1;
    });
}

extern "C" void shutdown_kernel(KERNEL_QUEUE_PARAM) {}
