// Raymarching Primitives — SDF primitives showcase with raymarching.
// Inspired by Inigo Quilez's #1 most popular Shadertoy.

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
static KernelDesc desc = {"raymarching_primitives",
    "SDF primitives: spheres, boxes, torii with raymarching lighting",
    1, 2, src};
extern "C" KernelDesc *get_kernel_desc() { return &desc; }

struct MapCtx { float time; };

static float map_sdf(st_float3 p, const MapCtx &ctx) {
    float gnd = p.y + 1.0f;
    float s1 = length(p + make_float3(-3.0f, 0.5f + 0.3f * sinf(ctx.time + p.x), 0.0f)) - 0.8f;
    st_float3 bq = {fabsf(p.x) - 0.6f, fabsf(p.y + 0.5f) - 0.6f, fabsf(p.z) - 0.6f};
    float sx = fmaxf(bq.x, 0.0f), sy = fmaxf(bq.y, 0.0f), sz = fmaxf(bq.z, 0.0f);
    float s2 = sqrtf(sx*sx + sy*sy + sz*sz) + fminf(fmaxf(bq.x, fmaxf(bq.y, bq.z)), 0.0f);
    return fminf(fminf(s1, s2), gnd);
}

static st_float3 calc_norm(st_float3 p, const MapCtx &ctx) {
    float e = 0.001f;
    float dx = map_sdf(p+make_float3(e,0,0),ctx) - map_sdf(p-make_float3(e,0,0),ctx);
    float dy = map_sdf(p+make_float3(0,e,0),ctx) - map_sdf(p-make_float3(0,e,0),ctx);
    float dz = map_sdf(p+make_float3(0,0,e),ctx) - map_sdf(p-make_float3(0,0,e),ctx);
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

    g_rt->foreach_pixel<class RaymarchingPrimitivesPixelKernel>(w, h, [=](int x, int y, int fi) {
        float as = (float)w/h;
        float px = (2.0f*(x+0.5f)/w-1)*as, py = 1-2.0f*(y+0.5f)/h;
        float ca = time * 0.15f;
        st_float3 ro = {6*sinf(ca),2.5f,6*cosf(ca)}, ta = {0,0,-1.5f};
        st_float3 fw = normalize(ta-ro), ri = normalize(cross(fw,{0,1,0})), up = cross(ri,fw);
        st_float3 rd = normalize(px*ri+py*up+2*fw);
        float t = 0; st_float3 col = {0,0,0};
        for (int i = 0; i < 80; ++i) {
            st_float3 pos = ro+rd*t;
            float d = map_sdf(pos, ctx);
            if (d < 0.002f) {
                st_float3 n = calc_norm(pos, ctx);
                st_float3 ld = normalize(make_float3(1,2,1));
                float df = fmaxf(dot(n,ld),0)*0.6f+0.3f+fmaxf(dot(n,{0,1,0}),0)*0.2f;
                col = lerp(make_float3(0.8f,0.2f,0.2f),make_float3(0.2f,0.6f,0.8f),0.5f+0.5f*sinf(pos.x*3))*df;
                if (pos.y < -0.9f) col = make_float3(0.3f*df,0.3f*df,0.3f*df);
                break;
            }
            t += d; if (t > 20) { float st = 0.5f*(rd.y+1); col = lerp(make_float3(0.1f,0.1f,0.2f),make_float3(0.4f,0.6f,0.8f),st); break; }
        }
        int b = fi*4; ac[b+0] += col.x; ac[b+1] += col.y; ac[b+2] += col.z; ac[b+3] += 1;
    });
}

extern "C" void shutdown_kernel(KERNEL_QUEUE_PARAM) {}
