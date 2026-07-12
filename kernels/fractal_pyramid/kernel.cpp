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
static KernelDesc desc = {"fractal_pyramid", "IFS fractal pyramid", 1, 2, src};
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
    g_rt->foreach_pixel<class FractalPyramidPixelKernel>(w, h, [=](int x, int y, int fi) {
        float as = (float)w/h;
        float px = (2.0f*(x+0.5f)/w-1)*as, py = 1-2.0f*(y+0.5f)/h;
        st_float3 ro = {2*sinf(time*0.1f),1.5f,2*cosf(time*0.1f)}, ta = {0,0,0};
        st_float3 fw = normalize(ta-ro), ri = normalize(cross(fw,{0,1,0})), up = cross(ri,fw);
        st_float3 rd = normalize(px*ri+py*up+1.5f*fw);
        float tr = 0; st_float3 col = {0,0,0};
        for (int i = 0; i < 100; ++i) {
            st_float3 p = ro+rd*tr;
            float sc = 1;
            for (int j = 0; j < 8; ++j) {
                p.y = fabsf(p.y)-0.3f/sc;
                float cx = cosf(1.57f*0.5f), sx = sinf(1.57f*0.5f);
                float ny = p.y*cx - p.z*sx, nz = p.y*sx + p.z*cx;
                p.y = ny; p.z = nz;
                float cy = cosf(time*0.2f), sy = sinf(time*0.2f);
                float nx = p.x*cy + p.z*sy; p.z = -p.x*sy + p.z*cy; p.x = nx;
                p.x = p.x*2 - 0.5f; p.y = p.y*2; p.z = p.z*2 - 0.5f;
                sc *= 2;
                p.x = fabsf(p.x); p.y = fabsf(p.y); p.z = fabsf(p.z);
            }
            float d = sqrtf(p.x*p.x+p.y*p.y+p.z*p.z)/sc - 0.2f;
            if (d < 0.002f) {
                col = lerp(make_float3(0.8f,0.2f,0.1f),make_float3(0.1f,0.2f,0.8f),fabsf(sinf(time+p.x*5)));
                col = col * (0.5f+0.5f*fabsf(sinf(time*3+i*0.5f)));
                break;
            }
            tr += d; if(tr>10){col=lerp(make_float3(0,0,0.05f),make_float3(0.1f,0.1f,0.2f),0.5f+0.5f*rd.y);break;}
        }
        col = tonemap(col);
        int b = fi*4; ac[b+0] += col.x; ac[b+1] += col.y; ac[b+2] += col.z; ac[b+3] += 1;
    });
}

extern "C" void shutdown_kernel(KERNEL_QUEUE_PARAM) {}
