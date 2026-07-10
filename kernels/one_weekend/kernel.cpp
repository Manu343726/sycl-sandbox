#include "sandbox_api.h"
#include "kernel.h"
#include "rt/math.h"
#include "rt/types.h"
#include "rt/camera.h"
#include "rt/trace.h"
#include "rt/params.h"
#include <cstdlib>
#include <cstring>

static ParamMeta params_meta[] = {
    { "spp_frame","Samples per frame", ParamType::INT, .range={.i={1,64,1}}, .default_i=1 },
    { "max_bounces","Max ray depth", ParamType::INT, .range={.i={1,100,1}}, .default_i=10 },
    { "cam_eye","Camera pos", ParamType::VEC3, .default_c3={13,2,3} },
    { "cam_at","Look-at", ParamType::VEC3, .default_c3={0,0,0} },
    { "cam_fov","FOV", ParamType::FLOAT, .range={.f={1,120,1}}, .default_f=20 },
    { "cam_aperture","Aperture", ParamType::FLOAT, .range={.f={0,1,0.01f}}, .default_f=0.1f },
    { "cam_up","Up", ParamType::VEC3, .default_c3={0,1,0} },
    { "num_spheres","Num random spheres", ParamType::INT, .range={.i={0,500,1}}, .default_i=11 },
    { "ground_color","Ground albedo", ParamType::COLOR_RGB, .default_c3={0.5f,0.5f,0.5f} },
    { "background","Sky", ParamType::COLOR_RGB, .default_c3={0.5f,0.7f,1} },
};

enum P { P_NUM=RT_NUM_STD_PARAMS, P_GROUND=P_NUM+1, P_BG=P_GROUND+3 };
static const char* srcs[] = {"kernel.cpp","kernel.h",nullptr};
static KernelDesc desc = {
    "one_weekend","Raytracing in One Weekend — random spheres",
    10,params_meta,0,4096,2,srcs
};
extern "C" KernelDesc* get_kernel_desc() {
    desc.params_buffer_size=0;
    for(int i=0;i<desc.param_count;i++) desc.params_buffer_size+=param_buffer_size(params_meta[i]);
    return &desc;
}

static rt::Object* g_objs=nullptr;
static int g_n=0;
static rt::float3 g_bg={0.5f,0.7f,1};

static float rndf() { return (float)rand()*(1.f/2147483647.f); }

extern "C" void init_kernel(sycl::queue* q,int,int,const void* params,size_t) {
    auto* p=(const float*)params;
    int n=(int)p[P_NUM];
    rt::float3 gc; memcpy(&gc,p+P_GROUND,12);
    memcpy(&g_bg,p+P_BG,12);
    if(g_objs){sycl::free(g_objs,*q);g_objs=nullptr;}
    srand(42);
    int cap=n+4+3;
    auto* objs=new rt::Object[cap];
    int c=0;

    auto sph=[&](rt::float3 cen,float rad,rt::MatType mt,rt::float3 alb,float fz,float ir,rt::float3 em){
        objs[c].geom.type=rt::GEOM_SPHERE; objs[c].geom.center=cen; objs[c].geom.radius=rad;
        objs[c].mat.type=mt; objs[c].mat.albedo=alb; objs[c].mat.fuzz=fz; objs[c].mat.ir=ir; objs[c].mat.emit=em;
        c++;
    };
    auto lm=[&](rt::float3 c_){ return rt::MAT_LAMBERTIAN; };
    auto mt=[&](rt::float3 c_,float f){ return rt::MAT_METAL; };
    auto di=rt::MAT_DIELECTRIC;

    sph({0,-1000,0},1000,rt::MAT_LAMBERTIAN,gc,0,0,{0,0,0});
    for(int k=0;k<n;){
        float x=-10+20*rndf(),z=-10+20*rndf();
        rt::float3 c={x,0.2f,z};
        if(rt::len(c)<=0.9f) continue;
        rt::float3 col={rndf()*rndf(),rndf()*rndf(),rndf()*rndf()};
        float d=rndf();
        if(d<0.6f)      sph(c,0.2f,rt::MAT_LAMBERTIAN,col,0,0,{0,0,0});
        else if(d<0.85f) sph(c,0.2f,rt::MAT_METAL,col,0.5f*rndf(),0,{0,0,0});
        else             sph(c,0.2f,rt::MAT_DIELECTRIC,{1,1,1},0,1.5f,{0,0,0});
        k++;
    }
    sph({4,1,0},1,rt::MAT_METAL,{0.7f,0.6f,0.5f},0,0,{0,0,0});
    sph({-4,1,0},1,rt::MAT_LAMBERTIAN,{0.4f,0.2f,0.1f},0,0,{0,0,0});
    sph({0,1,0},1,rt::MAT_DIELECTRIC,{1,1,1},0,1.5f,{0,0,0});

    g_objs=sycl::malloc_device<rt::Object>(c,*q);
    q->memcpy(g_objs,objs,c*sizeof(rt::Object)).wait();
    g_n=c;
    delete[] objs;
}

extern "C" void render_kernel(sycl::queue* q,int w,int h,const void* p,void* accum,int si) {
    rt::render_main(q,w,h,(const float*)p,(float*)accum,si,g_objs,g_n,
        [bg=g_bg](const rt::Ray& ray)->rt::float3{
            float t=0.5f*(ray.dir.y+1); return rt::lerp({1,1,1},bg,t);
        });
}

extern "C" void shutdown_kernel(sycl::queue* q) {
    if(g_objs){sycl::free(g_objs,*q);g_objs=nullptr;}
}
