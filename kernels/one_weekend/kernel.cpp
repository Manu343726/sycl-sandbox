#include "sandbox_api.h"
#include "kernel.h"
#include "rt/math.h"
#include "rt/types.h"
#include "rt/camera.h"
#include "rt/trace.h"
#include "rt/params.h"
#include "rt/hittables/sphere.h"
#include "rt/materials/lambertian.h"
#include "rt/materials/metal.h"
#include "rt/materials/dielectric.h"
#include <cstdlib>
#include <cstring>

using namespace rt;
using rt::hittables::make_sphere;
using rt::materials::make_lambertian;
using rt::materials::make_metal;
using rt::materials::make_dielectric;

static ParamMeta params_meta[] = {
    {"spp_frame","Samples per frame",ParamType::INT,.range={.i={1,64,1}},.default_i=1},
    {"max_bounces","Max ray depth",ParamType::INT,.range={.i={1,100,1}},.default_i=10},
    {"cam_eye","Camera pos",ParamType::VEC3,.default_c3={13,2,3}},
    {"cam_at","Look-at",ParamType::VEC3,.default_c3={0,0,0}},
    {"cam_fov","FOV",ParamType::FLOAT,.range={.f={1,120,1}},.default_f=20},
    {"cam_aperture","Aperture",ParamType::FLOAT,.range={.f={0,1,0.01f}},.default_f=0.1f},
    {"cam_up","Up",ParamType::VEC3,.default_c3={0,1,0}},
    {"num_spheres","Num spheres",ParamType::INT,.range={.i={0,500,1}},.default_i=11},
    {"ground_color","Ground albedo",ParamType::COLOR_RGB,.default_c3={0.5f,0.5f,0.5f}},
    {"background","Sky",ParamType::COLOR_RGB,.default_c3={0.5f,0.7f,1}},
};
enum P{P_N=RT_NUM_STD_PARAMS,P_GC=P_N+1,P_BG=P_GC+3};
static KernelDesc desc={
    "one_weekend","Raytracing in One Weekend — random spheres",
    10,params_meta,0,4096,2,
    (const char*[]){"kernel.cpp","kernel.h",nullptr}
};
extern "C" KernelDesc* get_kernel_desc(){
    desc.params_buffer_size=0;
    for(int i=0;i<desc.param_count;i++)desc.params_buffer_size+=param_buffer_size(params_meta[i]);
    return &desc;
}

static Object* g_objs=nullptr;
static int g_n=0;
static float3 g_bg={0.5f,0.7f,1};

static float rnd(){return(float)rand()*(1.f/2147483647.f);}

extern "C" void init_kernel(sycl::queue* q,int,int,const void* params,size_t){
    auto* p=(const float*)params;
    int n=(int)p[P_N];
    float3 gc; memcpy(&gc,p+P_GC,12); memcpy(&g_bg,p+P_BG,12);
    if(g_objs){sycl::free(g_objs,*q);g_objs=nullptr;}
    srand(42);
    auto* o=new Object[n+4+3];
    int c=0;

    o[c++]={make_sphere({0,-1000,0},1000), make_lambertian(gc)};

    for(int k=0;k<n;){
        float x=-10+20*rnd(),z=-10+20*rnd();
        float3 cen={x,0.2f,z};
        if(len(cen)<=0.9f)continue;
        float3 col={rnd()*rnd(),rnd()*rnd(),rnd()*rnd()};
        float d=rnd();
        if(d<0.6f)      o[c++]={make_sphere(cen,0.2f), make_lambertian(col)};
        else if(d<0.85f) o[c++]={make_sphere(cen,0.2f), make_metal(col,0.5f*rnd())};
        else             o[c++]={make_sphere(cen,0.2f), make_dielectric(1.5f)};
        k++;
    }
    o[c++]={make_sphere({4,1,0},1), make_metal({0.7f,0.6f,0.5f},0)};
    o[c++]={make_sphere({-4,1,0},1), make_lambertian({0.4f,0.2f,0.1f})};
    o[c++]={make_sphere({0,1,0},1), make_dielectric(1.5f)};

    g_objs=sycl::malloc_device<Object>(c,*q);
    q->memcpy(g_objs,o,c*sizeof(Object)).wait();
    g_n=c; delete[] o;
}

extern "C" void render_kernel(sycl::queue* q,int w,int h,const void* p,void* accum,int si){
    render_main(q,w,h,(const float*)p,(float*)accum,si,g_objs,g_n,
        [bg=g_bg](const Ray& ray)->float3{
            return lerp({1,1,1},bg,0.5f*(ray.dir.y+1));
        });
}

extern "C" void shutdown_kernel(sycl::queue* q){
    if(g_objs){sycl::free(g_objs,*q);g_objs=nullptr;}
}
