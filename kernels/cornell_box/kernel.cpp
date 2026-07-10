#include "sandbox_api.h"
#include "rt/math.h"
#include "rt/types.h"
#include "rt/camera.h"
#include "rt/trace.h"
#include "rt/params.h"
#include "rt/hittables/quad.h"
#include "rt/materials/lambertian.h"
#include "rt/materials/diffuse_light.h"
#include <cstring>

using namespace rt;
using rt::hittables::quad;
using rt::materials::lambertian;
using rt::materials::diffuse_light;

static ParamMeta params_meta[]={
    {"spp_frame","Samples per frame",ParamType::INT,.range={.i={1,64,1}},.default_i=1},
    {"max_bounces","Max ray depth",ParamType::INT,.range={.i={1,100,1}},.default_i=10},
    {"cam_eye","Camera pos",ParamType::VEC3,.default_c3={0,1.5f,4.5f}},
    {"cam_at","Look-at",ParamType::VEC3,.default_c3={0,1.2f,0}},
    {"cam_fov","FOV",ParamType::FLOAT,.range={.f={1,120,1}},.default_f=35},
    {"cam_aperture","Aperture",ParamType::FLOAT,.range={.f={0,1,0.01f}},.default_f=0},
    {"cam_up","Up",ParamType::VEC3,.default_c3={0,1,0}},
    {"light_color","Light color",ParamType::COLOR_RGB,.default_c3={1,1,1}},
    {"light_strength","Light intensity",ParamType::FLOAT,.range={.f={1,50,1}},.default_f=15},
};
enum P{P_LC=RT_NUM_STD_PARAMS,P_LS=P_LC+3};
static KernelDesc desc={
    "cornell_box","Cornell box with quads",9,params_meta,0,4096,1,
    (const char*[]){"kernel.cpp",nullptr}
};
extern "C" KernelDesc* get_kernel_desc(){
    desc.params_buffer_size=0;
    for(int i=0;i<desc.param_count;i++)desc.params_buffer_size+=param_buffer_size(params_meta[i]);
    return &desc;
}

static Object* g_objs=nullptr;
static int g_n=0;

static float3 p4(int ax,float av,float b0,float b1,float c0,float c1,int k){
    int bi=(ax+1)%3,ci=(ax+2)%3;
    float bb[2]={b0,b1},cc[2]={c0,c1},v[3]={0,0,0};
    v[ax]=av; v[bi]=bb[k&1]; v[ci]=cc[k>>1];
    return {v[0],v[1],v[2]};
}

extern "C" void init_kernel(sycl::queue* q,int,int,const void* params,size_t){
    auto* p=(const float*)params;
    float3 lc;memcpy(&lc,p+P_LC,12);
    float ls=p[P_LS];
    if(g_objs){sycl::free(g_objs,*q);g_objs=nullptr;}

    auto* o=new Object[64]; int i=0;
    float3 w={0.73f,0.73f,0.73f},r={0.65f,0.05f,0.05f},g={0.12f,0.45f,0.15f},z={0,0,0};
    float3 le=scale(lc,ls);

    auto add=[&](int ax,float av,float b0,float b1,float c0,float c1,Material mat){
        auto a=p4(ax,av,b0,b1,c0,c1,0),b=p4(ax,av,b0,b1,c0,c1,1);
        auto c=p4(ax,av,b0,b1,c0,c1,2),d=p4(ax,av,b0,b1,c0,c1,3);
        o[i++]={quad(a,b,c), mat};
        o[i++]={quad(a,c,d), mat};
    };

    add(1,0,   -2,2,-2,2, lambertian(w));
    add(1,3,   -2,2,-2,2, lambertian(w));
    add(2,-2,  -2,2,0,3,  lambertian(w));
    add(0,-2,  -2,2,0,3,  lambertian(r));
    add(0,2,   -2,2,0,3,  lambertian(g));
    add(1,2.99f,-1,1,-1,1, diffuse_light(le));

    auto box=[&](float bx,float by,float bz,float bw,float bh,float bd,Material m){
        add(1,by+bh, bx,bx+bw,bz,bz+bd, m); add(1,by, bx,bx+bw,bz,bz+bd, m);
        add(2,bz,   bx,bx+bw,by,by+bh, m); add(2,bz+bd,bx,bx+bw,by,by+bh, m);
        add(0,bx,   bz,bz+bd,by,by+bh, m); add(0,bx+bw,bz,bz+bd,by,by+bh, m);
    };
    box(-0.8f,0,-0.8f, 0.6f,1.5f,0.6f, lambertian({0.55f,0.55f,0.55f}));
    box(0.8f,0,-0.3f, 0.6f,0.6f,1.2f, lambertian({0.55f,0.55f,0.55f}));

    g_objs=sycl::malloc_device<Object>(i,*q);
    q->memcpy(g_objs,o,i*sizeof(Object)).wait();
    g_n=i; delete[] o;
}

extern "C" void render_kernel(sycl::queue* q,int w,int h,const void* p,void* accum,int si){
    render_main(q,w,h,(const float*)p,(float*)accum,si,g_objs,g_n,
        [](const Ray&)->float3{return{0,0,0};});
}

extern "C" void shutdown_kernel(sycl::queue* q){
    if(g_objs){sycl::free(g_objs,*q);g_objs=nullptr;}
}
