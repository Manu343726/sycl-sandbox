#include "sandbox_api.h"
#include "rt/math.h"
#include "rt/types.h"
#include "rt/camera.h"
#include "rt/trace.h"
#include "rt/params.h"
#include <cstring>

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

static rt::Object* g_objs=nullptr;
static int g_n=0;

static void add_quad(rt::Object* o,int& i,int ax,float av,float b0,float b1,float c0,float c1,
                     rt::Material mat){
    rt::float3 p[4];
    int bi=(ax+1)%3,ci=(ax+2)%3;
    float bb[2]={b0,b1},cc[2]={c0,c1};
    for(int k=0;k<4;k++){float*v=(float*)&p[k];v[ax]=av;v[bi]=bb[k&1];v[ci]=cc[k>>1];}
    o[i++]=rt::Object::make_quad(p[0],p[1],p[2],std::move(mat));
}

extern "C" void init_kernel(sycl::queue* q,int,int,const void* params,size_t){
    auto* p=(const float*)params;
    rt::float3 lc;memcpy(&lc,p+P_LC,12);
    float ls=p[P_LS];
    if(g_objs){sycl::free(g_objs,*q);g_objs=nullptr;}

    auto* o=new rt::Object[64]; int i=0;
    rt::float3 w={0.73f,0.73f,0.73f},r={0.65f,0.05f,0.05f},g={0.12f,0.45f,0.15f},z={0,0,0};
    rt::float3 le=rt::scale(lc,ls);

    add_quad(o,i,1,0,   -2,2,-2,2, rt::Lambertian(w));
    add_quad(o,i,1,3,   -2,2,-2,2, rt::Lambertian(w));
    add_quad(o,i,2,-2,  -2,2,0,3,  rt::Lambertian(w));
    add_quad(o,i,0,-2,  -2,2,0,3,  rt::Lambertian(r));
    add_quad(o,i,0,2,   -2,2,0,3,  rt::Lambertian(g));
    add_quad(o,i,1,2.99f,-1,1,-1,1, rt::DiffuseLight(le));

    auto box=[&](float bx,float by,float bz,float bw,float bh,float bd,rt::Material m){
        add_quad(o,i,1,by+bh, bx,bx+bw,bz,bz+bd, m);
        add_quad(o,i,1,by,    bx,bx+bw,bz,bz+bd, m);
        add_quad(o,i,2,bz,    bx,bx+bw,by,by+bh, m);
        add_quad(o,i,2,bz+bd, bx,bx+bw,by,by+bh, m);
        add_quad(o,i,0,bx,    bz,bz+bd,by,by+bh, m);
        add_quad(o,i,0,bx+bw, bz,bz+bd,by,by+bh, m);
    };
    box(-0.8f,0,-0.8f, 0.6f,1.5f,0.6f, rt::Lambertian({0.55f,0.55f,0.55f}));
    box(0.8f,0,-0.3f, 0.6f,0.6f,1.2f, rt::Lambertian({0.55f,0.55f,0.55f}));

    g_objs=sycl::malloc_device<rt::Object>(i,*q);
    q->memcpy(g_objs,o,i*sizeof(rt::Object)).wait();
    g_n=i; delete[] o;
}

extern "C" void render_kernel(sycl::queue* q,int w,int h,const void* p,void* accum,int si){
    rt::render_main(q,w,h,(const float*)p,(float*)accum,si,g_objs,g_n,
        [](const rt::Ray&)->rt::float3{return{0,0,0};});
}

extern "C" void shutdown_kernel(sycl::queue* q){
    if(g_objs){sycl::free(g_objs,*q);g_objs=nullptr;}
}
