#include "sandbox_api.h"
#include "rt/math.h"
#include "rt/types.h"
#include "rt/camera.h"
#include "rt/trace.h"
#include "rt/params.h"
#include <cstring>

static ParamMeta params_meta[] = {
    { "spp_frame",    "Samples per frame",
      ParamType::INT,  .range = { .i = { 1, 64, 1 } },  .default_i = 1 },
    { "max_bounces",  "Maximum ray path depth",
      ParamType::INT,  .range = { .i = { 1, 100, 1 } }, .default_i = 10 },
    { "cam_eye",      "Camera position",
      ParamType::VEC3, .default_c3 = { 0.f, 1.5f, 4.5f } },
    { "cam_at",       "Camera look-at target",
      ParamType::VEC3, .default_c3 = { 0.f, 1.2f, 0.f } },
    { "cam_fov",      "Vertical field of view (degrees)",
      ParamType::FLOAT, .range = { .f = { 1.f, 120.f, 1.f } }, .default_f = 35.f },
    { "cam_aperture","Depth of field aperture size",
      ParamType::FLOAT, .range = { .f = { 0.f, 1.f, 0.01f } }, .default_f = 0.f },
    { "cam_up",       "Camera up vector",
      ParamType::VEC3, .default_c3 = { 0.f, 1.f, 0.f } },
    { "light_color",  "Ceiling light color",
      ParamType::COLOR_RGB, .default_c3 = { 1.f, 1.f, 1.f } },
    { "light_strength","Ceiling light intensity multiplier",
      ParamType::FLOAT, .range = { .f = { 1.f, 50.f, 1.f } }, .default_f = 15.f },
};

enum ParamIdx : int {
    P_LIGHT_COLOR = RT_NUM_STD_PARAMS,
    P_LIGHT_STRENGTH = P_LIGHT_COLOR + 3,
};

static const char* source_files[] = { "kernel.cpp", nullptr };
static KernelDesc desc = {
    "cornell_box", "Cornell box scene with quads",
    9, params_meta, 0, 4096, 1, source_files
};

extern "C" KernelDesc* get_kernel_desc() {
    desc.params_buffer_size = 0;
    for (int i = 0; i < desc.param_count; i++)
        desc.params_buffer_size += param_buffer_size(params_meta[i]);
    return &desc;
}

static rt::Object* g_d_objects = nullptr;
static int         g_num_objects = 0;

// Build a quad as two triangles (two objects per quad)
static void add_quad(rt::Object* objs, int& idx, rt::float3 a, rt::float3 b, rt::float3 c,
                     rt::float3 d, rt::Material* mat) {
    // Quad as two triangles: (a,b,c) and (a,c,d)
    // For axis-aligned quads, the normal is computed by Quad constructor
    objs[idx++] = {new rt::Quad(a,b,c), mat};
    objs[idx++] = {new rt::Quad(a,c,d), mat};
}

extern "C" void init_kernel(sycl::queue* q, int, int,
                             const void* params, size_t) {
    auto* p = (const float*)params;
    rt::float3 lc; memcpy(&lc, p + P_LIGHT_COLOR, 12);
    float ls = p[P_LIGHT_STRENGTH];

    if (g_d_objects) { sycl::free(g_d_objects, *q); g_d_objects = nullptr; }

    rt::float3 w = {0.73f,0.73f,0.73f}, r = {0.65f,0.05f,0.05f};
    rt::float3 g = {0.12f,0.45f,0.15f}, z = {0,0,0};
    rt::float3 le = rt::scale(lc, ls);

    auto* objs = new rt::Object[64];
    int i = 0;

    // Helper: axis-aligned quad as two triangles
    auto quad = [&](int ax, float a_val, float b0, float b1, float c0, float c1,
                     rt::Material* mat) {
        rt::float3 p[4];
        int bi = (ax+1)%3, ci = (ax+2)%3;
        float bb[2]={b0,b1}, cc[2]={c0,c1};
        for (int k=0;k<4;k++) {
            float* v=(float*)&p[k];
            v[ax]=a_val; v[bi]=bb[k&1]; v[ci]=cc[k>>1];
        }
        add_quad(objs, i, p[0], p[1], p[2], p[3], mat);
    };

    // Room
    auto* white = new rt::Lambertian(w);
    quad(1,0,   -2,2,-2,2, white);
    quad(1,3,   -2,2,-2,2, white);
    quad(2,-2,  -2,2,0,3, white);
    quad(0,-2,  -2,2,0,3, new rt::Lambertian(r));
    quad(0,2,   -2,2,0,3, new rt::Lambertian(g));

    // Light
    quad(1,2.99f, -1,1,-1,1, new rt::DiffuseLight(le));

    // Boxes
    auto box = [&](float cx, float cy, float cz, float bw, float bh, float bd, rt::Material* m) {
        quad(1,cy+bh, cx,cx+bw, cz,cz+bd, m);
        quad(1,cy,   cx,cx+bw, cz,cz+bd, m);
        quad(2,cz,   cx,cx+bw, cy,cy+bh, m);
        quad(2,cz+bd,cx,cx+bw, cy,cy+bh, m);
        quad(0,cx,   cz,cz+bd, cy,cy+bh, m);
        quad(0,cx+bw,cz,cz+bd, cy,cy+bh, m);
    };
    box(-0.8f,0,-0.8f, 0.6f,1.5f,0.6f, new rt::Lambertian({0.55f,0.55f,0.55f}));
    box(0.8f,0,-0.3f, 0.6f,0.6f,1.2f, new rt::Lambertian({0.55f,0.55f,0.55f}));

    int cnt = i;
    g_d_objects = sycl::malloc_device<rt::Object>(cnt, *q);
    q->memcpy(g_d_objects, objs, cnt * sizeof(rt::Object)).wait();
    g_num_objects = cnt;
    delete[] objs;
}

extern "C" void render_kernel(sycl::queue* q, int w, int h,
                               const void* params, void* accum, int si) {
    auto* p = (const float*)params;
    rt::render_main(q, w, h, p, (float*)accum, si,
                    g_d_objects, g_num_objects,
                    [](const rt::Ray&) -> rt::float3 { return {0,0,0}; });
}

extern "C" void shutdown_kernel(sycl::queue* q) {
    if (g_d_objects) { sycl::free(g_d_objects, *q); g_d_objects = nullptr; }
}
