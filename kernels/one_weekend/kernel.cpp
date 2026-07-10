#include "sandbox_api.h"
#include "kernel.h"
#include "rt/math.h"
#include "rt/types.h"
#include "rt/camera.h"
#include "rt/trace.h"
#include "rt/params.h"
#include <cmath>
#include <cstdlib>
#include <cstring>

static ParamMeta params_meta[] = {
    { "spp_frame",    "Samples per frame",
      ParamType::INT,  .range = { .i = { 1, 64, 1 } },  .default_i = 1 },
    { "max_bounces",  "Maximum ray path depth",
      ParamType::INT,  .range = { .i = { 1, 100, 1 } }, .default_i = 10 },
    { "cam_eye",      "Camera position",
      ParamType::VEC3, .default_c3 = { 13.f, 2.f, 3.f } },
    { "cam_at",       "Camera look-at target",
      ParamType::VEC3, .default_c3 = { 0.f, 0.f, 0.f } },
    { "cam_fov",      "Vertical field of view (degrees)",
      ParamType::FLOAT, .range = { .f = { 1.f, 120.f, 1.f } }, .default_f = 20.f },
    { "cam_aperture","Depth of field aperture size",
      ParamType::FLOAT, .range = { .f = { 0.f, 1.f, 0.01f } }, .default_f = 0.1f },
    { "cam_up",       "Camera up vector",
      ParamType::VEC3, .default_c3 = { 0.f, 1.f, 0.f } },
    { "num_spheres",  "Number of random small spheres",
      ParamType::INT,  .range = { .i = { 0, 500, 1 } }, .default_i = 11 },
    { "ground_color", "Ground sphere albedo",
      ParamType::COLOR_RGB, .default_c3 = { 0.5f, 0.5f, 0.5f } },
    { "background",   "Sky colour",
      ParamType::COLOR_RGB, .default_c3 = { 0.5f, 0.7f, 1.0f } },
};

enum ParamIdx : int {
    P_NUM_SPHERES = RT_NUM_STD_PARAMS,
    P_GROUND_COLOR = P_NUM_SPHERES + 1,
    P_BACKGROUND = P_GROUND_COLOR + 3,
};

static const char* source_files[] = { "kernel.cpp", "kernel.h", nullptr };
static KernelDesc desc = {
    "one_weekend", "Raytracing in One Weekend — random spheres",
    10, params_meta, 0, 4096, 2, source_files
};

extern "C" KernelDesc* get_kernel_desc() {
    desc.params_buffer_size = 0;
    for (int i = 0; i < desc.param_count; i++)
        desc.params_buffer_size += param_buffer_size(params_meta[i]);
    return &desc;
}

static float rnd() { return (float)rand() * (1.f / 2147483647.f); }

static rt::Object* g_d_objects = nullptr;
static int         g_num_objects = 0;
static rt::float3  g_background = {0.5f, 0.7f, 1.0f};

extern "C" void init_kernel(sycl::queue* q, int, int,
                             const void* params, size_t) {
    auto* p = (const float*)params;
    int n = (int)p[P_NUM_SPHERES];
    rt::float3 gc; memcpy(&gc, p + P_GROUND_COLOR, 12);
    memcpy(&g_background, p + P_BACKGROUND, 12);

    if (g_d_objects) { sycl::free(g_d_objects, *q); g_d_objects = nullptr; }

    int max_obj = n + 4 + 3;
    auto* objs = new rt::Object[max_obj];
    int cnt = 0;
    srand(42);

    objs[cnt++] = {new rt::Sphere({0,-1000,0},1000), new rt::Lambertian(gc)};

    for (int k = 0; k < n; ) {
        float x = -10 + 20*rnd(), z = -10 + 20*rnd();
        rt::float3 c = {x,0.2f,z};
        if (rt::len(c) <= 0.9f) continue;
        float r = rnd()*rnd(), g = rnd()*rnd(), b = rnd()*rnd();
        float d = rnd();
        auto* s = new rt::Sphere(c, 0.2f);
        if (d < 0.6f)
            objs[cnt++] = {s, new rt::Lambertian({r,g,b})};
        else if (d < 0.85f)
            objs[cnt++] = {s, new rt::Metal({r,g,b}, 0.5f*rnd())};
        else
            objs[cnt++] = {s, new rt::Dielectric(1.5f)};
        k++;
    }

    objs[cnt++] = {new rt::Sphere({4,1,0},1), new rt::Metal({0.7f,0.6f,0.5f},0)};
    objs[cnt++] = {new rt::Sphere({-4,1,0},1), new rt::Lambertian({0.4f,0.2f,0.1f})};
    objs[cnt++] = {new rt::Sphere({0,1,0},1), new rt::Dielectric(1.5f)};

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
                    [bg = g_background](const rt::Ray& ray) -> rt::float3 {
                        float t = 0.5f * (ray.dir.y + 1.f);
                        return rt::lerp({1,1,1}, bg, t);
                    });
}

extern "C" void shutdown_kernel(sycl::queue* q) {
    if (g_d_objects) { sycl::free(g_d_objects, *q); g_d_objects = nullptr; }
}
