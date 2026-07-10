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

// Standard params (0-12) + kernel-specific (13+)
static ParamMeta params_meta[] = {
    // ── standard rt params (order fixed by rt_std_param) ────────────
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
    // ── kernel-specific params (index 13+) ──────────────────────────
    { "num_spheres",  "Number of random small spheres",
      ParamType::INT,  .range = { .i = { 0, 500, 1 } }, .default_i = 11 },
    { "ground_color", "Ground sphere albedo",
      ParamType::COLOR_RGB, .default_c3 = { 0.5f, 0.5f, 0.5f } },
    { "background",   "Sky colour",
      ParamType::COLOR_RGB, .default_c3 = { 0.5f, 0.7f, 1.0f } },
};

enum ParamIdx : int {
    P_NUM_SPHERES = RT_NUM_STD_PARAMS,       // 13
    P_GROUND_COLOR = P_NUM_SPHERES + 1,      // 14
    P_BACKGROUND = P_GROUND_COLOR + 1,       // 17 (COLOR_RGB is 3 floats)
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

// ── per-instance state ─────────────────────────────────────────────────
static rt::Sphere*  g_d_spheres = nullptr;
static int           g_num_spheres = 0;
static rt::float3    g_background = {0.5f, 0.7f, 1.0f};

// ── host scene builder ─────────────────────────────────────────────────
static rt::Sphere* build_random_spheres(int n, rt::float3 gc) {
    auto* s = new rt::Sphere[n + 4];
    int i = 0;
    s[i++] = {{0,-1000,0}, 1000, rt::MatType::LAMBERTIAN, gc, 0,0,{0,0,0}};
    rt::RNG rng{42};
    for (int k = 0; k < n; k++) {
        float x = -10 + 20*rng.next(), z = -10 + 20*rng.next();
        rt::float3 c = {x,0.2f,z};
        if (rt::len(c) <= 0.9f) continue;
        rt::float3 col = { rng.next()*rng.next(), rng.next()*rng.next(),
                       rng.next()*rng.next() };
        float d = rng.next();
        if      (d < 0.6f)  s[i++] = { c,0.2f, rt::MatType::LAMBERTIAN, col,0,0,{0,0,0} };
        else if (d < 0.85f) s[i++] = { c,0.2f, rt::MatType::METAL, col, 0.5f*rng.next(),0,{0,0,0} };
        else                s[i++] = { c,0.2f, rt::MatType::DIELECTRIC, {1,1,1},0,1.5f,{0,0,0} };
    }
    s[i++] = { {4,1,0}, 1, rt::MatType::METAL,      {0.7f,0.6f,0.5f},0,0,{0,0,0} };
    s[i++] = { {-4,1,0},1, rt::MatType::LAMBERTIAN,  {0.4f,0.2f,0.1f},0,0,{0,0,0} };
    s[i++] = { {0,1,0}, 1, rt::MatType::DIELECTRIC,  {1,1,1},0,1.5f,{0,0,0} };
    return s;
}

extern "C" void init_kernel(sycl::queue* q, int, int,
                             const void* params, size_t) {
    auto* p = (const float*)params;
    int n = (int)p[P_NUM_SPHERES];
    rt::float3 gc; memcpy(&gc, p + P_GROUND_COLOR, 12);
    memcpy(&g_background, p + P_BACKGROUND, 12);

    int cnt = n + 4;
    auto* hs = build_random_spheres(n, gc);
    if (g_d_spheres) sycl::free(g_d_spheres, *q);
    g_d_spheres = sycl::malloc_device<rt::Sphere>(cnt, *q);
    q->memcpy(g_d_spheres, hs, cnt*sizeof(rt::Sphere)).wait();
    g_num_spheres = cnt;
    delete[] hs;
}

extern "C" void render_kernel(sycl::queue* q, int w, int h,
                               const void* params, void* accum, int si) {
    auto* p = (const float*)params;
    rt::float3 bg = g_background;

    rt::render_main(q, w, h, p, (float*)accum, si,
                    g_d_spheres, g_num_spheres,
                    [](const rt::Sphere& sp, const rt::Ray& rr, float mn, float mx, rt::HitRecord& rec) {
                        return rt::hit_sphere(sp, rr, mn, mx, rec);
                    },
                    [bg](const rt::Ray& ray) -> rt::float3 {
                        float t = 0.5f * (ray.dir.y + 1.f);
                        return rt::lerp({1,1,1}, bg, t);
                    });
}

extern "C" void shutdown_kernel(sycl::queue* q) {
    if (g_d_spheres) { sycl::free(g_d_spheres, *q); g_d_spheres = nullptr; }
}
