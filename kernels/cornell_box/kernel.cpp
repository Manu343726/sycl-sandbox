#include "sandbox_api.h"
#include "rt/math.h"
#include "rt/types.h"
#include "rt/camera.h"
#include "rt/trace.h"
#include <cstring>

static ParamMeta params_meta[] = {
    { "spp_frame",    "Samples per frame",
      ParamType::INT,  .range = { .i = { 1, 64, 1 } },  .default_i = 1 },
    { "max_bounces",  "Maximum ray path depth",
      ParamType::INT,  .range = { .i = { 1, 100, 1 } }, .default_i = 10 },
    { "light_color",  "Ceiling light color",
      ParamType::COLOR_RGB, .default_c3 = { 1.f, 1.f, 1.f } },
    { "light_strength","Ceiling light intensity multiplier",
      ParamType::FLOAT, .range = { .f = { 1.f, 50.f, 1.f } }, .default_f = 15.f },
};

enum ParamIdx : int {
    P_SPP_FRAME=0, P_MAX_BOUNCES=1,
    P_LIGHT_COLOR=2, P_LIGHT_STRENGTH=5
};

static const char* source_files[] = { "kernel.cpp", nullptr };
static KernelDesc desc = {
    "cornell_box", "Cornell box scene with quads",
    4, params_meta, 0, 4096, 1, source_files
};

extern "C" KernelDesc* get_kernel_desc() {
    desc.params_buffer_size = 0;
    for (int i = 0; i < desc.param_count; i++)
        desc.params_buffer_size += param_buffer_size(params_meta[i]);
    return &desc;
}

// ── scene builder ──────────────────────────────────────────────────────
static void add_quad(rt::Quad* out, int& i, int ax, float a_val,
                     float b_lo, float b_hi, float c_lo, float c_hi,
                     rt::MatType mat, rt::float3 albedo, rt::float3 emit) {
    rt::float3 p[4];
    int bi = (ax + 1) % 3, ci = (ax + 2) % 3;
    float bb[2] = {b_lo, b_hi}, cc[2] = {c_lo, c_hi};
    for (int k = 0; k < 4; k++) {
        float* v = (float*)&p[k];
        v[ax] = a_val; v[bi] = bb[k & 1]; v[ci] = cc[k >> 1];
    }
    rt::float3 n = rt::norm(rt::cross(rt::sub(p[1], p[0]), rt::sub(p[2], p[0])));
    out[i] = {p[0], p[1], p[2], n, mat, albedo, emit};
    i++;
}

static int build_scene(rt::Quad* out, rt::float3 light_col, float light_strength) {
    int i = 0;
    rt::float3 w = {0.73f,0.73f,0.73f}, r = {0.65f,0.05f,0.05f};
    rt::float3 g = {0.12f,0.45f,0.15f}, z = {0,0,0};
    rt::float3 le = rt::scale(light_col, light_strength);

    add_quad(out,i,1,0.f,   -2,2,-2,2, rt::MatType::LAMBERTIAN, w,z);
    add_quad(out,i,1,3.f,   -2,2,-2,2, rt::MatType::LAMBERTIAN, w,z);
    add_quad(out,i,2,-2.f,  -2,2,0,3,  rt::MatType::LAMBERTIAN, w,z);
    add_quad(out,i,0,-2.f,  -2,2,0,3,  rt::MatType::LAMBERTIAN, r,z);
    add_quad(out,i,0,2.f,   -2,2,0,3,  rt::MatType::LAMBERTIAN, g,z);
    add_quad(out,i,1,2.99f, -1,1,-1,1, rt::MatType::DIFFUSE_LIGHT, w,le);

    float bx=-0.8f,by=0.f,bz=-0.8f,bw=0.6f,bh=1.5f,bd=0.6f;
    add_quad(out,i,1,by+bh, bx,bx+bw,bz,bz+bd, rt::MatType::LAMBERTIAN, w,z);
    add_quad(out,i,1,by,    bx,bx+bw,bz,bz+bd, rt::MatType::LAMBERTIAN, w,z);
    add_quad(out,i,2,bz,    bx,bx+bw,by,by+bh, rt::MatType::LAMBERTIAN, w,z);
    add_quad(out,i,2,bz+bd, bx,bx+bw,by,by+bh, rt::MatType::LAMBERTIAN, w,z);
    add_quad(out,i,0,bx,    bz,bz+bd,by,by+bh, rt::MatType::LAMBERTIAN, w,z);
    add_quad(out,i,0,bx+bw, bz,bz+bd,by,by+bh, rt::MatType::LAMBERTIAN, w,z);

    bx=0.8f;by=0.f;bz=-0.3f;bw=0.6f;bh=0.6f;bd=1.2f;
    add_quad(out,i,1,by+bh, bx,bx+bw,bz,bz+bd, rt::MatType::LAMBERTIAN, w,z);
    add_quad(out,i,1,by,    bx,bx+bw,bz,bz+bd, rt::MatType::LAMBERTIAN, w,z);
    add_quad(out,i,2,bz,    bx,bx+bw,by,by+bh, rt::MatType::LAMBERTIAN, w,z);
    add_quad(out,i,2,bz+bd, bx,bx+bw,by,by+bh, rt::MatType::LAMBERTIAN, w,z);
    add_quad(out,i,0,bx,    bz,bz+bd,by,by+bh, rt::MatType::LAMBERTIAN, w,z);
    add_quad(out,i,0,bx+bw, bz,bz+bd,by,by+bh, rt::MatType::LAMBERTIAN, w,z);
    return i;
}

static rt::Quad* g_d_quads = nullptr;
static int       g_num_quads = 0;

extern "C" void init_kernel(sycl::queue* q, int, int,
                             const void* params, size_t) {
    auto* p = (const float*)params;
    rt::float3 lc; memcpy(&lc, p + P_LIGHT_COLOR, 12);
    float ls = p[P_LIGHT_STRENGTH];
    rt::Quad hq[32];
    int nq = build_scene(hq, lc, ls);
    if (g_d_quads) sycl::free(g_d_quads, *q);
    g_d_quads = sycl::malloc_device<rt::Quad>(nq, *q);
    q->memcpy(g_d_quads, hq, nq * sizeof(rt::Quad)).wait();
    g_num_quads = nq;
}

extern "C" void render_kernel(sycl::queue* q, int w, int h,
                               const void* params, void* accum, int si) {
    auto* p = (const float*)params;
    int spp = (int)p[P_SPP_FRAME];
    int bounces = (int)p[P_MAX_BOUNCES];
    float aspect = (float)w / (float)h;

    rt::Camera cam = rt::lookat({0,1.5f,4.5f}, {0,1.2f,0}, {0,1,0}, 35.f, aspect);

    rt::render(q, w, h, cam, g_d_quads, g_num_quads, spp, bounces,
               (float*)accum, si, [](const rt::Quad& qq, const rt::Ray& rr, float mn, float mx, rt::HitRecord& rec) {
                   return rt::hit_quad(qq, rr, mn, mx, rec);
               });
}

extern "C" void shutdown_kernel(sycl::queue* q) {
    if (g_d_quads) { sycl::free(g_d_quads, *q); g_d_quads = nullptr; }
}
