#include "sandbox_api.h"
#include "kernel.h"
#include <cmath>
#include <cstdlib>
#include <cstring>

// ── parameter metadata ─────────────────────────────────────────────────
static ParamMeta params_meta[] = {
    { "num_spheres",  "Number of random small spheres",
      ParamType::INT,  .range = { .i = { 0, 500, 1 } }, .default_i = 11 },
    { "max_bounces",  "Maximum ray path depth",
      ParamType::INT,  .range = { .i = { 1, 100, 1 } }, .default_i = 10 },
    { "spp_frame",    "Samples per frame",
      ParamType::INT,  .range = { .i = { 1, 64, 1 } },  .default_i = 1 },
    { "ground_color", "Ground sphere albedo",
      ParamType::COLOR_RGB, .default_c3 = { 0.5f, 0.5f, 0.5f } },
    { "background",   "Sky colour",
      ParamType::COLOR_RGB, .default_c3 = { 0.5f, 0.7f, 1.0f } },
};

// float-index offsets into param buffer (matches param_buffer_size)
enum ParamIdx : int { P_NUM_SPHERES=0, P_MAX_BOUNCES=1, P_SPP_FRAME=2,
                      P_GROUND_COLOR=3, P_BACKGROUND=6 };

static const char* source_files[] = { "kernel.cpp", "kernel.h", nullptr };
static KernelDesc desc = {
    "one_weekend", "Raytracing in One Weekend — random spheres",
    5, params_meta, 0, 2, source_files
};

extern "C" KernelDesc* get_kernel_desc() {
    desc.params_buffer_size = 0;
    for (int i = 0; i < desc.param_count; i++)
        desc.params_buffer_size += param_buffer_size(params_meta[i]);
    return &desc;
}

// ── per-instance device state (persistent across frames) ───────────────
// Stored as USM so the kernel lambda can capture device pointers.
static Sphere*     g_d_spheres = nullptr;
static CameraData* g_d_cam     = nullptr;
static int         g_num_spheres = 0;

// ── host scene builder ─────────────────────────────────────────────────
static Sphere* build_random_spheres(int n, float3 gc) {
    auto* s = new Sphere[n + 4];
    int i = 0;
    s[i++] = { {0,-1000,0}, 1000, MatType::LAMBERTIAN, gc, 0,0,{0,0,0} };
    RNG rng{42};
    for (int k = 0; k < n; k++) {
        float x = -10 + 20*rng.next(), z = -10 + 20*rng.next();
        float3 c = {x,0.2f,z};
        if (vlen(c) <= 0.9f) continue;
        float3 col = { rng.next()*rng.next(), rng.next()*rng.next(),
                       rng.next()*rng.next() };
        float d = rng.next();
        if      (d < 0.6f)  s[i++] = { c,0.2f, MatType::LAMBERTIAN, col,0,0,{0,0,0} };
        else if (d < 0.85f) s[i++] = { c,0.2f, MatType::METAL, col, 0.5f*rng.next(),0,{0,0,0} };
        else                s[i++] = { c,0.2f, MatType::DIELECTRIC, {1,1,1},0,1.5f,{0,0,0} };
    }
    s[i++] = { {4,1,0}, 1, MatType::METAL,      {0.7f,0.6f,0.5f},0,0,{0,0,0} };
    s[i++] = { {-4,1,0},1, MatType::LAMBERTIAN,  {0.4f,0.2f,0.1f},0,0,{0,0,0} };
    s[i++] = { {0,1,0}, 1, MatType::DIELECTRIC,  {1,1,1},0,1.5f,{0,0,0} };
    return s;
}

static CameraData setup_camera(float aspect) {
    float3 from = {13,2,3};
    float theta = 20.f * 3.14159265f / 180.f;
    float h = tanf(theta/2), vh = 2*h, vw = aspect*vh, fd = 10.f, ap = 0.1f;
    float3 w = vnorm(vsub(from, {0,0,0})), u = vnorm(vcross({0,1,0}, w));
    float3 v = vcross(w, u);
    CameraData c;
    c.origin     = from;
    c.horizontal = vscale(u, vw*fd);
    c.vertical   = vscale(v, vh*fd);
    c.lower_left = vsub(vsub(vsub(c.origin, vscale(c.horizontal,.5f)),
                             vscale(c.vertical,.5f)), vscale(w, fd));
    c.u = u; c.v = v; c.w = w; c.lens_radius = ap/2;
    return c;
}

// ── device hit / scatter / trace ───────────────────────────────────────
SYCL_EXTERNAL inline bool hit(const Sphere& s, const Ray& r,
                               float mn, float mx, HitRecord& rec) {
    float3 oc = vsub(r.orig, s.center);
    float a = vdot(r.dir,r.dir), b = vdot(oc,r.dir);
    float c = vdot(oc,oc) - s.radius*s.radius;
    float d = b*b - a*c;
    if (d <= 0) return false;
    float t = (-b - sycl::sqrt(d))/a;
    if (t < mn || t > mx) t = (-b + sycl::sqrt(d))/a;
    if (t < mn || t > mx) return false;
    rec.t = t; rec.p = vadd(r.orig, vscale(r.dir,t));
    rec.normal = vscale(vsub(rec.p,s.center), 1/s.radius);
    rec.front_face = vdot(r.dir, rec.normal) < 0;
    if (!rec.front_face) rec.normal = vscale(rec.normal, -1);
    rec.mat_type = s.mat_type; rec.albedo = s.albedo;
    rec.fuzz = s.fuzz; rec.ir = s.ir; rec.emit = s.emit;
    return true;
}

SYCL_EXTERNAL inline float3 rnd_dir(RNG& r) {
    for (int i=0;i<100;i++) { float3 p = {2*r.next()-1,2*r.next()-1,2*r.next()-1}; if (vlen2(p)<1) return p; }
    return {0,0,1};
}
SYCL_EXTERNAL inline float3 reflect(float3 v, float3 n) { return vsub(v, vscale(n, 2*vdot(v,n))); }
SYCL_EXTERNAL inline bool refract(float3 v, float3 n, float eta, float3& o) {
    float3 uv = vnorm(v); float dt = vdot(uv,n), d = 1-eta*eta*(1-dt*dt);
    if (d<=0) return false;
    o = vscale(vsub(uv, vscale(n,dt)), eta); o = vsub(o, vscale(n, sycl::sqrt(d)));
    return true;
}
SYCL_EXTERNAL inline float schlick(float c, float ir) {
    float r0 = (1-ir)/(1+ir); r0*=r0; return r0+(1-r0)*sycl::pow(1-c,5);
}

SYCL_EXTERNAL float3 trace(const Ray& r, const Sphere* spheres, int n,
                            float3 bg, int bounces, RNG& rng) {
    float3 att = {1,1,1};
    Ray ray = r;
    for (int b=0; b<bounces; b++) {
        HitRecord rec; float closest = 1e30f; bool hit_any = false;
        for (int i=0; i<n; i++) if (hit(spheres[i], ray, 0.001f, closest, rec))
            { closest = rec.t; hit_any = true; }
        if (!hit_any) {
            float t = 0.5f*(ray.dir.y()+1);
            return vmul(att, vlerp({1,1,1}, bg, t));
        }
        switch (rec.mat_type) {
        case MatType::LAMBERTIAN: {
            float3 tg = vadd(rec.p, vadd(rec.normal, rnd_dir(rng)));
            ray = {rec.p, vsub(tg,rec.p)}; att = vmul(att, rec.albedo); break;
        }
        case MatType::METAL: {
            float3 rfl = reflect(vnorm(ray.dir), rec.normal);
            float3 f = vadd(rfl, vscale(rnd_dir(rng), rec.fuzz));
            if (vdot(f,rec.normal)<=0) return {0,0,0};
            ray = {rec.p,f}; att = vmul(att, rec.albedo); break;
        }
        case MatType::DIELECTRIC: {
            float3 out; float eta, cos;
            if (vdot(ray.dir, rec.normal) > 0) {
                out = vscale(rec.normal,-1); eta = rec.ir;
                cos = rec.ir*vdot(ray.dir,rec.normal)/vlen(ray.dir);
            } else {
                out = rec.normal; eta = 1/rec.ir;
                cos = -vdot(ray.dir,rec.normal)/vlen(ray.dir);
            }
            float3 refr; float rp;
            if (refract(ray.dir, out, eta, refr)) rp = schlick(cos, rec.ir);
            else rp = 1;
            ray = {rec.p, rng.next()<rp ? reflect(vnorm(ray.dir),rec.normal) : refr};
            att = vmul(att, rec.albedo); break;
        }
        case MatType::DIFFUSE_LIGHT: return vmul(att, rec.emit);
        }
    }
    return {0,0,0};
}

// ── init (host) ────────────────────────────────────────────────────────
extern "C" void init_kernel(sycl::queue* q, int w, int h,
                             const void* params, size_t) {
    auto* p = (const float*)params;
    int n = (int)p[P_NUM_SPHERES];
    float3 gc; memcpy(&gc, p + P_GROUND_COLOR, 12);

    int cnt = n + 4;
    auto* hs = build_random_spheres(n, gc);
    if (g_d_spheres) sycl::free(g_d_spheres, *q);
    g_d_spheres = sycl::malloc_device<Sphere>(cnt, *q);
    q->memcpy(g_d_spheres, hs, cnt*sizeof(Sphere)).wait();
    g_num_spheres = cnt;
    delete[] hs;

    auto cam = setup_camera((float)w/h);
    if (g_d_cam) sycl::free(g_d_cam, *q);
    g_d_cam = sycl::malloc_device<CameraData>(1, *q);
    q->memcpy(g_d_cam, &cam, sizeof(CameraData)).wait();
}

// ── render (host, called each frame) ──────────────────────────────────
extern "C" void render_kernel(sycl::queue* q, int w, int h,
                               const void* params, void* accum, int si) {
    auto* p = (const float*)params;
    int spp = (int)p[P_SPP_FRAME];
    int bounces = (int)p[P_MAX_BOUNCES];
    float3 bg; memcpy(&bg, p + P_BACKGROUND, 12);

    auto* acc = (float*)accum;
    auto* d_sph = g_d_spheres;
    auto* d_cam = g_d_cam;
    int   nsph = g_num_spheres;

    q->parallel_for(sycl::range<2>{(size_t)h, (size_t)w},
                    [=](sycl::item<2> it) {
        int x = it[1], y = it[0], idx = y * w + x;

        for (int s = 0; s < spp; s++) {
            RNG rng{ (uint32_t)(idx * 6364136223846793005ull
                                + (uint64_t)(si * 2654435761u)) };
            float u = (x + rng.next()) / (float)w;
            float v_ = (y + rng.next()) / (float)h;
            float3 rd = vscale(rnd_dir(rng), d_cam->lens_radius);
            float3 off = vadd(vscale(d_cam->u, rd.x()), vscale(d_cam->v, rd.y()));
            Ray ray;
            ray.orig = vadd(d_cam->origin, off);
            ray.dir  = vsub(vadd(vadd(d_cam->lower_left,
                                      vscale(d_cam->horizontal, u)),
                                 vscale(d_cam->vertical, v_)),
                           vadd(d_cam->origin, off));

            float3 col = trace(ray, d_sph, nsph, bg, bounces, rng);
            int base = idx * 4;
            acc[base+0] += col.x();
            acc[base+1] += col.y();
            acc[base+2] += col.z();
            acc[base+3] += 1;
        }
    }).wait();
}

extern "C" void shutdown_kernel(sycl::queue* q) {
    if (g_d_spheres) { sycl::free(g_d_spheres, *q); g_d_spheres = nullptr; }
    if (g_d_cam)     { sycl::free(g_d_cam, *q);     g_d_cam = nullptr; }
}
