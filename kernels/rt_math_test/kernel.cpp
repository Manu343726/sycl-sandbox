// RT Math Test kernel — raytracer written from scratch.
//
// Diagnostic kernel.  Uses ONLY the rt library building blocks:
//   - rt/math.h          (float3, RNG)
//   - rt/helpers.h       (random_in_unit_sphere)
//   - rt/hittables/*     (Quad, Box .hit())
//   - rt/materials/*     (Lambertian, DiffuseLight .scatter()/.emit())
//
// EVERYTHING else is written from scratch in this file:
// camera frustum from params, primary ray generation, the intersection
// loop, material dispatch, background gradient, accumulation.
//
// The scene replicates the Cornell box (same geometry/materials as
// scenes/cornell_box.yaml): 5 walls + ceiling light panel + 2 boxes.
//
// NO SceneBuilder, NO SceneView, NO render_main, NO set_scene_view,
// NO device allocations — the scene is a plain POD snapshot built on the
// host stack and captured by value into the per-pixel kernel lambda.
//
// Purpose: if this kernel renders correct colours while every YAML-loaded
// raytracer scene shows a grayish image, the bug lives in the scene
// loading/injection path (loader.cpp / host_scene.h / SceneBuilder),
// NOT in the raytracing math.

#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/helpers.h>
#include <sycl-sandbox/rt/hittables/quad.h>
#include <sycl-sandbox/rt/hittables/box.h>
#include <sycl-sandbox/rt/materials/lambertian.h>
#include <sycl-sandbox/rt/materials/diffuse_light.h>
#include <sycl-sandbox/kernel/params.h>
#include <sycl-sandbox/kernel/stats.h>
#include <sycl-sandbox/kernel/execution_context.h>

using namespace rt;

// ── Runtime (set by host to dispatch SYCL vs software) ───────────────
static rt::Runtime *g_rt = nullptr;

extern "C" void set_runtime(rt::Runtime *rt) {
    g_rt = rt;
}

// ── Param lookup (host-populated from YAML scene descriptor) ────────
static ParamLookup s_lookup;

extern "C" void set_param_lookup(const ParamLookup *lookup) {
    if (lookup) {
        s_lookup = *lookup;
    } else {
        s_lookup = ParamLookup{};
    }
}

// ── Stat writer (kernel-populated, host reads after each frame) ───────
static StatWriter s_stat_writer;

extern "C" void set_stat_lookup(const StatWriter *writer) {
    if (writer) {
        s_stat_writer = *writer;
    } else {
        s_stat_writer = StatWriter{};
    }
}

static KernelDesc desc = {
    "rt_math_test",
    "Diagnostic: hardcoded scene, renderer written from scratch (no scene framework)",
    4096,
    2,
    (const char *[]) {"kernel.cpp", nullptr}};

extern "C" KernelDesc *get_kernel_desc() {
    return &desc;
}

// ═════════════════════════════════════════════════════════════════════
//  Hardcoded scene — plain POD snapshot, built on the host stack.
//  Trivially copyable, captured by value into the device lambda.
// ═════════════════════════════════════════════════════════════════════

// Material kinds for the parallel arrays below.
enum MatKind : int {
    MAT_LAMBERTIAN = 0,
    MAT_LIGHT = 1,
};

struct TestScene {
    // ── Quads: 5 Cornell-box walls + 1 ceiling light panel ──────
    int num_quads = 6;
    hittables::Quad quads[6];
    int quad_kinds[6];                      // MatKind tag
    materials::Lambertian quad_lams[6];
    materials::DiffuseLight quad_lights[6];

    // ── Boxes: the two Cornell-box boxes ────────────────────────
    int num_boxes = 2;
    hittables::Box boxes[2];
    materials::Lambertian box_lams[2];
};

/// Build the hardcoded Cornell box on the HOST stack (cheap, a few PODs).
/// Geometry/materials mirror scenes/cornell_box.yaml exactly.
static TestScene make_test_scene() {
    TestScene s;

    auto wall = [&](int i, float3 base, float3 u, float3 v, float3 albedo) {
        s.quads[i] = hittables::quad(base, u, v);
        s.quad_kinds[i] = MAT_LAMBERTIAN;
        s.quad_lams[i] = materials::lambertian(albedo);
    };

    // Floor    (Y=0)
    wall(0, {-2.f, 0.f, -2.f}, {4.f, 0.f, 0.f}, {0.f, 0.f, 4.f}, {0.73f, 0.73f, 0.73f});
    // Ceiling  (Y=3)
    wall(1, {-2.f, 3.f, -2.f}, {4.f, 0.f, 0.f}, {0.f, 0.f, 4.f}, {0.73f, 0.73f, 0.73f});
    // Back wall (Z=-2)
    wall(2, {-2.f, 0.f, -2.f}, {4.f, 0.f, 0.f}, {0.f, 3.f, 0.f}, {0.73f, 0.73f, 0.73f});
    // Left wall (X=-2, red)
    wall(3, {-2.f, 0.f, -2.f}, {0.f, 3.f, 0.f}, {0.f, 0.f, 4.f}, {0.65f, 0.05f, 0.05f});
    // Right wall (X=2, green)
    wall(4, {2.f, 0.f, -2.f}, {0.f, 3.f, 0.f}, {0.f, 0.f, 4.f}, {0.12f, 0.45f, 0.15f});

    // Light panel on ceiling (white, intensity 15 — same as the YAML scene)
    s.quads[5] = hittables::quad({-1.f, 2.99f, -1.f}, {2.f, 0.f, 0.f}, {0.f, 0.f, 2.f});
    s.quad_kinds[5] = MAT_LIGHT;
    s.quad_lights[5] = materials::diffuse_light({15.f, 15.f, 15.f});

    // Short box (left):  min(-0.8, 0, -0.8), size (0.6, 1.5, 0.6)
    s.boxes[0] = hittables::box(-0.8f, 0.f, -0.8f, 0.6f, 1.5f, 0.6f);
    s.box_lams[0] = materials::lambertian({0.55f, 0.55f, 0.55f});
    // Tall box (right):  min(0.8, 0, -0.3), size (0.6, 0.6, 1.2)
    s.boxes[1] = hittables::box(0.8f, 0.f, -0.3f, 0.6f, 0.6f, 1.2f);
    s.box_lams[1] = materials::lambertian({0.55f, 0.55f, 0.55f});

    return s;
}

// ═════════════════════════════════════════════════════════════════════
//  Path tracing — written from scratch (same algorithm as rt/trace.h,
//  but with raw parallel arrays + a manual switch, no Handle/SceneView).
// ═════════════════════════════════════════════════════════════════════

/// Trace a ray: loop bounces, intersect the walls/boxes manually,
/// dispatch materials via the kinds[] tag, accumulate attenuation.
/// Returns black on miss (caller substitutes the background).
///
/// \param transparent_backfaces X-ray mode: when true, rays pass through
///        surfaces hit from behind (front_face == false), so a camera
///        outside the box can see inside through the walls.
inline float3 test_trace(const Ray &ray, const TestScene &scene,
                         int max_bounces, bool transparent_backfaces, RNG &rng) {
    float3 attenuation = {1, 1, 1};
    Ray current = ray;

    for ( int bounce = 0; bounce < max_bounces; bounce++ ) {
        // ── Find closest hit among the quads + boxes ──────────────
        optional<HitRecord> closest;
        int hit_obj = 0;   // 0 = none, 1 = quad, 2 = box
        int hit_idx = -1;
        float t_max = 1e30f;

        for ( int i = 0; i < scene.num_quads; i++ ) {
            auto hit = scene.quads[i].hit(current, 0.001f, t_max);
            if ( !hit ) continue;
            // X-ray mode: back faces are transparent — skip hits from
            // behind so the ray passes through the surface.
            if ( transparent_backfaces && !hit->front_face ) continue;
            closest = hit;
            hit_obj = 1;
            hit_idx = i;
            t_max = hit->t;
        }
        for ( int i = 0; i < scene.num_boxes; i++ ) {
            auto hit = scene.boxes[i].hit(current, 0.001f, t_max);
            if ( !hit ) continue;
            if ( transparent_backfaces && !hit->front_face ) continue;
            closest = hit;
            hit_obj = 2;
            hit_idx = i;
            t_max = hit->t;
        }

        if ( !closest ) {
            return {0, 0, 0};  // miss → background handled by caller
        }

        // ── Emission: the ceiling light panel ─────────────────────
        float3 emitted = {0, 0, 0};
        if ( hit_obj == 1 && scene.quad_kinds[hit_idx] == MAT_LIGHT ) {
            emitted = scene.quad_lights[hit_idx].emit(*closest);
        }
        if ( emitted.x != 0 || emitted.y != 0 || emitted.z != 0 ) {
            return mul(attenuation, emitted);
        }

        // ── Scatter: dispatch on the material kind tag ────────────
        optional<ScatterRecord> scattered;
        if ( hit_obj == 1 ) {
            if ( scene.quad_kinds[hit_idx] == MAT_LAMBERTIAN ) {
                scattered = scene.quad_lams[hit_idx].scatter(current, *closest, rng);
            }
            // MAT_LIGHT never reaches here (emitted above)
        } else if ( hit_obj == 2 ) {
            scattered = scene.box_lams[hit_idx].scatter(current, *closest, rng);
        }
        if ( !scattered ) {
            return {0, 0, 0};  // absorbed
        }

        attenuation = mul(attenuation, scattered->attenuation);
        current = scattered->scattered;
    }
    return {0, 0, 0};
}

// ═════════════════════════════════════════════════════════════════════
//  Kernel entry points
// ═════════════════════════════════════════════════════════════════════

// Nothing to construct in init — the scene lives in .so code, and the
// render loop builds it per frame on the host stack.  No device memory,
// no Runtime usage (avoids any init-time runtime dependency).

extern "C" void init_kernel(KERNEL_QUEUE_PARAM, int, int,
                            const void *, size_t) {
    PROFILER_ZONE("init_kernel");
}

// ENQUEUE-ONLY: submits the render job and returns without waiting.
extern "C" void render_kernel(KERNEL_QUEUE_PARAM, const RenderContext *ctx) {
    PROFILER_ZONE("render");
    if ( !ctx || !g_rt ) return;

    s_lookup.set_buffer(ctx->params);

    // ── Read params (host side, zero per-pixel overhead) ─────────
    int samples_per_frame = s_lookup.read<int>("spp_frame");
    int max_bounces = s_lookup.read<int>("max_bounces");
    bool transparent_backfaces = s_lookup.read<bool>("transparent_backfaces");
    float3 cam_eye = s_lookup.read_vec3<float3>("cam_eye");
    float3 cam_at = s_lookup.read_vec3<float3>("cam_at");
    float3 cam_up = s_lookup.read_vec3<float3>("cam_up");
    float fov = s_lookup.read<float>("cam_fov");
    float aperture = s_lookup.read<float>("cam_aperture");
    float frame_time = s_lookup.read<float>("time");
    float3 bg = s_lookup.read_vec3<float3>("background");

    // ── Camera frustum — from scratch ────────────────────────────
    float aspect = (float)ctx->width / (float)ctx->height;
    float theta = fov * 3.14159265f / 180.f;
    float half_h = math::tan(theta * 0.5f);
    float half_w = aspect * half_h;
    float3 w_axis = norm(sub(cam_eye, cam_at));
    float3 u_axis = norm(cross(cam_up, w_axis));
    float3 v_axis = cross(w_axis, u_axis);
    float3 lower_left = sub(sub(sub(cam_eye, scale(u_axis, half_w)),
                                scale(v_axis, half_h)),
                            w_axis);
    float3 horizontal = scale(u_axis, 2.f * half_w);
    float3 vertical = scale(v_axis, 2.f * half_h);

    // ── Scene snapshot (host stack, captured by value) ───────────
    TestScene scene = make_test_scene();

    int w = ctx->width;
    int h = ctx->height;
    float *accum = ctx->accum;
    uint32_t spp_total = ctx->spp_total;
    profiler::DeviceRing prof = ctx->prof;

    g_rt->foreach_pixel<class RtMathTestPixelKernel>(
        w, h, [=](int x, int y, int flat_index) {
            PROFILER_DEVICE_ZONE(prof, "trace_px", flat_index);
            for ( int sample = 0; sample < samples_per_frame; sample++ ) {
                // Per-pixel, per-sample RNG stream (same seeding as rt/trace.h).
                RNG rng {static_cast<uint32_t>(flat_index * 6364136223846793005ull +
                                               (uint64_t)(spp_total * 2654435761u) +
                                               sample)};

                // Stratified pixel coordinates with random offsets.
                float u = (x + rng.next()) / (float)w;
                float v = (y + rng.next()) / (float)h;

                // DOF jitter (aperture = 0 disables it).
                float3 origin = cam_eye;
                if ( aperture > 0.f ) {
                    float3 jitter = scale(random_in_unit_sphere(rng), aperture * 0.5f);
                    origin.x += jitter.x;
                    origin.y += jitter.y;
                }

                // Primary ray.
                Ray ray;
                ray.orig = origin;
                ray.dir = norm(sub(add(add(lower_left, scale(horizontal, u)),
                                       scale(vertical, v)),
                                   origin));
                ray.time = frame_time;

                // Trace + background gradient (white → background colour).
                float3 colour = test_trace(ray, scene, max_bounces,
                                           transparent_backfaces, rng);
                if ( colour.x == 0.f && colour.y == 0.f && colour.z == 0.f ) {
                    float t = 0.5f * (ray.dir.y + 1.0f);
                    colour = lerp({1, 1, 1}, bg, t);
                }

                // Accumulate.
                int base = flat_index * 4;
                accum[base + 0] += colour.x;
                accum[base + 1] += colour.y;
                accum[base + 2] += colour.z;
                accum[base + 3] += 1.0f;
            }
        });

    // ── Statistics (host side, race-free) ────────────────────────
    if ( ctx->stats ) {
        ctx->stats->write<int>("num_objects", scene.num_quads + scene.num_boxes);
        ctx->stats->write<int>("num_bvh_nodes", 0);
        ctx->stats->write<int>("num_lights", 1);
    }
}

extern "C" void shutdown_kernel(KERNEL_QUEUE_PARAM) {
    PROFILER_ZONE("shutdown_kernel");
    // Nothing to free — this kernel allocates no device memory.
}
