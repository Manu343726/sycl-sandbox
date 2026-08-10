#pragma once
#include <sycl-sandbox/context.h>
#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/rt/helpers.h>
#include <sycl-sandbox/rt/types.h>
#include <sycl-sandbox/rt/tonemap.h>
#include <sycl-sandbox/math.h>
#include <sycl-sandbox/scene/camera.h>
#include <sycl-sandbox/kernel/params.h>
#include <sycl-sandbox/scene/data.h>
#include <sycl-sandbox/kernel/execution_context.h>

namespace rt {

constexpr const char *KEY_SPP_FRAME    = "spp_frame";
constexpr const char *KEY_MAX_BOUNCES  = "max_bounces";
constexpr const char *KEY_TRANSPARENT_BACKFACES = "transparent_backfaces";
constexpr const char *KEY_CAM_EYE      = "cam_eye";
constexpr const char *KEY_CAM_AT       = "cam_at";
constexpr const char *KEY_CAM_UP       = "cam_up";
constexpr const char *KEY_CAM_FOV      = "cam_fov";
constexpr const char *KEY_CAM_APERTURE = "cam_aperture";
constexpr const char *KEY_TICK         = "tick";
constexpr const char *KEY_TIME         = "time";

// Deserialized standard parameters
struct Params {
    int spp_frame = 0;
    int max_bounces = 8;
    bool transparent_backfaces = false;
    float3 camera_eye = {0, 0, 0};
    float3 camera_at = {0, 0, -1};
    float3 camera_up = {0, 1, 0};
    float camera_fov = 60.0f;
    float camera_aperture = 0.0f;
    Camera camera;
    int tick = 0;
    float time = 0.0f;

    static Params from_lookup(const ParamLookup *lookup, const float aspectRatio) {
        Params p;
        p.spp_frame = lookup->read<int>(KEY_SPP_FRAME);
        p.max_bounces = lookup->read<int>(KEY_MAX_BOUNCES);
        p.transparent_backfaces = lookup->read<bool>(KEY_TRANSPARENT_BACKFACES);
        p.camera_eye = lookup->read<float3>(KEY_CAM_EYE);
        p.camera_at = lookup->read<float3>(KEY_CAM_AT);
        p.camera_up = lookup->read<float3>(KEY_CAM_UP);
        p.camera_fov = lookup->read<float>(KEY_CAM_FOV);
        p.camera_aperture = lookup->read<float>(KEY_CAM_APERTURE);
        p.camera = lookat(p.camera_eye, p.camera_at, p.camera_up, p.camera_fov, aspectRatio);
        p.tick = lookup->read<int>(KEY_TICK);
        p.time = lookup->read<float>(KEY_TIME);
        return p;
    }
};

// ── Path tracing ───────────────────────────────────────────────────────

/// Trace a ray through the scene and return the accumulated colour.
/// Uses per-handle dispatch to hit, scatter, and emit on the correct
/// per-type arrays.
///
/// When a ray escapes the scene (or hits the bounce limit), the
/// background function is evaluated at the current ray and folded into
/// the path, attenuated by the throughput so far — standard sky
/// lighting.  This is what makes sky-lit scenes (no emissive lights,
/// e.g. "one weekend" style) render at all: without it every path
/// ends black and the whole image collapses to the flat background.
///
/// \param transparent_backfaces X-ray mode: when true, rays pass through
///        surfaces hit from behind (front_face == false), so a camera
///        outside an enclosed scene can see inside through the walls.
/// \param ctx per-call kernel context: carries the device profiler ring
///        + per-work-item id (every pipeline stage below — trace, bounce,
///        hit tests, scatter, emit, texture sampling — records profiler
///        zones through it, shown as "gpu:*" zones on the device
///        timeline), the per-frame trace counters (num_hits /
///        num_bvh_hits, atomically counted on each closest hit), and the
///        trace collector.  The collector can be reached from ANY depth
///        of the pipeline, so debuggers append metadata beyond the
///        per-bounce steps recorded here.  An inactive context (default)
///        makes every zone/hook a no-op, so host-side callers
///        (scene-debug, tests) pay nothing.
template <typename BgFn>
inline float3 trace(const Ray &ray, const SceneView &scene, int max_bounces,
                    bool transparent_backfaces, RNG &rng,
                    BgFn &&background_fn,
                    const Context &ctx = Context{}) {

    PROFILER_FUNCTION();

    // Initialise the path throughput (attenuation) and the working ray
    float3 attenuation = {1, 1, 1};
    Ray ray_in_out = ray;

    // Trace the ray through successive bounces
    for ( int bounce = 0; bounce < max_bounces; bounce++ ) {
        PROFILER_ZONE("trace_bounce");

        // Find the closest object hit by the ray within [0.001, ∞).
        // Preferred path: closest-hit query over the flat BVH array
        // (skips whole subtrees via AABB culling).  Falls back to a
        // linear scan when no BVH was built for the scene.
        bool used_bvh = scene.bvh_nodes && scene.bvh_root >= 0;
        optional<HitRecord> closest_hit;
        Handle hit_handle = {};

        if ( used_bvh ) {
            auto bvh_result = bvh_hit(ray_in_out, 0.001f, 1e30f, scene,
                                      transparent_backfaces, ctx);

            if ( bvh_result ) {
                closest_hit = bvh_result->record;
                hit_handle = bvh_result->handle;
            }
        } else {
            for ( int i = 0; i < scene.num_handles; i++ ) {
                auto hit = handle_hit(scene.handles[i],
                                      ray_in_out,
                                      0.001f,
                                      closest_hit ? closest_hit->t : 1e30f,
                                      scene,
                                      ctx);
                if ( !hit ) continue;
                // X-ray mode: back faces are transparent — skip hits from
                // behind so the ray passes through the surface.
                if ( transparent_backfaces && !hit->front_face ) continue;
                closest_hit = hit;
                hit_handle = scene.handles[i];
            }
        }

        // If no object was hit the ray escapes to the void: fold the
        // sky colour into the path, attenuated by the throughput.
        if ( !closest_hit ) {
            TraceStepRecord record;
            record.ray = ray_in_out;
            record.escaped = true;
            record.throughput = attenuation;
            ctx.collector.record_step(record);
            return mul(attenuation, background_fn(ray_in_out));
        }

        // Count the closest hit (BVH provenance distinguishes the two
        // query paths — with the BVH active, every hit is a BVH hit).
        if ( ctx.trace_counters ) ctx.trace_counters->add_hit(used_bvh);

        // If the hit object emits light, return the attenuated emission
        float3 emitted = handle_emit(hit_handle, *closest_hit, scene, ctx);
        if ( emitted.x != 0 || emitted.y != 0 || emitted.z != 0 ) {
            TraceStepRecord record;
            record.ray = ray_in_out;
            record.hit = true;
            record.handle = hit_handle;
            record.record = *closest_hit;
            record.emit = emitted;
            record.throughput = attenuation;
            ctx.collector.record_step(record);
            return mul(attenuation, emitted);
        }

        // Scatter the ray; if the material absorbs it, terminate the path
        auto scattered = handle_scatter(hit_handle, ray_in_out, *closest_hit, rng, scene,
                                        ctx);
        if ( !scattered ) {
            TraceStepRecord record;
            record.ray = ray_in_out;
            record.hit = true;
            record.handle = hit_handle;
            record.record = *closest_hit;
            record.absorbed = true;
            record.throughput = attenuation;
            ctx.collector.record_step(record);
            return {0, 0, 0};
        }

        {
            TraceStepRecord record;
            record.ray = ray_in_out;
            record.hit = true;
            record.handle = hit_handle;
            record.record = *closest_hit;
            record.scattered = true;
            record.scattered_ray = scattered->scattered;
            record.scatter_attenuation = scattered->attenuation;
            record.throughput = attenuation;
            ctx.collector.record_step(record);
        }

        // Update the attenuation and continue with the scattered ray
        attenuation = mul(attenuation, scattered->attenuation);
        ray_in_out = scattered->scattered;
    }

    // Bounce limit reached — approximate the remaining path with the
    // sky, attenuated by the throughput so far.
    {
        TraceStepRecord record;
        record.ray = ray_in_out;
        record.bounce_limit = true;
        record.throughput = attenuation;
        ctx.collector.record_step(record);
    }
    return mul(attenuation, background_fn(ray_in_out));
}

// ── Render entry point ─────────────────────────────────────────────────

/// KernelName is an explicit SYCL kernel name tag, unique per calling
/// kernel .so — see rt::Runtime::foreach_pixel() for why this is required
/// (AdaptiveCpp's kernel identity is derived from the mangled name of the
/// enclosing function, which collides across .so's that reuse the same
/// function name, e.g. every kernel's `render_kernel` entry point).
///
/// The shared render path used by every rt::Context Render phase: the
/// kernel .so receives the context, then calls rt::render<KernelName>()
/// which reads the standard params, walks every pixel through the Runtime
/// abstraction, and accumulates samples into the RGBA accumulator.
///
/// \param ctx the rt::Context for this invocation — carries the scene
///        view (`ctx.scene`), profiler ring, per-frame trace counters,
///        and collector.  Copied into a per-work-item Context (stamped
///        with the pixel's linear id) that is threaded down through
///        every pipeline call.
/// \param background_fn evaluated when a ray escapes the scene; the sky
///        colour is folded into the path, attenuated by throughput.
template <typename KernelName, typename BgFn>
void render(const Context &ctx, BgFn &&background_fn) {
    PROFILER_FUNCTION();

    const auto &reader = *ctx.params;
    const auto params = Params::from_lookup(&reader, (float)ctx.width / (float)ctx.height);
    const SceneView &scene = *ctx.scene;

    // Iterate over every pixel through the Runtime abstraction.
    // In SYCL mode this uses queue->parallel_for; in software mode it
    // uses plain nested for-loops — no OpenMP, no SYCL in the kernel.
    bool  tm_on      = reader.read<bool>("tonemap_enabled");
    int   tm_op      = reader.read<int>("tonemap_operator");
    float tm_exp     = reader.read<float>("tonemap_exposure");
    float tm_gamma   = reader.read<float>("tonemap_gamma");

    ctx.runtime->foreach_pixel<KernelName>(ctx.width, ctx.height, [=](int x, int y, int flat_index) {
        PROFILER_ZONE("render_pixel");

        // Per-work-item context: a copy of the frame context with a
        // consistent linear id for this pixel (the profiler ring and the
        // per-frame hit counters carry over from the frame context; the
        // collector is inactive on the render path — empty hooks, zero
        // cost).
        rt::Context px = ctx;
        px.linear_id = static_cast<uint32_t>(flat_index);

        for ( int sample = 0; sample < params.spp_frame; sample++ ) {
            PROFILER_ZONE("render_sample");

            // Initialise the per-pixel, per-sample RNG from the pixel index and
            // total accumulated samples for stream decorrelation.
            RNG rng {static_cast<uint32_t>(flat_index * 6364136223846793005ull +
                                           (uint64_t)(ctx.spp_total * 2654435761u) +
                                           sample)};

            // Generate stratified pixel coordinates with random offsets
            float u = (x + rng.next()) / (float)ctx.width;
            float v = (y + rng.next()) / (float)ctx.height;

            // Optionally jitter the ray origin for depth-of-field effects
            float3 ray_origin = params.camera.origin;
            if ( params.camera_aperture > 0.f ) {
                float3 jitter = scale(random_in_unit_sphere(rng), params.camera_aperture * 0.5f);
                ray_origin.x += jitter.x;
                ray_origin.y += jitter.y;
            }

            // Construct the primary ray from the camera frustum
            Ray ray;
            ray.orig = ray_origin;
            ray.dir = norm(sub(add(add(params.camera.lower_left, scale(params.camera.horizontal, u)),
                                   scale(params.camera.vertical, v)),
                               ray_origin));
            ray.time = params.tick;

            // Trace the ray through the scene; trace() folds the sky
            // colour into the path when the ray escapes, attenuated by
            // the throughput (see rt::trace()).
            float3 colour = trace(ray, scene, params.max_bounces,
                                  params.transparent_backfaces, rng, background_fn, px);

            {
                PROFILER_ZONE("accumulate_pixel");

                // Accumulate the sample into the RGBA output buffer
                int base = flat_index * 4;
                ctx.accum[base + 0] += colour.x;
                ctx.accum[base + 1] += colour.y;
                ctx.accum[base + 2] += colour.z;
                ctx.accum[base + 3] += 1.0f;
            }
        }

        // ── Per-pixel tone-map: accum float4 → RGBA8 output ────
        tonemap_pixel(px, flat_index, tm_on, tm_op, tm_exp, tm_gamma);
    });
}

} // namespace rt
