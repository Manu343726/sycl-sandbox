// sycl-sandbox — Instrumented ray trace for the scene debug view
//
// The debug view runs the REAL rt::trace() (the same function the
// kernel .so compiles), with the rt::TraceCollector armed over host
// rings and the per-call rt::Context built from them, so everything the
// debugger needs to visualize a single ray path is captured:
//   - every bounce (ray, closest hit, material emission, scatter result)
//   - which BVH nodes the ray's traversal actually entered (AABB hits),
//     in traversal order — the "BVH hits" of the ray
//   - deep pipeline metadata (primitive hit tests, material scatters /
//     emits, texture samples) attached to each step via its event_start
//
// The kernel path leaves the collector inactive (empty hooks the
// compiler eliminates), so there is exactly ONE path-tracing
// implementation: the debugger cannot diverge from the renderer.
//
// Fully host-compilable: it only uses the rt runtime headers (the same
// ones the kernel .so compiles), so the debug render thread can run it
// against the host copy of the scene (SceneDebugScene) every frame.

#pragma once

#include <sycl-sandbox/context.h>
#include <sycl-sandbox/rt/trace.h>
#include <sycl-sandbox/rt/collector.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace rt {

/// One bounce of an instrumented ray trace.
struct RayTraceStep {
    Ray ray = {};                        ///< ray entering this step
    /// BVH nodes whose AABB the ray entered during closest-hit traversal,
    /// in traversal order (leaf nodes whose object was missed are still
    /// recorded — the ray did enter their box).  Empty when the scene has
    /// no BVH (linear-scan fallback).
    std::vector<uint32_t> bvh_visited;
    /// Per-mesh BVH nodes entered while intersecting a Mesh hittable
    /// (indices into SceneView::mesh_bvh_nodes), in traversal order.
    /// Distinct from bvh_visited (the scene BVH): a single mesh leaf of
    /// the scene BVH can expand into many per-mesh nodes.  Empty when no
    /// per-mesh BVH was built (linear triangle scan).
    std::vector<uint32_t> hittable_bvh_visited;

    bool hit = false;                    ///< did the ray hit an object
    Handle handle = {};                  ///< hit handle (valid when hit)
    HitRecord record = {};               ///< closest hit record
    float3 emit = {0, 0, 0};             ///< emission of the hit material
    bool scattered = false;              ///< did the material scatter
    Ray scattered_ray = {};              ///< continuation ray (when scattered)
    float3 attenuation = {1, 1, 1};      ///< scatter attenuation (when scattered)
    /// Running path throughput entering this step (the product of all
    /// previous scatter attenuations — what the real tracer attenuates
    /// the remaining path by).
    float3 throughput = {1, 1, 1};

    /// Terminal state after this step (exactly one is true when the path
    /// ended; none when the trace simply hit the bounce limit).
    bool escaped = false;                ///< ray left the scene (no hit)
    bool absorbed = false;               ///< material absorbed the ray

    /// Deep pipeline metadata recorded during this step (in evaluation
    /// order, via the collector hooks — the collector reaches into the
    /// primitives, materials, and textures, not just trace()):
    /// every primitive hit test performed while finding the closest hit
    /// (misses included), every material scatter/emit evaluated, and
    /// every texture sample taken by the scatter.
    std::vector<HittableType> hit_tests;
    std::vector<MaterialType> scatters;
    std::vector<MaterialType> emits;
    std::vector<uint32_t> texture_samples;

    /// Output colour of this hit: the radiance the path sends back
    /// along this step's ray (camera direction).  Computed AFTER the
    /// trace by back-propagation — path tracing goes camera → scene,
    /// so the colour of a hit is only known once its children are
    /// (the real tracer accumulates it as the recursion unwinds):
    ///     emit + attenuation * (colour of the continuation)
    /// where the continuation is the next step's colour, the sky
    /// gradient when the ray escaped, or black when absorbed.  For a
    /// path cut short by the bounce limit the rest is approximated
    /// with the sky, exactly like rt::trace().
    float3 color = {0, 0, 0};
};

/// Result of tracing one ray recursively through the scene.
struct RayTraceResult {
    Ray start = {};                      ///< the initial ray
    std::vector<RayTraceStep> steps;     ///< one entry per bounce (≤ max_bounces)
    /// True when the path ended because max_bounces was reached
    /// (steps.size() == max_bounces and the last step is not terminal).
    bool bounce_limit = false;

    /// Every BVH node entered across all steps, deduplicated.
    std::vector<uint32_t> visited_nodes() const {
        std::vector<uint32_t> nodes;
        for ( const auto &s : steps )
            for ( uint32_t n : s.bvh_visited )
                if ( std::find(nodes.begin(), nodes.end(), n) == nodes.end() )
                    nodes.push_back(n);
        return nodes;
    }

    /// Every per-mesh BVH node entered across all steps, deduplicated.
    std::vector<uint32_t> visited_hittable_nodes() const {
        std::vector<uint32_t> nodes;
        for ( const auto &s : steps )
            for ( uint32_t n : s.hittable_bvh_visited )
                if ( std::find(nodes.begin(), nodes.end(), n) == nodes.end() )
                    nodes.push_back(n);
        return nodes;
    }
};

/// Trace a single ray with the REAL rt::trace(), recording every bounce
/// through the rt::TraceCollector ring.  The kernel path leaves the
/// collector inactive (empty hooks — zero cost), so the debugger and
/// the renderer share ONE path-tracing implementation: same BVH
/// traversal, same dispatch, same RNG sequence, same early returns.
/// After the trace, every step's `color` is back-propagated
/// (emit + attenuation × continuation), so step 0's colour is the
/// final pixel colour of the traced path.
///
/// The collector rings are plain host stack arrays (power-of-two
/// capacities); trace() appends one TraceStepRecord per step and the
/// pipeline appends TraceEvents for the deep metadata, then the steps
/// are reconstructed in record order and each step attaches the events
/// in [prev.event_start, this.event_start).
///
/// \param rng        RNG for material scatter — pass a FIXED seed for a
///                   deterministic, stable visualization.
/// \param max_bounces recursion depth: number of bounces to trace.
/// \param transparent_backfaces X-ray mode (see rt::trace()).
/// \param background scene background colour (the kernel's "background"
///                   param): escaped rays fold in the same sky gradient
///                   the kernel uses — lerp(white, background, 0.5·(dir.y+1)).
inline RayTraceResult trace_ray_debug(const Ray &ray, const SceneView &scene,
                                      int max_bounces, RNG &rng,
                                      bool transparent_backfaces = false,
                                      float3 background = {0, 0, 0}) {
    RayTraceResult result;
    result.start = ray;
    if ( max_bounces <= 0 ) return result;

    // Ring capacities (power of two — records wrap via mask).  A debug
    // path is at most max_bounces + 1 steps (the +1 is the terminal
    // bounce-limit marker), and each step fires a handful of events per
    // linear-scan hit test / BVH node.  If a ring ever wraps, the
    // reconstruction below clamps to the newest captured records.
    constexpr uint32_t STEP_CAP = 1u << 6;    // 64 steps
    constexpr uint32_t EVENT_CAP = 1u << 12;  // 4096 events
    TraceCollectorHeader header = {};
    TraceStepRecord steps[STEP_CAP];
    TraceEvent events[EVENT_CAP];

    // Same sky gradient the kernel folds into escaped paths
    // (kernels/raytracer/kernel.cpp bg_fn).
    auto sky = [&background](const Ray &r) -> float3 {
        float t = 0.5f * (r.dir.y + 1.0f);
        return lerp({1.f, 1.f, 1.f}, background, t);
    };

    // Arm the per-call context with the collector and run the REAL path
    // tracer — trace() and every pipeline stage write into the rings;
    // the reconstruction below recovers the path and per-hit radiances.
    Context ctx;
    ctx.collector = TraceCollector {&header, events, steps, EVENT_CAP, STEP_CAP};
    trace(ray, scene, max_bounces, transparent_backfaces, rng, sky, ctx);

    // ── Reconstruct the path steps from the step ring ────────────────
    // Steps are appended in record order (one per completed bounce, plus
    // the terminal bounce-limit marker).  The bounce-limit marker is not
    // a real bounce — flag it on the result and drop it, keeping one
    // node per bounce in the treeview.
    const uint32_t step_count = header.step_pos;
    const uint32_t n_steps = std::min(step_count, STEP_CAP);
    const uint32_t first_step = step_count - n_steps;

    for ( uint32_t k = 0; k < n_steps; k++ ) {
        const TraceStepRecord &r = steps[(first_step + k) & (STEP_CAP - 1)];
        if ( r.bounce_limit ) {
            result.bounce_limit = true;
            continue;
        }

        RayTraceStep step;
        step.ray = r.ray;
        step.hit = r.hit;
        step.handle = r.handle;
        step.record = r.record;
        step.emit = r.emit;
        step.scattered = r.scattered;
        step.scattered_ray = r.scattered_ray;
        step.attenuation = r.scatter_attenuation;
        step.throughput = r.throughput;
        step.escaped = r.escaped;
        step.absorbed = r.absorbed;

        // Attach this step's deep-pipeline events: the range
        // [prev.event_start, this.event_start) in the event ring.
        uint32_t begin = k == 0 ? 0
                                : steps[(first_step + k - 1) & (STEP_CAP - 1)].event_start;
        uint32_t end = r.event_start;
        const uint32_t event_count = header.event_pos;
        const uint32_t event_first = event_count >= EVENT_CAP ? event_count - EVENT_CAP : 0;
        begin = std::max(begin, event_first);
        end = std::min(end, event_count);
        for ( uint32_t e = begin; e < end; e++ ) {
            const TraceEvent &ev = events[e & (EVENT_CAP - 1)];
            switch ( static_cast<TraceEventKind>(ev.kind) ) {
                case TraceEventKind::BvhNode:
                    step.bvh_visited.push_back(ev.payload);
                    break;
                case TraceEventKind::HittableBvhNode:
                    step.hittable_bvh_visited.push_back(ev.payload);
                    break;
                case TraceEventKind::HitTest:
                    step.hit_tests.push_back(static_cast<HittableType>(ev.payload));
                    break;
                case TraceEventKind::Scatter:
                    step.scatters.push_back(static_cast<MaterialType>(ev.payload));
                    break;
                case TraceEventKind::Emit:
                    step.emits.push_back(static_cast<MaterialType>(ev.payload));
                    break;
                case TraceEventKind::TextureSample:
                    step.texture_samples.push_back(ev.payload);
                    break;
            }
        }

        result.steps.push_back(std::move(step));
    }

    // ── Back-propagate the output colours ──────────────────────────
    // The radiance of a hit is only known after its children (the real
    // tracer accumulates it as the recursion unwinds): walk the steps
    // backwards, folding each continuation into its parent.
    for ( int i = (int)result.steps.size() - 1; i >= 0; i-- ) {
        RayTraceStep &s = result.steps[i];
        float3 child = {0.f, 0.f, 0.f};
        if ( s.escaped ) {
            child = sky(s.ray);
        } else if ( s.absorbed ) {
            child = {0.f, 0.f, 0.f};
        } else if ( i + 1 < (int)result.steps.size() ) {
            child = result.steps[i + 1].color;
        } else if ( s.scattered ) {
            // Bounce limit: the remaining path is approximated with
            // the sky, exactly like rt::trace().
            child = sky(s.scattered_ray);
        }
        // Emissive terminal steps (emit != 0, never scattered): the
        // continuation is black — the light ends the path.
        s.color = add(s.emit, mul(s.attenuation, child));
    }
    return result;
}

} // namespace rt
