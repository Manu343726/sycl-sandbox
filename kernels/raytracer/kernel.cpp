#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types.h>
#include <sycl-sandbox/rt/trace.h>
#include <sycl-sandbox/kernel/params.h>
#include <sycl-sandbox/kernel/stats.h>
#include <sycl-sandbox/scene/data.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include <cstring>

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
    "raytracer",
    "Generic raytracer — scene geometry and all params come from YAML",
    4096,
    2,
    (const char *[]) {"kernel.cpp", nullptr}};

extern "C" KernelDesc *get_kernel_desc() {
    return &desc;
}

// ── Scene state (externally supplied via set_scene_view / set_scene_debug_info) ──

static const SceneView *external_scene = nullptr;
static const SceneDebugInfo *external_debug = nullptr;
static float3 background = {0.0f, 0.0f, 0.0f};

extern "C" void set_scene_view(const SceneView *view) {
    external_scene = view;
}

extern "C" void set_scene_debug_info(const SceneDebugInfo *info) {
    external_debug = info;
}

// ── init_kernel ─────────────────────────────────────────────────────
// Reads all params from the buffer via the type-safe ParamLookup API.
// The scene geometry is built by the host from YAML and injected via
// set_scene_view — nothing to construct here.

extern "C" void init_kernel(KERNEL_QUEUE_PARAM, int, int, const void *params_buffer, size_t) {
    PROFILER_ZONE("init_kernel");
    s_lookup.set_buffer(params_buffer);
    background = s_lookup.read_vec3<float3>("background");
}

// ── render_kernel (ABI v2) ────────────────────────────────────────
// ENQUEUE-ONLY: submits the render job and returns without waiting.

extern "C" void render_kernel(KERNEL_QUEUE_PARAM, const RenderContext *ctx) {
    PROFILER_ZONE("render");
    if (!ctx || !g_rt) return;

    s_lookup.set_buffer(ctx->params);
    // Unique SYCL kernel name tag — see rt::Runtime::foreach_pixel() for
    // why every kernel .so needs a distinct one.
    render_main<class RaytracerPixelKernel>(
                *g_rt,
                ctx->width,
                ctx->height,
                s_lookup,
                ctx->accum,
                ctx->spp_total,
                external_scene ? *external_scene : SceneView{},
                [bg = background](const Ray &ray) -> float3 {
                    // Sky gradient from white to background colour
                    float t = 0.5f * (ray.dir.y + 1.0f);
                    return lerp({1, 1, 1}, bg, t);
                },
                ctx->prof,
                ctx->trace_counters);

    // Write statistics to the per-frame block
    if (external_scene && ctx->stats) {
        ctx->stats->write<int>("num_objects", external_scene->num_handles);
        ctx->stats->write<int>("num_bvh_nodes", external_scene->num_bvh_nodes);
        ctx->stats->write<int>("num_lights", external_scene->num_lights);
    }
}

// ── Per-frame statistics (optional export, ABI v2) ───────────────────
// Statistics are kernel outputs, written by the kernel.  The host calls
// this AFTER the frame's device work completed (render kernel + tone-map),
// so the device-side trace counters are final; we read them back and
// publish num_hits / num_bvh_hits into the stat block via ctx->stats.
// (The copy is blocking, but the queue is idle here — the host waited
// for the frame before calling us, so it returns immediately.)

extern "C" void collect_frame_stats(KERNEL_QUEUE_PARAM,
                                     const RenderContext *ctx) {
    PROFILER_ZONE("collect_frame_stats");
    if ( !ctx || !g_rt ) return;
    if ( !ctx->stats || !ctx->trace_counters ) return;

    rt::TraceCounters local;
    g_rt->copy_to_host(&local, ctx->trace_counters, sizeof(rt::TraceCounters));
    ctx->stats->write<int>("num_hits", (int)local.num_hits);
    ctx->stats->write<int>("num_bvh_hits", (int)local.num_bvh_hits);
}

// ── shutdown_kernel ─────────────────────────────────────────────────
// Does NOT free the scene — the host owns it.

extern "C" void shutdown_kernel(KERNEL_QUEUE_PARAM) {
    PROFILER_ZONE("shutdown_kernel");
    external_scene = nullptr;
    external_debug = nullptr;
}

// ── Debug info ──────────────────────────────────────────────────────

extern "C" const SceneDebugInfo *get_scene_debug_info() {
    return external_debug;
}
