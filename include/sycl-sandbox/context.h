#pragma once

/// @file
/// The single kernel entry point ABI — the one type every kernel .so
/// and the host share.  Each kernel exports exactly ONE function:
///
///   extern "C" void kernel_entry(const rt::Context *ctx)
///
/// called once per frame.  There is no op dispatch, no setup phase, no
/// shutdown, no metadata query: the caller owns every resource and
/// passes it in the Context on every call.  The kernel keeps NO state
/// between calls — local variables only (profiler zones record through
/// ctx->prof).  Because the kernel owns nothing, unloading the .so
/// needs no teardown.
///
/// Kernel metadata (name, max_spp, whether it consumes a scene) lives
/// host-side in the scene data, not in the kernel binary.

#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/rt/collector.h>

#include <cstdint>
#include <cstddef>

namespace rt {

struct Runtime;       ///< kernel/execution_context.h
struct ParamLookup;   ///< kernel/params.h
struct StatWriter;    ///< kernel/stats.h
struct SceneView;     ///< scene/data.h

/// Per-frame kernel context.  The host fills every field the kernel
/// needs for THIS frame; the kernel reads them, does its work, and
/// returns.  The pointer is valid only for the duration of the call.
/// POD and SYCL-free (the queue lives in rt::Runtime, not here).
struct Context {
    /// Kernel execution context — owns the queue and the memory
    /// allocators (foreach_pixel, buffers, transfers).  Never null.
    Runtime *runtime = nullptr;
    /// Device-visible cancellation flag (USM shared, null when
    /// uninitialised / software backend).  Device code reads this
    /// pointer BY VALUE with `ctx.cancel_flag && __atomic_load_n(...)`
    /// and must NEVER dereference `runtime` — it lives in host memory.
    int *cancel_flag = nullptr;
    /// Host-visible parameter snapshot.  Read HOST-side and capture
    /// scalars by value into kernel lambdas — device code must never
    /// dereference it.
    const ParamLookup *params = nullptr;
    /// Stat block writer (host-side float buffer).  Null when the scene
    /// has no stats.
    const StatWriter *stats = nullptr;
    /// Device scene view for this frame.  Null when the scene data has
    /// nothing to build.
    const SceneView *scene = nullptr;
    /// Device profiler ring.  Capture BY VALUE into kernel lambdas and
    /// use PROFILER_ZONE; an inactive ring makes it a no-op.
    profiler::DeviceRing prof = {};
    /// Per-frame trace counters (num_hits / num_bvh_hits).  Device-side
    /// scratch buffer, zeroed by the host before enqueuing the render
    /// kernel; the host reads it back after frame completion.  May be
    /// null.
    TraceCounters *trace_counters = nullptr;
    /// Trace collector ring — the pipeline writes path steps + deep
    /// metadata through it (rt::TraceCollector).  Inactive on the render
    /// path; the scene-debug view arms it over host memory and reads the
    /// rings back after a single-ray trace.
    TraceCollector collector = {};
    /// Profiler work-item id: the render loop stamps a per-pixel copy of
    /// the Context with each work item's linear id, so device profiler
    /// zones stay decimated across work items.  0 for host-side tracing.
    uint32_t linear_id = 0;

    // ── Render ──
    int32_t width = 0;
    int32_t height = 0;
    /// Persistent accumulation buffer: device float4[width*height].
    /// Kernels ADD samples (rgb += sample; w += 1 per sample) across
    /// frames for progressive rendering.  Never cleared by the kernel;
    /// the host resets it on accumulation restart.
    float *accum = nullptr;
    /// Tone-mapped RGBA8 output buffer (device-visible): the kernel
    /// reads accumulated samples from accum, applies the tone-map
    /// operator, and packs the final display frame here.  width*height*4
    /// bytes, RGBA order.  This IS the display slot buffer.
    uint8_t *output = nullptr;
    /// Samples to add in this call (host's read of "spp_frame").
    uint32_t spp_frame = 0;
    /// Samples already accumulated before this call (RNG decorrelation).
    uint32_t spp_total = 0;
    /// Monotone frame counter (animation tick).
    uint64_t frame_index = 0;
};

} // namespace rt

/// The single kernel entry point.
///
/// ENQUEUE-ONLY CONTRACT: the kernel must submit its device work to the
/// queue (via ctx->runtime) and return WITHOUT waiting for completion.
/// The host chains the tone-map and display publication after it on the
/// same in-order queue.  (Native mode executes synchronously — the
/// contract is trivially met.)
///
/// The kernel uses only local variables; every input arrives through
/// ctx.  No persistent kernel-side state, so no setup/shutdown ops.
extern "C" void kernel_entry(const rt::Context *ctx);
