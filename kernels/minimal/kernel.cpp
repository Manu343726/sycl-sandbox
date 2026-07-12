// Minimal stub kernel — does absolutely nothing.
// Exists only to exercise the kernel loading, buffer, and display infrastructure
// without any actual rendering logic.
// This kernel intentionally has no SYCL kernel launch whatsoever.

#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/kernel/params.h>
#include <sycl-sandbox/kernel/stats.h>
#include <sycl-sandbox/kernel/execution_context.h>

// ── Runtime (set by host to dispatch SYCL vs software) ───────────────
static rt::Runtime *g_rt = nullptr;

extern "C" void set_runtime(rt::Runtime *rt) {
    g_rt = rt;
}

// ── Param lookup (host-populated from YAML scene descriptor) ────────
static rt::ParamLookup s_lookup;

extern "C" void set_param_lookup(const rt::ParamLookup *lookup) {
    if (lookup) {
        s_lookup = *lookup;
    } else {
        s_lookup = rt::ParamLookup{};
    }
}

// ── Stat writer (kernel-populated, host reads after each frame) ───────
static rt::StatWriter s_stat_writer;

extern "C" void set_stat_lookup(const rt::StatWriter *writer) {
    if (writer) {
        s_stat_writer = *writer;
    } else {
        s_stat_writer = rt::StatWriter{};
    }
}

static const char *source_files[] = {"kernel.cpp", nullptr};
// max_spp=1  We render only once, no accumulation needed.
static KernelDesc desc =
    {"minimal", "Stub kernel — does nothing", 1, 1, source_files};

extern "C" KernelDesc *get_kernel_desc() {
    return &desc;
}

extern "C" void init_kernel(KERNEL_QUEUE_PARAM, int, int, const void *buf, size_t) {
    PROFILER_ZONE("init_kernel");
    s_lookup.set_buffer(buf);
}

extern "C" void render_kernel(KERNEL_QUEUE_PARAM, const RenderContext *ctx) {
    PROFILER_ZONE("render_kernel");
    if (!ctx || !g_rt) return;
    s_lookup.set_buffer(ctx->params);

    // Do absolutely nothing — no SYCL kernel launch, no pixel iteration.
    // The accum buffer stays zero-filled, display shows black.
}

extern "C" void shutdown_kernel(KERNEL_QUEUE_PARAM) {}
