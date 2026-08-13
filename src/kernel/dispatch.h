#pragma once
#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/context.h>
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/kernel/params.h>
#include <sycl-sandbox/kernel/stats.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include <sycl-sandbox/rt/collector.h>

#include <dlfcn.h>
#include <cstdio>

// ── Single-entry kernel dispatch ──────────────────────────────────────
// ABI: every kernel .so exports exactly ONE function,
// `kernel_entry(const rt::Context*)`, called once per frame.  The caller
// owns all resources and passes them in the Context; the kernel keeps no
// state between calls.  No op dispatch — one function, one purpose.
//
// q is null on the software (native) backend — the kernel .so was built
// with KERNEL_NATIVE; rt::Runtime owns the queue, so rt::Context carries
// no SYCL types at all.

using kernel_entry_fn = void (*)(const rt::Context *);

/// Resolve the single kernel entry point.
/// Called once at kernel load time (KernelLibrary::load()); the result
/// is cached in KernelHandle::entry so the per-frame dispatch path
/// never calls dlsym again.
inline kernel_entry_fn resolve_kernel_entry(void *handle) {
    return reinterpret_cast<kernel_entry_fn>(dlsym(handle, "kernel_entry"));
}

/// Render one frame: forward the fully-populated context to the kernel.
/// `entry` is the cached dlsym result for the kernel's "kernel_entry"
/// symbol (KernelHandle::entry), resolved once at load().  A null entry
/// only happens in hand-rolled paths like the GPU diag — the loader
/// refuses kernels that don't export the symbol.
inline void call_kernel_entry(void *entry, const rt::Context &ctx) {
    PROFILER_FUNCTION();
    auto fn = reinterpret_cast<kernel_entry_fn>(entry);
    if ( fn ) fn(&ctx);

    // Per-call Frame-scope enforcement: the kernel must leave no host-side
    // allocations behind (no-state-between-calls contract).  Any leftover
    // Frame-scoped buffer is self-healing-freed here; a nonzero count is a
    // kernel bug reported through stderr in debug builds.
    if ( ctx.runtime && ctx.runtime->pool ) {
        size_t leaked = ctx.runtime->pool->check_frame_clean();
#ifndef NDEBUG
        if ( leaked ) {
            std::fprintf(stderr,
                         "[kernel] kernel_entry leaked %zu Frame-scope "
                         "allocation(s); freed by the runtime\n",
                         leaked);
        }
#endif
    }
}
