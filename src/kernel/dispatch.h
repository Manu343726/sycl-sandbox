#pragma once
#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/context.h>
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/kernel/params.h>
#include <sycl-sandbox/kernel/stats.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include <sycl-sandbox/rt/collector.h>

#include <dlfcn.h>

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
inline kernel_entry_fn resolve_kernel_entry(void *handle) {
    return reinterpret_cast<kernel_entry_fn>(dlsym(handle, "kernel_entry"));
}

/// Render one frame: forward the fully-populated context to the kernel.
/// No-op when the kernel doesn't export the entry (old-ABI binary — the
/// loader refuses those, so in practice this only happens in hand-rolled
/// paths like the GPU diag).
inline void call_kernel_entry(void *handle, const rt::Context &ctx) {
    PROFILER_FUNCTION();
    auto *fn = resolve_kernel_entry(handle);
    if ( fn ) fn(&ctx);
}
