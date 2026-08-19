#pragma once

/// @file
/// Kernel execution context — the runtime abstraction passed to every
/// kernel .so via set_runtime().  Fully hides whether the kernel is
/// executing on a SYCL device or in software (plain C++).
///
/// SYCL mode:   queue points to a valid sycl::queue
/// Software mode: queue == nullptr, all operations use std::memcpy,
///                new[]/delete[], and nested for-loops.
///
/// When KERNEL_NATIVE is defined (CPU Software backend), the queue member
/// is a plain void* (always nullptr) so no SYCL runtime is linked into
/// the kernel .so.

#include <sycl-sandbox/profiler.h>
#include <cstddef>
#include <cstring>
#include <string>
#include <type_traits>

#ifndef KERNEL_NATIVE
#include <sycl/sycl.hpp>
#endif

// MemoryPool + Buffer live host-side; the SYCL device pass parses this
// header but never needs the pool's layout (device code never allocates),
// so the include is gated on __SYCL_DEVICE_ONLY__.  rt::Runtime carries
// the pool as an incomplete-type pointer member either way.
#ifndef __SYCL_DEVICE_ONLY__
#include <sycl-sandbox/kernel/memory.h>
#endif

namespace rt {

class MemoryPool;   ///< host-side allocation registry (defined in memory.h)
template <typename T> struct Buffer;  ///< owning handle (defined in memory.h)

/// Kernel execution context — memory allocator, data transfer, pixel
/// iteration primitive, profiler buffer, and queue access.
///
/// Every kernel receives a pointer to its Runtime via set_runtime().
/// The host (KernelRuntime) owns one instance and hands it to all
/// loaded kernels.
///
/// Usage in a kernel:
/// @code
///   #include <sycl-sandbox/kernel/execution_context.h>
///   static rt::Runtime *g_rt = nullptr;
///
///   extern "C" void set_runtime(rt::Runtime *rt) { g_rt = rt; }
///
///   extern "C" void render_kernel(void *, int w, int h, ...) {
///       // The kernel name tag must be unique per kernel .so — every
///       // kernel's entry point is named `render_kernel`, so without an
///       // explicit tag their per-pixel lambdas would mangle identically
///       // and AdaptiveCpp's kernel cache would confuse them.
///       g_rt->foreach_pixel<class MyKernelPixelTag>(w, h, [=](int x, int y, int idx) {
///           // pixel logic — must be by-value capturable (POD)
///       });
///   }
/// @endcode
struct Runtime {
#ifdef KERNEL_NATIVE
    void *queue = nullptr;         ///< nullptr in software mode
#else
    sycl::queue *queue = nullptr;  ///< valid in SYCL mode
#endif

    // ── Memory ────────────────────────────────────────────────────────
    /// Host-side allocation registry for all kernel-usable memory.  Set
    /// by the owner (KernelRuntime / diags / SceneDebugScene) to a
    /// `rt::MemoryPool` it owns.  Null when the Runtime is default
    /// constructed without an owner — the alloc methods then become no-ops
    /// / null returns, so owners MUST install a pool before allocating.
    /// Carried as an incomplete-type pointer so this header parses in the
    /// SYCL device pass (device code never allocates).
    MemoryPool *pool = nullptr;

    // ── Cancellation ────────────────────────────────────────────────
    /// Device-visible cancellation flag (USM shared allocation).
    /// The host sets this to 1 when it needs to abort in-flight kernel
    /// work (pause/resize/switch); kernels check it in hot loops and
    /// bail out early.  Reset to 0 at the start of each frame.
    /// Null when uninitialised.
    ///
    /// Plain int, NOT std::atomic: the pointer lives in USM shared
    /// memory, and std::atomic operations are not supported in device
    /// code on the CUDA backend.  Host and device both use the
    /// __atomic_* GCC/Clang builtins for relaxed access.
    int *cancel_flag = nullptr;

    /// True when the host has requested cancellation of the current
    /// frame.  Safe to call from device code (uses __atomic_load_n
    /// with relaxed ordering).
    bool cancelled() const {
        return cancel_flag &&
               __atomic_load_n(cancel_flag, __ATOMIC_RELAXED) != 0;
    }

    // ── Pixel iteration ──────────────────────────────────────────────
    // ENQUEUE-ONLY: in SYCL mode this submits the parallel_for to the
    // (in-order) queue and returns immediately — the host chains further
    // work (tone-map, display copy) after it and never blocks here.
    // In software/native mode execution is synchronous plain loops.
    //
    // KernelName is an explicit SYCL kernel name tag (a locally-declared
    // class, e.g. `class MyKernelPixelTag`) and MUST be unique per calling
    // kernel .so.  Without it, AdaptiveCpp derives the device kernel's
    // identity from the mangled name of the lambda's *enclosing function*
    // — and since every kernel .so implements its entry point in a
    // function literally called `render_kernel`, their per-pixel lambdas
    // all mangle to the exact same symbol.  AdaptiveCpp's SSCP kernel
    // cache then treats unrelated kernels from different .so's as the
    // same kernel, and switching between them can launch one kernel's
    // compiled binary against another's argument layout — observed as a
    // GPU MMU fault (Xid 31) when switching e.g. mandelbrot <-> cyber_fuji.
    template <typename KernelName, typename Fn>
    void foreach_pixel(int width, int height, Fn &&fn) const {
#ifdef KERNEL_NATIVE
        for (int y = 0; y < height; ++y) {
            if (__builtin_expect(cancelled(), 0)) break;
            for (int x = 0; x < width; ++x)
                fn(x, y, y * width + x);
        }
#else
        if (queue) {
            auto pixel_fn = std::forward<Fn>(fn);
            auto *cf = cancel_flag;
            queue->parallel_for<KernelName>(
                sycl::range<2>{(size_t)height, (size_t)width},
                [pixel_fn, width, cf](sycl::item<2> item) {
                    if (__builtin_expect(cf &&
                            __atomic_load_n(cf, __ATOMIC_RELAXED) != 0, 0))
                        return;
                    int x = item[1], y = item[0];
                    pixel_fn(x, y, y * width + x);
                });
            queue->wait();
        } else {
            for (int y = 0; y < height; ++y) {
                if (__builtin_expect(cancelled(), 0)) break;
                for (int x = 0; x < width; ++x)
                    fn(x, y, y * width + x);
            }
        }
#endif
    }

    /// Blocking variant for init-time work that must complete before the
    /// caller continues (e.g. one-off precomputation in init_kernel).
    /// See foreach_pixel() above for why KernelName must be unique.
    template <typename KernelName, typename Fn>
    void foreach_pixel_sync(int width, int height, Fn &&fn) const {
        foreach_pixel<KernelName>(width, height, std::forward<Fn>(fn));
#ifndef KERNEL_NATIVE
        if (queue) queue->wait();
#endif
    }

    // ── Memory ───────────────────────────────────────────────────────
    // All allocation/transfer/free calls delegate to the bound
    // `rt::MemoryPool` (host-side registry + drain-on-free + per-kind
    // byte counters).  Owners install the pool via `rt.runtime.pool`.
    // The kernel-facing surface (alloc_device/host, dealloc, make_input,
    // make_output, read_back, copy_to_*/fill) is unchanged from the
    // pre-pool API, so kernels and SceneBuilder keep working as-is.

    template <typename T>
    T *alloc_device(size_t count, MemScope scope = MemScope::Kernel) {
        if (!pool) return nullptr;
        return pool->alloc_device_raw<T>(count, scope);
    }

    template <typename T>
    T *alloc_host(size_t count, MemScope scope = MemScope::Kernel) {
        if (!pool) return nullptr;
        return pool->alloc_host_raw<T>(count, scope);
    }

    template <typename T>
    T *alloc_shared(size_t count, MemScope scope = MemScope::Kernel) {
        if (!pool) return nullptr;
        return pool->alloc_shared_raw<T>(count, scope);
    }

    template <typename T>
    void dealloc(T *ptr) {
        if (!pool || !ptr) return;
        pool->release(static_cast<void *>(ptr));
    }

    template <typename T>
    void dealloc_sized(T *ptr, size_t /*count*/, bool /*is_device*/) {
        if (!pool || !ptr) return;
        pool->release(static_cast<void *>(ptr));
    }

    // ── Transfers ────────────────────────────────────────────────────

    void copy_to_device(void *dst, const void *src, size_t bytes) {
        if (pool) pool->copy_to_device(dst, src, bytes);
    }

    void copy_to_host(void *dst, const void *src, size_t bytes) {
        if (pool) pool->copy_to_host(dst, src, bytes);
    }

    void fill(void *ptr, int val, size_t bytes) {
        if (pool) pool->fill(ptr, val, bytes);
    }

    // ── Generic buffers (kernel-facing, Frame scope) ───────────────
    // One-shot input/output buffers for kernels: upload a host array
    // (texture data, lookup tables, LUTs) as a device buffer, or
    // allocate an output buffer for per-pixel results.  The returned
    // rt::Buffer<T> OWNS the memory and frees it back to the pool on
    // destruction — never dealloc() it manually.  Allocated under
    // MemScope::Frame so the per-call clean check enforces the
    // no-state-between-calls contract.

    /// Allocate a device (heap in software mode) buffer and copy the
    /// host data into it.  Returns an empty buffer when count is 0.
    template <typename T>
    Buffer<T> make_input(const T *host_data, size_t count) {
        if (!pool || count == 0) return {};
        auto buf = pool->alloc_device<T>(count, MemScope::Frame);
        if (buf.data) pool->copy_to_device(static_cast<void *>(buf.data),
                                           host_data, count * sizeof(T));
        return buf;
    }

    /// Allocate a zero-initialized output buffer.  Returns an empty
    /// buffer when count is 0.
    template <typename T>
    Buffer<T> make_output(size_t count) {
        if (!pool || count == 0) return {};
        auto buf = pool->alloc_device<T>(count, MemScope::Frame);
        if (buf.data) pool->fill(static_cast<void *>(buf.data), 0,
                                 count * sizeof(T));
        return buf;
    }

    /// Copy a buffer back to host memory (synchronous).  No-op when the
    /// buffer is empty.
    template <typename T>
    void read_back(const Buffer<T> &buffer, T *dst) const {
        if (!buffer.data) return;
        copy_to_host(dst, buffer.data, buffer.count * sizeof(T));
    }

    std::string device_name() const {
#ifndef KERNEL_NATIVE
        if (queue) return queue->get_device().get_info<sycl::info::device::name>();
#endif
        return "CPU (Software)";
    }

    /// ── Kernel profiler buffer ─────────────────────────────────────
    /// Pointer to the profiler ring buffer allocated by the host.
    /// Kernels can install it via set_profiler_buffer() for Tracy-style
    /// GPU device-side profiling zones.
    void *profiler_buffer = nullptr;
};

} // namespace rt