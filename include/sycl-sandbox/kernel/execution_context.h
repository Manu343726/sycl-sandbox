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
#include <atomic>

#ifndef KERNEL_NATIVE
#include <sycl/sycl.hpp>
#endif

namespace rt {

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

    // ── Memory tracking counters ────────────────────────────────────
    std::atomic<int64_t> device_memory_used{0};
    std::atomic<int64_t> host_memory_used{0};
    int64_t peak_device_memory{0};
    int64_t peak_host_memory{0};

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
        PROFILER_ZONE("foreach_pixel");
#ifdef KERNEL_NATIVE
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                fn(x, y, y * width + x);
#else
        if (queue) {
            auto pixel_fn = std::forward<Fn>(fn);
            queue->parallel_for<KernelName>(
                sycl::range<2>{(size_t)height, (size_t)width},
                [pixel_fn, width](sycl::item<2> item) {
                    int x = item[1], y = item[0];
                    pixel_fn(x, y, y * width + x);
                });
        } else {
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    fn(x, y, y * width + x);
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

    template <typename T>
    T *alloc_device(size_t count) {
        size_t bytes = count * sizeof(T);
        T *ptr;
#ifndef KERNEL_NATIVE
        if (queue) { ptr = sycl::malloc_device<T>(count, *queue); }
        else
#endif
        { ptr = new T[count](); }
        device_memory_used += bytes;
        if (device_memory_used > peak_device_memory)
            peak_device_memory = device_memory_used;
        // TODO: uncomment when profiler works
        // PROFILER_ALLOC(ptr, bytes);
        // PROFILER_PLOT("Device memory", (float)device_memory_used.load());
        return ptr;
    }

    template <typename T>
    T *alloc_host(size_t count) {
        size_t bytes = count * sizeof(T);
        T *ptr;
#ifndef KERNEL_NATIVE
        if (queue) { ptr = sycl::malloc_host<T>(count, *queue); }
        else
#endif
        { ptr = new T[count](); }
        host_memory_used += bytes;
        if (host_memory_used > peak_host_memory)
            peak_host_memory = host_memory_used;
        // TODO: uncomment when profiler works
        // PROFILER_ALLOC(ptr, bytes);
        // PROFILER_PLOT("Host memory", (float)host_memory_used.load());
        return ptr;
    }

    template <typename T>
    void dealloc(T *ptr) {
        if (!ptr) return;
        // TODO: uncomment when profiler works
        // PROFILER_FREE(ptr);
#ifndef KERNEL_NATIVE
        if (queue) { sycl::free(ptr, *queue); return; }
#endif
        delete[] ptr;
    }

    template <typename T>
    void dealloc_sized(T *ptr, size_t count, bool is_device) {
        if (!ptr) return;
        size_t bytes = count * sizeof(T);
        if (is_device) {
            device_memory_used -= bytes;
            // TODO: uncomment when profiler works
            // PROFILER_PLOT("Device memory", (float)device_memory_used.load());
        } else {
            host_memory_used -= bytes;
            // TODO: uncomment when profiler works
            // PROFILER_PLOT("Host memory", (float)host_memory_used.load());
        }
        // TODO: uncomment when profiler works
        // PROFILER_FREE(ptr);
#ifndef KERNEL_NATIVE
        if (queue) { sycl::free(ptr, *queue); return; }
#endif
        delete[] ptr;
    }

    void event(const char *msg) const { PROFILER_MSG(msg); }

    // ── Transfers ────────────────────────────────────────────────────

    void copy_to_device(void *dst, const void *src, size_t bytes) {
#ifndef KERNEL_NATIVE
        if (queue) { queue->memcpy(dst, src, bytes).wait(); return; }
#endif
        std::memcpy(dst, src, bytes);
    }

    void copy_to_host(void *dst, const void *src, size_t bytes) {
#ifndef KERNEL_NATIVE
        if (queue) { queue->memcpy(dst, src, bytes).wait(); return; }
#endif
        std::memcpy(dst, src, bytes);
    }

    void fill(void *ptr, int val, size_t bytes) {
#ifndef KERNEL_NATIVE
        if (queue) { queue->memset(ptr, val, bytes).wait(); return; }
#endif
        std::memset(ptr, val, bytes);
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