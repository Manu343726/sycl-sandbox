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

template <typename T> struct Buffer;   ///< defined below (owns Runtime memory)

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

    // ── Generic buffers ────────────────────────────────────────────
    // One-shot input/output buffers for kernels: upload a host array
    // (texture data, lookup tables, LUTs) as a device buffer, or
    // allocate an output buffer for per-pixel results.  The returned
    // rt::Buffer<T> OWNS the memory and frees it through this Runtime
    // on destruction — never dealloc() it manually.

    /// Allocate a device (heap in software mode) buffer and copy the
    /// host data into it.  Returns an empty buffer when count is 0.
    template <typename T>
    Buffer<T> make_input(const T *host_data, size_t count) {
        if (count == 0) return {};
        T *ptr = alloc_device<T>(count);
        copy_to_device(ptr, host_data, count * sizeof(T));
        return Buffer<T>(this, ptr, count);
    }

    /// Allocate a zero-initialized output buffer.  Returns an empty
    /// buffer when count is 0.
    template <typename T>
    Buffer<T> make_output(size_t count) {
        if (count == 0) return {};
        T *ptr = alloc_device<T>(count);
        fill(ptr, 0, count * sizeof(T));
        return Buffer<T>(this, ptr, count);
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

/// Owning device (or heap, in software mode) buffer allocated through a
/// rt::Runtime via make_input() / make_output().  Frees its memory back
/// to the Runtime on destruction.  Move-only — copy ownership is
/// meaningless for a raw device pointer.
///
/// Usage in a kernel:
/// @code
///   auto palette = rt->make_input<float3>(host_colors, num_colors);
///   ...
///   // palette.data is a device pointer, usable inside parallel_for
///   // lambdas (captured by value); the buffer frees itself on scope
///   // exit, or can be released early with release().
/// @endcode
template <typename T>
struct Buffer {
    Runtime *runtime = nullptr;
    T *data = nullptr;
    size_t count = 0;

    Buffer() = default;
    Buffer(Runtime *rt, T *ptr, size_t n) : runtime(rt), data(ptr), count(n) {}

    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;

    Buffer(Buffer &&other) noexcept
        : runtime(other.runtime), data(other.data), count(other.count) {
        other.runtime = nullptr;
        other.data = nullptr;
        other.count = 0;
    }
    Buffer &operator=(Buffer &&other) noexcept {
        if (this != &other) {
            release();
            runtime = other.runtime;
            data = other.data;
            count = other.count;
            other.runtime = nullptr;
            other.data = nullptr;
            other.count = 0;
        }
        return *this;
    }

    ~Buffer() { release(); }

    T &operator[](size_t index) { return data[index]; }
    const T &operator[](size_t index) const { return data[index]; }

    explicit operator bool() const { return data != nullptr; }

    /// Return the memory to the Runtime and reset to empty.
    void release() {
        if (runtime && data) runtime->dealloc(data);
        runtime = nullptr;
        data = nullptr;
        count = 0;
    }
};

} // namespace rt