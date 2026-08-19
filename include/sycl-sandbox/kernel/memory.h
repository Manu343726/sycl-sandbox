#pragma once

/// @file
/// The kernel runtime's single allocation system.
///
/// Every kernel-usable buffer is allocated through one `rt::MemoryPool`,
/// which owns a registry of live allocations, per-kind byte counters, and
/// the drain protocol.  There is no second allocator path.
///
/// See docs/architecture.md → "Memory system" for the design.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#ifndef KERNEL_NATIVE
#include <sycl/sycl.hpp>
#endif

namespace rt {

class MemoryPool;

template <typename T> struct Buffer;   // owning handle (defined below)

// ── Locality ───────────────────────────────────────────────────────────

/// Where the bytes of an allocation live and which sides can touch them.
enum class MemoryKind : uint8_t {
    Device, ///< sycl::malloc_device  — device only; host copies via queue
    Host,   ///< sycl::malloc_host    — host-resident, device-visible (no copy)
    Shared, ///< sycl::malloc_shared  — every side accesses the same pointer
};

// ── Scope ──────────────────────────────────────────────────────────────

/// Lifetime class of an allocation.  Drives the per-call Frame-scope
/// enforcement; `release()`/`release_all()` free any scope.
///
///   App     — process-lifetime buffers owned by the app (display, rings)
///   Kernel  — app-allocated on the kernel's behalf (accum, params, scene)
///   Frame   — kernel host-side per-call buffers; MUST be empty when
///             kernel_entry returns (checked by MemoryPool::check_frame_clean)
enum class MemScope : uint8_t { App, Kernel, Frame };

// ── MemoryPool ─────────────────────────────────────────────────────────

/// Single allocator for all kernel-usable memory.  Bound to one queue
/// (null in software/native mode, where every op is a plain `new`/`memcpy`).
///
/// Owns the registry of live allocations.  `release()` and `release_all()`
/// always drain the queue first — the in-order queue plus the blocking
/// kernel (`foreach_pixel` waits internally) makes this free in steady state
/// but eliminates the footgun of freeing a buffer an enqueued op references.
///
/// Host-only: the SYCL device pass never instantiates this class
/// (`rt::Runtime` carries it as an incomplete-type pointer member, so the
/// device compiler never needs its layout).
class MemoryPool {
public:
    MemoryPool() = default;
    explicit MemoryPool(void *q) : queue_(q) {}
    ~MemoryPool() { release_all(); }

    MemoryPool(const MemoryPool &) = delete;
    MemoryPool &operator=(const MemoryPool &) = delete;
    MemoryPool(MemoryPool &&) = delete;
    MemoryPool &operator=(MemoryPool &&) = delete;

    /// Rebind to a new queue (after a backend switch).  Call `release_all()`
    /// on the OLD binding first — `sycl::free` must use the queue that
    /// allocated.
    void bind(void *q) { queue_ = q; }
    void *queue() const { return queue_; }

    // ── Typed allocation ───────────────────────────────────────────
    /// Owning allocation — returns a `Buffer<T>` whose destructor frees
    /// the memory back to this pool.  Use this when the caller stores the
    /// result in an `rt::Buffer` member.
    template <typename T>
    Buffer<T> alloc(MemoryKind kind, size_t count,
                    MemScope scope = MemScope::Kernel) {
        if (count == 0) return {};
        size_t bytes = count * sizeof(T);
        void *ptr = alloc_raw(kind, bytes, scope);
        if (!ptr) return {};
        return Buffer<T>(this, static_cast<T *>(ptr), count, kind);
    }

    template <typename T> Buffer<T> alloc_device(size_t n, MemScope s = MemScope::Kernel) {
        return alloc<T>(MemoryKind::Device, n, s);
    }
    template <typename T> Buffer<T> alloc_host(size_t n, MemScope s = MemScope::Kernel) {
        return alloc<T>(MemoryKind::Host, n, s);
    }
    template <typename T> Buffer<T> alloc_shared(size_t n, MemScope s = MemScope::Kernel) {
        return alloc<T>(MemoryKind::Shared, n, s);
    }

    /// Raw-pointer allocation — registers the entry (freed via
    /// `release(void*)`) but returns a borrowed `T*` with NO owning
    /// handle.  Use this for the legacy `rt::Runtime::alloc_device` API
    /// where the caller holds a raw pointer and frees it through
    /// `dealloc`.  Never call this when you can use the owning
    /// `alloc_device<T>` overload instead.
    template <typename T>
    T *alloc_device_raw(size_t count, MemScope scope = MemScope::Kernel) {
        return static_cast<T *>(alloc_raw(MemoryKind::Device, count * sizeof(T), scope));
    }
    template <typename T>
    T *alloc_host_raw(size_t count, MemScope scope = MemScope::Kernel) {
        return static_cast<T *>(alloc_raw(MemoryKind::Host, count * sizeof(T), scope));
    }
    template <typename T>
    T *alloc_shared_raw(size_t count, MemScope scope = MemScope::Kernel) {
        return static_cast<T *>(alloc_raw(MemoryKind::Shared, count * sizeof(T), scope));
    }

    /// Allocate a Device buffer and copy `host` into it (one-shot input).
    template <typename T>
    Buffer<T> upload(const T *host, size_t count,
                     MemScope scope = MemScope::Kernel) {
        auto buf = alloc_device<T>(count, scope);
        if (buf.data) copy_to_device(buf.data, host, count * sizeof(T));
        return buf;
    }

    /// Copy a buffer back to host memory (synchronous).  No-op when empty.
    template <typename T>
    void read(const Buffer<T> &buf, T *dst) const {
        if (!buf.data) return;
        copy_to_host(dst, buf.data, buf.count * sizeof(T));
    }

    // ── Raw byte ops (queue-backed; memcpy/memset in software mode) ──
    void copy_to_device(void *dst, const void *src, size_t bytes) {
#ifndef KERNEL_NATIVE
        if (queue_) { static_cast<sycl::queue *>(queue_)->memcpy(dst, src, bytes).wait(); return; }
#endif
        std::memcpy(dst, src, bytes);
    }
    void copy_to_host(void *dst, const void *src, size_t bytes) {
#ifndef KERNEL_NATIVE
        if (queue_) { static_cast<sycl::queue *>(queue_)->memcpy(dst, src, bytes).wait(); return; }
#endif
        std::memcpy(dst, src, bytes);
    }
    void fill(void *ptr, int val, size_t bytes) {
#ifndef KERNEL_NATIVE
        if (queue_) { static_cast<sycl::queue *>(queue_)->memset(ptr, val, bytes).wait(); return; }
#endif
        std::memset(ptr, val, bytes);
    }

    // ── Teardown ───────────────────────────────────────────────────
    /// Wait for all enqueued work on the bound queue.  No-op in software.
    void drain() {
#ifndef KERNEL_NATIVE
        if (queue_) {
            try { static_cast<sycl::queue *>(queue_)->wait(); }
            catch (...) { /* swallow — teardown path */ }
        }
#endif
    }

    /// Free one allocation (looked up by pointer).  Drains first.  No-op
    /// for an unknown/null pointer (safe to call on a released buffer).
    void release(void *ptr) {
        if (!ptr) return;
        drain();
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].ptr == ptr) {
                raw_free(entries_[i]);
                bytes_[(int)entries_[i].kind] -= (int64_t)entries_[i].size;
                entries_[i] = entries_.back();
                entries_.pop_back();
                return;
            }
        }
        // Unknown pointer — not in the registry.  Silent no-op so stale
        // Buffer handles and explicit release() calls are idempotent.
    }

    /// Free every allocation (any scope).  Drains first.  Used for backend
    /// switch / shutdown.
    void release_all() {
        drain();
        for (auto &e : entries_) {
            raw_free(e);
        }
        entries_.clear();
        bytes_[0] = bytes_[1] = bytes_[2] = 0;
    }

    // ── Introspection ─────────────────────────────────────────────
    bool empty() const { return entries_.empty(); }
    /// True when `ptr` is currently registered with this pool (allocated
    /// through it and not yet released).  Used to detect STALE Buffer
    /// handles after `release_all()` (e.g. a pool-rebound backend switch
    /// frees every allocation but leaves caller-held rt::Buffer members
    /// pointing at the freed memory).
    bool owns(const void *ptr) const {
        if (!ptr) return false;
        for (const auto &e : entries_)
            if (e.ptr == ptr) return true;
        return false;
    }
    int64_t bytes(MemoryKind k) const { return bytes_[(int)k]; }
    int64_t device_bytes() const { return bytes_[(int)MemoryKind::Device]; }
    int64_t host_bytes() const { return bytes_[(int)MemoryKind::Host] + bytes_[(int)MemoryKind::Shared]; }

    /// Per-call Frame-scope enforcement: free any Frame-tagged leftovers and
    /// return how many were found.  Zero means the kernel honoured the
    /// no-state-between-calls contract; non-zero is a kernel bug (logged +
    /// freed self-healing by the caller).
    size_t check_frame_clean() {
        size_t leaked = 0;
        drain();
        for (size_t i = 0; i < entries_.size(); ) {
            if (entries_[i].scope == MemScope::Frame) {
                ++leaked;
                raw_free(entries_[i]);
                bytes_[(int)entries_[i].kind] -= (int64_t)entries_[i].size;
                entries_[i] = entries_.back();
                entries_.pop_back();
            } else {
                ++i;
            }
        }
        return leaked;
    }

private:
    struct Entry {
        void *ptr = nullptr;
        size_t size = 0;
        MemoryKind kind = MemoryKind::Device;
        MemScope scope = MemScope::Kernel;
    };

    /// Allocate `bytes` of `kind`, register the entry, and return the
    /// raw pointer.  Ownership stays in the registry — free via
    /// `release(void*)` (or wrap the pointer in a `Buffer<T>` at the
    /// call site for RAII).
    void *alloc_raw(MemoryKind kind, size_t bytes, MemScope scope) {
        if (bytes == 0) return nullptr;
        void *ptr = raw_alloc(kind, bytes);
        if (!ptr) return nullptr;
        entries_.push_back(Entry{ptr, bytes, kind, scope});
        bytes_[(int)kind] += (int64_t)bytes;
        return ptr;
    }

    void *raw_alloc(MemoryKind kind, size_t bytes) {
#ifndef KERNEL_NATIVE
        if (queue_) {
            auto *q = static_cast<sycl::queue *>(queue_);
            switch (kind) {
                case MemoryKind::Device: return sycl::malloc_device(bytes, *q);
                case MemoryKind::Host:   return sycl::malloc_host(bytes, *q);
                case MemoryKind::Shared: return sycl::malloc_shared(bytes, *q);
            }
        }
#endif
        // Software mode: raw byte array + zero-fill (mirrors the old
        // `new T[count]()` value-initialisation the native path relied on).
        void *p = ::operator new(bytes);
        std::memset(p, 0, bytes);
        return p;
    }

    void raw_free(const Entry &e) {
#ifndef KERNEL_NATIVE
        if (queue_) { sycl::free(e.ptr, *static_cast<sycl::queue *>(queue_)); return; }
#endif
        ::operator delete(e.ptr);
    }

    void *queue_ = nullptr;       ///< sycl::queue*, null in software mode
    std::vector<Entry> entries_;  ///< registry (host-only)
    int64_t bytes_[3] = {0, 0, 0};
};

// ── Buffer<T> — owning handle ──────────────────────────────────────────

/// Owning view of `count` elements of `T` allocated through a `MemoryPool`.
/// Move-only; destruction frees the memory back to the pool (drains first).
/// A raw pointer obtained via `data`/`get` is a *borrowed view* — valid only
/// while this Buffer lives and never freed manually.
template <typename T>
struct Buffer {
    MemoryPool *pool = nullptr;
    T *data = nullptr;
    size_t count = 0;
    MemoryKind kind = MemoryKind::Device;

    Buffer() = default;
    Buffer(MemoryPool *p, T *ptr, size_t n, MemoryKind k)
        : pool(p), data(ptr), count(n), kind(k) {}

    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;

    Buffer(Buffer &&o) noexcept
        : pool(o.pool), data(o.data), count(o.count), kind(o.kind) {
        o.pool = nullptr; o.data = nullptr; o.count = 0;
    }
    Buffer &operator=(Buffer &&o) noexcept {
        if (this != &o) {
            release();
            pool = o.pool; data = o.data; count = o.count; kind = o.kind;
            o.pool = nullptr; o.data = nullptr; o.count = 0;
        }
        return *this;
    }

    ~Buffer() { release(); }

    T &operator[](size_t i) { return data[i]; }
    const T &operator[](size_t i) const { return data[i]; }
    explicit operator bool() const { return data != nullptr; }
    T *get() const { return data; }

    /// Return the memory to the pool and reset to empty.  Idempotent.
    void release() {
        if (pool && data) pool->release(static_cast<void *>(data));
        pool = nullptr; data = nullptr; count = 0;
    }

    /// Host → this buffer (copy `n` elements from `src`).
    void write(const T *src, size_t n) {
        if (!data || n == 0) return;
        n = (n < count) ? n : count;
        pool->copy_to_device(static_cast<void *>(data), src, n * sizeof(T));
    }

    /// This buffer → host (copy `n` elements to `dst`).
    void read(T *dst, size_t n) const {
        if (!data || n == 0) return;
        n = (n < count) ? n : count;
        pool->copy_to_host(dst, static_cast<void *>(data), n * sizeof(T));
    }
};

} // namespace rt