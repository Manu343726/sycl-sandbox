#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>

// ── Profiler configuration ────────────────────────────────────────────
// Profiling is always compiled in.  It can be disabled at runtime via
// profiler::set_enabled(false).
//
// Kernel code uses the device ring buffer (KERNEL_BUILD defined).
// Host code uses a thread-local ring buffer for zone events.

namespace profiler {

// ── Constants ─────────────────────────────────────────────────────────
static constexpr int MAX_RECORDS      = 1048576;  ///< kernel records per frame (1M)
static constexpr int MAX_HOST_RECORDS = 65536;  ///< host records between collects
static constexpr int MAX_NAME_LEN     = 48;

// ── Kernel record (shared type: written by kernel, read by host) ──────
/// A single profiling event (zone enter or exit) from kernel code.
/// Packed to 64 bytes for alignment-friendly buffer.
/// Record type values.
enum RecordType : uint8_t {
    REC_ZONE_ENTER = 0,  ///< Zone started (name is zone name)
    REC_ZONE_EXIT  = 1,  ///< Zone ended
    REC_MSG        = 2,  ///< Text message (name is message text)
    REC_PLOT       = 3,  ///< Named plot value (name is plot name, zone_id holds float bits)
    REC_ALLOC      = 4,  ///< Allocation tracking
    REC_FREE       = 5,  ///< Deallocation tracking
};

struct alignas(64) Record {
    uint64_t timestamp;   ///< cycles (rdtsc) or ns
    uint32_t zone_id;     ///< zone id / plot value bits
    uint8_t type;         ///< RecordType
    uint8_t _pad[3];
    char name[MAX_NAME_LEN]; ///< zone / message / plot name, null-terminated

    void set_name(const char *n) {
        for (int i = 0; i < MAX_NAME_LEN - 1 && n[i]; i++)
            name[i] = n[i];
        name[MAX_NAME_LEN - 1] = '\0';
    }

    /// For REC_PLOT records: extract the float value from zone_id.
    float plot_value() const {
        float v;
        memcpy(&v, &zone_id, sizeof(v));
        return v;
    }
    /// For REC_PLOT records: store a float value into zone_id.
    void set_plot_value(float v) {
        memcpy(&zone_id, &v, sizeof(v));
    }
};

/// Buffer header (at the start of the kernel profile buffer).
/// The record array is a ring buffer: device atomically increments write_pos
/// to claim the next slot, then writes to records[slot % MAX_RECORDS].
/// Oldest records are overwritten — newest records always survive overflow.
struct BufferHeader {
    volatile uint32_t write_pos;     ///< ring buffer write position (atomic fetch_add)
    uint32_t max_records;            ///< ring capacity (set by init, must be power of two)
    uint32_t sample_interval;        ///< 0 = record all, N = record 1 out of N zones
    uint32_t sample_counter;         ///< incremented for each zone (atomic on device)
};

// ── Host record ───────────────────────────────────────────────────────
struct HostRecord {
    uint64_t timestamp_ns;   ///< nanoseconds (steady_clock)
    uint64_t end_ns;         ///< end timestamp (0 for non-zone records)
    uint32_t id;             ///< zone id or 0
    uint8_t  type;           ///< 0=zone_enter, 1=zone_exit, 2=msg, 3=plot, 4=alloc, 5=free
    uint8_t  _pad[3];
    char     name[MAX_NAME_LEN];
    float    plot_value;

    void set_name(const char *n) {
        for (int i = 0; i < MAX_NAME_LEN - 1 && n[i]; i++) name[i] = n[i];
        name[MAX_NAME_LEN - 1] = '\0';
    }
};

// ── Result (for reading collected records) ────────────────────────────
struct ZoneEntry {
    std::string name;
    uint64_t start_ns;
    uint64_t end_ns;
    int depth;
};


// ── Platform dispatch ─────────────────────────────────────────────────
#ifdef KERNEL_BUILD

// ── Kernel-side implementation (host-side ring buffer access) ───────
//
// CAUTION: The global variables below (g_records, g_write_pos, etc.) live
// in the kernel .so's data segment.  Inside SYCL parallel_for lambdas
// running on GPU, these globals are NOT accessible — accessing them causes
// CUDA error 700 (illegal address).  All record functions are guarded with
// !__SYCL_DEVICE_ONLY__ so they become no-ops in SYCL device code.
// Host code within the .so (init_kernel, render_kernel entry point, etc.)
// still records correctly because __SYCL_DEVICE_ONLY__ is not defined there.

/// Global profiler state (plain pointers, set before each frame).
/// CAUTION: These globals live in the kernel .so's data segment.  They are
/// set by set_buffer() called from render_kernel() (host code in the .so).
/// SYCL device code MUST NOT access them — use __SYCL_DEVICE_ONLY__ guards.
inline Record *g_records = nullptr;
inline volatile uint32_t *g_write_pos = nullptr;
inline uint32_t g_max_records = 0;
inline uint32_t *g_sample_interval = nullptr;
inline uint32_t *g_sample_counter = nullptr;

/// Set the profiler output buffer.  Only call from HOST-side code in the
/// kernel .so (e.g., the render_kernel entry point, before parallel_for).
/// Never call from inside a SYCL parallel_for lambda.
inline void set_buffer(void *buffer, uint32_t max_records) {
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(__HIP_DEVICE_COMPILE__) && \
    !(defined(__CUDACC__) && !defined(__HOST__))
    if (!buffer) {
        g_records = nullptr;
        g_write_pos = nullptr;
        g_max_records = 0;
        g_sample_interval = nullptr;
        g_sample_counter = nullptr;
        return;
    }
    auto *header = static_cast<profiler::BufferHeader *>(buffer);
    header->write_pos = 0;
    header->max_records = max_records;
    header->sample_interval = 0;
    header->sample_counter = 0;
    g_write_pos = &header->write_pos;
    g_sample_interval = &header->sample_interval;
    g_sample_counter = &header->sample_counter;
    g_records = reinterpret_cast<profiler::Record *>(header + 1);
    g_max_records = max_records;
#else
    (void)buffer;
    (void)max_records;
#endif
}

/// Set the zone sample interval for sub-sampling.
/// interval=0 (default): record every zone.
/// interval=N: record only 1 out of every N zones.
/// Safe to call from host code only (not from SYCL device lambdas).
inline void set_sample_interval(uint32_t interval) {
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(__HIP_DEVICE_COMPILE__) && \
    !(defined(__CUDACC__) && !defined(__HOST__))
    if (g_sample_interval) *g_sample_interval = interval;
    if (g_sample_counter)  *g_sample_counter = 0;
#else
    (void)interval;
#endif
}

/// Portable timestamp.
///
/// Priority order:
///   1. __builtin_readcyclecounter — available in both Clang (host & CUDA
///      device) and GCC.  Lowers to rdtsc on x86 host and clock64 on CUDA.
///   2. x86 rdtsc inline asm — fallback for older compilers without the builtin.
///   3. ARM cntvct_el0 — fallback for aarch64.
///   4. 0 — unsupported architecture.
inline uint64_t read_timestamp() {
#if defined(__has_builtin) && __has_builtin(__builtin_readcyclecounter)
    // Clang (host x86 + CUDA device) and GCC both support this.
    // On x86 host → rdtsc; on CUDA device → clock64(); on ARM → cntvct_el0.
    return __builtin_readcyclecounter();
#elif defined(__x86_64__) || defined(__i386__)
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#elif defined(__aarch64__)
    uint64_t cnt;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(cnt));
    return cnt;
#else
    return 0;
#endif
}

/// Append a record to the profiler ring buffer (atomic slot claim).
/// Uses __atomic_fetch_add for both host and SYCL device safety.
/// When the ring wraps, the oldest record at the target slot is overwritten.
///
/// WARNING: Only call from HOST-side code (init_kernel, render_kernel entry).
/// Inside SYCL parallel_for lambdas (GPU device code), the global variables
/// g_records / g_write_pos are NOT accessible — this becomes a no-op.
inline void add_record(uint32_t zone_id, uint8_t type,
                        const char *name = nullptr) {
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(__HIP_DEVICE_COMPILE__) && \
    !(defined(__CUDACC__) && !defined(__HOST__))
    if (!g_records || !g_write_pos) return;
    uint32_t idx = __atomic_fetch_add(g_write_pos, 1, __ATOMIC_RELAXED);
    uint32_t slot = idx % MAX_RECORDS;

    auto &rec = g_records[slot];
    rec.timestamp = read_timestamp();
    rec.zone_id = zone_id;
    rec.type = type;
    if (name) {
        rec.set_name(name);
    } else {
        rec.name[0] = '\0';
    }
#else
    (void)zone_id;
    (void)type;
    (void)name;
#endif
}

/// FNV-1a hash for deterministic zone_id from a name string.
inline uint32_t hash_zone_name(const char *name) {
    uint32_t h = 2166136261u;
    for (const char *p = name; *p; ++p) {
        h ^= (uint8_t)*p;
        h *= 16777619u;
    }
    return h;
}

/// RAII zone — records enter on construction, exit on destruction.
/// Supports sub-sampling via g_sample_interval: when the counter fires,
/// both enter and exit are skipped atomically.
///
/// WARNING: Use ONLY from host-side code in the kernel .so.  Inside
/// SYCL parallel_for lambdas (GPU device code), Zone becomes a no-op
/// because the profiler global variables are not accessible from GPU.
class Zone {
public:
    Zone(const char *name)
        : zone_id_(hash_zone_name(name)), name_(name) {
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(__HIP_DEVICE_COMPILE__) && \
    !(defined(__CUDACC__) && !defined(__HOST__))
        // Sub-sampling: only record if the counter hits 0
        uint32_t interval = g_sample_interval ? *g_sample_interval : 0;
        if (interval > 0) {
            uint32_t c = __atomic_fetch_add(g_sample_counter, 1, __ATOMIC_RELAXED);
            if (c % interval != 0) {
                recorded_ = false;
                return;
            }
        }
        add_record(zone_id_, REC_ZONE_ENTER, name);
#else
        (void)name;
        recorded_ = false;
#endif
    }
    ~Zone() {
        if (!recorded_) return;
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(__HIP_DEVICE_COMPILE__) && \
    !(defined(__CUDACC__) && !defined(__HOST__))
        add_record(zone_id_, REC_ZONE_EXIT);
#endif
    }
    Zone(const Zone &) = delete;
    Zone &operator=(const Zone &) = delete;
    bool recorded_ = true;

private:
    uint32_t zone_id_;
    const char *name_;
};

/// Helper: atomically claim a record index (ring buffer).
inline uint32_t claim_record() {
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(__HIP_DEVICE_COMPILE__) && \
    !(defined(__CUDACC__) && !defined(__HOST__))
    if (!g_records || !g_write_pos) return ~0u;
    return __atomic_fetch_add(g_write_pos, 1, __ATOMIC_RELAXED);
#else
    return ~0u;
#endif
}

/// Emit a text message into the profiler buffer.
inline void message(const char *text) {
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(__HIP_DEVICE_COMPILE__) && \
    !(defined(__CUDACC__) && !defined(__HOST__))
    uint32_t idx = claim_record();
    uint32_t slot = idx % MAX_RECORDS;
    auto &rec = g_records[slot];
    rec.timestamp = read_timestamp();
    rec.type = REC_MSG;
    rec.zone_id = 0;
    rec.set_name(text ? text : "");
#else
    (void)text;
#endif
}

/// Emit a named plot value into the profiler buffer.
inline void plot(const char *name, float value) {
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(__HIP_DEVICE_COMPILE__) && \
    !(defined(__CUDACC__) && !defined(__HOST__))
    uint32_t idx = claim_record();
    uint32_t slot = idx % MAX_RECORDS;
    auto &rec = g_records[slot];
    rec.timestamp = read_timestamp();
    rec.type = REC_PLOT;
    rec.set_plot_value(value);
    rec.set_name(name ? name : "");
#else
    (void)name;
    (void)value;
#endif
}

/// Mark a heap allocation.
inline void alloc(const void *ptr, size_t size) {
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(__HIP_DEVICE_COMPILE__) && \
    !(defined(__CUDACC__) && !defined(__HOST__))
    uint32_t idx = claim_record();
    uint32_t slot = idx % MAX_RECORDS;
    auto &rec = g_records[slot];
    rec.timestamp = read_timestamp();
    rec.type = REC_ALLOC;
    rec.zone_id = 0;
    char buf[MAX_NAME_LEN];
    snprintf(buf, sizeof(buf), "%p %zu", ptr, size);
    rec.set_name(buf);
#else
    (void)ptr;
    (void)size;
#endif
}

/// Mark a heap deallocation.
inline void free(const void *ptr) {
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(__HIP_DEVICE_COMPILE__) && \
    !(defined(__CUDACC__) && !defined(__HOST__))
    uint32_t idx = claim_record();
    uint32_t slot = idx % MAX_RECORDS;
    auto &rec = g_records[slot];
    rec.timestamp = read_timestamp();
    rec.type = REC_FREE;
    rec.zone_id = 0;
    char buf[MAX_NAME_LEN];
    snprintf(buf, sizeof(buf), "%p", ptr);
    rec.set_name(buf);
#else
    (void)ptr;
#endif
}

#else
// ── Host-side implementation (mutex-guarded ring buffer) ──────────

inline std::mutex &host_mutex() {
    static std::mutex m;
    return m;
}
inline HostRecord *host_records() {
    static HostRecord buf[MAX_HOST_RECORDS];
    return buf;
}
inline std::atomic<uint32_t> &host_count() {
    static std::atomic<uint32_t> c{0};
    return c;
}
inline std::atomic<bool> &enabled() {
    static std::atomic<bool> e{true};
    return e;
}

inline void set_enabled(bool e) { enabled().store(e); }
inline bool is_enabled() { return enabled().load(); }

/// Collect all records since last call (returns and clears).
inline std::vector<HostRecord> collect() {
    std::lock_guard<std::mutex> lock(host_mutex());
    uint32_t n = host_count().exchange(0);
    if (n > MAX_HOST_RECORDS) n = MAX_HOST_RECORDS;
    std::vector<HostRecord> out(host_records(), host_records() + n);
    return out;
}

inline uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// Push a record (lock-free-ish: mutex-guarded).
inline void push_record(uint8_t type, const char *name, float val = 0) {
    if (!is_enabled()) return;
    std::lock_guard<std::mutex> lock(host_mutex());
    uint32_t idx = host_count().fetch_add(1);
    if (idx >= MAX_HOST_RECORDS) { host_count().store(MAX_HOST_RECORDS); return; }
    auto &r = host_records()[idx];
    r.timestamp_ns = now_ns();
    r.end_ns = 0;
    r.type = type;
    r.plot_value = val;
    r.id = 0;
    r.set_name(name ? name : "");
}

/// RAII zone.
struct Zone {
    Zone(const char *name = nullptr) {
        if (!is_enabled()) return;
        push_record(REC_ZONE_ENTER, name ? name : "unnamed");
        active_ = true;
    }
    ~Zone() {
        if (!active_) return;
        push_record(REC_ZONE_EXIT, nullptr);
    }
    Zone(const Zone &) = delete;
    Zone &operator=(const Zone &) = delete;
    bool active_ = false;
};

inline void message(const char *text) { push_record(REC_MSG, text); }
inline void plot(const char *name, float value) { push_record(REC_PLOT, name, value); }
inline void alloc(const void *ptr, size_t size) { (void)ptr; (void)size; /* future */ }
inline void free(const void *ptr) { (void)ptr; /* future */ }

#endif // !KERNEL_BUILD

} // namespace profiler

// ── Unified macros ────────────────────────────────────────────────────

/// RAII zone with a given name (__LINE__ - unique variable).
#define PROFILER_ZONE(name)   ::profiler::Zone _prof_zone_##__LINE__(name)

/// RAII zone auto-named from the enclosing C++ function via __FUNCTION__.
#define PROFILER_FUNCTION()   ::profiler::Zone _prof_func_zone_(__FUNCTION__)

/// RAII zone with a dynamically formatted name using printf-style format.
/// Uses a stack-allocated 256-byte buffer (no heap allocation).
/// Truncates if the formatted string exceeds the buffer.
#define PROFILER_FMT(fmt, ...)                                              \
    ::profiler::Zone _prof_fmt_zone_(                                     \
        ({ static char _prof_buf_[256];                                    \
           snprintf(_prof_buf_, sizeof(_prof_buf_), fmt, ##__VA_ARGS__);   \
           _prof_buf_; }))

/// Emit a text message visible in the profiler timeline.
#define PROFILER_MSG(text)    ::profiler::message(text)

/// Emit a formatted message (printf-style, stack buffer).
#define PROFILER_MSG_FMT(fmt, ...)                                         \
    do { if (::profiler::is_enabled()) {                                  \
        char _prof_buf_[256];                                              \
        snprintf(_prof_buf_, sizeof(_prof_buf_), fmt, ##__VA_ARGS__);     \
        ::profiler::message(_prof_buf_);                                   \
    } } while(0)

/// Emit a named plot value (time-series numeric data).
#define PROFILER_PLOT(n, v)   ::profiler::plot(n, v)

/// Mark a heap allocation for memory tracking.
#define PROFILER_ALLOC(p, s)  ::profiler::alloc(p, s)

/// Mark a heap deallocation for memory tracking.
#define PROFILER_FREE(p)      ::profiler::free(p)

/// Mark a frame boundary in the timeline.
#define PROFILER_FRAME_MARK   do { ::profiler::message("---FRAME---"); } while(0)

/// Stub zone for GPU operations (captured automatically by SYCL runtime).
#define PROFILER_GPU_ZONE(name) /* no-op */
