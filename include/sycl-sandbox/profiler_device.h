#pragma once

/// @file
/// Device-side profiler ring buffer.
///
/// Unlike the host-side profiler in profiler.h (which lives in the kernel
/// .so's data segment and is therefore unreachable from GPU device code),
/// the DeviceRing is a POD handle that travels INTO the kernel as a value
/// captured by the SYCL lambda (via RenderContext::prof).  The header and
/// record array live in device memory (sycl::malloc_device), so device
/// threads can append records with atomic slot claims.  The host copies
/// the ring back once per frame and merges it into the profiler UI.
///
/// Timestamp source:
///   - SYCL device code: __builtin_readcyclecounter() → rdtsc on CPU
///     backends, clock64() on CUDA.  Per-thread deltas (zone durations)
///     are meaningful; absolute cross-SM alignment is approximate.
///   - With SANDBOX_DEVICE_GLOBALTIMER defined (CUDA/PTX only, opt-in):
///     the PTX %globaltimer register — a nanosecond wall clock coherent
///     across all SMs — selected at JIT time via AdaptiveCpp's
///     ACPP_EXT_JIT_COMPILE_IF reflection so non-PTX backends keep the
///     cycle counter.
///   - Native (KERNEL_NATIVE) builds: rdtsc, appended with plain
///     __atomic intrinsics.

#include <cstdint>

#ifndef KERNEL_NATIVE
#include <sycl/sycl.hpp>
#endif

namespace profiler {

// Shared with profiler.h RecordType values (REC_ZONE_ENTER / REC_ZONE_EXIT).
inline constexpr uint8_t DEV_ZONE_ENTER = 0;
inline constexpr uint8_t DEV_ZONE_EXIT  = 1;
inline constexpr uint8_t DEV_PLOT       = 3;

/// Clock domain of the ring's timestamps (stored in the header so the
/// host-side ingest knows how to map them onto the timeline).
enum DeviceClockKind : uint32_t {
    DEV_CLOCK_CYCLES     = 0,  ///< __builtin_readcyclecounter (rdtsc / clock64)
    DEV_CLOCK_NS_GLOBAL  = 1,  ///< PTX %globaltimer (nanoseconds, SM-coherent)
};

/// One profiling event from device code.  16 bytes.
struct DeviceRecord {
    uint64_t timestamp;
    uint32_t zone_id;   ///< FNV-1a hash of the zone name (see hash_zone_name)
    uint8_t  type;      ///< DEV_ZONE_ENTER / DEV_ZONE_EXIT / DEV_PLOT
    uint8_t  _pad;
    uint16_t lane;      ///< low bits of the linear work-item id (enter/exit matching)
};

/// Ring header, first bytes of the device allocation.
struct DeviceRingHeader {
    uint32_t write_pos;    ///< atomically incremented slot counter
    uint32_t clock_kind;   ///< DeviceClockKind actually used by the kernel
    uint32_t _pad[2];
};

/// POD handle to the device ring.  Captured BY VALUE into kernel lambdas.
/// An inactive ring (header == nullptr) makes every operation a no-op, so
/// kernels can call the zone macros unconditionally.
struct DeviceRing {
    DeviceRingHeader *header = nullptr;  ///< device memory
    DeviceRecord *records = nullptr;     ///< device memory, `capacity` entries
    uint32_t capacity = 0;               ///< power of two
    uint32_t sample_interval = 1;        ///< record zones for 1/N work items

    bool active() const { return header != nullptr && records != nullptr; }

    /// Should this work item record its zones?  Decimation keeps the ring
    /// from overflowing on multi-megapixel launches.
    bool want(uint32_t linear_id) const {
        return active() &&
               (sample_interval <= 1 || linear_id % sample_interval == 0);
    }

    /// Read the device timestamp.
    ///
    /// __builtin_readcyclecounter lowers to llvm.readcyclecounter, which
    /// every backend the SSCP JIT targets implements (rdtsc on x86,
    /// clock64 on NVPTX) — inline x86 asm here would poison the device IR
    /// and break the JIT the moment a GPU work-item executed it, so the
    /// asm form is reserved for compilers without the builtin (GCC in
    /// KERNEL_NATIVE mode), which only ever run on the host.
    static uint64_t timestamp() {
#if defined(__has_builtin) && __has_builtin(__builtin_readcyclecounter)
        return __builtin_readcyclecounter();
#elif defined(__x86_64__) || defined(__i386__)
        uint64_t lo, hi;
        asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
        return (hi << 32) | lo;
#else
        return 0;   // no cycle counter — zones still count invocations
#endif
    }

    static constexpr uint32_t clock_kind_compiled() { return DEV_CLOCK_CYCLES; }

    /// Append one record (atomic slot claim; oldest records are overwritten
    /// on wrap so the newest events always survive).
    void push(uint32_t zone_id, uint8_t type, uint32_t linear_id) const {
        if (!active()) return;
        uint32_t idx;
#ifdef KERNEL_NATIVE
        idx = __atomic_fetch_add(&header->write_pos, 1, __ATOMIC_RELAXED);
#else
        ::sycl::atomic_ref<uint32_t, ::sycl::memory_order::relaxed,
                           ::sycl::memory_scope::device>
            wp(header->write_pos);
        idx = wp.fetch_add(1);
#endif
        DeviceRecord &rec = records[idx & (capacity - 1)];
        rec.timestamp = timestamp();
        rec.zone_id = zone_id;
        rec.type = type;
        rec._pad = 0;
        rec.lane = (uint16_t)(linear_id & 0xffff);
    }
};

/// Compile-time FNV-1a hash — same algorithm as profiler::hash_zone_name
/// so device zone ids resolve against the same name registry.
constexpr uint32_t device_zone_hash(const char *name) {
    uint32_t h = 2166136261u;
    for (const char *p = name; *p; ++p) {
        h ^= (uint8_t)*p;
        h *= 16777619u;
    }
    return h;
}

/// RAII device zone.  Records enter on construction and exit on
/// destruction, but only when the ring wants this work item.
class DeviceZone {
public:
    DeviceZone(const DeviceRing &ring, uint32_t zone_id, uint32_t linear_id)
        : ring_(ring), zone_id_(zone_id), linear_id_(linear_id),
          on_(ring.want(linear_id)) {
        if (on_) ring_.push(zone_id_, DEV_ZONE_ENTER, linear_id_);
    }
    ~DeviceZone() {
        if (on_) ring_.push(zone_id_, DEV_ZONE_EXIT, linear_id_);
    }
    DeviceZone(const DeviceZone &) = delete;
    DeviceZone &operator=(const DeviceZone &) = delete;

private:
    const DeviceRing &ring_;
    uint32_t zone_id_;
    uint32_t linear_id_;
    bool on_;
};

} // namespace profiler

/// Device-side RAII zone macro.  `ring` is a profiler::DeviceRing (usually
/// ctx.prof captured by value), `linear_id` the flat work-item index.
/// The zone-name hash is computed at compile time.
#define SANDBOX_PDZ_CONCAT2(a, b) a##b
#define SANDBOX_PDZ_CONCAT(a, b) SANDBOX_PDZ_CONCAT2(a, b)
#define PROFILER_DEVICE_ZONE(ring, name, linear_id)                        \
    ::profiler::DeviceZone SANDBOX_PDZ_CONCAT(_pdz_, __LINE__)(            \
        (ring), ::profiler::device_zone_hash(name), (uint32_t)(linear_id))
