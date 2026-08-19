#pragma once

#ifdef SANDBOX_ENABLE_TRACY

#include <sycl-sandbox/profiler.h>
#include <cstdint>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#ifndef KERNEL_NATIVE
#include <sycl/sycl.hpp>
#endif

namespace tracy_bridge {

class Bridge {
public:
    Bridge() = default;
    void init();
    void shutdown();
#ifndef KERNEL_NATIVE
    void submit_device_ring(profiler::DeviceRingHeader*, profiler::DeviceRecord*,
                            uint32_t cap, sycl::queue*);
#else
    void submit_device_ring(...) {}
#endif
    void frame_mark();

    /// Raw (unmasked) write_pos reported by the last drained frame —
    /// the true record count the kernel produced, even if it exceeded
    /// the ring capacity.  The UI reads this to recommend a capacity
    /// that covers a full frame without overflow.
    uint32_t last_write_pos() const { return last_write_pos_.load(std::memory_order_relaxed); }
    /// True if the last drained frame overflowed the ring (write_pos >
    /// cap), meaning some zone records were wrapped and dropped.
    bool last_overflow() const { return last_overflow_.load(std::memory_order_relaxed); }
    /// Zones emitted by the last submit_device_ring() — the
    /// profiler-system diagnostic compares this against the count the
    /// tracy::Worker captures to confirm the full pipeline (ring →
    /// bridge → client → server) carries every record.
    uint32_t last_emitted_zones() const { return last_emitted_zones_.load(std::memory_order_relaxed); }
private:
    /// Allocates a compact srcloc for a GPU zone. The alloc-serial zone
    /// begin API takes ownership and frees it after transmission, so a fresh
    /// one must be created per emit (never cached/reused).  `name` is the
    /// profiler zone name; `function` is the enclosing-zone chain shown as
    /// the call-stack-like path in the UI.
    uint64_t make_srcloc(const std::string& name,
                         const std::string& function = std::string{});
    /// Stable plot-name pointer (Tracy keys plots by pointer identity; the
    /// string must outlive the trace, so it is owned by plot_names_).
    const char* plot_name_for(const std::string& name);
    /// Resolve a device zone/msg/plot hash to its display name, caching the
    /// result (the zone-name registry is lock-guarded; device frames can
    /// emit >100k zone records, so only unknown hashes should ever touch
    /// the registry).  Hash → name is fixed once registered, so the cache
    /// never goes stale.
    std::string zone_name_for(uint32_t hash);
    std::mutex mtx_;
    std::unordered_map<std::string, uint64_t> plot_names_;
    std::unordered_map<uint32_t, std::string> zone_name_cache_;
    uint16_t query_id_ = 0;
    /// Host monotonic-clock ns of the previous calibration (or of the
    /// last frame that produced no records, so idle gaps don't inflate
    /// the next frame's ns-per-record rate).
    int64_t previous_cpu_time_ = 0;
    /// Cumulative record count drained across frames — the "device clock"
    /// domain used for GPU zone times.  Device records carry their ring
    /// write_pos index (single global atomic, monotonic and consistent
    /// across all SMs); submit_device_ring() adds this offset so record
    /// indices stay strictly increasing across frames, then emits a
    /// per-frame calibration mapping them into the frame's host window.
    uint64_t cumulative_records_ = 0;
    bool initialized_ = false;
    bool thread_named_ = false;
    /// Written by submit_device_ring (render thread), read by the UI
    /// thread for the overflow warning.
    std::atomic<uint32_t> last_write_pos_{0};
    std::atomic<bool> last_overflow_{false};
    std::atomic<uint32_t> last_emitted_zones_{0};
};

} // namespace tracy_bridge
#else
namespace tracy_bridge {
class Bridge {
public:
    void init() {}
    void shutdown() {}
    void submit_device_ring(...) {}
    void frame_mark() {}
    uint32_t last_write_pos() const { return 0; }
    bool last_overflow() const { return false; }
};
} // namespace tracy_bridge
#endif