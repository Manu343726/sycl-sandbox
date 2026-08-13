#pragma once

#ifdef SANDBOX_ENABLE_TRACY

#include <sycl-sandbox/profiler.h>
#include <cstdint>
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
                            uint32_t cap, sycl::queue*, int64_t frame,
                            uint64_t t0, uint64_t t1);
#else
    void submit_device_ring(...) {}
#endif
    void frame_mark();
private:
    /// Allocates a compact srcloc for a GPU zone. The alloc-serial zone
    /// begin API takes ownership and frees it after transmission, so a fresh
    /// one must be created per emit (never cached/reused).
    uint64_t make_srcloc(const std::string& name);
    /// Stable plot-name pointer (Tracy keys plots by pointer identity; the
    /// string must outlive the trace, so it is owned by plot_names_).
    const char* plot_name_for(const std::string& name);
    std::mutex mtx_;
    std::unordered_map<std::string, uint64_t> plot_names_;
    uint16_t query_id_ = 0;
    int64_t previous_cpu_time_ = 0;
    bool initialized_ = false;
    bool thread_named_ = false;
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
};
} // namespace tracy_bridge
#endif