#ifdef SANDBOX_ENABLE_TRACY
#include "tracy_bridge.h"
#include "kernel/zone_names.h"
#include <sycl-sandbox/profiler.h>
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>
#include <client/TracyProfiler.hpp>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>
#include <spdlog/spdlog.h>

namespace tracy_bridge {
namespace {
constexpr uint8_t kCtx = 0, kCtxType = 7, kCtxFlags = 1;
/// Host monotonic-clock nanoseconds.  Tracy's GpuCalibration protocol
/// expects cpuDelta in host NANOSECONDS (see TracyWorker.cpp
/// ProcessGpuCalibration: calibrationMod = cpuDelta / gpuDelta), while
/// Tracy::Profiler::GetTime() returns raw TSC cycles on x86_64 — those
/// two must NOT be mixed in the calibration delta.
int64_t host_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}
uint32_t clr(const std::string& n) {
    uint32_t h = 2166136261u;
    for (char c : n) { h ^= (uint8_t)c; h *= 16777619u; }
    return 0xFF000000u | (h & 0xFFFFFFu);
}
}

void Bridge::init() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (initialized_) return;
    initialized_ = true;
    // Device clock for the GPU timeline = the ring's write_pos record
    // index (single global atomic — monotonic and consistent across all
    // SMs, unlike clock64 which is per-SM with a private epoch).  Anchor
    // record-index 0 at init; submit_device_ring() emits a per-frame
    // calibration that maps record indices into the frame's host-time
    // window.
    auto n0 = host_ns();
    ___tracy_emit_gpu_new_context_serial({0,1.0f,kCtx,kCtxFlags,kCtxType});
    ___tracy_emit_gpu_context_name_serial({kCtx,"SYCL",4});
    int64_t delta_ns = host_ns() - n0;
    if (delta_ns <= 0) delta_ns = 1;
    // Provisional seed: anchor record-index 1 so the very first frame's
    // zones (emitted only after the first submit calibration) never see
    // an uncalibrated ctx with a division-by-zero gpuDelta.  No zones
    // exist yet, so the seed's wildly wrong rate is irrelevant.
    ___tracy_emit_gpu_calibration_serial({1,delta_ns,kCtx});
    previous_cpu_time_ = host_ns();
    cumulative_records_ = 0;
    spdlog::debug("[tracy] bridge initialised (ctx={}, record-index clock, "
                  "seed_cal_delta_ns={})", kCtx, delta_ns);
}

void Bridge::shutdown() {
    std::lock_guard<std::mutex> lk(mtx_);
    spdlog::debug("[tracy] bridge shutdown (releasing {} plot names)",
                  plot_names_.size());
    plot_names_.clear();
    zone_name_cache_.clear();
    initialized_ = false;
}

uint64_t Bridge::make_srcloc(const std::string& name,
                             const std::string& function) {
    // Compact srcloc: owned by the alloc-serial GPU zone begin and freed after
    // transmission, so never cache/reuse the returned id.  The `name` is the
    // profiler zone name (the srcloc.name field Tracy's GetZoneName prefers);
    // `function` carries the enclosing zone chain (render_pixel > render_sample
    // > ...), which the Tracy UI shows as the call-stack-like nesting path.
    return ___tracy_alloc_srcloc_name(0,"kernel",6,
                                      function.data(),function.size(),
                                      name.data(),name.size(),clr(name));
}

/// Resolve a device zone/msg/plot hash to its profiler name (from the
/// generated perfect-hash table + hot-reload runtime scan).  Falls back to
/// the legacy `gpu:zone_<hash>` style so unknown hashes still appear.
std::string Bridge::zone_name_for(uint32_t hash) {
    {
        // Fast path: already resolved this frame (or a previous one).  The
        // generated table is constexpr/lock-free; the runtime registry is
        // mutex-guarded, so we cache to avoid locking per record.
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = zone_name_cache_.find(hash);
        if (it != zone_name_cache_.end()) return it->second;
    }
    auto n = profiler::lookup_zone_name(hash);
    std::string out = n.empty() ? ("gpu:zone_" + std::to_string(hash)) : n;
    std::lock_guard<std::mutex> lk(mtx_);
    zone_name_cache_.emplace(hash, out);
    return out;
}

const char* Bridge::plot_name_for(const std::string& name) {
    std::lock_guard<std::mutex> lk(mtx_);
    return plot_names_.try_emplace(name, 0).first->first.c_str();
}

#ifndef KERNEL_NATIVE
void Bridge::submit_device_ring(profiler::DeviceRingHeader* dh,
                                profiler::DeviceRecord* dr, uint32_t cap,
                                sycl::queue* q) {
    if (!initialized_||!dh) return;
    if (!thread_named_) { ___tracy_set_thread_name("render"); thread_named_=true; }

    // Read back the header + records and reset write_pos. The ring
    // storage is the same address from both sides on every backend, but
    // the readback method differs:
    //  - GPU / SYCL-CPU (q != nullptr): ring is USM-shared; go through
    //    q->memcpy/q->memset so the reads/writes are ordered against
    //    in-flight device writes.
    //  - Software (q == nullptr): ring is plain host heap; the kernel
    //    .so runs synchronously, so all writes are already visible and a
    //    plain std::memcpy/std::memset suffices.
    profiler::DeviceRingHeader hdr{};
    if (q) q->memcpy(&hdr,dh,sizeof(hdr)).wait();
    else   std::memcpy(&hdr,dh,sizeof(hdr));
    uint32_t n = std::min(hdr.write_pos,cap);
    const bool overflow = hdr.write_pos > cap;
    // Publish the raw (unmasked) write_pos and overflow flag for the UI
    // thread — last_write_pos is the true record count the kernel
    // produced this frame, even if it exceeded the ring capacity.
    last_write_pos_.store(hdr.write_pos, std::memory_order_relaxed);
    last_overflow_.store(overflow, std::memory_order_relaxed);
    spdlog::debug("[tracy] drain ring: backend={} records={} write_pos={}{}",
                  q ? "sycl" : "software", n, hdr.write_pos,
                  overflow ? " (OVERFLOW — wrapped, dropped records)" : "");
    if (!n) {
        // No records this frame: advance the calibration baseline so a
        // later record-producing frame's cpuDelta measures only its own
        // duration, not the idle gap since the last records.
        previous_cpu_time_ = host_ns();
        return;
    }

    std::vector<profiler::DeviceRecord> s(n);
    if (q) {
        q->memcpy(s.data(),dr,n*sizeof(profiler::DeviceRecord)).wait();
        q->memset(dh,0,sizeof(profiler::DeviceRingHeader)).wait();
    } else {
        std::memcpy(s.data(),dr,n*sizeof(profiler::DeviceRecord));
        std::memset(dh,0,sizeof(profiler::DeviceRingHeader));
    }

    // Keep the device-clock range in the drained records (for the log
    // only — with the record-index clock it shows the span of the frame's
    // write_pos values, i.e. roughly n).
    uint64_t dmin=UINT64_MAX, dmax=0;
    for (auto& r : s) if (r.type != profiler::DEV_PLOT) {
        dmin=std::min(dmin,r.timestamp); dmax=std::max(dmax,r.timestamp);
    }

    // Calibrate the record-index domain into host time BEFORE emitting
    // this frame's zones: the server converts a zone's raw gpuTime with
    // the most recent calibration, so an anchor emitted first lands the
    // whole frame inside its own [start,end] host window.
    //   gpuTime  = cumulative record count at the END of this frame
    //   cpuDelta = host ns elapsed since the previous calibration
    // Tracy then computes calibrationMod = cpuDelta/gpuDelta = host
    // ns-per-record for this frame, and a zone with index c maps to
    // (c - end)*mod + now — the frame's start when c = start, its end
    // when c = end.  Record production is treated as uniform within the
    // frame, which is the best we can do without a real device-global
    // clock (%globaltimer is unreachable under generic SSCP).
    const uint64_t frame_base = cumulative_records_;
    cumulative_records_ += n;
    const int64_t now_ns = host_ns();
    const int64_t cal_delta = now_ns - previous_cpu_time_;
    previous_cpu_time_ = now_ns;
    if (cal_delta > 0)
        ___tracy_emit_gpu_calibration_serial({(int64_t)cumulative_records_, cal_delta, kCtx});

    struct DF { uint64_t t; uint32_t z; };
    std::unordered_map<uint16_t,std::vector<DF>> lanes;
    uint32_t cnt_zone=0, cnt_msg=0, cnt_plot=0, cnt_enter=0,
             cnt_unpaired_exit=0;
    for (auto& r : s) {
        if (r.type == profiler::DEV_MSG) {
            // Copying message variant — messageL would keep a pointer into a
            // temporary std::string that the UI later dereferences.
            std::string zn = zone_name_for(r.zone_id);
            ___tracy_emit_message(zn.data(), zn.size(), 0);
            ++cnt_msg;
            continue;
        }
        if (r.type == profiler::DEV_PLOT) {
            float v; memcpy(&v, &r.timestamp, sizeof(v));
            std::string pn = profiler::lookup_zone_name(r.zone_id);
            if (pn.empty()) pn = "gpu:plot_" + std::to_string(r.zone_id);
            ___tracy_emit_plot_float(plot_name_for(pn), v);
            ++cnt_plot;
            continue;
        }
        auto& st = lanes[r.lane];
        if (r.type==profiler::DEV_ZONE_ENTER) { st.push_back({r.timestamp,r.zone_id}); ++cnt_enter; }
        else if (r.type==profiler::DEV_ZONE_EXIT) {
            if (st.empty()) { ++cnt_unpaired_exit; continue; }
            DF f=std::move(st.back()); st.pop_back();
            uint16_t qid=query_id_++;
            // Display name: resolve the device zone-id hash to the real
            // profiler name (aabb_slab, render_sample, ...) instead of the
            // opaque gpu:zone_<hash>, so the UI timeline and the diag
            // walker show meaningful names.
            std::string zn = zone_name_for(f.z);
            // Call-stack-like path: the enclosing zone chain still on this
            // lane's stack (outer-most first).  The Tracy UI shows srcloc
            // function alongside the zone name.
            std::string chain;
            for (const auto& p : st) {
                if (!chain.empty()) chain += " > ";
                chain += zone_name_for(p.z);
            }
            ___tracy_emit_gpu_zone_begin_alloc_serial({make_srcloc(zn, chain),qid,kCtx});
            ___tracy_emit_gpu_time_serial({(int64_t)(frame_base + f.t),qid,kCtx});
            ___tracy_emit_gpu_zone_end_serial({qid,kCtx});
            ___tracy_emit_gpu_time_serial({(int64_t)(frame_base + r.timestamp),qid,kCtx});
            ++cnt_zone;
        }
    }
    uint32_t open = 0;
    for (auto& kv : lanes) open += (uint32_t)kv.second.size();
    last_emitted_zones_.store(cnt_zone, std::memory_order_relaxed);
    spdlog::debug("[tracy] emitted zones={} msgs={} plots={} "
                  "(enter={} unpaired_exit={} open={}) dev_clock=[{}..{}]",
                  cnt_zone, cnt_msg, cnt_plot,
                  cnt_enter, cnt_unpaired_exit, open,
                  dmin, dmax);
    if (cnt_unpaired_exit)
        spdlog::warn("[tracy] {} zone EXIT records had no matching ENTER "
                     "(likely ring overflow mid-zone)", cnt_unpaired_exit);
}
#endif

void Bridge::frame_mark() {
    if (!initialized_) return;
    // The per-frame calibration is emitted in submit_device_ring() (it
    // must precede the frame's zones so the server applies it to them).
    // This mark just fences each rendered frame on the timeline.
    FrameMarkNamed("render");
    spdlog::debug("[tracy] frame_mark");
}

} // namespace tracy_bridge
#endif