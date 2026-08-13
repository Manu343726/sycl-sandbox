#ifdef SANDBOX_ENABLE_TRACY
#include "tracy_bridge.h"
#include <sycl-sandbox/profiler.h>
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>
#include <client/TracyProfiler.hpp>
#include <algorithm>
#include <cstring>
#include <vector>
#include <spdlog/spdlog.h>

namespace tracy_bridge {
namespace {
constexpr uint8_t kCtx = 0, kCtxType = 7, kCtxFlags = 1;
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
    auto t = profiler::read_timestamp();
    ___tracy_emit_gpu_new_context_serial({(int64_t)t,1.0f,kCtx,kCtxFlags,kCtxType});
    ___tracy_emit_gpu_context_name_serial({kCtx,"SYCL",4});
    previous_cpu_time_ = tracy::Profiler::GetTime();
    spdlog::debug("[tracy] bridge initialised (ctx={}, t0_cycles={})",
                  kCtx, t);
}

void Bridge::shutdown() {
    std::lock_guard<std::mutex> lk(mtx_);
    spdlog::debug("[tracy] bridge shutdown (releasing {} plot names)",
                  plot_names_.size());
    plot_names_.clear();
    initialized_ = false;
}

uint64_t Bridge::make_srcloc(const std::string& name) {
    // Compact srcloc: owned by the alloc-serial GPU zone begin and freed after
    // transmission, so never cache/reuse the returned id.
    return ___tracy_alloc_srcloc_name(0,"kernel",6,"kernel",6,
                                      name.data(),name.size(),clr(name));
}

const char* Bridge::plot_name_for(const std::string& name) {
    std::lock_guard<std::mutex> lk(mtx_);
    return plot_names_.try_emplace(name, 0).first->first.c_str();
}

#ifndef KERNEL_NATIVE
void Bridge::submit_device_ring(profiler::DeviceRingHeader* dh,
                                profiler::DeviceRecord* dr, uint32_t cap,
                                sycl::queue* q, int64_t,
                                uint64_t host_t0, uint64_t host_t1) {
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
    spdlog::debug("[tracy] drain ring: backend={} records={}{}",
                  q ? "sycl" : "software", n,
                  overflow ? " (OVERFLOW — wrapped, dropped records)" : "");
    if (!n) return;

    std::vector<profiler::DeviceRecord> s(n);
    if (q) {
        q->memcpy(s.data(),dr,n*sizeof(profiler::DeviceRecord)).wait();
        q->memset(dh,0,sizeof(profiler::DeviceRingHeader)).wait();
    } else {
        std::memcpy(s.data(),dr,n*sizeof(profiler::DeviceRecord));
        std::memset(dh,0,sizeof(profiler::DeviceRingHeader));
    }

    uint64_t dmin=UINT64_MAX, dmax=0;
    for (auto& r : s) if (r.timestamp && r.type != profiler::DEV_PLOT) {
        dmin=std::min(dmin,r.timestamp); dmax=std::max(dmax,r.timestamp);
    }
    auto d2h=[&](uint64_t dt)->int64_t {
        if (dmax<=dmin) return (int64_t)host_t0;
        return (int64_t)(host_t0+(uint64_t)((double)(dt-dmin)/(double)(dmax-dmin)*(double)(host_t1-host_t0)));
    };

    struct DF { uint64_t t; uint32_t z; };
    std::unordered_map<uint16_t,std::vector<DF>> lanes;
    uint32_t cnt_zone=0, cnt_msg=0, cnt_plot=0, cnt_enter=0,
             cnt_unpaired_exit=0;
    for (auto& r : s) {
        if (r.type == profiler::DEV_MSG) {
            // Copying message variant — messageL would keep a pointer into a
            // temporary std::string that the UI later dereferences.
            std::string zn = "gpu:msg_" + std::to_string(r.zone_id);
            ___tracy_emit_message(zn.data(), zn.size(), 0);
            ++cnt_msg;
            continue;
        }
        if (r.type == profiler::DEV_PLOT) {
            float v; memcpy(&v, &r.timestamp, sizeof(v));
            ___tracy_emit_plot_float(plot_name_for("gpu:plot_" +
                                                   std::to_string(r.zone_id)),
                                     v);
            ++cnt_plot;
            continue;
        }
        if (!r.timestamp) continue;
        auto& st = lanes[r.lane];
        if (r.type==profiler::DEV_ZONE_ENTER) { st.push_back({r.timestamp,r.zone_id}); ++cnt_enter; }
        else if (r.type==profiler::DEV_ZONE_EXIT) {
            if (st.empty()) { ++cnt_unpaired_exit; continue; }
            DF f=std::move(st.back()); st.pop_back();
            uint16_t qid=query_id_++;
            std::string zn="gpu:zone_"+std::to_string(f.z);
            ___tracy_emit_gpu_zone_begin_alloc_serial({make_srcloc(zn),qid,kCtx});
            ___tracy_emit_gpu_time_serial({d2h(f.t),qid,kCtx});
            ___tracy_emit_gpu_zone_end_serial({qid,kCtx});
            ___tracy_emit_gpu_time_serial({d2h(r.timestamp),qid,kCtx});
            ++cnt_zone;
        }
    }
    uint32_t open = 0;
    for (auto& kv : lanes) open += (uint32_t)kv.second.size();
    spdlog::debug("[tracy] emitted zones={} msgs={} plots={} "
                  "(enter={} unpaired_exit={} open={}) "
                  "dev_clock=[{}..{}] host=[{}..{}]",
                  cnt_zone, cnt_msg, cnt_plot,
                  cnt_enter, cnt_unpaired_exit, open,
                  dmin, dmax, host_t0, host_t1);
    if (cnt_unpaired_exit)
        spdlog::warn("[tracy] {} zone EXIT records had no matching ENTER "
                     "(likely ring overflow mid-zone)", cnt_unpaired_exit);
}
#endif

void Bridge::frame_mark() {
    if (!initialized_) return;
    auto cyc = profiler::read_timestamp();
    int64_t now=tracy::Profiler::GetTime(), delta=now-previous_cpu_time_;
    previous_cpu_time_=now;
    ___tracy_emit_gpu_calibration_serial({(int64_t)cyc,delta,kCtx});
    FrameMarkNamed("render");
    spdlog::debug("[tracy] frame_mark: cycles={} cal_delta={}us", cyc, delta);
}

} // namespace tracy_bridge
#endif