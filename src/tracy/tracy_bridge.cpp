#ifdef SANDBOX_ENABLE_TRACY
#include "tracy_bridge.h"
#include <sycl-sandbox/profiler.h>
#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>
#include <client/TracyProfiler.hpp>
#include <algorithm>
#include <cstring>
#include <vector>

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
}

void Bridge::shutdown() { std::lock_guard<std::mutex> lk(mtx_); srclocs_.clear(); initialized_=false; }

uint64_t Bridge::srcloc_for(const std::string& name) {
    auto it = srclocs_.find(name);
    if (it != srclocs_.end()) return it->second;
    uint64_t s = ___tracy_alloc_srcloc_name(0,"kernel",6,"kernel",6,name.data(),name.size(),clr(name));
    srclocs_.emplace(name,s);
    return s;
}

#ifndef KERNEL_NATIVE
void Bridge::submit_device_ring(profiler::DeviceRingHeader* dh,
                                profiler::DeviceRecord* dr, uint32_t cap,
                                sycl::queue* q, int64_t,
                                uint64_t host_t0, uint64_t host_t1) {
    if (!initialized_||!dh||!q) return;
    if (!thread_named_) { ___tracy_set_thread_name("render"); thread_named_=true; }

    profiler::DeviceRingHeader hdr{};
    q->memcpy(&hdr,dh,sizeof(hdr)).wait();
    uint32_t n = std::min(hdr.write_pos,cap);
    if (!n) return;

    std::vector<profiler::DeviceRecord> s(n);
    q->memcpy(s.data(),dr,n*sizeof(profiler::DeviceRecord)).wait();
    q->memset(dh,0,sizeof(profiler::DeviceRingHeader)).wait();

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
    std::lock_guard<std::mutex> lk(mtx_);
    for (auto& r : s) {
        if (r.type == profiler::DEV_MSG) {
            // name hash in zone_id; host resolves via zone-name registry
            std::string zn = "gpu:msg_" + std::to_string(r.zone_id);
            ___tracy_emit_messageL(zn.c_str(), 0);
            continue;
        }
        if (r.type == profiler::DEV_PLOT) {
            std::string zn = "gpu:plot_" + std::to_string(r.zone_id);
            float v; memcpy(&v, &r.timestamp, sizeof(v));
            ___tracy_emit_plot_float(zn.c_str(), v);
            continue;
        }
        if (!r.timestamp) continue;
        auto& st = lanes[r.lane];
        if (r.type==profiler::DEV_ZONE_ENTER) st.push_back({r.timestamp,r.zone_id});
        else if (r.type==profiler::DEV_ZONE_EXIT&&!st.empty()) {
            DF f=std::move(st.back()); st.pop_back();
            uint16_t qid=query_id_++;
            std::string zn="gpu:zone_"+std::to_string(f.z);
            ___tracy_emit_gpu_zone_begin_serial({srcloc_for(zn),qid,kCtx});
            ___tracy_emit_gpu_time_serial({d2h(f.t),qid,kCtx});
            ___tracy_emit_gpu_zone_end_serial({qid,kCtx});
            ___tracy_emit_gpu_time_serial({d2h(r.timestamp),qid,kCtx});
        }
    }
}
#endif

void Bridge::frame_mark() {
    if (!initialized_) return;
    auto cyc = profiler::read_timestamp();
    int64_t now=tracy::Profiler::GetTime(), delta=now-previous_cpu_time_;
    previous_cpu_time_=now;
    ___tracy_emit_gpu_calibration_serial({(int64_t)cyc,delta,kCtx});
    FrameMarkNamed("render");
}

} // namespace tracy_bridge
#endif