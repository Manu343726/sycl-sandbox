#pragma once
/// @file
/// Profiler macros — single API, dispatched by compilation context.
///
/// All paths use PROFILER_ZONE("name") with an implicit `ctx` variable
/// (a reference to the per-call rt::Context):
///   ctx.prof       — profiler::DeviceRing (POD handle)
///   ctx.linear_id  — uint32_t work-item id (0 for host code)
///
/// The master switch is the CMake option `SANDBOX_ENABLE_PROFILER`
/// (compile definition when ON — the default).  When it is NOT defined
/// every macro below expands to nothing, so kernels and host code carry
/// ZERO profiling overhead regardless of the other paths.  When defined:
///   __SYCL_DEVICE_ONLY__  → device ring, ctx from SYCL lambda capture
///   KERNEL_BUILD          → host ring, ctx = the rt::Context& parameter
///   SANDBOX_ENABLE_TRACY  → Tracy C++ client API
///   else                  → no-ops
///
/// The KERNEL_BUILD path keeps NO state: the ring is read from ctx.prof
/// on every call, so kernels need no globals (single function, local
/// variables only).

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>

#ifndef KERNEL_NATIVE
#include <sycl/sycl.hpp>
#endif

// ═══════════════════════════════════════════════════════════════════════
//  profiler::DeviceRing / DeviceZone / helpers (shared by host + device)
// ═══════════════════════════════════════════════════════════════════════
namespace profiler {

inline constexpr uint8_t DEV_ZONE_ENTER = 0;
inline constexpr uint8_t DEV_ZONE_EXIT  = 1;
inline constexpr uint8_t DEV_MSG        = 2;
inline constexpr uint8_t DEV_PLOT       = 3;

struct DeviceRecord {
    uint64_t timestamp;
    uint32_t zone_id;
    uint8_t  type;
    uint8_t  _pad;
    uint16_t lane;
};

struct DeviceRingHeader {
    uint32_t write_pos;
    uint32_t clock_kind;
    uint32_t _pad[2];
};

constexpr uint32_t device_zone_hash(const char *name) {
    uint32_t h = 2166136261u;
    for (const char *p = name; *p; ++p) { h ^= (uint8_t)*p; h *= 16777619u; }
    return h;
}

struct DeviceRing {
    DeviceRingHeader *header = nullptr;
    DeviceRecord *records = nullptr;
    uint32_t capacity = 0;
    uint32_t sample_interval = 1;

    bool active() const { return header && records; }
    bool want(uint32_t lid) const { return active() && (sample_interval <= 1 || lid % sample_interval == 0); }

    static uint64_t timestamp() {
#if defined(__has_builtin) && __has_builtin(__builtin_readcyclecounter)
        return __builtin_readcyclecounter();
#elif defined(__x86_64__) || defined(__i386__)
        uint32_t lo, hi; asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
#else
        return 0;
#endif
    }

    void push(uint32_t zone_id, uint8_t type, uint32_t lid) const {
        if (!active()) return;
        uint32_t idx;
#ifdef KERNEL_NATIVE
        idx = __atomic_fetch_add(&header->write_pos, 1, __ATOMIC_RELAXED);
#else
        ::sycl::atomic_ref<uint32_t,::sycl::memory_order::relaxed,::sycl::memory_scope::device> wp(header->write_pos);
        idx = wp.fetch_add(1);
#endif
        DeviceRecord &rec = records[idx & (capacity - 1)];
        rec.timestamp = timestamp(); rec.zone_id = zone_id; rec.type = type;
        rec._pad = 0; rec.lane = (uint16_t)(lid & 0xffff);
    }
    void push_msg(uint32_t h, uint32_t lid) const { push(h, DEV_MSG, lid); }
    void push_plot(uint32_t h, float v, uint32_t lid) const {
        if (!active()) return;
        uint32_t idx;
#ifdef KERNEL_NATIVE
        idx = __atomic_fetch_add(&header->write_pos, 1, __ATOMIC_RELAXED);
#else
        ::sycl::atomic_ref<uint32_t,::sycl::memory_order::relaxed,::sycl::memory_scope::device> wp(header->write_pos);
        idx = wp.fetch_add(1);
#endif
        DeviceRecord &rec = records[idx & (capacity - 1)];
        uint32_t bits; memcpy(&bits, &v, sizeof(bits));
        rec.timestamp = bits; rec.zone_id = h; rec.type = DEV_PLOT;
        rec._pad = 0; rec.lane = (uint16_t)(lid & 0xffff);
    }
};

class DeviceZone {
public:
    DeviceZone(const DeviceRing &ring, uint32_t zid, uint32_t lid)
        : ring_(ring), zid_(zid), on_(ring.want(lid)) {
        if (on_) ring_.push(zid_, DEV_ZONE_ENTER, lid);
    }
    ~DeviceZone() { if (on_) ring_.push(zid_, DEV_ZONE_EXIT, 0); }
private:
    DeviceRing ring_; uint32_t zid_; bool on_;
};

inline uint64_t read_timestamp() { return DeviceRing::timestamp(); }
inline uint32_t hash_zone_name(const char *n) { return device_zone_hash(n); }

} // namespace profiler

// ── Token paste ───────────────────────────────────────────────────────
#define SANDBOX_PZ_CAT2(a,b) a##b
#define SANDBOX_PZ_CAT(a,b) SANDBOX_PZ_CAT2(a,b)

#if !defined(SANDBOX_ENABLE_PROFILER)
// ═══════════════════════════════════════════════════════════════════════
//  Profiler support disabled at compile time (CMake option
//  SANDBOX_ENABLE_PROFILER=OFF) — every macro is a no-op so kernels and
//  host code carry ZERO profiling overhead.  This takes precedence over
//  the device-ring / KERNEL_BUILD / Tracy paths below.
// ═══════════════════════════════════════════════════════════════════════
#define PROFILER_ZONE(n)     ((void)0)
#define PROFILER_FUNCTION()  ((void)0)
#define PROFILER_FMT(...)    ((void)0)
#define PROFILER_MSG(...)    ((void)0)
#define PROFILER_MSG_FMT(...) ((void)0)
#define PROFILER_PLOT(...)   ((void)0)
#define PROFILER_ALLOC(...)  ((void)0)
#define PROFILER_FREE(...)   ((void)0)
#define PROFILER_FRAME_MARK  ((void)0)

#elif defined(__SYCL_DEVICE_ONLY__) || defined(KERNEL_BUILD)
// Device code, or kernel .so host pipeline code — the implicit `ctx` is a
// reference to the per-call rt::Context (in a SYCL lambda it is the
// captured kernel-context object, possibly a per-work-item copy stamped
// with linear_id): `ctx.prof` is the device profiler ring, `ctx.linear_id`
// the work-item id (0 for host-side tracing).  An inactive ring makes
// every zone a no-op.  The KERNEL_BUILD path keeps NO state: the ring is
// read from ctx.prof on every call, so kernels need no globals (single
// function, local variables only).
#define PROFILER_ZONE(name) \
    ::profiler::DeviceZone SANDBOX_PZ_CAT(_pz_,__LINE__)( \
        ctx.prof, ::profiler::device_zone_hash(name), (uint32_t)ctx.linear_id)
#define PROFILER_FUNCTION() PROFILER_ZONE(__PRETTY_FUNCTION__)
#define PROFILER_FMT(...)   PROFILER_MSG_FMT(__VA_ARGS__)
#define PROFILER_MSG(t)  do{if(ctx.prof.active())ctx.prof.push_msg(::profiler::device_zone_hash(t),(uint32_t)ctx.linear_id);}while(0)
#define PROFILER_MSG_FMT(f,...) do{if(ctx.prof.active()){char _b_[64];snprintf(_b_,sizeof(_b_),f,##__VA_ARGS__);ctx.prof.push_msg(::profiler::device_zone_hash(_b_),(uint32_t)ctx.linear_id);}}while(0)
#define PROFILER_PLOT(n,v) do{if(ctx.prof.active())ctx.prof.push_plot(::profiler::device_zone_hash(n),(float)(v),(uint32_t)ctx.linear_id);}while(0)
#define PROFILER_ALLOC(...)  ((void)0)
#define PROFILER_FREE(...)   ((void)0)
#define PROFILER_FRAME_MARK  ((void)0)

#elif defined(SANDBOX_ENABLE_TRACY)
// ═══════════════════════════════════════════════════════════════════════
//  App host code — Tracy C++ client API
// ═══════════════════════════════════════════════════════════════════════
#include <tracy/Tracy.hpp>
#define PROFILER_ZONE(name)  ZoneNamedN(___tracy_scoped_zone, name, true)
#define PROFILER_FUNCTION()  ZoneScoped
#define PROFILER_FMT(f,...)  ZoneNamedN(SANDBOX_PZ_CAT(___tracy_fmt_zone,__LINE__),({static char _b_[256];snprintf(_b_,sizeof(_b_),f,##__VA_ARGS__);_b_;}),true)
#define PROFILER_MSG(t)      TracyMessage(t, 0)
#define PROFILER_MSG_FMT(f,...) do{char _b_[256];snprintf(_b_,sizeof(_b_),f,##__VA_ARGS__);TracyMessage(_b_,0);}while(0)
#define PROFILER_PLOT(n,v)   TracyPlot(n, (double)(v))
#define PROFILER_ALLOC(p,s)  TracyAlloc(p,s)
#define PROFILER_FREE(p)     TracyFree(p)
#define PROFILER_FRAME_MARK   FrameMarkNamed("render")

#else
// ═══════════════════════════════════════════════════════════════════════
//  Profiler disabled
// ═══════════════════════════════════════════════════════════════════════
#define PROFILER_ZONE(n)     ((void)0)
#define PROFILER_FUNCTION()  ((void)0)
#define PROFILER_FMT(...)    ((void)0)
#define PROFILER_MSG(...)    ((void)0)
#define PROFILER_MSG_FMT(...) ((void)0)
#define PROFILER_PLOT(...)   ((void)0)
#define PROFILER_ALLOC(...)  ((void)0)
#define PROFILER_FREE(...)   ((void)0)
#define PROFILER_FRAME_MARK  ((void)0)
// hash_zone_name / read_timestamp are defined inline above (shared by all
// paths) — no redefinition here.
#endif
