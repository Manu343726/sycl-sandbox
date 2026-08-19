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
    /// Interest-zone sampling percentage (0-100), written by the host
    /// each frame (see PROFILER_INTEREST_BEGIN).  The kernel reads it at
    /// the start of an interest zone to decide whether THIS work-item
    /// gets recorded or dropped.
    uint32_t interest_pct;
    uint32_t _pad;
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
    /// Per-lane interest-zone state (one byte per work-item, host-
    /// allocated USM shared / host heap).  Values:
    ///   0 — not inside an interest zone (default → record everything)
    ///   1 — inside a SAMPLED interest zone  → record
    ///   2 — inside a DROPPED interest zone  → drop every record
    /// Written by the PROFILER_INTEREST_BEGIN guard; cleared to 0 when
    /// the zone closes.  Null when the feature is unused/unsized — every
    /// push then records (backward compatible).
    uint8_t *sample_flags = nullptr;
    uint32_t sample_count = 0;

    bool active() const { return header && records; }

    /// True when lane `lid` is inside a dropped interest zone and must
    /// NOT emit any profiler record.
    bool dropped_lane(uint32_t lid) const {
        return sample_flags && lid < sample_count &&
               sample_flags[lid] == 2;
    }

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

    /// True when the ring holds a full frame's records and pushes must
    /// stop.  write_pos is monotonic within a frame (the host memsets the
    /// header to zero after draining), so once it reaches capacity every
    /// further push would only wrap and overwrite the earliest records —
    /// and pay for a contended global atomic per record.  The early-out
    /// caps the per-frame atomic traffic at ~capacity and keeps the
    /// FIRST `capacity` records of the frame instead.  A relaxed read is
    /// enough: a stale value costs at most a couple of extra atomics,
    /// which the idx >= capacity guard below then drops without writing.
    bool full() const {
#ifdef KERNEL_NATIVE
        return __atomic_load_n(&header->write_pos, __ATOMIC_RELAXED) >= capacity;
#else
        return ::sycl::atomic_ref<uint32_t,::sycl::memory_order::relaxed,::sycl::memory_scope::device>(header->write_pos).load() >= capacity;
#endif
    }

    void push(uint32_t zone_id, uint8_t type, uint32_t lid) const {
        if (!active() || dropped_lane(lid) || full()) return;
        uint32_t idx;
#ifdef KERNEL_NATIVE
        idx = __atomic_fetch_add(&header->write_pos, 1, __ATOMIC_RELAXED);
#else
        ::sycl::atomic_ref<uint32_t,::sycl::memory_order::relaxed,::sycl::memory_scope::device> wp(header->write_pos);
        idx = wp.fetch_add(1);
#endif
        if (idx >= capacity) return;
        DeviceRecord &rec = records[idx];
        // Device clock for the Tracy timeline = the record's write_pos
        // index.  It is the only globally-consistent, monotonic device
        // clock: clock64()/readcyclecounter is per-SM with a private
        // epoch, so raw cycle counts from different SMs are not
        // comparable.  The bridge converts record indices to host time
        // with a per-frame calibration.
        rec.timestamp = idx; rec.zone_id = zone_id; rec.type = type;
        rec._pad = 0; rec.lane = (uint16_t)(lid & 0xffff);
    }
    void push_msg(uint32_t h, uint32_t lid) const { push(h, DEV_MSG, lid); }
    void push_plot(uint32_t h, float v, uint32_t lid) const {
        if (!active() || dropped_lane(lid) || full()) return;
        uint32_t idx;
#ifdef KERNEL_NATIVE
        idx = __atomic_fetch_add(&header->write_pos, 1, __ATOMIC_RELAXED);
#else
        ::sycl::atomic_ref<uint32_t,::sycl::memory_order::relaxed,::sycl::memory_scope::device> wp(header->write_pos);
        idx = wp.fetch_add(1);
#endif
        if (idx >= capacity) return;
        DeviceRecord &rec = records[idx];
        uint32_t bits; memcpy(&bits, &v, sizeof(bits));
        rec.timestamp = bits; rec.zone_id = h; rec.type = DEV_PLOT;
        rec._pad = 0; rec.lane = (uint16_t)(lid & 0xffff);
    }
};

class DeviceZone {
public:
    DeviceZone(const DeviceRing &ring, uint32_t zid, uint32_t lid)
        : ring_(ring), zid_(zid), lid_(lid) {
        ring_.push(zid_, DEV_ZONE_ENTER, lid_);
    }
    ~DeviceZone() { ring_.push(zid_, DEV_ZONE_EXIT, lid_); }
private:
    DeviceRing ring_; uint32_t zid_; uint32_t lid_;
};

/// Interest-zone sampling guard (RAII — ends at scope exit).  Mark the
/// region you want sampled with PROFILER_INTEREST_BEGIN(ctxt, "name").
///
/// On construction it reads `header->interest_pct` (host-updated every
/// frame) and decides deterministically whether THIS work-item is
/// sampled in: `hash(lid) % 100 < pct`.  Sampled lanes record the zone
/// ENTER plus everything inside; dropped lanes set their per-lane flag
/// so every nested PROFILER_ZONE / MSG / PLOT record is skipped — the
/// whole trace for that work-item is dropped, which is what keeps the
/// ring from overflowing when a full frame's traces don't fit.
class InterestZone {
public:
    InterestZone(const DeviceRing &ring, uint32_t lid, uint32_t zid)
        : ring_(ring), lid_(lid), zid_(zid) {
        const uint32_t pct = ring_.header ? ring_.header->interest_pct : 100u;
        sample_ = (pct >= 100u) ||
                  (pct > 0u && (lane_hash(lid_) % 100u) < pct);
        if (ring_.sample_flags && lid_ < ring_.sample_count) {
            ring_.sample_flags[lid_] = sample_ ? 1u : 2u;
            if (sample_) ring_.push(zid_, DEV_ZONE_ENTER, lid_);
        } else {
            // No flags array sized for this lane → behave like a plain
            // zone (always record) so a missing buffer never loses data.
            ring_.push(zid_, DEV_ZONE_ENTER, lid_);
        }
    }
    ~InterestZone() { close(); }

    /// Deterministic per-lane decision hash (integer mix — cheap, no
    /// collisions along sequential pixel ids).
    static uint32_t lane_hash(uint32_t lane) {
        uint32_t h = lane;
        h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13;
        h *= 0xc2b2ae35u; h ^= h >> 16;
        return h;
    }
private:
    void close() {
        if (closed_) return;
        closed_ = true;
        if (ring_.sample_flags && lid_ < ring_.sample_count) {
            ring_.sample_flags[lid_] = 0u;
            if (sample_) ring_.push(zid_, DEV_ZONE_EXIT, lid_);
        } else {
            ring_.push(zid_, DEV_ZONE_EXIT, lid_);
        }
    }
    DeviceRing ring_;
    uint32_t lid_;
    uint32_t zid_;
    bool sample_ = true;
    bool closed_ = false;
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
#define PROFILER_ZONE_IN(...) ((void)0)
#define PROFILER_FUNCTION()  ((void)0)
#define PROFILER_FMT(...)    ((void)0)
#define PROFILER_MSG(...)    ((void)0)
#define PROFILER_MSG_FMT(...) ((void)0)
#define PROFILER_PLOT(...)   ((void)0)
#define PROFILER_ALLOC(...)  ((void)0)
#define PROFILER_FREE(...)   ((void)0)
#define PROFILER_FRAME_MARK  ((void)0)
#define PROFILER_INTEREST_BEGIN(...) ((void)0)

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
// Variant with an explicit per-work-item context (`ctxt`), for zones
// inside a foreach_pixel lambda where the implicit `ctx` is the frame
// context (linear_id 0 for every work-item) — e.g. `px` in rt::render.
#define PROFILER_ZONE_IN(ctxt, name) \
    ::profiler::DeviceZone SANDBOX_PZ_CAT(_pz_,__LINE__)( \
        (ctxt).prof, ::profiler::device_zone_hash(name), (uint32_t)(ctxt).linear_id)
#define PROFILER_FUNCTION() PROFILER_ZONE(__PRETTY_FUNCTION__)
#define PROFILER_FMT(...)   PROFILER_MSG_FMT(__VA_ARGS__)
#define PROFILER_MSG(t)  do{if(ctx.prof.active())ctx.prof.push_msg(::profiler::device_zone_hash(t),(uint32_t)ctx.linear_id);}while(0)
#define PROFILER_MSG_FMT(f,...) do{if(ctx.prof.active()){char _b_[64];snprintf(_b_,sizeof(_b_),f,##__VA_ARGS__);ctx.prof.push_msg(::profiler::device_zone_hash(_b_),(uint32_t)ctx.linear_id);}}while(0)
#define PROFILER_PLOT(n,v) do{if(ctx.prof.active())ctx.prof.push_plot(::profiler::device_zone_hash(n),(float)(v),(uint32_t)ctx.linear_id);}while(0)
#define PROFILER_ALLOC(...)  ((void)0)
#define PROFILER_FREE(...)   ((void)0)
#define PROFILER_FRAME_MARK  ((void)0)
// Interest-zone sampling: `ctxt` is the per-work-item Context in scope
// (e.g. `px` inside rt::render's foreach_pixel lambda); `name` is the
// zone name shown in Tracy.  RAII — the zone ends at scope exit.
#define PROFILER_INTEREST_BEGIN(ctxt, name) \
    ::profiler::InterestZone SANDBOX_PZ_CAT(_pzi_,__LINE__)( \
        (ctxt).prof, (uint32_t)(ctxt).linear_id, ::profiler::device_zone_hash(name))

#elif defined(SANDBOX_ENABLE_TRACY)
// ═══════════════════════════════════════════════════════════════════════
//  App host code — Tracy C++ client API
// ═══════════════════════════════════════════════════════════════════════
#include <tracy/Tracy.hpp>
#define PROFILER_ZONE(name)  ZoneNamedN(___tracy_scoped_zone, name, true)
#define PROFILER_ZONE_IN(ctxt, name) PROFILER_ZONE(name)
#define PROFILER_FUNCTION()  ZoneScoped
#define PROFILER_FMT(f,...)  ZoneNamedN(SANDBOX_PZ_CAT(___tracy_fmt_zone,__LINE__),({static char _b_[256];snprintf(_b_,sizeof(_b_),f,##__VA_ARGS__);_b_;}),true)
#define PROFILER_MSG(t)      TracyMessage(t, 0)
#define PROFILER_MSG_FMT(f,...) do{char _b_[256];snprintf(_b_,sizeof(_b_),f,##__VA_ARGS__);TracyMessage(_b_,0);}while(0)
#define PROFILER_PLOT(n,v)   TracyPlot(n, (double)(v))
#define PROFILER_ALLOC(p,s)  TracyAlloc(p,s)
#define PROFILER_FREE(p)     TracyFree(p)
#define PROFILER_FRAME_MARK   FrameMarkNamed("render")
#define PROFILER_INTEREST_BEGIN(...) ((void)0)

#else
// ═══════════════════════════════════════════════════════════════════════
//  Profiler disabled
// ═══════════════════════════════════════════════════════════════════════
#define PROFILER_ZONE(n)     ((void)0)
#define PROFILER_ZONE_IN(...) ((void)0)
#define PROFILER_FUNCTION()  ((void)0)
#define PROFILER_FMT(...)    ((void)0)
#define PROFILER_MSG(...)    ((void)0)
#define PROFILER_MSG_FMT(...) ((void)0)
#define PROFILER_PLOT(...)   ((void)0)
#define PROFILER_ALLOC(...)  ((void)0)
#define PROFILER_FREE(...)   ((void)0)
#define PROFILER_FRAME_MARK  ((void)0)
#define PROFILER_INTEREST_BEGIN(...) ((void)0)
// hash_zone_name / read_timestamp are defined inline above (shared by all
// paths) — no redefinition here.
#endif
