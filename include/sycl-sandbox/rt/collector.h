#pragma once

/// @file
/// The trace collector — the per-call observation buffer threaded through
/// the whole raytracing pipeline inside the kernel rt::Context.
///
/// It is a device-safe POD ring (mirroring profiler::DeviceRing): the
/// pipeline appends tagged TraceEvent records (deep metadata: BVH nodes
/// entered, primitive hit tests, material scatters/emits, texture samples)
/// and TraceStepRecord snapshots (one per completed path step) into two
/// flat rings.  The caller arms it over host or device memory and reads
/// the rings back after the trace completes — "written by the kernel,
/// read by the app".
///
/// The render path leaves the collector inactive ({}): every hook guards
/// on active() and compiles away to a branch on a null pointer, so the
/// instrumentation costs nothing on the device.  The scene-debug view
/// (src/ui/scene_debug/ray_trace.h) arms it on the host and runs the REAL
/// rt::trace() against it — one implementation, one observation channel.
///
/// No vtables, no templates, no polymorphism: the hooks are non-virtual
/// member calls on the concrete TraceCollector, so the same headers
/// compile for SYCL device code, the kernel .so host pipeline, and the
/// host app.
///
/// Rings are sized by the caller and MUST be power-of-two: records wrap
/// via a (capacity - 1) mask, exactly like the profiler ring.

#include <sycl-sandbox/rt/types_fwd.h>

#include <cstdint>
#include <cstddef>

#ifndef KERNEL_NATIVE
#include <sycl/sycl.hpp>
#endif

namespace rt {

struct TraceStepRecord;   ///< defined in scene/data.h

// ── TraceCounters ─────────────────────────────────────────────────────

/// Per-frame ray tracing counters, accumulated on the DEVICE during a
/// render call (atomic increments inside rt::trace()) and read back by
/// the host after the frame completes.  The host zeroes the buffer
/// before enqueuing each render kernel.
struct TraceCounters {
    uint32_t num_hits = 0;      ///< closest hits found (all bounces)
    uint32_t num_bvh_hits = 0;  ///< of those, found via BVH traversal

    /// Device-safe atomic increment of the hit counters.
    /// Software (KERNEL_NATIVE) builds render single-threaded — plain
    /// increments suffice there.
    void add_hit(bool via_bvh) {
#ifdef KERNEL_NATIVE
        ++num_hits;
        if (via_bvh) ++num_bvh_hits;
#else
        ::sycl::atomic_ref<uint32_t, ::sycl::memory_order::relaxed,
                           ::sycl::memory_scope::device> h(num_hits);
        h.fetch_add(1);
        if (via_bvh) {
            ::sycl::atomic_ref<uint32_t, ::sycl::memory_order::relaxed,
                               ::sycl::memory_scope::device> b(num_bvh_hits);
            b.fetch_add(1);
        }
#endif
    }
};

// ── Collector event ring ───────────────────────────────────────────────

/// Kind of a deep-pipeline trace event (see TraceEvent).  Each fires
/// while the pipeline evaluates one path step; the reader attaches the
/// events to the step via TraceStepRecord::event_start.
enum class TraceEventKind : uint8_t {
    BvhNode         = 0, ///< payload: scene-BVH node index entered
    HitTest         = 1, ///< payload: HittableType (fires even on misses)
    Scatter         = 2, ///< payload: MaterialType
    Emit            = 3, ///< payload: MaterialType
    TextureSample   = 4, ///< payload: texture variant index
    HittableBvhNode = 5, ///< payload: per-mesh BVH node index entered
                         ///< (into SceneView::mesh_bvh_nodes)
};

/// One deep-pipeline event written by the collector hooks.
struct TraceEvent {
    uint8_t kind;       ///< TraceEventKind
    uint8_t _pad[3];
    uint32_t payload;   ///< node index / type tag / texture kind
};

/// Ring headers for the collector rings (host zeroes before arming).
struct TraceCollectorHeader {
    uint32_t event_pos = 0;  ///< next free TraceEvent slot
    uint32_t step_pos = 0;   ///< next free TraceStepRecord slot
    uint32_t _pad[2];
};

/// Per-call trace collector — POD ring handle carried in rt::Context.
///
/// The pipeline writes through it at every depth: BVH traversal reports
/// node visits, primitives report hit tests, materials report scatter/
/// emit, textures report samples, and rt::trace() appends one
/// TraceStepRecord per completed path step.  Inactive by default on the
/// render path — every hook is a guard on active() and is eliminated.
///
/// event_capacity / step_capacity MUST be powers of two (records wrap
/// via mask).  Steps reference their events via event_start: the events
/// belonging to step i are [step[i-1].event_start, step[i].event_start)
/// (or [0, step[0].event_start) for the first step).
struct TraceCollector {
    TraceCollectorHeader *header = nullptr;
    TraceEvent *events = nullptr;
    TraceStepRecord *steps = nullptr;   ///< ring of path step snapshots
    uint32_t event_capacity = 0;        ///< power of two
    uint32_t step_capacity = 0;         ///< power of two

    bool active() const {
        return header && events && steps && event_capacity > 0 &&
               step_capacity > 0;
    }

    /// Relaxed fetch-add of a ring write position.  Device code uses a
    /// SYCL atomic (work items race on the shared rings); host code
    /// (kernel host pipeline, scene-debug render thread) uses a plain
    /// GCC/Clang atomic builtin.
    static uint32_t bump(uint32_t *pos) {
#ifdef __SYCL_DEVICE_ONLY__
        ::sycl::atomic_ref<uint32_t, ::sycl::memory_order::relaxed,
                           ::sycl::memory_scope::device> r(*pos);
        return r.fetch_add(1);
#else
        return __atomic_fetch_add(pos, 1, __ATOMIC_RELAXED);
#endif
    }

    // ── Deep-pipeline hooks ────────────────────────────────────────

    void on_bvh_node(uint32_t node_index) const {
        if (!active()) return;
        TraceEvent e{};
        e.kind = (uint8_t)TraceEventKind::BvhNode;
        e.payload = node_index;
        events[bump(&header->event_pos) & (event_capacity - 1)] = e;
    }
    void on_hittable_bvh_node(uint32_t node_index) const {
        if (!active()) return;
        TraceEvent e{};
        e.kind = (uint8_t)TraceEventKind::HittableBvhNode;
        e.payload = node_index;
        events[bump(&header->event_pos) & (event_capacity - 1)] = e;
    }
    void on_hit_test(HittableType kind, const Ray &, float, float) const {
        if (!active()) return;
        TraceEvent e{};
        e.kind = (uint8_t)TraceEventKind::HitTest;
        e.payload = (uint32_t)kind;
        events[bump(&header->event_pos) & (event_capacity - 1)] = e;
    }
    void on_scatter(MaterialType kind, const Ray &, const HitRecord &) const {
        if (!active()) return;
        TraceEvent e{};
        e.kind = (uint8_t)TraceEventKind::Scatter;
        e.payload = (uint32_t)kind;
        events[bump(&header->event_pos) & (event_capacity - 1)] = e;
    }
    void on_emit(MaterialType kind, const HitRecord &) const {
        if (!active()) return;
        TraceEvent e{};
        e.kind = (uint8_t)TraceEventKind::Emit;
        e.payload = (uint32_t)kind;
        events[bump(&header->event_pos) & (event_capacity - 1)] = e;
    }
    void on_texture_sample(uint32_t kind) const {
        if (!active()) return;
        TraceEvent e{};
        e.kind = (uint8_t)TraceEventKind::TextureSample;
        e.payload = kind;
        events[bump(&header->event_pos) & (event_capacity - 1)] = e;
    }

    /// Append a completed path step (rt::trace()).  Defined in
    /// scene/data.h, where TraceStepRecord is complete; stamps the
    /// step's event_start so the reader can attach the events that
    /// fired while this step was evaluated.
    void record_step(const TraceStepRecord &r) const;
};

} // namespace rt
