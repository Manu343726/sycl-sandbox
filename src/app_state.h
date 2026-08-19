#pragma once

#include <sycl-sandbox/kernel/execution_context.h>
#include <sycl-sandbox/scene_loader.h>
#include <sycl-sandbox/profiler.h>
#include "kernel/runtime.h"
#include "scene/registry.h"
#include "scene/host_scene.h"
#include "camera/orbit.h"
#include "display/texture.h"
#include "display/display_target.h"
#include "render/param_store.h"
#include "render/stat_publish.h"
#include "io/log_sink.h"
#include "ui/stat/panel.h"
#include "ui/build_monitor/panel.h"
#include "ui/metrics/system_metrics.h"
#include "tracy/tracy_bridge.h"

#include <sycl/sycl.hpp>
#include <GLFW/glfw3.h>
#include <atomic>
#include <thread>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <deque>
#include <mutex>

// ── AppState ─────────────────────────────────────────────────────────
/// Groups all mutable state for the sycl-sandbox application.
/// Passed by reference to all panel / rendering functions.
///
/// The KernelRuntime member owns all kernel/backend/scene subsystems and
/// orchestrates ordered reload sequences.  AppState owns UI state, render
/// thread control, profiling/stats, the camera, and the OpenGL texture.
struct AppState {
    // ── Window ───────────────────────────────────────────────────
    GLFWwindow *window = nullptr;
    int width = 1280;
    int height = 720;

    // ── Logging ──────────────────────────────────────────────────
    std::shared_ptr<LogSink> log_sink;

    // ── Kernel Runtime (orchestrator for all kernel/backend state) ─
    std::unique_ptr<KernelRuntime> kr;

    // ── Display pipeline ────────────────────────────────────────
    std::shared_ptr<DisplayTarget> display_target;
    GLuint tex = 0;
    FrameInfo last_frame_info{};

    // ── Thread-safe parameter / stat hand-off ───────────────────
    ParamStore param_store;
    PublishedStats published_stats;

    // ── Command mailbox (UI → render thread) ─────────────────────
    /// The UI thread never touches device/kernel state directly; it posts
    /// work here and the render thread drains the queue at the top of
    /// every loop iteration.  All device/queue operations therefore run
    /// on the render thread and the UI can never block on them.
    std::mutex cmd_mtx;
    std::deque<std::function<void()>> cmd_queue;

    /// UI thread: enqueue work for the render thread.  Capture by value;
    /// never capture UI-owned buffers the render thread would race on.
    void post_cmd(std::function<void()> fn) {
        std::lock_guard<std::mutex> lk(cmd_mtx);
        cmd_queue.push_back(std::move(fn));
    }

    /// Render thread only: run every queued command in FIFO order.
    /// Commands posted *by* commands land in the queue for the next pass.
    void drain_cmds() {
        std::deque<std::function<void()>> local;
        {
            std::lock_guard<std::mutex> lk(cmd_mtx);
            local.swap(cmd_queue);
        }
        for (auto &fn : local) fn();
    }

    // ── Deferred-op request helpers (UI thread) ──────────────────
    /// Request a scene switch.  The pipeline is gated (kernel_ready down)
    /// until the render thread finishes applying the switch and the UI has
    /// re-run on_scene_changed() for the new scene generation.  `scene`
    /// points into the (stable) SceneRegistry.  `w`/`h` default to -1 =
    /// keep the render thread's current resolution (runtime switches);
    /// the startup load passes the live framebuffer size so the first
    /// switch also sizes the accumulation buffer.
    void request_scene_switch(const SceneDef *scene, int w = -1, int h = -1) {
        kernel_ready.store(false);
        pending_device_ops.fetch_add(1);
        post_cmd([this, scene, w, h] {
            // Size the per-lane profiler sample flags to the new pixel
            // count (runtime switches keep the current resolution).
            int nw = w, nh = h;
            if (nw < 0 || nh < 0) { nw = kr->width(); nh = kr->height(); }
            if (nw > 0 && nh > 0)
                ensure_profiler_sample_flags((uint32_t)(nw * nh));
            bool ok = kr->apply_scene_switch(*scene, w, h);
            if (ok) {
                current_spp.store(0);
                target_spp.store(scene->max_spp);
                tick.store(0);
                scene_start_time.store(std::chrono::steady_clock::now());
                scene_generation.fetch_add(1);
            }
            pending_device_ops.fetch_sub(1);
        });
    }

    /// Flag a pending resize at `w`×`h`; a UI-frame step posts the actual
    /// (coalesced) command.  Gating: kernel_ready stays down until the
    /// render thread re-allocates the accumulation buffer and the UI has
    /// resized the display target.
    void request_resize(int w, int h) {
        std::lock_guard<std::mutex> lk(resize_mtx);
        if ( w != resize_w || h != resize_h ) {
            // The requested size changed — restart the settle clock.  The
            // viewport re-requests on every frame its region differs from
            // the applied size, so identical requests must NOT reset it
            // (otherwise the debounce could never expire).
            resize_w = w;
            resize_h = h;
            resize_last_change = std::chrono::steady_clock::now();
        }
        resize_pending = true;
    }

    /// Bounded wait for the render thread to leave its frame scope.
    /// Only used for backend switches, where the GL display target must be
    /// torn down before the render thread swaps queues.  Returns once the
    /// current frame completes — never waits on builds or reloads.
    void wait_render_idle() {
        for (int i = 0; i < 5000 && render_busy.load(); i++)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    

    // ── Deferred UI→render operations ───────────────────────────
    /// The UI takes kernel_ready down *before* posting a gating command
    /// (scene switch, backend switch, resize, profiler toggle).  The
    /// render thread drains commands at the top of every loop iteration —
    /// the device is always idle there, since a frame completes before the
    /// loop returns to the top — applies the op, and acks.  The UI does
    /// any GL/display-side work when it sees the ack, then re-raises
    /// kernel_ready once no ops are outstanding.  Commands that don't
    /// touch the display (re-init) never gate rendering.
    std::atomic<int> pending_device_ops{0};  ///< gating ops in flight

    /// Bumped by the render thread after it applied any op that may have
    /// changed the scene *layout* (scene switch, backend switch, hot
    /// reload, profiler toggle).  The UI then re-runs on_scene_changed()
    /// (camera refs, stat store, param store reset).
    std::atomic<uint64_t> scene_generation{0};
    uint64_t last_processed_generation = 0;  ///< UI thread only

    /// Render-thread ack for a resize: the device-side accumulation buffer
    /// was re-allocated; the UI now finishes the GL/display side
    /// (display_target->resize, texture refresh) and re-raises kernel_ready.
    std::atomic<bool> resize_applied{false};

    /// Render-thread ack for a backend switch: the device side swapped
    /// queues and reloaded the kernel; the UI now recreates the display
    /// target on the new queue and re-raises kernel_ready.
    std::atomic<bool> backend_switch_applied{false};
    /// Resolution the render thread just applied (read by the UI when it
    /// finishes the GL/display side of a resize).
    std::atomic<int> applied_resize_w{0};
    std::atomic<int> applied_resize_h{0};
    /// Param values to restore after a backend switch (same scene re-parsed,
    /// so the layout matches).  UI thread only.
    std::vector<float> backend_restore_params;

    // ── Coalesced resize (window drags fire many events) ────────
    /// The viewport panel flags a pending resize; a UI-frame step snapshots
    /// the latest dims and posts at most one command at a time, so a drag
    /// collapses into a single device-side resize with the final size.
    /// Requests are also debounced: the viewport re-requests on every frame
    /// its region differs from the applied size, so a region that never
    /// settles (e.g. a dock-layout fight at startup) would otherwise gate
    /// the pipeline forever.  The kickoff only fires once the requested
    /// size has held still for `resize_settle_ms`.
    std::mutex resize_mtx;
    int resize_w = 0, resize_h = 0;          ///< UI thread only
    bool resize_pending = false;             ///< UI thread only
    std::chrono::steady_clock::time_point resize_last_change{};  ///< UI thread only
    std::atomic<bool> resize_posted{false};  ///< at most one cmd in flight

    // ── Coalesced re-init (param change → render thread) ─────────
    /// Params are edited in scene_desc->current_buffer (UI thread only).
    /// On a param change the UI snapshots the latest values here and
    /// posts at most one reinit command at a time; rapid drags collapse
    /// into a single re-init with the final values.
    std::mutex reinit_mtx;
    std::vector<float> reinit_params;
    bool reinit_pending = false;
    std::atomic<bool> reinit_posted{false};

    /// Rebind everything that depends on the active scene/kernel: camera
    /// pointers, the stat seqlock size, pending param snapshots.  Call
    /// from every scene-switch / backend-switch / hot-reload completion
    /// path, while kernel_ready is still false.
    void on_scene_changed() {
        refresh_camera_ptrs();
        published_stats.resize(kr ? kr->stat_buffer().size() : 0);
        param_store.reset();
        auto scene_desc = kr ? kr->scene_desc() : nullptr;
        if (scene_desc) stat_store.init_from(scene_desc->stats);
        orbit_init = false;
    }

    // ── Scene system (loaded at startup, static list) ──────────
    std::unique_ptr<SceneRegistry> scenes;

    // ── Render state ────────────────────────────────────────────
    std::atomic<int> current_spp{0};
    std::atomic<int> target_spp{1};
    /// Animation frame counter — increments each sample (like Shadertoy iFrame).
    std::atomic<uint64_t> tick{0};
    /// Animation clock — steady time-point for elapsed-time computation.
    /// Atomic: written by the UI (epoch processing) on scene changes,
    /// read by the render thread for the `time` param.
    std::atomic<std::chrono::steady_clock::time_point> scene_start_time{};

    // ── Device profiler ring ─────────────────────────────────
    rt::Buffer<profiler::DeviceRingHeader> d_ring_header_;
    rt::Buffer<profiler::DeviceRecord> d_ring_records_;
    /// Ring capacity (record count).  MUST be a power of two —
    /// DeviceRing::push masks with idx & (capacity - 1).  Atomic: the
    /// UI thread reads it for display; the render thread re-allocates
    /// the ring (request_ring_resize → render-thread closure →
    /// init_profiler_buffers).  The closure updates this AFTER the new
    /// buffers are live so device_ring() never sees a mismatched cap.
    std::atomic<uint32_t> ring_capacity{256 * 1024};
    /// Default initial capacity (also the minimum the UI offers).
    static constexpr uint32_t RING_CAPACITY_DEFAULT = 256 * 1024;

    /// Interest-zone sampling percentage (0-100), set from the UI
    /// ("Profiler" panel) and uploaded to the device ring header every
    /// frame by the render loop.  PROFILER_INTEREST_BEGIN regions decide
    /// per work-item whether to record (hash(lane) % 100 < pct) or drop.
    std::atomic<uint32_t> profiler_interest_pct{10};
    /// Per-lane interest-zone state array (see profiler::DeviceRing::
    /// sample_flags).  Sized to the current pixel count; managed by
    /// ensure_profiler_sample_flags on the render thread.
    rt::Buffer<uint8_t> d_prof_sample_flags_;

    /// Round x up to the next power of two (or itself if already a
    /// power of two).  Used by the UI to recommend a capacity that
    /// covers the last frame's record count without overflow.
    static uint32_t next_pow2(uint32_t x) {
        if (x <= 1) return 1;
        --x;
        x |= x >> 1;  x |= x >> 2;  x |= x >> 4;
        x |= x >> 8;  x |= x >> 16;
        return x + 1;
    }

    void init_profiler_buffers(rt::Runtime *rt, sycl::queue *q,
                                uint32_t cap = 0) {
        free_profiler_buffers();
        if (!rt->pool) return;
        if (cap == 0) cap = ring_capacity.load();
        // The device profiler ring. On every backend the storage lives at
        // one address that is both writable from kernel code and readable
        // from the app side:
        //  - GPU / SYCL-CPU backend (q != nullptr): USM shared via
        //    sycl::malloc_shared, written from parallel_for via
        //    sycl::atomic_ref and read back by the Tracy bridge with
        //    q->memcpy (ordered against in-flight device writes).
        //  - Software backend (q == nullptr): plain host heap (the pool's
        //    raw_alloc falls back to ::operator new + memset). The kernel
        //    .so runs synchronously under OpenMP, so by the time
        //    call_kernel_entry returns all atomic writes are visible to
        //    the host; the bridge reads them with std::memcpy.
        // Both go through alloc_shared so backend-switch teardown
        // (release_all on the old queue / pool) is uniform.
        d_ring_header_ = rt->pool->alloc_shared<profiler::DeviceRingHeader>(
            1, ::rt::MemScope::App);
        d_ring_records_ = rt->pool->alloc_shared<profiler::DeviceRecord>(
            cap, ::rt::MemScope::App);
        rt->fill(d_ring_header_.data, 0, sizeof(profiler::DeviceRingHeader));
        ring_capacity.store(cap);
    }
    void free_profiler_buffers() {
        d_ring_header_ = {};
        d_ring_records_ = {};
        // NOTE: d_prof_sample_flags_ is intentionally NOT freed here —
        // this runs on ring resize too (request_ring_resize), where the
        // flags array (sized to pixels, independent of the ring) must
        // survive.  Backend switch frees it implicitly via
        // pool_.release_all(); ensure_profiler_sample_flags detects the
        // stale handle (pool->owns() is false) and re-allocates against
        // the rebound pool.
    }
    profiler::DeviceRing device_ring(size_t work_items) const {
        if (!d_ring_header_.data) return {};
        (void)work_items;
        profiler::DeviceRing ring;
        ring.header = d_ring_header_.data;
        ring.records = d_ring_records_.data;
        ring.capacity = ring_capacity.load();
        ring.sample_flags = d_prof_sample_flags_.count
                                ? d_prof_sample_flags_.data : nullptr;
        ring.sample_count = (uint32_t)d_prof_sample_flags_.count;
        return ring;
    }

    /// Re-allocate the per-lane sample-flags array to cover at least `n`
    /// work items (called on the render thread whenever the pixel count
    /// changes — scene switch / resize — and after a backend switch, so
    /// the array is allocated against the current pool).  Zero-fills it:
    /// value 0 means "not inside an interest zone → record", the default
    /// before any guard runs.  Only ever grows; shrinking keeps the
    /// larger allocation.
    ///
    /// The `pool->owns()` check is the backend-switch safety net: a
    /// backend switch rebinds the pool and calls release_all(), which
    /// FREES this array while the rt::Buffer handle keeps pointing at the
    /// freed memory (free_profiler_buffers deliberately does NOT reset
    /// it — ring resizes must keep the flags).  Without the check the
    /// early-return above would keep the stale pointer and the device
    /// kernel would read freed memory (crash in DeviceRing::dropped_lane
    /// on `sample_flags[lid] == 2`).
    void ensure_profiler_sample_flags(uint32_t n) {
        if (d_prof_sample_flags_.data &&
            d_prof_sample_flags_.count >= n &&
            d_prof_sample_flags_.pool &&
            d_prof_sample_flags_.pool->owns(d_prof_sample_flags_.data))
            return;
        if (!kr || !kr->runtime().pool) return;
        d_prof_sample_flags_ = {};
        d_prof_sample_flags_ =
            kr->runtime().pool->alloc_shared<uint8_t>(n, ::rt::MemScope::App);
        if (d_prof_sample_flags_.data)
            kr->runtime().fill(d_prof_sample_flags_.data, 0, n);
    }

    /// Push the current interest-sampling percentage to the device ring
    /// header.  Called by the render loop BEFORE every dispatch: the
    /// Tracy bridge memsets the whole header to zero after draining, so
    /// this must happen every frame, not just on change.  One 4-byte
    /// in-order memcpy on SYCL; plain store on the software backend.
    void upload_profiler_pct(sycl::queue *q, uint32_t pct) {
        if (!d_ring_header_.data) return;
        if (pct > 100) pct = 100;
        if (q)
            q->memcpy(&d_ring_header_.data->interest_pct, &pct, sizeof(pct))
                 .wait();
        else
            std::memcpy(&d_ring_header_.data->interest_pct, &pct, sizeof(pct));
    }

    /// Request a ring-capacity change.  `new_cap` must be a power of
    /// two.  The actual re-alloc (free old buffers + alloc new ones via
    /// init_profiler_buffers) runs as a command on the render thread at
    /// loop top, where no frame is in flight — the per-frame kernel
    /// writes and the bridge drain are sequenced around it so there is
    /// no race on the device-ring pointers.  ring_capacity is updated
    /// INSIDE the closure (after the new buffers are live) so
    /// device_ring() never sees a mismatched cap.
    void request_ring_resize(uint32_t new_cap) {
        // Enforce power-of-two (the ring masks with idx & (cap-1)).
        if (new_cap < 2) return;
        if ((new_cap & (new_cap - 1)) != 0) new_cap = next_pow2(new_cap);
        post_cmd([this, new_cap] {
            init_profiler_buffers(&kr->runtime(), kr->queue_ptr(), new_cap);
        });
    }

    StatStore stat_store;
    BuildMonitor build_monitor;

    // ── Tracy profiler (optional, SANDBOX_ENABLE_TRACY) ───────────
    /// Forwards device-ring records to the Tracy client as GPU zones.
    /// The profiler UI is Tracy's own standalone tracy-profiler app,
    /// launched from the Controls panel (see tracy/tracy_launcher.*).
    tracy_bridge::Bridge tracy_bridge;

    // ── System resource metrics (CPU / RAM / GPU) ────────────────
    SystemMetrics system_metrics;

    // ── Render thread ──────────────────────────────────────────
    std::atomic<bool> render_paused{false};
    std::atomic<bool> render_running{true};
    std::atomic<bool> kernel_ready{false};
    /// True while the render thread is inside a frame iteration or the
    /// loop-top op window.  The UI uses this to avoid raising kernel_ready
    /// while the render thread is applying an op / reloading, and for the
    /// bounded idle-wait before display-target teardown (backend switch).
    std::atomic<bool> render_busy{false};
    std::thread render_thread;

    // ── Timing diagnostics (ms, published by render thread, read by UI) ─
    /// Pure GPU kernel time (call_kernel_entry → queue wait), ms.
    std::atomic<double> kernel_time_ms{0.0};
    /// call_kernel_entry alone (enqueue on GPU, synchronous render on CPU).
    std::atomic<double> kernel_call_ms{0.0};
    /// queue wait time (GPU backend only; 0 on software).
    std::atomic<double> kernel_wait_ms{0.0};
    /// Render-loop wall-clock interval between iterations, ms (includes spin).
    std::atomic<double> render_loop_time_ms{0.0};
    /// Wall-clock interval between productive frames, ms (SPP-incrementing).
    std::atomic<double> render_interval_ms{0.0};
    /// Main-thread frame time (poll → composite), ms.
    std::atomic<double> ui_frame_time_ms{0.0};
    /// Main-thread phase breakdowns, ms.
    std::atomic<double> ui_poll_events_ms{0.0};
    std::atomic<double> ui_rebuild_ms{0.0};
    std::atomic<double> ui_stats_ms{0.0};
    std::atomic<double> ui_imgui_ms{0.0};
    std::atomic<double> ui_upload_ms{0.0};
    std::atomic<double> ui_composite_ms{0.0};

    // ── Camera ─────────────────────────────────────────────────
    OrbitCam orbit;
    bool orbit_init = false;

    // ── Cached camera ParamRefs (refreshed each scene change) ──
    scene_loader::ParamRef camera_eye;
    scene_loader::ParamRef camera_at;
    scene_loader::ParamRef camera_up;
    scene_loader::ParamRef fov;
    scene_loader::ParamRef aperture;
    scene_loader::ParamRef center_x;
    scene_loader::ParamRef center_y;
    scene_loader::ParamRef zoom;

    // ── UI toggles ──────────────────────────────────────────────
    bool show_builds = false;
    bool show_logs = false;
    bool show_metrics = false;
    bool show_test_engine = false;
    /// Whether on-the-fly kernel builds compile with profiler support
    /// (SANDBOX_ENABLE_PROFILER in the CMake cache).  Mirrors the cache
    /// value at startup; the Controls checkbox toggles it by
    /// reconfiguring + rebuilding the active kernel.
    bool profiler_enabled = true;

    // ── Convenience helpers ────────────────────────────────────
    /// Build directory the app was launched from (build/ or build_debug/),
    /// used to locate the standalone tracy-profiler executable.
    std::string build_dir;

    bool has_3d() const { return camera_eye.valid() && camera_at.valid() && fov.valid(); }
    bool has_2d() const { return center_x.valid() && center_y.valid() && zoom.valid(); }

    void refresh_camera_ptrs() {
        auto scene_desc = kr ? kr->scene_desc() : nullptr;
        if (!scene_desc) {
            camera_eye = camera_at = camera_up = scene_loader::ParamRef{};
            fov = aperture = scene_loader::ParamRef{};
            center_x = center_y = zoom = scene_loader::ParamRef{};
            return;
        }
        camera_eye  = scene_desc->find_param_ref("cam_eye");
        camera_at   = scene_desc->find_param_ref("cam_at");
        fov         = scene_desc->find_param_ref("cam_fov");
        aperture    = scene_desc->find_param_ref("cam_aperture");
        camera_up   = scene_desc->find_param_ref("cam_up");
        center_x    = scene_desc->find_param_ref("center_x");
        center_y    = scene_desc->find_param_ref("center_y");
        zoom        = scene_desc->find_param_ref("zoom");
    }
};
