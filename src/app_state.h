#pragma once

#include <sycl-sandbox/kernel/execution_context.h>
#include <sycl-sandbox/scene_loader.h>
#include <sycl-sandbox/profiler.h>
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

    /// Stop the render thread from starting a new frame and drain the
    /// device queue.  Main thread only.  Restore with resume_pipeline().
    /// Required before any operation that frees device memory the render
    /// thread may be using (resize, backend switch).
    /// Sets the kernel cancellation flag so in-flight work can bail out
    /// early while we wait for the frame to finish.
    bool pause_pipeline() {
        if (kr) kr->cancel();
        bool was_ready = kernel_ready.exchange(false);
        while (render_busy.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (kr) kr->drain();
        return was_ready;
    }

    void resume_pipeline(bool was_ready) {
        kernel_ready.store(was_ready);
    }

    void recreate_buffers(int w, int h) {
        bool was_ready = pause_pipeline();
        width = w;
        height = h;
        if (kr) {
            kr->resize(w, h);
            kr->reinit_kernel();   // kernels re-derive per-resolution state
        }
        if (display_target) {
            display_target->resize(w, h);
            tex = display_target->texture();
        }
        current_spp = 0;
        resume_pipeline(was_ready);
    }

    /// Rebind everything that depends on the active scene/kernel: camera
    /// pointers, the stat seqlock size, pending param snapshots.  Call
    /// from every scene-switch / backend-switch / hot-reload completion
    /// path, while kernel_ready is still false.
    void on_scene_changed() {
        refresh_camera_ptrs();
        published_stats.resize(kr ? kr->stat_buffer().size() : 0);
        param_store.reset();
        auto *scene_desc = kr ? kr->scene_desc() : nullptr;
        if (scene_desc) stat_store.init_from(scene_desc->stats);
    }

    // ── Scene system (loaded at startup, static list) ──────────
    std::unique_ptr<SceneRegistry> scenes;

    // ── Render state ────────────────────────────────────────────
    std::atomic<int> current_spp{0};
    std::atomic<int> target_spp{1};
    /// Animation frame counter — increments each sample (like Shadertoy iFrame).
    std::atomic<uint64_t> tick{0};
    /// Animation clock — steady time-point for elapsed-time computation.
    std::chrono::steady_clock::time_point scene_start_time;

    // ── Device profiler ring ─────────────────────────────────
    profiler::DeviceRingHeader *d_ring_header_ = nullptr;
    profiler::DeviceRecord *d_ring_records_ = nullptr;
    sycl::queue *ring_q_ = nullptr;
    static constexpr uint32_t RING_CAPACITY = 256 * 1024;
    static constexpr size_t   SAMPLED_ITEMS = 1024;

    void init_profiler_buffers(rt::Runtime *, sycl::queue *q) {
        free_profiler_buffers();
        if (!q) return;
        ring_q_ = q;
        // USM shared — the ring is written from device code (via
        // sycl::atomic_ref) and read from host code (kernel .so
        // KERNEL_BUILD path via sycl::atomic_ref).  malloc_device
        // is not host-accessible on CUDA.
        d_ring_header_ = sycl::malloc_shared<profiler::DeviceRingHeader>(1, *q);
        d_ring_records_ = sycl::malloc_shared<profiler::DeviceRecord>(RING_CAPACITY, *q);
        q->memset(d_ring_header_, 0, sizeof(profiler::DeviceRingHeader)).wait();
    }
    void free_profiler_buffers() {
        if (ring_q_) {
            if (d_ring_header_) sycl::free(d_ring_header_, *ring_q_);
            if (d_ring_records_) sycl::free(d_ring_records_, *ring_q_);
        }
        d_ring_header_ = nullptr;
        d_ring_records_ = nullptr;
        ring_q_ = nullptr;
    }
    profiler::DeviceRing device_ring(size_t work_items) const {
        if (!d_ring_header_) return {};
        profiler::DeviceRing ring;
        ring.header = d_ring_header_;
        ring.records = d_ring_records_;
        ring.capacity = RING_CAPACITY;
        ring.sample_interval =
            (uint32_t)std::max<size_t>(1, work_items / SAMPLED_ITEMS);
        return ring;
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
    /// True while the render thread is inside a frame iteration.  Paired
    /// with kernel_ready by pause_pipeline() to stop the pipeline safely.
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
        auto *scene_desc = kr ? kr->scene_desc() : nullptr;
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
