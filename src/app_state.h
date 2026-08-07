#pragma once

#include <sycl-sandbox/kernel/execution_context.h>
#include <sycl-sandbox/scene_loader.h>
#include "kernel/runtime.h"
#include "kernel/profiler_host.h"
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

#include <sycl/sycl.hpp>
#include <GLFW/glfw3.h>
#include <atomic>
#include <thread>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

// ── Forward declarations for kernel API functions ────────────────────
using get_scene_debug_info_fn = const SceneDebugInfo *(*)();

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
    bool pause_pipeline() {
        bool was_ready = kernel_ready.exchange(false);
        while (render_busy.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (kr) kr->drain();
        return was_ready;
    }

    void resume_pipeline(bool was_ready) { kernel_ready.store(was_ready); }

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

    // ── Profiler + stats ────────────────────────────────────────
    KernelProfiler kernel_profiler;
    StatStore stat_store;
    BuildMonitor build_monitor;

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
    bool show_profiler = false;
    bool show_metrics = false;
    bool show_test_engine = false;

    // ── Convenience helpers ────────────────────────────────────
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
