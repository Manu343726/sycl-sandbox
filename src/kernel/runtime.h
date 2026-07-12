#pragma once

#include "library.h"
#include "build_system.h"
#include "kernel/dispatch.h"
#include "scene/registry.h"
#include "scene/host_scene.h"
#include "io/watcher.h"

#ifndef KERNEL_NATIVE
#include <sycl/sycl.hpp>
#endif

#include <string>
#include <vector>
#include <memory>
#include <functional>

// ── Backend info ──────────────────────────────────────────────────────
struct BackendInfo {
    std::string label;
    std::string id;
    bool available;
};

// ── KernelRuntime ─────────────────────────────────────────────────────
/// Central orchestrator that owns all kernel / backend / scene subsystems.
///
/// Ownership:
///   - Backend probing + selection
///   - SYCL queue + rt::Runtime (memory abstraction)
///   - KernelLibrary (kernel .so loading)
///   - KernelBuildSystem (CMake builds)
///   - SourceWatcher (inotify + .d file dependency tracking)
///   - Active kernel handle + scene descriptor
///   - Accumulation buffers (allocated through Runtime)
///   - HostScene (YAML scene builder state)
///
/// Any change (scene switch, backend switch, hot reload, param update)
/// goes through the internal ordered_reload() sequence that guarantees
/// correct shutdown → rebuild → load → init ordering.
class KernelRuntime {
public:
    /// @param build_dir     CMake build directory (absolute).
    /// @param project_root  Project root for source-file filtering.
    /// @param preferred_backend "gpu" or "cpu" for initial selection.
    KernelRuntime(const std::string &build_dir,
                  const std::string &project_root,
                  const std::string &preferred_backend);
    ~KernelRuntime();

    KernelRuntime(const KernelRuntime &) = delete;
    KernelRuntime &operator=(const KernelRuntime &) = delete;

    // ── Setup ────────────────────────────────────────────────────
    /// Register all known kernels with the watcher + build system.
    /// Call once after construction, before the event loop.
    void setup_all_kernels(SceneRegistry &scenes);

    // ── Lifecycle operations ─────────────────────────────────────
    /// Full ordered scene switch: cancel → build → load → init.
    /// On success the caller should:
    ///   - Reset SPP counters (current_spp = 0, target_spp = kernel()->desc.max_spp)
    ///   - Set tick = 0, scene_start_time = now
    ///   - Set kernel_ready = true, render_paused = false
    ///   - Re-init orbit camera
    /// Returns true if the kernel was loaded successfully.
    bool switch_scene(const SceneDef &scene, int width, int height);

    /// Full ordered backend switch: free resources → new queue → rebuild → reload.
    /// @return The index of the newly activated backend.
    int switch_backend(int new_backend, int width, int height);

    /// Re-init the kernel after a param or camera change (no rebuild).
    /// Copies the latest param buffer to the device and calls init_kernel
    /// with the last-known width/height.
    void reinit_kernel();

    /// Poll the source watcher for dirty files and check build results.
    /// If a kernel was hot-reloaded, triggers the full ordered_reload
    /// and returns the kernel name (empty string otherwise).
    /// `pre_reload` (optional) runs right before the reload swaps state —
    /// the app uses it to stop the render thread and drain the queue.
    std::string poll_hot_reload(
        const std::function<void()> &pre_reload = {});

    // ── Window resize ────────────────────────────────────────────
    /// Update internal width/height and re-allocate accumulation buffers.
    void resize(int w, int h);

    // ── Accumulation ops ─────────────────────────────────────────
    // Single persistent accumulation buffer.  Kernels ADD samples into it
    // (rgb += sample; alpha += 1); the host clears it only when the sample
    // stream restarts (scene/param/camera change, resize).  Progressive
    // convergence therefore just keeps summing into one buffer — no
    // ping-pong, no cross-thread pointer swap.
    void clear_accum();
    /// Enqueue-only clear (render thread, per-frame for animated kernels).
    /// The in-order queue sequences it before the next render kernel.
    void clear_accum_async();
    void fill_zero(float *buf);

    /// Block until all enqueued device work has completed.  MUST be called
    /// before freeing any allocation an enqueued kernel may reference —
    /// sycl::free does not wait for in-flight work.
    void drain();

    // ── Accessors ────────────────────────────────────────────────
    sycl::queue &queue() { return *q_; }
    /// Null on the software (native) backend — callers must branch to the
    /// synchronous CPU path instead of dereferencing.
    sycl::queue *queue_ptr() { return q_; }
    rt::Runtime &runtime() { return krt_; }

    /// Per-frame stat writer for RenderContext::stats.  Points at a
    /// runtime-owned buffer (NOT the scene descriptor's — the UI thread
    /// reads that one); the render thread publishes it through the
    /// PublishedStats seqlock after each frame.  Null when the scene has
    /// no stats.
    const rt::StatWriter *stat_writer() const {
        return stat_buffer_.empty() ? nullptr : &stat_writer_;
    }
    const std::vector<float> &stat_buffer() const { return stat_buffer_; }

    KernelHandle *kernel() const { return active_kernel_; }
    const SceneDef *scene() const { return active_scene_; }
    scene_loader::SceneDescriptor *scene_desc() const {
        return active_scene_desc_.get();
    }
    get_scene_debug_info_fn scene_debug_fn() const {
        return scene_debug_fn_;
    }

    float *d_params() const { return d_params_; }

    float *d_accum() const { return d_accum_; }
    size_t accum_bytes() const { return accum_bytes_; }
    int width() const { return width_; }
    int height() const { return height_; }
    size_t pixel_count() const { return (size_t)width_ * height_; }

    const std::vector<BackendInfo> &backends() const { return backends_; }
    int active_backend() const { return active_backend_; }
    bool is_software() const { return is_software_; }
    std::string device_name();

    HostScene &host_scene() { return host_scene_; }

    KernelLibrary &lib() { return *lib_; }
    KernelBuildSystem &builder() { return *builder_; }
    SourceWatcher &watcher() { return *watcher_; }

private:
    // ── Ordered reload (internal) ────────────────────────────────
    /// Core ordered sequence shared by switch_scene, backend switch,
    /// and hot-reload.  Assumes the kernel binary already exists.
    /// Uses internal width_/height_ for init_kernel dimensions.
    bool ordered_reload(const SceneDef &scene);

    // ── Backend helpers ──────────────────────────────────────────
    void free_device_resources();
    void alloc_device_resources();
    sycl::queue make_queue(int backend_idx);
    void probe_and_select_backend(const std::string &preferred_backend);

    // ── Subsystems (owned) ───────────────────────────────────────
    std::unique_ptr<KernelLibrary> lib_;
    std::unique_ptr<KernelBuildSystem> builder_;
    std::unique_ptr<SourceWatcher> watcher_;

    // ── Backend state ────────────────────────────────────────────
    std::vector<BackendInfo> backends_;
    int active_backend_ = 0;
    bool is_software_ = false;
    sycl::queue q_storage_;
    sycl::queue *q_ = nullptr;
    rt::Runtime krt_;

    // ── Active scene / kernel ────────────────────────────────────
    const SceneDef *active_scene_ = nullptr;
    KernelHandle *active_kernel_ = nullptr;
    std::unique_ptr<scene_loader::SceneDescriptor> active_scene_desc_;
    float *d_params_ = nullptr;
    get_scene_debug_info_fn scene_debug_fn_ = nullptr;
    HostScene host_scene_;

    // ── Per-frame stats (render-thread private, seqlock-published) ─
    std::vector<float> stat_buffer_;
    rt::StatWriter stat_writer_;

    // ── Accumulation buffer (single, persistent) ─────────────────
    float *d_accum_ = nullptr;
    size_t accum_bytes_ = 0;
    int width_ = 0;
    int height_ = 0;
};
