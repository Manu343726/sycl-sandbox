#pragma once

#include "library.h"
#include "build_system.h"
#include "kernel/dispatch.h"
#include "scene/registry.h"
#include "scene/host_scene.h"
#include "io/watcher.h"
#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/kernel/memory.h>

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

    // ── Re-init (param / camera change) ───────────────────────────
    /// Re-init the kernel after a param or camera change (no rebuild).
    /// Copies the latest param buffer to the device and calls init_kernel
    /// with the last-known width/height.  Startup/synchronous path.
    void reinit_kernel();
    /// Render-thread variant: re-init using an explicit parameter snapshot
    /// (the UI may be editing current_buffer concurrently).  Silently
    /// dropped when the snapshot's layout doesn't match the active scene
    /// (scene switched while the re-init was queued).
    void reinit_kernel(const std::vector<float> &params);

    /// Poll the source watcher for dirty files and check build results.
    /// If a kernel was hot-reloaded, triggers the full ordered_reload
    /// and returns the kernel name (empty string otherwise).
    /// `pre_reload` (optional) runs right before the reload swaps state —
    /// the app uses it to stop the render thread and drain the queue.
    std::string poll_hot_reload(
        const std::function<void()> &pre_reload = {});

    // ── Render-thread apply variants ─────────────────────────────
    /// All of these run ON the render thread (posted via the UI command
    /// mailbox).  They may block on a background build (the render thread
    /// is allowed to block — the UI never is).  Callers are responsible
    /// for the surrounding handshake (kernel_ready/apply_epoch).

    /// Scene switch with background build + wait.  Returns false when the
    /// build failed (no reload performed, old scene stays active).
    bool apply_scene_switch(const SceneDef &scene, int w, int h);

    /// Backend switch with background build + wait.  Frees resources on
    /// the old queue, creates the new queue, rebuilds + reloads the
    /// active scene.  Returns the new backend index.
    int apply_backend_switch(int new_backend, int w, int h);

    /// Resolution change: re-allocate the accumulation buffer for w×h.
    void apply_resize(int w, int h);

    /// Reconfigure the CMake cache for SANDBOX_ENABLE_PROFILER and force a
    /// rebuild + reload of the active kernel with the new flag.
    /// Returns true when the kernel was successfully rebuilt+reloaded.
    bool apply_profiler_toggle(bool enabled);

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

    /// Per-frame stat writer.  Points at a
    /// runtime-owned buffer (NOT the scene descriptor's — the UI thread
    /// reads that one); the render thread publishes it through the
    /// PublishedStats seqlock after each frame.  Null when the scene has
    /// no stats.  Handed to the kernel in ctx->stats each frame.
    const rt::StatWriter *stat_writer() const {
        return stat_buffer_.empty() ? nullptr : &stat_writer_;
    }
    const std::vector<float> &stat_buffer() const { return stat_buffer_; }

    // ── Per-frame trace counters (device side) ─────────────────────
    /// Device buffer holding num_hits / num_bvh_hits for the current
    /// frame.  Zeroed before each render enqueue (in-order: sequenced
    /// before the render kernel), read back after frame completion and
    /// folded into the per-frame stat block.  May be null in degenerate
    /// states.
    rt::TraceCounters *trace_counters() { return d_trace_counters_.data; }

    /// Zero the trace counters before enqueuing the render kernel
    /// (in-order queue sequences the memset ahead of the kernel).
    void zero_trace_counters_async();

    KernelHandle *kernel() const { return active_kernel_.load(); }
    const SceneDef *scene() const { return active_scene_.load(); }
    /// Current scene descriptor (parsed + laid out).  Returns a shared_ptr
    /// so UI-side readers can hold the descriptor across a reload that
    /// swaps in a new one.  Thread-safe.
    std::shared_ptr<scene_loader::SceneDescriptor> scene_desc() const {
        std::lock_guard<std::mutex> lk(desc_mtx_);
        return active_scene_desc_;
    }

    float *d_params() const { return d_params_.data; }

    float *d_accum() const { return d_accum_.data; }
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

    // ── Cancellation ────────────────────────────────────────────
    /// Signal in-flight kernel work to abort.  Safe to call from any
    /// thread — operates on a host-side atomic, never touches USM.
    void cancel();
    /// Called by the render thread at the start of each frame.
    /// Syncs the host cancel state to device-visible USM, then resets
    /// the host flag.  Only the render thread touches USM.
    void begin_frame();

    // ── Profiler flag (on-the-fly kernel builds) ─────────────────
    /// Whether the CMake cache currently has SANDBOX_ENABLE_PROFILER=ON
    /// (the value the UI checkbox starts from).  Defaults to true when
    /// the cache file cannot be read.
    bool profiler_enabled() const;

private:
    // ── Ordered reload (internal) ────────────────────────────────
    /// Core ordered sequence shared by backend switch, scene switch,
    /// and hot-reload.  Assumes the kernel binary already exists.
    /// Uses internal width_/height_ for init_kernel dimensions.
    bool ordered_reload(const SceneDef &scene);

    /// Swap in a new scene descriptor (render thread; desc_mtx_ guarded).
    void set_scene_desc(std::shared_ptr<scene_loader::SceneDescriptor> d) {
        std::lock_guard<std::mutex> lk(desc_mtx_);
        active_scene_desc_ = std::move(d);
    }

    /// Render-thread helper: make sure `kernel`'s binary is built, kicking
    /// a background build and waiting for it when missing/stale (or always
    /// when `force` — used when a compile-time define changed).  Returns
    /// false on build failure.
    bool ensure_built_async(const std::string &kernel, bool force);

    // ── Backend helpers ──────────────────────────────────────────
    void free_device_resources();
    void alloc_device_resources();
    sycl::queue make_queue(int backend_idx);
    void probe_and_select_backend(const std::string &preferred_backend);

    // ── Subsystems (owned) ───────────────────────────────────────
    std::unique_ptr<KernelLibrary> lib_;
    std::unique_ptr<KernelBuildSystem> builder_;
    std::unique_ptr<SourceWatcher> watcher_;
    /// CMake build directory (absolute) — used to reconfigure the cache
    /// when the on-the-fly kernel build flags change.
    std::string build_dir_;

    // ── Backend state ────────────────────────────────────────────
    std::vector<BackendInfo> backends_;
    int active_backend_ = 0;
    bool is_software_ = false;
    sycl::queue q_storage_;
    sycl::queue *q_ = nullptr;
    /// Single allocation registry for all kernel-usable memory.  Bound to
    /// `q_` (null in software mode).  Declared before `krt_` so the pool
    /// outlives the Runtime that references it; released explicitly in the
    /// destructor while the queue is still alive.
    rt::MemoryPool pool_;
    rt::Runtime krt_;

    // ── Active scene / kernel ────────────────────────────────────
    /// Pointers into stable, owned storage (SceneRegistry / KernelLibrary),
    /// so concurrent UI reads see either the old or the new entry — never
    /// a dangling one.  Atomic because the render thread swaps them on
    /// reload while the UI reads them every frame.
    std::atomic<const SceneDef *> active_scene_{nullptr};
    std::atomic<KernelHandle *> active_kernel_{nullptr};
    /// Scene descriptor.  Swap-protected by desc_mtx_; readers get a
    /// shared_ptr copy that keeps the (possibly replaced) descriptor alive
    /// for as long as they use it.
    mutable std::mutex desc_mtx_;
    std::shared_ptr<scene_loader::SceneDescriptor> active_scene_desc_;
    rt::Buffer<float> d_params_;
    HostScene host_scene_;

    // ── Per-frame stats (render-thread private, seqlock-published) ─
    std::vector<float> stat_buffer_;
    rt::StatWriter stat_writer_;

    // ── Trace counters (device buffer, zeroed per frame) ─────────
    rt::Buffer<rt::TraceCounters> d_trace_counters_;

    // ── Accumulation buffer (single, persistent) ─────────────────
    rt::Buffer<float> d_accum_;
    size_t accum_bytes_ = 0;
    int width_ = 0;
    int height_ = 0;

    // ── Cancellation ────────────────────────────────────────────
    /// Host-side flag — always valid (plain member, never freed).
    /// Set by cancel(), read by the render thread each frame start.
    /// This is the canonical cancel signal for host code; the USM
    /// pointer below is a mirror for device-code visibility.
    std::atomic<int> cancel_requested_{0};
    /// USM-shared flag mirror for device code.  Allocated through the
    /// pool (Shared kind) so backend-switch teardown is uniform.  May
    /// be empty (software backend) or briefly empty during a backend
    /// switch.
    rt::Buffer<int> cancel_flag_;
};
