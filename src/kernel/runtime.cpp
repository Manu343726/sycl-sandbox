#include "runtime.h"

#include "kernel/dispatch.h"
#include "scene/host_scene.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstdio>
#include <fstream>

// ── Construction ──────────────────────────────────────────────────────

KernelRuntime::KernelRuntime(const std::string &build_dir,
                             const std::string &project_root,
                             const std::string &preferred_backend)
    : lib_(std::make_unique<KernelLibrary>(build_dir, "")) // suffix set below
    , builder_(std::make_unique<KernelBuildSystem>(build_dir, *lib_))
    , watcher_(std::make_unique<SourceWatcher>(build_dir, project_root))
    , build_dir_(build_dir)
{
    // ── Probe + select backend ───────────────────────────────────
    probe_and_select_backend(preferred_backend);

    // ── Update native suffix after backend selection ──────────────
    lib_->set_native_mode(is_software_);

    // ── Wire watcher into build system ────────────────────────────
    builder_->set_watcher(*watcher_);

    // ── Build notification callback ───────────────────────────────
    builder_->set_notification_callback([](const BuildNotification &n) {
        switch (n.type) {
        case BuildNotification::BuildStarted:
            spdlog::info("[build] {}: {}", n.kernel_name, n.text);
            break;
        case BuildNotification::BuildProgress:
            spdlog::debug("[build] {}: {:.0f}%", n.kernel_name, n.progress * 100);
            break;
        case BuildNotification::BuildCompleted:
            spdlog::info("[build] {}: {}", n.kernel_name, n.text);
            break;
        case BuildNotification::BuildFailed:
            spdlog::error("[build] {}: {}", n.kernel_name, n.text);
            break;
        case BuildNotification::BuildLogLine:
            spdlog::trace("[build] {}: {}", n.kernel_name, n.text);
            break;
        }
    });

    // ── Create queue ─────────────────────────────────────────────
    if (!is_software_) {
        q_storage_ = make_queue(active_backend_);
        q_ = &q_storage_;
    }
    krt_.queue = q_;
    pool_.bind(q_);
    krt_.pool = &pool_;

    // ── Cancellation flag (USM shared when queue exists; plain heap
    //    in software mode — device/host visible in either case).
    //    Plain int, not std::atomic — see execution_context.h rationale.
    cancel_flag_ = pool_.alloc_shared<int>(1, rt::MemScope::Kernel);
    krt_.cancel_flag = cancel_flag_.data;

    // Per-frame trace counters (num_hits / num_bvh_hits)
    d_trace_counters_ = pool_.alloc_device<rt::TraceCounters>(1);
    krt_.fill(d_trace_counters_.data, 0, sizeof(rt::TraceCounters));

    spdlog::info("[startup] {}", device_name());
}

KernelRuntime::~KernelRuntime() {
    // Free everything while the queue is still alive.  release_all()
    // drains first, then frees accum/params/trace counters/cancel flag
    // and the scene arrays (all pool-registered).  The rt::Buffer
    // members' destructors then see an empty registry and no-op.
    pool_.release_all();
    host_scene_ = HostScene{};

    krt_.cancel_flag = nullptr;
    krt_.pool = nullptr;
}

// ── Setup ─────────────────────────────────────────────────────────────

void KernelRuntime::setup_all_kernels(SceneRegistry &scenes) {
    for (auto &s : scenes.all()) {
        builder_->setup_kernel(s.kernel);
    }
}

// ── Scene switch (render thread, background build) ────────────────────

bool KernelRuntime::apply_scene_switch(const SceneDef &scene, int w, int h) {
    spdlog::info("[scene] switching to '{}' (kernel '{}')",
                 scene.name, scene.kernel);
    width_ = w > 0 ? w : width_;
    height_ = h > 0 ? h : height_;

    // Size the accumulation buffer for the target resolution (no-op on
    // runtime switches that keep the current size).  This is the ONLY
    // allocation path for d_accum_: the startup load reaches it through
    // (w,h) = the framebuffer size, and a plain resize goes through
    // apply_resize(); both call resize(), which reallocates + zeroes.
    size_t needed = (size_t)width_ * height_ * 4 * sizeof(float);
    if (d_accum_.data == nullptr || accum_bytes_ != needed)
        resize(width_, height_);

    if (!ensure_built_async(scene.kernel, /*force=*/false)) return false;
    return ordered_reload(scene);
}

// ── Backend switch (render thread) ────────────────────────────────────

int KernelRuntime::apply_backend_switch(int new_backend, int w, int h) {
    // ── Phase 1: Ordered shutdown ─────────────────────────────────
    spdlog::info("[backend] switching to {}", backends_[new_backend].label);

    // Cancel in-progress builds
    builder_->cancel_all();

    // Save scene info before freeing
    const SceneDef *prev_scene = active_scene_.load();
    std::string prev_kernel_name;
    if (prev_scene) prev_kernel_name = prev_scene->kernel;

    // Free all device resources tied to the OLD backend
    free_device_resources();
    active_kernel_.store(nullptr);

    // ── Phase 2: Create new queue ─────────────────────────────────
    is_software_ = (backends_[new_backend].id == "cpu_software");
    lib_->set_native_mode(is_software_);

    if (is_software_) {
        q_ = nullptr;
    } else {
        q_storage_ = make_queue(new_backend);
        q_ = &q_storage_;
    }
    krt_.queue = q_;
    pool_.bind(q_);
    active_backend_ = new_backend;
    spdlog::info("[backend] now using device: {}", device_name());

    // Update stored dimensions
    width_ = w > 0 ? w : width_;
    height_ = h > 0 ? h : height_;

    // ── Re-allocate accumulation buffers ──────────────────────────
    alloc_device_resources();

    // ── Phase 3: Rebuild + reload ─────────────────────────────────
    if (prev_scene && !prev_kernel_name.empty()) {
        // Background build: missing binary, stale sources, or (critical)
        // an existing .so compiled against an older ABI (RenderContext /
        // ParamLookup layout) would make the device kernel read garbage
        // pointers → sticky CUDA error 700.
        if (!ensure_built_async(prev_kernel_name, /*force=*/false)) {
            spdlog::error("[backend] kernel '{}' build failed for new backend",
                          prev_kernel_name);
            return new_backend;
        }
        ordered_reload(*prev_scene);
    }

    return new_backend;
}

// ── Resolution change (render thread) ─────────────────────────────────

void KernelRuntime::apply_resize(int w, int h) {
    // Pure resolution change: only the accumulation buffer depends on the
    // pixel count (params/scene/stat-writer are resolution-independent),
    // so a plain resize() is sufficient — reinit would redundantly rebuild
    // the YAML scene on every window drag.
    resize(w, h);
}

// ── Build helper (render thread) ──────────────────────────────────────

bool KernelRuntime::ensure_built_async(const std::string &kernel, bool force) {
    if (!force && lib_->so_exists(kernel) && lib_->is_up_to_date(kernel))
        return true;

    // Cancel any in-flight build for this kernel and purge cancelled /
    // finished entries so build_async() can start a fresh one (a
    // cancelled build never pushes a result, so we must not wait on it).
    builder_->cancel(kernel);
    builder_->poll_results();

    spdlog::info("[build] kernel '{}' {}...", kernel,
                 force ? "rebuilding" : "missing or stale, building");
    builder_->build_async(kernel);

    bool success = false;
    bool got = false;
    while (!got) {
        for (auto &r : builder_->poll_results()) {
            if (r.kernel_name == kernel) {
                success = r.success;
                got = true;
            }
        }
        if (!got)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!success)
        spdlog::error("[build] kernel '{}' build failed", kernel);
    return success;
}

// ── Re-init (param / camera change) ───────────────────────────────────

void KernelRuntime::reinit_kernel() {
    auto desc = scene_desc();
    if (!desc) return;
    reinit_kernel(desc->current_buffer);
}

void KernelRuntime::reinit_kernel(const std::vector<float> &params) {
    // Works on every backend — q_ is legitimately null in software mode.
    if (!active_kernel_.load()) return;

    auto desc = scene_desc();
    if (!desc) return;

    auto sz = desc->buffer_size();
    // The snapshot was taken from a previous scene's descriptor (the UI
    // queued a re-init and a scene switch happened in between) — drop it
    // rather than copy a mismatched layout into d_params.
    if (params.size() * sizeof(float) < sz) {
        spdlog::warn("[param] re-init dropped: param snapshot layout "
                     "({} floats) doesn't match active scene ({})",
                     params.size(), desc->buffer_size() / sizeof(float));
        return;
    }
    krt_.copy_to_device(d_params_.data, params.data(), sz);

    // Rebuild the YAML scene from the current live params (kernels that
    // accept a host-built scene; others skip it entirely).
    const SceneDef *scene = active_scene_.load();
    if (scene && scene->uses_scene) {
        rebuild_yaml_scene(&krt_,
                           scene->yaml_path,
                           *desc,
                           host_scene_);
    }

    // Refresh the host-side stat writer (the kernel writes into it via
    // ctx->stats during the render call — no device round-trip).
    {
        stat_buffer_.assign(desc->current_stat_buffer.size(), 0.f);
        stat_writer_ = rt::StatWriter{};
        if (!stat_buffer_.empty()) {
            stat_writer_.set_buffer(stat_buffer_.data());
            stat_writer_.set_entries(
                desc->stat_lookup_entries.data(),
                (int)desc->stat_lookup_entries.size());
        }
    }

    clear_accum();
}

// ── Hot reload ────────────────────────────────────────────────────────

std::string KernelRuntime::poll_hot_reload(
    const std::function<void()> &pre_reload) {
    bool triggered = false;
    std::string reloaded_kernel;

    for (auto &dirty_name : watcher_->poll()) {
        spdlog::debug("[watcher] {} changed, starting background build...",
                      dirty_name);
        builder_->build_async(dirty_name);
    }

    for (auto &res : builder_->poll_results()) {
        if (res.success) {
            auto *new_kh = lib_->load(res.kernel_name);
            if (new_kh && active_scene_.load() && active_kernel_.load() &&
                active_kernel_.load()->name == res.kernel_name) {
                spdlog::info("[hotreload] rebuilt and reloading '{}' (gen {})",
                             res.kernel_name, new_kh->generation);
                triggered = true;
                reloaded_kernel = res.kernel_name;
            }
        } else {
            spdlog::error("[hotreload] build failed for '{}'", res.kernel_name);
        }
    }

    // If a kernel was reloaded, run the full ordered reload
    if (triggered && active_scene_) {
        if (pre_reload) pre_reload();
        ordered_reload(*active_scene_);
    }

    // Ingest build monitor notifications
    // (build_monitor is owned by AppState, notifications ingested by caller)

    return reloaded_kernel;
}

// ── Accumulation ops ──────────────────────────────────────────────────

void KernelRuntime::clear_accum() {
    if (d_accum_.data) krt_.fill(d_accum_.data, 0, accum_bytes_);
}

void KernelRuntime::clear_accum_async() {
    if (!d_accum_.data) return;
    if (q_) q_->memset(d_accum_.data, 0, accum_bytes_);   // in-order: sequenced
    else std::memset(d_accum_.data, 0, accum_bytes_);     // native: synchronous
}

void KernelRuntime::fill_zero(float *buf) {
    krt_.fill(buf, 0, accum_bytes_);
}

void KernelRuntime::zero_trace_counters_async() {
    if (!d_trace_counters_.data) return;
    if (q_) q_->memset(d_trace_counters_.data, 0, sizeof(rt::TraceCounters));
    else std::memset(d_trace_counters_.data, 0, sizeof(rt::TraceCounters));
}

void KernelRuntime::drain() {
    if (!q_) return;
    try {
        q_->wait();
    } catch (const std::exception &e) {
        spdlog::error("[sycl] drain: {}", e.what());
    }
}

void KernelRuntime::resize(int w, int h) {
    width_ = w;
    height_ = h;
    size_t pixel_count = (size_t)w * h;
    size_t bytes = pixel_count * 4 * sizeof(float);

    // Free old (the move-assign drains via the pool before freeing) and
    // allocate new.
    d_accum_ = {};
    accum_bytes_ = bytes;
    d_accum_ = pool_.alloc_device<float>(pixel_count * 4);
    krt_.fill(d_accum_.data, 0, accum_bytes_);
}

// ── Device name ───────────────────────────────────────────────────────

std::string KernelRuntime::device_name() {
    return krt_.device_name();
}

// ── Internal: ordered reload ──────────────────────────────────────────

bool KernelRuntime::ordered_reload(const SceneDef &scene) {
    // Free old params — after draining: enqueued frames read them at
    // dispatch time, but a memcpy targeting them may still be queued.
    drain();
    d_params_ = {};

    // Load kernel
    active_kernel_.store(lib_->load(scene.kernel));
    if (!active_kernel_.load()) {
        spdlog::error("[scene] failed to load kernel '{}'", scene.kernel);
        return false;
    }

    spdlog::info("[scene] switch complete: kernel loaded (gen {})",
                 active_kernel_.load()->generation);

    // Load + build scene descriptor
    active_scene_.store(&scene);
    set_scene_desc(std::make_shared<scene_loader::SceneDescriptor>(
        scene_loader::load_scene_descriptor(scene.yaml_path)));
    scene_desc()->build_layout();

    // Allocate + upload params.  Host-visible (alloc_host): the kernel
    // reads its param snapshot through a ParamLookup over this buffer.
    auto desc = scene_desc();
    auto sz = desc->buffer_size();
    d_params_ = pool_.alloc_host<float>(sz / sizeof(float));
    krt_.copy_to_device(d_params_.data,
                        desc->current_buffer.data(),
                        sz);

    // Rebuild the YAML scene first — the per-frame ctx carries the scene
    // view, so the geometry must exist before the first render call.
    if (scene.uses_scene) {
        rebuild_yaml_scene(&krt_,
                           scene.yaml_path,
                           *desc,
                           host_scene_);
    }

    // Host-side stat writer for this scene (handed to the kernel via
    // ctx->stats each frame — no Setup op, no kernel call).
    {
        stat_buffer_.assign(desc->current_stat_buffer.size(), 0.f);
        stat_writer_ = rt::StatWriter{};
        if (!stat_buffer_.empty()) {
            stat_writer_.set_buffer(stat_buffer_.data());
            stat_writer_.set_entries(
                desc->stat_lookup_entries.data(),
                (int)desc->stat_lookup_entries.size());
        }
    }

    // Clear accum
    clear_accum();

    return true;
}

// ── Backend helpers ───────────────────────────────────────────────────

void KernelRuntime::free_device_resources() {
    // release_all drains first, then frees every pool-registered buffer
    // (accum/params/trace counters/cancel flag/scene arrays) with the
    // OLD queue — which is still bound here, before the backend switch
    // rebinds the pool.
    pool_.release_all();
    d_accum_ = {};
    d_params_ = {};
    d_trace_counters_ = {};
    cancel_flag_ = {};
    krt_.cancel_flag = nullptr;

    // Free host scene (its arrays were pool-registered and freed above;
    // this resets the SceneData view's counts/pointers).
    if (!host_scene_.data.empty()) {
        host_scene_.data.free(&krt_);
        host_scene_ = HostScene{};
    }
}

void KernelRuntime::alloc_device_resources() {
    size_t pixel_count = (size_t)width_ * height_;
    accum_bytes_ = pixel_count * 4 * sizeof(float);

    d_accum_ = pool_.alloc_device<float>(pixel_count * 4);
    krt_.fill(d_accum_.data, 0, accum_bytes_);

    // Trace counters — persistent across scenes, re-created per backend
    d_trace_counters_ = pool_.alloc_device<rt::TraceCounters>(1);
    krt_.fill(d_trace_counters_.data, 0, sizeof(rt::TraceCounters));

    // Re-allocate cancellation flag after a backend switch (freed by
    // free_device_resources() above).  Guard against double-allocation.
    if (!cancel_flag_.data) {
        cancel_flag_ = pool_.alloc_shared<int>(1, rt::MemScope::Kernel);
        krt_.cancel_flag = cancel_flag_.data;
    }
}

sycl::queue KernelRuntime::make_queue(int backend_idx) {
    // In-order queue: behaves like a single stream — every submitted
    // operation implicitly depends on the previous one, so the render →
    // tone-map → display-copy chain needs no manual event wiring and can
    // never race device-vs-device.  An async_handler surfaces async SYCL
    // exceptions through spdlog instead of std::terminate.
    static auto async_handler = [](sycl::exception_list el) {
        for (auto &e : el) {
            try { std::rethrow_exception(e); }
            catch (const sycl::exception &ex) {
                spdlog::error("[sycl] async exception: {}", ex.what());
            }
        }
    };
    const sycl::property_list props{sycl::property::queue::in_order{}};

    const auto &b = backends_[backend_idx];
    if (b.id == "cpu_software") {
        return sycl::queue(sycl::cpu_selector_v, async_handler, props);
    }
    if (b.id == "gpu_sycl") {
        try {
            return sycl::queue(sycl::gpu_selector_v, async_handler, props);
        } catch (const std::exception &e) {
            spdlog::warn("[backend] gpu_selector_v failed ({}), falling back to cpu",
                         e.what());
        }
    }
    return sycl::queue(sycl::cpu_selector_v, async_handler, props);
}

void KernelRuntime::probe_and_select_backend(const std::string &preferred_backend) {
    // CPU (Software): always available with AdaptiveCpp (OpenMP CPU backend)
    backends_.push_back({"CPU (Software)", "cpu_software", true});

    // ── Force software backend before probing SYCL queues ────────
    // Select cpu_software immediately so the queue is created with
    // cpu_selector_v.  We still probe SYCL backends afterward so the
    // UI backend-switcher can see them.
    bool force_software = (preferred_backend == "software");
    if (force_software) {
        active_backend_ = 0;
        is_software_ = true;
        spdlog::info("[startup] using CPU Software backend (--backend software)");
    }

    try {
        sycl::queue probe_cpu(sycl::cpu_selector_v);
        (void)probe_cpu;
        backends_.push_back({"CPU (SYCL)", "cpu_sycl", true});
    } catch (...) {
        backends_.push_back({"CPU (SYCL)", "cpu_sycl", false});
    }

    try {
        sycl::queue probe_gpu(sycl::gpu_selector_v);
        (void)probe_gpu;
        backends_.push_back({"GPU (SYCL)", "gpu_sycl", true});
    } catch (...) {
        backends_.push_back({"GPU (SYCL)", "gpu_sycl", false});
    }

    // ── Select initial backend ───────────────────────────────────
    // "auto": always use cascading best-available (GPU SYCL > CPU SYCL > CPU Software).
    // "gpu" / "cpu" / "software": explicit preferences with fallback.
    bool try_auto = (preferred_backend == "auto");

    if (preferred_backend == "gpu") {
        for (int i = 0; i < (int)backends_.size(); i++) {
            if (backends_[i].id == "gpu_sycl") {
                if (backends_[i].available) {
                    active_backend_ = i;
                    is_software_ = false;
                    spdlog::info("[startup] using GPU SYCL backend (--backend gpu)");
                } else {
                    spdlog::warn("[startup] GPU SYCL not available (--backend gpu), "
                                 "falling back to best available");
                    try_auto = true;
                }
                break;
            }
        }
    } else if (preferred_backend == "cpu") {
        for (int i = 0; i < (int)backends_.size(); i++) {
            if (backends_[i].id == "cpu_sycl" && backends_[i].available) {
                active_backend_ = i;
                is_software_ = false;
                break;
            }
        }
        if (backends_[active_backend_].id != "cpu_sycl") {
            active_backend_ = 0;
            is_software_ = true;
        }
        spdlog::info("[startup] using {} backend (--backend cpu)",
                     backends_[active_backend_].label);
    }
    // preferred_backend == "software" was already handled above —
    // active_backend_ and is_software_ are already set.

    // Auto-select best available backend.
    // Priority: GPU SYCL > CPU SYCL > CPU Software
    if (try_auto || !backends_[active_backend_].available) {
        for (int i = (int)backends_.size() - 1; i >= 0; i--) {
            if (backends_[i].available) {
                active_backend_ = i;
                break;
            }
        }
        spdlog::info("[startup] {} {} backend{}",
                     (preferred_backend == "auto") ? "auto-selected" : "fell back to",
                     backends_[active_backend_].label,
                     (preferred_backend == "auto") ? "" : " (best available)");
    }

    // Ultimate fallback
    if (!backends_[active_backend_].available) {
        for (int i = (int)backends_.size() - 1; i >= 0; i--) {
            if (backends_[i].available) {
                active_backend_ = i;
                spdlog::info("[startup] fell back to {} backend",
                             backends_[active_backend_].label);
                break;
            }
        }
    }

    is_software_ = (backends_[active_backend_].id == "cpu_software");
}

// ── Cancellation ──────────────────────────────────────────────────────

void KernelRuntime::cancel() {
    // Host-side only — never touches USM.  Safe to call from any thread
    // regardless of queue lifecycle (backend switch, shutdown, etc.).
    cancel_requested_.store(1, std::memory_order_relaxed);
}

void KernelRuntime::begin_frame() {
    // Mirror the host cancel state to device-visible USM before the
    // kernel dispatch, then clear the host flag.  Only the render
    // thread calls this — it owns the queue, so USM is always valid.
    int *cf = cancel_flag_.data;
    if (cf) {
        int v = cancel_requested_.load(std::memory_order_relaxed);
        __atomic_store_n(cf, v, __ATOMIC_RELAXED);
    }
    cancel_requested_.store(0, std::memory_order_relaxed);
}

// ── Profiler flag (on-the-fly kernel builds) ──────────────────────────

bool KernelRuntime::apply_profiler_toggle(bool enabled) {
    std::string value = enabled ? "ON" : "OFF";
    spdlog::info("[profiler] reconfiguring kernel builds with "
                 "SANDBOX_ENABLE_PROFILER={}", value);

    // Re-run CMake configure so the cache variable changes.  Passing the
    // existing build dir as the argument re-uses the cached source dir;
    // the rebuilt targets pick up the new compile definition.
    std::string cmd = "cmake -DSANDBOX_ENABLE_PROFILER=" + value +
                      " " + build_dir_ + " 2>&1";
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        spdlog::error("[profiler] popen failed for cmake reconfigure");
        return false;
    }
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) {
        // Reconfigure output is short — route through spdlog at debug.
        std::string line(buf);
        if (!line.empty() && line.back() == '\n') line.pop_back();
        spdlog::debug("[profiler] cmake: {}", line);
    }
    int rc = pclose(pipe);
    if (rc != 0) {
        spdlog::error("[profiler] cmake reconfigure failed (exit {}), "
                      "kernel builds keep the previous flag", rc);
        return false;
    }

    // Rebuild + reload the active kernel with the new flag.  The compile
    // definition changed, so the .so is stale even though its timestamps
    // look current — force the rebuild.
    const SceneDef *scene = active_scene_.load();
    KernelHandle *kh = active_kernel_.load();
    if (scene && kh) {
        std::string name = kh->name;
        if (!ensure_built_async(name, /*force=*/true)) {
            spdlog::error("[profiler] kernel '{}' rebuild failed with "
                          "profiler={} — falling back to old binary", name, value);
            return false;
        }
        ordered_reload(*scene);
        spdlog::info("[profiler] kernel '{}' rebuilt (SANDBOX_ENABLE_PROFILER={})",
                     name, value);
        return true;
    }
    return true;
}

bool KernelRuntime::profiler_enabled() const {
    std::ifstream f(build_dir_ + "/CMakeCache.txt");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("SANDBOX_ENABLE_PROFILER:", 0) == 0) {
            auto pos = line.find('=');
            if (pos != std::string::npos)
                return line.compare(pos + 1, 2, "ON") == 0;
        }
    }
    return true;
}
