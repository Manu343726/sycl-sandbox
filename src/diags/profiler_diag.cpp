// `diag profiler` — full profiler-system diagnostic.
//
// Drives the REAL raytracer kernel .so in-process (like `diag gpu`) but
// with a LIVE device profiler ring + Tracy bridge, while a tracy::Worker
// (the same server-side capture machinery `diag tracy` uses) connects to
// the in-process Tracy client over loopback and captures every emitted
// event.  This end-to-end exercises the whole pipeline:
//
//   kernel PROFILER_ZONE → device ring (atomic records) → bridge drain
//   → srcloc + GPU-zone/calibration serial emits → Tracy client → TCP
//   loopback → tracy::Worker → capture analysis.
//
// For each of the three backends (software, cpu, gpu) it renders the
// mesh_demo scene at 1080p for 10 frames and verifies that every
// category of profile data arrives with correct content:
//   * GPU context + per-frame calibration (calibrationMod > 0)
//   * frame marks (≥ frames × backends)
//   * GPU zones (> 0 per backend, names resolved to the real profiler
//     names — aabb_slab, render_sample, … — not opaque hashes)
//   * plots (SPP) and messages (backend boundaries)
//   * pipeline integrity (bridge-emitted zone count ≈ worker-captured)
//
// It also measures the per-frame record count and extrapolates to 100%
// interest, reporting the ring size and sampling ratio needed to capture
// a full frame without overflow for each backend.
//
// The diag is self-contained: the in-process Tracy client binds to the
// process's TRACY_PORT (or 8086 — read at static-init, see below) and the
// loopback Worker connects to that same port.

#include "diags.h"
#include "profiler_diag.h"
#include "kernel/zone_names.h"
#include "kernel/dispatch.h"
#include "tracy/tracy_bridge.h"

#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/scene_loader.h>
#include <sycl-sandbox/scene/data.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include <sycl-sandbox/kernel/memory.h>
#include <sycl-sandbox/kernel/stats.h>
#include <sycl-sandbox/profiler.h>

#include <sycl/sycl.hpp>
#include <tracy/TracyC.h>
#include <TracyFileWrite.hpp>
#include <TracyPrint.hpp>
#include <TracyWorker.hpp>

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

// ── Expected device zone names (PROFILER_ZONE / PROFILER_ZONE_IN
// literals — the generated name table resolves these).  __PRETTY_FUNCTION__
// zones (PROFILER_FUNCTION) are not in the table and show as
// gpu:zone_<hash>; they are expected but unnamed.
const std::vector<std::string> kExpectedZoneNames = {
    "aabb_slab", "accumulate_pixel", "bvh_traverse", "pixel_path",
    "render_pixel", "render_sample", "trace_bounce"
};

uint64_t next_pow2(uint64_t x) {
    if (x <= 1) return 1;
    --x;
    x |= x >> 1;  x |= x >> 2;  x |= x >> 4;
    x |= x >> 8;  x |= x >> 16; x |= x >> 32;
    return x + 1;
}

// ── tracy::Worker capture (background thread) ─────────────────────────
// The Worker is created, polled, and destroyed inside the capture thread
// (ObtainLockForMainThread is taken there).  The main thread reads the
// accumulators only after join — no concurrent access.
struct Capture {
    std::atomic<bool> stop{false};
    std::atomic<bool> connected{false};
    std::atomic<bool> has_ctx{false};
    std::atomic<bool> has_calibration{false};
    std::atomic<double> calibration_mod{0};
    std::atomic<size_t> gpu_zones{0};
    std::atomic<size_t> frame_marks{0};
    std::atomic<size_t> messages{0};
    std::string handshake_error;     // set once before connected
    // Written only by the capture thread (under the worker lock).
    std::unordered_set<const tracy::GpuEvent *> seen_zones;  // completed only
    std::unordered_set<std::string> zone_names;
    std::vector<std::string> plot_names;
    std::vector<std::string> msg_texts;
};

void collect_gpu_zones(const tracy::Worker &worker,
                       const tracy::Vector<tracy::short_ptr<tracy::GpuEvent>> &vec,
                       Capture &c) {
    for (const auto &sp : vec) {
        const tracy::GpuEvent *ev = sp;
        if (!ev) continue;
        // Count each completed zone exactly once.  In-flight zones are not
        // marked seen, so a later poll logs them when the tree settles.
        if (ev->GpuStart() >= 0 && ev->GpuEnd() >= 0 &&
            c.seen_zones.insert(ev).second) {
            c.gpu_zones.fetch_add(1, std::memory_order_relaxed);
            c.zone_names.insert(worker.GetZoneName(*ev));
        }
        if (ev->Child() >= 0)
            collect_gpu_zones(worker, worker.GetGpuChildren(ev->Child()), c);
    }
}

void poll_worker(tracy::Worker &worker, Capture &c) {
    auto lock = worker.ObtainLockForMainThread();

    const auto &gpu = worker.GetGpuData();
    for (auto *ctx : gpu) {
        c.has_ctx.store(true, std::memory_order_relaxed);
        if (ctx->hasCalibration) {
            c.has_calibration.store(true, std::memory_order_relaxed);
            c.calibration_mod.store(ctx->calibrationMod, std::memory_order_relaxed);
        }
        for (auto &td : ctx->threadData)
            collect_gpu_zones(worker, td.second.timeline, c);
    }
    if (getenv("PROFILER_DIAG_DEBUG")) {
        std::fprintf(stderr, "[diag] poll: connected=%d gpu_ctxs=%zu gpu_zone_count=%llu threadData=%zu\n",
                     (int)worker.IsConnected(), gpu.size(),
                     (unsigned long long)worker.GetGpuZoneCount(),
                     gpu.empty() ? 0 : gpu[0]->threadData.size());
    }

    size_t frames = 0;
    for (const auto *fd : worker.GetFrames())
        frames += worker.GetFrameCount(*fd);
    c.frame_marks.store(frames, std::memory_order_relaxed);

    const auto &plots = worker.GetPlots();
    for (const auto *p : plots) {
        const char *name = worker.GetString(p->name);
        if (std::find(c.plot_names.begin(), c.plot_names.end(), name)
            == c.plot_names.end())
            c.plot_names.push_back(name);
    }

    const auto &messages = worker.GetMessages();
    if (messages.size() > c.msg_texts.size()) {
        for (size_t i = c.msg_texts.size(); i < messages.size(); i++) {
            const auto *m = messages[i].get();
            if (m) c.msg_texts.emplace_back(worker.GetString(m->ref));
        }
    }
    c.messages.store(messages.size(), std::memory_order_relaxed);
}

void capture_thread(Capture *c, const std::string &address, int port) {
    tracy::Worker worker(address.c_str(), uint16_t(port), -1);
    // Wait for the handshake (the main thread inits the bridge / client
    // before starting this thread, so the listener is already up).
    while (!worker.HasData()) {
        const auto hs = worker.GetHandshakeStatus();
        if (hs == tracy::HandshakeProtocolMismatch) { c->handshake_error = "protocol mismatch"; return; }
        if (hs == tracy::HandshakeNotAvailable)  { c->handshake_error = "client already has a server"; return; }
        if (hs == tracy::HandshakeDropped)       { c->handshake_error = "connection dropped"; return; }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    c->connected.store(true, std::memory_order_relaxed);

    while (!c->stop.load(std::memory_order_relaxed)) {
        if (worker.IsConnected()) poll_worker(worker, *c);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    // Final drain so the last frames' events are captured.
    for (int i = 0; i < 10 && worker.IsConnected(); i++) {
        poll_worker(worker, *c);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    worker.Disconnect();
}

// ── Per-backend run ───────────────────────────────────────────────────
struct BackendResult {
    std::string id, label;
    bool ran = false;
    std::string error;
    int frames_run = 0;
    double avg_frame_ms = 0;
    double avg_records = 0;       // records/frame at the test interest
    uint64_t max_records = 0;
    bool overflow = false;
    uint64_t emitted_zones = 0;   // bridge side
    double full_frame_records = 0;  // extrapolated to 100% interest
    uint64_t required_ring = 0;    // next_pow2(full_frame_records)
};

BackendResult run_backend(const std::string &id, const std::string &label,
                          const std::string &yaml, const std::string &so_path,
                          int W, int H, int frames, uint32_t interest,
                          uint32_t ring_cap, tracy_bridge::Bridge &bridge) {
    BackendResult r;
    r.id = id;
    r.label = label;

    // ── Queue (null for software) ──────────────────────────────────
    sycl::queue q_storage;
    sycl::queue *q = nullptr;
    if (id == "software") {
        q = nullptr;
    } else if (id == "cpu") {
        try {
            q_storage = sycl::queue{sycl::cpu_selector_v,
                                    sycl::property::queue::in_order()};
            q = &q_storage;
        } catch (const std::exception &e) {
            r.error = std::string("CPU queue: ") + e.what();
            return r;
        }
    } else {  // gpu
        try {
            q_storage = sycl::queue{sycl::gpu_selector_v,
                                    sycl::property::queue::in_order()};
            q = &q_storage;
        } catch (const std::exception &e) {
            r.error = std::string("GPU queue: ") + e.what();
            return r;
        }
    }

    rt::MemoryPool pool;
    pool.bind(q);
    rt::Runtime rt;
    rt.queue = q;
    rt.pool = &pool;

    // ── Scene ─────────────────────────────────────────────────────
    auto config = scene_loader::load_and_resolve(yaml);
    rt::SceneBuilder builder;
    scene_loader::build_scene(builder, config);
    builder.build_bvh();
    builder.build_mesh_bvhs();
    rt::SceneData data = builder.build(&rt);
    rt::SceneView v = data.view();

    auto desc = scene_loader::load_scene_descriptor(yaml);
    desc.build_layout();
    size_t pbytes = desc.buffer_size();
    float *d_params = rt.alloc_host<float>(pbytes / sizeof(float));
    rt.copy_to_device(d_params, desc.current_buffer.data(), pbytes);
    rt::ParamLookup lookup = desc.make_lookup();
    lookup.set_buffer(d_params);

    const size_t pixels = size_t(W) * H;
    float *d_accum = rt.alloc_device<float>(pixels * 4);
    rt.fill(d_accum, 0, pixels * 4 * sizeof(float));
    uint8_t *d_output = rt.alloc_device<uint8_t>(pixels * 4);

    // ── Device profiler ring (header + records + per-lane flags) ──
    auto *d_hdr = pool.alloc_shared_raw<profiler::DeviceRingHeader>(
        1, rt::MemScope::App);
    auto *d_rec = pool.alloc_shared_raw<profiler::DeviceRecord>(
        ring_cap, rt::MemScope::App);
    auto *d_flags = pool.alloc_shared_raw<uint8_t>(pixels, rt::MemScope::App);
    if (d_hdr)  pool.fill(d_hdr, 0, sizeof(profiler::DeviceRingHeader));
    if (d_flags) pool.fill(d_flags, 0, pixels);

    // ── Kernel .so ─────────────────────────────────────────────────
    void *so = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!so) {
        r.error = std::string("dlopen(") + so_path + "): " + dlerror();
        data.free(&rt); rt.dealloc(d_params); rt.dealloc(d_accum);
        rt.dealloc(d_output);
        pool.release(d_hdr); pool.release(d_rec); pool.release(d_flags);
        return r;
    }
    auto entry = reinterpret_cast<void (*)(const rt::Context *)>(
        dlsym(so, "kernel_entry"));
    if (!entry) {
        r.error = std::string("dlsym kernel_entry: ") + dlerror();
        dlclose(so);
        data.free(&rt); rt.dealloc(d_params); rt.dealloc(d_accum);
        rt.dealloc(d_output);
        pool.release(d_hdr); pool.release(d_rec); pool.release(d_flags);
        return r;
    }

    std::printf("[%s] device=%s  scene=%s  %dx%d  frames=%d  interest=%u%%  "
                "ring=%u\n", label.c_str(),
                id == "software" ? "native CPU (KERNEL_NATIVE)"
                                 : (q ? q->get_device()
                                         .get_info<sycl::info::device::name>()
                                         .c_str()
                                      : "?"),
                yaml.c_str(), W, H, frames, interest, ring_cap);
    std::fflush(stdout);

    // Mark the backend boundary in the Tracy stream so the capture
    // confirms message delivery.
    {
        std::string m = "=== profiler-diag: backend " + label + " ===";
        ___tracy_emit_message(m.data(), m.size(), 0);
    }

    // ── Frame loop ────────────────────────────────────────────────
    double sum_ms = 0;
    uint64_t sum_rec = 0, max_rec = 0, emitted = 0;
    bool overflow = false;
    for (int f = 0; f < frames; f++) {
        // Upload interest % (the bridge memsets the header to 0 after
        // draining, so this must happen every frame).
        uint32_t pct = interest;
        if (d_hdr) {
            if (q) q->memcpy(&d_hdr->interest_pct, &pct, sizeof(pct)).wait();
            else   std::memcpy(&d_hdr->interest_pct, &pct, sizeof(pct));
        }

        profiler::DeviceRing ring;
        ring.header = d_hdr;
        ring.records = d_rec;
        ring.capacity = ring_cap;
        ring.sample_flags = d_flags;
        ring.sample_count = uint32_t(pixels);

        rt::Context ctx;
        ctx.runtime = &rt;
        ctx.cancel_flag = rt.cancel_flag;
        ctx.params = &lookup;
        ctx.scene = &v;
        ctx.width = W;
        ctx.height = H;
        ctx.accum = d_accum;
        ctx.output = d_output;
        ctx.spp_frame = 1;
        ctx.spp_total = uint32_t(f);
        ctx.frame_index = uint64_t(f);
        ctx.prof = ring;

        auto t0 = std::chrono::steady_clock::now();
        entry(&ctx);
        if (q) q->wait();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        bridge.submit_device_ring(d_hdr, d_rec, ring_cap, q);
        bridge.frame_mark();
        PROFILER_PLOT("SPP", float(f + 1));

        uint32_t wp = bridge.last_write_pos();
        bool ov = bridge.last_overflow();
        uint32_t ez = bridge.last_emitted_zones();
        sum_ms += ms;
        sum_rec += wp;
        max_rec = std::max<uint64_t>(max_rec, wp);
        overflow |= ov;
        emitted += ez;
        std::printf("  [%s] frame %2d: %6.1fms  records=%6u  zones=%6u%s\n",
                    label.c_str(), f, ms, wp, ez,
                    ov ? "  (OVERFLOW)" : "");
        std::fflush(stdout);
    }

    r.ran = true;
    r.frames_run = frames;
    r.avg_frame_ms = sum_ms / frames;
    r.avg_records = double(sum_rec) / frames;
    r.max_records = max_rec;
    r.overflow = overflow;
    r.emitted_zones = emitted;
    r.full_frame_records = r.avg_records * 100.0 / interest;
    r.required_ring = next_pow2(uint64_t(std::ceil(r.full_frame_records)));

    // ── Cleanup ────────────────────────────────────────────────────
    data.free(&rt);
    rt.dealloc(d_params);
    rt.dealloc(d_accum);
    rt.dealloc(d_output);
    pool.release(d_hdr);
    pool.release(d_rec);
    pool.release(d_flags);
    dlclose(so);
    return r;
}

// ── Reporting ────────────────────────────────────────────────────────
void print_zone_name_check(const Capture &c) {
    std::printf("  captured zone names (%zu distinct):\n", c.zone_names.size());
    for (const auto &n : c.zone_names)
        std::printf("    %s\n", n.c_str());
    std::printf("  expected device zone names:\n");
    for (const auto &e : kExpectedZoneNames) {
        bool ok = c.zone_names.count(e) > 0;
        std::printf("    [%s] %s\n", ok ? "OK" : "MISSING", e.c_str());
    }
}

int run_profiler_diag(const std::string &yaml, int width, int height,
                      int frames, uint32_t interest, uint32_t ring_cap,
                      int port, const std::vector<std::string> &backends,
                      const std::string &out_path) {
    const int W = width, H = height;

    // The Tracy client is a static global (TracyProfiler.cpp
    // `static Profiler init_order(105) s_profiler`) — it reads TRACY_PORT
    // during static initialisation, BEFORE main() runs.  So the client is
    // already bound to the port the process launched with; the capture
    // Worker must connect THERE.  A custom port therefore requires the
    // process to be launched with TRACY_PORT=<port> set.
    const char *env_port = getenv("TRACY_PORT");
    int client_port = env_port ? atoi(env_port) : 8086;
    if (port != client_port) {
        std::printf("note: --port %d != TRACY_PORT(%s) → the in-process "
                    "client listens on %d (static-init); connecting there.\n",
                    port, env_port ? env_port : "unset", client_port);
        port = client_port;
    }

    std::printf("=== profiler system diagnostic ===\n");
    std::printf("scene: %s   resolution: %dx%d   frames/backend: %d\n"
                "interest: %u%%   ring: %s   tracy port: %d\n",
                yaml.c_str(), W, H, frames, interest,
                ring_cap ? std::to_string(ring_cap).c_str() : "auto",
                port);
    std::fflush(stdout);

    tracy_bridge::Bridge bridge;
    bridge.init();  // starts the in-process Tracy client (listening).

    // Start the capture thread (connects to our own client).
    Capture cap;
    std::thread capt(capture_thread, &cap, "127.0.0.1", port);
    // Wait for the worker to connect (or fail).
    while (!cap.connected.load(std::memory_order_relaxed) &&
           cap.handshake_error.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!cap.handshake_error.empty()) {
        std::printf("FAIL: tracy handshake: %s\n", cap.handshake_error.c_str());
        cap.stop.store(true);
        capt.join();
        bridge.shutdown();
        return 1;
    }
    std::printf("tracy capture connected on port %d\n", port);
    std::fflush(stdout);

    // Resolve .so paths (native variant for software, SYCL for cpu/gpu).
    const std::string so_sycl  = "build/kernels/raytracer/libraytracer.so";
    const std::string so_native = "build/kernels/raytracer/libraytracer_native.so";

    std::vector<BackendResult> results;
    for (const auto &b : backends) {
        const std::string &so = (b == "software") ? so_native : so_sycl;
        BackendResult r = run_backend(b, b, yaml, so, W, H, frames,
                                       interest, ring_cap, bridge);
        results.push_back(std::move(r));
        std::printf("\n");
        std::fflush(stdout);
    }

    // Let the worker drain the last frames, then stop + join.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    cap.stop.store(true);
    capt.join();
    bridge.shutdown();

    // ── Verification (from the tracy::Worker capture) ───────────────
    std::printf("=== capture verification ===\n");
    bool ok = true;
    auto check = [&](const char *name, bool cond) {
        std::printf("  [%s] %s\n", cond ? "OK" : "FAIL", name);
        if (!cond) ok = false;
    };

    check("tracy client connected", cap.connected.load());
    check("GPU context present", cap.has_ctx.load());
    check("GPU context calibrated (calibrationMod > 0)",
          cap.has_calibration.load() && cap.calibration_mod.load() > 0);
    const size_t total_frames = cap.frame_marks.load();
    check("frame marks arrived (>= frames x backends run)",
          total_frames >= size_t(frames) * results.size());
    const size_t total_gpu_zones = cap.gpu_zones.load();
    check("GPU zones captured (total > 0)", total_gpu_zones > 0);
    check("messages arrived (backend boundaries)",
          cap.messages.load() >= results.size());

    // Every expected device zone name resolved to a real name.
    std::printf("\n=== zone-name resolution ===\n");
    print_zone_name_check(cap);
    for (const auto &e : kExpectedZoneNames)
        if (!cap.zone_names.count(e)) ok = false;
    // Plots: confirm "SPP" was captured.
    bool spp_plot = std::find(cap.plot_names.begin(), cap.plot_names.end(),
                              "SPP") != cap.plot_names.end();
    check("SPP plot captured", spp_plot);

    // Pipeline integrity: bridge-emitted zones ≈ worker-captured zones.
    uint64_t total_emitted = 0;
    for (const auto &r : results) total_emitted += r.emitted_zones;
    std::printf("\n=== pipeline integrity ===\n");
    std::printf("  bridge emitted zones (all backends): %llu\n"
                "  worker captured GPU zones:           %llu\n",
                (unsigned long long)total_emitted,
                (unsigned long long)total_gpu_zones);
    // The worker only logs zones it has walked so far; allow a small
    // lag but require the same order of magnitude.
    check("emitted vs captured in same ballpark",
          total_gpu_zones >= total_emitted / 4 &&
          total_gpu_zones <= total_emitted * 4);

    // ── Per-backend measurement + recommendations ─────────────────
    std::printf("\n=== per-backend measurement ===\n");
    for (const auto &r : results) {
        std::printf("\n[%s] %s\n", r.label.c_str(),
                    r.ran ? "ran" : "SKIPPED");
        if (!r.error.empty()) {
            std::printf("  error: %s\n", r.error.c_str());
            ok = false;
            continue;
        }
        std::printf("  frames run           : %d\n", r.frames_run);
        std::printf("  avg frame time       : %.1f ms\n", r.avg_frame_ms);
        std::printf("  avg records/frame    : %.0f  (at %u%% interest)\n",
                    r.avg_records, interest);
        std::printf("  max records/frame    : %llu\n",
                    (unsigned long long)r.max_records);
        std::printf("  bridge zones emitted  : %llu\n",
                    (unsigned long long)r.emitted_zones);
        std::printf("  overflowed           : %s\n",
                    r.overflow ? "YES (ring too small — data truncated)"
                               : "no");
        std::printf("  extrapolated 100%%     : %.0f records/frame\n",
                    r.full_frame_records);
        std::printf("  required ring (100%%) : %llu records (%s)\n",
                    (unsigned long long)r.required_ring,
                    tracy::MemSizeToString(r.required_ring *
                                           sizeof(profiler::DeviceRecord)));
        // Recommended sampling ratio for a few standard ring sizes.
        std::printf("  max interest for standard rings:\n");
        static const uint64_t std_rings[] = {
            256 * 1024, 1024 * 1024, 4 * 1024 * 1024,
            16 * 1024 * 1024, 64 * 1024 * 1024
        };
        for (uint64_t cap : std_rings) {
            if (r.full_frame_records <= 0) break;
            double max_pct = double(cap) * 100.0 / r.full_frame_records;
            if (max_pct > 100) max_pct = 100;
            std::printf("    ring %5llu rec (%6s): %.2f%% interest\n",
                        (unsigned long long)cap,
                        tracy::MemSizeToString(cap *
                                               sizeof(profiler::DeviceRecord)),
                        max_pct);
        }
        if (r.overflow) ok = false;
    }

    // ── Optional trace save ───────────────────────────────────────
    if (!out_path.empty()) {
        std::printf("\n(use `diag tracy --port %d --out %s` against a "
                    "running app to save a trace; this in-process diag "
                    "does not persist the loopback capture)\n",
                    port, out_path.c_str());
    }

    std::printf("\n=== result: %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

} // namespace

void register_profiler_diag(argparse::ArgumentParser &diag,
                            std::vector<DiagCommand> &commands) {
    DiagCommand cmd;
    cmd.name = "profiler";
    cmd.parser = std::make_unique<argparse::ArgumentParser>("profiler");
    auto int_action = [](const std::string &s) { return std::stoi(s); };
    auto uint_action = [](const std::string &s) { return uint32_t(std::stoul(s)); };
    cmd.parser->add_argument("yaml")
        .nargs(argparse::nargs_pattern::optional)
        .default_value(std::string("scenes/mesh_demo.yaml"))
        .help("scene YAML file (default: mesh_demo)");
    cmd.parser->add_argument("--width")
        .default_value(1920)
        .action(int_action)
        .help("render width (default 1920 = 1080p)");
    cmd.parser->add_argument("--height")
        .default_value(1080)
        .action(int_action)
        .help("render height (default 1080)");
    cmd.parser->add_argument("--frames")
        .default_value(10)
        .action(int_action)
        .help("frames per backend (default 10)");
    cmd.parser->add_argument("--interest")
        .default_value(uint32_t(1))
        .action(uint_action)
        .help("profiler interest sampling %% (default 1 — keeps the ring "
              "small; the diag extrapolates to 100%%)");
    cmd.parser->add_argument("--ring")
        .default_value(uint32_t(8 * 1024 * 1024))
        .action(uint_action)
        .help("device ring capacity in records (default 8M = 128 MiB; must "
              "hold a frame at --interest without overflow)");
    cmd.parser->add_argument("--port")
        .default_value(8086)
        .action(int_action)
        .help("Tracy capture port — the in-process client binds to the "
              "process's TRACY_PORT env (static-init; default 8086), and "
              "the loopback capture connects there.  Use --port only "
              "together with TRACY_PORT=<port> at launch.");
    cmd.parser->add_argument("--backends")
        .default_value(std::string("software,cpu,gpu"))
        .help("comma-separated backends to test (default all three)");
    cmd.parser->add_argument("--out")
        .default_value(std::string(""))
        .help("trace output path (informational — loopback capture is not "
              "persisted by this diag)");
    cmd.run = [](argparse::ArgumentParser &p) {
        std::string backends_str = p.get<std::string>("--backends");
        std::vector<std::string> backends;
        std::string cur;
        for (char ch : backends_str) {
            if (ch == ',') { if (!cur.empty()) backends.push_back(cur); cur.clear(); }
            else cur.push_back(ch);
        }
        if (!cur.empty()) backends.push_back(cur);
        return run_profiler_diag(p.get<std::string>("yaml"),
                                 p.get<int>("--width"), p.get<int>("--height"),
                                 p.get<int>("--frames"),
                                 p.get<uint32_t>("--interest"),
                                 p.get<uint32_t>("--ring"),
                                 p.get<int>("--port"), backends,
                                 p.get<std::string>("--out"));
    };
    diag.add_subparser(*cmd.parser);
    commands.push_back(std::move(cmd));
}