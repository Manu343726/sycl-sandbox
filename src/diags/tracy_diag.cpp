// `diag tracy` — Tracy capture/receiver diagnostics.
//
// Connects to a RUNNING sandbox instance (the app's embedded Tracy client,
// which listens on port 8086 by default) using Tracy's server-side capture
// machinery — tracy::Worker + FileWrite + TracyPrint — i.e. exactly the code
// tracy-capture / tracy-csvexport use.  While connected it logs everything
// the client sends:
//
//   * connection/handshake + timer resolution + app info
//   * GPU contexts (calibration: period, calibrationMod, calibrated times)
//   * GPU zones (name, thread, calibrated host-ns start/end/duration) —
//     the sandbox feeds these from its device-side profiler ring
//   * frames (per frame set, incl. the app's named "render" set)
//   * plots (DEV_PLOT series: gpu_kernel_ms etc.)
//   * messages (DEV_MSG)
//
// On disconnect (or --seconds / Ctrl-C) it prints a summary and can save the
// raw trace with `--out trace.tracy` (same FileWrite::Open + Worker::Write
// path tracy-capture uses), keeping the capture reproducible.

#include "diags.h"

#include "TracyFileWrite.hpp"
#include "TracyPrint.hpp"
#include "TracyWorker.hpp"

#include <signal.h>

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

std::atomic<bool> s_disconnect{false};

void sigint_handler(int) {
    s_disconnect.store(true, std::memory_order_relaxed);
}

/// Resolve a frame set's display name, mirroring View::GetFrameSetName.
/// name==0 is the base set; name>>63 is a Vsync-count set; otherwise the
/// name is a string index into the worker.
const char *frame_set_name(const tracy::Worker &worker,
                           const tracy::FrameData &fd) {
    if (fd.name == 0) return "Frames";
    if (fd.name >> 63 != 0) {
        static char buf[64];
        std::snprintf(buf, sizeof(buf), "[%u] Vsync", uint32_t(fd.name));
        return buf;
    }
    return worker.GetString(fd.name);
}

/// Walk a GPU zone timeline (one thread of one context) recursively and log
/// any zone that wasn't logged before.  `seen` tracks per-zone pointers so
/// in-flight / partially-built trees are logged exactly once.  Timing is the
/// already-calibrated host-ns GpuStart()/GpuEnd().
void walk_gpu_timeline(const tracy::Worker &worker,
                       const tracy::Vector<tracy::short_ptr<tracy::GpuEvent>> &vec,
                       uint64_t thread_id, const char *thread_name, int depth,
                       std::unordered_set<const tracy::GpuEvent *> &seen) {
    for (const auto &sp : vec) {
        const tracy::GpuEvent *ev = sp;
        if (!ev) continue;
        // Log a zone only once it's complete (GpuStart/GpuEnd set by the
        // trailing gpu_time events).  Zones still in flight are NOT marked
        // as seen, so a later poll logs them when the tree settles.
        if (ev->GpuStart() >= 0 && ev->GpuEnd() >= 0 &&
            seen.insert(ev).second) {
            const auto start = ev->GpuStart();
            const auto end = ev->GpuEnd();
            std::printf("  %*sGPU zone  %-40s thread=%-24s start=%10.3fms "
                        "end=%10.3fms  dur=%8.1fus\n",
                        depth * 2, "", worker.GetZoneName(*ev), thread_name,
                        double(start) / 1e6, double(end) / 1e6,
                        double(end - start) / 1e3);
        }
        if (ev->Child() >= 0) {
            walk_gpu_timeline(worker, worker.GetGpuChildren(ev->Child()),
                              thread_id, thread_name, depth + 1, seen);
        }
    }
}

/// Log newly-arrived profiler data since the last poll.  All reads happen
/// under the worker's main-thread lock (the same protocol the Tracy GUI uses
/// for live capture); the Worker's network/exec threads hand over the lock
/// so we see a consistent snapshot.
struct DiagSession {
    std::unordered_map<const tracy::FrameData *, size_t> logged_frames;
    std::unordered_map<const tracy::PlotData *, size_t> logged_plot_points;
    std::unordered_set<const tracy::GpuEvent *> logged_gpu_zones;
    size_t logged_messages = 0;
    std::vector<bool> logged_contexts;
};

void log_new_data(tracy::Worker &worker, DiagSession &s) {
    auto lock = worker.ObtainLockForMainThread();

    // ---- frames ----
    const auto &frames = worker.GetFrames();
    for (const auto *fd : frames) {
        const size_t n = worker.GetFrameCount(*fd);
        const auto &prev = s.logged_frames[fd];
        if (n > prev) {
            std::printf("  frames[%s]: %zu -> %zu frames\n",
                        frame_set_name(worker, *fd), prev, n);
            for (size_t i = prev; i < n; i++) {
                const auto begin = worker.GetFrameBegin(*fd, i);
                const auto end = worker.GetFrameEnd(*fd, i);
                if (begin < 0 || end < 0) continue;
                std::printf("    #%4zu  %10.3fms -> %10.3fms  (%8.3fms)\n", i,
                            double(begin) / 1e6, double(end) / 1e6,
                            double(worker.GetFrameTime(*fd, i)) / 1e6);
            }
        }
        s.logged_frames[fd] = n;
    }

    // ---- GPU contexts + zones ----
    const auto &gpu = worker.GetGpuData();
    if (s.logged_contexts.size() < gpu.size()) {
        s.logged_contexts.resize(gpu.size(), false);
    }
    for (size_t ci = 0; ci < gpu.size(); ci++) {
        const auto *ctx = gpu[ci];
        if (!s.logged_contexts[ci]) {
            s.logged_contexts[ci] = true;
            std::printf("  gpu ctx #%zu: count=%" PRIu64 " period=%g ns "
                        "hasPeriod=%d hasCalibration=%d "
                        "calibratedGpuTime=%" PRId64 " calibratedCpuTime=%" PRId64
                        " calibrationMod=%g timeDiff=%" PRId64 " overflow=%" PRIu64
                        "\n",
                        ci, ctx->count, ctx->period, int(ctx->hasPeriod),
                        int(ctx->hasCalibration), ctx->calibratedGpuTime,
                        ctx->calibratedCpuTime, ctx->calibrationMod,
                        ctx->timeDiff, ctx->overflow);
        }
        for (auto &td : ctx->threadData) {
            walk_gpu_timeline(worker, td.second.timeline, td.first,
                              worker.GetThreadName(td.first), 0,
                              s.logged_gpu_zones);
        }
    }

    // ---- plots ----
    const auto &plots = worker.GetPlots();
    for (const auto *plot : plots) {
        const size_t n = plot->data.size();
        const auto &prev = s.logged_plot_points[plot];
        if (n > prev) {
            std::printf("  plot[%s]: %zu -> %zu points (min=%.3g max=%.3g "
                        "sum=%.3g)\n",
                        worker.GetString(plot->name), prev, n, plot->min,
                        plot->max, plot->sum);
            for (size_t i = prev; i < n; i++) {
                const auto &pt = plot->data[i];
                std::printf("    %12.3fms  %10.3g\n", double(pt.time.Val()) / 1e6,
                            pt.val);
            }
        }
        s.logged_plot_points[plot] = n;
    }

    // ---- messages ----
    const auto &messages = worker.GetMessages();
    for (size_t i = s.logged_messages; i < messages.size(); i++) {
        const auto *msg = messages[i].get();
        if (!msg) continue;
        const auto tid = worker.DecompressThread(msg->thread);
        std::printf("  msg  %12.3fms  [%s]  %s\n", double(msg->time) / 1e6,
                    worker.GetThreadName(tid), worker.GetString(msg->ref));
    }
    s.logged_messages = messages.size();
}

int run_tracy_diag(const std::string &address, int port, int seconds,
                   const std::string &out_path) {
    std::printf("Connecting to %s:%i (Tracy capture)...\n", address.c_str(),
                port);
    std::fflush(stdout);

    tracy::Worker worker(address.c_str(), uint16_t(port), -1);
    while (!worker.HasData()) {
        const auto hs = worker.GetHandshakeStatus();
        if (hs == tracy::HandshakeProtocolMismatch) {
            std::printf("protocol mismatch: incompatible Tracy versions on "
                        "client and server\n");
            return 1;
        }
        if (hs == tracy::HandshakeNotAvailable) {
            std::printf("client already has a server connected\n");
            return 2;
        }
        if (hs == tracy::HandshakeDropped) {
            std::printf("client dropped the connection during handshake\n");
            return 3;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::printf("Connected.  Timer resolution: %s\n",
                tracy::TimeToString(worker.GetResolution()));
    const auto &app_info = worker.GetAppInfo();
    for (const auto &line : app_info) {
        std::printf("  app info: %s\n", worker.GetString(line));
    }
    std::fflush(stdout);

    struct sigaction sa, old_sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = sigint_handler;
    sigaction(SIGINT, &sa, &old_sa);

    DiagSession session;
    const auto t0 = std::chrono::high_resolution_clock::now();

    std::printf("Capturing profiler data...\n");
    std::fflush(stdout);
    while (worker.IsConnected()) {
        if (s_disconnect.load(std::memory_order_relaxed)) {
            worker.Disconnect();
            s_disconnect.store(false, std::memory_order_relaxed);
            break;
        }
        log_new_data(worker, session);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (seconds != -1) {
            const auto dur = std::chrono::high_resolution_clock::now() - t0;
            if (std::chrono::duration_cast<std::chrono::seconds>(dur).count() >=
                seconds) {
                worker.Disconnect();
            }
        }
    }

    // One final sweep for data that arrived during the last poll, then summary.
    log_new_data(worker, session);

    const auto elapsed = std::chrono::high_resolution_clock::now() - t0;
    const auto first = worker.GetFirstTime();
    const auto last = worker.GetLastTime();

    std::printf("\nDisconnected.  Summary:\n");
    std::printf("  frames:  %" PRIu64 "\n",
                worker.GetFrameCount(*worker.GetFramesBase()));
    std::printf("  zones:   %s (cpu)  %s (gpu)\n",
                tracy::RealToString(worker.GetZoneCount()),
                tracy::RealToString(worker.GetGpuZoneCount()));
    std::printf("  time:    %s\n",
                tracy::TimeToString(last - first));
    std::printf("  wall:    %s\n",
                tracy::TimeToString(std::chrono::duration_cast<
                                    std::chrono::nanoseconds>(elapsed).count()));
    std::fflush(stdout);

    if (!out_path.empty()) {
        std::printf("Saving trace...\n");
        std::fflush(stdout);
        auto f = std::unique_ptr<tracy::FileWrite>(
            tracy::FileWrite::Open(out_path.c_str(), tracy::FileCompression::Zstd,
                                   3, 4));
        if (f) {
            worker.Write(*f, false);
            f->Finish();
            const auto stats = f->GetCompressionStatistics();
            std::printf("Trace size %s (%.2f%% ratio)\n",
                        tracy::MemSizeToString(stats.second),
                        100.f * stats.second / stats.first);
        } else {
            std::printf("Failed to open %s for writing!\n", out_path.c_str());
            return 4;
        }
    }

    return 0;
}

} // namespace

void register_tracy_diag(argparse::ArgumentParser &diag,
                         std::vector<DiagCommand> &commands) {
    DiagCommand cmd;
    cmd.name = "tracy";
    cmd.parser = std::make_unique<argparse::ArgumentParser>("tracy");
    auto int_action = [](const std::string &s) { return std::stoi(s); };
    cmd.parser->add_argument("--address")
        .default_value(std::string("127.0.0.1"))
        .help("address of the running sandbox Tracy client");
    cmd.parser->add_argument("--port")
        .default_value(8086)
        .action(int_action)
        .help("Tracy data port (TRACY_PORT override on the client)");
    cmd.parser->add_argument("--seconds")
        .default_value(-1)
        .action(int_action)
        .help("stop capture after N seconds (-1 = until disconnect)");
    cmd.parser->add_argument("--out")
        .default_value(std::string(""))
        .help("save the raw capture to a .tracy file (tracy-capture format)");
    cmd.run = [](argparse::ArgumentParser &p) {
        return run_tracy_diag(p.get<std::string>("--address"),
                              p.get<int>("--port"), p.get<int>("--seconds"),
                              p.get<std::string>("--out"));
    };
    diag.add_subparser(*cmd.parser);
    commands.push_back(std::move(cmd));
}
