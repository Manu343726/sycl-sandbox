#include "build_system.h"
#include "../io/watcher.h"
#include <sycl-sandbox/profiler.h>
#include <spdlog/spdlog.h>
#include <cstdio>
#include <cstring>
#include <memory>
#include <algorithm>
#include <regex>
#include <sstream>
#include <pthread.h>

// ── Constructor / Destructor ──────────────────────────────────────────

KernelBuildSystem::KernelBuildSystem(std::string build_dir, KernelLibrary &lib)
    : build_dir_(std::move(build_dir))
    , lib_(lib) {
    spdlog::debug("[build] KernelBuildSystem created (build_dir='{}')", build_dir_);
}

KernelBuildSystem::~KernelBuildSystem() {
    shutdown();
}

// ── Lifecycle ─────────────────────────────────────────────────────────

void KernelBuildSystem::shutdown() {
    cancel_all();

    std::vector<std::unique_ptr<ActiveBuild>> builds;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        builds = std::move(active_builds_);
    }
    for (auto &b : builds) {
        if (b->thread.joinable()) {
            b->thread.join();
        }
    }
    spdlog::debug("[build] shutdown complete");
}

// ── Build control ─────────────────────────────────────────────────────

void KernelBuildSystem::build_async(const std::string &kernel_name) {
    // Check if already building
    {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto &b : active_builds_) {
            if (b->kernel_name == kernel_name) {
                spdlog::debug("[build] '{}' already building, skipping", kernel_name);
                return;
            }
        }
        // Check concurrency limit
        if ((int)active_builds_.size() >= max_concurrent_) {
            spdlog::warn("[build] max concurrent builds ({}) reached, queuing will wait",
                         max_concurrent_);
            // Drop the oldest build to make room
            active_builds_.erase(active_builds_.begin());
        }
    }

    auto cancel_flag = std::make_shared<std::atomic<bool>>(false);
    auto done = std::make_shared<std::atomic<bool>>(false);
    auto thread = std::thread(&KernelBuildSystem::build_thread_fn, this,
                               kernel_name, cancel_flag, done);

    std::lock_guard<std::mutex> lock(mtx_);
    auto build = std::make_unique<ActiveBuild>();
    build->thread = std::move(thread);
    build->cancel_flag = std::move(cancel_flag);
    build->done = std::move(done);
    build->kernel_name = kernel_name;
    active_builds_.push_back(std::move(build));
}

void KernelBuildSystem::cancel(const std::string &kernel_name) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto &b : active_builds_) {
        if (b->kernel_name == kernel_name) {
            b->cancel_flag->store(true);
            spdlog::debug("[build] cancel requested for '{}'", kernel_name);
            break;
        }
    }
}

void KernelBuildSystem::cancel_all() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto &b : active_builds_) {
        b->cancel_flag->store(true);
    }
    spdlog::debug("[build] all builds cancelled");
}

// ── Kernel setup (build target + source dir + watcher) ────────────────

void KernelBuildSystem::setup_kernel(const std::string &kernel_name) {
    std::string build_target = lib_.build_target_name(kernel_name);
    std::string source_dir = std::string("kernels/") + kernel_name;

    lib_.add_kernel_source_dir(kernel_name, source_dir);
    spdlog::debug("[build] setup kernel '{}' (target={}, src={})",
                 kernel_name, build_target, source_dir);

    if (watcher_) {
        watcher_->watch_kernel(kernel_name, source_dir, build_target);
    }
}

// ── Polling ───────────────────────────────────────────────────────────

std::vector<BuildResult> KernelBuildSystem::poll_results() {
    // Join finished threads and move results
    std::lock_guard<std::mutex> lock(mtx_);

    // Reap entries whose build thread has finished.  A build is finished
    // when either:
    //  - its `done` flag is set (natural completion: the thread ran to
    //    the end of build_thread_fn and the RAII guard flipped it), or
    //  - it was cancelled (cancel_flag set — preserved for the explicit
    //    cancel path).
    // Previously only cancelled builds were reaped, so a naturally-
    // completed build stayed in active_builds_ forever and blocked
    // subsequent rebuilds with "already building, skipping".
    for (auto it = active_builds_.begin(); it != active_builds_.end(); ) {
        const bool finished = (*it)->done->load(std::memory_order_acquire) ||
                              (*it)->cancel_flag->load();
        if (finished) {
            if ((*it)->thread.joinable()) {
                (*it)->thread.detach(); // don't block — thread already exited
            }
            it = active_builds_.erase(it);
        } else {
            ++it;
        }
    }

    std::vector<BuildResult> results = std::move(pending_results_);
    pending_results_.clear();
    return results;
}

std::vector<BuildNotification> KernelBuildSystem::poll_notifications() {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<BuildNotification> notifications = std::move(pending_notifications_);
    pending_notifications_.clear();
    return notifications;
}

// ── Queries ───────────────────────────────────────────────────────────

bool KernelBuildSystem::is_building(const std::string &kernel_name) const {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto &b : active_builds_) {
        if (b->kernel_name == kernel_name) return true;
    }
    return false;
}

int KernelBuildSystem::active_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return (int)active_builds_.size();
}

// ── Notification helpers ──────────────────────────────────────────────

static void push_notification(std::vector<BuildNotification> &vec,
                               const std::string &kn, BuildNotification::Type t,
                               const std::string &text, float progress) {
    vec.push_back({t, kn, text, progress});
}

// ── Build output parsing ──────────────────────────────────────────────

/// Parse a build output line and log it to spdlog with the appropriate
/// level.  Returns the progress (0..1) if detected, or -1.0f.
///
/// Handles these output formats:
///   CMake/Make:   [35%] Building CXX object ...
///   Ninja:        [5/42] Building CXX object ...
///   GCC/Clang error:   file.cpp:12:5: error: ...
///   GCC/Clang warning: file.cpp:12:5: warning: ...
///   AdaptiveCpp/SYCL:  file.cpp:12: error: ...
///   ld error:          ld: ... undefined reference ...
static float parse_and_log_build_line(const std::string &kernel_name,
                                      const std::string &line) {
    // Clang/GCC error:   file:line:col: error:   or  file:line: error:
    // ld error:          ld: ... undefined reference ...
    static std::regex re_error(
        R"(^(?:.*:\d+:\d+:\s*error:|.*:\d+:\s*error:|ld:\s|.*fatal error|.*Error\s))",
        std::regex::ECMAScript | std::regex::icase);
    // Clang/GCC warning: file:line:col: warning:
    // AdaptiveCpp deprecation: file:line:col: warning:
    static std::regex re_warning(
        R"(^(?:.*:\d+:\d+:\s*warning:|.*:\d+:\s*warning:|.*warning:))",
        std::regex::ECMAScript | std::regex::icase);

    // Progress patterns: [35%] or [5/42]
    static std::regex re_pct(R"(\[(\d+)%\])");
    static std::regex re_count(R"(\[(\d+)/(\d+)\])");

    float progress = -1.0f;

    // Check for progress first
    std::smatch m;
    if (std::regex_search(line, m, re_pct)) {
        progress = std::stoi(m[1]) / 100.0f;
    } else if (std::regex_search(line, m, re_count)) {
        int num = std::stoi(m[1]);
        int den = std::stoi(m[2]);
        progress = (den > 0) ? (float)num / den : 0.0f;
    }

    // Route to spdlog based on content
    if (std::regex_search(line, m, re_error)) {
        spdlog::error("[build] {}: {}", kernel_name, line);
    } else if (std::regex_search(line, m, re_warning)) {
        spdlog::warn("[build] {}: {}", kernel_name, line);
    } else if (progress >= 0.0f) {
        spdlog::debug("[build] {}: {:.0f}%  {}", kernel_name, progress * 100.0f, line);
    } else {
        // Generic build output — trace level (Building CXX, Linking, etc.
        // is extremely verbose during a build and drowns out app messages).
        spdlog::trace("[build] {}: {}", kernel_name, line);
    }

    return progress;
}

// ── Build thread function ─────────────────────────────────────────────

void KernelBuildSystem::build_thread_fn(const std::string &kernel_name,
                                          std::shared_ptr<std::atomic<bool>> cancel_flag,
                                          std::shared_ptr<std::atomic<bool>> done) {
    pthread_setname_np(pthread_self(), "sycl-build");
    PROFILER_FUNCTION();

    // RAII guard: flip the entry's `done` flag on every return path
    // (natural completion, popen failure, or cancellation) so
    // poll_results can reap this active_builds_ entry.  Without it the
    // build would stay forever in active_builds_ and later build_async
    // calls would refuse with "already building".
    struct DoneGuard {
        std::shared_ptr<std::atomic<bool>> done;
        ~DoneGuard() { done->store(true, std::memory_order_release); }
    } guard{done};

    // Use the KernelLibrary to determine the CMake target name
    std::string target = lib_.build_target_name(kernel_name);

    // Build command: run cmake and capture stderr + stdout combined.
    // Use --verbose to ensure error messages are visible in output.
    std::string cmd = "cmake --build " + build_dir_ + " --target " + target +
                      " -- -j4 2>&1";

    // Notify: build started
    {
        std::lock_guard<std::mutex> lock(mtx_);
        push_notification(pending_notifications_, kernel_name,
                          BuildNotification::BuildStarted,
                          "Starting build for " + kernel_name + " (" + target + ")", 0.0f);
        NotificationCallback cb;
        {
            std::lock_guard<std::mutex> cblock(cb_mtx_);
            cb = callback_;
        }
        if (cb) {
            cb(BuildNotification{BuildNotification::BuildStarted, kernel_name,
                                 "Starting build", 0.0f});
        }
    }

    spdlog::info("[build] building target '{}' for kernel '{}'", target, kernel_name);

    // Run cmake via popen
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        spdlog::error("[build] popen failed for '{}'", kernel_name);
        std::lock_guard<std::mutex> lock(mtx_);
        push_notification(pending_notifications_, kernel_name,
                          BuildNotification::BuildFailed,
                          "popen failed", 0.0f);
        return;
    }

    float peak_progress = 0.0f;
    std::string full_log;
    char line_buf[4096];
    while (fgets(line_buf, sizeof(line_buf), pipe)) {
        if (cancel_flag->load()) {
            spdlog::debug("[build] '{}' cancelled, terminating build process", kernel_name);
            pclose(pipe);
            {
                std::lock_guard<std::mutex> lock(mtx_);
                push_notification(pending_notifications_, kernel_name,
                                  BuildNotification::BuildFailed,
                                  "Cancelled", peak_progress);
            }
            return;
        }

        std::string line(line_buf);
        if (!line.empty() && line.back() == '\n') line.pop_back();
        if (!line.empty() && line.back() == '\r') line.pop_back();

        full_log += line + "\n";

        // Parse and log each line with appropriate severity
        float line_progress = parse_and_log_build_line(kernel_name, line);
        float effective_progress = line_progress;
        if (line_progress >= 0.0f) {
            peak_progress = std::max(peak_progress, line_progress);
            effective_progress = peak_progress;
        } else {
            effective_progress = peak_progress;
        }

        // Push notification for UI layer
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (line_progress >= 0.0f) {
                push_notification(pending_notifications_, kernel_name,
                                  BuildNotification::BuildProgress,
                                  line, line_progress);
            } else {
                push_notification(pending_notifications_, kernel_name,
                                  BuildNotification::BuildLogLine,
                                  line, peak_progress);
            }
        }

        // Thread-safe callback
        {
            NotificationCallback cb;
            {
                std::lock_guard<std::mutex> cblock(cb_mtx_);
                cb = callback_;
            }
            if (cb) {
                BuildNotification::Type t = (line_progress >= 0.0f)
                                              ? BuildNotification::BuildProgress
                                              : BuildNotification::BuildLogLine;
                cb(BuildNotification{t, kernel_name, line, effective_progress});
            }
        }
    }

    int exit_code = pclose(pipe);
    bool success = (exit_code == 0);

    if (!success) {
        spdlog::error("[build] '{}' FAILED (exit {})", kernel_name, exit_code);
    } else {
        spdlog::info("[build] '{}' succeeded", kernel_name);
    }

    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (success) {
            push_notification(pending_notifications_, kernel_name,
                              BuildNotification::BuildCompleted,
                              "Build succeeded", 1.0f);
        } else {
            push_notification(pending_notifications_, kernel_name,
                              BuildNotification::BuildFailed,
                              "Build failed (exit " + std::to_string(exit_code) + ")",
                              peak_progress);
        }

        // Store full build log for diagnostics
        pending_results_.push_back({kernel_name, success, exit_code,
                                     std::move(full_log), peak_progress});

        NotificationCallback cb;
        {
            std::lock_guard<std::mutex> cblock(cb_mtx_);
            cb = callback_;
        }
        if (cb) {
            auto t = success ? BuildNotification::BuildCompleted
                             : BuildNotification::BuildFailed;
            cb(BuildNotification{t, kernel_name,
                                 success ? "Build succeeded" : "Build failed",
                                 success ? 1.0f : peak_progress});
        }
    }
}

