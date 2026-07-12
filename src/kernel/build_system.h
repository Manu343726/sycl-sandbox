#pragma once
#include "library.h"
#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <functional>
#include <atomic>
#include <memory>

class SourceWatcher; // forward decl — included in .cpp

// ── BuildResult ───────────────────────────────────────────────────────
/// Result of a completed kernel build, ready for the main thread to consume.
struct BuildResult {
    std::string kernel_name;
    bool        success     = false;
    int         exit_code   = -1;
    std::string log;                ///< full build output
    float       peak_progress = 0.0f; ///< highest progress % seen
};

// ── BuildNotification ─────────────────────────────────────────────────
/// Lightweight notification for the UI thread (e.g. progress bar text).
struct BuildNotification {
    enum Type : uint8_t {
        BuildStarted,
        BuildProgress,
        BuildCompleted,
        BuildFailed,
        BuildLogLine,   ///< a single line of build output (for live log view)
    };
    Type        type;
    std::string kernel_name;
    std::string text;
    float       progress = 0.0f;   ///< 0.0 – 1.0
};

// ── KernelBuildSystem ─────────────────────────────────────────────────
/// Manages background kernel compilation with real-time progress reporting.
///
/// Usage:
/// @code
///   KernelBuildSystem builder(build_dir, kernels);
///   builder.build_async("raytracer");
///   // ... on main thread each frame:
///   for (auto &n : builder.poll_notifications()) { /* update UI */ }
///   for (auto &r : builder.poll_results()) {
///       if (r.success) { KernelHandle *kh = kernels.load(r.kernel_name); }
///   }
/// @endcode
class KernelBuildSystem {
public:
    using NotificationCallback = std::function<void(const BuildNotification&)>;

    /// @param build_dir  Absolute path to the CMake build directory.
    /// @param lib        Reference to the KernelLibrary for rebuild calls.
    KernelBuildSystem(std::string build_dir, KernelLibrary &lib);

    ~KernelBuildSystem();

    // ── Lifecycle ────────────────────────────────────────────────────

    /// Cancel all running builds and wait for them to finish.
    void shutdown();

    // ── Build control ────────────────────────────────────────────────

    /// Start building a kernel in a background thread.
    /// Does nothing if a build for that kernel is already running.
    void build_async(const std::string &kernel_name);

    /// Build a kernel synchronously (blocks until complete).
    /// Use for first-time builds or backend switches where the kernel
    /// must be ready before proceeding.
    BuildResult build_sync(const std::string &kernel_name);

    /// Cancel a running build (best-effort — sets a flag, thread cleans up).
    void cancel(const std::string &kernel_name);

    /// Cancel all running builds.
    void cancel_all();

    // ── Polling (main thread) ────────────────────────────────────────

    /// Collect completed build results since last call.
    std::vector<BuildResult> poll_results();

    /// Collect pending notifications since last call.
    std::vector<BuildNotification> poll_notifications();

    // ── Queries ──────────────────────────────────────────────────────

    /// Whether a build is currently in progress for this kernel.
    bool is_building(const std::string &kernel_name) const;

    /// Number of builds currently running.
    int active_count() const;

    /// Maximum number of concurrent build threads.
    int max_concurrent() const { return max_concurrent_; }
    void set_max_concurrent(int n) { max_concurrent_ = std::max(1, n); }

    // ── Watch registration ──────────────────────────────────────────
    /// Attach a SourceWatcher so the build system can register kernels
    /// for file-change monitoring.  Ownership is not transferred.
    void set_watcher(SourceWatcher &watcher) { watcher_ = &watcher; }

    /// Set up a kernel for building + watching — computes build target,
    /// registers its source directory for dependency tracking, and
    /// registers it with the attached watcher (if any).
    /// Call once per kernel at startup.
    void setup_kernel(const std::string &kernel_name);

    // ── Notification callback ────────────────────────────────────────
    /// If set, called from the build thread for each progress update.
    /// The callback must be thread-safe (e.g. post to a thread-safe queue).
    void set_notification_callback(NotificationCallback cb) {
        std::lock_guard<std::mutex> lock(cb_mtx_);
        callback_ = std::move(cb);
    }

private:
    // ── Internal types ───────────────────────────────────────────────
    struct ActiveBuild {
        std::thread                  thread;
        std::shared_ptr<std::atomic<bool>> cancel_flag;
        std::string                  kernel_name;
    };

    // ── Build thread entry point ─────────────────────────────────────
    void build_thread_fn(const std::string &kernel_name,
                         std::shared_ptr<std::atomic<bool>> cancel_flag);

    // ── Progress parsing ─────────────────────────────────────────────
    /// Parse a line of build output and extract progress (0..1) if present.
    static float parse_progress(const std::string &line);

    // ── Members ──────────────────────────────────────────────────────
    std::string build_dir_;
    KernelLibrary &lib_;
    SourceWatcher *watcher_ = nullptr;   ///< non-owning, set via set_watcher()

    mutable std::mutex mtx_;
    std::vector<std::unique_ptr<ActiveBuild>> active_builds_;
    std::vector<BuildResult> pending_results_;
    std::vector<BuildNotification> pending_notifications_;

    int max_concurrent_ = 4;

    // Optional callback invoked from build threads
    NotificationCallback callback_;
    mutable std::mutex cb_mtx_;
};
