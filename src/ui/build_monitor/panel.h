#pragma once
#include "kernel/build_system.h"
#include <imgui.h>
#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <memory>
#include <algorithm>

// ── LiveBuildJob ──────────────────────────────────────────────────────
/// Persistent per-job state accumulated from BuildNotifications on the main thread.
struct LiveBuildJob {
    std::string kernel_name;
    float       progress = 0.0f;   ///< 0..1
    std::string last_status;       ///< last status text
    std::deque<std::string> log;   ///< recent log lines (capped)
    int         log_line_count = 0;
    bool        active = false;    ///< still building
    bool        completed = false;
    bool        failed = false;

    static constexpr int MAX_LOG_LINES = 200;
};

// ── BuildMonitor ──────────────────────────────────────────────────────
/// Main-thread-only helper that consumes BuildNotifications and maintains
/// per-job state for the UI to render.
class BuildMonitor {
public:
    BuildMonitor() = default;

    /// Call once per frame from the main thread with all pending notifications.
    void ingest(std::vector<BuildNotification> notifications) {
        for (auto &n : notifications) {
            auto &job = jobs_[n.kernel_name];
            job.kernel_name = n.kernel_name;

            switch (n.type) {
            case BuildNotification::BuildStarted:
                job = LiveBuildJob{};
                job.kernel_name = n.kernel_name;
                job.active = true;
                job.progress = 0.0f;
                job.last_status = n.text;
                break;

            case BuildNotification::BuildProgress:
                job.active = true;
                job.progress = n.progress;
                job.last_status = n.text;
                break;

            case BuildNotification::BuildLogLine:
                job.active = true;
                add_log(job, n.text);
                break;

            case BuildNotification::BuildCompleted:
                job.active = false;
                job.completed = true;
                job.failed = false;
                job.progress = 1.0f;
                job.last_status = n.text;
                break;

            case BuildNotification::BuildFailed:
                job.active = false;
                job.completed = true;
                job.failed = true;
                job.progress = n.progress;
                job.last_status = n.text;
                break;
            }
        }
    }

    /// Remove all completed/failed jobs.
    void clear_completed() {
        for (auto it = jobs_.begin(); it != jobs_.end(); ) {
            if (!it->second.active) {
                it = jobs_.erase(it);
            } else {
                ++it;
            }
        }
    }

    /// Remove a specific job by name.
    void clear_job(const std::string &name) {
        jobs_.erase(name);
    }

    /// Access all tracked jobs.
    const std::unordered_map<std::string, LiveBuildJob> &jobs() const { return jobs_; }

    /// Number of currently building jobs.
    int active_count() const {
        int n = 0;
        for (auto &[_, j] : jobs_) if (j.active) n++;
        return n;
    }

private:
    void add_log(LiveBuildJob &job, const std::string &line) {
        if ((int)job.log.size() >= LiveBuildJob::MAX_LOG_LINES)
            job.log.pop_front();
        job.log.push_back(line);
    }

    std::unordered_map<std::string, LiveBuildJob> jobs_;
};

// ── Render helpers ────────────────────────────────────────────────────

/// Render a single build job row (used inside the build monitor window).
static void render_build_job(const LiveBuildJob &job) {
    ImGui::PushID(job.kernel_name.c_str());

    // Header: kernel name + status badge
    float line_h = ImGui::GetTextLineHeight();
    ImVec4 status_col = job.active ? ImVec4(0.6f, 0.8f, 1.0f, 1)
                     : job.failed ? ImVec4(1.0f, 0.3f, 0.3f, 1)
                     : ImVec4(0.3f, 1.0f, 0.3f, 1);
    const char *badge = job.active   ? "BUILDING"
                      : job.failed   ? "FAILED"
                      : "OK";

    ImGui::TextUnformatted(job.kernel_name.c_str());
    ImGui::SameLine();
    ImGui::TextColored(status_col, " [%s]", badge);

    // Progress bar
    ImGui::ProgressBar(job.progress, ImVec2(-1, line_h));
    if (!job.last_status.empty()) {
        ImGui::TextUnformatted(job.last_status.c_str());
    }

    // Log (collapsible)
    if (!job.log.empty()) {
        char label[64];
        snprintf(label, sizeof(label), "Log (%zu lines)", job.log.size());
        if (ImGui::TreeNode(label)) {
            ImGui::BeginChild("##log_scroll", ImVec2(0, std::min((float)job.log.size() * line_h, 200.0f)),
                              true, ImGuiWindowFlags_HorizontalScrollbar);
            for (auto &line : job.log) {
                ImGui::TextUnformatted(line.c_str());
            }
            if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
                ImGui::SetScrollHereY(1.0f);
            ImGui::EndChild();
            ImGui::TreePop();
        }
    }

    ImGui::Separator();
    ImGui::PopID();
}

/// Render the full build monitor panel (to be called inside an ImGui window).
static void render_build_monitor_panel(BuildMonitor &monitor) {
    if (monitor.active_count() == 0 && monitor.jobs().empty()) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1), "No build jobs.");
        return;
    }

    // Toolbar
    if (ImGui::SmallButton("Clear Completed")) {
        monitor.clear_completed();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("  %d active / %zu total", monitor.active_count(), monitor.jobs().size());
    ImGui::Separator();

    // One section per job
    for (auto &[name, job] : monitor.jobs()) {
        render_build_job(job);
    }
}
