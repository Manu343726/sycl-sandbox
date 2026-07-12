#pragma once
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/base_sink.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <ctime>
#include <cstdint>
#include <algorithm>

/// spdlog sink that captures log entries into a circular buffer and
/// renders them in an ImGui window with filtering and colour-coding.
///
/// Usage:
/// @code
///   #include "io/log_sink.h"
///   auto g_logSink = std::make_shared<LogSink>();
///   spdlog::default_logger()->sinks().push_back(g_logSink);
///
///   // Each frame:
///   g_logSink->draw_imgui("Logs", &show_logs);
/// @endcode
class LogSink final : public spdlog::sinks::base_sink<std::mutex> {
public:
    static constexpr int MAX_ENTRIES = 2048;

    struct Entry {
        std::chrono::system_clock::time_point timestamp;
        spdlog::level::level_enum level;
        std::string message;
    };

    LogSink() { set_pattern("%v"); }

    /// Return a subset of entries matching the current filter.
    /// Caller should hold the sink_mutex_ if reading concurrently.
    const std::vector<Entry> &entries() const { return entries_; }

    /// Clear all buffered log entries.
    /// NOTE: caller must hold this->mutex_ if clear may race with sink_it_.
    void clear() {
        entries_.clear();
        write_cursor_ = 0;
        full_ = false;
    }

    /// ── ImGui log viewer widget ─────────────────────────────────────
    ///
    /// Renders the log buffer with:
    ///   • Level-based colour coding (trace=grey, debug=white, info=green,
    ///     warn=yellow, err=red, critical=bold red)
    ///   • Text filter (case-insensitive substring match)
    ///   • Per-level filter checkboxes
    ///   • Auto-scroll toggle + scroll-to-bottom button
    ///   • Clear button
    ///
    /// Call once per frame inside an ImGui window.
    ///
    /// @param title  Window title passed to ImGui::Begin.
    /// @param p_open Optional pointer to bool for closeable window.
    void draw_imgui(const char *title, bool *p_open = nullptr);

private:
    std::vector<Entry> entries_;
    size_t write_cursor_ = 0;
    bool full_ = false;

    /// spdlog sink callback.
    void sink_it_(const spdlog::details::log_msg &msg) override {
        spdlog::memory_buf_t formatted;
        this->formatter_->format(msg, formatted);

        Entry e;
        e.timestamp = msg.time;
        e.level = msg.level;
        e.message = fmt::to_string(formatted);
        // Strip trailing newlines from spdlog's formatter
        while (!e.message.empty() && (e.message.back() == '\n' || e.message.back() == '\r'))
            e.message.pop_back();

        if (entries_.size() < MAX_ENTRIES) {
            entries_.push_back(std::move(e));
        } else {
            entries_[write_cursor_] = std::move(e);
            write_cursor_ = (write_cursor_ + 1) % MAX_ENTRIES;
            full_ = true;
        }
    }

    void flush_() override {}
};

// ── Helpers ───────────────────────────────────────────────────────────

/// Map spdlog level to an ImGui colour for the level badge.
inline ImU32 log_level_color(spdlog::level::level_enum lvl) {
    switch (lvl) {
        case spdlog::level::trace:    return IM_COL32(140, 140, 140, 255);
        case spdlog::level::debug:    return IM_COL32(200, 200, 200, 255);
        case spdlog::level::info:     return IM_COL32(100, 220, 100, 255);
        case spdlog::level::warn:     return IM_COL32(240, 220,  80, 255);
        case spdlog::level::err:      return IM_COL32(240,  80,  80, 255);
        case spdlog::level::critical: return IM_COL32(255,  60,  60, 255);
        default:                      return IM_COL32(200, 200, 200, 255);
    }
}

/// Convert spdlog level to a short display string.
inline const char *log_level_str(spdlog::level::level_enum lvl) {
    switch (lvl) {
        case spdlog::level::trace:    return "TRC";
        case spdlog::level::debug:    return "DBG";
        case spdlog::level::info:     return "INF";
        case spdlog::level::warn:     return "WRN";
        case spdlog::level::err:      return "ERR";
        case spdlog::level::critical: return "CRT";
        default:                      return "???";
    }
}

inline void LogSink::draw_imgui(const char *title, bool *p_open) {
    // ── Filter state ──────────────────────────────────────────────
    static bool show_trace   = true;
    static bool show_debug   = true;
    static bool show_info    = true;
    static bool show_warn    = true;
    static bool show_error   = true;
    static bool show_critical = true;
    static bool auto_scroll  = true;
    static char filter_buf[128] = "";
    const char *filter_str = filter_buf;

    ImGui::Begin(title, p_open);

    // ── Toolbar ───────────────────────────────────────────────────
    ImGui::Checkbox("Auto-scroll", &auto_scroll); ImGui::SameLine();
    if (ImGui::SmallButton("Clear")) { clear(); } ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    ImGui::InputText("Filter", filter_buf, sizeof(filter_buf));

    // ── Level filter toggles (clickable colour badges) ────────────
    ImGui::Text("Level: ");
    ImGui::SameLine();
    auto level_toggle = [](bool &flag, const char *label, ImU32 col) {
        ImGui::PushStyleColor(ImGuiCol_Text, col);
        if (ImGui::SmallButton(label)) flag = !flag;
        ImGui::PopStyleColor(1);
        if (ImGui::IsItemActive()) flag = !flag; // click toggle
        ImGui::SameLine();
    };
    level_toggle(show_trace,    "TRC", log_level_color(spdlog::level::trace));
    level_toggle(show_debug,    "DBG", log_level_color(spdlog::level::debug));
    level_toggle(show_info,     "INF", log_level_color(spdlog::level::info));
    level_toggle(show_warn,     "WRN", log_level_color(spdlog::level::warn));
    level_toggle(show_error,    "ERR", log_level_color(spdlog::level::err));
    level_toggle(show_critical, "CRT", log_level_color(spdlog::level::critical));
    ImGui::TextDisabled(" (%zu entries, %zu shown)", entries_.size(), entries_.size());

    ImGui::Separator();

    // ── Log body ──────────────────────────────────────────────────
    bool filter_active = filter_str[0] != '\0';
    ImGui::BeginChild("##log_body", ImVec2(0, 0), true);

    size_t shown = 0;
    for (size_t i = 0; i < entries_.size(); i++) {
        const Entry &e = entries_[i];
        // Level filter
        switch (e.level) {
            case spdlog::level::trace:    if (!show_trace)    continue; break;
            case spdlog::level::debug:    if (!show_debug)    continue; break;
            case spdlog::level::info:     if (!show_info)     continue; break;
            case spdlog::level::warn:     if (!show_warn)     continue; break;
            case spdlog::level::err:      if (!show_error)    continue; break;
            case spdlog::level::critical: if (!show_critical) continue; break;
            default: break;
        }
        // Text filter
        if (filter_active) {
            bool match = false;
            // Case-insensitive substring search
            std::string lower_msg = e.message;
            std::string lower_filter = filter_str;
            std::transform(lower_msg.begin(), lower_msg.end(), lower_msg.begin(), ::tolower);
            std::transform(lower_filter.begin(), lower_filter.end(), lower_filter.begin(), ::tolower);
            if (lower_msg.find(lower_filter) != std::string::npos) match = true;
            if (!match) continue;
        }

        shown++;

        // Timestamp
        auto tt = std::chrono::system_clock::to_time_t(e.timestamp);
        struct tm *lt = localtime(&tt);
        char ts[16];
        strftime(ts, sizeof(ts), "%H:%M:%S", lt);

        // Level badge
        ImU32 lcol = log_level_color(e.level);
        ImGui::TextColored(ImVec4(ImColor(lcol)), "%s", log_level_str(e.level));
        ImGui::SameLine();

        // Timestamp
        ImGui::TextDisabled("[%s]", ts);
        ImGui::SameLine();

        // Message
        ImGui::TextUnformatted(e.message.c_str());
    }

    if (auto_scroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 10)
        ImGui::SetScrollHereY(1.0f);

    ImGui::EndChild();
    ImGui::End();
}
