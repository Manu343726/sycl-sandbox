#pragma once

#include "kernel/profiler_host.h"
#include <imgui.h>

// ── Kernel messages log ──────────────────────────────────────────────

static void render_kernel_messages(KernelProfiler& profiler) {
    auto msgs = profiler.messages();
    if (msgs.empty()) {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1), "No kernel messages.");
        return;
    }

    ImGui::BeginChild("##msg_list", ImVec2(0, 0), true);
    for (auto& m : msgs) {
        ImGui::Text("[%lld] %s", (long long)m.frame_index, m.text.c_str());
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
        ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();
}
