#pragma once

#include "app_state.h"
#include "render_loop.h"
#include "scene/host_scene.h"
#include "ui/controls/panel.h"
#include "ui/viewport/panel.h"
#include "ui/scene_debug/panel.h"
#include "ui/profiler/panel.h"
#include "ui/build_monitor/panel.h"
#include "io/log_sink.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_test_engine/imgui_te_engine.h"
#include "imgui_test_engine/imgui_te_ui.h"
#include <GLFW/glfw3.h>

// ── Test engine global (set from main.cpp) ────────────────────────────
extern ImGuiTestEngine* g_test_engine;

// ── Internal frame-step helpers ───────────────────────────────────────
// These break the frame loop into single-responsibility steps, each
// delegating to the appropriate subsystem.

namespace {

// ── Poll OS events ───────────────────────────────────────────────────
void poll_events(AppState &state) {
    PROFILER_ZONE("Frame begin");
    glfwPollEvents();
}

// ── Source-watcher → background build → hot-reload ──────────────────
void handle_kernel_rebuild(AppState &state) {
    // The pre-reload hook runs only when a finished build is about to be
    // swapped in: it stops the render thread and drains the queue so the
    // reload never frees memory a frame in flight still references.
    std::string reloaded = state.kr->poll_hot_reload(
        [&state] { state.pause_pipeline(); });
    if (!reloaded.empty()) {
        state.kernel_profiler.clear_data();
        state.current_spp = 0;
        state.tick.store(0);
        state.scene_start_time = std::chrono::steady_clock::now();
        state.on_scene_changed();
        state.kernel_ready.store(true);
        state.render_paused.store(false);
    }
    state.build_monitor.ingest(state.kr->builder().poll_notifications());
}

// ── Update frame statistics ──────────────────────────────────────────
void update_frame_stats(AppState &state) {
    auto *scene_desc = state.kr->scene_desc();
    if (!scene_desc) return;
    // 1. Pull the render thread's latest kernel stats through the seqlock
    //    into the UI-owned stat buffer (never touches render-thread memory).
    if (state.published_stats.size() == scene_desc->current_stat_buffer.size())
        state.published_stats.try_read(scene_desc->current_stat_buffer.data());
    // 2. Standard stats (fps, frame time, ...) are computed here on the
    //    UI thread and overwrite their own fields on top.
    update_standard_stats(state);
    // 3. Feed the merged block into the history graphs.
    state.stat_store.ingest(*scene_desc);
}

// ── Render the full ImGui frame (dockspace + all panels) ────────────
void render_ui(AppState &state) {
    PROFILER_ZONE("ImGui frame");
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Full-window dockspace (always-on background)
    {
        ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
        ImGui::Begin("##dockspace", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
            ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMouseInputs);
        ImGui::PopStyleVar(3);
        ImGui::DockSpace(ImGui::GetID("dockspace"),
                         ImVec2(0.f, 0.f),
                         ImGuiDockNodeFlags_PassthruCentralNode);
        ImGui::End();
    }

    // Controls panel
    render_controls_panel(state);

    // Viewport panel
    render_viewport_panel(state);

    // Scene debug view
    {
        static DebugViewFlags debug_flags;
        if (state.kr) {
            const SceneDebugInfo *dbg = state.kr->scene_debug_fn()
                ? state.kr->scene_debug_fn()() : nullptr;
            ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
            if (ImGui::Begin("Scene Debug")) {
                render_scene_debug(dbg,
                                   state.camera_eye.ptr(),
                                   state.camera_at.ptr(),
                                   state.camera_up.ptr(),
                                   state.fov.valid() ? state.fov.as_float() : 45.f,
                                   (float)state.width / (float)state.height,
                                   debug_flags);
            }
            ImGui::End();
        }
    }

    // Log viewer
    if (state.show_logs) {
        state.log_sink->draw_imgui("Logs", &state.show_logs);
    }

    // Kernel profiler
    if (state.show_profiler) {
        ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Kernel Profiler", &state.show_profiler)) {
            render_profiler_panel(state.kernel_profiler);
        }
        ImGui::End();
    }

    // Build monitor
    if (state.show_builds) {
        ImGui::SetNextWindowSize(ImVec2(550, 400), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Build Monitor", &state.show_builds)) {
            render_build_monitor_panel(state.build_monitor);
        }
        ImGui::End();
    }

    // Test Engine UI (Dear ImGui automation/testing panel)
    if (state.show_test_engine && g_test_engine) {
        ImGuiTestEngine_ShowTestEngineWindows(g_test_engine, &state.show_test_engine);
    }
}

// ── Present the newest finished frame ────────────────────────────────
void upload_display(AppState &state) {
    if (!state.display_target) return;
    int slot = -1;
    FrameInfo info{};
    if (!state.display_target->latest_ready(slot, info)) return;

    state.tex = state.display_target->present(slot);
    state.last_frame_info = info;
    spdlog::trace("[frame] displayed frame {}, spp={}",
                  info.frame_index, info.spp);
}

} // anonymous namespace

// ── Main application frame loop ───────────────────────────────────────
// Runs the entire main loop until the window is closed.
inline void frame_loop(AppState &state) {
    while (!glfwWindowShouldClose(state.window)) {
        poll_events(state);
        handle_kernel_rebuild(state);
        update_frame_stats(state);
        render_ui(state);
        upload_display(state);
        composite_frame(state);
    }
}

// ── Shutdown sequence ─────────────────────────────────────────────────
// Tears down the render thread, scene, device allocations, and ImGui.
inline void shutdown_sandbox(AppState &state) {
    spdlog::debug("[render] shutting down render thread");
    state.render_running.store(false);
    if (state.render_thread.joinable()) {
        state.render_thread.join();
        spdlog::info("[render] thread joined \u2014 shutdown complete");
    }

    shutdown_scene_debug();

    // The display target and the profiler's device ring hold allocations
    // on the KernelRuntime's queue — release them while it is still alive.
    if (state.display_target) {
        state.display_target->destroy();
        state.display_target.reset();
    }
    state.kernel_profiler.free_device_ring();

    // KernelRuntime destructor handles all kernel/device resource cleanup
    state.kr.reset();

    // Shutdown test engine (must be before ImGui::DestroyContext)
    if (g_test_engine) {
        ImGuiTestEngine_Stop(g_test_engine);
        ImGuiTestEngine_DestroyContext(g_test_engine);
        g_test_engine = nullptr;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(state.window);
    glfwTerminate();
}
