#pragma once

#include "app_state.h"
#include "render_loop.h"
#include "ui/controls/panel.h"
#include "ui/viewport/panel.h"
#include "ui/scene_debug/panel.h"
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

// ── Deferred-op coordination + build notifications ───────────────────
// Hot-reload build results are now consumed on the RENDER thread (see
// render_loop.h); this UI-frame step handles the UI side of the deferred
// command protocol: coalesced command kickoff, ack processing, and
// re-raising kernel_ready once the pipeline is quiescent.
void process_frame_ops(AppState &state) {
    // 1. Coalesced re-init kickoff (non-gating: only touches
    //    render-thread-owned device state, never the display).
    if ( state.reinit_pending && !state.reinit_posted.load() ) {
        std::vector<float> params;
        {
            std::lock_guard<std::mutex> lk(state.reinit_mtx);
            state.reinit_pending = false;
            params.swap(state.reinit_params);
        }
        state.reinit_posted.store(true);
        state.post_cmd([&state, params = std::move(params)]() mutable {
            if ( !params.empty() ) state.kr->reinit_kernel(params);
            state.current_spp.store(0);
            state.reinit_posted.store(false);
        });
    }

    // 2. Coalesced resize kickoff (gating: the UI's display-target resize
    //    must not overlap the render thread acquiring slots).  Kicks off
    //    only once the requested size has been stable for a short window
    //    (resize_debounce_ms) — the viewport re-requests every frame its
    //    region differs from the applied size, so without the debounce an
    //    unstable region (a dock-layout fight at startup) would re-post a
    //    resize forever and keep kernel_ready down permanently.
    if ( state.resize_pending && !state.resize_posted.load() ) {
        constexpr std::chrono::milliseconds resize_debounce_ms(150);
        int w, h;
        bool settled = false;
        {
            std::lock_guard<std::mutex> lk(state.resize_mtx);
            settled = std::chrono::steady_clock::now() - state.resize_last_change
                          >= resize_debounce_ms;
            w = state.resize_w;
            h = state.resize_h;
            if ( settled ) state.resize_pending = false;
        }
        if ( settled ) {
            state.resize_posted.store(true);
            state.applied_resize_w.store(w);
            state.applied_resize_h.store(h);
            state.kernel_ready.store(false);
            state.pending_device_ops.fetch_add(1);
            // Pure resolution change: apply_resize() reallocates the
            // accumulation buffer and updates the render-thread-owned
            // width/height.  The kernel receives the new resolution via
            // ctx.width/ctx.height on the next dispatch (render loop), so
            // no scene reload or param re-init is needed — reloading would
            // redundantly rebuild the YAML scene on every window drag.
            state.post_cmd([&state, w, h] {
                state.kr->apply_resize(w, h);
                state.resize_applied.store(true);
                state.resize_posted.store(false);
                state.pending_device_ops.fetch_sub(1);
            });
        }
    }

    // 3. Process the latest scene generation (scene switch / backend
    //    switch / hot reload / profiler toggle completion).
    uint64_t gen = state.scene_generation.load();
    if ( gen != state.last_processed_generation ) {
        state.last_processed_generation = gen;
        state.on_scene_changed();
    }

    // 4. Resize ack → finish the GL/display side of a resize.
    if ( state.resize_applied.exchange(false) ) {
        int w = state.applied_resize_w.load();
        int h = state.applied_resize_h.load();
        if ( state.display_target ) state.display_target->resize(w, h);
        state.tex = state.display_target ? state.display_target->texture() : 0;
        state.width = w;
        state.height = h;
        state.current_spp.store(0);
        spdlog::info("[viewport] resize applied {}x{}", w, h);
    }

    // 5. Backend-switch ack → recreate the GL display target on the new
    //    queue (its old allocations died with the old queue) and restore
    //    the live param values the switch reload wiped.
    if ( state.backend_switch_applied.exchange(false) ) {
        state.display_target.reset(create_display_target(
            state.kr->queue_ptr(), state.kr->width(), state.kr->height()));
        state.tex = state.display_target->texture();
        auto desc = state.kr->scene_desc();
        if ( desc && !state.backend_restore_params.empty() &&
             state.backend_restore_params.size() == desc->current_buffer.size() ) {
            desc->current_buffer = state.backend_restore_params;
            state.param_store.publish(desc->current_buffer.data(),
                                      desc->buffer_size(),
                                      /*restart_accum=*/true);
        }
        state.backend_restore_params.clear();
        state.current_spp.store(0);
        spdlog::info("[backend] display target re-created: {}",
                     state.display_target ? state.display_target->name() : "?");
    }

    // 6. Re-raise kernel_ready once every gating op has been applied AND
    //    acknowledged.  The re-checks (render_busy / generation) make sure
    //    we never raise while the render thread is mid-reload or a newer
    //    generation is still unprocessed.
    if ( !state.kernel_ready.load() && !state.render_busy.load() &&
         state.pending_device_ops.load() == 0 &&
         !state.backend_switch_applied.load() &&
         !state.resize_applied.load() &&
         state.scene_generation.load() == state.last_processed_generation ) {
        state.kernel_ready.store(true);
    }

    // Build notifications (rendered by the BuildMonitor panel) are
    // consumed here on the UI thread — poll_results() already drained the
    // build results on the render thread.
    state.build_monitor.ingest(state.kr->builder().poll_notifications());
}

// ── Update frame statistics ──────────────────────────────────────────
void update_frame_stats(AppState &state) {
    auto scene_desc = state.kr->scene_desc();
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

    // Scene debug view — the window (ImGui + presented texture) is
    // managed by the panel; the 3D rendering happens on the background
    // debug thread from a host copy of the scene (read-only).
    //
    // The WINDOW is created on every frame (it may be docked, and a
    // docked window that pops in and out re-flows the dock layout — which
    // shifts the viewport region and can re-trigger a resize → feedback
    // loop while the pipeline is gated).
    //
    // The CONTENT is gated on kernel_ready: the debug camera uses the
    // cached camera ParamRefs, which point into the active
    // SceneDescriptor's buffer.  The descriptor is swapped in by the
    // RENDER thread (hot reload, scene/backend switch), so those refs are
    // only valid once the UI has re-run on_scene_changed() for the latest
    // scene generation.
    {
        static DebugViewFlags debug_flags;
        if (state.kr && state.kernel_ready.load()) {
            // Render parameters mirrored from the kernel: the debug ray
            // is traced with the same background / max_bounces /
            // transparent_backfaces the kernel uses, and overlay
            // colours are displayed with the same tone-map + gamma as
            // the framebuffer.
            SceneRenderParams render;
            if (auto desc = state.kr->scene_desc()) {
                auto ref = desc->find_param_ref("background");
                if (ref.valid()) ref.as_vec3(render.background);
                ref = desc->find_param_ref("max_bounces");
                if (ref.valid()) render.max_bounces = ref.as_int();
                ref = desc->find_param_ref("transparent_backfaces");
                if (ref.valid())
                    render.transparent_backfaces = ref.as_bool();
                ref = desc->find_param_ref("tonemap_enabled");
                if (ref.valid()) render.tonemap_enabled = ref.as_bool();
                ref = desc->find_param_ref("tonemap_operator");
                if (ref.valid()) render.tonemap_operator = ref.as_int();
                ref = desc->find_param_ref("tonemap_exposure");
                if (ref.valid()) render.tonemap_exposure = ref.as_float();
                ref = desc->find_param_ref("tonemap_gamma");
                if (ref.valid()) render.tonemap_gamma = ref.as_float();
            }
            render_scene_debug(&state.kr->host_scene(),
                               state.camera_eye.ptr(),
                               state.camera_at.ptr(),
                               state.camera_up.ptr(),
                               state.fov.valid() ? state.fov.as_float() : 45.f,
                               (float)state.width / (float)state.height,
                               state.width,
                               state.height,
                               render,
                               debug_flags);

            if (debug_apply_scene_camera(
                    const_cast<float *>(state.camera_eye.ptr()),
                    const_cast<float *>(state.camera_at.ptr()),
                    const_cast<float *>(state.camera_up.ptr()))) {
                auto desc = state.kr->scene_desc();
                if (desc) {
                    state.param_store.publish(desc->current_buffer.data(),
                                              desc->buffer_size(),
                                              /*restart_accum=*/true);
                    state.current_spp = 0;
                }
            }
        } else if (state.kr) {
            // Kernel exists but the pipeline is gated (startup, scene /
            // backend switch, reload): keep the docked Scene Debug window
            // alive with a placeholder so the dock layout stays put.
            ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
            if ( ImGui::Begin("Scene Debug") ) {
                ImGui::TextUnformatted("Loading scene...");
            }
            ImGui::End();
        }
    }

    // Log viewer
    if (state.show_logs) {
        state.log_sink->draw_imgui("Logs", &state.show_logs);
    }

    // Build monitor
    if (state.show_builds) {
        ImGui::SetNextWindowSize(ImVec2(550, 400), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Build Monitor", &state.show_builds)) {
            render_build_monitor_panel(state.build_monitor);
        }
        ImGui::End();
    }

    // System metrics (CPU / RAM / GPU usage graphs)
    if (state.show_metrics) {
        ImGui::SetNextWindowSize(ImVec2(420, 300), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("System Metrics", &state.show_metrics)) {
            state.system_metrics.sample();
            state.system_metrics.draw();
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
        auto ui_t0 = std::chrono::steady_clock::now();
        poll_events(state);
        auto t1 = std::chrono::steady_clock::now();
        process_frame_ops(state);
        auto t2 = std::chrono::steady_clock::now();
        update_frame_stats(state);
        auto t3 = std::chrono::steady_clock::now();
        render_ui(state);
        auto t4 = std::chrono::steady_clock::now();
        upload_display(state);
        auto t5 = std::chrono::steady_clock::now();
        composite_frame(state);
        auto t6 = std::chrono::steady_clock::now();

        state.ui_frame_time_ms.store(
            std::chrono::duration<double, std::milli>(t6 - ui_t0).count());
        state.ui_poll_events_ms.store(
            std::chrono::duration<double, std::milli>(t1 - ui_t0).count());
        state.ui_rebuild_ms.store(
            std::chrono::duration<double, std::milli>(t2 - t1).count());
        state.ui_stats_ms.store(
            std::chrono::duration<double, std::milli>(t3 - t2).count());
        state.ui_imgui_ms.store(
            std::chrono::duration<double, std::milli>(t4 - t3).count());
        state.ui_upload_ms.store(
            std::chrono::duration<double, std::milli>(t5 - t4).count());
        state.ui_composite_ms.store(
            std::chrono::duration<double, std::milli>(t6 - t5).count());
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
    state.free_profiler_buffers();

    // KernelRuntime destructor handles all kernel/device resource cleanup
    state.kr.reset();

    // Shutdown test engine (must be before ImGui::DestroyContext)
    if (g_test_engine) {
        ImGuiTestEngine_Stop(g_test_engine);
        ImGuiTestEngine_DestroyContext(g_test_engine);
        g_test_engine = nullptr;
    }

    // Tracy profiler: stop the bridge (render thread already joined).
    // The standalone tracy-profiler process (if running) is independent
    // of the sandbox and keeps running.
    state.tracy_bridge.shutdown();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(state.window);
    glfwTerminate();
}
