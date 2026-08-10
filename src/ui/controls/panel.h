#pragma once

#include "app_state.h"
#ifdef SANDBOX_ENABLE_TRACY
#include "tracy/tracy_launcher.h"
#endif
#include "ui/params/controls.h"
#include "ui/scene_debug/panel.h"
#include "ui/stat/panel.h"
#include "ui/build_monitor/panel.h"
#include "io/log_sink.h"

#include <imgui.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>

// ── Helpers ───────────────────────────────────────────────────────────

/// Update auto-generated standard stats (FPS, SPP, memory, etc.).
inline void update_standard_stats(AppState &state) {
    auto *scene_desc = state.kr ? state.kr->scene_desc() : nullptr;
    if ( !scene_desc ) return;
    auto &krt = state.kr->runtime();
    auto fps = scene_desc->find_stat_ref("fps");
    if ( fps.valid() ) {
        double t = state.render_interval_ms.load();
        fps.set(t > 0.0 ? (float)(1000.0 / t) : 0.0f);
    }
    auto ft = scene_desc->find_stat_ref("frame_time_ms");
    if ( ft.valid() ) ft.set((float)state.render_interval_ms.load());
    auto spp = scene_desc->find_stat_ref("spp");
    if ( spp.valid() ) spp.set((int)state.current_spp.load());
    auto px = scene_desc->find_stat_ref("pixel_count");
    if ( px.valid() ) px.set((float)(state.width * state.height) / 1e6f);
    auto dev_mem = scene_desc->find_stat_ref("device_memory_mb");
    if ( dev_mem.valid() ) dev_mem.set((float)krt.device_memory_used / (1024.0f * 1024.0f));
    auto host_mem = scene_desc->find_stat_ref("host_memory_mb");
    if ( host_mem.valid() ) host_mem.set((float)krt.host_memory_used / (1024.0f * 1024.0f));
}

// ── Controls panel ────────────────────────────────────────────────────

/// Render the Controls ImGui window containing scene selection,
/// parameter controls, camera controls, statistics, backend switcher, etc.
inline void render_controls_panel(AppState &state) {
    if (!state.kr) return;
    auto *scene_desc = state.kr->scene_desc();
    auto *scene_def  = state.kr->scene();
    auto &krt = state.kr->runtime();

    ImGui::Begin("Controls");

    // ---- Scene selector ----
    std::string cur = scene_def ? scene_def->name : "\u2014";
    if ( ImGui::BeginCombo("Scene", cur.c_str()) ) {
        for ( auto &s : state.scenes->all() ) {
            bool sel = scene_def && scene_def->name == s.name;
            if ( ImGui::Selectable(s.name.c_str(), sel) ) {
                if ( !sel ) {
                    state.render_paused.store(true);
                    state.pause_pipeline();
                    if (state.kr->switch_scene(s, state.width, state.height)) {
                        state.current_spp = 0;
                        state.target_spp = state.kr->scene()
                            ? state.kr->scene()->max_spp : 1;
                        state.tick.store(0);
                        state.scene_start_time = std::chrono::steady_clock::now();
                        state.orbit_init = false;
                        state.on_scene_changed();
                        state.kernel_ready.store(true);
                        state.render_paused.store(false);
                    } else {
                        state.render_paused.store(false);
                    }
                }
            }
        }
        ImGui::EndCombo();
    }

    // A scene switch above replaces state.kr's active SceneDescriptor
    // (the old one is destroyed) — re-fetch so the rest of this function
    // never dereferences the stale pointer captured before the switch.
    scene_desc = state.kr->scene_desc();
    scene_def  = state.kr->scene();

    // ---- Param controls (procedural from SceneDescriptor) ----
    if ( scene_desc ) {
        bool param_changed = false;
        if ( render_param_controls(*scene_desc,
                                    scene_desc->current_buffer.data(),
                                    false,
                                    scene_loader::ParamCategory::Render) ) {
            param_changed = true;
        }
        if ( render_param_controls(*scene_desc,
                                    scene_desc->current_buffer.data(),
                                    false,
                                    scene_loader::ParamCategory::Camera3D) ) {
            param_changed = true;
        }
        if ( render_param_controls(*scene_desc,
                                    scene_desc->current_buffer.data(),
                                    false,
                                    scene_loader::ParamCategory::Camera2D) ) {
            param_changed = true;
        }
        if ( render_param_controls(*scene_desc,
                                    scene_desc->current_buffer.data(),
                                    false,
                                    scene_loader::ParamCategory::Kernel) ) {
            param_changed = true;
        }
        if ( param_changed ) {
            spdlog::debug("[param] param changed, re-init kernel and reset accum");
            // Kernel-category params may feed init-time precomputation, so
            // run the full re-init — under the pause handshake, because it
            // rewrites d_params and kernel .so globals the render thread reads.
            bool was_ready = state.pause_pipeline();
            state.kr->reinit_kernel();
            state.current_spp = 0;
            state.resume_pipeline(was_ready);
        }
    }

    // ---- Kernel execution play/pause ----
    {
        ImGui::SeparatorText("Kernel Execution");
        bool paused = state.render_paused.load();
        if ( paused ) {
            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.33f, 0.6f, 0.6f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.33f, 0.7f, 0.7f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.33f, 0.8f, 0.8f));
            if ( ImGui::Button("> Play") ) {
                spdlog::info("[render] play \u2014 starting execution loop");
                state.render_paused.store(false);
                state.kr->clear_accum();
                state.current_spp.store(0);
            }
            ImGui::PopStyleColor(3);
            ImGui::SameLine();
            ImGui::TextDisabled("Paused");
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.0f, 0.6f, 0.6f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.0f, 0.7f, 0.7f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.0f, 0.8f, 0.8f));
            if ( ImGui::Button("|| Stop") ) {
                spdlog::info("[render] pause \u2014 halting execution loop");
                state.render_paused.store(true);
                state.current_spp.store(0);
                state.kr->clear_accum();
            }
            ImGui::PopStyleColor(3);
            ImGui::SameLine();
            ImGui::Text("SPP %d/%d", (int)state.current_spp, (int)state.target_spp);
        }
    }

    // ---- Statistics ----
    if ( scene_desc && !scene_desc->stats.empty() ) {
        ImGui::SeparatorText("Statistics");
        render_stat_panel(state.stat_store, *scene_desc);
    }

    // ── Timing diagnostics ─────────────────────────────────────────
    {
        ImGui::SeparatorText("Timing");
        double kt = state.kernel_time_ms.load();
        double rt = state.render_interval_ms.load();
        double spin = state.render_loop_time_ms.load();
        double ut = state.ui_frame_time_ms.load();

        ImGui::Text("Kernel:         %5.1f ms", kt);
        ImGui::Text("Render interval:%5.1f ms  (%.0f FPS)", rt,
                    rt > 0.0 ? 1000.0 / rt : 0.0);
        ImGui::Text("  display wait: %5.1f ms", std::max(0.0, rt - kt));
        ImGui::Text("Loop spin:      %5.1f ms", spin);
        ImGui::Separator();
        ImGui::Text("UI thread:      %5.1f ms", ut);
        ImGui::Text("  poll:         %5.1f ms", state.ui_poll_events_ms.load());
        ImGui::Text("  rebuild:      %5.1f ms", state.ui_rebuild_ms.load());
        ImGui::Text("  stats:        %5.1f ms", state.ui_stats_ms.load());
        ImGui::Text("  ImGui:        %5.1f ms", state.ui_imgui_ms.load());
        ImGui::Text("  upload:       %5.1f ms", state.ui_upload_ms.load());
        ImGui::Text("  composite:    %5.1f ms", state.ui_composite_ms.load());

        static int lag_frame = 0;
        lag_frame++;
        if (lag_frame % 120 == 0) {
            double wait = rt - kt;
            if (wait > 10.0)
                spdlog::warn("[timing] display back-pressure {:.1f} ms -- "
                             "render thread waiting for main thread", wait);
            if (ut > rt && rt > 0.0)
                spdlog::warn("[timing] UI frame ({:.1f} ms) > render interval ({:.1f} ms)",
                             ut, rt);
        }
    }

    // ---- Backend switcher ----
    {
        ImGui::SeparatorText("Backend");
        bool backend_changed = false;
        int new_backend = state.kr->active_backend();
        auto &backends = state.kr->backends();
        std::string cur_label = backends[state.kr->active_backend()].label;
        if ( ImGui::BeginCombo("##backend", cur_label.c_str()) ) {
            for ( int i = 0; i < (int)backends.size(); i++ ) {
                if ( !backends[i].available ) continue;
                ImGui::PushID(i);
                bool sel = i == state.kr->active_backend();
                if ( ImGui::Selectable(backends[i].label.c_str(), sel) ) {
                    if ( !sel ) {
                        new_backend = i;
                        backend_changed = true;
                    }
                }
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }

        // Show current device name
        ImGui::TextUnformatted(state.kr->device_name().c_str());

        if ( backend_changed && new_backend != state.kr->active_backend() ) {
            state.render_paused.store(true);
            state.pause_pipeline();

            // Preserve live param values across the backend switch:
            // switch_backend() reloads the scene from YAML, which rebuilds
            // current_buffer from defaults and would wipe user edits.
            std::vector<float> saved_params;
            if ( scene_desc ) saved_params = scene_desc->current_buffer;

            // Display target + device profiler ring hold allocations on the
            // OLD queue — tear them down while it is still alive, and
            // rebuild against the new one after the switch.
            if (state.display_target) {
                state.display_target->destroy();
                state.display_target.reset();
            }
            state.kr->switch_backend(new_backend, state.width, state.height);
            state.display_target.reset(create_display_target(
                state.kr->queue_ptr(), state.width, state.height));
            state.tex = state.display_target->texture();
            state.init_profiler_buffers(&state.kr->runtime(),
                                        state.kr->queue_ptr());
            state.current_spp = 0;
            state.target_spp = state.kr->scene()
                ? state.kr->scene()->max_spp : 1;
            state.tick.store(0);
            state.scene_start_time = std::chrono::steady_clock::now();
            state.on_scene_changed();

            // Restore the saved params.  Same scene re-parsed, so the
            // layout (param order/offsets) is identical; re-publish so the
            // render thread picks them up before its first frame.  Must
            // re-fetch scene_desc — the switch destroyed the old one.
            scene_desc = state.kr->scene_desc();
            if ( scene_desc && saved_params.size() == scene_desc->current_buffer.size() ) {
                scene_desc->current_buffer = std::move(saved_params);
                state.param_store.publish(scene_desc->current_buffer.data(),
                                          scene_desc->buffer_size(),
                                          /*restart_accum=*/true);
                state.current_spp = 0;
            }

            state.kernel_ready.store(true);
            state.render_paused.store(false);
        }
    }

    // ---- Window toggles ----
    {
        ImGui::SeparatorText("Windows");
        ImGui::Checkbox("Show Build Monitor", &state.show_builds);
        ImGui::Checkbox("Show Logs", &state.show_logs);
#ifdef SANDBOX_ENABLE_TRACY
        // The profiler UI is Tracy's own standalone tracy-profiler
        // executable (built from Tracy's CMake) — this button just
        // launches it; it connects to the in-process client and starts
        // capturing (on-demand recording).
        if (ImGui::Button("Launch Tracy Profiler")) {
            tracy_launcher::launch_profiler(state.build_dir);
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Starts the standalone tracy-profiler app, which connects "
                "to the in-process Tracy client (127.0.0.1:8086). Recording "
                "runs while the profiler is connected.");
#endif
        ImGui::Checkbox("Show System Metrics", &state.show_metrics);
    }

    // ---- Profiler (kernel build compile-time flag) ----
    {
        ImGui::SeparatorText("Profiler");
        bool enabled = state.profiler_enabled;
        if ( ImGui::Checkbox("Enable profiler (kernel build)", &enabled) &&
             enabled != state.profiler_enabled ) {
            // Reconfigure the CMake cache + rebuild the active kernel
            // with the new compile-time flag (mirrors a backend switch:
            // stop the pipeline first, rebuild, then resume).
            state.render_paused.store(true);
            state.pause_pipeline();
            state.kr->set_profiler_enabled(enabled);
            state.profiler_enabled = enabled;
            state.current_spp = 0;
            state.target_spp = state.kr->scene()
                ? state.kr->scene()->max_spp : 1;
            state.tick.store(0);
            state.scene_start_time = std::chrono::steady_clock::now();
            state.on_scene_changed();
            state.kernel_ready.store(true);
            state.render_paused.store(false);
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if ( ImGui::IsItemHovered() ) {
            ImGui::SetTooltip(
                "Toggles the SANDBOX_ENABLE_PROFILER compile-time flag for "
                "on-the-fly kernel builds.  Off makes every PROFILER_* "
                "macro a no-op — the kernel carries zero profiling "
                "overhead.  Changing it reconfigures CMake and rebuilds "
                "the active kernel.");
        }
    }

    ImGui::End();
}
