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
#include <cstdio>
#if defined(__linux__)
#include <sys/sysinfo.h>
#endif

// ── Helpers ───────────────────────────────────────────────────────────

/// Total memory the device profiler ring is drawn from, in bytes:
/// GPU backend → device global memory, CPU / software → system RAM.
inline uint64_t ring_total_mem_bytes(AppState &state) {
    if ( state.kr && state.kr->queue_ptr() ) {
        const auto &backends = state.kr->backends();
        const int idx = state.kr->active_backend();
        if ( idx >= 0 && idx < (int)backends.size() &&
             backends[idx].id == "gpu_sycl" ) {
            return state.kr->queue().get_device()
                .get_info<sycl::info::device::global_mem_size>();
        }
    }
#if defined(__linux__)
    struct sysinfo si;
    if ( sysinfo(&si) == 0 ) return (uint64_t)si.totalram * si.mem_unit;
#endif
    return 0;
}

/// Update auto-generated standard stats (FPS, SPP, memory, etc.).
inline void update_standard_stats(AppState &state) {
    auto scene_desc = state.kr ? state.kr->scene_desc() : nullptr;
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
    if ( dev_mem.valid() )
        dev_mem.set((float)(krt.pool ? krt.pool->device_bytes() : 0) / (1024.0f * 1024.0f));
    auto host_mem = scene_desc->find_stat_ref("host_memory_mb");
    if ( host_mem.valid() )
        host_mem.set((float)(krt.pool ? krt.pool->host_bytes() : 0) / (1024.0f * 1024.0f));
}

// ── Controls panel ────────────────────────────────────────────────────

/// Render the Controls ImGui window containing scene selection,
/// parameter controls, camera controls, statistics, backend switcher, etc.
inline void render_controls_panel(AppState &state) {
    if (!state.kr) return;
    // Shared-ptr copies: the render thread can swap in a new descriptor
    // (hot reload, scene/backend switch) while this function runs — the
    // copies keep the old one alive, so nothing here ever dereferences a
    // freed descriptor.
    auto scene_desc = state.kr->scene_desc();
    const SceneDef *scene_def  = state.kr->scene();
    auto &krt = state.kr->runtime();

    ImGui::Begin("Controls");

    // ---- Scene selector ----
    std::string cur = scene_def ? scene_def->name : "\u2014";
    if ( ImGui::BeginCombo("Scene", cur.c_str()) ) {
        for ( auto &s : state.scenes->all() ) {
            bool sel = scene_def && scene_def->name == s.name;
            if ( ImGui::Selectable(s.name.c_str(), sel) ) {
                if ( !sel ) {
                    // Deferred scene switch: the render thread applies the
                    // build + reload; the UI re-runs on_scene_changed() for
                    // the new generation and re-raises kernel_ready.
                    state.request_scene_switch(&s);
                }
            }
        }
        ImGui::EndCombo();
    }

    // A scene switch above replaces state.kr's active SceneDescriptor —
    // re-fetch so the rest of this function uses the current one.
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
            spdlog::debug("[param] param changed, coalesced re-init");
            // Kernel-category params may feed init-time precomputation, so
            // the full re-init (rewrite d_params + rebuild the host scene)
            // runs as a coalesced render-thread command — at most one is in
            // flight and rapid drags collapse to the latest values.
            {
                std::lock_guard<std::mutex> lk(state.reinit_mtx);
                state.reinit_params = scene_desc->current_buffer;
                state.reinit_pending = true;
            }
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
                // The accumulator clear is a device op → run it on the
                // render thread.  The loop drains commands at the top of
                // every iteration, so the clear lands before the first
                // frame of the resumed pipeline.
                state.post_cmd([&state] {
                    state.kr->clear_accum();
                    state.current_spp.store(0);
                });
                state.render_paused.store(false);
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
                state.post_cmd([&state] {
                    state.kr->clear_accum();
                    state.current_spp.store(0);
                });
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
            // Preserve live param values across the backend switch:
            // apply_backend_switch() reloads the scene from YAML, which
            // rebuilds current_buffer from defaults and would wipe edits.
            state.backend_restore_params.clear();
            if ( scene_desc ) state.backend_restore_params = scene_desc->current_buffer;

            // Stop the pipeline.  wait_render_idle() parks the render
            // thread out of its frame scope (bounded — it's not waiting
            // on device work, just on the frame-scope flag flipping).
            // The actual device-side teardown (drain the in-flight
            // tone-map, free the staging USM allocated on the OLD
            // queue) happens INSIDE the render-thread closure below —
            // the UI thread never blocks on a SYCL queue wait.  The
            // display target object stays alive until the ack
            // (process_frame_ops step 5) calls destroy() on the UI
            // thread to release its GL/CUDA-GL resources, then
            // recreates it on the new queue.
            state.kernel_ready.store(false);
            state.wait_render_idle();

            state.pending_device_ops.fetch_add(1);
            state.post_cmd([&state, new_backend] {
                // Render-thread: drain the queue + free staging USM
                // (release_device_buffers — allowed to block here)
                // BEFORE apply_backend_switch frees the old queue.
                if (state.display_target)
                    state.display_target->release_device_buffers();
                state.kr->apply_backend_switch(
                    new_backend, state.kr->width(), state.kr->height());
                // The device profiler ring holds allocations on the old
                // queue (freed by the switch) — re-create it against the
                // new queue here, on the render thread.
                state.init_profiler_buffers(&state.kr->runtime(),
                                            state.kr->queue_ptr());
                // Same for the per-lane sample-flags array (freed with
                // the ring above) — re-alloc against the new pool at the
                // current resolution.
                state.ensure_profiler_sample_flags(
                    (uint32_t)(state.kr->width() * state.kr->height()));
                state.backend_switch_applied.store(true);
                state.current_spp.store(0);
                state.tick.store(0);
                state.scene_start_time.store(std::chrono::steady_clock::now());
                state.scene_generation.fetch_add(1);
                state.pending_device_ops.fetch_sub(1);
            });
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
            // Deferred: the CMake reconfigure + rebuild + reload runs as a
            // command on the render thread (the UI stays responsive for the
            // whole rebuild); the UI re-runs on_scene_changed() for the new
            // generation and re-raises kernel_ready.
            state.profiler_enabled = enabled;
            state.kernel_ready.store(false);
            state.pending_device_ops.fetch_add(1);
            state.post_cmd([&state, enabled] {
                if ( state.kr->apply_profiler_toggle(enabled) ) {
                    state.current_spp.store(0);
                    state.target_spp.store(state.kr->scene()
                        ? state.kr->scene()->max_spp : 1);
                    state.tick.store(0);
                    state.scene_start_time.store(std::chrono::steady_clock::now());
                    state.scene_generation.fetch_add(1);
                }
                state.pending_device_ops.fetch_sub(1);
            });
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

        // ---- Ring capacity (runtime, resize-on-the-fly) ----
        // Sized as a percentage of the active backend's total memory
        // (GPU global memory on the GPU backend, system RAM on CPU /
        // software).  The slider is log-scaled because useful ring sizes
        // are tiny fractions of device memory; the capacity snaps to a
        // power of two (the ring masks with idx & (cap-1)).
        const uint32_t cap = state.ring_capacity.load();
        uint64_t total_mem = ring_total_mem_bytes(state);
        // If the memory query returned 0 (transient — e.g. during the
        // very first frame before the sysinfo/SYCL query completes),
        // derive a usable total from the current capacity so the
        // percent slider and resize logic still work.
        if ( total_mem == 0 && cap > 0 )
            total_mem = (uint64_t)cap * 16;   // assume ring ≈ 100%
        const double cur_pct = total_mem > 0
            ? (double)cap * 16.0 / (double)total_mem * 100.0
            : 0.0;

        char total_txt[64];
        if ( total_mem >= (1024ull * 1024ull * 1024ull) )
            snprintf(total_txt, sizeof(total_txt), "%.2f GiB",
                     (double)total_mem / (1024.0 * 1024.0 * 1024.0));
        else if ( total_mem >= (1024ull * 1024ull) )
            snprintf(total_txt, sizeof(total_txt), "%.1f MiB",
                     (double)total_mem / (1024.0 * 1024.0));
        else
            snprintf(total_txt, sizeof(total_txt), "?");
        ImGui::Text("Device ring: %.3f%% of %s", cur_pct, total_txt);

        // Persistent slider state so a live drag doesn't fight the
        // actual ring capacity (updated only after the render thread
        // applies the resize); re-synced from the ring whenever idle.
        static float s_ring_pct = -1.0f;
        static bool s_ring_editing = false;
        if ( !s_ring_editing ) s_ring_pct = (float)cur_pct;
        constexpr float kMinPct = 0.001f, kMaxPct = 100.0f;
        ImGui::SliderFloat("##ring_pct", &s_ring_pct, kMinPct, kMaxPct,
                           "%.3f%%", ImGuiSliderFlags_Logarithmic);
        s_ring_editing = ImGui::IsItemActive();

        // Size for the current slider position (live preview, snapped to
        // a power of two) — shown alongside the percent above.
        uint64_t want_rec = total_mem > 0
            ? (uint64_t)((double)s_ring_pct / 100.0 * (double)total_mem / 16.0)
            : cap;
        if ( want_rec < 2 ) want_rec = 2;
        if ( want_rec > (1ull << 30) ) want_rec = 1ull << 30;
        const uint32_t target = AppState::next_pow2((uint32_t)want_rec);
        ImGui::Text("= %u records (%.2f MiB)", target,
                    target * 16.0 / (1024.0 * 1024.0));

        if ( ImGui::IsItemDeactivatedAfterEdit() && target != cap ) {
            state.request_ring_resize(target);
        }
        if ( ImGui::IsItemHovered() )
            ImGui::SetTooltip(
                "Device-side profiler ring as a percentage of the active "
                "backend's total memory (GPU global memory on the GPU "
                "backend, system RAM on CPU / software).  Records are "
                "16 bytes each; the capacity snaps to the nearest power "
                "of two.  The ring wraps per frame, so if a frame writes "
                "more records than this, the tail is dropped.  Resize "
                "applies at the next frame boundary on the render thread "
                "(no kernel reload).");

        // ---- Overflow warning ----
        const uint32_t last_wp = state.tracy_bridge.last_write_pos();
        const bool ovf = state.tracy_bridge.last_overflow();
        if ( ovf ) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.2f, 1.0f));
            ImGui::TextWrapped("⚠ Ring overflow — dropped records.");
            ImGui::TextWrapped("Last frame wrote %u records (cap %u).  "
                               "Full-frame trace needs ~%u records (%.0f MB).",
                               last_wp, cap,
                               state.next_pow2(last_wp),
                               state.next_pow2(last_wp) * 16.0 / (1024.0 * 1024.0));
            ImGui::PopStyleColor();
        }

        // ---- Interest-zone sampling ----
        // PROFILER_INTEREST_BEGIN regions (e.g. the per-pixel path in the
        // raytracing kernel) record or drop their ENTIRE trace per
        // work-item depending on this percentage: with 0% nothing inside
        // an interest zone is recorded, with 100% everything is.  This
        // trades trace completeness for ring headroom — sampling ~10-30%
        // of the pixels usually keeps a full frame inside the default
        // 256K ring.
        {
            int pct = (int)state.profiler_interest_pct.load();
            ImGui::Text("Interest-zone sampling: %d%%", pct);
            if ( ImGui::SliderInt("Sampled traces", &pct, 0, 100,
                                  "%d%%") &&
                 pct != (int)state.profiler_interest_pct.load() ) {
                // Applied next frame by the render loop (uploaded to the
                // ring header before dispatch) — no device op needed.
                state.profiler_interest_pct.store((uint32_t)pct);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if ( ImGui::IsItemHovered() ) {
                ImGui::SetTooltip(
                    "%% of work-items whose PROFILER_INTEREST_BEGIN region "
                    "(the per-pixel raytrace path) is fully recorded; the "
                    "rest drop every profiler record.  Lower it to keep a "
                    "full frame's traces inside the ring (see warning above) "
                    "without dropping the tail mid-zone.");
            }
        }
    }

    ImGui::End();
}
