#pragma once

#include "app_state.h"
#include "kernel/dispatch.h"
#include "render/tonemap.h"

#include <sycl/sycl.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>
#include <algorithm>

#include "imgui_test_engine/imgui_te_engine.h"
extern ImGuiTestEngine* g_test_engine;
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>

// ── Formatting helpers ────────────────────────────────────────────────

/// Convert a byte count to a human-readable string with auto-scaled units
/// (B, KiB, MiB, GiB) and 2‑digit fractional precision.
inline std::string format_bytes(size_t bytes) {
    static constexpr const char *units[] = {"B", "KiB", "MiB", "GiB"};
    double v = static_cast<double>(bytes);
    int u = 0;
    while (v >= 1024.0 && u < 3) { v /= 1024.0; ++u; }
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%.2f %s", v, units[u]);
    return buf;
}

// ── Compositing (main thread) ─────────────────────────────────────────

/// Render ImGui draw data and swap buffers.
/// Also fires test engine swap hooks for screen capture support.
inline void composite_frame(AppState &state) {
    PROFILER_ZONE("Compositing");

    ImGui::Render();

    int dw, dh;
    glfwGetFramebufferSize(state.window, &dw, &dh);
    glViewport(0, 0, dw, dh);
    glClearColor(0.0f, 0.0f, 0.0f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (g_test_engine) ImGuiTestEngine_PreSwap(g_test_engine);
    glfwSwapBuffers(state.window);
    if (g_test_engine) ImGuiTestEngine_PostSwap(g_test_engine);
}

// ── Background render thread ──────────────────────────────────────────

/// Continuous kernel execution loop running on a background thread.
///
/// Per frame:
///   1. apply any pending parameter snapshot from the UI (ParamStore),
///      clearing the accumulator when the change invalidates samples
///   2. animated kernels (target_spp == 1): clear the accumulator every
///      frame so each frame stands alone; progressive kernels: idle once
///      the sample target is reached
///   3. acquire a display slot (may be unavailable — progressive kernels
///      keep accumulating without publishing, animated kernels wait)
///   4. enqueue render kernel + GPU tone-map on the in-order queue,
///      then wait the tone-map's event (bounds work in flight to one
///      frame and makes the host-side counters truthful)
///   5. publish the slot, the stat seqlock, and both profiler rings
inline void render_thread_func(AppState &state) {
    spdlog::debug("[render] thread started");

    bool spp_logged = false;
    bool wait_logged = false;
    int  tick_counter = 0;
    std::vector<uint8_t> param_scratch;

    // Resolve device zone-id hashes for the profiler UI.
    state.kernel_profiler.register_device_zone("trace_px");
    state.kernel_profiler.register_device_zone("tonemap_px");

    while ( state.render_running.load() ) {
        // ── Wait until kernel is fully initialized ───────────────
        if ( state.render_paused.load() || !state.kernel_ready.load() ) {
            if ( !wait_logged ) {
                spdlog::debug(state.render_paused.load()
                                  ? "[render] paused — waiting for resume"
                                  : "[render] kernel not ready — waiting");
                wait_logged = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        wait_logged = false;

        // ── Frame scope: render_busy pairs with pause_pipeline() ─
        state.render_busy.store(true);
        struct BusyClear {
            std::atomic<bool> &flag;
            ~BusyClear() { flag.store(false); }
        } busy_clear{state.render_busy};
        if ( !state.kernel_ready.load() ) continue;  // pause raced in

        // ── Apply pending parameter snapshot ─────────────────────
        {
            bool restart = false;
            if ( state.param_store.fetch(param_scratch, restart) ) {
                auto *desc = state.kr->scene_desc();
                if ( desc && !param_scratch.empty() && state.kr->d_params() &&
                     param_scratch.size() == desc->buffer_size() ) {
                    std::memcpy(state.kr->d_params(), param_scratch.data(),
                                param_scratch.size());
                }
                if ( restart ) {
                    // In-order queue sequences the clear after any frame
                    // still in flight and before the next render kernel.
                    state.kr->clear_accum_async();
                    state.current_spp.store(0);
                }
            }
        }

        // ── Progressive kernels idle at the sample target ────────
        int cur_spp = state.current_spp.load();
        int tgt_spp = state.target_spp.load();
        bool animated = tgt_spp <= 1;   // Shadertoy-style: one frame per tick
        if ( !animated && cur_spp >= tgt_spp ) {
            if ( !spp_logged ) {
                spdlog::debug("[render] SPP {} >= target {} — rendering idle",
                              cur_spp, tgt_spp);
                spp_logged = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        spp_logged = false;

        // ── Acquire a display slot (back-pressure point) ─────────
        int slot = -1;
        uint8_t *staging = nullptr;
        if ( state.display_target ) {
            slot = state.display_target->acquire();
            if ( slot >= 0 ) staging = state.display_target->staging_ptr(slot);
        }
        if ( animated && slot < 0 ) {
            // Nothing would ever see this frame — pace to display rate.
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        // ── Update tick/time in the render-thread-owned d_params ─
        uint64_t new_tick = state.tick.load() + 1;
        state.tick.store(new_tick);
        float secs;
        {
            auto now = std::chrono::steady_clock::now();
            secs = std::chrono::duration<float>(now - state.scene_start_time)
                       .count();
            auto *desc = state.kr->scene_desc();
            if (desc) {
                // find_param_ref() gives a type-safe accessor; the offset()
                // is used to write into d_params, which only this thread touches.
                auto write_param = [&](const char *name, float v) {
                    auto ref = desc->find_param_ref(name);
                    if (!ref.valid()) return;
                    *(float *)((char *)state.kr->d_params() + ref.offset()) = v;
                };
                write_param("tick", (float)new_tick);
                write_param("time", secs);
            }
            if ( ++tick_counter % 60 == 0 ) {
                spdlog::trace("[render] tick={}, time={:.3f}s, SPP={}/{}",
                              new_tick, secs, cur_spp, tgt_spp);
            }
        }

        // ── Samples this frame (kernel honors the spp_frame param) ─
        uint32_t spp_frame = 1;
        if ( auto *desc = state.kr->scene_desc() ) {
            auto ref = desc->find_param_ref("spp_frame");
            if ( ref.valid() ) {
                float v = *(float *)((char *)state.kr->d_params() + ref.offset());
                if ( v >= 1.f ) spp_frame = (uint32_t)v;
            }
        }

        // ── Animated kernels: every frame stands alone ───────────
        if ( animated ) state.kr->clear_accum_async();

        // ── Dispatch render kernel + tone-map (enqueue-only) ─────
        auto &krt = state.kr->runtime();
        krt.profiler_buffer = state.kernel_profiler.buffer();
        state.kernel_profiler.reset_frame();

        sycl::queue *q = state.kr->queue_ptr();
        int32_t w = state.width, h = state.height;
        size_t pixels = (size_t)w * h;
        profiler::DeviceRing ring = state.kernel_profiler.device_ring(pixels);

        uint64_t host_t0 = profiler::DeviceRing::timestamp();

        RenderContext ctx = {
            w, h,
            state.kr->d_params(),
            state.kr->d_accum(),
            spp_frame,
            (uint32_t)cur_spp,       // samples accumulated before this call
            new_tick,                // animation frame counter
            ring,
            state.kr->stat_writer()
        };
        call_render_kernel(state.kr->kernel()->handle, q, ctx);

        sycl::event done{};
        if ( slot >= 0 && staging ) {
            if ( q ) {
                done = tonemap::enqueue(*q, state.kr->d_accum(), staging,
                                        w, h, ring);
            } else {
                // Software backend: render already completed synchronously.
                tonemap::run_cpu(state.kr->d_accum(), staging, w, h);
            }
        }

        // ── Wait for frame completion ────────────────────────────
        // Bounds in-flight work to one frame: keeps current_spp truthful,
        // the queue shallow (UI operations never stall behind a backlog),
        // and gives the device profiler an exact host time bracket.
        if ( q ) {
            try {
                if ( slot >= 0 ) done.wait();
                else q->wait();
            } catch (const std::exception &e) {
                spdlog::error("[render] frame failed: {}", e.what());
            }
        }
        uint64_t host_t1 = profiler::DeviceRing::timestamp();

        // ── Publish frame ────────────────────────────────────────
        if ( slot >= 0 && staging ) {
            FrameInfo info;
            info.frame_index = new_tick;
            info.spp = animated ? spp_frame : cur_spp + spp_frame;
            info.time_sec = secs;
            state.display_target->publish(slot, info, done);
        }

        // ── Publish stats (seqlock — UI never blocks this thread) ─
        const auto &stats = state.kr->stat_buffer();
        if ( !stats.empty() )
            state.published_stats.set_data(stats.data(), stats.size());

        // ── Bookkeeping + profiler collection ────────────────────
        if ( animated ) state.current_spp.store(1);
        else state.current_spp.fetch_add((int)spp_frame);

        PROFILER_PLOT("SPP", (float)state.current_spp.load());
        state.kernel_profiler.collect(state.current_spp.load());
        state.kernel_profiler.collect_device((int64_t)new_tick,
                                             host_t0, host_t1);
    }
    spdlog::debug("[render] thread exiting");
}
