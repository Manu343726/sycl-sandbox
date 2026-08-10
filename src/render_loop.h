#pragma once

#include "app_state.h"
#include "kernel/dispatch.h"
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
#include <pthread.h>

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
    pthread_setname_np(pthread_self(), "sycl-render");
    spdlog::debug("[render] thread started");

    bool spp_logged = false;
    bool wait_logged = false;
    int  tick_counter = 0;
    std::vector<uint8_t> param_scratch;
    auto last_end = std::chrono::steady_clock::now();
    auto last_productive_end = last_end;

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

        // ── Sync cancellation state to device-visible USM ──
        // The main thread sets cancel_requested_ (host-side atomic,
        // always valid).  We mirror it to the USM cancel_flag before
        // dispatch so device code can bail out of long-running kernels.
        // Only the render thread touches the USM pointer — the main
        // thread never accesses USM for cancellation, avoiding crashes
        // when the queue is being torn down (backend switch).
        if (state.kr) state.kr->begin_frame();

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
        bool render_this_frame = true;
        if ( animated && slot < 0 ) {
            // Nothing would ever see this frame — pace to display rate.
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            render_this_frame = false;
        }
        if (render_this_frame) {

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

        sycl::queue *q = state.kr->queue_ptr();
        int32_t w = state.width, h = state.height;
        size_t pixels = (size_t)w * h;
        profiler::DeviceRing ring = state.device_ring(pixels);

#ifdef SANDBOX_ENABLE_PROFILER
        // Host timestamp bracket for the device ring → Tracy bridge.
        uint64_t host_t0 = profiler::DeviceRing::timestamp();
#endif

        // ── Frame context for the kernel (single entry, no ops) ──
        // Params travel as a ParamLookup over the render-thread-owned
        // d_params buffer (tick/time/spp_frame are written there above).
        rt::ParamLookup lookup;
        if ( auto *desc = state.kr->scene_desc() ) {
            lookup.set_buffer(state.kr->d_params());
            lookup.set_entries(desc->lookup_entries.data(),
                               (int)desc->lookup_entries.size());
        }

        rt::Context ctx;
        ctx.runtime = &state.kr->runtime();
        ctx.cancel_flag = state.kr->runtime().cancel_flag;
        ctx.params = &lookup;
        ctx.stats = state.kr->stat_writer();
        ctx.scene = state.kr->host_scene().view.num_handles > 0
                        ? &state.kr->host_scene().view : nullptr;
        ctx.prof = ring;
        ctx.trace_counters = state.kr->trace_counters();
        ctx.width = w;
        ctx.height = h;
        ctx.accum = state.kr->d_accum();
        ctx.spp_frame = spp_frame;
        ctx.spp_total = (uint32_t)cur_spp;   // samples before this call
        ctx.frame_index = new_tick;          // animation frame counter

        // Zero the per-frame trace counters (in-order: sequenced before
        // the render kernel, so debug overrides see a clean 0).
        state.kr->zero_trace_counters_async();

        auto k_t0 = std::chrono::steady_clock::now();

        ctx.output = staging;
        if (!ctx.output) ctx.output = krt.alloc_device<uint8_t>(pixels * 4);
        call_kernel_entry(state.kr->kernel()->handle, ctx);

        auto k_t1 = std::chrono::steady_clock::now();
        state.kernel_time_ms.store(
            std::chrono::duration<double, std::milli>(k_t1 - k_t0).count());
        if (!staging) krt.dealloc(ctx.output);
#ifdef SANDBOX_ENABLE_PROFILER
        uint64_t host_t1 = profiler::DeviceRing::timestamp();
#endif

        // ── Publish frame ────────────────────────────────────────
        if ( slot >= 0 && staging ) {
            FrameInfo info;
            info.frame_index = new_tick;
            info.spp = animated ? spp_frame : cur_spp + spp_frame;
            info.time_sec = secs;
            state.display_target->publish(slot, info, sycl::event{});
        }

        // ── Publish stats (seqlock — UI never blocks this thread) ─
        const auto &stats = state.kr->stat_buffer();
        if ( !stats.empty() )
            state.published_stats.set_data(stats.data(), stats.size());

        // ── Bookkeeping + profiler collection ────────────────────
        if ( animated ) state.current_spp.store(1);
        else state.current_spp.fetch_add((int)spp_frame);

        PROFILER_PLOT("SPP", (float)state.current_spp.load());
#ifdef SANDBOX_ENABLE_PROFILER
        // Host overhead that only exists when profiler support was
        // compiled into the app (SANDBOX_ENABLE_PROFILER=ON): the device
        // ring → Tracy bridge submission.
        state.tracy_bridge.submit_device_ring(state.d_ring_header_,
                                              state.d_ring_records_,
                                              state.RING_CAPACITY,
                                              q, (int64_t)new_tick,
                                              host_t0, host_t1);
        state.tracy_bridge.frame_mark();
#endif
        } // if (render_this_frame)

        auto now = std::chrono::steady_clock::now();
        state.render_loop_time_ms.store(
            std::chrono::duration<double, std::milli>(now - last_end).count());
        if (render_this_frame) {
            state.render_interval_ms.store(
                std::chrono::duration<double, std::milli>(now - last_productive_end).count());
            last_productive_end = now;
        }
        last_end = now;
    }
    spdlog::debug("[render] thread exiting");
}
