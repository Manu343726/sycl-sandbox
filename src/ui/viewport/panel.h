#pragma once

#include "app_state.h"
#include "camera/orbit.h"

#include <imgui.h>
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>

// ── Viewport panel ────────────────────────────────────────────────────

/// Render the main viewport ImGui window.
///
/// Displays the kernel output texture, handles viewport resize,
/// and provides interactive camera controls (2D pan/zoom and 3D orbit).
/// The function writes camera changes directly into the SceneDescriptor
/// buffer and re-initialises the kernel when the camera changes.
inline void render_viewport_panel(AppState &state) {
    if (!state.kr) { ImGui::Begin("Viewport"); ImGui::End(); return; }
    auto &io = ImGui::GetIO();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::Begin("Viewport", nullptr,
                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    ImVec2 region = ImGui::GetContentRegionAvail();
    if ( region.x > 0 && region.y > 0 ) {
        // Recreate render buffers at viewport pixel size if needed
        ImVec2 scale = io.DisplayFramebufferScale;
        int vp_w = std::max(1, (int)(region.x * scale.x));
        int vp_h = std::max(1, (int)(region.y * scale.y));

        if ( vp_w != state.width || vp_h != state.height ) {
            PROFILER_ZONE("Viewport resize");
            spdlog::info("[viewport] resize {}x{} -> {}x{}",
                         state.width, state.height, vp_w, vp_h);
            // Deferred resize: the render thread re-allocates the device
            // accumulation buffer; the UI finishes the GL/display side when
            // it sees the ack.  Requests are coalesced — a drag collapses
            // into a single device-side resize with the final size.
            state.request_resize(vp_w, vp_h);
        }

        // Image fills the whole region (texture matches region size)
        ImVec2 cursor = ImGui::GetCursorScreenPos();
        ImVec2 p0 = cursor;
        ImVec2 p1 = ImVec2(cursor.x + region.x, cursor.y + region.y);

        // Hit-testable area
        ImGui::InvisibleButton("##vp_area", region,
            ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle |
            ImGuiButtonFlags_MouseButtonRight);
        bool vp_hovered = ImGui::IsItemHovered();

        // Display kernel texture.  The GPU tone-map already writes the
        // image Y-flipped for OpenGL's bottom-left origin, so standard
        // (0,0)-(1,1) UVs are correct here — flipping again would show
        // the image upside down.
        ImGui::GetWindowDrawList()->AddImage(
            (ImTextureID)(intptr_t)state.tex,
            p0, p1,
            ImVec2(0, 0), ImVec2(1, 1));

        // ── Loading overlay (render thread not ready) ───────────
        // Startup, scene switches, backend switches, and profiler toggles
        // gate rendering until the render thread has finished the build +
        // reload.  The UI never blocks on that (the first kernel build can
        // take seconds) — draw a scrim + spinner over the stale/empty
        // texture so the app doesn't look frozen.
        if ( !state.kernel_ready.load() ) {
            const char *status = state.render_busy.load()
                ? "Building / loading kernel..." : "Loading scene...";
            auto *dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(p0, p1, IM_COL32(12, 12, 14, 220));

            float cx = (p0.x + p1.x) * 0.5f;
            float cy = (p0.y + p1.y) * 0.5f;
            float r  = 16.f;
            float t  = (float)ImGui::GetTime();
            for ( int i = 0; i < 8; i++ ) {
                float a = t * 3.0f + i * (float)(IM_PI * 2.0 / 8.0);
                ImVec2 c(cx + cosf(a) * r, cy + sinf(a) * r);
                dl->AddCircleFilled(
                    c, 4.f,
                    IM_COL32(255, 255, 255,
                             (int)(100 + 155 * (0.5f + 0.5f * sinf(a)))),
                    8);
            }
            ImVec2 ts = ImGui::CalcTextSize(status);
            dl->AddText(ImVec2(cx - ts.x * 0.5f, cy + r + 14.f),
                        IM_COL32(220, 220, 220, 255), status);
        }

        // ── Camera controls (2D pan / 3D orbit) ─────────────────
        // Gated on kernel_ready: the cached camera ParamRefs point into
        // the active SceneDescriptor's buffer, which the RENDER thread
        // swaps in during reloads / scene / backend switches.  The refs are
        // only valid once the UI has re-run on_scene_changed() for the
        // latest generation (i.e. while kernel_ready is true).
        if ( state.kr->kernel() && state.kr->scene_desc() &&
             state.kernel_ready.load() ) {
            // Refresh camera pointers each frame (in case SceneDescriptor changed)
            state.refresh_camera_ptrs();
            bool has_3d = state.has_3d();
            bool has_2d = state.has_2d();
            bool changed = false;

            // ── 2D camera (center + zoom) ──────────────────────
            if ( has_2d && state.center_x.valid() && state.center_y.valid() && state.zoom.valid() ) {
                if ( vp_hovered && ImGui::IsMouseDragging(ImGuiMouseButton_Left) ) {
                    auto delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                    state.center_x.set(state.center_x.as_float() - delta.x / region.x * state.zoom.as_float() * (region.x / region.y));
                    state.center_y.set(state.center_y.as_float() + delta.y / region.y * state.zoom.as_float());
                    ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
                    changed = true;
                }
                if ( vp_hovered && io.MouseWheel != 0.f ) {
                    float z = state.zoom.as_float() * ((io.MouseWheel > 0.f) ? 0.9f : 1.1f);
                    state.zoom.set(std::max(0.001f, std::min(1000.f, z)));
                    changed = true;
                }
            }

            // ── 3D orbit camera ────────────────────────────────
            if ( has_3d ) {
                if ( !state.orbit_init ) {
                    const float *eye = state.camera_eye.ptr();
                    const float *at = state.camera_at.ptr();
                    orbit_from_eye(state.orbit, eye, at);
                    state.orbit_init = true;
                }

                // Mouse controls (only when viewport hovered)
                if ( vp_hovered ) {
                    float ct = cosf(state.orbit.theta), st = sinf(state.orbit.theta);
                    float cp = cosf(state.orbit.phi),  sp = sinf(state.orbit.phi);
                    float forward[3] = {cp * st, sp, cp * ct};
                    float right[3]   = {ct, 0.f, -st};
                    float up[3]       = {-sp * st, cp, -sp * ct};

                    // MMB drag = rotate camera (yaw/pitch)
                    if ( ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ) {
                        auto d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle);
                        state.orbit.theta -= d.x * 0.005f;
                        state.orbit.phi   -= d.y * 0.005f;
                        state.orbit.phi = std::max(-1.5f, std::min(1.5f, state.orbit.phi));
                        ImGui::ResetMouseDragDelta(ImGuiMouseButton_Middle);
                        changed = true;
                    }
                    // LMB drag = pan camera in camera-local space
                    if ( ImGui::IsMouseDragging(ImGuiMouseButton_Left) ) {
                        auto d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                        float speed = state.orbit.dist * 0.002f;
                        if (io.KeyCtrl) {
                            state.orbit.eye[0] += d.y * forward[0] * speed;
                            state.orbit.eye[1] += d.y * forward[1] * speed;
                            state.orbit.eye[2] += d.y * forward[2] * speed;
                        } else {
                            state.orbit.eye[0] -= d.x * right[0] * speed + d.y * up[0] * speed;
                            state.orbit.eye[1] -= d.x * right[1] * speed + d.y * up[1] * speed;
                            state.orbit.eye[2] -= d.x * right[2] * speed + d.y * up[2] * speed;
                        }
                        ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
                        changed = true;
                    }

                    // Scroll = move along look axis, Ctrl+scroll = roll
                    float mw = io.MouseWheel;
                    if ( mw != 0.f ) {
                        if (io.KeyCtrl) {
                            state.orbit.roll += mw * 0.05f;
                        } else {
                            float speed = state.orbit.dist * 0.2f;
                            state.orbit.eye[0] -= mw * forward[0] * speed;
                            state.orbit.eye[1] -= mw * forward[1] * speed;
                            state.orbit.eye[2] -= mw * forward[2] * speed;
                        }
                        changed = true;
                    }
                }

                // Keyboard controls (when viewport is hovered and focused)
                if ( vp_hovered && ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) ) {
                    float ct = cosf(state.orbit.theta), st = sinf(state.orbit.theta);
                    float cp = cosf(state.orbit.phi),  sp = sinf(state.orbit.phi);
                    float forward[3] = {cp * st, sp, cp * ct};
                    float right[3]   = {ct, 0.f, -st};
                    float up[3]       = {-sp * st, cp, -sp * ct};
                    float tspeed = 0.3f;

                    if ( ImGui::IsKeyDown(ImGuiKey_W) ) { state.orbit.eye[0] += forward[0] * tspeed; state.orbit.eye[1] += forward[1] * tspeed; state.orbit.eye[2] += forward[2] * tspeed; changed = true; }
                    if ( ImGui::IsKeyDown(ImGuiKey_S) ) { state.orbit.eye[0] -= forward[0] * tspeed; state.orbit.eye[1] -= forward[1] * tspeed; state.orbit.eye[2] -= forward[2] * tspeed; changed = true; }
                    if ( ImGui::IsKeyDown(ImGuiKey_A) ) { state.orbit.eye[0] += right[0] * tspeed;  state.orbit.eye[1] += right[1] * tspeed;  state.orbit.eye[2] += right[2] * tspeed; changed = true; }
                    if ( ImGui::IsKeyDown(ImGuiKey_D) ) { state.orbit.eye[0] -= right[0] * tspeed;  state.orbit.eye[1] -= right[1] * tspeed;  state.orbit.eye[2] -= right[2] * tspeed; changed = true; }
                    if ( ImGui::IsKeyDown(ImGuiKey_Q) ) { state.orbit.eye[0] -= up[0] * 0.3f; state.orbit.eye[1] -= up[1] * 0.3f; state.orbit.eye[2] -= up[2] * 0.3f; changed = true; }
                    if ( ImGui::IsKeyDown(ImGuiKey_E) ) { state.orbit.eye[0] += up[0] * 0.3f; state.orbit.eye[1] += up[1] * 0.3f; state.orbit.eye[2] += up[2] * 0.3f; changed = true; }
                }

                if ( changed ) {
                    float lookat[3], up[3];
                    orbit_to_lookat(state.orbit, lookat);
                    orbit_up(state.orbit, up);
                    state.camera_eye.set(state.orbit.eye[0], state.orbit.eye[1], state.orbit.eye[2]);
                    state.camera_at.set(lookat[0], lookat[1], lookat[2]);
                    if ( state.camera_up.valid() ) {
                        state.camera_up.set(up[0], up[1], up[2]);
                    }
                }
            }

            if ( changed ) {
                // Camera edits go through the ParamStore: the render
                // thread picks the snapshot up at its next frame start,
                // clears the accumulator, and restarts sampling — no
                // lock is held while the GPU works, no re-init needed
                // (camera params are read per frame from d_params).
                auto desc = state.kr->scene_desc();
                state.param_store.publish(desc->current_buffer.data(),
                                          desc->buffer_size(),
                                          /*restart_accum=*/true);
                state.current_spp = 0;
            }
        }

        // SPP counter overlay
        {
            char info[64];
            snprintf(info, sizeof(info), "SPP: %d/%d",
                     state.current_spp.load(), state.target_spp.load());
            ImGui::GetWindowDrawList()->AddText(
                ImVec2(p0.x + 8, p0.y + 8),
                IM_COL32(200, 200, 200, 200), info);
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
}
