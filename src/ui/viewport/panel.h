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
            // Pauses the pipeline, resizes buffers + display target, and
            // re-inits the kernel at the new resolution.
            state.recreate_buffers(vp_w, vp_h);
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

        // ── Camera controls (2D pan / 3D orbit) ─────────────────
        if ( state.kr->kernel() && state.kr->scene_desc() ) {
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
                    float dx = eye[0] - at[0],
                          dy = eye[1] - at[1],
                          dz = eye[2] - at[2];
                    state.orbit.dist   = sqrtf(dx * dx + dy * dy + dz * dz);
                    state.orbit.theta  = atan2f(dx, dz);
                    state.orbit.phi    = asinf(dy / state.orbit.dist);
                    state.orbit.target[0] = at[0];
                    state.orbit.target[1] = at[1];
                    state.orbit.target[2] = at[2];
                    state.orbit.theta = std::fmod(state.orbit.theta, 2.f * 3.14159265f);
                    state.orbit.phi   = std::max(-1.5f, std::min(1.5f, state.orbit.phi));
                    state.orbit_init = true;
                }

                // Mouse controls (only when viewport hovered)
                if ( vp_hovered ) {
                    // LMB = pan target
                    if ( ImGui::IsMouseDragging(ImGuiMouseButton_Left) ) {
                        auto d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                        float ct = cosf(state.orbit.theta), st = sinf(state.orbit.theta);
                        float cp = cosf(state.orbit.phi),  sp = sinf(state.orbit.phi);
                        float right[3] = {ct, 0.f, -st};
                        float up[3]    = {-sp * st, cp, -sp * ct};
                        float speed = state.orbit.dist * 0.002f;
                        state.orbit.target[0] += (-d.x * right[0] + d.y * up[0]) * speed;
                        state.orbit.target[1] += (-d.x * right[1] + d.y * up[1]) * speed;
                        state.orbit.target[2] += (-d.x * right[2] + d.y * up[2]) * speed;
                        ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
                        changed = true;
                    }
                    // MMB = orbit
                    if ( ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ) {
                        auto d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle);
                        state.orbit.theta -= d.x * 0.005f;
                        state.orbit.phi   += d.y * 0.005f;
                        state.orbit.phi = std::max(-1.5f, std::min(1.5f, state.orbit.phi));
                        ImGui::ResetMouseDragDelta(ImGuiMouseButton_Middle);
                        changed = true;
                    }
                    // Scroll: zoom / aperture / FOV / roll
                    float mw = io.MouseWheel;
                    if ( mw != 0.f ) {
                        bool ctrl  = io.KeyCtrl;
                        bool shift = io.KeyShift;
                        bool alt   = io.KeyAlt;
                        if ( ctrl && alt ) {
                            state.orbit.roll += mw * 0.05f;
                            changed = true;
                        } else if ( ctrl && shift && state.fov.valid() ) {
                            state.fov.set(std::max(1.f, std::min(120.f, state.fov.as_float() + mw * 2.f)));
                            changed = true;
                        } else if ( ctrl && state.aperture.valid() ) {
                            state.aperture.set(std::max(0.f, std::min(1.f, state.aperture.as_float() + mw * 0.02f)));
                            changed = true;
                        } else {
                            state.orbit.dist *= (mw > 0.f) ? 0.9f : 1.1f;
                            state.orbit.dist = std::max(1.f, std::min(100.f, state.orbit.dist));
                            changed = true;
                        }
                    }
                }

                // Keyboard controls (when viewport is hovered)
                if ( vp_hovered && !io.WantCaptureKeyboard ) {
                    float kspeed = 0.04f;
                    bool shift = io.KeyShift;

                    if ( !shift && ImGui::IsKeyDown(ImGuiKey_LeftArrow) )  { state.orbit.theta -= kspeed; changed = true; }
                    if ( !shift && ImGui::IsKeyDown(ImGuiKey_RightArrow) ) { state.orbit.theta += kspeed; changed = true; }
                    if ( !shift && ImGui::IsKeyDown(ImGuiKey_UpArrow) )    { state.orbit.phi += kspeed; state.orbit.phi = std::min(1.5f, state.orbit.phi); changed = true; }
                    if ( !shift && ImGui::IsKeyDown(ImGuiKey_DownArrow) )  { state.orbit.phi -= kspeed; state.orbit.phi = std::max(-1.5f, state.orbit.phi); changed = true; }

                    if ( shift && ImGui::IsKeyDown(ImGuiKey_LeftArrow) ) {
                        state.orbit.target[0] -= 0.2f * cosf(state.orbit.theta);
                        state.orbit.target[2] -= 0.2f * sinf(state.orbit.theta);
                        changed = true;
                    }
                    if ( shift && ImGui::IsKeyDown(ImGuiKey_RightArrow) ) {
                        state.orbit.target[0] += 0.2f * cosf(state.orbit.theta);
                        state.orbit.target[2] += 0.2f * sinf(state.orbit.theta);
                        changed = true;
                    }
                    if ( shift && ImGui::IsKeyDown(ImGuiKey_UpArrow) )    { state.orbit.target[1] += 0.2f; changed = true; }
                    if ( shift && ImGui::IsKeyDown(ImGuiKey_DownArrow) )  { state.orbit.target[1] -= 0.2f; changed = true; }

                    // WASD
                    float ct = cosf(state.orbit.theta), st = sinf(state.orbit.theta);
                    float cp = cosf(state.orbit.phi),  sp = sinf(state.orbit.phi);
                    float forward[3] = {cp * st, sp, cp * ct};
                    float right[3]   = {ct, 0.f, -st};
                    float tspeed = 0.3f;

                    if ( ImGui::IsKeyDown(ImGuiKey_W) ) { state.orbit.target[0] += forward[0] * tspeed; state.orbit.target[1] += forward[1] * tspeed; state.orbit.target[2] += forward[2] * tspeed; changed = true; }
                    if ( ImGui::IsKeyDown(ImGuiKey_S) ) { state.orbit.target[0] -= forward[0] * tspeed; state.orbit.target[1] -= forward[1] * tspeed; state.orbit.target[2] -= forward[2] * tspeed; changed = true; }
                    if ( ImGui::IsKeyDown(ImGuiKey_A) ) { state.orbit.target[0] -= right[0] * tspeed;  state.orbit.target[1] -= right[1] * tspeed;  state.orbit.target[2] -= right[2] * tspeed; changed = true; }
                    if ( ImGui::IsKeyDown(ImGuiKey_D) ) { state.orbit.target[0] += right[0] * tspeed;  state.orbit.target[1] += right[1] * tspeed;  state.orbit.target[2] += right[2] * tspeed; changed = true; }
                    if ( ImGui::IsKeyDown(ImGuiKey_Q) ) { state.orbit.target[1] -= 0.3f; changed = true; }
                    if ( ImGui::IsKeyDown(ImGuiKey_E) ) { state.orbit.target[1] += 0.3f; changed = true; }
                }

                if ( changed ) {
                    float eye[3], up[3];
                    orbit_to_eye(state.orbit, eye);
                    orbit_up(state.orbit, up);
                    state.camera_eye.set(eye[0], eye[1], eye[2]);
                    state.camera_at.set(state.orbit.target[0], state.orbit.target[1], state.orbit.target[2]);
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
                auto *desc = state.kr->scene_desc();
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
