// sycl-sandbox — Scene Debug window (ImGui side)
//
// The window itself is plain ImGui: toolbar checkboxes, a stats line,
// orbit-camera mouse handling, and the 3D view drawn as a texture via
// ImGui::Image.  All OpenGL rendering happens on the background render
// thread (SceneDebugRenderer); this panel only feeds it shared state
// (scene snapshot, flags, camera input, render resolution) and presents
// the newest ready slot texture.

#include "panel.h"

#include "imgui.h"

#include <sycl-sandbox/rt/math.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <spdlog/spdlog.h>

namespace {

SceneDebugRenderer g_renderer;

/// Label of one bounce step in the hits tree.
static std::string step_label(const rt::RayTraceResult &trace, size_t i) {
    const rt::RayTraceStep &s = trace.steps[i];
    const char *hn = hittable_name(static_cast<rt::HittableType>(
        rt::handle_tag(s.handle.hittable)));
    const char *mn = material_name(static_cast<rt::MaterialType>(
        rt::handle_tag(s.handle.material)));
    char buf[160];
    if ( s.escaped ) {
        std::snprintf(buf, sizeof(buf), "bounce %zu — escaped (sky)", i);
    } else if ( s.absorbed ) {
        std::snprintf(buf, sizeof(buf),
                      "bounce %zu — absorbed by %s#%u (%s)", i, hn,
                      rt::handle_index(s.handle.hittable), mn);
    } else {
        std::snprintf(buf, sizeof(buf),
                      "bounce %zu — %s#%u (%s) t=%.3f", i, hn,
                      rt::handle_index(s.handle.hittable), mn, s.record.t);
    }
    return buf;
}

/// Square colour indicator + label + numeric values (HDR radiance can
/// exceed [0,1], so the swatch is clamped while the values stay raw).
static void draw_color_row(const char *label, rt::float3 c, float size = 18.f) {
    ImGui::ColorButton(label,
                       ImVec4(std::clamp(c.x, 0.f, 1.f),
                              std::clamp(c.y, 0.f, 1.f),
                              std::clamp(c.z, 0.f, 1.f), 1.f),
                       ImGuiColorEditFlags_NoTooltip |
                           ImGuiColorEditFlags_NoPicker,
                       ImVec2(size, size));
    ImGui::SameLine();
    ImGui::Text("%s = (%.3f, %.3f, %.3f)", label, c.x, c.y, c.z);
}

/// Details of one recorded hit: full HitRecord + scatter result.
static void draw_step_details(const rt::RayTraceResult &trace, int idx,
                              const SceneRenderParams &render) {
    const rt::RayTraceStep &s = trace.steps[idx];
    char buf[320];

    std::snprintf(buf, sizeof(buf), "bounce %d of %zu", idx,
                  trace.steps.size());
    ImGui::TextUnformatted(buf);
    ImGui::Separator();

    if ( s.escaped ) {
        ImGui::TextColored(ImVec4(0.65f, 0.8f, 1.f, 1.f),
                           "state: escaped (sky)");
    } else if ( s.absorbed ) {
        ImGui::TextColored(ImVec4(1.f, 0.65f, 0.45f, 1.f),
                           "state: absorbed");
    } else {
        ImGui::TextColored(ImVec4(0.55f, 1.f, 0.65f, 1.f),
                           "state: hit, scattered");
    }

    // Back-propagated output colour of this hit: the radiance the path
    // sends back along this step's ray (emit + attenuation × the folded
    // continuation).
    draw_color_row("output color", s.color);
    // How the framebuffer actually shows this radiance (exposure →
    // tone-map operator → gamma) — matches the 3D overlay colour.
    draw_color_row("displayed", scene_debug_display_color(s.color, render));

    if ( s.hit ) {
        std::snprintf(buf, sizeof(buf), "handle: %s#%u · %s material",
                      hittable_name(static_cast<rt::HittableType>(
                          rt::handle_tag(s.handle.hittable))),
                      rt::handle_index(s.handle.hittable),
                      material_name(static_cast<rt::MaterialType>(
                          rt::handle_tag(s.handle.material))));
        ImGui::TextUnformatted(buf);
        std::snprintf(buf, sizeof(buf), "t = %.4f   u = %.4f   v = %.4f",
                      s.record.t, s.record.u, s.record.v);
        ImGui::TextUnformatted(buf);
        ImGui::TextUnformatted(s.record.front_face ? "front face: yes"
                                                   : "front face: no");
        rt::float3 p =
            rt::add(s.ray.orig, rt::scale(s.ray.dir, s.record.t));
        std::snprintf(buf, sizeof(buf), "point  = (%.3f, %.3f, %.3f)", p.x,
                      p.y, p.z);
        ImGui::TextUnformatted(buf);
        std::snprintf(buf, sizeof(buf), "normal = (%.3f, %.3f, %.3f)",
                      s.record.normal.x, s.record.normal.y, s.record.normal.z);
        ImGui::TextUnformatted(buf);
        draw_color_row("emit", s.emit);
        if ( s.record.is_portal ) {
            std::snprintf(buf, sizeof(buf),
                          "portal origin = (%.3f, %.3f, %.3f)",
                          s.record.portal_origin.x, s.record.portal_origin.y,
                          s.record.portal_origin.z);
            ImGui::TextUnformatted(buf);
            std::snprintf(buf, sizeof(buf),
                          "portal dir    = (%.3f, %.3f, %.3f)",
                          s.record.portal_dir.x, s.record.portal_dir.y,
                          s.record.portal_dir.z);
            ImGui::TextUnformatted(buf);
        }
        ImGui::Separator();
        if ( s.scattered ) {
            draw_color_row("attenuation", s.attenuation);
            std::snprintf(buf, sizeof(buf),
                          "scattered ray: orig (%.3f, %.3f, %.3f) dir "
                          "(%.3f, %.3f, %.3f)",
                          s.scattered_ray.orig.x, s.scattered_ray.orig.y,
                          s.scattered_ray.orig.z, s.scattered_ray.dir.x,
                          s.scattered_ray.dir.y, s.scattered_ray.dir.z);
            ImGui::TextUnformatted(buf);
        } else {
            ImGui::TextDisabled("no scatter");
        }
        ImGui::Separator();
    }
    std::snprintf(buf, sizeof(buf), "BVH nodes entered: %zu",
                  s.bvh_visited.size());
    ImGui::TextUnformatted(buf);
    if ( !s.bvh_visited.empty() ) {
        std::string nodes;
        for ( size_t i = 0; i < s.bvh_visited.size(); i++ ) {
            if ( i ) nodes += ", ";
            nodes += std::to_string(s.bvh_visited[i]);
        }
        ImGui::TextWrapped("%s", nodes.c_str());
    }
    std::snprintf(buf, sizeof(buf), "mesh BVH nodes entered: %zu",
                  s.hittable_bvh_visited.size());
    ImGui::TextUnformatted(buf);
    if ( !s.hittable_bvh_visited.empty() ) {
        std::string nodes;
        for ( size_t i = 0; i < s.hittable_bvh_visited.size(); i++ ) {
            if ( i ) nodes += ", ";
            nodes += std::to_string(s.hittable_bvh_visited[i]);
        }
        ImGui::TextWrapped("%s", nodes.c_str());
    }
}

/// Selection state shared between the hits tree and the details
/// dialog: clicking a node selects the bounce and opens (or updates)
/// the "Hit Details" dialog; closing that dialog keeps the selection
/// so the next click reopens it for the same hit.
static int sel_step = -1;
static bool details_open = false;

/// Hits treeview (one node per bounce).  Clicking a node opens the
/// hit-details dialog for that bounce.  Renders from a snapshot of
/// the latest trace; the selection index survives between frames.
static void draw_trace_inspector(const rt::RayTraceResult &trace,
                                 const SceneRenderParams &render) {    // Overall statistics of the trace
    size_t hits = 0, nodes = 0, mesh_nodes = 0;
    bool escaped = false, absorbed = false;
    for ( const auto &s : trace.steps ) {
        hits += s.hit ? 1 : 0;
        nodes += s.bvh_visited.size();
        mesh_nodes += s.hittable_bvh_visited.size();
        escaped |= s.escaped;
        absorbed |= s.absorbed;
    }
    char buf[256];
    // Step 0's back-propagated colour is the final pixel colour of the
    // traced ray.
    const rt::float3 &final = trace.steps.empty()
                                  ? rt::float3{0.f, 0.f, 0.f}
                                  : trace.steps[0].color;
    std::snprintf(buf, sizeof(buf),
                  "bounces: %zu  ·  hits: %zu  ·  bvh nodes entered: %zu"
                  "  ·  mesh bvh nodes: %zu  ·  end: %s%s  ·  color: "
                  "(%.3f, %.3f, %.3f)",
                  trace.steps.size(), hits, nodes, mesh_nodes,
                  escaped ? "escaped (sky)" : (absorbed ? "absorbed" : "hit"),
                  trace.bounce_limit ? "  ·  (bounce limit)" : "",
                  final.x, final.y, final.z);
    ImGui::TextDisabled("%s", buf);
    ImGui::Separator();

    if ( sel_step >= (int)trace.steps.size() ) sel_step = -1;

    ImGui::BeginChild("##hit_tree", ImVec2(0, 0), ImGuiChildFlags_Borders);
    for ( size_t i = 0; i < trace.steps.size(); i++ ) {
        const rt::RayTraceStep &s = trace.steps[i];
        ImGuiTreeNodeFlags fl = ImGuiTreeNodeFlags_OpenOnArrow |
                                ImGuiTreeNodeFlags_SpanAvailWidth;
        if ( (int)i == sel_step ) fl |= ImGuiTreeNodeFlags_Selected;
        std::string label = step_label(trace, i);
        bool open = ImGui::TreeNodeEx((void *)(intptr_t)(i + 1), fl, "%s",
                                      label.c_str());
        if ( ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen() ) {
            sel_step = (int)i;
            details_open = true;  // (re)open the details dialog
        }
        if ( open ) {
            if ( s.hit ) {
                ImGui::TextDisabled("handle: %s#%u (%s)",
                                    hittable_name(static_cast<rt::HittableType>(
                                        rt::handle_tag(s.handle.hittable))),
                                    rt::handle_index(s.handle.hittable),
                                    material_name(static_cast<rt::MaterialType>(
                                        rt::handle_tag(s.handle.material))));
                ImGui::TextDisabled("t = %.4f · u = %.4f · v = %.4f",
                                    s.record.t, s.record.u, s.record.v);
                draw_color_row("color", s.color, 14.f);
                draw_color_row("displayed",
                               scene_debug_display_color(s.color, render),
                               14.f);
                draw_color_row("emit", s.emit, 14.f);
                if ( s.scattered )
                    draw_color_row("attenuation", s.attenuation, 14.f);
            }
            // BVH traversal of this bounce: the scene-BVH nodes the ray
            // entered while finding the closest hit, then the per-mesh
            // (hittable) BVH nodes entered inside Mesh primitives.  The
            // indices index into SceneView::bvh_nodes / mesh_bvh_nodes.
            ImGui::PushID((int)i);
            std::string nodes;
            std::snprintf(buf, sizeof(buf), "scene bvh nodes (%zu)",
                          s.bvh_visited.size());
            if ( ImGui::TreeNode(buf) ) {
                nodes.clear();
                for ( size_t k = 0; k < s.bvh_visited.size(); k++ ) {
                    if ( k ) nodes += ", ";
                    nodes += std::to_string(s.bvh_visited[k]);
                }
                ImGui::TextWrapped("%s", nodes.empty() ? "(none)" : nodes.c_str());
                ImGui::TreePop();
            }
            std::snprintf(buf, sizeof(buf), "mesh bvh nodes (%zu)",
                          s.hittable_bvh_visited.size());
            if ( ImGui::TreeNode(buf) ) {
                nodes.clear();
                for ( size_t k = 0; k < s.hittable_bvh_visited.size(); k++ ) {
                    if ( k ) nodes += ", ";
                    nodes += std::to_string(s.hittable_bvh_visited[k]);
                }
                ImGui::TextWrapped("%s", nodes.empty() ? "(none)" : nodes.c_str());
                ImGui::TreePop();
            }
            ImGui::PopID();
            ImGui::TreePop();
        }
    }
    ImGui::EndChild();
}

/// "Hit Details" dialog: full details of the selected hit (t/u/v,
/// front face, point, normal, emit, portal, attenuation, scattered
/// ray, BVH nodes), opened by clicking a node in the Ray Inspector
/// tree.  A regular dockable window with a title-bar close button;
/// selecting another hit updates it in place.
static void draw_hit_details_window(const rt::RayTraceResult &trace,
                                    const SceneRenderParams &render) {
    if ( !details_open ) return;
    ImGui::SetNextWindowSize(ImVec2(360, 420), ImGuiCond_FirstUseEver);
    if ( !ImGui::Begin("Hit Details", &details_open) ) {
        ImGui::End();
        return;
    }
    if ( sel_step >= 0 && sel_step < (int)trace.steps.size() ) {
        draw_step_details(trace, sel_step, render);
    } else {
        ImGui::TextDisabled("no hit selected");
    }
    ImGui::End();
}

/// Dockable "Ray Inspector" window: the trace summary + hits treeview
/// (all raytracing details — the Scene Debug window keeps only its
/// toolbar and 3D view, so its size never depends on the trace data).
/// As a regular ImGui window inside the dockspace it can be docked
/// anywhere.  The hit-details dialog is an independent window drawn
/// alongside it (Tracy-style: click a hit to open it).
static void draw_ray_inspector_window(SceneDebugRenderer::State &st) {
    rt::RayTraceResult trace;
    std::string summary;
    SceneRenderParams render;
    {
        std::lock_guard<std::mutex> lk(st.mutex);
        trace = st.ray_trace;
        summary = st.ray_trace_text;
        render = st.render;
    }

    // Independent dialog — stays open (showing the last trace) even if
    // the tree window is collapsed or closed.
    draw_hit_details_window(trace, render);

    ImGui::SetNextWindowSize(ImVec2(520, 320), ImGuiCond_FirstUseEver);
    if ( !ImGui::Begin("Ray Inspector") ) {
        ImGui::End();
        return;
    }
    if ( !summary.empty() ) {
        ImGui::TextWrapped("%s", summary.c_str());
        ImGui::Separator();
    }
    if ( trace.steps.empty() ) {
        ImGui::TextDisabled(
            "no trace yet — pick a ray in the Scene Debug view (RMB) to "
            "inspect its hits");
        ImGui::End();
        return;
    }
    draw_trace_inspector(trace, render);
    ImGui::End();
}

} // namespace

void init_scene_debug(GLFWwindow *share_window) {
    if ( !g_renderer.init(share_window) ) {
        spdlog::error("[scene_debug] renderer initialization failed");
    }
}

void shutdown_scene_debug() {
    g_renderer.shutdown();
}

bool debug_apply_scene_camera(float *cam_eye, float *cam_at, float *cam_up) {
    auto &st = g_renderer.state();
    std::lock_guard<std::mutex> lk(st.mutex);
    if (!st.set_scene_camera) return false;
    st.set_scene_camera = false;
    cam_eye[0] = st.set_scene_eye[0];
    cam_eye[1] = st.set_scene_eye[1];
    cam_eye[2] = st.set_scene_eye[2];
    cam_at[0] = st.set_scene_at[0];
    cam_at[1] = st.set_scene_at[1];
    cam_at[2] = st.set_scene_at[2];
    cam_up[0] = st.set_scene_up[0];
    cam_up[1] = st.set_scene_up[1];
    cam_up[2] = st.set_scene_up[2];
    return true;
}

void render_scene_debug(const HostScene *host_scene,
                        const float *cam_eye,
                        const float *cam_at,
                        const float *cam_up,
                        float cam_fov,
                        float cam_aspect,
                        int fb_w,
                        int fb_h,
                        const SceneRenderParams &render,
                        DebugViewFlags &flags) {
    auto &st = g_renderer.state();

    // Mirror the kernel's render parameters to the render thread (the
    // debug trace + overlay display follow the kernel exactly).
    {
        std::lock_guard<std::mutex> lk(st.mutex);
        st.render = render;
    }

    // Default the debug bounce depth to the kernel's max_bounces (the
    // path the debugger traces then matches the rendered path).  The
    // user can still override it with the slider; on a scene change
    // (or a param edit) the default is restored.
    static int last_scene_bounces = -1;
    if ( render.max_bounces != last_scene_bounces ) {
        flags.ray_bounces = std::max(render.max_bounces, 1);
        last_scene_bounces = render.max_bounces;
    }

    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
    if ( !ImGui::Begin("Scene Debug") ) {
        // Collapsed/hidden — tell the renderer to skip frames.
        std::lock_guard<std::mutex> lk(st.mutex);
        st.window_open = false;
        ImGui::End();
        // The Ray Inspector is an independent dockable window: keep it
        // alive (showing the last trace) even while the 3D view is
        // collapsed.
        if ( flags.show_ray ) draw_ray_inspector_window(st);
        return;
    }

    // ── Toolbar ─────────────────────────────────────────────────────
    ImGui::Checkbox("Floor", &flags.show_floor);
    ImGui::SameLine();
    ImGui::Checkbox("Grid", &flags.show_grid);
    ImGui::SameLine();
    ImGui::Checkbox("Objects", &flags.show_objects);
    ImGui::SameLine();
    ImGui::Checkbox("Wireframe", &flags.show_wireframe);
    ImGui::SameLine();
    ImGui::Checkbox("AABBs", &flags.show_aabbs);
    ImGui::Separator();
    ImGui::Checkbox("Frustum", &flags.show_frustum);
    ImGui::SameLine();
    ImGui::Checkbox("Camera", &flags.show_camera);
    ImGui::SameLine();
    ImGui::Checkbox("Visibility", &flags.show_visibility);
    ImGui::SameLine();
    ImGui::Checkbox("BVH", &flags.show_bvh);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110.f);
    ImGui::SliderInt("BVH depth", &flags.bvh_depth, 1, 16);
    ImGui::SameLine();
    if ( ImGui::Button("Reset View") ) {
        std::lock_guard<std::mutex> lk(st.mutex);
        st.reset_view = true;
    }
    ImGui::SameLine();
    if ( ImGui::Button("Set Scene Camera") && cam_eye && cam_at && cam_up ) {
        std::lock_guard<std::mutex> lk(st.mutex);
        float cp = std::cos(st.orbit_pitch), sp = std::sin(st.orbit_pitch);
        float sn = std::sin(st.orbit_yaw), ct = std::cos(st.orbit_yaw);
        float forward[3] = {cp * sn, sp, cp * ct};
        st.set_scene_eye[0] = st.orbit_eye[0];
        st.set_scene_eye[1] = st.orbit_eye[1];
        st.set_scene_eye[2] = st.orbit_eye[2];
        st.set_scene_at[0] = st.orbit_eye[0] + st.orbit_dist * forward[0];
        st.set_scene_at[1] = st.orbit_eye[1] + st.orbit_dist * forward[1];
        st.set_scene_at[2] = st.orbit_eye[2] + st.orbit_dist * forward[2];
        st.set_scene_up[0] = 0.f; st.set_scene_up[1] = 1.f; st.set_scene_up[2] = 0.f;
        st.set_scene_camera = true;
    }
    ImGui::Separator();
    ImGui::Checkbox("Ray", &flags.show_ray);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110.f);
    ImGui::SliderInt("bounces", &flags.ray_bounces, 0, 16);
    ImGui::SameLine();
    if ( ImGui::Button("Reset Ray") ) {
        std::lock_guard<std::mutex> lk(st.mutex);
        st.reset_ray = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("RMB: pick ray");
    ImGui::Separator();

    // ── Scene stats ─────────────────────────────────────────────────
    if ( host_scene && host_scene->debug_scene ) {
        const auto &v = host_scene->debug_scene->view;
        ImGui::TextDisabled("%d objects · %d BVH nodes · %d portals · scene v%llu",
                            v.num_handles, v.num_bvh_nodes, v.num_portals,
                            (unsigned long long)host_scene->debug_scene->version);
    } else {
        ImGui::TextDisabled("no scene (kernel without YAML scene)");
    }

    // ── Ray trace inspector ────────────────────────────────────────
    // All raytracing details (trace summary, hits treeview, hit
    // details dialog) live in their OWN dockable windows — nothing
    // here but the fixed toolbar, so the 3D view size never changes
    // with the trace data.
    if ( flags.show_ray ) draw_ray_inspector_window(st);

    // ── 3D view: presented texture + mouse input ────────────────────
    ImVec2 avail = ImGui::GetContentRegionAvail();
    float w = std::max(avail.x, 64.f);
    float h = std::max(avail.y - ImGui::GetTextLineHeightWithSpacing(), 64.f);

    // Release the previous slot + pick the newest ready frame.  The
    // returned texture is sampled during ImGui::Render (composite_frame)
    // and is never overwritten before the next present() call.
    GLuint tex = g_renderer.present();
    if ( tex ) {
        // Present V-flipped: GL FBO textures are bottom-up (v=0 is the
        // rendered image's bottom row), but ImGui::Image maps uv (0,0)
        // to the top of the widget — the same convention the main
        // viewport compensates for in the tone-mapper (Y-flip).  With
        // default UVs the whole scene would display upside down.
        ImGui::Image((ImTextureID)(intptr_t)tex, ImVec2(w, h),
                     ImVec2(0.f, 1.f), ImVec2(1.f, 0.f));
    } else {
        // No frame yet (renderer starting up / unavailable)
        ImGui::InvisibleButton("##scene_debug_empty", ImVec2(w, h));
        ImVec2 pos = ImGui::GetItemRectMin();
        ImGui::SetCursorScreenPos(ImVec2(pos.x + 8.f, pos.y + 8.f));
        ImGui::TextDisabled("scene debug renderer unavailable");
    }

    bool hovered = ImGui::IsItemHovered();
    // Camera controls mirror the main Viewport exactly: LMB pans the
    // Right-click picks a ray; dragging keeps re-picking every frame so
    // the ray and its trace/stats follow the cursor.  Clicking ON the
    // scene-camera framebuffer rectangle selects the exact framebuffer
    // pixel there (the render thread resolves the rectangle hit).
    bool pick_click = hovered &&
                      (ImGui::IsMouseClicked(ImGuiMouseButton_Right) ||
                       ImGui::IsMouseDragging(ImGuiMouseButton_Right));
    const ImGuiIO &io = ImGui::GetIO();

    {
        std::lock_guard<std::mutex> lk(st.mutex);

        // Ray pick: convert the click to NDC (y up); the render thread
        // resolves it with the exact camera used for the frame.
        if ( pick_click ) {
            ImVec2 rect_min = ImGui::GetItemRectMin();
            ImVec2 rect_max = ImGui::GetItemRectMax();
            ImVec2 mouse = ImGui::GetMousePos();
            ImVec2 frac = {(mouse.x - rect_min.x) / (rect_max.x - rect_min.x),
                           (mouse.y - rect_min.y) / (rect_max.y - rect_min.y)};
            st.pick_ray_request = true;
            st.pick_ndc_x = 2.f * frac.x - 1.f;
            st.pick_ndc_y = 1.f - 2.f * frac.y;
        }

        // Scene snapshot (shared_ptr hand-off keeps it alive while the
        // render thread is mid-frame with it).
        st.scene = host_scene ? host_scene->debug_scene : nullptr;

        // Scene camera (frustum + visibility overlays only — the debug
        // camera is independent).
        st.scene_cam_valid = cam_eye && cam_at && cam_up;
        if ( st.scene_cam_valid ) {
            std::memcpy(st.scene_cam_eye, cam_eye, sizeof(st.scene_cam_eye));
            std::memcpy(st.scene_cam_at, cam_at, sizeof(st.scene_cam_at));
            std::memcpy(st.scene_cam_up, cam_up, sizeof(st.scene_cam_up));
        }
        st.scene_cam_fov = cam_fov;
        st.scene_cam_aspect = cam_aspect;
        st.fb_w = fb_w;
        st.fb_h = fb_h;
        st.size_w = (int)w;
        st.size_h = (int)h;
        st.flags = flags;
        st.window_open = true;

        // ── First-person camera input (matches the main Viewport). ──
        bool cam_changed = false;

        // MMB drag = rotate camera (yaw/pitch)
        if ( ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ) {
            auto d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle);
            st.orbit_yaw -= d.x * 0.005f;
            st.orbit_pitch -= d.y * 0.005f;
            st.orbit_pitch = std::clamp(st.orbit_pitch, -1.5f, 1.5f);
            ImGui::ResetMouseDragDelta(ImGuiMouseButton_Middle);
            cam_changed = true;
        }
        // LMB drag = pan in camera-local plane
        if ( ImGui::IsMouseDragging(ImGuiMouseButton_Left) ) {
            auto d = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
            float ct = std::cos(st.orbit_yaw), sn = std::sin(st.orbit_yaw);
            float cp = std::cos(st.orbit_pitch), sp = std::sin(st.orbit_pitch);
            float forward[3] = {cp * sn, sp, cp * ct};
            float right[3]   = {ct, 0.f, -sn};
            float up[3]       = {-sp * sn, cp, -sp * ct};
            float speed = st.orbit_dist * 0.002f;
            if (io.KeyCtrl) {
                st.orbit_eye[0] += d.y * forward[0] * speed;
                st.orbit_eye[1] += d.y * forward[1] * speed;
                st.orbit_eye[2] += d.y * forward[2] * speed;
            } else {
                st.orbit_eye[0] -= d.x * right[0] * speed + d.y * up[0] * speed;
                st.orbit_eye[1] -= d.x * right[1] * speed + d.y * up[1] * speed;
                st.orbit_eye[2] -= d.x * right[2] * speed + d.y * up[2] * speed;
            }
            ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
            cam_changed = true;
        }

        // Scroll = move along look axis
        if ( hovered && io.MouseWheel != 0.f ) {
            float ct = std::cos(st.orbit_yaw), sn = std::sin(st.orbit_yaw);
            float cp = std::cos(st.orbit_pitch), sp = std::sin(st.orbit_pitch);
            float forward[3] = {cp * sn, sp, cp * ct};
            float speed = st.orbit_dist * 0.2f;
            st.orbit_eye[0] -= io.MouseWheel * forward[0] * speed;
            st.orbit_eye[1] -= io.MouseWheel * forward[1] * speed;
            st.orbit_eye[2] -= io.MouseWheel * forward[2] * speed;
            cam_changed = true;
        }

        // Keyboard: WASD/QE
        if ( hovered && ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) ) {
            float ct = std::cos(st.orbit_yaw), sn = std::sin(st.orbit_yaw);
            float cp = std::cos(st.orbit_pitch), sp = std::sin(st.orbit_pitch);
            float forward[3] = {cp * sn, sp, cp * ct};
            float right[3]   = {ct, 0.f, -sn};
            float up[3]       = {-sp * sn, cp, -sp * ct};
            float tspeed = 0.3f;

            if ( ImGui::IsKeyDown(ImGuiKey_W) ) { st.orbit_eye[0] += forward[0] * tspeed; st.orbit_eye[1] += forward[1] * tspeed; st.orbit_eye[2] += forward[2] * tspeed; cam_changed = true; }
            if ( ImGui::IsKeyDown(ImGuiKey_S) ) { st.orbit_eye[0] -= forward[0] * tspeed; st.orbit_eye[1] -= forward[1] * tspeed; st.orbit_eye[2] -= forward[2] * tspeed; cam_changed = true; }
            if ( ImGui::IsKeyDown(ImGuiKey_A) ) { st.orbit_eye[0] += right[0] * tspeed;  st.orbit_eye[1] += right[1] * tspeed;  st.orbit_eye[2] += right[2] * tspeed; cam_changed = true; }
            if ( ImGui::IsKeyDown(ImGuiKey_D) ) { st.orbit_eye[0] -= right[0] * tspeed;  st.orbit_eye[1] -= right[1] * tspeed;  st.orbit_eye[2] -= right[2] * tspeed; cam_changed = true; }
            if ( ImGui::IsKeyDown(ImGuiKey_Q) ) { st.orbit_eye[0] -= up[0] * 0.3f; st.orbit_eye[1] -= up[1] * 0.3f; st.orbit_eye[2] -= up[2] * 0.3f; cam_changed = true; }
            if ( ImGui::IsKeyDown(ImGuiKey_E) ) { st.orbit_eye[0] += up[0] * 0.3f; st.orbit_eye[1] += up[1] * 0.3f; st.orbit_eye[2] += up[2] * 0.3f; cam_changed = true; }
        }

        if ( cam_changed ) st.camera_user_interacted = true;

        if ( cam_changed ) st.camera_user_interacted = true;
    }

    ImGui::TextDisabled("LMB pan · MMB orbit · wheel zoom · arrows orbit · "
                        "Shift+arrows pan · WASD/QE move · RMB pick/drag ray "
                        "(on the framebuffer rect: pixel pick)");
    ImGui::End();
}
