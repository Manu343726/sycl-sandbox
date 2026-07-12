#include "panel.h"
#include "gl_loader.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <spdlog/spdlog.h>

// Scene debug panel implementation
// Renders a 3D OpenGL viewport showing scene geometry, BVH volumes, and camera frustum.

// ── OpenGL resources ──────────────────────────────────────────────────
static GLuint g_debug_vao = 0;
static GLuint g_debug_vbo = 0;
static bool g_debug_inited = false;

void init_scene_debug() {
    if (g_debug_inited) return;
    glGenVertexArrays(1, &g_debug_vao);
    glGenBuffers(1, &g_debug_vbo);
    g_debug_inited = true;
    spdlog::debug("[scene_debug] OpenGL resources created (vao={}, vbo={})", g_debug_vao, g_debug_vbo);
}

void shutdown_scene_debug() {
    if (!g_debug_inited) return;
    glDeleteVertexArrays(1, &g_debug_vao);
    glDeleteBuffers(1, &g_debug_vbo);
    g_debug_vao = 0;
    g_debug_vbo = 0;
    g_debug_inited = false;
    spdlog::debug("[scene_debug] OpenGL resources freed");
}

// ── Simple 3D line rendering helper ──────────────────────────────────
static void draw_lines(const float *verts, int count, const float color[3], float line_width = 1.0f) {
    if (count < 2) return;
    glLineWidth(line_width);
    glColor3fv(color);
    glBindVertexArray(g_debug_vao);
    glBindBuffer(GL_ARRAY_BUFFER, g_debug_vbo);
    glBufferData(GL_ARRAY_BUFFER, count * 3 * sizeof(float), verts, GL_DYNAMIC_DRAW);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, nullptr);
    glDrawArrays(GL_LINES, 0, count);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// ── AABB wireframe helper (8 vertices → 12 edges) ────────────────────
static void draw_aabb(const float *aabb_min, const float *aabb_max, const float color[3]) {
    // 8 corners
    float corners[8][3] = {
        {aabb_min[0], aabb_min[1], aabb_min[2]},
        {aabb_max[0], aabb_min[1], aabb_min[2]},
        {aabb_max[0], aabb_max[1], aabb_min[2]},
        {aabb_min[0], aabb_max[1], aabb_min[2]},
        {aabb_min[0], aabb_min[1], aabb_max[2]},
        {aabb_max[0], aabb_min[1], aabb_max[2]},
        {aabb_max[0], aabb_max[1], aabb_max[2]},
        {aabb_min[0], aabb_max[1], aabb_max[2]},
    };
    // 12 edges (index pairs)
    int edges[24] = {
        0,1, 1,2, 2,3, 3,0,
        4,5, 5,6, 6,7, 7,4,
        0,4, 1,5, 2,6, 3,7
    };
    float verts[72];
    for (int i = 0; i < 24; i++) {
        verts[i*3+0] = corners[edges[i]][0];
        verts[i*3+1] = corners[edges[i]][1];
        verts[i*3+2] = corners[edges[i]][2];
    }
    draw_lines(verts, 24, color, 0.5f);
}

// ── Sphere wireframe helper (latitude/longitude rings) ───────────────
static void draw_sphere_wireframe(const float *center, float radius,
                                   const float color[3], int segments = 12) {
    // Meridians (longitude rings)
    float verts[1024];
    int count = 0;
    for (int m = 0; m < segments; m++) {
        float theta = (float)m / segments * 2.0f * 3.14159265f;
        float ct = cosf(theta), st = sinf(theta);
        for (int i = 0; i <= segments; i++) {
            float phi = (float)i / segments * 3.14159265f;
            float cp = cosf(phi), sp = sinf(phi);
            verts[count*3+0] = center[0] + radius * sp * ct;
            verts[count*3+1] = center[1] + radius * cp;
            verts[count*3+2] = center[2] + radius * sp * st;
            count++;
        }
    }
    draw_lines(verts, count, color, 0.5f);
}

// ── Floor grid ───────────────────────────────────────────────────────
static void draw_floor_grid(float size, int divs, const float color[3]) {
    float half = size * 0.5f;
    float step = size / divs;
    std::vector<float> verts;
    for (int i = 0; i <= divs; i++) {
        float x = -half + i * step;
        verts.push_back(x); verts.push_back(0); verts.push_back(-half);
        verts.push_back(x); verts.push_back(0); verts.push_back(half);
        float z = -half + i * step;
        verts.push_back(-half); verts.push_back(0); verts.push_back(z);
        verts.push_back(half);  verts.push_back(0); verts.push_back(z);
    }
    draw_lines(verts.data(), (int)verts.size() / 3, color, 0.5f);
}

// ── Frustum (camera view volume) ─────────────────────────────────────
static void draw_frustum(const float *eye, const float *at, const float *up,
                          float fov_deg, float aspect, float near_d, float far_d,
                          const float color[3]) {
    // Calculate frustum corners at near/far planes
    float forward[3] = {at[0]-eye[0], at[1]-eye[1], at[2]-eye[2]};
    float dist = sqrtf(forward[0]*forward[0] + forward[1]*forward[1] + forward[2]*forward[2]);
    if (dist < 1e-6f) return;
    forward[0] /= dist; forward[1] /= dist; forward[2] /= dist;

    // Right vector (forward × world_up)
    float world_up[3] = {0,1,0};
    float right[3];
    right[0] = forward[1]*world_up[2] - forward[2]*world_up[1];
    right[1] = forward[2]*world_up[0] - forward[0]*world_up[2];
    right[2] = forward[0]*world_up[1] - forward[1]*world_up[0];
    float rlen = sqrtf(right[0]*right[0]+right[1]*right[1]+right[2]*right[2]);
    if (rlen < 1e-6f) return;
    right[0] /= rlen; right[1] /= rlen; right[2] /= rlen;

    // Real up vector
    float cam_up[3] = {up[0], up[1], up[2]};
    float ulen = sqrtf(cam_up[0]*cam_up[0]+cam_up[1]*cam_up[1]+cam_up[2]*cam_up[2]);
    if (ulen < 1e-6f) return;
    cam_up[0] /= ulen; cam_up[1] /= ulen; cam_up[2] /= ulen;

    float fov_rad = fov_deg * 3.14159265f / 180.0f;
    float half_h_near = tanf(fov_rad * 0.5f) * near_d;
    float half_w_near = half_h_near * aspect;
    float half_h_far  = tanf(fov_rad * 0.5f) * far_d;
    float half_w_far  = half_h_far * aspect;

    // 8 corners of the frustum
    float corners[8][3];
    auto frustum_corner = [&](int idx, float half_w, float half_h, float d) {
        float cx = eye[0] + forward[0]*d + right[0]*(idx&1 ? half_w : -half_w) + cam_up[0]*(idx&2 ? half_h : -half_h);
        float cy = eye[1] + forward[1]*d + right[1]*(idx&1 ? half_w : -half_w) + cam_up[1]*(idx&2 ? half_h : -half_h);
        float cz = eye[2] + forward[2]*d + right[2]*(idx&1 ? half_w : -half_w) + cam_up[2]*(idx&2 ? half_h : -half_h);
        corners[idx][0] = cx; corners[idx][1] = cy; corners[idx][2] = cz;
    };
    for (int i = 0; i < 4; i++) frustum_corner(i, half_w_near, half_h_near, near_d);
    for (int i = 0; i < 4; i++) frustum_corner(4+i, half_w_far, half_h_far, far_d);

    float verts[72]; // 24 edges * 3
    int e[24] = {0,1, 1,3, 3,2, 2,0,  4,5, 5,7, 7,6, 6,4,  0,4, 1,5, 2,6, 3,7};
    for (int i = 0; i < 24; i++) {
        verts[i*3+0] = corners[e[i]][0];
        verts[i*3+1] = corners[e[i]][1];
        verts[i*3+2] = corners[e[i]][2];
    }
    draw_lines(verts, 24, color, 1.5f);
}

// ── Main render function ──────────────────────────────────────────────
void render_scene_debug(const SceneDebugInfo *dbg,
                         const float *cam_eye,
                         const float *cam_at,
                         const float *cam_up,
                         float cam_fov,
                         float aspect,
                         DebugViewFlags &flags) {
    ImGui::BeginGroup();
    ImVec2 region = ImGui::GetContentRegionAvail();
    float size = std::min(region.x, region.y);
    if (size < 32) { ImGui::EndGroup(); return; }

    // ── Controls ──────────────────────────────────────────────────
    ImGui::Checkbox("Floor", &flags.show_floor); ImGui::SameLine();
    ImGui::Checkbox("Grid", &flags.show_grid); ImGui::SameLine();
    ImGui::Checkbox("Objects", &flags.show_objects); ImGui::SameLine();
    ImGui::Checkbox("Wireframe", &flags.show_wireframe); ImGui::SameLine();
    ImGui::Checkbox("AABBs", &flags.show_aabbs); ImGui::SameLine();
    ImGui::Checkbox("Frustum", &flags.show_frustum); ImGui::SameLine();
    ImGui::Checkbox("Camera", &flags.show_camera);

    // ── Viewport ──────────────────────────────────────────────────
    ImVec2 vp_pos = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##debug_vp", ImVec2(size, size));
    bool hovered = ImGui::IsItemHovered();

    // Save OpenGL state
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    // Set up viewport
    glViewport((int)vp_pos.x, (int)(ImGui::GetIO().DisplaySize.y - vp_pos.y - size),
               (int)size, (int)size);
    glScissor((int)vp_pos.x, (int)(ImGui::GetIO().DisplaySize.y - vp_pos.y - size),
              (int)size, (int)size);
    glEnable(GL_SCISSOR_TEST);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    // Simple perspective (fov 45°, near 0.1, far 200)
    float fov_rad = 45.0f * 3.14159265f / 180.0f;
    float top = tanf(fov_rad * 0.5f) * 0.1f;
    float bottom = -top;
    float right_val = top * (size / size); // square aspect
    float left_val = -right_val;
    glFrustum(left_val, right_val, bottom, top, 0.1f, 200.0f);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    if (cam_eye && cam_at && cam_up) {
        // Manual look-at matrix (replaces gluLookAt)
        {
            float fwd[3] = {cam_at[0]-cam_eye[0], cam_at[1]-cam_eye[1], cam_at[2]-cam_eye[2]};
            float len = sqrtf(fwd[0]*fwd[0]+fwd[1]*fwd[1]+fwd[2]*fwd[2]);
            if (len > 0.f) { fwd[0]/=len; fwd[1]/=len; fwd[2]/=len; }
            float side[3];
            side[0] = fwd[1]*cam_up[2] - fwd[2]*cam_up[1];
            side[1] = fwd[2]*cam_up[0] - fwd[0]*cam_up[2];
            side[2] = fwd[0]*cam_up[1] - fwd[1]*cam_up[0];
            float slen = sqrtf(side[0]*side[0]+side[1]*side[1]+side[2]*side[2]);
            if (slen > 0.f) { side[0]/=slen; side[1]/=slen; side[2]/=slen; }
            float up2[3];
            up2[0] = side[1]*fwd[2] - side[2]*fwd[1];
            up2[1] = side[2]*fwd[0] - side[0]*fwd[2];
            up2[2] = side[0]*fwd[1] - side[1]*fwd[0];
            float m[16] = {side[0], up2[0], -fwd[0], 0,
                           side[1], up2[1], -fwd[1], 0,
                           side[2], up2[2], -fwd[2], 0,
                           0,0,0,1};
            glMultMatrixf(m);
            glTranslatef(-cam_eye[0], -cam_eye[1], -cam_eye[2]);
        }
    } else {
        // Default fallback camera (no lookAt — identity view)
    }

    glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    // ── Floor + Grid ──────────────────────────────────────────────
    float grid_color[3] = {0.3f, 0.3f, 0.35f};
    if (flags.show_floor) {
        float floor_col[3] = {0.2f, 0.22f, 0.25f};
        float floor_verts[12] = {-10,0,-10, 10,0,-10, 10,0,10, -10,0,10};
        draw_lines(floor_verts, 4, floor_col);
    }
    if (flags.show_grid) {
        draw_floor_grid(20, 20, grid_color);
    }

    // ── Scene objects ─────────────────────────────────────────────
    if (dbg && flags.show_objects) {
        float sphere_color[3] = {0.4f, 0.8f, 1.0f};
        float aabb_color[3] = {1.0f, 0.2f, 0.2f};

        // Spheres
        if (dbg->num_spheres > 0 && dbg->spheres) {
            for (int i = 0; i < dbg->num_spheres; i++) {
                const float *s = reinterpret_cast<const float*>(&dbg->spheres[i]);
                if (flags.show_wireframe)
                    draw_sphere_wireframe(s, s[3], sphere_color);
            }
        }

        // AABBs
        if (flags.show_aabbs && dbg->num_aabbs > 0 && dbg->aabb_data) {
            for (int i = 0; i < dbg->num_aabbs; i++) {
                const float *aabb = dbg->aabb_data + i * 6;
                draw_aabb(aabb, aabb + 3, aabb_color);
            }
        }
    }

    // ── Camera frustum ────────────────────────────────────────────
    if (flags.show_frustum && cam_eye && cam_at && cam_up) {
        float fcol[3] = {0.2f, 1.0f, 0.2f};
        draw_frustum(cam_eye, cam_at, cam_up, cam_fov, aspect, 0.5f, 5.0f, fcol);
    }

    // ── Camera position indicator ─────────────────────────────────
    if (flags.show_camera && cam_eye) {
        float ccol[3] = {1.0f, 1.0f, 0.2f};
        // Small cross at camera position
        float cross_sz = 0.3f;
        float cv[6] = {
            cam_eye[0]-cross_sz, cam_eye[1], cam_eye[2],
            cam_eye[0]+cross_sz, cam_eye[1], cam_eye[2]
        };
        draw_lines(cv, 2, ccol, 2.0f);
    }

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();

    ImGui::EndGroup();
}
