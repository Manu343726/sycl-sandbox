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

// ── Solid sphere (lat/long quad grid) ────────────────────────────────
static void draw_sphere_solid(const float *center, float radius,
                               const float color[3], int segments = 16) {
    glColor3fv(color);
    glBegin(GL_QUADS);
    for (int lat = 0; lat < segments; lat++) {
        float phi0 = (float)lat / segments * 3.14159265f;
        float phi1 = (float)(lat + 1) / segments * 3.14159265f;
        for (int lon = 0; lon < segments; lon++) {
            float th0 = (float)lon / segments * 2.0f * 3.14159265f;
            float th1 = (float)(lon + 1) / segments * 2.0f * 3.14159265f;
            auto corner = [&](float phi, float theta) {
                float cp = cosf(phi), sp = sinf(phi);
                float ct = cosf(theta), st = sinf(theta);
                glVertex3f(center[0] + radius * sp * ct,
                           center[1] + radius * cp,
                           center[2] + radius * sp * st);
            };
            corner(phi0, th0); corner(phi0, th1);
            corner(phi1, th1); corner(phi1, th0);
        }
    }
    glEnd();
}

// ── Solid quad (parallelogram base/u/v, 2 triangles) ─────────────────
static void draw_quad_solid(const float *base, const float *u, const float *v,
                             const float color[3]) {
    glColor3fv(color);
    glBegin(GL_TRIANGLES);
    // base, base+u, base+u+v
    glVertex3fv(base);
    glVertex3f(base[0]+u[0], base[1]+u[1], base[2]+u[2]);
    glVertex3f(base[0]+u[0]+v[0], base[1]+u[1]+v[1], base[2]+u[2]+v[2]);
    // base, base+u+v, base+v
    glVertex3fv(base);
    glVertex3f(base[0]+u[0]+v[0], base[1]+u[1]+v[1], base[2]+u[2]+v[2]);
    glVertex3f(base[0]+v[0], base[1]+v[1], base[2]+v[2]);
    glEnd();
}

// ── Quad wireframe edges ─────────────────────────────────────────────
static void draw_quad_wireframe(const float *base, const float *u, const float *v,
                                 const float color[3]) {
    float verts[12];
    verts[0]=base[0]; verts[1]=base[1]; verts[2]=base[2];
    verts[3]=base[0]+u[0]; verts[4]=base[1]+u[1]; verts[5]=base[2]+u[2];
    verts[6]=base[0]+u[0]+v[0]; verts[7]=base[1]+u[1]+v[1]; verts[8]=base[2]+u[2]+v[2];
    verts[9]=base[0]+v[0]; verts[10]=base[1]+v[1]; verts[11]=base[2]+v[2];
    // closed loop: 4 segments
    float loop[24];
    for (int i = 0; i < 4; i++) {
        int a = i, b = (i + 1) % 4;
        loop[i*6+0]=verts[a*3+0]; loop[i*6+1]=verts[a*3+1]; loop[i*6+2]=verts[a*3+2];
        loop[i*6+3]=verts[b*3+0]; loop[i*6+4]=verts[b*3+1]; loop[i*6+5]=verts[b*3+2];
    }
    draw_lines(loop, 8, color, 1.0f);
}

// ── Solid box (6 faces, per-face colours) ────────────────────────────
// Face order: 0=-x 1=+x 2=-y 3=+y 4=-z 5=+z.  The near/far constant
// coordinate is taken from min/max; the other two span the face.
static void draw_box_solid(const float *mn, const float *mx,
                            const float colors[6][3]) {
    auto face = [&](int axis, float coord, float lo_a, float hi_a,
                    float lo_b, float hi_b, const float color[3]) {
        glColor3fv(color);
        float v[4][3] = {
            {coord, lo_a, lo_b}, {coord, hi_a, lo_b},
            {coord, hi_a, hi_b}, {coord, lo_a, hi_b},
        };
        // Reorder so the quad is (axis-consistent) CCW for a nice look;
        // lighting is off so winding only matters for culling (off too).
        glBegin(GL_QUADS);
        glVertex3fv(v[0]); glVertex3fv(v[1]);
        glVertex3fv(v[2]); glVertex3fv(v[3]);
        glEnd();
    };
    face(0, mn[0], mn[1], mx[1], mn[2], mx[2], colors[0]);
    face(1, mx[0], mn[1], mx[1], mn[2], mx[2], colors[1]);
    face(2, mn[1], mn[0], mx[0], mn[2], mx[2], colors[2]);
    face(3, mx[1], mn[0], mx[0], mn[2], mx[2], colors[3]);
    face(4, mn[2], mn[0], mx[0], mn[1], mx[1], colors[4]);
    face(5, mx[2], mn[0], mx[0], mn[1], mx[1], colors[5]);
}

// ── Box wireframe (12 edges) ─────────────────────────────────────────
static void draw_box_wireframe(const float *mn, const float *mx,
                                const float color[3]) {
    float corners[8][3] = {
        {mn[0], mn[1], mn[2]}, {mx[0], mn[1], mn[2]},
        {mx[0], mx[1], mn[2]}, {mn[0], mx[1], mn[2]},
        {mn[0], mn[1], mx[2]}, {mx[0], mn[1], mx[2]},
        {mx[0], mx[1], mx[2]}, {mn[0], mx[1], mx[2]},
    };
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

// ── CPU ray helpers (occlusion tests against debug geometry) ─────────
struct CpuRay {
    float o[3];
    float d[3];   // normalized
};

// Ray/sphere: returns nearest positive hit distance.
static bool ray_hit_sphere(const float *center, float radius, const CpuRay &r,
                           float &t) {
    float oc[3] = {r.o[0]-center[0], r.o[1]-center[1], r.o[2]-center[2]};
    float a = r.d[0]*r.d[0] + r.d[1]*r.d[1] + r.d[2]*r.d[2];
    float half_b = oc[0]*r.d[0] + oc[1]*r.d[1] + oc[2]*r.d[2];
    float c = oc[0]*oc[0] + oc[1]*oc[1] + oc[2]*oc[2] - radius*radius;
    float disc = half_b*half_b - a*c;
    if (disc <= 0.f) return false;
    float sq = sqrtf(disc);
    t = (-half_b - sq) / a;
    if (t <= 0.f) t = (-half_b + sq) / a;   // inside sphere → exit
    return t > 0.f;
}

// Ray/parallelogram (base + u·edge_u + v·edge_v, u,v in [0,1]).
static bool ray_hit_quad(const float *base, const float *u, const float *v,
                         const CpuRay &r, float &t) {
    float n[3] = {
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0],
    };
    float denom = n[0]*r.d[0] + n[1]*r.d[1] + n[2]*r.d[2];
    if (fabsf(denom) < 1e-8f) return false;
    float w[3] = {base[0]-r.o[0], base[1]-r.o[1], base[2]-r.o[2]};
    t = (n[0]*w[0] + n[1]*w[1] + n[2]*w[2]) / denom;
    if (t <= 0.f) return false;
    float p[3] = {r.o[0]+r.d[0]*t, r.o[1]+r.d[1]*t, r.o[2]+r.d[2]*t};
    float duu = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
    float dvv = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    if (duu < 1e-12f || dvv < 1e-12f) return false;
    float a = ((p[0]-base[0])*u[0] + (p[1]-base[1])*u[1] + (p[2]-base[2])*u[2]) / duu;
    float b = ((p[0]-base[0])*v[0] + (p[1]-base[1])*v[1] + (p[2]-base[2])*v[2]) / dvv;
    return a >= -1e-4f && a <= 1.f + 1e-4f && b >= -1e-4f && b <= 1.f + 1e-4f;
}

// Ray/AABB (slab method): returns the entry distance.
static bool ray_hit_box(const float *mn, const float *mx, const CpuRay &r,
                        float &t) {
    float tmin = -1e30f, tmax = 1e30f;
    for (int axis = 0; axis < 3; axis++) {
        if (fabsf(r.d[axis]) < 1e-9f) {
            if (r.o[axis] < mn[axis] || r.o[axis] > mx[axis]) return false;
            continue;
        }
        float inv = 1.f / r.d[axis];
        float t1 = (mn[axis] - r.o[axis]) * inv;
        float t2 = (mx[axis] - r.o[axis]) * inv;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
    }
    if (tmax < tmin || tmax <= 0.f) return false;
    t = fmaxf(tmin, 0.f);
    return true;
}

// ── Visibility from the scene camera ─────────────────────────────────
// Casts a ray from `eye` to `p` and checks whether ANY debug object
// (sphere/quad/box) occludes the segment before the target point.
static bool point_visible_from(const float p[3], const float eye[3],
                               const SceneDebugInfo *dbg) {
    CpuRay ray;
    ray.o[0] = eye[0]; ray.o[1] = eye[1]; ray.o[2] = eye[2];
    float dx = p[0]-eye[0], dy = p[1]-eye[1], dz = p[2]-eye[2];
    float len = sqrtf(dx*dx + dy*dy + dz*dz);
    if (len < 1e-6f) return true;
    ray.d[0] = dx/len; ray.d[1] = dy/len; ray.d[2] = dz/len;
    // Slightly short of the target so the target surface itself (which
    // lies exactly at `len`) is not counted as an occluder.
    float t_target = len - 0.002f;
    float t;
    if (dbg->spheres) {
        for (int i = 0; i < dbg->num_spheres; i++) {
            const float *s = reinterpret_cast<const float*>(&dbg->spheres[i]);
            if (ray_hit_sphere(s, s[3], ray, t) && t < t_target) return false;
        }
    }
    if (dbg->quads) {
        for (int i = 0; i < dbg->num_quads; i++) {
            const float *q = reinterpret_cast<const float*>(&dbg->quads[i]);
            if (ray_hit_quad(q, q+3, q+6, ray, t) && t < t_target) return false;
        }
    }
    if (dbg->boxes) {
        for (int i = 0; i < dbg->num_boxes; i++) {
            const float *b = reinterpret_cast<const float*>(&dbg->boxes[i]);
            if (ray_hit_box(b, b+3, ray, t) && t < t_target) return false;
        }
    }
    return true;
}

// Shade a colour by visibility: visible → material colour, occluded →
// darkened (used by the visibility overlay).
static void shade_by_visibility(float out[3], const float color[3], bool visible) {
    if (visible) {
        out[0] = color[0]; out[1] = color[1]; out[2] = color[2];
    } else {
        out[0] = color[0] * 0.12f + 0.02f;
        out[1] = color[1] * 0.12f + 0.02f;
        out[2] = color[2] * 0.12f + 0.02f;
    }
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
    ImGui::Checkbox("Camera", &flags.show_camera); ImGui::SameLine();
    ImGui::Checkbox("Visibility", &flags.show_visibility);

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
        // Fixed-function shading is off; glColor3f is the material colour.
        glDisable(GL_LIGHTING);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0f, 1.0f);
        float aabb_color[3] = {1.0f, 0.2f, 0.2f};
        // Neutral edge colour used by the visibility overlay so the
        // shaded fills stay readable.
        float edge_vis[3] = {0.85f, 0.85f, 0.95f};

        // Spheres — solid (material colour) + wireframe overlay.
        // Visibility: the front point facing the camera is tested.
        if (dbg->num_spheres > 0 && dbg->spheres) {
            for (int i = 0; i < dbg->num_spheres; i++) {
                const float *s = reinterpret_cast<const float*>(&dbg->spheres[i]);
                const float *mat_col = s + 4;   // DebugSphere color[3]
                float fill[3];
                if (flags.show_visibility && cam_eye) {
                    // Front point of the sphere as seen from the camera
                    float fp[3];
                    float dx = s[0]-cam_eye[0], dy = s[1]-cam_eye[1], dz = s[2]-cam_eye[2];
                    float len = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (len > 1e-6f) {
                        fp[0] = s[0] + dx/len*s[3];
                        fp[1] = s[1] + dy/len*s[3];
                        fp[2] = s[2] + dz/len*s[3];
                    } else {
                        fp[0] = s[0]; fp[1] = s[1]; fp[2] = s[2];
                    }
                    shade_by_visibility(fill, mat_col,
                                        point_visible_from(fp, cam_eye, dbg));
                } else {
                    fill[0] = mat_col[0]; fill[1] = mat_col[1]; fill[2] = mat_col[2];
                }
                draw_sphere_solid(s, s[3], fill);
                if (flags.show_wireframe) {
                    draw_sphere_wireframe(s, s[3],
                        flags.show_visibility ? edge_vis : mat_col);
                }
            }
        }

        // Quads — solid + wireframe edges.  Visibility: quad centre.
        if (dbg->num_quads > 0 && dbg->quads) {
            for (int i = 0; i < dbg->num_quads; i++) {
                const float *q = reinterpret_cast<const float*>(&dbg->quads[i]);
                const float *mat_col = q + 9;   // DebugQuad color[3]
                float fill[3];
                if (flags.show_visibility && cam_eye) {
                    float c[3] = {
                        q[0] + 0.5f*q[3] + 0.5f*q[6],
                        q[1] + 0.5f*q[4] + 0.5f*q[7],
                        q[2] + 0.5f*q[5] + 0.5f*q[8],
                    };
                    shade_by_visibility(fill, mat_col,
                                        point_visible_from(c, cam_eye, dbg));
                } else {
                    fill[0] = mat_col[0]; fill[1] = mat_col[1]; fill[2] = mat_col[2];
                }
                draw_quad_solid(q, q+3, q+6, fill);
                if (flags.show_wireframe) {
                    draw_quad_wireframe(q, q+3, q+6,
                        flags.show_visibility ? edge_vis : mat_col);
                }
            }
        }

        // Boxes — solid with PER-FACE visibility shading + wireframe.
        if (dbg->num_boxes > 0 && dbg->boxes) {
            for (int i = 0; i < dbg->num_boxes; i++) {
                const float *b = reinterpret_cast<const float*>(&dbg->boxes[i]);
                const float *mn = b, *mx = b + 3;
                const float *mat_col = b + 6;   // DebugBox color[3]
                float face_cols[6][3];
                if (flags.show_visibility && cam_eye) {
                    // Face centres (see draw_box_solid face order)
                    float centers[6][3] = {
                        {mn[0], 0.5f*(mn[1]+mx[1]), 0.5f*(mn[2]+mx[2])},
                        {mx[0], 0.5f*(mn[1]+mx[1]), 0.5f*(mn[2]+mx[2])},
                        {0.5f*(mn[0]+mx[0]), mn[1], 0.5f*(mn[2]+mx[2])},
                        {0.5f*(mn[0]+mx[0]), mx[1], 0.5f*(mn[2]+mx[2])},
                        {0.5f*(mn[0]+mx[0]), 0.5f*(mn[1]+mx[1]), mn[2]},
                        {0.5f*(mn[0]+mx[0]), 0.5f*(mn[1]+mx[1]), mx[2]},
                    };
                    for (int f = 0; f < 6; f++) {
                        shade_by_visibility(face_cols[f], mat_col,
                                            point_visible_from(centers[f], cam_eye, dbg));
                    }
                } else {
                    for (int f = 0; f < 6; f++) {
                        face_cols[f][0] = mat_col[0];
                        face_cols[f][1] = mat_col[1];
                        face_cols[f][2] = mat_col[2];
                    }
                }
                draw_box_solid(mn, mx, face_cols);
                if (flags.show_wireframe) {
                    draw_box_wireframe(mn, mx,
                        flags.show_visibility ? edge_vis : mat_col);
                }
            }
        }

        glDisable(GL_POLYGON_OFFSET_FILL);

        // AABBs
        if (flags.show_aabbs && dbg->num_aabbs > 0 && dbg->aabb_data) {
            for (int i = 0; i < dbg->num_aabbs; i++) {
                const float *aabb = dbg->aabb_data + i * 6;
                draw_box_wireframe(aabb, aabb + 3, aabb_color);
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
        // Cross on the other two axes so the eye is easy to spot
        float cv2[6] = {
            cam_eye[0], cam_eye[1]-cross_sz, cam_eye[2],
            cam_eye[0], cam_eye[1]+cross_sz, cam_eye[2]
        };
        draw_lines(cv2, 2, ccol, 2.0f);

        // View direction: line from the eye along forward with an
        // arrowhead, plus the camera up vector (short, cyan).
        if (cam_at) {
            float fwd[3] = {cam_at[0]-cam_eye[0], cam_at[1]-cam_eye[1],
                            cam_at[2]-cam_eye[2]};
            float len = sqrtf(fwd[0]*fwd[0]+fwd[1]*fwd[1]+fwd[2]*fwd[2]);
            if (len > 1e-6f) {
                float inv = 1.f/len;
                fwd[0] *= inv; fwd[1] *= inv; fwd[2] *= inv;
                float arrow_len = 2.5f;
                float tip[3] = {cam_eye[0]+fwd[0]*arrow_len,
                                cam_eye[1]+fwd[1]*arrow_len,
                                cam_eye[2]+fwd[2]*arrow_len};
                float shaft[6] = {cam_eye[0], cam_eye[1], cam_eye[2],
                                  tip[0], tip[1], tip[2]};
                draw_lines(shaft, 2, ccol, 2.0f);
                // Arrowhead: two short segments at ±~30° from the shaft
                float up_dir[3] = {0.f, 1.f, 0.f};
                float side[3];
                side[0] = fwd[1]*up_dir[2] - fwd[2]*up_dir[1];
                side[1] = fwd[2]*up_dir[0] - fwd[0]*up_dir[2];
                side[2] = fwd[0]*up_dir[1] - fwd[1]*up_dir[0];
                float sl = sqrtf(side[0]*side[0]+side[1]*side[1]+side[2]*side[2]);
                if (sl > 1e-6f) {
                    side[0] /= sl; side[1] /= sl; side[2] /= sl;
                    float up2[3];
                    up2[0] = side[1]*fwd[2] - side[2]*fwd[1];
                    up2[1] = side[2]*fwd[0] - side[0]*fwd[2];
                    up2[2] = side[0]*fwd[1] - side[1]*fwd[0];
                    float back = arrow_len * 0.35f, wing = 0.18f;
                    float b0[3] = {tip[0]-fwd[0]*back + up2[0]*wing,
                                   tip[1]-fwd[1]*back + up2[1]*wing,
                                   tip[2]-fwd[2]*back + up2[2]*wing};
                    float b1[3] = {tip[0]-fwd[0]*back - up2[0]*wing,
                                   tip[1]-fwd[1]*back - up2[1]*wing,
                                   tip[2]-fwd[2]*back - up2[2]*wing};
                    float w1[6] = {tip[0], tip[1], tip[2], b0[0], b0[1], b0[2]};
                    float w2[6] = {tip[0], tip[1], tip[2], b1[0], b1[1], b1[2]};
                    draw_lines(w1, 2, ccol, 2.0f);
                    draw_lines(w2, 2, ccol, 2.0f);
                }
                // Camera up vector (short cyan line from the eye)
                float ucol[3] = {0.4f, 1.0f, 1.0f};
                float ulen2 = 0.8f;
                float uv[6] = {cam_eye[0], cam_eye[1], cam_eye[2],
                               cam_eye[0]+cam_up[0]*ulen2,
                               cam_eye[1]+cam_up[1]*ulen2,
                               cam_eye[2]+cam_up[2]*ulen2};
                draw_lines(uv, 2, ucol, 1.5f);
            }
        }
    }

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();

    ImGui::EndGroup();
}
