#include "scene_debug.h"
#include "gl_loader.h"   // pulls in <GLFW/glfw3.h> + missing GL 3.x+ declarations

#include "imgui.h"
#include "imgui_internal.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ── OpenGL shader helpers ──────────────────────────────────────────────

static GLuint compile_shader(GLenum type, const char *source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if ( !ok ) {
        char buf[512];
        glGetShaderInfoLog(shader, sizeof(buf), nullptr, buf);
        fprintf(stderr, "[scene_debug] shader compile error:\n%s\n", buf);
    }
    return shader;
}

static GLuint create_shader_program(const char *vs_src, const char *fs_src) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if ( !ok ) {
        char buf[512];
        glGetProgramInfoLog(prog, sizeof(buf), nullptr, buf);
        fprintf(stderr, "[scene_debug] program link error:\n%s\n", buf);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

static const char *kVS = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

static const char *kFS = R"(
#version 330 core
uniform vec3 uColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(uColor, 1.0);
}
)";

// ── Shader program + uniform locations ─────────────────────────────────

static GLuint g_shader = 0;
static GLint g_uMVP = -1;
static GLint g_uColor = -1;

static void ensure_shader() {
    if ( g_shader ) {
        return;
    }
    g_shader = create_shader_program(kVS, kFS);
    g_uMVP = glGetUniformLocation(g_shader, "uMVP");
    g_uColor = glGetUniformLocation(g_shader, "uColor");
}

// ── Debug orbit camera (mirrors OrbitCam from main.cpp) ─────────────────

struct DebugCam {
    float theta = 1.8f;     // azimuth
    float phi = 0.6f;       // elevation
    float dist = 15.f;      // distance from target
    float roll = 0.f;       // rotation of the up vector around forward
    float target[3] = {0, 0, 0};
};

static DebugCam g_debug_cam;

/// Rotate the default up vector {0,1,0} around the forward (lookat) axis
/// by the camera's roll angle.  Mirrors orbit_up() in main.cpp.
static void debug_cam_orbit_up(float theta, float phi, float roll, float up[3]) {
    float cos_t = cosf(theta), sin_t = sinf(theta);
    float cos_p = cosf(phi), sin_p = sinf(phi);
    float forward[3] = {-cos_p * sin_t, -sin_p, -cos_p * cos_t};
    float default_up[3] = {0.f, 1.f, 0.f};
    float cos_r = cosf(roll), sin_r = sinf(roll);
    float dot = forward[0] * default_up[0] + forward[1] * default_up[1] + forward[2] * default_up[2];
    up[0] = default_up[0] * cos_r +
            (forward[1] * default_up[2] - forward[2] * default_up[1]) * sin_r +
            forward[0] * dot * (1.f - cos_r);
    up[1] = default_up[1] * cos_r +
            (forward[2] * default_up[0] - forward[0] * default_up[2]) * sin_r +
            forward[1] * dot * (1.f - cos_r);
    up[2] = default_up[2] * cos_r +
            (forward[0] * default_up[1] - forward[1] * default_up[0]) * sin_r +
            forward[2] * dot * (1.f - cos_r);
}

static void debug_cam_get_view(float view[16], float proj[16], float fb_w, float fb_h) {
    float cos_t = cosf(g_debug_cam.theta), sin_t = sinf(g_debug_cam.theta);
    float cos_p = cosf(g_debug_cam.phi), sin_p = sinf(g_debug_cam.phi);

    float eye[3] = {
        g_debug_cam.target[0] + g_debug_cam.dist * cos_p * sin_t,
        g_debug_cam.target[1] + g_debug_cam.dist * sin_p,
        g_debug_cam.target[2] + g_debug_cam.dist * cos_p * cos_t,
    };

    float up[3];
    debug_cam_orbit_up(g_debug_cam.theta, g_debug_cam.phi, g_debug_cam.roll, up);
    float fwd[3] = {g_debug_cam.target[0] - eye[0],
                    g_debug_cam.target[1] - eye[1],
                    g_debug_cam.target[2] - eye[2]};
    float fwd_len = sqrtf(fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]);
    if ( fwd_len > 1e-8f ) {
        fwd[0] /= fwd_len;
        fwd[1] /= fwd_len;
        fwd[2] /= fwd_len;
    }

    // right = normalize(cross(up, fwd))
    float right[3] = {up[1]*fwd[2] - up[2]*fwd[1],
                      up[2]*fwd[0] - up[0]*fwd[2],
                      up[0]*fwd[1] - up[1]*fwd[0]};
    float rlen = sqrtf(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    if ( rlen > 1e-8f ) {
        right[0] /= rlen; right[1] /= rlen; right[2] /= rlen;
    }
    // cam_up = cross(fwd, right)
    float cam_up[3] = {fwd[1]*right[2] - fwd[2]*right[1],
                       fwd[2]*right[0] - fwd[0]*right[2],
                       fwd[0]*right[1] - fwd[1]*right[0]};

    // View matrix (look-at)
    view[0] = right[0]; view[1] = cam_up[0]; view[2] = -fwd[0]; view[3] = 0;
    view[4] = right[1]; view[5] = cam_up[1]; view[6] = -fwd[1]; view[7] = 0;
    view[8] = right[2]; view[9] = cam_up[2]; view[10] = -fwd[2]; view[11] = 0;
    view[12] = -(right[0]*eye[0] + right[1]*eye[1] + right[2]*eye[2]);
    view[13] = -(cam_up[0]*eye[0] + cam_up[1]*eye[1] + cam_up[2]*eye[2]);
    view[14] = fwd[0]*eye[0] + fwd[1]*eye[1] + fwd[2]*eye[2];
    view[15] = 1;

    // Perspective projection
    float aspect = fb_w / fb_h;
    float fov_y = 45.f * 3.14159265f / 180.f;
    float f = 1.f / tanf(fov_y * 0.5f);
    float znear = 0.1f, zfar = 200.f;
    proj[0] = f / aspect; proj[1] = 0; proj[2] = 0; proj[3] = 0;
    proj[4] = 0; proj[5] = f; proj[6] = 0; proj[7] = 0;
    proj[8] = 0; proj[9] = 0; proj[10] = (zfar+znear)/(znear-zfar); proj[11] = -1;
    proj[12] = 0; proj[13] = 0; proj[14] = 2*zfar*znear/(znear-zfar); proj[15] = 0;
}

/// Multiply two 4×4 matrices: out = a * b
static void mat_mul(const float a[16], const float b[16], float out[16]) {
    for ( int i = 0; i < 4; i++ ) {
        for ( int j = 0; j < 4; j++ ) {
            out[i*4 + j] = 0;
            for ( int k = 0; k < 4; k++ ) {
                out[i*4 + j] += a[i*4 + k] * b[k*4 + j];
            }
        }
    }
}

// ── FBO for offscreen 3D rendering ─────────────────────────────────────

static GLuint g_fbo = 0;
static GLuint g_fbo_tex = 0;
static int g_fbo_w = 0;
static int g_fbo_h = 0;

static void destroy_fbo() {
    if ( g_fbo ) {
        glDeleteFramebuffers(1, &g_fbo);
        g_fbo = 0;
    }
    if ( g_fbo_tex ) {
        glDeleteTextures(1, &g_fbo_tex);
        g_fbo_tex = 0;
    }
    g_fbo_w = 0;
    g_fbo_h = 0;
}

static void ensure_fbo(int w, int h) {
    if ( w == g_fbo_w && h == g_fbo_h && g_fbo ) {
        return;
    }
    destroy_fbo();

    glGenFramebuffers(1, &g_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);

    glGenTextures(1, &g_fbo_tex);
    glBindTexture(GL_TEXTURE_2D, g_fbo_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_fbo_tex, 0);

    GLuint rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);

    if ( glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE ) {
        fprintf(stderr, "[scene_debug] FBO not complete!\n");
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    g_fbo_w = w;
    g_fbo_h = h;
}

// ── Geometry builders ──────────────────────────────────────────────────

/// Push vertices for a wireframe box (12 edges) into the given vectors.
static void build_box_wireframe(float min_x, float min_y, float min_z,
                                float max_x, float max_y, float max_z,
                                std::vector<float> &verts) {
    // 8 corners
    float c[8][3] = {
        {min_x, min_y, min_z}, // 0
        {max_x, min_y, min_z}, // 1
        {max_x, max_y, min_z}, // 2
        {min_x, max_y, min_z}, // 3
        {min_x, min_y, max_z}, // 4
        {max_x, min_y, max_z}, // 5
        {max_x, max_y, max_z}, // 6
        {min_x, max_y, max_z}, // 7
    };
    int edges[12][2] = {
        {0,1},{1,2},{2,3},{3,0}, // bottom
        {4,5},{5,6},{6,7},{7,4}, // top
        {0,4},{1,5},{2,6},{3,7}, // vertical
    };
    for ( int e = 0; e < 12; e++ ) {
        verts.push_back(c[edges[e][0]][0]);
        verts.push_back(c[edges[e][0]][1]);
        verts.push_back(c[edges[e][0]][2]);
        verts.push_back(c[edges[e][1]][0]);
        verts.push_back(c[edges[e][1]][1]);
        verts.push_back(c[edges[e][1]][2]);
    }
}

/// Build a quad as two triangles (for wireframe, we draw the 4 edges)
static void build_quad_wireframe(const float base[3], const float eu[3], const float ev[3],
                                 std::vector<float> &verts) {
    float c[4][3] = {
        {base[0],          base[1],          base[2]},
        {base[0]+eu[0],    base[1]+eu[1],    base[2]+eu[2]},
        {base[0]+eu[0]+ev[0], base[1]+eu[1]+ev[1], base[2]+eu[2]+ev[2]},
        {base[0]+ev[0],    base[1]+ev[1],    base[2]+ev[2]},
    };
    int edges[4][2] = {{0,1},{1,2},{2,3},{3,0}};
    for ( int e = 0; e < 4; e++ ) {
        verts.push_back(c[edges[e][0]][0]);
        verts.push_back(c[edges[e][0]][1]);
        verts.push_back(c[edges[e][0]][2]);
        verts.push_back(c[edges[e][1]][0]);
        verts.push_back(c[edges[e][1]][1]);
        verts.push_back(c[edges[e][1]][2]);
    }
}

/// Build a wireframe sphere as latitude/longitude line loops.
static void build_sphere_wireframe(float cx, float cy, float cz, float r,
                                   int segments,
                                   std::vector<float> &verts) {
    // Longitude lines (meridians)
    for ( int i = 0; i < segments; i++ ) {
        float theta = 2.f * 3.14159265f * i / segments;
        float sin_t = sinf(theta), cos_t = cosf(theta);
        // Each meridian is a line loop from south to north pole
        for ( int j = 0; j <= segments; j++ ) {
            float phi = 3.14159265f * j / segments - 3.14159265f * 0.5f;
            float sin_p = sinf(phi), cos_p = cosf(phi);
            verts.push_back(cx + r * cos_p * cos_t);
            verts.push_back(cy + r * sin_p);
            verts.push_back(cz + r * cos_p * sin_t);
            // Connect to next point on same meridian
            if ( j < segments ) {
                float phi2 = 3.14159265f * (j+1) / segments - 3.14159265f * 0.5f;
                float sin_p2 = sinf(phi2), cos_p2 = cosf(phi2);
                verts.push_back(cx + r * cos_p2 * cos_t);
                verts.push_back(cy + r * sin_p2);
                verts.push_back(cz + r * cos_p2 * sin_t);
            }
        }
    }

    // Latitude lines (parallels)
    for ( int j = 1; j < segments; j++ ) {
        float phi = 3.14159265f * j / segments - 3.14159265f * 0.5f;
        float sin_p = sinf(phi), cos_p = cosf(phi);
        for ( int i = 0; i <= segments; i++ ) {
            float theta = 2.f * 3.14159265f * i / segments;
            float sin_t = sinf(theta), cos_t = cosf(theta);
            verts.push_back(cx + r * cos_p * cos_t);
            verts.push_back(cy + r * sin_p);
            verts.push_back(cz + r * cos_p * sin_t);
            if ( i < segments ) {
                float theta2 = 2.f * 3.14159265f * (i+1) / segments;
                float sin_t2 = sinf(theta2), cos_t2 = cosf(theta2);
                verts.push_back(cx + r * cos_p * cos_t2);
                verts.push_back(cy + r * sin_p);
                verts.push_back(cz + r * cos_p * sin_t2);
            }
        }
    }
}

/// Build a grid floor on the XZ plane (Y=0) with the given extent and subdivisions.
static void build_grid_floor(float extent, int divisions, std::vector<float> &verts) {
    float half = extent * 0.5f;
    float step = extent / divisions;

    // Lines along X
    for ( int i = 0; i <= divisions; i++ ) {
        float z = -half + i * step;
        verts.push_back(-half); verts.push_back(0); verts.push_back(z);
        verts.push_back(half);  verts.push_back(0); verts.push_back(z);
    }
    // Lines along Z
    for ( int i = 0; i <= divisions; i++ ) {
        float x = -half + i * step;
        verts.push_back(x); verts.push_back(0); verts.push_back(-half);
        verts.push_back(x); verts.push_back(0); verts.push_back(half);
    }
}

// ── Rendering helpers ──────────────────────────────────────────────────

/// Draw a collection of line segments.
static void draw_lines(const std::vector<float> &verts,
                       const float color[3],
                       float mvp[16],
                       float line_width = 1.5f) {
    if ( verts.empty() ) {
        return;
    }
    glUseProgram(g_shader);
    glUniformMatrix4fv(g_uMVP, 1, GL_FALSE, mvp);
    glUniform3f(g_uColor, color[0], color[1], color[2]);

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STREAM_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    glLineWidth(line_width);
    glDrawArrays(GL_LINES, 0, (GLsizei)(verts.size() / 3));

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}

// ── Camera frustum computation ─────────────────────────────────────────

static void compute_frustum_corners(const float eye[3],
                                    const float at[3],
                                    const float up[3],
                                    float vfov_deg,
                                    float aspect,
                                    float focus_dist,
                                    float corners[4][3]) {
    float fwd[3] = {at[0] - eye[0], at[1] - eye[1], at[2] - eye[2]};
    float fwd_len = sqrtf(fwd[0]*fwd[0] + fwd[1]*fwd[1] + fwd[2]*fwd[2]);
    if ( fwd_len > 1e-6f ) {
        fwd[0] /= fwd_len; fwd[1] /= fwd_len; fwd[2] /= fwd_len;
    }

    float right[3] = {up[1]*fwd[2] - up[2]*fwd[1],
                      up[2]*fwd[0] - up[0]*fwd[2],
                      up[0]*fwd[1] - up[1]*fwd[0]};
    float rlen = sqrtf(right[0]*right[0] + right[1]*right[1] + right[2]*right[2]);
    if ( rlen > 1e-6f ) {
        right[0] /= rlen; right[1] /= rlen; right[2] /= rlen;
    }

    float cup[3] = {fwd[1]*right[2] - fwd[2]*right[1],
                    fwd[2]*right[0] - fwd[0]*right[2],
                    fwd[0]*right[1] - fwd[1]*right[0]};

    float theta = vfov_deg * 3.14159265f / 180.f;
    float half_h = tanf(theta * 0.5f);
    float half_w = aspect * half_h;

    float vc[3] = {eye[0] + fwd[0] * focus_dist,
                   eye[1] + fwd[1] * focus_dist,
                   eye[2] + fwd[2] * focus_dist};

    corners[0][0] = vc[0] - half_w * right[0] - half_h * cup[0];
    corners[0][1] = vc[1] - half_w * right[1] - half_h * cup[1];
    corners[0][2] = vc[2] - half_w * right[2] - half_h * cup[2];
    corners[1][0] = vc[0] + half_w * right[0] - half_h * cup[0];
    corners[1][1] = vc[1] + half_w * right[1] - half_h * cup[1];
    corners[1][2] = vc[2] + half_w * right[2] - half_h * cup[2];
    corners[2][0] = vc[0] + half_w * right[0] + half_h * cup[0];
    corners[2][1] = vc[1] + half_w * right[1] + half_h * cup[1];
    corners[2][2] = vc[2] + half_w * right[2] + half_h * cup[2];
    corners[3][0] = vc[0] - half_w * right[0] + half_h * cup[0];
    corners[3][1] = vc[1] - half_w * right[1] + half_h * cup[1];
    corners[3][2] = vc[2] - half_w * right[2] + half_h * cup[2];
}

// ── Public API ─────────────────────────────────────────────────────────

void init_scene_debug() {
    ensure_shader();
}

void shutdown_scene_debug() {
    destroy_fbo();
    if ( g_shader ) {
        glDeleteProgram(g_shader);
        g_shader = 0;
    }
}

void render_scene_debug(const SceneDebugInfo *dbg,
                        const float *cam_eye,
                        const float *cam_at,
                        const float *cam_up,
                        float cam_fov,
                        float aspect,
                        DebugViewFlags &flags) {
    // ── Compute focus distance from camera params ─────────────────────
    float focus_dist = 1.0f;
    if ( cam_eye && cam_at ) {
        float dx = cam_at[0] - cam_eye[0];
        float dy = cam_at[1] - cam_eye[1];
        float dz = cam_at[2] - cam_eye[2];
        focus_dist = sqrtf(dx * dx + dy * dy + dz * dz);
        if ( focus_dist < 1e-6f ) {
            focus_dist = 1.0f;
        }
    }

    // ── Controls panel at the top of the window ───────────────────────
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 2));
    if ( ImGui::CollapsingHeader("View Controls", ImGuiTreeNodeFlags_DefaultOpen) ) {
        ImGui::Checkbox("Floor", &flags.show_floor);
        ImGui::SameLine();
        ImGui::Checkbox("Grid", &flags.show_grid);
        ImGui::SameLine();
        ImGui::Checkbox("Objects", &flags.show_objects);
        ImGui::SameLine();
        ImGui::Checkbox("Wireframe", &flags.show_wireframe);

        ImGui::Checkbox("AABBs", &flags.show_aabbs);
        ImGui::SameLine();
        ImGui::Checkbox("Frustum", &flags.show_frustum);
        ImGui::SameLine();
        ImGui::Checkbox("Camera", &flags.show_camera);

        ImGui::Text("LMB drag = orbit  |  scroll = zoom");
        ImGui::Text("Ctrl+scroll = aperture  |  Ctrl+Shift+scroll = FOV");
        ImGui::Text("Ctrl+Alt+scroll = roll  |  WASD = move camera");
        ImGui::Text("Arrows = orbit  |  Shift+arrows = pan target");
        ImGui::Text("Q/E = up/down");
    }
    ImGui::PopStyleVar();

    // ── Determine the content region size for the 3D viewport ─────────
    ImVec2 region = ImGui::GetContentRegionAvail();
    int fb_w = (int)region.x;
    int fb_h = (int)region.y;
    if ( fb_w < 16 || fb_h < 16 ) {
        return;
    }

    // ── Ensure FBO matches the viewport size ──────────────────────────
    ensure_fbo(fb_w, fb_h);

    // ── Handle mouse + keyboard interaction on the 3D viewport ───────
    ImVec2 cursor_pos = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##debug_viewport", region, ImGuiButtonFlags_MouseButtonLeft |
                           ImGuiButtonFlags_MouseButtonMiddle | ImGuiButtonFlags_MouseButtonRight);
    bool hovered = ImGui::IsItemHovered();
    auto &io = ImGui::GetIO();

    if ( hovered ) {
        // Pan target (left drag) — mirrors main.cpp LMB behaviour
        if ( ImGui::IsMouseDragging(ImGuiMouseButton_Left) ) {
            ImVec2 drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
            float ct = cosf(g_debug_cam.theta), st = sinf(g_debug_cam.theta);
            float cp = cosf(g_debug_cam.phi), sp = sinf(g_debug_cam.phi);
            float rgt[3] = {ct, 0.f, -st};
            float udir[3] = {-sp*st, cp, -sp*ct};
            float speed = g_debug_cam.dist * 0.002f;
            g_debug_cam.target[0] += (-drag.x * rgt[0] + drag.y * udir[0]) * speed;
            g_debug_cam.target[1] += (-drag.x * rgt[1] + drag.y * udir[1]) * speed;
            g_debug_cam.target[2] += (-drag.x * rgt[2] + drag.y * udir[2]) * speed;
            ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
        }
        // Orbit (middle drag) — mirrors main.cpp MMB behaviour
        if ( ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ) {
            ImVec2 drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle);
            g_debug_cam.theta -= drag.x * 0.005f;
            g_debug_cam.phi += drag.y * 0.005f;
            g_debug_cam.phi = std::max(-1.5f, std::min(1.5f, g_debug_cam.phi));
            ImGui::ResetMouseDragDelta(ImGuiMouseButton_Middle);
        }
        // Zoom / roll (scroll)
        float scroll = io.MouseWheel;
        if ( scroll != 0.0f ) {
            bool ctrl = io.KeyCtrl;
            bool shift = io.KeyShift;
            bool alt = io.KeyAlt;
            if ( ctrl && alt ) {
                g_debug_cam.roll += scroll * 0.05f;
            } else {
                g_debug_cam.dist *= (scroll > 0) ? 0.9f : 1.1f;
                g_debug_cam.dist = std::max(0.5f, std::min(200.f, g_debug_cam.dist));
            }
        }
    }

    // ── Keyboard camera controls (active when viewport hovered or unfocused) ──
    if ( !io.WantCaptureKeyboard ) {
        float kspeed = 0.04f;
        float tspeed = 0.3f;
        bool shift = io.KeyShift;

        // Arrow keys: orbit (no shift) / pan target (shift)
        if ( !shift && ImGui::IsKeyDown(ImGuiKey_LeftArrow) )  g_debug_cam.theta -= kspeed;
        if ( !shift && ImGui::IsKeyDown(ImGuiKey_RightArrow) ) g_debug_cam.theta += kspeed;
        if ( !shift && ImGui::IsKeyDown(ImGuiKey_UpArrow) ) {
            g_debug_cam.phi += kspeed;
            g_debug_cam.phi = std::min(1.5f, g_debug_cam.phi);
        }
        if ( !shift && ImGui::IsKeyDown(ImGuiKey_DownArrow) ) {
            g_debug_cam.phi -= kspeed;
            g_debug_cam.phi = std::max(-1.5f, g_debug_cam.phi);
        }
        if ( shift && ImGui::IsKeyDown(ImGuiKey_LeftArrow) ) {
            g_debug_cam.target[0] -= 0.2f * cosf(g_debug_cam.theta);
            g_debug_cam.target[2] -= 0.2f * sinf(g_debug_cam.theta);
        }
        if ( shift && ImGui::IsKeyDown(ImGuiKey_RightArrow) ) {
            g_debug_cam.target[0] += 0.2f * cosf(g_debug_cam.theta);
            g_debug_cam.target[2] += 0.2f * sinf(g_debug_cam.theta);
        }
        if ( shift && ImGui::IsKeyDown(ImGuiKey_UpArrow) )    g_debug_cam.target[1] += 0.2f;
        if ( shift && ImGui::IsKeyDown(ImGuiKey_DownArrow) )  g_debug_cam.target[1] -= 0.2f;

        // WASD: move camera (translate target) — mirrors main.cpp
        float ct = cosf(g_debug_cam.theta), st = sinf(g_debug_cam.theta);
        float cp = cosf(g_debug_cam.phi),   sp = sinf(g_debug_cam.phi);
        float forward[3] = {cp * st, sp, cp * ct};
        float right[3] = {ct, 0.f, -st};

        if ( ImGui::IsKeyDown(ImGuiKey_W) ) {
            g_debug_cam.target[0] += forward[0] * tspeed;
            g_debug_cam.target[1] += forward[1] * tspeed;
            g_debug_cam.target[2] += forward[2] * tspeed;
        }
        if ( ImGui::IsKeyDown(ImGuiKey_S) ) {
            g_debug_cam.target[0] -= forward[0] * tspeed;
            g_debug_cam.target[1] -= forward[1] * tspeed;
            g_debug_cam.target[2] -= forward[2] * tspeed;
        }
        if ( ImGui::IsKeyDown(ImGuiKey_A) ) {
            g_debug_cam.target[0] -= right[0] * tspeed;
            g_debug_cam.target[1] -= right[1] * tspeed;
            g_debug_cam.target[2] -= right[2] * tspeed;
        }
        if ( ImGui::IsKeyDown(ImGuiKey_D) ) {
            g_debug_cam.target[0] += right[0] * tspeed;
            g_debug_cam.target[1] += right[1] * tspeed;
            g_debug_cam.target[2] += right[2] * tspeed;
        }
        if ( ImGui::IsKeyDown(ImGuiKey_Q) ) g_debug_cam.target[1] -= 0.3f;
        if ( ImGui::IsKeyDown(ImGuiKey_E) ) g_debug_cam.target[1] += 0.3f;
    }

    // ── Compute MVP matrix ────────────────────────────────────────────
    float view[16], proj[16], mvp[16];
    debug_cam_get_view(view, proj, (float)fb_w, (float)fb_h);
    mat_mul(proj, view, mvp);

    // ── Render 3D scene to FBO ────────────────────────────────────────
    glBindFramebuffer(GL_FRAMEBUFFER, g_fbo);
    glViewport(0, 0, fb_w, fb_h);

    // Dark background
    glClearColor(0.12f, 0.12f, 0.14f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    // ── Floor (large grid) ────────────────────────────────────────────
    if ( flags.show_floor ) {
        std::vector<float> floor_verts;
        build_grid_floor(20.f, 20, floor_verts);
        float floor_color[3] = {0.25f, 0.25f, 0.3f};
        draw_lines(floor_verts, floor_color, mvp, 1.f);
    }

    // ── Small reference grid ──────────────────────────────────────────
    if ( flags.show_grid ) {
        std::vector<float> grid_verts;
        build_grid_floor(12.f, 12, grid_verts);
        float grid_color[3] = {0.35f, 0.35f, 0.4f};
        draw_lines(grid_verts, grid_color, mvp, 0.8f);
    }

    // ── Scene objects ─────────────────────────────────────────────────
    if ( dbg && flags.show_objects ) {
        float line_w = flags.show_wireframe ? 1.5f : 2.f;

        // Spheres
        for ( int i = 0; i < dbg->num_spheres; i++ ) {
            std::vector<float> sv;
            build_sphere_wireframe(dbg->spheres[i].center[0],
                                   dbg->spheres[i].center[1],
                                   dbg->spheres[i].center[2],
                                   dbg->spheres[i].radius,
                                   16, sv);
            draw_lines(sv, dbg->spheres[i].color, mvp, line_w);
        }

        // Quads
        for ( int i = 0; i < dbg->num_quads; i++ ) {
            std::vector<float> qv;
            build_quad_wireframe(dbg->quads[i].base,
                                 dbg->quads[i].edge_u,
                                 dbg->quads[i].edge_v,
                                 qv);
            draw_lines(qv, dbg->quads[i].color, mvp, line_w);
        }

        // Boxes
        for ( int i = 0; i < dbg->num_boxes; i++ ) {
            std::vector<float> bv;
            build_box_wireframe(dbg->boxes[i].box_min[0],
                                dbg->boxes[i].box_min[1],
                                dbg->boxes[i].box_min[2],
                                dbg->boxes[i].box_max[0],
                                dbg->boxes[i].box_max[1],
                                dbg->boxes[i].box_max[2],
                                bv);
            draw_lines(bv, dbg->boxes[i].color, mvp, line_w);
        }
    }

    // ── AABBs (optional overlay, red) ─────────────────────────────────
    if ( dbg && flags.show_aabbs && dbg->aabb_data && dbg->num_aabbs > 0 ) {
        float aabb_col[3] = {1.f, 0.3f, 0.3f};
        for ( int i = 0; i < dbg->num_aabbs; i++ ) {
            const float *box = dbg->aabb_data + i * 6;
            std::vector<float> av;
            build_box_wireframe(box[0], box[1], box[2],
                                box[3], box[4], box[5],
                                av);
            draw_lines(av, aabb_col, mvp, 1.f);
        }
    }

    // ── Camera frustum ────────────────────────────────────────────────
    if ( cam_eye && cam_at && cam_up && flags.show_frustum ) {
        float corners[4][3];
        compute_frustum_corners(cam_eye, cam_at, cam_up, cam_fov, aspect, focus_dist, corners);

        float frustum_col[3] = {1.f, 0.8f, 0.2f};
        std::vector<float> fv;

        // Eye to 4 corners
        for ( int c = 0; c < 4; c++ ) {
            fv.push_back(cam_eye[0]); fv.push_back(cam_eye[1]); fv.push_back(cam_eye[2]);
            fv.push_back(corners[c][0]); fv.push_back(corners[c][1]); fv.push_back(corners[c][2]);
        }

        // Viewport rectangle
        for ( int c = 0; c < 4; c++ ) {
            int n = c, n2 = (c+1) % 4;
            fv.push_back(corners[n][0]);  fv.push_back(corners[n][1]);  fv.push_back(corners[n][2]);
            fv.push_back(corners[n2][0]); fv.push_back(corners[n2][1]); fv.push_back(corners[n2][2]);
        }

        draw_lines(fv, frustum_col, mvp, 1.5f);

        // Camera position marker (small cross)
        if ( flags.show_camera ) {
            float cam_mark_col[3] = {0.f, 1.f, 0.f};
            float s = 0.3f;
            std::vector<float> cv;
            for ( int axis = 0; axis < 3; axis++ ) {
                float p[3] = {cam_eye[0], cam_eye[1], cam_eye[2]};
                float off[3] = {0,0,0};
                off[axis] = s;
                cv.push_back(p[0]-off[0]); cv.push_back(p[1]-off[1]); cv.push_back(p[2]-off[2]);
                cv.push_back(p[0]+off[0]); cv.push_back(p[1]+off[1]); cv.push_back(p[2]+off[2]);
            }
            draw_lines(cv, cam_mark_col, mvp, 2.5f);

            // Look-at target marker (small diamond cross)
            float target_col[3] = {0.f, 0.8f, 0.2f};
            std::vector<float> tv;
            float ts = 0.2f;
            for ( int axis = 0; axis < 3; axis++ ) {
                float p[3] = {cam_at[0], cam_at[1], cam_at[2]};
                float off[3] = {0,0,0};
                off[axis] = ts;
                tv.push_back(p[0]-off[0]); tv.push_back(p[1]-off[1]); tv.push_back(p[2]-off[2]);
                tv.push_back(p[0]+off[0]); tv.push_back(p[1]+off[1]); tv.push_back(p[2]+off[2]);
            }
            draw_lines(tv, target_col, mvp, 2.f);
        }
    }

    // ── Axis indicators at origin ─────────────────────────────────────
    {
        float axis_len = 1.5f;
        float red[3] = {1, 0.2f, 0.2f};
        std::vector<float> ax = {0,0,0, axis_len,0,0};
        draw_lines(ax, red, mvp, 2.5f);
        float green[3] = {0.2f, 1, 0.2f};
        std::vector<float> ay = {0,0,0, 0,axis_len,0};
        draw_lines(ay, green, mvp, 2.5f);
        float blue[3] = {0.2f, 0.2f, 1};
        std::vector<float> az = {0,0,0, 0,0,axis_len};
        draw_lines(az, blue, mvp, 2.5f);
    }

    // ── Restore default framebuffer ──────────────────────────────────
    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // ── Display the FBO texture in the ImGui window ──────────────────
    ImGui::SetCursorScreenPos(cursor_pos);
    ImGui::Image((ImTextureID)(intptr_t)g_fbo_tex, region,
                 ImVec2(0, 1), ImVec2(1, 0));  // flip Y
}
