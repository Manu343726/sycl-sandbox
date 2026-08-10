// sycl-sandbox — Scene debug renderer
// Background-thread OpenGL renderer for the "Scene Debug" window.
//
// Renders the scene (from a host copy, read-only) with an independent
// orbit camera into a triple-buffered slot texture; the UI thread
// presents the newest ready slot via ImGui::Image.  See renderer.h for
// the architecture overview and threading contract.

#include "ui/scene_debug/renderer.h"
#include "ui/scene_debug/ray_trace.h"

#include "gl_loader.h"

#include <sycl-sandbox/rt/aabb.h>
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types.h>
#include <sycl-sandbox/scene/camera.h>
#include <sycl-sandbox/scene/data.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <utility>
#include <vector>
#include <pthread.h>

namespace {

constexpr float kPi = 3.14159265358979f;
constexpr int kMinSize = 64;
constexpr int kMaxSize = 4096;
constexpr uint64_t kFrameIntervalUs = 16000; // ~60 fps cap
constexpr float kFov = 45.f;                 // debug camera vertical FOV
constexpr float kZNear = 0.05f;
constexpr float kZFar = 2000.f;

// ── Interleaved vertex: position + flat color (alpha for translucent
//    overlays like the scene framebuffer window) ────────────────────
struct Vertex {
    float x, y, z;
    float r, g, b, a;
};

static void push_vertex(std::vector<Vertex> &v, rt::float3 p, rt::float3 c,
                        float a = 1.f) {
    v.push_back({p.x, p.y, p.z, c.x, c.y, c.z, a});
}

// ── Column-major 4x4 matrix (OpenGL convention) ──────────────────────
struct Mat4 {
    float m[16];
};

static Mat4 mat4_mul(const Mat4 &a, const Mat4 &b) {
    Mat4 r{};
    for ( int col = 0; col < 4; col++ ) {
        for ( int row = 0; row < 4; row++ ) {
            float sum = 0.f;
            for ( int k = 0; k < 4; k++ )
                sum += a.m[k * 4 + row] * b.m[col * 4 + k];
            r.m[col * 4 + row] = sum;
        }
    }
    return r;
}

static Mat4 mat4_perspective(float fovy_deg, float aspect, float znear, float zfar) {
    float f = 1.f / std::tan(fovy_deg * kPi / 360.f);
    Mat4 m{};
    m.m[0] = f / aspect;
    m.m[5] = f;
    m.m[10] = (zfar + znear) / (znear - zfar);
    m.m[11] = -1.f;
    m.m[14] = 2.f * zfar * znear / (znear - zfar);
    return m;
}

static Mat4 mat4_look_at(rt::float3 eye, rt::float3 at, rt::float3 up) {
    rt::float3 f = rt::norm(rt::sub(at, eye));
    rt::float3 s = rt::norm(rt::cross(f, up));
    rt::float3 u = rt::cross(s, f);
    Mat4 m{};
    m.m[0] = s.x; m.m[4] = s.y; m.m[8] = s.z;
    m.m[1] = u.x; m.m[5] = u.y; m.m[9] = u.z;
    m.m[2] = -f.x; m.m[6] = -f.y; m.m[10] = -f.z;
    m.m[12] = -rt::dot(s, eye);
    m.m[13] = -rt::dot(u, eye);
    m.m[14] = rt::dot(f, eye);
    m.m[15] = 1.f;
    return m;
}

// ── Shader helper ────────────────────────────────────────────────────
static GLuint compile_shader(const char *vs_src, const char *fs_src) {
    auto compile = [](GLenum type, const char *src) -> GLuint {
        GLuint sh = glCreateShader(type);
        glShaderSource(sh, 1, &src, nullptr);
        glCompileShader(sh);
        GLint ok = 0;
        glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if ( !ok ) {
            char log[1024] = {};
            GLsizei len = 0;
            glGetShaderInfoLog(sh, sizeof(log), &len, log);
            spdlog::error("[scene-debug] shader compile failed: {}", log);
            glDeleteShader(sh);
            return 0;
        }
        return sh;
    };

    GLuint vs = compile(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fs_src);
    if ( !vs || !fs ) {
        if ( vs ) glDeleteShader(vs);
        if ( fs ) glDeleteShader(fs);
        return 0;
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if ( !ok ) {
        char log[1024] = {};
        GLsizei len = 0;
        glGetProgramInfoLog(prog, sizeof(log), &len, log);
        spdlog::error("[scene-debug] program link failed: {}", log);
        glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

// ── Material color for a handle (mirror of SceneBuilder's helper, but
//    reading straight from the host SceneView arrays) ─────────────────
static rt::float3 material_color(const rt::Handle &h, const rt::SceneView &v) {
    auto mtype = static_cast<rt::MaterialType>(rt::handle_tag(h.material));
    uint32_t midx = rt::handle_index(h.material);
    switch ( mtype ) {
        case rt::MaterialType::Lambertian:
            return v.lambertians[midx].albedo;
        case rt::MaterialType::Metal:
            return v.metals[midx].albedo;
        case rt::MaterialType::Dielectric:
            return {0.7f, 0.8f, 1.0f};
        case rt::MaterialType::DiffuseLight: {
            rt::float3 e = v.diffuse_lights[midx].emit_color;
            float max_c = std::fmax(std::fmax(e.x, e.y), e.z);
            if ( max_c > 1.f ) e = rt::scale(e, 1.f / max_c);
            return e;
        }
        case rt::MaterialType::TexturedLambertian: {
            rt::RNG rng{0xdeadbeefu};
            return rt::textures::sample(v.textured_lambertians[midx].texture,
                                        0.5f, 0.5f, 0.f, rng);
        }
    }
    return {1, 1, 1};
}

// ── Per-handle AABB (from SceneView or recomputed from the shape) ────
static rt::Aabb aabb_for_handle(const rt::SceneView &v, int i) {
    if ( v.aabbs ) return v.aabbs[i];
    const rt::Handle &h = v.handles[i];
    switch ( static_cast<rt::HittableType>(rt::handle_tag(h.hittable)) ) {
        case rt::HittableType::Sphere:
            return v.spheres[rt::handle_index(h.hittable)].aabb();
        case rt::HittableType::Triangle:
            return v.triangles[rt::handle_index(h.hittable)].aabb();
        case rt::HittableType::Quad:
            return v.quads[rt::handle_index(h.hittable)].aabb();
        case rt::HittableType::Box:
            return v.boxes[rt::handle_index(h.hittable)].aabb();
        case rt::HittableType::Portal:
            return v.portals[rt::handle_index(h.hittable)].aabb();
        case rt::HittableType::Mesh:
            return v.meshes[rt::handle_index(h.hittable)].aabb(v.triangles);
    }
    return {{0, 0, 0}, {0, 0, 0}};
}

// ── Visibility from the scene camera ──────────────────────────────────
//
// The old per-object representative-point raycasts were replaced by a
// pixel-accurate pass: the scene is rendered from the scene camera's
// POV into an offscreen depth buffer (the z-test result — the "stencil"
// of pixels the framebuffer sees), and the orbit view's solid pass
// shades every fragment against that buffer (see vis_fs_src in
// gl_init() and the camera-visibility pass in thread_main()).

// ── Primitive geometry builders ───────────────────────────────────────

static void add_box_solid(std::vector<Vertex> &v, rt::float3 lo, rt::float3 hi,
                          rt::float3 c) {
    // 6 faces, 12 triangles — axis aligned
    auto face = [&](rt::float3 a, rt::float3 b, rt::float3 cc, rt::float3 d) {
        push_vertex(v, a, c);
        push_vertex(v, b, c);
        push_vertex(v, cc, c);
        push_vertex(v, a, c);
        push_vertex(v, cc, c);
        push_vertex(v, d, c);
    };
    face({lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, lo.y, hi.z}, {lo.x, lo.y, hi.z}); // -Y
    face({lo.x, hi.y, lo.z}, {hi.x, hi.y, lo.z}, {hi.x, hi.y, hi.z}, {lo.x, hi.y, hi.z}); // +Y
    face({lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, hi.y, lo.z}, {lo.x, hi.y, lo.z}); // -Z
    face({lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z}, {hi.x, hi.y, hi.z}, {lo.x, hi.y, hi.z}); // +Z
    face({lo.x, lo.y, lo.z}, {lo.x, lo.y, hi.z}, {lo.x, hi.y, hi.z}, {lo.x, hi.y, lo.z}); // -X
    face({hi.x, lo.y, lo.z}, {hi.x, lo.y, hi.z}, {hi.x, hi.y, hi.z}, {hi.x, hi.y, lo.z}); // +X
}

static void add_box_wire(std::vector<Vertex> &v, rt::float3 lo, rt::float3 hi,
                         rt::float3 c, float a = 1.f) {
    rt::float3 corners[8] = {
        {lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, lo.y, hi.z}, {lo.x, lo.y, hi.z},
        {lo.x, hi.y, lo.z}, {hi.x, hi.y, lo.z}, {hi.x, hi.y, hi.z}, {lo.x, hi.y, hi.z},
    };
    const int edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };
    for ( const auto &e : edges ) {
        push_vertex(v, corners[e[0]], c, a);
        push_vertex(v, corners[e[1]], c, a);
    }
}

static void add_quad_solid(std::vector<Vertex> &v, rt::float3 base, rt::float3 eu,
                           rt::float3 ev, rt::float3 c, float a = 1.f) {
    rt::float3 b = base;
    rt::float3 d = rt::add(base, rt::add(eu, ev));
    push_vertex(v, b, c, a);
    push_vertex(v, rt::add(base, eu), c, a);
    push_vertex(v, d, c, a);
    push_vertex(v, b, c, a);
    push_vertex(v, d, c, a);
    push_vertex(v, rt::add(base, ev), c, a);
}

static void add_quad_wire(std::vector<Vertex> &v, rt::float3 base, rt::float3 eu,
                          rt::float3 ev, rt::float3 c) {
    rt::float3 pts[4] = {base, rt::add(base, eu), rt::add(base, rt::add(eu, ev)),
                         rt::add(base, ev)};
    for ( int i = 0; i < 4; i++ ) {
        push_vertex(v, pts[i], c);
        push_vertex(v, pts[(i + 1) % 4], c);
    }
}

static void add_tri_solid(std::vector<Vertex> &v, rt::float3 a, rt::float3 b,
                          rt::float3 cc, rt::float3 c) {
    push_vertex(v, a, c);
    push_vertex(v, b, c);
    push_vertex(v, cc, c);
}

static void add_tri_wire(std::vector<Vertex> &v, rt::float3 a, rt::float3 b,
                         rt::float3 cc, rt::float3 c) {
    push_vertex(v, a, c);
    push_vertex(v, b, c);
    push_vertex(v, b, c);
    push_vertex(v, cc, c);
    push_vertex(v, cc, c);
    push_vertex(v, a, c);
}

static void add_sphere_solid(std::vector<Vertex> &v, rt::float3 center, float r,
                             rt::float3 c) {
    constexpr int kLon = 16;
    constexpr int kLat = 12;
    for ( int lat = 0; lat < kLat; lat++ ) {
        float theta0 = lat * kPi / kLat;            // 0..pi
        float theta1 = (lat + 1) * kPi / kLat;
        for ( int lon = 0; lon < kLon; lon++ ) {
            float phi0 = lon * 2.f * kPi / kLon;
            float phi1 = (lon + 1) * 2.f * kPi / kLon;
            auto pt = [&](float th, float ph) {
                return rt::add(center, rt::scale(
                    {std::sin(th) * std::cos(ph), std::cos(th), std::sin(th) * std::sin(ph)},
                    r));
            };
            rt::float3 p00 = pt(theta0, phi0), p01 = pt(theta0, phi1);
            rt::float3 p10 = pt(theta1, phi0), p11 = pt(theta1, phi1);
            push_vertex(v, p00, c);
            push_vertex(v, p10, c);
            push_vertex(v, p11, c);
            push_vertex(v, p00, c);
            push_vertex(v, p11, c);
            push_vertex(v, p01, c);
        }
    }
}

static void add_sphere_wire(std::vector<Vertex> &v, rt::float3 center, float r,
                            rt::float3 c) {
    constexpr int kSegs = 32;
    auto ring = [&](rt::float3 axis_u, rt::float3 axis_v) {
        for ( int i = 0; i < kSegs; i++ ) {
            float a0 = i * 2.f * kPi / kSegs;
            float a1 = (i + 1) * 2.f * kPi / kSegs;
            push_vertex(v, rt::add(center, rt::scale(rt::add(rt::scale(axis_u, std::cos(a0)),
                                                             rt::scale(axis_v, std::sin(a0))), r)), c);
            push_vertex(v, rt::add(center, rt::scale(rt::add(rt::scale(axis_u, std::cos(a1)),
                                                             rt::scale(axis_v, std::sin(a1))), r)), c);
        }
    };
    ring({1, 0, 0}, {0, 1, 0});
    ring({1, 0, 0}, {0, 0, 1});
    ring({0, 0, 1}, {0, 1, 0});
}

static void add_floor(std::vector<Vertex> &v, float half) {
    // Reference plane, deliberately NOT at y=0: scene geometry often has
    // a floor quad exactly on y=0 and a coplanar opaque quad would
    // z-fight with it.  Drawn slightly below (+ translucent) it stays
    // out of the scene floor's depth range when viewed from above, and
    // tints rather than hides the scene when the camera is below it.
    rt::float3 lo{-half, -0.002f, -half};
    rt::float3 hi{half, -0.002f, half};
    rt::float3 c{0.11f, 0.12f, 0.15f};
    add_quad_solid(v, lo, {hi.x - lo.x, 0, 0}, {0, 0, hi.z - lo.z}, c, 0.15f);
}

static void add_grid(std::vector<Vertex> &v, float half, float step) {
    for ( float x = -half; x <= half + 0.01f; x += step ) {
        rt::float3 c = (std::fabs(x) < 0.01f)
                           ? rt::float3{0.35f, 0.35f, 0.42f}
                           : rt::float3{0.16f, 0.16f, 0.20f};
        push_vertex(v, {x, 0.001f, -half}, c);
        push_vertex(v, {x, 0.001f, half}, c);
        push_vertex(v, {-half, 0.001f, x}, c);
        push_vertex(v, {half, 0.001f, x}, c);
    }
}

static void add_frustum(std::vector<Vertex> &lines, std::vector<Vertex> &fill,
                        rt::float3 eye, rt::float3 at, rt::float3 up,
                        float fov_deg, float aspect) {
    // The raytracer projects onto an image plane exactly 1 unit in front
    // of the eye (rt::lookat's default focus_dist) — that plane IS the
    // scene framebuffer window.  The frustum's near plane is placed
    // there, not at an arbitrary fraction of the eye→at distance, so the
    // debug view shows where the rendered image lives in the scene.
    rt::Camera cam = rt::lookat(eye, at, up, fov_deg, aspect);
    rt::float3 nc[4] = {
        cam.lower_left,
        rt::add(cam.lower_left, cam.horizontal),
        rt::add(cam.lower_left, rt::add(cam.horizontal, cam.vertical)),
        rt::add(cam.lower_left, cam.vertical),
    };

    // Framebuffer window: translucent fill + bright outline so it reads
    // as the "screen" the raytracer renders onto.
    rt::float3 fb{1.f, 1.f, 0.72f};
    push_vertex(fill, nc[0], fb, 0.16f);
    push_vertex(fill, nc[1], fb, 0.16f);
    push_vertex(fill, nc[2], fb, 0.16f);
    push_vertex(fill, nc[0], fb, 0.16f);
    push_vertex(fill, nc[2], fb, 0.16f);
    push_vertex(fill, nc[3], fb, 0.16f);
    for ( int i = 0; i < 4; i++ ) {
        push_vertex(lines, nc[i], fb);
        push_vertex(lines, nc[(i + 1) % 4], fb);
    }

    // Far plane + connectors (the raytracer has no far clip plane; the
    // far plane here is just visual context).
    rt::float3 fwd = rt::norm(rt::sub(at, eye));
    rt::float3 right = rt::norm(rt::cross(fwd, up));
    rt::float3 upv = rt::cross(right, fwd);
    float dist = rt::len(rt::sub(at, eye));
    float far_d = std::fmax(dist * 3.f, 1.f);
    float tan_h = std::tan(fov_deg * kPi / 360.f);
    float hw = tan_h * far_d * aspect;
    float hh = tan_h * far_d;
    rt::float3 center = rt::add(eye, rt::scale(fwd, far_d));
    rt::float3 fc[4] = {
        rt::add(center, rt::add(rt::scale(right, -hw), rt::scale(upv, -hh))),
        rt::add(center, rt::add(rt::scale(right, hw), rt::scale(upv, -hh))),
        rt::add(center, rt::add(rt::scale(right, hw), rt::scale(upv, hh))),
        rt::add(center, rt::add(rt::scale(right, -hw), rt::scale(upv, hh))),
    };
    rt::float3 c{1.f, 0.85f, 0.2f};
    for ( int i = 0; i < 4; i++ ) {
        push_vertex(lines, fc[i], c);
        push_vertex(lines, fc[(i + 1) % 4], c);
        push_vertex(lines, nc[i], c);
        push_vertex(lines, fc[i], c);
    }
}

static void add_camera_indicator(std::vector<Vertex> &v, rt::float3 eye, rt::float3 at) {
    constexpr float s = 0.35f;
    // World-axis cross at the scene camera eye
    push_vertex(v, rt::add(eye, {-s, 0, 0}), {1.f, 0.2f, 0.2f});
    push_vertex(v, rt::add(eye, {s, 0, 0}), {1.f, 0.2f, 0.2f});
    push_vertex(v, rt::add(eye, {0, -s, 0}), {0.2f, 1.f, 0.2f});
    push_vertex(v, rt::add(eye, {0, s, 0}), {0.2f, 1.f, 0.2f});
    push_vertex(v, rt::add(eye, {0, 0, -s}), {0.2f, 0.4f, 1.f});
    push_vertex(v, rt::add(eye, {0, 0, s}), {0.2f, 0.4f, 1.f});
    // Look direction
    rt::float3 dir = rt::norm(rt::sub(at, eye));
    rt::float3 tip = rt::add(eye, rt::scale(dir, rt::len(rt::sub(at, eye)) * 0.3f));
    push_vertex(v, eye, {1.f, 1.f, 1.f});
    push_vertex(v, tip, {1.f, 1.f, 1.f});
}

// ── BVH tree overlay ─────────────────────────────────────────────────

/// Draw the BVH tree as wireframe AABBs down to `max_depth`, coloured by
/// depth (root red → deep blue).  Nodes entered by the traced ray (the
/// ray's "BVH hits") are drawn bright yellow, fully opaque.
static void add_bvh_overlay(std::vector<Vertex> &lines, const rt::SceneView &v,
                            uint32_t node_index, int depth, int max_depth,
                            const std::vector<uint32_t> *visited) {
    const rt::BvhNode &node = v.bvh_nodes[node_index];
    const bool is_visited =
        visited && std::find(visited->begin(), visited->end(), node_index) !=
                       visited->end();

    rt::float3 c;
    float a;
    if ( is_visited ) {
        c = {1.f, 0.85f, 0.2f};
        a = 1.f;
    } else {
        float t = max_depth > 1 ? (float)depth / (float)(max_depth - 1) : 1.f;
        c = rt::lerp({1.f, 0.35f, 0.2f}, {0.25f, 0.55f, 1.f}, t);
        a = 0.30f;
    }
    add_box_wire(lines, node.bounds.min, node.bounds.max, c, a);

    if ( depth >= max_depth || node.left == rt::BVH_LEAF ) return;
    add_bvh_overlay(lines, v, node.left, depth + 1, max_depth, visited);
    add_bvh_overlay(lines, v, node.right, depth + 1, max_depth, visited);
}

/// Draw one mesh's per-mesh BVH tree as wireframe AABBs, coloured green
/// (distinct from the red→blue scene BVH) by depth.  Nodes entered by the
/// traced ray (the mesh's interior "BVH hits") are drawn bright yellow,
/// fully opaque.  `max_depth` bounds the DIM part of the tree only —
/// visited nodes are always drawn (they are the actual traversal, the
/// interesting part); a dense mesh's tree (e.g. the 15k-tri duck) would
/// otherwise draw tens of thousands of boxes.
static void add_mesh_bvh_overlay(std::vector<Vertex> &lines, const rt::SceneView &v,
                                 uint32_t node_index, int depth, int max_depth,
                                 const std::vector<uint32_t> *visited) {
    const rt::BvhNode &node = v.mesh_bvh_nodes[node_index];
    const bool is_visited =
        visited && std::find(visited->begin(), visited->end(), node_index) !=
                       visited->end();

    rt::float3 c;
    float a;
    if ( is_visited ) {
        c = {1.f, 0.85f, 0.2f};
        a = 1.f;
    } else {
        float t = max_depth > 1 ? (float)depth / (float)(max_depth - 1) : 1.f;
        c = rt::lerp({0.15f, 0.85f, 0.3f}, {0.15f, 0.55f, 0.9f}, t);
        a = 0.30f;
    }
    add_box_wire(lines, node.bounds.min, node.bounds.max, c, a);

    // Recurse: visited nodes descend at ANY depth (they are the actual
    // traversal — their visited children are the rest of the path the ray
    // took); the dim part of the tree is capped at max_depth so a dense
    // mesh (e.g. the 15k-tri duck) doesn't draw tens of thousands of boxes.
    if ( node.left == rt::BVH_LEAF ) {
        return;
    }
    if ( !is_visited && depth >= max_depth ) {
        return;
    }
    add_mesh_bvh_overlay(lines, v, node.left, depth + 1, max_depth, visited);
    add_mesh_bvh_overlay(lines, v, node.right, depth + 1, max_depth, visited);
}

// ── Single-ray trace overlay ──────────────────────────────────────────

/// Draw one traced ray: a segment per bounce, each coloured with the
/// back-propagated output colour of that hit (the radiance the path
/// sends back along the segment, displayed through the SAME tone-map
/// operator + gamma as the framebuffer — see
/// scene_debug_display_color), a cross marker at every hit point, and
/// the escaped ray drawn out to a fixed visible length.
static void add_ray_overlay(std::vector<Vertex> &lines,
                            const rt::RayTraceResult &trace,
                            const SceneRenderParams &render) {
    auto add_marker = [&](rt::float3 p, rt::float3 c) {
        constexpr float s = 0.07f;
        push_vertex(lines, rt::add(p, {-s, 0, 0}), c);
        push_vertex(lines, rt::add(p, {s, 0, 0}), c);
        push_vertex(lines, rt::add(p, {0, -s, 0}), c);
        push_vertex(lines, rt::add(p, {0, s, 0}), c);
        push_vertex(lines, rt::add(p, {0, 0, -s}), c);
        push_vertex(lines, rt::add(p, {0, 0, s}), c);
    };
    // Display exactly what the framebuffer shows: exposure → tone-map
    // operator → gamma (hard clamp when tone-mapping is disabled).
    auto display = [&render](rt::float3 c) {
        return scene_debug_display_color(c, render);
    };

    if ( trace.steps.empty() ) {
        // No bounces traced — draw the start ray as-is.
        rt::float3 end = rt::add(trace.start.orig, rt::scale(trace.start.dir, 50.f));
        push_vertex(lines, trace.start.orig, {1.f, 1.f, 1.f});
        push_vertex(lines, end, {1.f, 1.f, 1.f});
        return;
    }

    for ( size_t i = 0; i < trace.steps.size(); i++ ) {
        const rt::RayTraceStep &s = trace.steps[i];
        rt::float3 c = display(s.color);

        rt::float3 end;
        if ( s.hit ) {
            end = rt::add(s.ray.orig, rt::scale(s.ray.dir, s.record.t));
            add_marker(end, c);
        } else {
            // Escaped — draw the ray out to a fixed visible length.
            end = rt::add(s.ray.orig, rt::scale(s.ray.dir, 50.f));
        }
        push_vertex(lines, s.ray.orig, c);
        push_vertex(lines, end, c);
    }
}

/// Human-readable summary of the traced ray, shown in the Ray Inspector
/// window.  Just the ray itself — the per-bounce hits are displayed by
/// the treeview, not duplicated as text.
static std::string build_trace_summary(const rt::Ray &start,
                                       const rt::RayTraceResult &trace,
                                       bool ray_picked,
                                       bool scene_cam_valid) {
    char buf[384];
    std::string out;
    std::snprintf(buf, sizeof(buf),
                  "ray origin=(%.2f, %.2f, %.2f) dir=(%.2f, %.2f, %.2f) [%s]\n",
                  start.orig.x, start.orig.y, start.orig.z, start.dir.x,
                  start.dir.y, start.dir.z,
                  ray_picked ? "picked" : (scene_cam_valid ? "scene camera center"
                                                           : "orbit camera target"));
    out += buf;
    if ( trace.bounce_limit )
        out += "  (bounce limit reached)\n";
    return out;
}

} // anonymous namespace

// ── SceneDebugRenderer ────────────────────────────────────────────────

bool SceneDebugRenderer::init(GLFWwindow *share_window) {
    if ( running_.load() || window_ ) {
        spdlog::warn("[scene-debug] already initialized");
        return true;
    }
    if ( !share_window ) {
        spdlog::error("[scene-debug] init requires the main GLFW window");
        return false;
    }

    // Hidden window sharing the main window's GL context.  GLFW requires
    // the share context to not be current on any thread during creation,
    // so release it, create, then restore.
    glfwMakeContextCurrent(nullptr);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window_ = glfwCreateWindow(512, 512, "sycl-sandbox scene-debug",
                               nullptr, share_window);
    glfwMakeContextCurrent(share_window);
    if ( !window_ ) {
        spdlog::error("[scene-debug] failed to create hidden GLFW window "
                      "(shared context)");
        return false;
    }

    running_.store(true);
    thread_ = std::thread(&SceneDebugRenderer::thread_main, this);
    return true;
}

void SceneDebugRenderer::shutdown() {
    if ( thread_.joinable() ) {
        running_.store(false);
        thread_.join();
    }
}

GLuint SceneDebugRenderer::present() {
    if ( !running_.load() ) return 0;

    // Newest READY frame wins.
    int best = -1;
    uint64_t best_gen = 0;
    for ( int i = 0; i < num_slots_; i++ ) {
        if ( slots_[i].state.load(std::memory_order_acquire) == 1 &&
             slots_[i].generation > best_gen ) {
            best = i;
            best_gen = slots_[i].generation;
        }
    }

    if ( best < 0 ) {
        // No new frame ready — the UI outran the render thread (it is
        // capped at ~60 fps and may be mid-render into the only free
        // slot).  Keep showing the previously presented texture: it is
        // still PRESENTING, so the render thread cannot overwrite it.
        // Without this, present() returned 0 and the panel flashed the
        // "unavailable" placeholder between frames.
        if ( presented_slot_ >= 0 ) return slots_[presented_slot_].tex;
        return 0;
    }

    if ( presented_slot_ == best ) {
        // Same slot still the newest — nothing to do (do NOT release it:
        // releasing to FREE then re-presenting would open a window where
        // the render thread grabs it while ImGui samples it).
        return slots_[best].tex;
    }

    // Release the slot presented last frame — ImGui sampled it during
    // ImGui::Render (composite_frame) after the previous present(), and
    // the current texture (`best`) is kept PRESENTING, so it is safe to
    // hand the old one back to the render thread now.
    if ( presented_slot_ >= 0 ) {
        slots_[presented_slot_].state.store(0, std::memory_order_release);
    }
    presented_slot_ = best;
    slots_[best].state.store(2, std::memory_order_release);
    return slots_[best].tex;
}

bool SceneDebugRenderer::gl_init() {
    const char *vs_src = R"(#version 330 core
layout(location=0) in vec3 a_pos;
layout(location=1) in vec4 a_color;
uniform mat4 u_mvp;
out vec4 v_color;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    v_color = a_color;
}
)";
    const char *fs_src = R"(#version 330 core
in vec4 v_color;
out vec4 frag_color;
void main() {
    frag_color = v_color;
}
)";
    program_ = compile_shader(vs_src, fs_src);
    if ( !program_ ) return false;

    // Camera-visibility shader: shades the orbit view's solid pass
    // per-pixel against the scene camera's depth pass (below + the
    // camera-visibility pass in thread_main).
    const char *vis_vs_src = R"(#version 330 core
layout(location=0) in vec3 a_pos;
layout(location=1) in vec4 a_color;
uniform mat4 u_mvp;
out vec3 v_world;
out vec4 v_color;
void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    v_world = a_pos;
    v_color = a_color;
}
)";
    const char *vis_fs_src = R"(#version 330 core
in vec3 v_world;
in vec4 v_color;
uniform mat4 u_cam_vp;
uniform sampler2D u_cam_depth;
uniform float u_eps;
out vec4 frag_color;

float lin_depth(float ndc_z) {
    // Linear view-space depth from NDC (n = 0.05, f = 2000 — must match
    // kZNear/kZFar): the depth texture's non-linear values are only
    // comparable after unprojecting.
    const float n = 0.05;
    const float f = 2000.0;
    return (2.0 * n * f) / (f + n - ndc_z * (f - n));
}

void main() {
    vec4 clip = u_cam_vp * vec4(v_world, 1.0);
    vec3 ndc = clip.xyz / clip.w;
    vec3 visible = v_color.rgb;
    vec3 occluded = v_color.rgb * 0.12;
    if ( clip.w <= 0.0 || ndc.x < -1.0 || ndc.x > 1.0 ||
         ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0 ) {
        frag_color = vec4(occluded, v_color.a);
        return;
    }
    // The depth texture is the z-test result of the scene from the
    // scene camera's POV — the "stencil" of pixels the framebuffer
    // sees.  Fragments at (or in front of) it are pixels the camera
    // sees; anything behind it is occluded from the camera.
    float tex_depth = texture(u_cam_depth, ndc.xy * 0.5 + 0.5).r;
    float zf = lin_depth(ndc.z);
    float zt = lin_depth(tex_depth);
    frag_color = vec4((zf <= zt * (1.0 + u_eps)) ? visible : occluded,
                      v_color.a);
}
)";
    vis_program_ = compile_shader(vis_vs_src, vis_fs_src);
    if ( !vis_program_ ) return false;
    vis_u_mvp_ = glGetUniformLocation(vis_program_, "u_mvp");
    vis_u_cam_vp_ = glGetUniformLocation(vis_program_, "u_cam_vp");
    vis_u_depth_ = glGetUniformLocation(vis_program_, "u_cam_depth");
    vis_u_eps_ = glGetUniformLocation(vis_program_, "u_eps");
    glUseProgram(vis_program_);
    glUniform1i(vis_u_depth_, 1);   // depth texture lives on unit 1
    glUseProgram(0);

    // Camera-visibility depth pass buffers (sized per-frame to the
    // raytraced framebuffer resolution).
    glGenTextures(1, &cam_color_tex_);
    glBindTexture(GL_TEXTURE_2D, cam_color_tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenTextures(1, &cam_depth_tex_);
    glBindTexture(GL_TEXTURE_2D, cam_depth_tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &cam_fbo_);
    resize_cam_fbo(512, 512);

    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, (GLsizei)sizeof(Vertex),
                          (const void *)offsetof(Vertex, x));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, (GLsizei)sizeof(Vertex),
                          (const void *)offsetof(Vertex, r));
    glEnableVertexAttribArray(1);

    for ( auto &s : slots_ ) {
        glGenTextures(1, &s.tex);
        glBindTexture(GL_TEXTURE_2D, s.tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glGenFramebuffers(1, &s.fbo);
        glGenRenderbuffers(1, &s.depth);
    }
    resize_slots(512, 512);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth(1.5f);
    return true;
}

void SceneDebugRenderer::gl_shutdown() {
    for ( auto &s : slots_ ) {
        if ( s.fbo ) glDeleteFramebuffers(1, &s.fbo);
        if ( s.tex ) glDeleteTextures(1, &s.tex);
        if ( s.depth ) glDeleteRenderbuffers(1, &s.depth);
        s.fbo = 0;
        s.tex = 0;
        s.depth = 0;
    }
    if ( vbo_ ) glDeleteBuffers(1, &vbo_);
    if ( vao_ ) glDeleteVertexArrays(1, &vao_);
    if ( program_ ) glDeleteProgram(program_);
    if ( cam_fbo_ ) glDeleteFramebuffers(1, &cam_fbo_);
    if ( cam_color_tex_ ) glDeleteTextures(1, &cam_color_tex_);
    if ( cam_depth_tex_ ) glDeleteTextures(1, &cam_depth_tex_);
    if ( vis_program_ ) glDeleteProgram(vis_program_);
    vbo_ = 0;
    vao_ = 0;
    program_ = 0;
    cam_fbo_ = 0;
    cam_color_tex_ = 0;
    cam_depth_tex_ = 0;
    vis_program_ = 0;
    cam_w_ = 0;
    cam_h_ = 0;
}

void SceneDebugRenderer::resize_slots(int w, int h) {
    w = std::clamp(w, kMinSize, kMaxSize);
    h = std::clamp(h, kMinSize, kMaxSize);
    for ( auto &s : slots_ ) {
        glBindFramebuffer(GL_FRAMEBUFFER, s.fbo);
        glBindTexture(GL_TEXTURE_2D, s.tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, nullptr);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, s.tex, 0);
        glBindRenderbuffer(GL_RENDERBUFFER, s.depth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                  GL_RENDERBUFFER, s.depth);
    }
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if ( status != GL_FRAMEBUFFER_COMPLETE ) {
        spdlog::warn("[scene-debug] framebuffer incomplete: 0x{:x}", (unsigned)status);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    size_w_ = w;
    size_h_ = h;
}

void SceneDebugRenderer::resize_cam_fbo(int w, int h) {
    w = std::clamp(w, kMinSize, kMaxSize);
    h = std::clamp(h, kMinSize, kMaxSize);
    if ( cam_fbo_ && w == cam_w_ && h == cam_h_ ) return;
    if ( !cam_fbo_ ) return;

    glBindTexture(GL_TEXTURE_2D, cam_color_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE,
                 nullptr);
    glBindTexture(GL_TEXTURE_2D, cam_depth_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, w, h, 0,
                 GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glBindFramebuffer(GL_FRAMEBUFFER, cam_fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, cam_color_tex_, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                           GL_TEXTURE_2D, cam_depth_tex_, 0);
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if ( status != GL_FRAMEBUFFER_COMPLETE ) {
        spdlog::warn("[scene-debug] camera-visibility framebuffer "
                     "incomplete: 0x{:x}", (unsigned)status);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    cam_w_ = w;
    cam_h_ = h;
}

namespace {

/// Snapshot of the shared state, taken under the mutex by the render
/// thread each frame.
struct Snap {
    std::shared_ptr<SceneDebugScene> scene;
    bool scene_cam_valid = false;
    rt::float3 eye{0, 0, 0};
    rt::float3 at{0, 0, 0};
    rt::float3 up{0, 1, 0};
    float fov = kFov;
    float aspect = 1.f;
    int w = 512;
    int h = 512;
    DebugViewFlags flags{};
    float yaw = -45.f;
    float pitch = -30.f;
    float dist = 12.f;
    rt::float3 eye_pos{0, 0, 0};
    bool reset_view = false;
    bool window_open = true;
    bool user_interacted = false;
    // Ray trace overlay state
    bool pick_ray_request = false;
    float pick_ndc_x = 0.f;
    float pick_ndc_y = 0.f;
    bool ray_picked = false;
    rt::float3 ray_origin{0, 0, 0};
    rt::float3 ray_dir{0, 0, -1};
    bool reset_ray = false;
    bool ray_from_rect = false;
    float picked_rect_u = 0.f;
    float picked_rect_v = 0.f;
    int fb_w = 0;
    int fb_h = 0;
    SceneRenderParams render{};
};

rt::float3 orbit_lookat(const Snap &s) {
    float cp = std::cos(s.pitch);
    float sp = std::sin(s.pitch);
    rt::float3 dir{cp * std::sin(s.yaw), sp, cp * std::cos(s.yaw)};
    return rt::add(s.eye_pos, rt::scale(dir, s.dist));
}

void place_at_scene_camera(Snap &s) {
    s.eye_pos = s.eye;  // orbit eye at scene camera position
    rt::float3 off = rt::sub(s.at, s.eye);
    float dist = rt::len(off);
    if ( dist < 1e-3f ) return;
    s.dist = dist;
    s.pitch = std::asin(std::clamp(off.y / dist, -1.f, 1.f));
    s.yaw = std::atan2(off.x, off.z);
}

/// Frame the camera around the scene bounds (fallback when the scene
/// has no camera; also used on explicit "Reset View" without one).
void autoframe(const rt::SceneView &v, Snap &s) {
    bool have = false;
    rt::float3 lo{0, 0, 0};
    rt::float3 hi{0, 0, 0};
    for ( int i = 0; i < v.num_handles; i++ ) {
        rt::Aabb b = aabb_for_handle(v, i);
        if ( !have ) {
            lo = b.min;
            hi = b.max;
            have = true;
        } else {
            lo = {std::fmin(lo.x, b.min.x), std::fmin(lo.y, b.min.y),
                  std::fmin(lo.z, b.min.z)};
            hi = {std::fmax(hi.x, b.max.x), std::fmax(hi.y, b.max.y),
                  std::fmax(hi.z, b.max.z)};
        }
    }
    if ( !have ) return;
    rt::float3 center = rt::scale(rt::add(lo, hi), 0.5f);
    rt::float3 ext = rt::sub(hi, lo);
    float max_ext = std::fmax(std::fmax(ext.x, ext.y), ext.z);
    float dist = max_ext * 0.8f / std::tan(kFov * kPi / 360.f);
    if ( dist < 0.5f ) dist = 0.5f;
    s.eye_pos = center;
    s.dist = dist;
    s.yaw = -45.f;
    s.pitch = 30.f;     // look DOWN at the scene (right-side up view)
}

// ── Framebuffer-rectangle picking ─────────────────────────────────────

/// Ray/triangle intersection (Möller–Trumbore).  Returns the hit
/// distance and barycentric coordinates (u along e1, v along e2).
/// Boundary points are accepted with a small epsilon so rays aimed
/// exactly at a quad edge/corner are watertight (float rounding at
/// the edges otherwise rejects ~half of exact-boundary rays).
static bool ray_tri_hit(rt::float3 o, rt::float3 d, rt::float3 a,
                        rt::float3 b, rt::float3 c, float &t, float &u,
                        float &v) {
    constexpr float kEps = 1e-5f;
    rt::float3 e1 = rt::sub(b, a);
    rt::float3 e2 = rt::sub(c, a);
    rt::float3 pv = rt::cross(d, e2);
    float det = rt::dot(e1, pv);
    if ( std::fabs(det) < 1e-12f ) return false;
    float inv = 1.f / det;
    rt::float3 tv = rt::sub(o, a);
    u = rt::dot(tv, pv) * inv;
    if ( u < -kEps || u > 1.f + kEps ) return false;
    rt::float3 qv = rt::cross(tv, e1);
    v = rt::dot(d, qv) * inv;
    if ( v < -kEps || u + v > 1.f + kEps ) return false;
    t = rt::dot(e2, qv) * inv;
    return t > 1e-4f;
}

/// Intersect a ray with the framebuffer rectangle quad (two triangles
/// matching add_frustum's fill).  On hit, `u`/`v` are the quad
/// coordinates (point = ll + u*h + v*vv) — the raytracer kernel's
/// pixel convention (u = x/W, v = y/H).  The barycentric results are
/// clamped so boundary hits land exactly on the edge.
static bool ray_quad_hit(const rt::Ray &ray, rt::float3 ll, rt::float3 h,
                         rt::float3 vv, float &u, float &v) {
    rt::float3 c1 = rt::add(ll, h);
    rt::float3 c2 = rt::add(c1, vv);
    rt::float3 c3 = rt::add(ll, vv);
    float t = 0.f, bu = 0.f, bv = 0.f;
    if ( ray_tri_hit(ray.orig, ray.dir, ll, c1, c2, t, bu, bv) ) {
        // point = ll + bu*h + bv*(h+vv) = ll + (bu+bv)*h + bv*vv
        u = std::clamp(bu + bv, 0.f, 1.f);
        v = std::clamp(bv, 0.f, 1.f);
        return true;
    }
    if ( ray_tri_hit(ray.orig, ray.dir, ll, c2, c3, t, bu, bv) ) {
        // point = ll + bu*(h+vv) + bv*vv = ll + bu*h + (bu+bv)*vv
        u = std::clamp(bu, 0.f, 1.f);
        v = std::clamp(bu + bv, 0.f, 1.f);
        return true;
    }
    return false;
}

/// Marker for the picked framebuffer pixel: a cross + the one-pixel
/// cell outline on the scene-camera framebuffer rectangle.
static void add_picked_pixel_marker(std::vector<Vertex> &overlay,
                                    const Snap &snap) {
    if ( !snap.scene_cam_valid ) return;
    rt::Camera cam = rt::lookat(snap.eye, snap.at, snap.up, snap.fov,
                                snap.aspect);
    rt::float3 pt = rt::add(
        rt::add(cam.lower_left, rt::scale(cam.horizontal, snap.picked_rect_u)),
        rt::scale(cam.vertical, snap.picked_rect_v));
    rt::float3 c{0.3f, 1.f, 0.45f};
    const float s = 0.06f;
    push_vertex(overlay, rt::add(pt, {-s, 0, 0}), c);
    push_vertex(overlay, rt::add(pt, {s, 0, 0}), c);
    push_vertex(overlay, rt::add(pt, {0, -s, 0}), c);
    push_vertex(overlay, rt::add(pt, {0, s, 0}), c);
    push_vertex(overlay, rt::add(pt, {0, 0, -s}), c);
    push_vertex(overlay, rt::add(pt, {0, 0, s}), c);

    // One framebuffer pixel, at the rectangle's proportions
    float pw = rt::len(cam.horizontal) /
               (float)(snap.fb_w > 0 ? snap.fb_w : 1024);
    float ph = rt::len(cam.vertical) /
               (float)(snap.fb_h > 0 ? snap.fb_h : 1024);
    rt::float3 hh = rt::scale(rt::norm(cam.horizontal), pw * 0.5f);
    rt::float3 hv = rt::scale(rt::norm(cam.vertical), ph * 0.5f);
    rt::float3 corners[4] = {
        rt::add(pt, rt::add(hh, hv)),
        rt::add(pt, rt::sub(hh, hv)),
        rt::sub(pt, rt::add(hh, hv)),
        rt::sub(pt, rt::sub(hh, hv)),
    };
    for ( int i = 0; i < 4; i++ ) {
        push_vertex(overlay, corners[i], c);
        push_vertex(overlay, corners[(i + 1) % 4], c);
    }
}

/// Build the scene geometry for one frame.  Four vertex lists:
///   - `solid`    — opaque triangles (write depth)
///   - `lines`    — wireframe/grid/overlay lines (write depth)
///   - `overlay`  — depth-test-DISABLED lines (BVH tree, traced ray),
///                  always visible through scene geometry
///   - `translucent` — alpha-blended fills (floor, frustum screen),
///                    drawn LAST with depth write disabled so they can
///                    never occlude geometry or z-fight with it.
/// `trace` (optional) is the current ray trace: its BVH node entries are
/// highlighted in the BVH overlay and the ray path is drawn on top.
void build_scene_geometry(const Snap &snap, std::vector<Vertex> &solid,
                          std::vector<Vertex> &lines,
                          std::vector<Vertex> &overlay,
                          std::vector<Vertex> &translucent,
                          const rt::RayTraceResult *trace = nullptr) {
    if ( snap.flags.show_floor ) add_floor(translucent, 20.f);
    if ( snap.flags.show_grid ) add_grid(lines, 20.f, 1.f);
    if ( !snap.scene ) return;

    const rt::SceneView &v = snap.scene->view;
    if ( snap.flags.show_objects || snap.flags.show_aabbs ) {
        for ( int i = 0; i < v.num_handles; i++ ) {
            const rt::Handle &h = v.handles[i];
            auto htype = static_cast<rt::HittableType>(rt::handle_tag(h.hittable));
            uint32_t hidx = rt::handle_index(h.hittable);

            rt::float3 color = material_color(h, v);
            rt::float3 wire = rt::scale(color, 0.6f);
            const bool wireframe = snap.flags.show_wireframe;

            switch ( htype ) {
                case rt::HittableType::Sphere: {
                    const auto &s = v.spheres[hidx];
                    if ( snap.flags.show_objects ) {
                        add_sphere_solid(solid, s.center, s.radius, color);
                        if ( wireframe ) add_sphere_wire(lines, s.center, s.radius, wire);
                    }
                    break;
                }
                case rt::HittableType::Quad: {
                    const auto &q = v.quads[hidx];
                    if ( snap.flags.show_objects ) {
                        add_quad_solid(solid, q.base, q.edge_u, q.edge_v, color);
                        if ( wireframe ) add_quad_wire(lines, q.base, q.edge_u, q.edge_v, wire);
                    }
                    break;
                }
                case rt::HittableType::Triangle: {
                    const auto &t = v.triangles[hidx];
                    if ( snap.flags.show_objects ) {
                        add_tri_solid(solid, t.a, t.b, t.c, color);
                        if ( wireframe ) add_tri_wire(lines, t.a, t.b, t.c, wire);
                    }
                    break;
                }
                case rt::HittableType::Box: {
                    const auto &b = v.boxes[hidx];
                    if ( snap.flags.show_objects ) {
                        add_box_solid(solid, b.box_min, b.box_max, color);
                        if ( wireframe ) add_box_wire(lines, b.box_min, b.box_max, wire);
                    }
                    break;
                }
                case rt::HittableType::Mesh: {
                    const auto &m = v.meshes[hidx];
                    if ( snap.flags.show_objects ) {
                        for ( uint32_t t = 0; t < m.num_triangles; t++ ) {
                            const auto &tri = v.triangles[m.first_triangle + t];
                            add_tri_solid(solid, tri.a, tri.b, tri.c, color);
                            if ( wireframe )
                                add_tri_wire(lines, tri.a, tri.b, tri.c, wire);
                        }
                    }
                    break;
                }
                case rt::HittableType::Portal: {
                    // Show BOTH portal shapes in the portal colour (both
                    // surfaces matter; portals are bidirectional).
                    const auto &p = v.portals[hidx];
                    rt::float3 pc{0.2f, 0.8f, 0.9f};
                    auto draw_shape = [&](const auto &shape) {
                        using S = std::decay_t<decltype(shape)>;
                        if constexpr ( std::is_same_v<S, rt::hittables::Sphere> ) {
                            if ( snap.flags.show_objects ) {
                                add_sphere_solid(solid, shape.center, shape.radius, pc);
                                if ( wireframe )
                                    add_sphere_wire(lines, shape.center, shape.radius,
                                                    rt::scale(pc, 0.6f));
                            }
                        } else if constexpr ( std::is_same_v<S, rt::hittables::Quad> ) {
                            if ( snap.flags.show_objects ) {
                                add_quad_solid(solid, shape.base, shape.edge_u, shape.edge_v, pc);
                                if ( wireframe )
                                    add_quad_wire(lines, shape.base, shape.edge_u, shape.edge_v,
                                                  rt::scale(pc, 0.6f));
                            }
                        } else if constexpr ( std::is_same_v<S, rt::hittables::Triangle> ) {
                            if ( snap.flags.show_objects ) {
                                add_tri_solid(solid, shape.a, shape.b, shape.c, pc);
                                if ( wireframe )
                                    add_tri_wire(lines, shape.a, shape.b, shape.c,
                                                 rt::scale(pc, 0.6f));
                            }
                        }
                    };
                    visit(p.entry, draw_shape);
                    visit(p.exit, draw_shape);
                    break;
                }
            }

            // Per-handle AABB overlay
            if ( snap.flags.show_aabbs ) {
                rt::Aabb b = aabb_for_handle(v, i);
                add_box_wire(lines, b.min, b.max, {0.9f, 0.5f, 0.2f});
            }
        }
    }

    // BVH tree overlay (nodes entered by the traced ray highlighted)
    if ( snap.flags.show_bvh && v.bvh_nodes && v.bvh_root >= 0 ) {
        std::vector<uint32_t> visited =
            trace ? trace->visited_nodes() : std::vector<uint32_t>{};
        add_bvh_overlay(overlay, v, (uint32_t)v.bvh_root, 0,
                        std::clamp(snap.flags.bvh_depth, 1, 32),
                        trace ? &visited : nullptr);
    }

    // Per-mesh BVH overlay (distinct from the scene BVH): each mesh's
    // interior tree, green depth-gradient; nodes the traced ray entered
    // are highlighted bright yellow (drawn at any depth).
    if ( snap.flags.show_bvh && v.mesh_bvh_nodes && v.num_mesh_bvh_nodes > 0 ) {
        std::vector<uint32_t> visited =
            trace ? trace->visited_hittable_nodes() : std::vector<uint32_t>{};
        // Cap the dim part of the tree: a dense mesh's full tree is far
        // larger than the scene BVH's (the duck alone has ~30k nodes),
        // so drawing it entirely would flood the frame.  Visited nodes
        // ignore the cap — they are the actual traversal.
        const int max_depth = std::clamp(snap.flags.bvh_depth, 1, 8);
        for ( int m = 0; m < v.num_meshes; m++ ) {
            const auto &mesh = v.meshes[m];
            if ( mesh.bvh_root < 0 ) continue;
            add_mesh_bvh_overlay(overlay, v, (uint32_t)mesh.bvh_root, 0,
                                 max_depth, trace ? &visited : nullptr);
        }
    }

    // Single-ray trace overlay
    if ( snap.flags.show_ray && trace )
        add_ray_overlay(overlay, *trace, snap.render);

    // Selected framebuffer pixel marker (picked on the framebuffer
    // rectangle — the pixel under the pick, at pixel size)
    if ( snap.ray_from_rect ) add_picked_pixel_marker(overlay, snap);

    // Scene-camera overlays
    if ( snap.scene_cam_valid ) {
        if ( snap.flags.show_frustum )
            add_frustum(lines, translucent, snap.eye, snap.at, snap.up, snap.fov,
                        snap.aspect);
        if ( snap.flags.show_camera ) add_camera_indicator(lines, snap.eye, snap.at);
    }
}

} // anonymous namespace

void SceneDebugRenderer::thread_main() {
    pthread_setname_np(pthread_self(), "sycl-scenedb");
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(0);

    if ( !gl_init() ) {
        spdlog::error("[scene-debug] GL initialization failed on debug thread");
        glfwMakeContextCurrent(nullptr);
        glfwDestroyWindow(window_);
        window_ = nullptr;
        return;
    }
    spdlog::info("[scene-debug] render thread GL ready (hidden shared context, "
                 "{}x{} slots, {} slots)", size_w_, size_h_, num_slots_);

    auto next_frame = std::chrono::steady_clock::now();
    int round_robin = 0;

    while ( running_.load() ) {
        // ── Throttle to ~60 fps ────────────────────────────────────
        next_frame += std::chrono::microseconds(kFrameIntervalUs);
        std::this_thread::sleep_until(next_frame);

        // ── Snapshot shared state ──────────────────────────────────
        Snap snap;
        {
            std::lock_guard<std::mutex> lk(state_.mutex);
            snap.scene = state_.scene;
            snap.scene_cam_valid = state_.scene_cam_valid;
            snap.eye = {state_.scene_cam_eye[0], state_.scene_cam_eye[1],
                        state_.scene_cam_eye[2]};
            snap.at = {state_.scene_cam_at[0], state_.scene_cam_at[1],
                       state_.scene_cam_at[2]};
            snap.up = {state_.scene_cam_up[0], state_.scene_cam_up[1],
                       state_.scene_cam_up[2]};
            snap.fov = state_.scene_cam_fov;
            snap.aspect = state_.scene_cam_aspect;
            snap.w = state_.size_w;
            snap.h = state_.size_h;
            snap.flags = state_.flags;
            snap.yaw = state_.orbit_yaw;
            snap.pitch = state_.orbit_pitch;
            snap.dist = state_.orbit_dist;
            snap.eye_pos = {state_.orbit_eye[0], state_.orbit_eye[1],
                            state_.orbit_eye[2]};
            snap.reset_view = state_.reset_view;
            state_.reset_view = false;
            snap.window_open = state_.window_open;
            snap.user_interacted = state_.camera_user_interacted;
            snap.pick_ray_request = state_.pick_ray_request;
            snap.pick_ndc_x = state_.pick_ndc_x;
            snap.pick_ndc_y = state_.pick_ndc_y;
            snap.ray_picked = state_.ray_picked;
            snap.ray_origin = {state_.ray_origin[0], state_.ray_origin[1],
                               state_.ray_origin[2]};
            snap.ray_dir = {state_.ray_dir[0], state_.ray_dir[1],
                            state_.ray_dir[2]};
            snap.reset_ray = state_.reset_ray;
            state_.reset_ray = false;
            snap.render = state_.render;
        }
        if ( !snap.window_open ) continue;

        // ── Place the camera on scene change (only until the user
        //    takes control; afterwards the camera stays put).  The
        //    default view matches the scene camera when one exists;
        //    autoframe around the scene bounds is the fallback.
        uint64_t version = snap.scene ? snap.scene->version : 0;
        if ( version != last_scene_version_ ) {
            last_scene_version_ = version;
            if ( snap.scene && !snap.reset_view && !snap.user_interacted &&
                 !camera_framed_ ) {
                if ( snap.scene_cam_valid ) {
                    place_at_scene_camera(snap);
                } else {
                    autoframe(snap.scene->view, snap);
                }
                camera_framed_ = true;
                // Write back so the UI state agrees.
                {
                    std::lock_guard<std::mutex> lk(state_.mutex);
                    state_.orbit_yaw = snap.yaw;
                    state_.orbit_pitch = snap.pitch;
                    state_.orbit_dist = snap.dist;
                    state_.orbit_eye[0] = snap.eye_pos.x;
                    state_.orbit_eye[1] = snap.eye_pos.y;
                    state_.orbit_eye[2] = snap.eye_pos.z;
                }
            }
        }
        if ( snap.reset_view ) {
            // Reset restores the default view: the scene camera when
            // available, otherwise an autoframe (or the default orbit
            // when there is no scene at all).
            if ( snap.scene_cam_valid ) {
                place_at_scene_camera(snap);
            } else if ( snap.scene ) {
                autoframe(snap.scene->view, snap);
            } else {
                snap.yaw = -45.f;
                snap.pitch = 30.f;
                snap.dist = 12.f;
                snap.eye_pos = {0, 0, 0};
            }
            camera_framed_ = true;
            // Write back to UI state so the panel doesn't overwrite
            // on the next frame.
            {
                std::lock_guard<std::mutex> lk(state_.mutex);
                state_.orbit_yaw = snap.yaw;
                state_.orbit_pitch = snap.pitch;
                state_.orbit_dist = snap.dist;
                state_.orbit_eye[0] = snap.eye_pos.x;
                state_.orbit_eye[1] = snap.eye_pos.y;
                state_.orbit_eye[2] = snap.eye_pos.z;
                state_.camera_user_interacted = false;
            }
        }

        // ── Resize if the UI requested a new resolution ───────────
        int w = std::clamp(snap.w, kMinSize, kMaxSize);
        int h = std::clamp(snap.h, kMinSize, kMaxSize);
        if ( w != size_w_ || h != size_h_ ) resize_slots(w, h);

        // ── Acquire a free slot (skip the frame if the UI is slow) ─
        int slot = -1;
        for ( int i = 0; i < num_slots_; i++ ) {
            int idx = (round_robin + i) % num_slots_;
            if ( slots_[idx].state.load(std::memory_order_acquire) == 0 ) {
                slot = idx;
                break;
            }
        }
        round_robin = (round_robin + 1) % num_slots_;
        if ( slot < 0 ) continue;

        // ── Camera ─────────────────────────────────────────────────
        rt::float3 eye = snap.eye_pos;
        rt::float3 lookat = orbit_lookat(snap);
        Mat4 view = mat4_look_at(eye, lookat, {0, 1, 0});
        Mat4 proj = mat4_perspective(kFov, (float)size_w_ / (float)size_h_,
                                     kZNear, kZFar);
        Mat4 mvp = mat4_mul(proj, view);

        // ── Ray trace overlay state ────────────────────────────────
        // A pick request (UI clicked the view) is resolved here with the
        // exact camera used for this frame, and the resulting ray is
        // stored back for the panel's summary and future frames.
        bool ray_picked = snap.ray_picked;
        rt::float3 picked_origin = snap.ray_origin;
        rt::float3 picked_dir = snap.ray_dir;
        bool ray_from_rect = snap.ray_from_rect;
        float picked_rect_u = snap.picked_rect_u;
        float picked_rect_v = snap.picked_rect_v;
        {
            std::lock_guard<std::mutex> lk(state_.mutex);
            if ( state_.pick_ray_request ) {
                state_.pick_ray_request = false;
                // Unproject the clicked NDC point onto the view plane
                // (1 unit in front of the eye) along the camera basis —
                // matches the frustum/framebuffer overlay math.
                float tan_h = std::tan(kFov * kPi / 360.f);
                float hw = tan_h * (float)size_w_ / (float)size_h_;
                float hh = tan_h;
                rt::float3 fwd = rt::norm(rt::sub(lookat, eye));
                rt::float3 right = rt::norm(rt::cross(fwd, {0, 1, 0}));
                rt::float3 upv = rt::cross(right, fwd);
                rt::float3 dir = rt::norm(rt::add(
                    fwd, rt::add(rt::scale(right, state_.pick_ndc_x * hw),
                                 rt::scale(upv, state_.pick_ndc_y * hh))));

                // Click on the scene-camera framebuffer rectangle → pick
                // the actual framebuffer pixel there: the scene camera's
                // ray through that pixel (same construction as the
                // raytracer kernel), so the traced ray matches the
                // rendered image pixel-for-pixel.  Anything else picks
                // a ray through the debug orbit camera.
                bool from_rect = false;
                float ru = 0.f;
                float rv = 0.f;
                if ( snap.scene_cam_valid ) {
                    rt::Camera cam = rt::lookat(snap.eye, snap.at, snap.up,
                                                snap.fov, snap.aspect);
                    if ( ray_quad_hit({eye, dir, 0.f}, cam.lower_left,
                                      cam.horizontal, cam.vertical, ru, rv) &&
                         ru >= 0.f && ru <= 1.f && rv >= 0.f && rv <= 1.f ) {
                        from_rect = true;
                        dir = rt::norm(rt::sub(
                            rt::add(rt::add(cam.lower_left,
                                            rt::scale(cam.horizontal, ru)),
                                    rt::scale(cam.vertical, rv)),
                            snap.eye));
                    }
                }
                rt::float3 origin = from_rect ? snap.eye : eye;
                state_.ray_origin[0] = origin.x;
                state_.ray_origin[1] = origin.y;
                state_.ray_origin[2] = origin.z;
                state_.ray_dir[0] = dir.x;
                state_.ray_dir[1] = dir.y;
                state_.ray_dir[2] = dir.z;
                state_.ray_picked = true;
                state_.ray_from_rect = from_rect;
                state_.picked_rect_u = ru;
                state_.picked_rect_v = rv;
                ray_picked = true;
                ray_from_rect = from_rect;
                picked_rect_u = ru;
                picked_rect_v = rv;
                picked_origin = origin;
                picked_dir = dir;
            }
            if ( snap.reset_ray ) {
                state_.ray_picked = false;
                state_.ray_from_rect = false;
                ray_picked = false;
                ray_from_rect = false;
            }
        }
        // Keep the pick marker in sync for this frame's geometry.
        snap.ray_from_rect = ray_from_rect;
        snap.picked_rect_u = picked_rect_u;
        snap.picked_rect_v = picked_rect_v;

        // ── Trace the single ray (deterministic RNG → stable view) ─
        rt::RayTraceResult trace;
        bool have_trace = false;
        std::string trace_text;
        if ( snap.flags.show_ray && snap.scene ) {
            rt::Ray trace_ray;
            if ( ray_picked ) {
                trace_ray.orig = picked_origin;
                trace_ray.dir = picked_dir;
            } else if ( snap.scene_cam_valid ) {
                // Default: scene camera center ray (eye → look-at).
                trace_ray.orig = snap.eye;
                trace_ray.dir = rt::norm(rt::sub(snap.at, snap.eye));
            } else {
                trace_ray.orig = eye;
                trace_ray.dir = rt::norm(rt::sub(lookat, eye));
            }
            rt::RNG rng{0x5eed5eedu};
            trace = rt::trace_ray_debug(
                trace_ray, snap.scene->view,
                std::clamp(snap.flags.ray_bounces, 0, 64), rng,
                snap.render.transparent_backfaces,
                {snap.render.background[0], snap.render.background[1],
                 snap.render.background[2]});
            have_trace = true;
            trace_text = build_trace_summary(trace_ray, trace, ray_picked,
                                             snap.scene_cam_valid);
        }
        {
            std::lock_guard<std::mutex> lk(state_.mutex);
            state_.ray_trace_text = std::move(trace_text);
            state_.ray_trace = have_trace ? trace : rt::RayTraceResult{};
        }

        // ── Build geometry + render ────────────────────────────────
        std::vector<Vertex> solid;
        std::vector<Vertex> lines;
        std::vector<Vertex> overlay;
        std::vector<Vertex> translucent;
        build_scene_geometry(snap, solid, lines, overlay, translucent,
                             have_trace ? &trace : nullptr);

        std::vector<Vertex> all;
        all.reserve(solid.size() + lines.size() + overlay.size() +
                    translucent.size());
        all.insert(all.end(), solid.begin(), solid.end());
        all.insert(all.end(), lines.begin(), lines.end());
        all.insert(all.end(), overlay.begin(), overlay.end());
        all.insert(all.end(), translucent.begin(), translucent.end());

        const GLsizei solid_count = (GLsizei)solid.size();
        const GLsizei lines_count = (GLsizei)lines.size();
        const GLsizei overlay_count = (GLsizei)overlay.size();

        // ── Camera-visibility pass: render the scene from the scene
        //    camera's POV into an offscreen depth buffer — the z-test
        //    result (the "stencil" of pixels the framebuffer sees).
        //    The orbit view's solid pass then shades against it.
        const bool vis_pass = snap.flags.show_visibility &&
                              snap.scene_cam_valid && !solid.empty();
        Mat4 cam_vp{};
        if ( vis_pass ) {
            int cw = snap.fb_w > 0 ? snap.fb_w : 1024;
            int ch = snap.fb_h > 0
                         ? snap.fb_h
                         : std::max(64, (int)(1024.f / std::max(snap.aspect,
                                                                0.01f)));
            resize_cam_fbo(cw, ch);
            Mat4 cam_view = mat4_look_at(snap.eye, snap.at, snap.up);
            Mat4 cam_proj = mat4_perspective(snap.fov, snap.aspect, kZNear,
                                             kZFar);
            cam_vp = mat4_mul(cam_proj, cam_view);
            glBindFramebuffer(GL_FRAMEBUFFER, cam_fbo_);
            glViewport(0, 0, cam_w_, cam_h_);
            glClear(GL_DEPTH_BUFFER_BIT);
            glUseProgram(program_);
            glUniformMatrix4fv(glGetUniformLocation(program_, "u_mvp"), 1,
                               GL_FALSE, cam_vp.m);
            glDrawArrays(GL_TRIANGLES, 0, solid_count);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        // ── Render into the slot ───────────────────────────────────
        glBindFramebuffer(GL_FRAMEBUFFER, slots_[slot].fbo);
        glViewport(0, 0, size_w_, size_h_);
        glClearColor(0.10f, 0.10f, 0.12f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(program_);
        glUniformMatrix4fv(glGetUniformLocation(program_, "u_mvp"), 1, GL_FALSE,
                           mvp.m);
        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, (ptrdiff_t)(all.size() * sizeof(Vertex)),
                     all.empty() ? nullptr : all.data(), GL_DYNAMIC_DRAW);
        if ( !solid.empty() ) {
            if ( vis_pass ) {
                // Per-pixel visibility overlay: fragments the scene
                // camera sees keep their material colour, occluded ones
                // are darkened (see vis_fs_src in gl_init).
                glUseProgram(vis_program_);
                glUniformMatrix4fv(vis_u_mvp_, 1, GL_FALSE, mvp.m);
                glUniformMatrix4fv(vis_u_cam_vp_, 1, GL_FALSE, cam_vp.m);
                glUniform1f(vis_u_eps_, 1e-3f);
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, cam_depth_tex_);
                glDrawArrays(GL_TRIANGLES, 0, solid_count);
                glUseProgram(program_);
            } else {
                glDrawArrays(GL_TRIANGLES, 0, solid_count);
            }
        }
        if ( !lines.empty() )
            glDrawArrays(GL_LINES, solid_count, lines_count);
        // Depth-test-disabled overlay pass: BVH tree + traced ray stay
        // visible through scene geometry (debug overlays must not be
        // occluded by the boxes they describe).
        if ( !overlay.empty() ) {
            glDisable(GL_DEPTH_TEST);
            glDrawArrays(GL_LINES, solid_count + lines_count, overlay_count);
            glEnable(GL_DEPTH_TEST);
        }
        // Translucent pass: blend, but do NOT write depth — the floor
        // and frustum screen tint surfaces without occluding anything
        // behind them (also prevents z-fighting with coplanar scene
        // geometry like the floor quad).
        if ( !translucent.empty() ) {
            glDepthMask(GL_FALSE);
            glDrawArrays(GL_TRIANGLES,
                         solid_count + lines_count + overlay_count,
                         (GLsizei)translucent.size());
            glDepthMask(GL_TRUE);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // Make the texture visible to the shared context BEFORE the UI
        // thread can present it (slot becomes READY only after glFinish).
        glFinish();
        slots_[slot].generation = ++generation_;
        slots_[slot].state.store(1, std::memory_order_release);
        glfwSwapBuffers(window_);
    }

    gl_shutdown();
    glfwMakeContextCurrent(nullptr);
    glfwDestroyWindow(window_);
    window_ = nullptr;
}
