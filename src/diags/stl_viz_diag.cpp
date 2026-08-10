// `diag stl-viz <file.stl> <outprefix> [--rx] [--ry] [--rz]` — render an
// STL mesh to shaded orthographic PNGs to inspect its true orientation.
//
// Uses the project's own stl_loader + apply_transform, so any candidate
// Euler rotation can be verified end-to-end (same Z → Y → X fixed-axis
// convention as scene YAML `rotation:`) before committing it to a scene.
//
// Writes four images (front/side/top/three-quarter), each 640x480:
//   <outprefix>_xy.png   screen right=+X, up=+Y  (front, looking down -Z)
//   <outprefix>_zy.png   screen right=+Z, up=+Y  (side,  looking from +X)
//   <outprefix>_xz.png   screen right=+X, up=+Z  (top,    looking down -Y)
//   <outprefix>_iso.png  three-quarter view
//
// PNG output via stb_image_write (FetchContent).  The renderer is a tiny
// software z-buffered rasterizer (Lambert headlight, two-sided normals) —
// no GL, so it runs headless.

#include "diags.h"

#include <sycl-sandbox/stl_loader.h>
#include <sycl-sandbox/rt/math.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;

struct VizImage {
    std::vector<unsigned char> px = std::vector<unsigned char>(kWidth * kHeight * 3, 18);
    std::vector<float> zb = std::vector<float>(kWidth * kHeight, 1e30f); // nearest = smallest
};

struct VizView {
    rt::float3 right, up, dir; // ortho basis; camera looks along `dir`
};

// Lambert shade from a fixed headlight slightly off the view direction.
unsigned char shade(rt::float3 n, rt::float3 view_dir) {
    rt::float3 l = rt::norm(rt::add(view_dir, rt::float3{0.3f, 0.5f, 0.4f}));
    float d = std::fmax(0.f, rt::dot(n, l));
    float a = 0.15f + 0.85f * d;
    return (unsigned char)std::clamp(220.f * a, 0.f, 255.f);
}

void render_view(const std::vector<rt::hittables::Triangle> &tris,
                 const VizView &view, const std::string &file) {
    VizImage img;
    // Fit: project triangle bounds onto the right/up axes.
    float rmin = 1e30f, rmax = -1e30f, umin = 1e30f, umax = -1e30f;
    for (const auto &t : tris)
        for (const auto *p : {&t.a, &t.b, &t.c}) {
            float r = rt::dot(*p, view.right), u = rt::dot(*p, view.up);
            rmin = std::fmin(rmin, r); rmax = std::fmax(rmax, r);
            umin = std::fmin(umin, u); umax = std::fmax(umax, u);
        }
    float span = std::fmax(rmax - rmin, umax - umin) * 1.08f;
    float rmid = 0.5f * (rmin + rmax), umid = 0.5f * (umin + umax);
    float world_per_px = span / kHeight;

    for (const auto &t : tris) {
        rt::float3 n = rt::norm(rt::cross(rt::sub(t.b, t.a), rt::sub(t.c, t.a)));
        if (rt::dot(n, view.dir) > 0) n = rt::scale(n, -1.f); // two-sided
        unsigned char g = shade(n, rt::scale(view.dir, -1.f));

        float sx[3], sy[3], sz[3];
        const rt::float3 *v[3] = {&t.a, &t.b, &t.c};
        for (int i = 0; i < 3; i++) {
            sx[i] = kWidth * 0.5f + (rt::dot(*v[i], view.right) - rmid) / world_per_px;
            sy[i] = kHeight * 0.5f - (rt::dot(*v[i], view.up) - umid) / world_per_px;
            sz[i] = rt::dot(*v[i], view.dir);
        }
        int x0 = std::max(0, (int)std::floor(std::fmin(sx[0], std::fmin(sx[1], sx[2]))));
        int x1 = std::min(kWidth - 1, (int)std::ceil(std::fmax(sx[0], std::fmax(sx[1], sx[2]))));
        int y0 = std::max(0, (int)std::floor(std::fmin(sy[0], std::fmin(sy[1], sy[2]))));
        int y1 = std::min(kHeight - 1, (int)std::ceil(std::fmax(sy[0], std::fmax(sy[1], sy[2]))));
        float area = (sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0]);
        if (std::fabs(area) < 1e-9f) continue;
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++) {
                float px = x + 0.5f, py = y + 0.5f;
                float w0 = ((sx[1] - px) * (sy[2] - py) - (sx[2] - px) * (sy[1] - py)) / area;
                float w1 = ((sx[2] - px) * (sy[0] - py) - (sx[0] - px) * (sy[2] - py)) / area;
                float w2 = 1.f - w0 - w1;
                if (w0 < 0 || w1 < 0 || w2 < 0) continue;
                float z = w0 * sz[0] + w1 * sz[1] + w2 * sz[2];
                int idx = y * kWidth + x;
                if (z >= img.zb[idx]) continue; // keep nearest
                img.zb[idx] = z;
                img.px[idx * 3 + 0] = g;
                img.px[idx * 3 + 1] = (unsigned char)(g * 0.85f);
                img.px[idx * 3 + 2] = (unsigned char)(g * 0.55f);
            }
    }
    stbi_write_png(file.c_str(), kWidth, kHeight, 3, img.px.data(), kWidth * 3);
    std::printf("wrote %s\n", file.c_str());
}

int run_stl_viz_diag(const std::string &file, const std::string &prefix,
                     float rx, float ry, float rz) {
    auto res = stl::load_stl(file);
    if (!res.ok) {
        std::printf("LOAD FAILED: %s\n", res.error.c_str());
        return 1;
    }
    if (rx != 0.f || ry != 0.f || rz != 0.f) {
        stl::MeshTransform tf;
        tf.rotation_degrees = {rx, ry, rz};
        stl::apply_transform(res.triangles, tf);
        std::printf("applied rotation (%.1f, %.1f, %.1f)\n", rx, ry, rz);
    }
    rt::float3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (const auto &t : res.triangles)
        for (const auto *p : {&t.a, &t.b, &t.c}) {
            lo.x = std::fmin(lo.x, p->x); lo.y = std::fmin(lo.y, p->y); lo.z = std::fmin(lo.z, p->z);
            hi.x = std::fmax(hi.x, p->x); hi.y = std::fmax(hi.y, p->y); hi.z = std::fmax(hi.z, p->z);
        }
    std::printf("aabb min=(%.2f, %.2f, %.2f) max=(%.2f, %.2f, %.2f)\n",
                lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);

    render_view(res.triangles, {{1, 0, 0}, {0, 1, 0}, {0, 0, -1}}, prefix + "_xy.png");
    render_view(res.triangles, {{0, 0, 1}, {0, 1, 0}, {-1, 0, 0}}, prefix + "_zy.png");
    render_view(res.triangles, {{1, 0, 0}, {0, 0, 1}, {0, -1, 0}}, prefix + "_xz.png");
    rt::float3 r = rt::norm(rt::float3{1, 0, -1});
    rt::float3 u = rt::norm(rt::float3{-0.4f, 1.2f, -0.4f});
    render_view(res.triangles, {r, u, rt::norm(rt::cross(r, u))}, prefix + "_iso.png");
    return 0;
}

} // namespace

void register_stl_viz_diag(argparse::ArgumentParser &diag,
                           std::vector<DiagCommand> &commands) {
    DiagCommand cmd;
    cmd.name = "stl-viz";
    cmd.parser = std::make_unique<argparse::ArgumentParser>("stl-viz");
    cmd.parser->add_argument("file").help("STL file to render");
    cmd.parser->add_argument("outprefix").help("output PNG prefix");
    cmd.parser->add_argument("--rx").default_value(0.0f).scan<'g', float>()
        .help("Euler X rotation (degrees), applied Z->Y->X");
    cmd.parser->add_argument("--ry").default_value(0.0f).scan<'g', float>()
        .help("Euler Y rotation (degrees)");
    cmd.parser->add_argument("--rz").default_value(0.0f).scan<'g', float>()
        .help("Euler Z rotation (degrees)");
    cmd.run = [](argparse::ArgumentParser &p) {
        return run_stl_viz_diag(p.get<std::string>("file"),
                                p.get<std::string>("outprefix"),
                                p.get<float>("--rx"),
                                p.get<float>("--ry"),
                                p.get<float>("--rz"));
    };
    diag.add_subparser(*cmd.parser);
    commands.push_back(std::move(cmd));
}
