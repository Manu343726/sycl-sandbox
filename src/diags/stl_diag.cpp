// `diag stl` / `diag stl-center` — STL analysis and re-centering.
//
// Ported from the /tmp/duck_analyze.cpp + /tmp/duck_center.cpp harnesses:
// analysis loads the file with the project's own STL loader (ASCII +
// binary, auto-detected) and reports triangle count, AABB, and how many
// zero-area (degenerate) triangles it contains — the degenerate count is
// the regression guard for the GPU NaN-normal bug (see docs/raytracing.md).
// Re-centering translates the model to be origin-centered in X/Z with
// its bottom at y=0 and rewrites it as a clean binary STL.

#include "diags.h"

#include <sycl-sandbox/stl_loader.h>
#include <sycl-sandbox/rt/math.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace {

int run_stl_diag(const std::string &path) {
    auto res = stl::load_stl(path);
    if (!res.ok) {
        std::printf("LOAD FAILED: %s\n", res.error.c_str());
        return 1;
    }
    rt::float3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    size_t degen = 0;
    for (const auto &t : res.triangles) {
        for (const auto *p : {&t.a, &t.b, &t.c}) {
            lo.x = std::fmin(lo.x, p->x); lo.y = std::fmin(lo.y, p->y);
            lo.z = std::fmin(lo.z, p->z);
            hi.x = std::fmax(hi.x, p->x); hi.y = std::fmax(hi.y, p->y);
            hi.z = std::fmax(hi.z, p->z);
        }
        rt::float3 n = rt::cross(rt::sub(t.b, t.a), rt::sub(t.c, t.a));
        if (rt::len2(n) < 1e-12f) degen++;
    }
    std::printf("== stl: '%s'\n", path.c_str());
    std::printf("triangles=%zu\n", res.triangles.size());
    std::printf("aabb: min=(%.4f, %.4f, %.4f) max=(%.4f, %.4f, %.4f)\n",
                lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
    std::printf("size: x=%.4f y=%.4f z=%.4f\n",
                hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);
    std::printf("degenerate (zero-area) triangles: %zu %s\n", degen,
                degen == 0 ? "✓" : "⚠  (filtered at load; GPU NaN hazard)");
    return 0;
}

int run_stl_center_diag(const std::string &in, const std::string &out) {
    auto res = stl::load_stl(in);
    if (!res.ok) {
        std::printf("load failed: %s\n", res.error.c_str());
        return 1;
    }

    rt::float3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (const auto &t : res.triangles) {
        for (const auto *p : {&t.a, &t.b, &t.c}) {
            lo.x = std::fmin(lo.x, p->x); lo.y = std::fmin(lo.y, p->y);
            lo.z = std::fmin(lo.z, p->z);
            hi.x = std::fmax(hi.x, p->x); hi.y = std::fmax(hi.y, p->y);
            hi.z = std::fmax(hi.z, p->z);
        }
    }
    rt::float3 off{(lo.x + hi.x) * -0.5f, -lo.y, (lo.z + hi.z) * -0.5f};
    std::printf("offset=(%.4f, %.4f, %.4f)\n", off.x, off.y, off.z);

    // Clean binary STL: 80-byte header | uint32 count | 50 bytes/tri
    std::vector<uint8_t> bytes;
    bytes.reserve(84 + 50 * res.triangles.size());
    bytes.resize(84);
    const char *hdr = "sycl-sandbox stl-center: origin-centered X/Z, bottom at y=0";
    std::memcpy(bytes.data(), hdr, std::min<size_t>(79, std::strlen(hdr)));
    uint32_t count = (uint32_t)res.triangles.size();
    std::memcpy(bytes.data() + 80, &count, 4);

    for (const auto &t : res.triangles) {
        rt::float3 a = rt::add(t.a, off), b = rt::add(t.b, off),
                   c = rt::add(t.c, off);
        rt::float3 n = rt::norm(rt::cross(rt::sub(b, a), rt::sub(c, a)));
        float tri[12] = {n.x, n.y, n.z, a.x, a.y, a.z,
                         b.x, b.y, b.z, c.x, c.y, c.z};
        uint8_t buf[50];
        std::memcpy(buf, tri, 48);
        std::memset(buf + 48, 0, 2); // attribute byte count
        bytes.insert(bytes.end(), buf, buf + 50);
    }

    FILE *f = std::fopen(out.c_str(), "wb");
    if (!f) {
        std::printf("cannot write %s\n", out.c_str());
        return 1;
    }
    std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    std::printf("wrote %s: %zu bytes, %u tris (new aabb: x=[%.3f,%.3f] "
                "y=[0,%.3f] z=[%.3f,%.3f])\n",
                out.c_str(), bytes.size(), count,
                lo.x + off.x, hi.x + off.x, hi.y + off.y,
                lo.z + off.z, hi.z + off.z);
    return 0;
}

} // namespace

void register_stl_diag(argparse::ArgumentParser &diag,
                       std::vector<DiagCommand> &commands) {
    {   // diag stl [file]
        DiagCommand cmd;
        cmd.name = "stl";
        cmd.parser = std::make_unique<argparse::ArgumentParser>("stl");
        cmd.parser->add_argument("file")
            .nargs(argparse::nargs_pattern::optional)
            .default_value(std::string("scenes/models/duck.stl"))
            .help("STL file to analyze");
        cmd.run = [](argparse::ArgumentParser &p) {
            return run_stl_diag(p.get<std::string>("file"));
        };
        diag.add_subparser(*cmd.parser);
        commands.push_back(std::move(cmd));
    }
    {   // diag stl-center <input> <output>
        DiagCommand cmd;
        cmd.name = "stl-center";
        cmd.parser = std::make_unique<argparse::ArgumentParser>("stl-center");
        cmd.parser->add_argument("input").help("source STL");
        cmd.parser->add_argument("output").help("output STL (clean binary)");
        cmd.run = [](argparse::ArgumentParser &p) {
            return run_stl_center_diag(p.get<std::string>("input"),
                                       p.get<std::string>("output"));
        };
        diag.add_subparser(*cmd.parser);
        commands.push_back(std::move(cmd));
    }
}
