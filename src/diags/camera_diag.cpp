// `diag orbit-camera` — verify the scene-debug orbit camera placement.

#include "diags.h"

#include <sycl-sandbox/rt/math.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>

namespace {

rt::float3 orbit_lookat(float yaw, float pitch, float dist, rt::float3 eye) {
    float cp = std::cos(pitch), sp = std::sin(pitch);
    rt::float3 dir{cp * std::sin(yaw), sp, cp * std::cos(yaw)};
    return rt::add(eye, rt::scale(dir, dist));
}

void place_at_scene_camera(rt::float3 eye, rt::float3 at, float &yaw,
                           float &pitch, float &dist) {
    rt::float3 off = rt::sub(at, eye);
    dist = rt::len(off);
    if (dist < 1e-3f) return;
    pitch = std::asin(std::clamp(off.y / dist, -1.f, 1.f));
    yaw = std::atan2(off.x, off.z);
}

int run_camera_diag() {
    struct Cam {
        rt::float3 eye, at;
        const char *name;
    };
    const Cam cams[] = {
        {{0.f, 2.5f, 6.f}, {0.f, 1.f, 0.f}, "cornell-ish"},
        {{-0.9f, 4.f, 2.5f}, {-0.9f, 1.2f, 0.4f}, "mesh demo-ish"},
        {{1.f, -3.f, -1.f}, {0.f, 0.f, 0.f}, "below looking up"},
        {{3.f, 0.5f, 3.f}, {0.f, 0.5f, 0.f}, "level, same height"},
        {{-2.f, 8.f, -2.f}, {0.f, 0.f, 0.f}, "steep top-down"},
    };

    bool ok = true;
    for (const auto &c : cams) {
        float yaw = 0, pitch = 0, dist = 0;
        place_at_scene_camera(c.eye, c.at, yaw, pitch, dist);
        rt::float3 lookat = orbit_lookat(yaw, pitch, dist, c.eye);
        rt::float3 err = rt::sub(lookat, c.at);
        float e = rt::len(err);
        bool pass = e < 1e-4f;
        ok = ok && pass;
        std::printf("%-22s eye=(%7.3f,%7.3f,%7.3f) at=(%7.3f,%7.3f,%7.3f) "
                    "yaw=%7.3f pitch=%7.3f dist=%7.3f -> err=%.2e %s\n",
                    c.name, c.eye.x, c.eye.y, c.eye.z,
                    c.at.x, c.at.y, c.at.z,
                    yaw, pitch, dist, e, pass ? "PASS" : "FAIL");
    }
    std::printf(ok ? "OK\n" : "FAILURES\n");
    return ok ? 0 : 1;
}

} // namespace

void register_camera_diag(argparse::ArgumentParser &diag,
                          std::vector<DiagCommand> &commands) {
    DiagCommand cmd;
    cmd.name = "orbit-camera";
    cmd.parser = std::make_unique<argparse::ArgumentParser>("orbit-camera");
    cmd.run = [](argparse::ArgumentParser &) { return run_camera_diag(); };
    diag.add_subparser(*cmd.parser);
    commands.push_back(std::move(cmd));
}
