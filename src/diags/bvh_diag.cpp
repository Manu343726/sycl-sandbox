// `diag bvh` — BVH-vs-linear-scan equivalence check.
//
// Ported from the /tmp/bvh_diag.cpp harness.  Builds a synthetic
// Cornell-box scene (quads + boxes + 40 spheres) in software mode and
// fires N random rays, verifying the BVH closest-hit matches a linear
// scan (same handle, same t) and that the skip_backfaces flag never
// returns backface records.

#include "diags.h"

#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/rt/types.h>
#include <sycl-sandbox/rt/trace.h>
#include <sycl-sandbox/rt/hittables/quad.h>
#include <sycl-sandbox/rt/hittables/box.h>
#include <sycl-sandbox/rt/hittables/sphere.h>
#include <sycl-sandbox/rt/materials/lambertian.h>
#include <sycl-sandbox/rt/materials/diffuse_light.h>
#include <sycl-sandbox/rt/materials/metal.h>

#include <cstdio>
#include <random>
#include <string>
#include <utility>

using namespace rt;

namespace {

int run_bvh_diag(int num_rays) {
    rt::Runtime rt; // null queue -> software mode

    // Cornell-box-like scene + some spheres (mix of types)
    SceneBuilder b;
    b.add({hittables::quad({0.f, 0.f, 0.f}, {555.f, 0.f, 0.f}, {0.f, 555.f, 0.f}),
           materials::lambertian({0.12f, 0.45f, 0.15f})}); // floor
    b.add({hittables::quad({0.f, 555.f, 0.f}, {555.f, 0.f, 0.f}, {0.f, 0.f, -555.f}),
           materials::lambertian({0.12f, 0.45f, 0.15f})}); // ceiling
    b.add({hittables::quad({0.f, 0.f, 0.f}, {0.f, 555.f, 0.f}, {0.f, 0.f, -555.f}),
           materials::lambertian({0.65f, 0.05f, 0.05f})}); // left
    b.add({hittables::quad({555.f, 0.f, 0.f}, {0.f, 555.f, 0.f}, {0.f, 0.f, -555.f}),
           materials::lambertian({0.73f, 0.73f, 0.73f})}); // right
    b.add({hittables::quad({0.f, 0.f, -555.f}, {555.f, 0.f, 0.f}, {0.f, 555.f, 0.f}),
           materials::lambertian({0.73f, 0.73f, 0.73f})}); // back
    b.add({hittables::box(130.f, 0.f, 65.f, 165.f, 330.f, 230.f),
           materials::lambertian({0.73f, 0.73f, 0.73f})});
    b.add({hittables::box(265.f, 0.f, 295.f, 165.f, 165.f, 165.f),
           materials::lambertian({0.73f, 0.73f, 0.73f})});
    for (int i = 0; i < 40; ++i) {
        float x = (float)((i * 97) % 500) + 20.f;
        float y = (float)((i * 61) % 300) + 20.f;
        float z = (float)((i * 43) % 500) - 500.f + 20.f;
        b.add({hittables::sphere({x, y, z}, 25.f),
               materials::metal({0.5f, 0.5f, 0.8f}, 0.1f)});
    }
    b.build_bvh();
    b.build_mesh_bvhs();
    SceneData data = b.build(&rt);
    SceneView scene = data.view();

    std::printf("handles=%d bvh_nodes=%d root=%d\n",
                scene.num_handles, scene.num_bvh_nodes, scene.bvh_root);

    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> ud(-600.f, 600.f);

    int mismatches = 0;
    int bvh_miss = 0, lin_miss = 0;
    for (int i = 0; i < num_rays; ++i) {
        Ray ray;
        ray.orig = {ud(gen), ud(gen), ud(gen)};
        ray.dir = {ud(gen), ud(gen), ud(gen)};
        float len = std::sqrt(ray.dir.x * ray.dir.x + ray.dir.y * ray.dir.y +
                              ray.dir.z * ray.dir.z);
        if (len < 1e-6f) { --i; continue; }
        ray.dir = {ray.dir.x / len, ray.dir.y / len, ray.dir.z / len};

        // Linear scan (reference)
        optional<HitRecord> lin;
        Handle lin_h = {};
        for (int h = 0; h < scene.num_handles; ++h) {
            auto hit = handle_hit(scene.handles[h], ray, 0.001f,
                                  lin ? lin->t : 1e30f, scene);
            if (hit) { lin = hit; lin_h = scene.handles[h]; }
        }
        // BVH query
        auto bvh = bvh_hit(ray, 0.001f, 1e30f, scene);

        bool l_miss = !lin, b_miss = !bvh;
        if (l_miss) ++lin_miss;
        if (b_miss) ++bvh_miss;
        if (l_miss != b_miss) {
            ++mismatches;
            if (mismatches <= 5)
                std::printf("MISMATCH miss: ray=(%f,%f,%f)->(%f,%f,%f) "
                            "lin=%d bvh=%d\n",
                            ray.orig.x, ray.orig.y, ray.orig.z,
                            ray.dir.x, ray.dir.y, ray.dir.z, l_miss, b_miss);
            continue;
        }
        if (!l_miss) {
            float dt = lin->t - bvh->record.t;
            if (dt < 0) dt = -dt;
            if (dt > 1e-4f || lin_h.hittable != bvh->handle.hittable ||
                lin_h.material != bvh->handle.material) {
                ++mismatches;
                if (mismatches <= 10)
                    std::printf("MISMATCH hit: t_lin=%f t_bvh=%f "
                                "h_lin=(%u,%u) h_bvh=(%u,%u)\n",
                                lin->t, bvh->record.t,
                                lin_h.hittable, lin_h.material,
                                bvh->handle.hittable, bvh->handle.material);
            }
        }
    }
    std::printf("rays=%d mismatches=%d (lin_miss=%d bvh_miss=%d)\n",
                num_rays, mismatches, lin_miss, bvh_miss);
    bool ok = mismatches == 0;
    if (ok)
        std::printf("PASS: BVH closest-hit == linear scan on %d rays\n",
                    num_rays);

    // Also verify the X-ray flag semantics: skip_backfaces should never
    // return a backface record.
    int backface_found = 0;
    for (int i = 0; i < num_rays; ++i) {
        Ray ray;
        ray.orig = {ud(gen), ud(gen), ud(gen)};
        ray.dir = {ud(gen), ud(gen), ud(gen)};
        float len = std::sqrt(ray.dir.x * ray.dir.x + ray.dir.y * ray.dir.y +
                              ray.dir.z * ray.dir.z);
        if (len < 1e-6f) { --i; continue; }
        ray.dir = {ray.dir.x / len, ray.dir.y / len, ray.dir.z / len};
        auto bvh = bvh_hit(ray, 0.001f, 1e30f, scene,
                           /*skip_backfaces=*/true);
        if (bvh && !bvh->record.front_face) ++backface_found;
    }
    std::printf("skip_backfaces: backface records returned = %d (expect 0)\n",
                backface_found);

    data.free(&rt);
    ok = ok && backface_found == 0;
    std::printf(ok ? "OK\n" : "FAILURES\n");
    return ok ? 0 : 1;
}

} // namespace

void register_bvh_diag(argparse::ArgumentParser &diag,
                       std::vector<DiagCommand> &commands) {
    DiagCommand cmd;
    cmd.name = "bvh";
    cmd.parser = std::make_unique<argparse::ArgumentParser>("bvh");
    // NOTE: argparse v3.2 stores CLI-supplied values as std::string (bad
    // any_cast for get<int>), so convert at parse time via .action().
    cmd.parser->add_argument("rays")
        .nargs(argparse::nargs_pattern::optional)
        .default_value(5000)
        .action([](const std::string &s) { return std::stoi(s); })
        .help("number of random rays to test");
    cmd.run = [](argparse::ArgumentParser &p) {
        return run_bvh_diag(p.get<int>("rays"));
    };
    diag.add_subparser(*cmd.parser);
    commands.push_back(std::move(cmd));
}
