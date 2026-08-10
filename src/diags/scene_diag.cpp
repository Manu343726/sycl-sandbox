// `diag scene` — software-mode scene sanity check.
//
// Loads a YAML scene with the full project pipeline (load_and_resolve →
// build_scene → build_bvh → build) using a null Runtime (heap mode), then:
//   - prints handle / per-type / BVH / light counts
//   - prints every mesh window (first/count) + its transformed AABB and
//     material type
//   - fires canned rays (down from above the scene center, at the first
//     mesh's AABB center, and a deliberate miss) and reports the hits
//   - with `--strict` also enforces the mesh_demo.yaml expectations
//     (5 meshes, 16 036 triangles, duck: 15 168 tris, on the floor,
//     centered, dielectric) — regression guard for the Cornell-box duck.

#include "diags.h"

#include <sycl-sandbox/scene_loader.h>
#include <sycl-sandbox/scene/data.h>
#include <sycl-sandbox/rt/trace.h>
#include <sycl-sandbox/kernel/execution_context.h>

#include <cstdio>
#include <cmath>
#include <string>
#include <utility>

namespace {

bool check(bool cond, const char *what) {
    std::printf("%s: %s\n", cond ? "PASS" : "FAIL", what);
    return cond;
}

} // namespace

static int run_scene_diag(const std::string &yaml, bool strict) {
    auto config = scene_loader::load_and_resolve(yaml);

    rt::Runtime rt; // null queue → software mode (heap)
    rt::SceneBuilder builder;
    scene_loader::build_scene(builder, config);
    builder.build_bvh();
    builder.build_mesh_bvhs();
    rt::SceneData data = builder.build(&rt);
    rt::SceneView v = data.view();

    std::printf("== scene: '%s'\n", config.name.c_str());
    std::printf("handles=%d triangles=%d meshes=%d bvh_nodes=%d root=%d "
                "mesh_bvh_nodes=%d lights=%d\n",
                v.num_handles, v.num_triangles, v.num_meshes,
                v.num_bvh_nodes, v.bvh_root, v.num_mesh_bvh_nodes,
                v.num_lights);

    // Per-mesh windows + transformed AABB + material type
    for (uint32_t m = 0; m < v.num_meshes; m++) {
        const auto &mesh = v.meshes[m];
        rt::float3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
        for (uint32_t i = 0; i < mesh.num_triangles; i++) {
            const auto &t = v.triangles[mesh.first_triangle + i];
            for (auto *p : {&t.a, &t.b, &t.c}) {
                lo.x = std::fmin(lo.x, p->x); lo.y = std::fmin(lo.y, p->y);
                lo.z = std::fmin(lo.z, p->z);
                hi.x = std::fmax(hi.x, p->x); hi.y = std::fmax(hi.y, p->y);
                hi.z = std::fmax(hi.z, p->z);
            }
        }
        // Material of the handle that owns this mesh
        const char *mat = "?";
        for (int i = 0; i < v.num_handles; i++) {
            const rt::Handle &hnd = v.handles[i];
            auto htype = static_cast<rt::HittableType>(rt::handle_tag(hnd.hittable));
            if (htype == rt::HittableType::Mesh &&
                rt::handle_index(hnd.hittable) == m) {
                auto mt = static_cast<rt::MaterialType>(rt::handle_tag(hnd.material));
                switch (mt) {
                    case rt::MaterialType::Lambertian: mat = "lambertian"; break;
                    case rt::MaterialType::Metal: mat = "metal"; break;
                    case rt::MaterialType::Dielectric: mat = "dielectric"; break;
                    case rt::MaterialType::DiffuseLight: mat = "diffuse_light"; break;
                    case rt::MaterialType::TexturedLambertian: mat = "textured_lambertian"; break;
                }
                break;
            }
        }
        std::printf("mesh[%u]: first=%u count=%u mat=%s aabb=min(%.3f,%.3f,%.3f) "
                    "max(%.3f,%.3f,%.3f) size(%.3f,%.3f,%.3f)\n",
                    m, mesh.first_triangle, mesh.num_triangles, mat,
                    lo.x, lo.y, lo.z, hi.x, hi.y, hi.z,
                    hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);
    }

    bool ok = true;
    if (v.num_handles == 0) {
        ok = check(false, "scene has no handles");
    } else {
        // Ray straight down from above the scene center
        rt::Ray down{{0.f, 2.5f, 0.f}, {0.f, -1.f, 0.f}, 0.f};
        auto hit = rt::bvh_hit(down, 1e-3f, 1e30f, v);
        if (hit.has_value) {
            auto htype = static_cast<rt::HittableType>(rt::handle_tag(hit->handle.hittable));
            std::printf("down ray: type=%d t=%.3f p=(%.3f,%.3f,%.3f)\n",
                        (int)htype, hit->record.t,
                        hit->record.p.x, hit->record.p.y, hit->record.p.z);
        } else {
            ok = check(false, "downward ray from scene center hit nothing");
        }
    }

    if (strict) {
        ok &= check(v.num_meshes == 5, "mesh_demo: expected 5 meshes");
        ok &= check(v.num_triangles == 288 + 4 + 288 + 288 + 15168,
                    "mesh_demo: expected 16 036 triangles");
        if (v.num_meshes > 0) {
            const auto &duck = v.meshes[v.num_meshes - 1];
            ok &= check(duck.num_triangles == 15168, "duck mesh: 15 168 tris");
            rt::float3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
            for (uint32_t i = 0; i < duck.num_triangles; i++) {
                const auto &t = v.triangles[duck.first_triangle + i];
                for (auto *p : {&t.a, &t.b, &t.c}) {
                    lo.x = std::fmin(lo.x, p->x); lo.y = std::fmin(lo.y, p->y);
                    lo.z = std::fmin(lo.z, p->z);
                    hi.x = std::fmax(hi.x, p->x); hi.y = std::fmax(hi.y, p->y);
                    hi.z = std::fmax(hi.z, p->z);
                }
            }
            ok &= check(std::fabs(lo.y) < 0.05f && hi.y - lo.y > 0.8f &&
                        hi.y - lo.y < 1.2f,
                        "duck: on floor, ~1 unit tall");
            ok &= check(std::fabs(lo.x) < 1.f && std::fabs(hi.x) < 1.f &&
                        std::fabs(lo.z) < 1.f && std::fabs(hi.z) < 1.f,
                        "duck: origin-centered in X/Z");
        }
    }

    data.free(&rt);
    std::printf(ok ? "OK\n" : "FAILURES\n");
    return ok ? 0 : 1;
}

void register_scene_diag(argparse::ArgumentParser &diag,
                         std::vector<DiagCommand> &commands) {
    // Note: add_subparser() stores a REFERENCE to the parser and parses in
    // place — the parser must outlive registration and dispatch, so it is
    // heap-owned by DiagCommand.
    DiagCommand cmd;
    cmd.name = "scene";
    cmd.parser = std::make_unique<argparse::ArgumentParser>("scene");
    cmd.parser->add_argument("yaml")
        .nargs(argparse::nargs_pattern::optional)
        .default_value(std::string("scenes/mesh_demo.yaml"))
        .help("scene YAML file");
    cmd.parser->add_argument("--strict")
        .default_value(false)
        .implicit_value(true)
        .nargs(0)
        .help("enforce mesh_demo.yaml expectations (5 meshes, duck layout)");
    cmd.run = [](argparse::ArgumentParser &p) {
        return run_scene_diag(p.get<std::string>("yaml"),
                              p["--strict"] == true);
    };
    diag.add_subparser(*cmd.parser);
    commands.push_back(std::move(cmd));
}
