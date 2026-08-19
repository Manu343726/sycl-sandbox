// `diag loader` — headless scene-descriptor + built-scene dump.
//
// Ported from the /tmp/loader_diag.cpp harness.  Prints exactly what
// build_scene() produces for a YAML scene (software mode): the initial
// camera from the descriptor, all params/data sources, and every handle
// with its hittable + material tags and per-type payloads.

#include "diags.h"

#include <sycl-sandbox/scene_loader.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include <sycl-sandbox/scene/data.h>

#include <cstdio>
#include <string>
#include <utility>

namespace {

int run_loader_diag(const std::string &path) {
    // ── Descriptor check: does the YAML camera override land in the
    //    initial params buffer the kernel sees? ──
    auto desc = scene_loader::load_scene_descriptor(path);
    desc.build_layout();
    auto e = desc.find_param_ref("cam_eye");
    auto a = desc.find_param_ref("cam_at");
    auto f = desc.find_param_ref("cam_fov");
    auto ap = desc.find_param_ref("cam_aperture");
    if (e.valid() && a.valid()) {
        float ev[3], av[3];
        e.as_vec3(ev);
        a.as_vec3(av);
        std::printf("== descriptor initial camera: eye=(%f,%f,%f) "
                    "at=(%f,%f,%f) fov=%f aperture=%f\n",
                    ev[0], ev[1], ev[2], av[0], av[1], av[2],
                    f.valid() ? f.as_float() : -1.f,
                    ap.valid() ? ap.as_float() : -1.f);
    } else {
        std::printf("== descriptor has no 3D camera params (2d scene?)\n");
    }

    auto config = scene_loader::load_and_resolve(path);
    std::printf("== load_and_resolve: name='%s' params=%zu data_sources=%zu\n",
                config.name.c_str(), config.params.size(),
                config.data_sources.size());
    for (const auto &kv : config.params) {
        std::printf("   param %s type=%d int=%d float=%f vec3=(%f,%f,%f)\n",
                    kv.first.c_str(), (int)kv.second.type, kv.second.int_val,
                    kv.second.float_val, kv.second.vec3_val.x,
                    kv.second.vec3_val.y, kv.second.vec3_val.z);
    }
    for (const auto &kv : config.data_sources) {
        std::printf("   source %s type=%d len=%zu\n", kv.first.c_str(),
                    (int)kv.second.type, kv.second.array_length());
    }

    rt::SceneBuilder builder;
    scene_loader::build_scene(builder, config);
    builder.build_bvh();
    builder.build_mesh_bvhs();

    rt::Runtime rt; // queue == nullptr → software mode (heap)
    rt::MemoryPool pool; // heap allocation mode
    rt.pool = &pool;
    rt::SceneData data = builder.build(&rt);

    std::printf("== built scene: handles=%d  quads=%d  boxes=%d  spheres=%d\n",
                data.num_handles, data.num_quads, data.num_boxes,
                data.num_spheres);
    std::printf("   lambertians=%d  diffuse_lights=%d  bvh_nodes=%d  lights=%d\n",
                data.num_lambertians, data.num_diffuse_lights,
                data.num_bvh_nodes, data.num_lights);

    for (int i = 0; i < data.num_handles; i++) {
        rt::Handle h = data.handles[i];
        uint32_t htag = h.hittable >> 24, hidx = h.hittable & 0xFFFFFF;
        uint32_t mtag = h.material >> 24, midx = h.material & 0xFFFFFF;
        std::printf("handle %d: hittable tag=%u idx=%u  material tag=%u idx=%u",
                    i, htag, hidx, mtag, midx);
        if (htag == 2 && hidx < (uint32_t)data.num_quads) {
            auto &q = data.quads[hidx];
            std::printf("  quad base=(%f,%f,%f) u=(%f,%f,%f) v=(%f,%f,%f)",
                        q.base.x, q.base.y, q.base.z,
                        q.edge_u.x, q.edge_u.y, q.edge_u.z,
                        q.edge_v.x, q.edge_v.y, q.edge_v.z);
        }
        if (htag == 0 && hidx < (uint32_t)data.num_spheres) {
            auto &s = data.spheres[hidx];
            std::printf("  sphere center=(%f,%f,%f) radius=%f",
                        s.center.x, s.center.y, s.center.z, s.radius);
        }
        if (htag == 3 && hidx < (uint32_t)data.num_boxes) {
            auto &b = data.boxes[hidx];
            std::printf("  box min=(%f,%f,%f) max=(%f,%f,%f)",
                        b.box_min.x, b.box_min.y, b.box_min.z,
                        b.box_max.x, b.box_max.y, b.box_max.z);
        }
        if (mtag == 0 && midx < (uint32_t)data.num_lambertians) {
            auto &m = data.lambertians[midx];
            std::printf("  lambert albedo=(%f,%f,%f)",
                        m.albedo.x, m.albedo.y, m.albedo.z);
        }
        if (mtag == 3 && midx < (uint32_t)data.num_diffuse_lights) {
            auto &m = data.diffuse_lights[midx];
            std::printf("  diffuse emit=(%f,%f,%f)",
                        m.emit_color.x, m.emit_color.y, m.emit_color.z);
        }
        std::printf("\n");
    }
    data.free(&rt);
    return 0;
}

} // namespace

void register_loader_diag(argparse::ArgumentParser &diag,
                          std::vector<DiagCommand> &commands) {
    DiagCommand cmd;
    cmd.name = "loader";
    cmd.parser = std::make_unique<argparse::ArgumentParser>("loader");
    cmd.parser->add_argument("yaml")
        .nargs(argparse::nargs_pattern::optional)
        .default_value(std::string("scenes/cornell_box.yaml"))
        .help("scene YAML file");
    cmd.run = [](argparse::ArgumentParser &p) {
        return run_loader_diag(p.get<std::string>("yaml"));
    };
    diag.add_subparser(*cmd.parser);
    commands.push_back(std::move(cmd));
}
