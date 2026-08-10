#pragma once
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/scene_loader.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include <sycl-sandbox/scene/data.h>
#include "kernel/dispatch.h"

#include <string>
#include <cstdio>
#include <memory>
#include <spdlog/spdlog.h>

// ── SceneDebugScene (host copy for the debug renderer) ───────────────
/// Full host-side copy of the generated scene, built for the scene-debug
/// renderer thread.
///
/// The raytracing runtime uploads the scene straight into device memory,
/// which the debug view cannot read.  This is the same SceneBuilder data
/// re-uploaded through a NULL-queue `rt::Runtime`, so every array lands
/// on the heap and stays fully CPU-accessible (hittables, materials,
/// handles, AABBs, BVH nodes, lights).  That gives the debug renderer
/// read-only access to the *whole* scene — enough to trace rays against
/// it (BVH traversal, hit visualization, portal recursion, ...) without
/// touching device memory.
///
/// Lifetime: owned via `std::shared_ptr` and handed to the debug render
/// thread.  A rebuild (scene switch, param change, hot reload) swaps in a
/// new instance; the old one is freed only when the render thread drops
/// its reference — the renderer can never read freed memory.
struct SceneDebugScene {
    rt::Runtime rt;          ///< NULL-queue runtime (heap allocation mode)
    rt::SceneData data = {}; ///< OWNING heap arrays
    rt::SceneView view = {}; ///< NON-OWNING view of data
    uint64_t version = 0;    ///< bumped on every rebuild
};

// ── HostScene ────────────────────────────────────────────────────────
/// Persistent host-side scene state for YAML-defined scenes.
/// The SceneBuilder and SceneData live here so that SceneDebugInfo
/// pointers into the builder's buffers remain valid across frames.
///
/// Ownership split:
///   - data  — OWNING buffers (device/heap), allocated via SceneBuilder.
///             Never passed to the render function.
///   - view  — NON-OWNING snapshot of data, handed to the kernel in
///             ctx->scene each frame; the render path only sees this.
///   - debug_scene — host copy of the scene for the scene-debug render
///             thread (see SceneDebugScene above).
struct HostScene {
    rt::SceneData data = {};
    rt::SceneView view = {};
    rt::SceneBuilder builder;
    SceneDebugInfo debug_info = {};
    std::shared_ptr<SceneDebugScene> debug_scene;
};

/// Rebuild the scene from the YAML scene definition.
/// Only does work when the scene consumes a host-built scene — the
/// caller gates on SceneDef::uses_scene.
/// Frees the old host-side scene and rebuilds with current param values.
/// Uses the Runtime abstraction for all memory operations (SYCL or software).
inline void rebuild_yaml_scene(rt::Runtime *rt,
                                const std::string &yaml_path,
                                const scene_loader::SceneDescriptor &scene_desc,
                                HostScene &host_scene) {
    PROFILER_FUNCTION();

    // Free old scene via Runtime (handles SYCL or software dealloc)
    if ( !host_scene.data.empty() ) {
        host_scene.data.free(rt);
        host_scene = HostScene{};
    }

    // Load YAML and resolve data sources
    auto config = scene_loader::load_and_resolve(yaml_path);

    // Override with current live param values via typed ParamRef accessors
    for (const auto &pd : scene_desc.params) {
        auto it = config.params.find(pd.name);
        if (it == config.params.end()) continue;
        scene_loader::ParamRef ref = scene_desc.find_param_ref(pd.name);
        if (!ref.valid()) continue;
        switch (pd.type) {
            case ParamType::INT:
                it->second.int_val = ref.as_int();
                break;
            case ParamType::FLOAT:
                it->second.float_val = ref.as_float();
                break;
            case ParamType::COLOR_RGB:
            case ParamType::VEC3: {
                float v3[3];
                ref.as_vec3(v3);
                it->second.type = scene_loader::ValueType::Vec3;
                it->second.vec3_val = {v3[0], v3[1], v3[2]};
                break;
            }
            case ParamType::BOOL:
                it->second.int_val = ref.as_bool() ? 1 : 0;
                break;
            default:
                break;
        }
    }

    // Data sources may reference params ($name) — regenerate the
    // arrays with the live param values (e.g. num_spheres).
    scene_loader::resolve_data_sources(config);

    // Build geometry
    scene_loader::build_scene(host_scene.builder, config);
    host_scene.builder.build_bvh();
    host_scene.builder.build_mesh_bvhs();

    // Build the SceneData through the Runtime abstraction (owns the
    // buffers); the kernel renders with the non-owning view snapshot.
    host_scene.data = host_scene.builder.build(rt);
    host_scene.view = host_scene.data.view();

    host_scene.builder.build_debug_geometry();

    // Build debug info (pointers into the builder's persistent storage)
    SceneDebugInfo dbg{};
    dbg.aabb_data = host_scene.builder.debug_aabbs();
    dbg.num_aabbs = host_scene.builder.debug_aabb_count();
    dbg.spheres = host_scene.builder.debug_spheres();
    dbg.num_spheres = host_scene.builder.debug_sphere_count();
    dbg.quads = host_scene.builder.debug_quads();
    dbg.num_quads = host_scene.builder.debug_quad_count();
    dbg.boxes = host_scene.builder.debug_boxes();
    dbg.num_boxes = host_scene.builder.debug_box_count();
    host_scene.debug_info = dbg;

    // Host copy of the scene for the scene-debug render thread.
    // Built through a NULL-queue Runtime so all arrays are plain heap
    // (the renderer reads them from its own thread; device memory is not
    // CPU-accessible).  The shared_ptr handoff keeps the previous copy
    // alive until the render thread releases it, so swapping in a new
    // scene can never free memory the renderer is still reading.
    auto debug_scene = std::make_shared<SceneDebugScene>();
    debug_scene->data = host_scene.builder.build(&debug_scene->rt);
    debug_scene->view = debug_scene->data.view();
    debug_scene->version = host_scene.debug_scene
        ? host_scene.debug_scene->version + 1 : 1;
    host_scene.debug_scene = std::move(debug_scene);

    spdlog::info("[scene] YAML scene rebuilt: {} objects via SceneBuilder",
                 host_scene.view.num_handles);
}
