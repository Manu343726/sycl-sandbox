#pragma once
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/scene_loader.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include <sycl-sandbox/scene/data.h>
#include "kernel/dispatch.h"

#include <string>
#include <cstdio>
#include <spdlog/spdlog.h>

// ── HostScene ────────────────────────────────────────────────────────
/// Persistent host-side scene state for YAML-defined scenes.
/// The SceneBuilder and SceneData live here so that SceneDebugInfo
/// pointers into the builder's buffers remain valid across frames.
///
/// Ownership split:
///   - data  — OWNING buffers (device/heap), allocated via SceneBuilder.
///             Never passed to the render function.
///   - view  — NON-OWNING snapshot of data, handed to the kernel via
///             set_scene_view(); the render path only ever sees this.
struct HostScene {
    rt::SceneData data = {};
    rt::SceneView view = {};
    rt::SceneBuilder builder;
    SceneDebugInfo debug_info = {};
};

/// Rebuild the scene from the YAML scene definition and pass it to the kernel.
/// Only does work if the kernel exports `set_scene_view` (YAML-aware kernels).
/// Frees the old host-side scene and rebuilds with current param values.
/// Uses the Runtime abstraction for all memory operations (SYCL or software).
inline void rebuild_yaml_scene(rt::Runtime *rt,
                                const std::string &yaml_path,
                                void *kernel_handle,
                                const scene_loader::SceneDescriptor &scene_desc,
                                HostScene &host_scene) {
    PROFILER_FUNCTION();
    auto set_scene_fn = resolve_set_scene_view(kernel_handle);
    if ( !set_scene_fn ) {
        return; // Kernel does not support YAML scene injection
    }

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

    // Pass scene and debug info to the kernel
    set_scene_fn(&host_scene.view);
    auto set_dbg_fn = resolve_set_scene_debug_info(kernel_handle);
    if ( set_dbg_fn ) {
        set_dbg_fn(&host_scene.debug_info);
    }

    spdlog::info("[scene] YAML scene rebuilt: {} objects via SceneBuilder",
                 host_scene.view.num_handles);
}
