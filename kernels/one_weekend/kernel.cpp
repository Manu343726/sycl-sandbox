#include <sycl-sandbox/sandbox_api.h>
#include "kernel.h"
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types.h>
#include <sycl-sandbox/rt/camera.h>
#include <sycl-sandbox/rt/trace.h>
#include <sycl-sandbox/rt/params.h>
#include <sycl-sandbox/rt/scene_data.h>
#include <sycl-sandbox/rt/hittables/sphere.h>
#include <sycl-sandbox/rt/materials/lambertian.h>
#include <sycl-sandbox/rt/materials/metal.h>
#include <sycl-sandbox/rt/materials/dielectric.h>
#include <cstdlib>
#include <cstring>

using namespace rt;
using rt::hittables::sphere;
using rt::materials::lambertian;
using rt::materials::metal;
using rt::materials::dielectric;

static ParamMeta params_meta[] = {
    {"num_spheres",
     "Number of random small spheres",
     ParamType::INT,
     .buffer_offset = RT_NUM_STD_PARAMS * sizeof(float),
     .range = {.i = {0, 500, 1}},
     .default_i = 11},
    {"ground_color",
     "Ground sphere albedo",
     ParamType::COLOR_RGB,
     .default_c3 = {0.5f, 0.5f, 0.5f}},
    {"background", "Sky colour", ParamType::COLOR_RGB, .default_c3 = {0.5f, 0.7f, 1}},
};

enum {
    PARAM_NUM_SPHERES = 0,
    PARAM_GROUND_COLOR = PARAM_NUM_SPHERES + 1,
    PARAM_BACKGROUND = PARAM_GROUND_COLOR + 3,
};

static KernelDesc desc = {"one_weekend",
                          "Raytracing in One Weekend — random spheres",
                          3,
                          params_meta,
                          0,
                          4096,
                          2,
                          (const char *[]) {"kernel.cpp", "kernel.h", nullptr}};
extern "C" KernelDesc *get_kernel_desc() {
    return &desc;
}

static SceneBuilder scene;
static SceneView scene_view = {};
static float3 background = {0.5f, 0.7f, 1.0f};

static float random_float() {
    return (float)rand() * (1.f / 2147483647.f);
}

extern "C" void init_kernel(sycl::queue *queue, int, int, const void *params_buffer, size_t) {
    const float *params = (const float *)params_buffer;
    int offset = RT_NUM_STD_PARAMS;
    int num_spheres = (int)params[offset + PARAM_NUM_SPHERES];
    float3 ground_color;
    memcpy(&ground_color, params + offset + PARAM_GROUND_COLOR, 12);
    memcpy(&background, params + offset + PARAM_BACKGROUND, 12);

    if ( scene_view.handles ) {
        scene_view.free(*queue);
    }
    srand(42);

    scene = SceneBuilder();

    scene.add({sphere({0, -1000, 0}, 1000), lambertian(ground_color)});

    for ( int k = 0; k < num_spheres; ) {
        float x = -10.0f + 20.0f * random_float();
        float z = -10.0f + 20.0f * random_float();
        float3 center = {x, 0.2f, z};
        if ( len(center) <= 0.9f ) {
            continue;
        }

        float3 color = {random_float() * random_float(),
                        random_float() * random_float(),
                        random_float() * random_float()};
        float choice = random_float();

        if ( choice < 0.6f ) {
            scene.add({sphere(center, 0.2f), lambertian(color)});
        } else if ( choice < 0.85f ) {
            scene.add({sphere(center, 0.2f), metal(color, 0.5f * random_float())});
        } else {
            scene.add({sphere(center, 0.2f), dielectric(1.5f)});
        }
        k++;
    }

    scene.add({sphere({4, 1, 0}, 1), metal({0.7f, 0.6f, 0.5f}, 0)});
    scene.add({sphere({-4, 1, 0}, 1), lambertian({0.4f, 0.2f, 0.1f})});
    scene.add({sphere({0, 1, 0}, 1), dielectric(1.5f)});

    scene.build_bvh();
    scene_view = scene.build(*queue);
    scene.build_debug_geometry();
}

extern "C" void render_kernel(sycl::queue *queue,
                              int width,
                              int height,
                              const void *params_buffer,
                              void *accum_buffer,
                              int sample_index) {
    const float *params = (const float *)params_buffer;
    render_main(queue,
                width,
                height,
                params,
                (float *)accum_buffer,
                sample_index,
                scene_view,
                [background = background](const Ray &ray) -> float3 {
                    float t = 0.5f * (ray.dir.y + 1.0f);
                    return lerp({1, 1, 1}, background, t);
                });
}

extern "C" void shutdown_kernel(sycl::queue *queue) {
    if ( scene_view.handles ) {
        scene_view.free(*queue);
        scene_view = {};
    }
}

extern "C" const SceneDebugInfo *get_scene_debug_info() {
    if ( scene.debug_aabb_count() == 0 ) {
        return nullptr;
    }
    static SceneDebugInfo info;
    info.aabb_data = scene.debug_aabbs();
    info.num_aabbs = scene.debug_aabb_count();
    info.spheres = scene.debug_spheres();
    info.num_spheres = scene.debug_sphere_count();
    info.quads = scene.debug_quads();
    info.num_quads = scene.debug_quad_count();
    info.boxes = scene.debug_boxes();
    info.num_boxes = scene.debug_box_count();
    return &info;
}
