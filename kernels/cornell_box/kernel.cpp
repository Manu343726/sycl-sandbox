#include "sandbox_api.h"
#include "rt/math.h"
#include "rt/types.h"
#include "rt/camera.h"
#include "rt/trace.h"
#include "rt/params.h"
#include "rt/scene.h"
#include "rt/hittables/quad.h"
#include "rt/materials/lambertian.h"
#include "rt/materials/diffuse_light.h"
#include <cstring>

using namespace rt;
using rt::materials::lambertian;
using rt::materials::diffuse_light;
using rt::Axis;

static ParamMeta params_meta[] = {
    {"light_color", "Ceiling light color", ParamType::COLOR_RGB, .default_c3 = {1, 1, 1}},
    {"light_strength",
     "Ceiling light intensity",
     ParamType::FLOAT,
     .range = {.f = {1, 50, 1}},
     .default_f = 15},
};

enum { PARAM_LIGHT_COLOR = 0, PARAM_LIGHT_STRENGTH = PARAM_LIGHT_COLOR + 3 };

static KernelDesc desc = {"cornell_box",
                          "Cornell box scene with quads",
                          2,
                          params_meta,
                          0,
                          4096,
                          1,
                          (const char *[]) {"kernel.cpp", nullptr}};
extern "C" KernelDesc *get_kernel_desc() {
    desc.params_buffer_size = RT_NUM_STD_PARAMS * sizeof(float);
    for ( int i = 0; i < desc.param_count; i++ ) {
        desc.params_buffer_size += param_buffer_size(params_meta[i]);
    }
    return &desc;
}

static Object *g_scene_objects = nullptr;
static int g_num_objects = 0;

/// Build four corners of an axis-aligned rectangle and add it as a Quad.
static void add_quad(Object *objects,
                     int &count,
                     Axis primary,
                     float value,
                     float min_s,
                     float max_s,
                     float min_t,
                     float max_t,
                     Material material) {
    float3 p0 = quad_corner(primary, value, min_s, max_s, min_t, max_t, 0);
    float3 p1 = quad_corner(primary, value, min_s, max_s, min_t, max_t, 1);
    float3 p2 = quad_corner(primary, value, min_s, max_s, min_t, max_t, 2);
    add(objects, count, {hittables::quad_from_corners(p0, p1, p2), std::move(material)});
}

/// Build an axis-aligned box (six Quad faces).
static void add_box(Object *objects,
                    int &count,
                    float cx,
                    float cy,
                    float cz,
                    float sx,
                    float sy,
                    float sz,
                    Material material) {
    add_quad(objects, count, Axis::Y, cy + sy, cx, cx + sx, cz, cz + sz, material);
    add_quad(objects, count, Axis::Y, cy, cx, cx + sx, cz, cz + sz, material);
    add_quad(objects, count, Axis::Z, cz, cx, cx + sx, cy, cy + sy, material);
    add_quad(objects, count, Axis::Z, cz + sz, cx, cx + sx, cy, cy + sy, material);
    add_quad(objects, count, Axis::X, cx, cz, cz + sz, cy, cy + sy, material);
    add_quad(objects, count, Axis::X, cx + sx, cz, cz + sz, cy, cy + sy, material);
}

extern "C" void init_kernel(sycl::queue *queue, int, int, const void *params_buffer, size_t) {
    const float *params = (const float *)params_buffer;
    int std_offset = RT_NUM_STD_PARAMS;
    float3 light_color;
    memcpy(&light_color, params + std_offset + PARAM_LIGHT_COLOR, 12);
    float light_strength = params[std_offset + PARAM_LIGHT_STRENGTH];

    if ( g_scene_objects ) {
        sycl::free(g_scene_objects, *queue);
        g_scene_objects = nullptr;
    }

    auto *objects = new Object[64];
    int object_count = 0;

    float3 white = {0.73f, 0.73f, 0.73f};
    float3 red = {0.65f, 0.05f, 0.05f};
    float3 green = {0.12f, 0.45f, 0.15f};
    float3 light_emission = scale(light_color, light_strength);

    add_quad(objects, object_count, Axis::Y, 0.0f, -2, 2, -2, 2, lambertian(white));
    add_quad(objects, object_count, Axis::Y, 3.0f, -2, 2, -2, 2, lambertian(white));
    add_quad(objects, object_count, Axis::Z, -2.0f, -2, 2, 0, 3, lambertian(white));
    add_quad(objects, object_count, Axis::X, -2.0f, -2, 2, 0, 3, lambertian(red));
    add_quad(objects, object_count, Axis::X, 2.0f, -2, 2, 0, 3, lambertian(green));
    add_quad(objects, object_count, Axis::Y, 2.99f, -1, 1, -1, 1, diffuse_light(light_emission));

    add_box(objects,
            object_count,
            -0.8f,
            0.0f,
            -0.8f,
            0.6f,
            1.5f,
            0.6f,
            lambertian({0.55f, 0.55f, 0.55f}));
    add_box(objects,
            object_count,
            0.8f,
            0.0f,
            -0.3f,
            0.6f,
            0.6f,
            1.2f,
            lambertian({0.55f, 0.55f, 0.55f}));

    g_scene_objects = sycl::malloc_device<Object>(object_count, *queue);
    queue->memcpy(g_scene_objects, objects, object_count * sizeof(Object)).wait();
    g_num_objects = object_count;
    delete[] objects;
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
                g_scene_objects,
                g_num_objects,
                [](const Ray &) -> float3 {
                    return {0, 0, 0};
                });
}

extern "C" void shutdown_kernel(sycl::queue *queue) {
    if ( g_scene_objects ) {
        sycl::free(g_scene_objects, *queue);
        g_scene_objects = nullptr;
    }
}
