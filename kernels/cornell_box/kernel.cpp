#include <sycl-sandbox/sandbox_api.h>
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types.h>
#include <sycl-sandbox/rt/trace.h>
#include <sycl-sandbox/rt/params.h>
#include <sycl-sandbox/rt/scene.h>
#include <sycl-sandbox/rt/hittables/quad.h>
#include <sycl-sandbox/rt/hittables/box.h>
#include <sycl-sandbox/rt/materials/lambertian.h>
#include <sycl-sandbox/rt/materials/diffuse_light.h>

#include <cstring>

using namespace rt;
using rt::materials::lambertian;
using rt::materials::diffuse_light;

static ParamMeta params_meta[] = {
    {"light_color",
     "Ceiling light color",
     ParamType::COLOR_RGB,
     .buffer_offset = RT_NUM_STD_PARAMS * sizeof(float),
     .default_c3 = {1, 1, 1}},
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
    return &desc;
}

static constexpr int NUM_OBJECTS = 8;
static Object *g_scene_host = nullptr;
static Object *g_scene_device = nullptr;
static sycl::queue *g_queue = nullptr;

extern "C" void init_kernel(sycl::queue *queue, int, int, const void *params_buffer, size_t) {
    const float *params = (const float *)params_buffer;
    int std_offset = RT_NUM_STD_PARAMS;
    float3 light_color;
    memcpy(&light_color, params + std_offset + PARAM_LIGHT_COLOR, 12);
    float light_strength = params[std_offset + PARAM_LIGHT_STRENGTH];

    g_queue = queue;

    float3 white = {0.73f, 0.73f, 0.73f};
    float3 red = {0.65f, 0.05f, 0.05f};
    float3 green = {0.12f, 0.45f, 0.15f};
    float3 light_emission = scale(light_color, light_strength);

    using hittables::quad;
    using hittables::box;

    g_scene_host = sycl::malloc_host<Object>(NUM_OBJECTS, *queue);

    g_scene_host[0] = {quad(1, 0.0f, -2, 2, -2, 2), lambertian(white)};
    g_scene_host[1] = {quad(1, 3.0f, -2, 2, -2, 2), lambertian(white)};
    g_scene_host[2] = {quad(2, -2.0f, -2, 2, 0, 3), lambertian(white)};
    g_scene_host[3] = {quad(0, -2.0f, -2, 2, 0, 3), lambertian(red)};
    g_scene_host[4] = {quad(0, 2.0f, -2, 2, 0, 3), lambertian(green)};
    g_scene_host[5] = {quad(1, 2.99f, -1, 1, -1, 1), diffuse_light(light_emission)};
    g_scene_host[6] = {box(-0.8f, 0.0f, -0.8f, 0.6f, 1.5f, 0.6f),
                       lambertian({0.55f, 0.55f, 0.55f})};
    g_scene_host[7] = {box(0.8f, 0.0f, -0.3f, 0.6f, 0.6f, 1.2f),
                       lambertian({0.55f, 0.55f, 0.55f})};

    g_scene_device = sycl::malloc_device<Object>(NUM_OBJECTS, *queue);
    queue->memcpy(g_scene_device, g_scene_host, NUM_OBJECTS * sizeof(Object)).wait();
    sycl::free(g_scene_host, *queue);
    g_scene_host = nullptr;
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
                g_scene_device,
                NUM_OBJECTS,
                [](const Ray &) -> float3 {
                    return {0, 0, 0};
                });
}

extern "C" void shutdown_kernel(sycl::queue *queue) {
    if (g_scene_host)   sycl::free(g_scene_host, *queue);
    if (g_scene_device) sycl::free(g_scene_device, *queue);
    g_scene_host   = nullptr;
    g_scene_device = nullptr;
}
