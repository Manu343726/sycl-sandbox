#include "sandbox_api.h"
#include "kernel.h"
#include "rt/math.h"
#include "rt/types.h"
#include "rt/camera.h"
#include "rt/trace.h"
#include "rt/params.h"
#include "rt/hittables/sphere.h"
#include "rt/materials/lambertian.h"
#include "rt/materials/metal.h"
#include "rt/materials/dielectric.h"
#include <cstdlib>
#include <cstring>

using namespace rt;
using rt::hittables::sphere;
using rt::materials::lambertian;
using rt::materials::metal;
using rt::materials::dielectric;

static ParamMeta params_meta[] = {
    {"spp_frame","Samples per frame",ParamType::INT,.range={.i={1,64,1}},.default_i=1},
    {"max_bounces","Maximum ray path depth",ParamType::INT,.range={.i={1,100,1}},.default_i=10},
    {"cam_eye","Camera position",ParamType::VEC3,.default_c3={13,2,3}},
    {"cam_at","Camera look-at target",ParamType::VEC3,.default_c3={0,0,0}},
    {"cam_fov","Vertical field of view",ParamType::FLOAT,.range={.f={1,120,1}},.default_f=20},
    {"cam_aperture","Depth of field aperture",ParamType::FLOAT,.range={.f={0,1,0.01f}},.default_f=0.1f},
    {"cam_up","Camera up vector",ParamType::VEC3,.default_c3={0,1,0}},
    {"num_spheres","Number of random small spheres",ParamType::INT,.range={.i={0,500,1}},.default_i=11},
    {"ground_color","Ground sphere albedo",ParamType::COLOR_RGB,.default_c3={0.5f,0.5f,0.5f}},
    {"background","Sky colour",ParamType::COLOR_RGB,.default_c3={0.5f,0.7f,1}},
};
enum ParamIndex : int {
    PARAM_NUM_SPHERES    = static_cast<int>(rt_std_param::RT_NUM_STD_PARAMS),
    PARAM_GROUND_COLOR   = PARAM_NUM_SPHERES + 1,
    PARAM_BACKGROUND     = PARAM_GROUND_COLOR + 3,
};
static KernelDesc desc={
    "one_weekend","Raytracing in One Weekend — random spheres",
    10,params_meta,0,4096,2,
    (const char*[]){"kernel.cpp","kernel.h",nullptr}
};
extern "C" KernelDesc* get_kernel_desc(){
    desc.params_buffer_size=0;
    for(int i=0;i<desc.param_count;i++)
        desc.params_buffer_size+=param_buffer_size(params_meta[i]);
    return &desc;
}

static Object*  g_scene_objects   = nullptr;
static int      g_num_objects     = 0;
static float3   g_background      = {0.5f, 0.7f, 1.0f};

static float random_float() {
    return (float)rand() * (1.f / 2147483647.f);
}

extern "C" void init_kernel(sycl::queue* queue, int, int,
                            const void* params_buffer, size_t) {
    const float* params = (const float*)params_buffer;
    int num_spheres = (int)params[PARAM_NUM_SPHERES];
    float3 ground_color; memcpy(&ground_color, params + PARAM_GROUND_COLOR, 12);
    memcpy(&g_background, params + PARAM_BACKGROUND, 12);

    if (g_scene_objects) { sycl::free(g_scene_objects, *queue); g_scene_objects = nullptr; }
    srand(42);

    int max_objects = num_spheres + 4 + 3;
    auto* objects = new Object[max_objects];
    int object_count = 0;

    objects[object_count++] = {sphere({0,-1000,0}, 1000), lambertian(ground_color)};

    for (int k = 0; k < num_spheres; ) {
        float x = -10.0f + 20.0f * random_float();
        float z = -10.0f + 20.0f * random_float();
        float3 center = {x, 0.2f, z};
        if (len(center) <= 0.9f) continue;

        float3 color = {random_float()*random_float(),
                        random_float()*random_float(),
                        random_float()*random_float()};
        float choice = random_float();

        if (choice < 0.6f)
            objects[object_count++] = {sphere(center, 0.2f), lambertian(color)};
        else if (choice < 0.85f)
            objects[object_count++] = {sphere(center, 0.2f), metal(color, 0.5f*random_float())};
        else
            objects[object_count++] = {sphere(center, 0.2f), dielectric(1.5f)};
        k++;
    }

    objects[object_count++] = {sphere({4,1,0}, 1), metal({0.7f,0.6f,0.5f}, 0)};
    objects[object_count++] = {sphere({-4,1,0}, 1), lambertian({0.4f,0.2f,0.1f})};
    objects[object_count++] = {sphere({0,1,0}, 1), dielectric(1.5f)};

    g_scene_objects = sycl::malloc_device<Object>(object_count, *queue);
    queue->memcpy(g_scene_objects, objects, object_count * sizeof(Object)).wait();
    g_num_objects = object_count;
    delete[] objects;
}

extern "C" void render_kernel(sycl::queue* queue, int width, int height,
                               const void* params_buffer, void* accum_buffer, int sample_index) {
    const float* params = (const float*)params_buffer;
    render_main(queue, width, height, params, (float*)accum_buffer, sample_index,
                g_scene_objects, g_num_objects,
                [background = g_background](const Ray& ray) -> float3 {
                    float t = 0.5f * (ray.dir.y + 1.0f);
                    return lerp({1,1,1}, background, t);
                });
}

extern "C" void shutdown_kernel(sycl::queue* queue) {
    if (g_scene_objects) { sycl::free(g_scene_objects, *queue); g_scene_objects = nullptr; }
}
