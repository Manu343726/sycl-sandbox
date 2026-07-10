#include "sandbox_api.h"
#include "rt/math.h"
#include "rt/types.h"
#include "rt/camera.h"
#include "rt/trace.h"
#include "rt/params.h"
#include "rt/hittables/quad.h"
#include "rt/materials/lambertian.h"
#include "rt/materials/diffuse_light.h"
#include <cstring>

using namespace rt;
using rt::hittables::quad;
using rt::materials::lambertian;
using rt::materials::diffuse_light;

static ParamMeta params_meta[]={
    {"spp_frame","Samples per frame",ParamType::INT,.range={.i={1,64,1}},.default_i=1},
    {"max_bounces","Maximum ray path depth",ParamType::INT,.range={.i={1,100,1}},.default_i=10},
    {"cam_eye","Camera position",ParamType::VEC3,.default_c3={0,1.5f,4.5f}},
    {"cam_at","Camera look-at target",ParamType::VEC3,.default_c3={0,1.2f,0}},
    {"cam_fov","Vertical field of view",ParamType::FLOAT,.range={.f={1,120,1}},.default_f=35},
    {"cam_aperture","Depth of field aperture",ParamType::FLOAT,.range={.f={0,1,0.01f}},.default_f=0},
    {"cam_up","Camera up vector",ParamType::VEC3,.default_c3={0,1,0}},
    {"light_color","Ceiling light color",ParamType::COLOR_RGB,.default_c3={1,1,1}},
    {"light_strength","Ceiling light intensity",ParamType::FLOAT,.range={.f={1,50,1}},.default_f=15},
};
enum ParamIndex {
    PARAM_LIGHT_COLOR     = RT_NUM_STD_PARAMS,
    PARAM_LIGHT_STRENGTH  = PARAM_LIGHT_COLOR + 3,
};
static KernelDesc desc={
    "cornell_box","Cornell box scene with quads",9,params_meta,0,4096,1,
    (const char*[]){"kernel.cpp",nullptr}
};
extern "C" KernelDesc* get_kernel_desc(){
    desc.params_buffer_size=0;
    for(int i=0;i<desc.param_count;i++)
        desc.params_buffer_size+=param_buffer_size(params_meta[i]);
    return &desc;
}

static Object* g_scene_objects = nullptr;
static int     g_num_objects   = 0;

static float3 compute_corner(int axis, float axis_value,
                              float min_b, float max_b,
                              float min_c, float max_c,
                              int corner_index) {
    int second_axis = (axis + 1) % 3;
    int third_axis  = (axis + 2) % 3;
    float bounds_second[2] = {min_b, max_b};
    float bounds_third[2]  = {min_c, max_c};
    float result[3] = {0, 0, 0};
    result[axis]        = axis_value;
    result[second_axis] = bounds_second[corner_index & 1];
    result[third_axis]  = bounds_third[corner_index >> 1];
    return {result[0], result[1], result[2]};
}

extern "C" void init_kernel(sycl::queue* queue, int, int,
                            const void* params_buffer, size_t) {
    const float* params = (const float*)params_buffer;
    float3 light_color; memcpy(&light_color, params + PARAM_LIGHT_COLOR, 12);
    float light_strength = params[PARAM_LIGHT_STRENGTH];

    if (g_scene_objects) { sycl::free(g_scene_objects, *queue); g_scene_objects = nullptr; }

    auto* objects = new Object[64];
    int object_count = 0;

    float3 white  = {0.73f, 0.73f, 0.73f};
    float3 red    = {0.65f, 0.05f, 0.05f};
    float3 green  = {0.12f, 0.45f, 0.15f};
    float3 zero   = {0, 0, 0};
    float3 light_emission = scale(light_color, light_strength);

    auto add_quad_as_two_triangles = [&](int axis, float axis_value,
                                          float min_b, float max_b,
                                          float min_c, float max_c,
                                          Material material) {
        float3 p0 = compute_corner(axis, axis_value, min_b, max_b, min_c, max_c, 0);
        float3 p1 = compute_corner(axis, axis_value, min_b, max_b, min_c, max_c, 1);
        float3 p2 = compute_corner(axis, axis_value, min_b, max_b, min_c, max_c, 2);
        float3 p3 = compute_corner(axis, axis_value, min_b, max_b, min_c, max_c, 3);
        objects[object_count++] = {quad(p0, p1, p2), material};
        objects[object_count++] = {quad(p0, p2, p3), material};
    };

    auto add_box = [&](float center_x, float center_y, float center_z,
                        float box_width, float box_height, float box_depth,
                        Material material) {
        add_quad_as_two_triangles(1, center_y + box_height, center_x, center_x + box_width,  center_z, center_z + box_depth,  material);
        add_quad_as_two_triangles(1, center_y,              center_x, center_x + box_width,  center_z, center_z + box_depth,  material);
        add_quad_as_two_triangles(2, center_z,              center_x, center_x + box_width,  center_y, center_y + box_height, material);
        add_quad_as_two_triangles(2, center_z + box_depth,  center_x, center_x + box_width,  center_y, center_y + box_height, material);
        add_quad_as_two_triangles(0, center_x,              center_z, center_z + box_depth,  center_y, center_y + box_height, material);
        add_quad_as_two_triangles(0, center_x + box_width,  center_z, center_z + box_depth,  center_y, center_y + box_height, material);
    };

    // Room walls (axis: 0=X, 1=Y, 2=Z)
    add_quad_as_two_triangles(1, 0.0f,   -2, 2, -2, 2, lambertian(white));
    add_quad_as_two_triangles(1, 3.0f,   -2, 2, -2, 2, lambertian(white));
    add_quad_as_two_triangles(2, -2.0f,  -2, 2,  0, 3, lambertian(white));
    add_quad_as_two_triangles(0, -2.0f,  -2, 2,  0, 3, lambertian(red));
    add_quad_as_two_triangles(0, 2.0f,   -2, 2,  0, 3, lambertian(green));

    // Ceiling light
    add_quad_as_two_triangles(1, 2.99f, -1, 1, -1, 1, diffuse_light(light_emission));

    // Tall box
    add_box(-0.8f, 0.0f, -0.8f,  0.6f, 1.5f, 0.6f, lambertian({0.55f, 0.55f, 0.55f}));
    // Short box
    add_box(0.8f, 0.0f, -0.3f,  0.6f, 0.6f, 1.2f, lambertian({0.55f, 0.55f, 0.55f}));

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
                [](const Ray&) -> float3 { return {0, 0, 0}; });
}

extern "C" void shutdown_kernel(sycl::queue* queue) {
    if (g_scene_objects) { sycl::free(g_scene_objects, *queue); g_scene_objects = nullptr; }
}
