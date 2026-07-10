#include "sandbox_api.h"
#include <sycl/sycl.hpp>
#include <cstring>

static ParamMeta params_meta[] = {
    {"center_x",
     "Real component of center",
     ParamType::FLOAT,
     .range = {.f = {-2.0f, 2.0f, 0.01f}},
     .default_f = -0.5f},
    {"center_y",
     "Imaginary component of center",
     ParamType::FLOAT,
     .range = {.f = {-2.0f, 2.0f, 0.01f}},
     .default_f = 0.0f},
    {"zoom",
     "Zoom level (larger = more zoomed in)",
     ParamType::FLOAT,
     .range = {.f = {0.1f, 100.0f, 0.1f}},
     .default_f = 3.0f},
    {"max_iterations",
     "Maximum iterations per point",
     ParamType::INT,
     .range = {.i = {10, 10000, 10}},
     .default_i = 300},
};

enum ParamIdx : int {
    P_CENTER_X = 0,
    P_CENTER_Y = 1,
    P_ZOOM = 2,
    P_MAX_ITER = 3,
};

static const char *source_files[] = {"kernel.cpp", nullptr};
static KernelDesc desc =
    {"mandelbrot", "Colored Mandelbrot fractal", 4, params_meta, 0, 1, 2, source_files};

extern "C" KernelDesc *get_kernel_desc() {
    desc.params_buffer_size = 0;
    for ( int i = 0; i < desc.param_count; i++ )
        desc.params_buffer_size += param_buffer_size(params_meta[i]);
    return &desc;
}

extern "C" void init_kernel(sycl::queue *, int, int, const void *, size_t) {
}

extern "C" void render_kernel(sycl::queue *queue,
                              int width,
                              int height,
                              const void *params,
                              void *accum_buffer,
                              int) {
    auto *params_data = (const float *)params;
    float center_x = params_data[P_CENTER_X];
    float center_y = params_data[P_CENTER_Y];
    float zoom_level = params_data[P_ZOOM];
    int max_iterations = (int)params_data[P_MAX_ITER];
    float aspect_ratio = (float)width / (float)height;

    auto *accum = (float *)accum_buffer;

    queue
        ->parallel_for(sycl::range<2> {(size_t)height, (size_t)width},
                       [=](sycl::item<2> pixel) {
                           int x = pixel[1], y = pixel[0], flat_index = y * width + x;

                           // Map the pixel to a point in the complex plane
                           float px = (x + 0.5f) / (float)width;
                           float py = (y + 0.5f) / (float)height;
                           float map_x = center_x + (px - 0.5f) * zoom_level * aspect_ratio;
                           float map_y = center_y + (py - 0.5f) * zoom_level;

                           // Iterate the Mandelbrot recurrence:  z ← z² + c
                           float zx = 0.0f, zy = 0.0f;
                           float zx2 = 0.0f, zy2 = 0.0f;
                           int iteration = 0;

                           while ( iteration < max_iterations && zx2 + zy2 < 4.0f ) {
                               zy = 2.0f * zx * zy + map_y;
                               zx = zx2 - zy2 + map_x;
                               zx2 = zx * zx;
                               zy2 = zy * zy;
                               iteration++;
                           }

                           // Convert the iteration count to an HSV colour, then to RGB
                           float red, green, blue;
                           if ( iteration == max_iterations ) {
                               // Interior of the set — render as black
                               red = 0.0f;
                               green = 0.0f;
                               blue = 0.0f;
                           } else {
                               // Map iteration count to a hue, then apply HSV-to-RGB conversion
                               float t = (float)iteration / 50.0f;
                               if ( t > 1.0f )
                                   t = 1.0f;

                               float hue = t;
                               float saturation = 0.9f;
                               float value = 0.8f + t * 0.2f;

                               // Standard HSV-to-RGB conversion
                               float hue_sector = hue * 6.0f;
                               int sector_index = (int)hue_sector;
                               float fractional_part = hue_sector - (float)sector_index;

                               float p = value * (1.0f - saturation);
                               float q = value * (1.0f - saturation * fractional_part);
                               float t_component =
                                   value * (1.0f - saturation * (1.0f - fractional_part));

                               switch ( sector_index % 6 ) {
                                   case 0:
                                       red = value;
                                       green = t_component;
                                       blue = p;
                                       break;
                                   case 1:
                                       red = q;
                                       green = value;
                                       blue = p;
                                       break;
                                   case 2:
                                       red = p;
                                       green = value;
                                       blue = t_component;
                                       break;
                                   case 3:
                                       red = p;
                                       green = q;
                                       blue = value;
                                       break;
                                   case 4:
                                       red = t_component;
                                       green = p;
                                       blue = value;
                                       break;
                                   default:
                                       red = value;
                                       green = p;
                                       blue = q;
                                       break;
                               }
                           }

                           // Accumulate the colour (single frame: max_spp = 1, no averaging needed)
                           int base = flat_index * 4;
                           accum[base + 0] += red;
                           accum[base + 1] += green;
                           accum[base + 2] += blue;
                           accum[base + 3] += 1;
                       })
        .wait();
}

extern "C" void shutdown_kernel(sycl::queue *) {
}
