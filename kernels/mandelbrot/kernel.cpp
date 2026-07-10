#include "sandbox_api.h"
#include <sycl/sycl.hpp>

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

extern "C" void render_kernel(sycl::queue *q, int w, int h, const void *params, void *accum, int) {
    auto *p = (const float *)params;
    float cx = p[P_CENTER_X];
    float cy = p[P_CENTER_Y];
    float zoom = p[P_ZOOM];
    int max_iter = (int)p[P_MAX_ITER];
    float aspect = (float)w / (float)h;

    auto *acc = (float *)accum;

    q->parallel_for(sycl::range<2> {(size_t)h, (size_t)w}, [=](sycl::item<2> it) {
         int x = it[1], y = it[0], idx = y * w + x;

         float px = (x + 0.5f) / (float)w;
         float py = (y + 0.5f) / (float)h;
         float map_x = cx + (px - 0.5f) * zoom * aspect;
         float map_y = cy + (py - 0.5f) * zoom;

         float zx = 0.0f, zy = 0.0f;
         float zx2 = 0.0f, zy2 = 0.0f;
         int iter = 0;
         while ( iter < max_iter && zx2 + zy2 < 4.0f ) {
             zy = 2.0f * zx * zy + map_y;
             zx = zx2 - zy2 + map_x;
             zx2 = zx * zx;
             zy2 = zy * zy;
             iter++;
         }

         float col_r, col_g, col_b;
         if ( iter == max_iter ) {
             col_r = 0.0f;
             col_g = 0.0f;
             col_b = 0.0f;
         } else {
             float t = (float)iter / 50.0f;
             if ( t > 1.0f )
                 t = 1.0f;
             float h = t;
             float s = 0.9f;
             float v = 0.8f + t * 0.2f;
             float h6 = h * 6.0f;
             int hi = (int)h6;
             float f = h6 - (float)hi;
             float p = v * (1.0f - s);
             float q = v * (1.0f - s * f);
             float tt = v * (1.0f - s * (1.0f - f));
             switch ( hi % 6 ) {
                 case 0:
                     col_r = v;
                     col_g = tt;
                     col_b = p;
                     break;
                 case 1:
                     col_r = q;
                     col_g = v;
                     col_b = p;
                     break;
                 case 2:
                     col_r = p;
                     col_g = v;
                     col_b = tt;
                     break;
                 case 3:
                     col_r = p;
                     col_g = q;
                     col_b = v;
                     break;
                 case 4:
                     col_r = tt;
                     col_g = p;
                     col_b = v;
                     break;
                 default:
                     col_r = v;
                     col_g = p;
                     col_b = q;
                     break;
             }
         }

         int base = idx * 4;
         acc[base + 0] += col_r;
         acc[base + 1] += col_g;
         acc[base + 2] += col_b;
         acc[base + 3] += 1;
     }).wait();
}

extern "C" void shutdown_kernel(sycl::queue *) {
}
