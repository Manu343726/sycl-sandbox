#pragma once
#include <sycl-sandbox/param_types.h>
#include <sycl/sycl.hpp>
#include <cstdint>

struct KernelDesc {
    const char*      name;
    const char*      description;
    int32_t          param_count;
    const ParamMeta* params;
    size_t           params_buffer_size;
    int32_t          max_spp;       // max samples/pixel this kernel benefits from (1 = single-frame)
    int32_t          source_count;
    const char**     sources;
};

extern "C" KernelDesc* get_kernel_desc();
extern "C" void        init_kernel(sycl::queue*, int w, int h,
                                   const void* params, size_t params_size);
extern "C" void        render_kernel(sycl::queue*, int w, int h,
                                     const void* params,
                                     void* accum_buffer, int sample_index);
extern "C" void        shutdown_kernel(sycl::queue*);
