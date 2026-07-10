#pragma once

// Standard parameter indices for all raytracer kernels.
// Each kernel MUST place these at the beginning of its ParamMeta array
// in this exact order. Kernel-specific params start at index 13.
enum rt_std_param {
    RT_SPP_FRAME    = 0,   // int
    RT_MAX_BOUNCES  = 1,   // int
    RT_CAM_EYE      = 2,   // VEC3 (floats 2,3,4)
    RT_CAM_AT       = 5,   // VEC3 (floats 5,6,7)
    RT_CAM_FOV      = 8,   // float
    RT_CAM_APERTURE = 9,   // float
    RT_CAM_UP       = 10,  // VEC3 (floats 10,11,12)
    RT_NUM_STD_PARAMS = 13 // first kernel-specific index
};
