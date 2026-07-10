#pragma once

/// Standard parameter indices for all raytracer kernels.
///
/// Every raytracer kernel MUST place these seven parameters at the
/// beginning of its param_meta array, in this exact order, so that
/// rt::render_main() can read them without knowing the kernel's full
/// parameter layout.  Kernel-specific parameters start at index 13.
enum class rt_std_param : int {
    RT_SPP_FRAME    = 0,   ///< int,   samples per frame
    RT_MAX_BOUNCES  = 1,   ///< int,   maximum ray path depth
    RT_CAM_EYE      = 2,   ///< VEC3,  camera position (floats 2,3,4)
    RT_CAM_AT       = 5,   ///< VEC3,  look-at target (floats 5,6,7)
    RT_CAM_FOV      = 8,   ///< float, vertical field of view (degrees)
    RT_CAM_APERTURE = 9,   ///< float, depth-of-field aperture size
    RT_CAM_UP       = 10,  ///< VEC3,  camera up vector (floats 10,11,12)
    RT_NUM_STD_PARAMS = 13 ///< first index available for kernel-specific params
};

/// Convenience: cast an rt_std_param enumerator to int for array indexing.
inline constexpr int rti(rt_std_param e) { return static_cast<int>(e); }
