#pragma once
#include "math.h"
#include "types.h"

/// Host-side scene-building helpers for raytracer kernels.
namespace rt {

/// Which axis of a 3D coordinate system.
enum class Axis : int { X = 0, Y = 1, Z = 2 };

/// Compute one corner of an axis-aligned rectangle.
inline float3 quad_corner(Axis primary_axis,
                          float axis_value,
                          float min_second_axis,
                          float max_second_axis,
                          float min_third_axis,
                          float max_third_axis,
                          int corner_index) {
    int primary = static_cast<int>(primary_axis);
    int second_axis = (primary + 1) % 3;
    int third_axis = (primary + 2) % 3;
    float bounds_second[2] = {min_second_axis, max_second_axis};
    float bounds_third[2] = {min_third_axis, max_third_axis};
    float result[3] = {0, 0, 0};
    result[primary] = axis_value;
    result[second_axis] = bounds_second[corner_index & 1];
    result[third_axis] = bounds_third[corner_index >> 1];
    return {result[0], result[1], result[2]};
}

/// Append a pre-built Object to a scene array.
inline void add(Object *objects, int &object_count, Object object) {
    objects[object_count++] = std::move(object);
}

} // namespace rt
