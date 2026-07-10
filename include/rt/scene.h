#pragma once
#include "math.h"
#include "types.h"

/// Host-side scene-building helpers for raytracer kernels.
namespace rt {

/// Which axis of a 3D coordinate system.
enum class Axis : int { X = 0, Y = 1, Z = 2 };

/// Compute one corner of an axis-aligned rectangle/quad.
inline float3 quad_corner(Axis primary_axis,
                          float axis_value,
                          float min_second_axis,
                          float max_second_axis,
                          float min_third_axis,
                          float max_third_axis,
                          int corner_index) {
    int pa = static_cast<int>(primary_axis);
    int second_axis = (pa + 1) % 3;
    int third_axis = (pa + 2) % 3;
    float bounds_second[2] = {min_second_axis, max_second_axis};
    float bounds_third[2] = {min_third_axis, max_third_axis};
    float result[3] = {0, 0, 0};
    result[pa] = axis_value;
    result[second_axis] = bounds_second[corner_index & 1];
    result[third_axis] = bounds_third[corner_index >> 1];
    return {result[0], result[1], result[2]};
}

/// Add an axis-aligned rectangle as a single Quad to an Object array.
inline void add_quad(Object *objects,
                     int &object_count,
                     Axis axis,
                     float axis_value,
                     float min_second,
                     float max_second,
                     float min_third,
                     float max_third,
                     Material material) {
    float3 p0 = quad_corner(axis, axis_value, min_second, max_second, min_third, max_third, 0);
    float3 p1 = quad_corner(axis, axis_value, min_second, max_second, min_third, max_third, 1);
    float3 p2 = quad_corner(axis, axis_value, min_second, max_second, min_third, max_third, 2);
    objects[object_count++] = {hittables::quad_from_corners(p0, p1, p2), material};
}

/// Add an axis-aligned box (6 faces, one Quad each) to an Object array.
inline void add_box(Object *objects,
                    int &object_count,
                    float corner_x,
                    float corner_y,
                    float corner_z,
                    float size_x,
                    float size_y,
                    float size_z,
                    Material material) {
    add_quad(objects,
             object_count,
             Axis::Y,
             corner_y + size_y,
             corner_x,
             corner_x + size_x,
             corner_z,
             corner_z + size_z,
             material);
    add_quad(objects,
             object_count,
             Axis::Y,
             corner_y,
             corner_x,
             corner_x + size_x,
             corner_z,
             corner_z + size_z,
             material);
    add_quad(objects,
             object_count,
             Axis::Z,
             corner_z,
             corner_x,
             corner_x + size_x,
             corner_y,
             corner_y + size_y,
             material);
    add_quad(objects,
             object_count,
             Axis::Z,
             corner_z + size_z,
             corner_x,
             corner_x + size_x,
             corner_y,
             corner_y + size_y,
             material);
    add_quad(objects,
             object_count,
             Axis::X,
             corner_x,
             corner_z,
             corner_z + size_z,
             corner_y,
             corner_y + size_y,
             material);
    add_quad(objects,
             object_count,
             Axis::X,
             corner_x + size_x,
             corner_z,
             corner_z + size_z,
             corner_y,
             corner_y + size_y,
             material);
}

} // namespace rt
