#pragma once
#include "math.h"
#include "types.h"

/// Host-side scene-building helpers for raytracer kernels.
///
/// These functions run on the host (in init_kernel) to construct scene
/// geometry arrays.  They are not used in device (kernel) code.
namespace rt {

/// Compute one corner of an axis-aligned rectangle/quad.
///
/// @param primary_axis    Which axis is fixed (0=X, 1=Y, 2=Z).
/// @param axis_value      Constant coordinate on the primary axis.
/// @param min_second_axis Minimum on the second axis (primary+1 mod 3).
/// @param max_second_axis Maximum on the second axis.
/// @param min_third_axis  Minimum on the third axis (primary+2 mod 3).
/// @param max_third_axis  Maximum on the third axis.
/// @param corner_index    0..3 selecting which corner to compute.
inline float3 quad_corner(int primary_axis, float axis_value,
                          float min_second_axis, float max_second_axis,
                          float min_third_axis, float max_third_axis,
                          int corner_index) {
    int second_axis = (primary_axis + 1) % 3;
    int third_axis  = (primary_axis + 2) % 3;
    float bounds_second[2] = {min_second_axis, max_second_axis};
    float bounds_third[2]  = {min_third_axis, max_third_axis};
    float result[3] = {0, 0, 0};
    result[primary_axis]   = axis_value;
    result[second_axis]    = bounds_second[corner_index & 1];
    result[third_axis]     = bounds_third[corner_index >> 1];
    return {result[0], result[1], result[2]};
}

/// Add an axis-aligned rectangle as two triangles to an Object array.
///
/// This is an alternative to adding individual quads — it builds two
/// triangles (p0,p1,p2) and (p0,p2,p3) from the axis-aligned bounds.
///
/// @param objects       Mutable pointer into the scene array.
/// @param object_count  Current number of objects (incremented by 2).
/// @param axis          Primary axis (0=X, 1=Y, 2=Z).
/// @param axis_value    Constant coordinate on the primary axis.
/// @param min_second, max_second  Range on the second axis.
/// @param min_third, max_third    Range on the third axis.
/// @param material      Material for the quad.
inline void add_quad(Object* objects, int& object_count,
                     int axis, float axis_value,
                     float min_second, float max_second,
                     float min_third, float max_third,
                     Material material) {
    float3 p0 = quad_corner(axis, axis_value, min_second, max_second, min_third, max_third, 0);
    float3 p1 = quad_corner(axis, axis_value, min_second, max_second, min_third, max_third, 1);
    float3 p2 = quad_corner(axis, axis_value, min_second, max_second, min_third, max_third, 2);
    float3 p3 = quad_corner(axis, axis_value, min_second, max_second, min_third, max_third, 3);
    objects[object_count++] = {hittables::quad(p0, p1, p2), material};
    objects[object_count++] = {hittables::quad(p0, p2, p3), material};
}

/// Add an axis-aligned box (6 faces, 12 triangles) to an Object array.
///
/// @param objects, object_count  Scene array (incremented by 12).
/// @param corner_x, corner_y, corner_z  Minimum corner of the box.
/// @param size_x, size_y, size_z        Extents of the box along each axis.
/// @param material                      Material for all six faces.
inline void add_box(Object* objects, int& object_count,
                    float corner_x, float corner_y, float corner_z,
                    float size_x, float size_y, float size_z,
                    Material material) {
    add_quad(objects, object_count, 1, corner_y + size_y, corner_x, corner_x + size_x, corner_z, corner_z + size_z, material);
    add_quad(objects, object_count, 1, corner_y,           corner_x, corner_x + size_x, corner_z, corner_z + size_z, material);
    add_quad(objects, object_count, 2, corner_z,           corner_x, corner_x + size_x, corner_y, corner_y + size_y, material);
    add_quad(objects, object_count, 2, corner_z + size_z,  corner_x, corner_x + size_x, corner_y, corner_y + size_y, material);
    add_quad(objects, object_count, 0, corner_x,           corner_z, corner_z + size_z, corner_y, corner_y + size_y, material);
    add_quad(objects, object_count, 0, corner_x + size_x,  corner_z, corner_z + size_z, corner_y, corner_y + size_y, material);
}

} // namespace rt
