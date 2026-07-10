#pragma once
#include <sycl-sandbox/rt/math.h>
#include <sycl-sandbox/rt/types_fwd.h>
#include <sycl-sandbox/rt/hittables/quad.h>
#include <sycl-sandbox/optional.h>
#include <array>

/// Box primitive composed from six Quad faces.
///
/// The box is axis-aligned, defined by its minimum corner and extents.
/// Intersection is tested against all six faces — the closest hit wins.
///
/// Factory:  rt::hittables::box(corner_x, corner_y, corner_z,
///                               size_x, size_y, size_z) -> Box
namespace rt::hittables {

class Box {
public:
    std::array<Quad, 6> faces;

    Box() = default;

    /// Build an axis-aligned box from a minimum corner and extents.
    Box(float cx, float cy, float cz, float sx, float sy, float sz) {
        float x0 = cx, x1 = cx + sx;
        float y0 = cy, y1 = cy + sy;
        float z0 = cz, z1 = cz + sz;
        faces[0] = Quad({x0, y1, z0}, {sx, 0, 0}, {0, 0, sz}); // top    (+Y)
        faces[1] = Quad({x0, y0, z0}, {0, 0, sz}, {sx, 0, 0}); // bottom (-Y)
        faces[2] = Quad({x0, y0, z0}, {0, sy, 0}, {sx, 0, 0}); // front  (-Z)
        faces[3] = Quad({x0, y0, z1}, {sx, 0, 0}, {0, sy, 0}); // back   (+Z)
        faces[4] = Quad({x0, y0, z0}, {0, sy, 0}, {0, 0, sz}); // left   (-X)
        faces[5] = Quad({x1, y0, z0}, {0, 0, sz}, {0, sy, 0}); // right  (+X)
    }

    /// Iterate over all six faces and return the closest hit.
    optional<HitRecord> hit(const Ray &ray, float t_min, float t_max) const {
        optional<HitRecord> closest;
        for ( int i = 0; i < 6; i++ ) {
            auto hit = faces[i].hit(ray, t_min, closest ? closest->t : t_max);
            if ( hit ) {
                closest = hit;
            }
        }
        return closest;
    }
};

/// Creates an axis-aligned Box from its minimum corner and extents.
inline Box box(float cx, float cy, float cz, float sx, float sy, float sz) {
    return Box(cx, cy, cz, sx, sy, sz);
}

} // namespace rt::hittables
