#pragma once
#include "../math.h"
#include "../types_fwd.h"
#include "quad.h"
#include <optional>
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
        faces[0] = quad(1, cy + sy, cx, cx + sx, cz, cz + sz); // top
        faces[1] = quad(1, cy, cx, cx + sx, cz, cz + sz);      // bottom
        faces[2] = quad(2, cz, cx, cx + sx, cy, cy + sy);      // front
        faces[3] = quad(2, cz + sz, cx, cx + sx, cy, cy + sy); // back
        faces[4] = quad(0, cx, cz, cz + sz, cy, cy + sy);      // left
        faces[5] = quad(0, cx + sx, cz, cz + sz, cy, cy + sy); // right
    }

    /// Iterate over all six faces and return the closest hit.
    std::optional<HitRecord> hit(const Ray &ray, float t_min, float t_max) const {
        std::optional<HitRecord> closest;
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
