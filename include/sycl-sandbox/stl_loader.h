#pragma once

/// @file
/// Host-side STL (stereolithography) triangle mesh loader.
///
/// Reads both the ASCII and the binary STL variants, normalizes the
/// result to `rt::hittables::Triangle` values, and applies optional
/// per-object placement transforms (position / rotation / scale).
///
/// HOST-ONLY: uses standard library file I/O, so this header must NOT
/// be included from code compiled for SYCL device execution (rt/
/// headers and kernels).  The scene loader (`src/scene/loader.cpp`)
/// includes it to turn `type: mesh` YAML objects into scene geometry.
///
/// File format (both variants):
///   - ASCII: `solid <name>` … `facet normal` / `outer loop` / three
///     `vertex x y z` / `endloop` / `endfacet` … `endsolid <name>`.
///     Whitespace- and case-tolerant token parsing (CRLF-safe).
///   - Binary: 80-byte header, little-endian uint32 triangle count,
///     then per triangle: 3×float3 normal (IGNORED — normals are
///     recomputed from the vertices; STL normals are often bogus),
///     3×float3 vertices, uint16 attribute byte count.  50 bytes per
///     triangle, so a valid binary file is exactly 84 + 50·count bytes.
///
/// Detection: the size/count invariant decides binary first; if it does
/// not hold and the file starts with `solid`, ASCII parsing is tried;
/// only if that also fails is the file reported as invalid.

#include <sycl-sandbox/rt/hittables/triangle.h>
#include <sycl-sandbox/rt/math.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace stl {

/// Result of loading an STL file.
struct StlLoadResult {
    /// Parsed triangles (empty on failure).
    std::vector<rt::hittables::Triangle> triangles;
    bool ok = false;
    std::string error;
};

/// Load an STL file (ASCII or binary, auto-detected) from disk.
StlLoadResult load_stl(const std::string &file_path);

/// Per-object placement transform applied to every triangle vertex
/// after loading: scale → rotate → translate.
struct MeshTransform {
    rt::float3 position{0, 0, 0};         ///< translation
    rt::float3 rotation_degrees{0, 0, 0}; ///< Euler angles, applied Z → Y → X
    rt::float3 scale{1, 1, 1};            ///< per-axis scaling
};

/// Transform every triangle of the mesh in place.
void apply_transform(std::vector<rt::hittables::Triangle> &triangles,
                     const MeshTransform &transform);

// ── Implementation ────────────────────────────────────────────────────

namespace detail {

inline float read_le_float(const uint8_t *bytes) {
    uint32_t bits = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
                    ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

/// Rotate a point by Euler angles (degrees) applied in the order
/// Z → Y → X about the fixed world axes (three.js default).
inline rt::float3 rotate_degrees(rt::float3 p, rt::float3 degrees) {
    constexpr float deg_to_rad = 3.14159265358979f / 180.f;

    float rz = degrees.z * deg_to_rad;
    float c = std::cos(rz), s = std::sin(rz);
    float x = c * p.x - s * p.y;
    float y = s * p.x + c * p.y;
    p.x = x;
    p.y = y;

    float ry = degrees.y * deg_to_rad;
    c = std::cos(ry);
    s = std::sin(ry);
    x = c * p.x + s * p.z;
    float z = -s * p.x + c * p.z;
    p.x = x;
    p.z = z;

    float rx = degrees.x * deg_to_rad;
    c = std::cos(rx);
    s = std::sin(rx);
    y = c * p.y - s * p.z;
    z = s * p.y + c * p.z;
    p.y = y;
    p.z = z;
    return p;
}

/// Parse the ASCII STL token stream into triangles.
/// Robust to CRLF, extra whitespace, missing normals, and any keyword
/// casing — only the three `vertex` lines per facet are meaningful.
inline StlLoadResult parse_ascii_stl(const std::string &text) {
    StlLoadResult result;
    std::istringstream stream(text);
    std::string word;
    int vertex_index = -1;   // 0..2 of the current facet, -1 = no facet
    rt::float3 v0, v1, v2;

    auto flush_facet = [&]() {
        if ( vertex_index == 3 ) {
            result.triangles.emplace_back(v0, v1, v2);
        }
        vertex_index = -1;
    };

    while ( stream >> word ) {
        if ( word == "facet" ) {
            flush_facet();
            vertex_index = 0;   // next `vertex` starts the facet
        } else if ( word == "vertex" ) {
            if ( vertex_index < 0 ) {
                vertex_index = 0;   // tolerate a vertex before any facet
            }
            float x, y, z;
            stream >> x >> y >> z;
            if ( !stream ) {
                // Malformed vertex (missing coordinates) — stop.
                break;
            }
            if ( vertex_index == 0 ) {
                v0 = {x, y, z};
            } else if ( vertex_index == 1 ) {
                v1 = {x, y, z};
            } else if ( vertex_index == 2 ) {
                v2 = {x, y, z};
            }
            vertex_index++;
        }
    }
    flush_facet();

    if ( !result.triangles.empty() ) {
        result.ok = true;
    } else {
        result.error = "ASCII STL contains no complete facets";
    }
    return result;
}

} // namespace detail

inline StlLoadResult load_stl(const std::string &file_path) {
    StlLoadResult result;

    // ── Read the whole file ──────────────────────────────────────────
    FILE *file = std::fopen(file_path.c_str(), "rb");
    if ( !file ) {
        result.error = "cannot open file";
        return result;
    }
    std::fseek(file, 0, SEEK_END);
    long file_size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if ( file_size < 0 ) {
        std::fclose(file);
        result.error = "cannot determine file size";
        return result;
    }
    std::vector<uint8_t> bytes((size_t)file_size);
    size_t read = std::fread(bytes.data(), 1, (size_t)file_size, file);
    std::fclose(file);
    if ( read != (size_t)file_size ) {
        result.error = "short read";
        return result;
    }

    bool starts_with_solid =
        bytes.size() >= 5 && std::memcmp(bytes.data(), "solid", 5) == 0;

    // ── Try binary (size/count invariant) ────────────────────────────
    if ( bytes.size() >= 84 ) {
        uint32_t count = (uint32_t)bytes[80] | ((uint32_t)bytes[81] << 8) |
                         ((uint32_t)bytes[82] << 16) | ((uint32_t)bytes[83] << 24);
        bool invariant_holds = 84ull + 50ull * count == bytes.size();
        if ( invariant_holds ) {
            result.triangles.reserve(count);
            for ( uint32_t i = 0; i < count; i++ ) {
                const uint8_t *tri = bytes.data() + 84 + (size_t)i * 50;
                // Skip the 3-float normal (recomputed by Triangle).
                rt::float3 v0{detail::read_le_float(tri + 12),
                              detail::read_le_float(tri + 16),
                              detail::read_le_float(tri + 20)};
                rt::float3 v1{detail::read_le_float(tri + 24),
                              detail::read_le_float(tri + 28),
                              detail::read_le_float(tri + 32)};
                rt::float3 v2{detail::read_le_float(tri + 36),
                              detail::read_le_float(tri + 40),
                              detail::read_le_float(tri + 44)};
                result.triangles.emplace_back(v0, v1, v2);
            }
            result.ok = true;
            return result;
        }
    }

    // ── Try ASCII ────────────────────────────────────────────────────
    if ( starts_with_solid ) {
        std::string text(reinterpret_cast<const char *>(bytes.data()),
                         bytes.size());
        result = detail::parse_ascii_stl(text);
        if ( result.ok ) {
            return result;
        }
    }

    result.error = "unsupported or malformed STL file (neither a valid "
                   "binary file — size must be 84 + 50·count — nor a "
                   "parseable ASCII file)";
    return result;
}

inline void apply_transform(std::vector<rt::hittables::Triangle> &triangles,
                            const MeshTransform &transform) {
    if ( triangles.empty() ) {
        return;
    }
    for ( auto &t : triangles ) {
        rt::float3 a = t.a;
        rt::float3 b = t.b;
        rt::float3 c = t.c;
        // scale
        a = {a.x * transform.scale.x, a.y * transform.scale.y, a.z * transform.scale.z};
        b = {b.x * transform.scale.x, b.y * transform.scale.y, b.z * transform.scale.z};
        c = {c.x * transform.scale.x, c.y * transform.scale.y, c.z * transform.scale.z};
        // rotate
        a = detail::rotate_degrees(a, transform.rotation_degrees);
        b = detail::rotate_degrees(b, transform.rotation_degrees);
        c = detail::rotate_degrees(c, transform.rotation_degrees);
        // translate
        a = rt::add(a, transform.position);
        b = rt::add(b, transform.position);
        c = rt::add(c, transform.position);
        t = rt::hittables::Triangle(a, b, c);
    }
}

} // namespace stl
