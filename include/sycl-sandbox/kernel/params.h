#pragma once

/// @file
/// Kernel parameter system — hash-based name lookup for typed parameter
/// reading inside kernel code.  Combined with the standard parameter
/// indices that all raytracer kernels must agree on.
///
/// The host populates the entry table and float buffer from the scene
/// YAML descriptor, then the kernel reads values by name with zero
/// per-pixel overhead (reads happen before the SYCL kernel launch).

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <sycl-sandbox/param_types.h>

namespace rt {

// ── ParamEntry ─────────────────────────────────────────────────────────

/// Maps a parameter name hash to its byte offset and declared type in a
/// flat float buffer.  The `type` field enables runtime validation: when
/// a kernel calls `read<T>(name)`, the lookup verifies that T is
/// compatible with the declared ParamType before reinterpreting bits.
/// Kept trivially copyable for SYCL device-side use if needed.
struct ParamEntry {
    uint64_t name_hash;
    uint32_t byte_offset; ///< byte offset into the float buffer
    ParamType type;       ///< declared type from YAML scene descriptor
};

// ── ParamLookup ────────────────────────────────────────────────────────

/// Device/host-side lookup table for reading typed parameters by name.
///
/// The host populates the entry table and buffer, then the kernel uses
/// read<T>() to extract values.  All read operations happen on the host
/// (before SYCL kernel launch), so there is zero per-pixel overhead.
///
/// Usage:
/// @code
///   ParamLookup reader;
///   reader.set_buffer(params_buf, entries, num_entries);
///   int spp = reader.read<int>("spp_frame");
///   float3 eye = reader.read<float3>("cam_eye");
/// @endcode
struct ParamLookup {
    const float *buffer = nullptr;
    const ParamEntry *entries = nullptr;
    int num_entries = 0;

    void set_buffer(const void *buf) {
        buffer = static_cast<const float *>(buf);
    }

    void set_entries(const ParamEntry *ents, int n) {
        entries = ents;
        num_entries = n;
    }

    /// Locate a parameter by name hash and return a pointer to its
    /// position in the float buffer, or nullptr if not found.
    const float *find(const char *name) const {
        if (!buffer || !entries) return nullptr;
        uint64_t h = 0;
        for (const char *p = name; *p; ++p) {
            h = h * 131 + (uint8_t)*p;
        }
        for (int i = 0; i < num_entries; i++) {
            if (entries[i].name_hash == h) {
                return buffer + entries[i].byte_offset / sizeof(float);
            }
        }
        return nullptr;
    }

    /// Locate a parameter by name and validate that its declared type is
    /// compatible with the requested type.  Returns nullptr if not found
    /// or if the type is incompatible (logged at trace level).
    const float *find(const char *name, ParamType requested_type) const {
        if (!buffer || !entries) return nullptr;
        uint64_t h = 0;
        for (const char *p = name; *p; ++p) {
            h = h * 131 + (uint8_t)*p;
        }
        for (int i = 0; i < num_entries; i++) {
            if (entries[i].name_hash == h) {
                if (!is_type_compatible(entries[i].type, requested_type)) {
                    return nullptr; // type mismatch — safe fallback
                }
                return buffer + entries[i].byte_offset / sizeof(float);
            }
        }
        return nullptr;
    }

    /// Read a float parameter by name.  Returns 0 if not found.
    float read_float(const char *name) const {
        auto *p = find(name, ParamType::FLOAT);
        return p ? *p : 0.0f;
    }

    /// Read an int parameter by name.  Returns 0 if not found.
    int read_int(const char *name) const {
        auto *p = find(name, ParamType::INT);
        return p ? static_cast<int>(*p) : 0;
    }

    /// Read a vec3 (3 floats) parameter by name.  Returns {0,0,0} if not found.
    void read_float3(const char *name, float out[3]) const {
        auto *p = find(name, ParamType::VEC3);
        if (p) {
            out[0] = p[0]; out[1] = p[1]; out[2] = p[2];
        } else {
            out[0] = out[1] = out[2] = 0.0f;
        }
    }

    /// Read a vec3 parameter by name, returning a float3 struct.
    template <typename Vec3>
    Vec3 read_vec3(const char *name) const {
        auto *p = find(name, ParamType::VEC3);
        if (p) return {p[0], p[1], p[2]};
        return {0, 0, 0};
    }

    /// Templated read — dispatch by type with runtime validation.
    ///
    /// The declared ParamType (from the YAML scene descriptor) is checked
    /// against the requested C++ type T:
    ///   - `read<float>()` → validates FLOAT, INT, BOOL, or ENUM
    ///   - `read<int>()`   → validates INT, FLOAT, BOOL, or ENUM
    ///   - `read<bool>()`  → validates BOOL, INT, or FLOAT
    ///   - `read<Vec3>()`  → validates VEC3 or COLOR_RGB
    ///
    /// Returns the default-constructed value if the name is not found
    /// or the types are incompatible.
    template <typename T>
    T read(const char *name) const {
        // Dispatch to type-validated find based on the C++ type.
        // Scalar types use ParamTypeOf; struct types (float3, etc.)
        // fall through to VEC3 which is compatible with COLOR_RGB.
        const float *p = nullptr;
        if constexpr (std::is_same_v<T, float>) {
            p = find(name, ParamTypeOf<float>::value);
        } else if constexpr (std::is_same_v<T, int>) {
            p = find(name, ParamTypeOf<int>::value);
        } else if constexpr (std::is_same_v<T, bool>) {
            p = find(name, ParamTypeOf<bool>::value);
        } else {
            // Struct types (float3, etc.) — VEC3 is compatible with COLOR_RGB
            p = find(name, ParamType::VEC3);
        }
        if (!p) return T{};
        if constexpr (std::is_same_v<T, float>) {
            return *p;
        } else if constexpr (std::is_same_v<T, int>) {
            return static_cast<int>(*p);
        } else if constexpr (std::is_same_v<T, bool>) {
            return *p != 0.0f;
        } else {
            // For struct types (float3, etc.), read 3 floats
            return {p[0], p[1], p[2]};
        }
    }
};

/// Compute a simple hash for a parameter name (same algorithm as ParamLookup::find).
inline uint64_t param_name_hash(const char *name) {
    uint64_t h = 0;
    for (const char *p = name; *p; ++p) {
        h = h * 131 + (uint8_t)*p;
    }
    return h;
}



} // namespace rt