#pragma once

/// @file
/// Kernel statistics system — hash-based name lookup for typed statistic
/// writing from inside kernel code.  The host reads the float buffer
/// after each frame to populate the UI stat panel.

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <sycl-sandbox/param_types.h>

namespace rt {

// ── StatEntry ─────────────────────────────────────────────────────────

/// Maps a statistic name hash to its byte offset and declared type in a
/// flat float buffer.  The `type` field enables runtime validation: when
/// a kernel calls `write<T>(name, value)`, the lookup verifies that T is
/// compatible with the declared ParamType before writing bits.
/// Kept trivially copyable for SYCL device-side use if needed.
struct StatEntry {
    uint64_t name_hash;
    uint32_t byte_offset; ///< byte offset into the float buffer
    ParamType type;       ///< declared type from YAML scene descriptor
};

// ── StatWriter ────────────────────────────────────────────────────────

/// Device/host-side write-only table for publishing statistics by name.
///
/// The host populates the entry table and buffer, then the kernel writes
/// values via write<T>().  All write operations happen on the host side
/// (after the SYCL kernel launch), so there is zero per-pixel overhead.
///
/// Usage:
/// @code
///   StatWriter writer;
///   writer.set_buffer(stats_buf);
///   writer.set_entries(entries, num_entries);
///   writer.write<int>("num_objects", 42);
///   writer.write<float>("avg_depth", 3.14f);
/// @endcode
struct StatWriter {
    float *buffer = nullptr;
    const StatEntry *entries = nullptr;
    int num_entries = 0;

    void set_buffer(void *buf) {
        buffer = static_cast<float *>(buf);
    }

    void set_entries(const StatEntry *ents, int n) {
        entries = ents;
        num_entries = n;
    }

    /// Locate a statistic by name hash and return a pointer to its
    /// position in the float buffer, or nullptr if not found.
    float *find(const char *name) const {
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

    /// Locate a statistic by name and validate that its declared type is
    /// compatible with the requested type.  Returns nullptr if not found
    /// or if the type is incompatible.
    float *find(const char *name, ParamType requested_type) const {
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

    /// Write a float value.  No-op if the stat doesn't exist.
    void write_float(const char *name, float value) const {
        auto *p = find(name, ParamType::FLOAT);
        if (p) *p = value;
    }

    /// Write an int value.  No-op if the stat doesn't exist.
    void write_int(const char *name, int value) const {
        auto *p = find(name, ParamType::INT);
        if (p) *p = (float)value;
    }

    /// Write a vec3 value.  No-op if the stat doesn't exist.
    void write_float3(const char *name, const float value[3]) const {
        auto *p = find(name, ParamType::VEC3);
        if (p) { p[0] = value[0]; p[1] = value[1]; p[2] = value[2]; }
    }

    /// Templated write — dispatch by type with runtime validation.
    ///
    /// The declared ParamType (from the YAML scene descriptor) is checked
    /// against the requested C++ type T.  No-op if the stat doesn't
    /// exist or the types are incompatible.
    template <typename T>
    void write(const char *name, const T &value) const {
        // Dispatch to type-validated find based on the C++ type.
        float *p = nullptr;
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
        if (!p) return;
        if constexpr (std::is_same_v<T, float>) {
            *p = value;
        } else if constexpr (std::is_same_v<T, int>) {
            *p = (float)value;
        } else if constexpr (std::is_same_v<T, bool>) {
            *p = value ? 1.0f : 0.0f;
        } else {
            // For struct types (float3, etc.), write 3 floats
            p[0] = value[0]; p[1] = value[1]; p[2] = value[2];
        }
    }
};

/// Compute a simple hash for a stat name (same algorithm as StatWriter::find).
inline uint64_t stat_name_hash(const char *name) {
    uint64_t h = 0;
    for (const char *p = name; *p; ++p) {
        h = h * 131 + (uint8_t)*p;
    }
    return h;
}

} // namespace rt