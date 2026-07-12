#pragma once
#include <cstdint>

enum class ParamType : uint8_t {
    FLOAT,
    INT,
    BOOL,
    COLOR_RGB,
    COLOR_RGBA,
    VEC3,
    ENUM,
};
// ── Type compatibility ────────────────────────────────────────────────
// Maps C++ types to their canonical ParamType for compile-time dispatch.
// Used by ParamLookup::read<T>() and StatWriter::write<T>() to validate
// that the caller's C++ type matches the declared ParamType at runtime.

template <typename T> struct ParamTypeOf;

template <> struct ParamTypeOf<float>  { static constexpr ParamType value = ParamType::FLOAT; };
template <> struct ParamTypeOf<int>    { static constexpr ParamType value = ParamType::INT; };
template <> struct ParamTypeOf<bool>   { static constexpr ParamType value = ParamType::BOOL; };

/// Check whether a declared ParamType is compatible with the requested
/// read/write type.  The request parameter is the ParamType that the
/// C++ type T would map to (e.g. float → FLOAT, int → INT).
///
/// Compatibility rules:
///   - Identical types are always compatible.
///   - Scalar types (FLOAT, INT, BOOL, ENUM) share 4-byte storage and
///     are mutually compatible — kernels frequently read an INT param as
///     float for arithmetic, or vice versa.
///   - VEC3 and COLOR_RGB share 12-byte (3×float) storage.
///   - COLOR_RGBA is 16 bytes and only compatible with itself.
inline constexpr bool is_type_compatible(ParamType declared, ParamType requested) {
    if (declared == requested) return true;

    // 4-byte scalar group: FLOAT, INT, BOOL, ENUM
    bool dec_scalar = (declared == ParamType::FLOAT  || declared == ParamType::INT ||
                       declared == ParamType::BOOL   || declared == ParamType::ENUM);
    bool req_scalar = (requested == ParamType::FLOAT  || requested == ParamType::INT ||
                       requested == ParamType::BOOL   || requested == ParamType::ENUM);
    if (dec_scalar && req_scalar) return true;

    // 12-byte vector group: VEC3, COLOR_RGB
    if ((declared == ParamType::VEC3 || declared == ParamType::COLOR_RGB) &&
        (requested == ParamType::VEC3 || requested == ParamType::COLOR_RGB))
        return true;

    return false;
}