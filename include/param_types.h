#pragma once
#include <cstdint>
#include <cstddef>

enum class ParamType : uint8_t {
    FLOAT,
    INT,
    BOOL,
    COLOR_RGB,
    COLOR_RGBA,
    VEC3,
    ENUM,
};

struct ParamMeta {
    const char*      name;
    const char*      description;
    ParamType        type;

    union {
        struct { float   min_f, max_f, step_f; } f;
        struct { int32_t min_i, max_i, step_i; } i;
    } range;

    union {
        float                     default_f;
        int32_t                   default_i;
        bool                      default_b;
        struct { float r,g,b; }   default_c3;
        struct { float r,g,b,a; } default_c4;
    };

    int32_t          enum_count;
    const char**     enum_labels;

    uint32_t         buffer_offset;
    uint32_t         buffer_size;
};

static constexpr uint32_t param_buffer_size(const ParamMeta& p) {
    switch (p.type) {
    case ParamType::FLOAT:    return 4;
    case ParamType::INT:      return 4;
    case ParamType::BOOL:     return 4;
    case ParamType::COLOR_RGB: return 12;
    case ParamType::COLOR_RGBA: return 16;
    case ParamType::VEC3:     return 12;
    case ParamType::ENUM:     return 4;
    }
    return 0;
}
