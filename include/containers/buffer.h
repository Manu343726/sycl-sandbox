#pragma once
#include <cstddef>
#include <alloc/tag.h>

namespace containers {

/// Typed view of a contiguous chunk of memory (host or device).
/// Unlike alloc::raw::Buffer<Tag> (void*, bytes), this carries the element type.
template <alloc::Target Tag, typename T>
struct Buffer {
    T *data = nullptr;
    size_t count = 0; ///< number of elements, not bytes

    bool is_valid() const {
        return data != nullptr;
    }

    size_t size_bytes() const {
        return count * sizeof(T);
    }
};

} // namespace containers
