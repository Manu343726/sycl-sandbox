#pragma once
#include <cstddef>
#include "tag.h"

/// Non-owning view of a chunk of allocated memory (host or device).
/// Size is in bytes.  Allocators return Buffer<Tag> instead of raw pointers.
template <AllocatorTag Tag>
struct Buffer {
    void   *data = nullptr;
    size_t  size = 0;
    bool is_valid() const { return data != nullptr; }
};
