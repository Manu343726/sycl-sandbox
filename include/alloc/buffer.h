#pragma once
#include <cstddef>
#include <cstdint>
#include "tag.h"
#include "../containers/buffer.h"

namespace alloc::raw {

/// Non-owning view of a chunk of allocated memory (host or device).
/// Size is in bytes.  Allocators return Buffer<Tag> instead of raw pointers.
template <alloc::Target Tag>
struct Buffer {
    void *data = nullptr;
    size_t size = 0;

    bool is_valid() const {
        return data != nullptr;
    }

    /// Return a typed view aligned to T, skipping misaligned bytes at the start.
    /// The returned pointer is T-aligned; buf.data[i] gives the correct address.
    template <typename T>
    containers::Buffer<Tag, T> typed() const {
        uintptr_t addr = reinterpret_cast<uintptr_t>(data);
        uintptr_t aligned = (addr + alignof(T) - 1) & ~(alignof(T) - 1);
        size_t offset = aligned - addr;
        if ( offset >= size ) {
            return {};
        }
        T *t_data = reinterpret_cast<T *>(aligned);
        size_t count = (size - offset) / sizeof(T);
        return {t_data, count};
    }
};

} // namespace alloc::raw
