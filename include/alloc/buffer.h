#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <alloc/tag.h>

namespace alloc {

/// Typed view of a contiguous chunk of memory.
/// Pointer is T-aligned; count is number of full elements that fit.
template <alloc::Target Tag, typename T>
struct Buffer {
    static_assert(std::is_trivially_copyable_v<T>,
                  "SYCL device types must be trivially copyable");

    T      *data  = nullptr;
    size_t  count = 0;

    bool is_valid() const { return data != nullptr; }
    size_t size_bytes() const { return count * sizeof(T); }
};

} // namespace alloc

namespace alloc::raw {

/// Non-owning view of a chunk of allocated memory (host or device).
/// Size is in bytes.  Untyped — allocators return this.
template <alloc::Target Tag>
struct Buffer {
    void   *data = nullptr;
    size_t  size = 0;

    bool is_valid() const { return data != nullptr; }

    /// Return a typed view aligned to T.
    /// The returned pointer is T-aligned; buf.data[i] gives the correct address.
    template <typename T>
    auto typed() const -> alloc::Buffer<Tag, T> {
        uintptr_t addr    = reinterpret_cast<uintptr_t>(data);
        uintptr_t aligned = (addr + alignof(T) - 1) & ~(alignof(T) - 1);
        size_t    offset  = aligned - addr;
        if (offset >= size) return {};
        T    *t_data = reinterpret_cast<T *>(aligned);
        size_t count  = (size - offset) / sizeof(T);
        return {t_data, count};
    }
};

} // namespace alloc::raw
