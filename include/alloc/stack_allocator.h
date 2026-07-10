#pragma once
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <alloc/buffer.h>
#include <alloc/tag.h>

namespace alloc::raw {

/// Stack allocator — supports push/pop via markers.
template <alloc::Target Tag_>
class StackAllocator {
public:
    static constexpr alloc::Target tag = Tag_;
    using Marker = size_t;

    StackAllocator() = default;

    explicit StackAllocator(Buffer<Tag_> pool)
        : pool_(pool), cursor_(static_cast<char *>(pool.data)) {
    }

    StackAllocator(StackAllocator &&) = default;
    StackAllocator &operator=(StackAllocator &&) = default;

    Buffer<Tag_> allocate(size_t bytes, size_t alignment = 1) {
        uintptr_t aligned = (reinterpret_cast<uintptr_t>(cursor_) + alignment - 1)
                          & ~(alignment - 1);
        size_t padded_mark = aligned - reinterpret_cast<uintptr_t>(pool_.data);
        if (padded_mark + bytes > pool_.size) return {};
        cursor_ = reinterpret_cast<char *>(aligned) + bytes;
        marker_ = padded_mark + bytes;
        return {reinterpret_cast<void *>(aligned), bytes};
    }

    Marker mark() const { return marker_; }

    void pop(Marker m) {
        marker_ = m;
        cursor_ = static_cast<char *>(pool_.data) + m;
    }

    void reset() { pop(0); }

    void reset_and_free(sycl::queue &queue) {
        if (pool_.data) sycl::free(pool_.data, queue);
        pool_.data = nullptr;
        pool_.size = 0;
        cursor_    = nullptr;
        marker_    = 0;
    }

private:
    Buffer<Tag_> pool_;
    char   *cursor_ = nullptr;
    size_t  marker_ = 0;
};

} // namespace alloc::raw
