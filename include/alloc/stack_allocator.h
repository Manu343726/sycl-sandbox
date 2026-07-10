#pragma once
#include <cstddef>
#include <sycl/sycl.hpp>
#include "buffer.h"
#include "tag.h"

/// Stack allocator — supports push/pop via markers.
///
/// Like LinearAllocator but with a marker system that allows rolling back
/// to a previous point (pop).  Useful for nested scopes where you allocate
/// and then undo a group of allocations.
template <AllocatorTag Tag_>
class StackAllocator {
public:
    static constexpr AllocatorTag tag = Tag_;

    using Marker = size_t;

    StackAllocator() = default;

    explicit StackAllocator(Buffer<Tag_> pool)
        : pool_(pool), cursor_(static_cast<char *>(pool.data)) {
    }

    StackAllocator(StackAllocator &&) = default;
    StackAllocator &operator=(StackAllocator &&) = default;

    Buffer<Tag_> allocate(size_t bytes) {
        if (marker_ + bytes > pool_.size) {
            return {};
        }
        void *p = cursor_;
        cursor_  += bytes;
        marker_  += bytes;
        return {p, bytes};
    }

    /// Snapshot the current position.
    Marker mark() const { return marker_; }

    /// Roll back to a previous marker (pops allocations).
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
