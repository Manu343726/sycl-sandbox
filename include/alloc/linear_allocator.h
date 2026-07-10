#pragma once
#include <cstddef>
#include <cassert>
#include <sycl/sycl.hpp>
#include "buffer.h"
#include "tag.h"

/// Linear (bump) allocator.
///
/// Obtains a large block from an upstream allocator once, then sub-allocates
/// linearly.  Individual deallocations are a no-op; the entire pool is
/// reclaimed via reset().  Ideal for per-frame scratch allocation.
template <AllocatorTag Tag_>
class LinearAllocator {
public:
    static constexpr AllocatorTag tag = Tag_;

    LinearAllocator() = default;

    /// Steal an existing buffer as the pool.  The allocator takes ownership.
    explicit LinearAllocator(Buffer<Tag_> pool)
        : pool_(pool), cursor_(static_cast<char *>(pool.data)) {
    }

    LinearAllocator(LinearAllocator &&other) noexcept { *this = std::move(other); }
    LinearAllocator &operator=(LinearAllocator &&other) noexcept {
        pool_   = other.pool_;
        cursor_ = other.cursor_;
        other.pool_.data = nullptr;
        other.pool_.size = 0;
        other.cursor_    = nullptr;
        return *this;
    }

    LinearAllocator(const LinearAllocator &) = delete;
    LinearAllocator &operator=(const LinearAllocator &) = delete;

    ~LinearAllocator() {
        // Ownership: the creator of the allocator is responsible for
        // deallocating the pool (typically via reset_and_free).
    }

    Buffer<Tag_> allocate(size_t bytes) {
        if (offset_ + bytes > pool_.size) {
            return {};   // OOM
        }
        void *p = cursor_;
        cursor_       += bytes;
        offset_       += bytes;
        return {p, bytes};
    }

    /// Reset cursor without freeing the pool.
    void reset() {
        cursor_ = static_cast<char *>(pool_.data);
        offset_ = 0;
    }

    /// Free the pool and reset.
    void reset_and_free(sycl::queue &queue) {
        if (pool_.data) sycl::free(pool_.data, queue);
        pool_.data = nullptr;
        pool_.size = 0;
        cursor_    = nullptr;
        offset_    = 0;
    }

    Buffer<Tag_> pool() const { return pool_; }

private:
    Buffer<Tag_> pool_;
    char   *cursor_ = nullptr;
    size_t  offset_ = 0;
};
