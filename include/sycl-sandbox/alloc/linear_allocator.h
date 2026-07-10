#pragma once
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <sycl-sandbox/alloc/buffer.h>
#include <sycl-sandbox/alloc/tag.h>

namespace alloc::raw {

/// Linear (bump) allocator.
///
/// Obtains a large block from an upstream allocator once, then sub-allocates
/// linearly.  Individual deallocations are a no-op; the entire pool is
/// reclaimed via reset().  Ideal for per-frame scratch allocation.
template <alloc::Target Tag_>
class LinearAllocator {
public:
    static constexpr alloc::Target tag = Tag_;

    LinearAllocator() = default;

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

    ~LinearAllocator() = default;

    Buffer<Tag_> allocate(size_t bytes, size_t alignment = 1) {
        uintptr_t aligned = (reinterpret_cast<uintptr_t>(cursor_) + alignment - 1)
                          & ~(alignment - 1);
        size_t padded_offset = aligned - reinterpret_cast<uintptr_t>(pool_.data);
        if (padded_offset + bytes > pool_.size) {
            return {};
        }
        cursor_ = reinterpret_cast<char *>(aligned) + bytes;
        offset_ = padded_offset + bytes;
        return {reinterpret_cast<void *>(aligned), bytes};
    }

    void reset() {
        cursor_ = static_cast<char *>(pool_.data);
        offset_ = 0;
    }

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

} // namespace alloc::raw
