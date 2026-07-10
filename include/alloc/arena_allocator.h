#pragma once
#include <cstddef>
#include <vector>
#include <sycl/sycl.hpp>
#include "buffer.h"
#include "tag.h"

namespace alloc::raw {

/// Arena allocator — allocates from a chain of fixed-size blocks.
template <AllocatorTag Tag_>
class ArenaAllocator {
public:
    static constexpr AllocatorTag tag = Tag_;

    ArenaAllocator(sycl::queue &queue, size_t block_size)
        : block_size_(block_size), queue_(queue) {
    }

    ~ArenaAllocator() { free_all(); }

    ArenaAllocator(ArenaAllocator &&) = default;
    ArenaAllocator &operator=(ArenaAllocator &&) = default;

    Buffer<Tag_> allocate(size_t bytes, size_t alignment = 1);

    void reset() {
        current_       = nullptr;
        current_offset_ = 0;
    }

    void free_all();

private:
    Buffer<Tag_> fresh_block();

    size_t                    block_size_;
    sycl::queue              &queue_;
    std::vector<Buffer<Tag_>> blocks_;
    Buffer<Tag_>             *current_       = nullptr;
    size_t                    current_offset_ = 0;
};

} // namespace alloc::raw

// ── template definitions (must be visible to callers) ───────────────────

namespace alloc::raw {

template <AllocatorTag Tag_>
Buffer<Tag_> ArenaAllocator<Tag_>::allocate(size_t bytes, size_t alignment) {
    // Check current block
    if (current_) {
        uintptr_t base = reinterpret_cast<uintptr_t>(current_->data);
        uintptr_t aligned = (base + current_offset_ + alignment - 1) & ~(alignment - 1);
        size_t needed = (aligned - base - current_offset_) + bytes;
        if (current_offset_ + needed <= block_size_) {
            current_offset_ += needed;
            return {reinterpret_cast<void *>(aligned), bytes};
        }
    }
    // Need a new block
    auto block = fresh_block();
    if (!block.is_valid()) return {};
    blocks_.push_back(block);
    current_ = &blocks_.back();
    uintptr_t base = reinterpret_cast<uintptr_t>(current_->data);
    uintptr_t aligned = (base + alignment - 1) & ~(alignment - 1);
    current_offset_ = (aligned - base) + bytes;
    return {reinterpret_cast<void *>(aligned), bytes};
}

template <AllocatorTag Tag_>
void ArenaAllocator<Tag_>::free_all() {
    RootAllocator<Tag_> root;
    for (auto &b : blocks_) {
        root.deallocate(b, queue_);
    }
    blocks_.clear();
    current_       = nullptr;
    current_offset_ = 0;
}

template <AllocatorTag Tag_>
Buffer<Tag_> ArenaAllocator<Tag_>::fresh_block() {
    return RootAllocator<Tag_>().allocate(block_size_, queue_);
}

} // namespace alloc::raw
