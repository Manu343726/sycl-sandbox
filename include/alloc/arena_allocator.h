#pragma once
#include <cstddef>
#include <vector>
#include <sycl/sycl.hpp>
#include "buffer.h"
#include "tag.h"

/// Arena allocator — allocates from a chain of fixed-size blocks.
///
/// When the current block is exhausted, a new block is requested from the
/// upstream allocator.  Individual deallocations are no-ops; the entire
/// arena is freed at once.  Unlike LinearAllocator, arena can grow
/// dynamically.
template <AllocatorTag Tag_, typename Upstream>
class ArenaAllocator {
public:
    static constexpr AllocatorTag tag = Tag_;

    ArenaAllocator(Upstream upstream, size_t block_size, sycl::queue &queue)
        : upstream_(std::move(upstream)), block_size_(block_size), queue_(queue) {
    }

    ~ArenaAllocator() { free_all(); }

    ArenaAllocator(ArenaAllocator &&) = default;
    ArenaAllocator &operator=(ArenaAllocator &&) = default;

    Buffer<Tag_> allocate(size_t bytes) {
        if (current_ && current_offset_ + bytes <= block_size_) {
            void *p = static_cast<char *>(current_->data) + current_offset_;
            current_offset_ += bytes;
            return {p, bytes};
        }
        // Need a new block
        auto block = upstream_.allocate(block_size_, queue_);
        if (!block.is_valid()) return {};
        blocks_.push_back(block);
        current_ = &blocks_.back();
        current_offset_ = bytes;
        return {block.data, bytes};
    }

    void reset() {
        current_       = nullptr;
        current_offset_ = 0;
    }

    void free_all() {
        for (auto &b : blocks_) {
            upstream_.deallocate(b, queue_);
        }
        blocks_.clear();
        current_       = nullptr;
        current_offset_ = 0;
    }

private:
    Upstream                    upstream_;
    size_t                      block_size_;
    sycl::queue                &queue_;
    std::vector<Buffer<Tag_>>   blocks_;
    Buffer<Tag_>               *current_       = nullptr;
    size_t                      current_offset_ = 0;
};
