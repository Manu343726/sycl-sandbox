#pragma once
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <sycl/sycl.hpp>
#include "buffer.h"
#include "tag.h"
#include "linear_allocator.h"
#include "transfer.h"
#include "host_allocator.h"
#include "device_allocator.h"

// ────────────────────────────────────────────────────────────────────────
//  Untyped vector<Tag> — fixed-max-length buffer built on LinearAllocator.
// ────────────────────────────────────────────────────────────────────────

template <AllocatorTag Tag>
class vector {
public:
    /// Construct a vector that can hold up to `max_elements` of `element_size`
    /// bytes each, aligned to `alignment`.  Memory is obtained by allocating
    /// a pool from an upstream root allocator.
    vector(size_t max_elements, size_t element_size, size_t alignment, sycl::queue &queue)
        : element_size_(element_size), count_(0) {
        size_t total = max_elements * element_size + alignment;
        Buffer<Tag> pool;
        if constexpr ( Tag == AllocatorTag::Host ) {
            pool = HostAllocator().allocate(total, queue);
        } else {
            pool = DeviceAllocator().allocate(total, queue);
        }
        allocator_ = LinearAllocator<Tag>(pool);
    }

    ~vector() {
        // Pool is freed via the allocator's destructor if we call reset_and_free,
        // but since we don't have the queue here, we rely on explicit teardown.
        // For safety, leak detection: user must call transfer() or discard().
    }

    vector(vector &&) = default;
    vector &operator=(vector &&) = default;

    vector(const vector &) = delete;
    vector &operator=(const vector &) = delete;

    void push_back(const void *element) {
        auto buf = allocator_.allocate(element_size_);
        if (!buf.is_valid()) return;   // OOM
        std::memcpy(buf.data, element, element_size_);
        count_++;
    }

    /// View of the current contiguous data (count × element_size bytes).
    Buffer<Tag> data() const {
        // LinearAllocator lays out sequentially, so data starts at pool base.
        Buffer<Tag> view = allocator_.pool();
        view.size = count_ * element_size_;
        return view;
    }

    size_t size()  const { return count_; }
    size_t max_size() const { return allocator_.pool().size / element_size_; }

    /// Transfer data to a different tag.  Returns a new vector on the target
    /// side.  Clears (frees) the source vector's memory.
    template <AllocatorTag TargetTag>
    vector<TargetTag> transfer(sycl::queue &queue) {
        size_t n = count_;
        size_t elem_sz = element_size_;
        vector<TargetTag> result(n, elem_sz, elem_sz, queue);
        ::transfer(data(), result.allocator_.pool(), queue);
        result.count_ = n;
        // Clear source
        allocator_.reset_and_free(queue);
        count_ = 0;
        return result;
    }

    /// Discard all data and free the pool.
    void discard(sycl::queue &queue) {
        allocator_.reset_and_free(queue);
        count_ = 0;
    }

private:
    LinearAllocator<Tag> allocator_;
    size_t element_size_;
    size_t count_;
};

// ────────────────────────────────────────────────────────────────────────
//  Typed vector<Tag, T> — type-safe wrapper around untyped vector.
// ────────────────────────────────────────────────────────────────────────

template <AllocatorTag Tag, typename T>
class typed_vector {
public:
    typed_vector(size_t max_elements, sycl::queue &queue)
        : impl_(max_elements, sizeof(T), alignof(T), queue) {
    }

    void push_back(const T &element) {
        impl_.push_back(&element);
    }

    Buffer<Tag> data() const { return impl_.data(); }
    size_t size()       const { return impl_.size(); }
    size_t max_size()   const { return impl_.max_size(); }

    template <AllocatorTag TargetTag>
    typed_vector<TargetTag, T> transfer(sycl::queue &queue) {
        auto untyped = impl_.template transfer<TargetTag>(queue);
        typed_vector<TargetTag, T> result(std::move(untyped));
        return result;
    }

    void discard(sycl::queue &queue) { impl_.discard(queue); }

private:
    // Construct from an existing untyped vector (used by transfer).
    explicit typed_vector(vector<Tag> &&impl) : impl_(std::move(impl)) {}

    vector<Tag> impl_;
};
