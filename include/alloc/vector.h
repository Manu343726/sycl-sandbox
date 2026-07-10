#pragma once
#include <cstddef>
#include <cstring>
#include <sycl/sycl.hpp>
#include "buffer.h"
#include "tag.h"
#include "linear_allocator.h"
#include "root_allocator.h"
#include "transfer.h"

// ────────────────────────────────────────────────────────────────────────
//  Untyped vector<Tag> — fixed-max-length buffer on LinearAllocator.
// ────────────────────────────────────────────────────────────────────────

template <AllocatorTag Tag>
class vector {
public:
    vector(size_t max_elements, size_t element_size, size_t alignment, sycl::queue &queue)
        : element_size_(element_size), count_(0) {
        size_t total = max_elements * element_size + alignment;
        auto pool = RootAllocator<Tag>().allocate(total, queue);
        allocator_ = LinearAllocator<Tag>(pool);
    }

    ~vector() = default;
    vector(vector &&) = default;
    vector &operator=(vector &&) = default;
    vector(const vector &) = delete;
    vector &operator=(const vector &) = delete;

    void push_back(const void *element) {
        auto buf = allocator_.allocate(element_size_);
        if (!buf.is_valid()) return;
        std::memcpy(buf.data, element, element_size_);
        count_++;
    }

    Buffer<Tag> data() const {
        Buffer<Tag> view = allocator_.pool();
        view.size = count_ * element_size_;
        return view;
    }

    size_t size()     const { return count_; }
    size_t max_size() const { return allocator_.pool().size / element_size_; }

    template <AllocatorTag TargetTag>
    vector<TargetTag> transfer(sycl::queue &queue) {
        size_t n = count_;
        size_t elem_sz = element_size_;
        vector<TargetTag> result(n, elem_sz, elem_sz, queue);
        ::transfer(data(), result.allocator_.pool(), queue);
        result.count_ = n;
        allocator_.reset_and_free(queue);
        count_ = 0;
        return result;
    }

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
    explicit typed_vector(vector<Tag> &&impl) : impl_(std::move(impl)) {}
    vector<Tag> impl_;
};
