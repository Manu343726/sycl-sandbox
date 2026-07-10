#pragma once
#include <cstddef>
#include <cstring>
#include <sycl/sycl.hpp>
#include <alloc/buffer.h>
#include <alloc/tag.h>
#include <alloc/linear_allocator.h>
#include <alloc/root_allocator.h>
#include <alloc/transfer.h>
#include "buffer.h"

namespace containers::raw {

/// Untyped vector<Tag> — fixed-max-length buffer on LinearAllocator.
template <alloc::Target Tag>
class vector {
public:
    vector(size_t max_elements, size_t element_size, size_t alignment, sycl::queue &queue)
        : queue_(&queue), element_size_(element_size), alignment_(alignment), count_(0) {
        size_t total = max_elements * element_size + alignment;
        auto pool = alloc::raw::RootAllocator<Tag>().allocate(total, queue);
        allocator_ = alloc::raw::LinearAllocator<Tag>(pool);
    }

    ~vector() = default;
    vector(vector &&) = default;
    vector &operator=(vector &&) = default;
    vector(const vector &) = delete;
    vector &operator=(const vector &) = delete;

    void push_back(const void *element) {
        auto buf = allocator_.allocate(element_size_, alignment_);
        if (!buf.is_valid()) return;
        std::memcpy(buf.data, element, element_size_);
        count_++;
    }

    alloc::raw::Buffer<Tag> data() const {
        alloc::raw::Buffer<Tag> view = allocator_.pool();
        view.size = count_ * element_size_;
        return view;
    }

    size_t size()     const { return count_; }
    size_t max_size() const { return allocator_.pool().size / element_size_; }

    template <alloc::Target TargetTag>
    vector<TargetTag> transfer() {
        size_t n = count_;
        vector<TargetTag> result(n, element_size_, alignment_, *queue_);
        alloc::raw::transfer(data(), result.allocator_.pool(), *queue_);
        result.count_ = n;
        allocator_.reset_and_free(*queue_);
        count_ = 0;
        return result;
    }

    void discard() {
        allocator_.reset_and_free(*queue_);
        count_ = 0;
    }

private:
    sycl::queue                          *queue_;
    alloc::raw::LinearAllocator<Tag>      allocator_;
    size_t                                element_size_;
    size_t                                alignment_;
    size_t                                count_;
};

} // namespace containers::raw

namespace containers {

/// Typed vector<Tag, T> — type-safe wrapper around raw::vector.
template <alloc::Target Tag, typename T>
class vector {
public:
    vector(size_t max_elements, sycl::queue &queue)
        : impl_(max_elements, sizeof(T), alignof(T), queue) {
    }

    void push_back(const T &element) {
        impl_.push_back(&element);
    }

    containers::Buffer<Tag, T> data() const {
        auto raw = impl_.data();
        return {static_cast<T *>(raw.data), impl_.size()};
    }
    size_t size()       const { return impl_.size(); }
    size_t max_size()   const { return impl_.max_size(); }

    template <alloc::Target TargetTag>
    vector<TargetTag, T> transfer() {
        auto untyped = impl_.template transfer<TargetTag>();
        vector<TargetTag, T> result(std::move(untyped));
        return result;
    }

    void discard() { impl_.discard(); }

private:
    explicit vector(containers::raw::vector<Tag> &&impl) : impl_(std::move(impl)) {}
    containers::raw::vector<Tag>   impl_;
};

} // namespace containers
