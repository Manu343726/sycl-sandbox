#pragma once
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <alloc/buffer.h>
#include <alloc/tag.h>
#include <alloc/linear_allocator.h>
#include <alloc/root_allocator.h>
#include <alloc/transfer.h>
#include "buffer.h"

namespace containers::raw {

/// Untyped vector<Tag> — fixed-max-length buffer on LinearAllocator.
///
/// RAII: destructor frees the pool.  Move transfers ownership (source empties).
/// Copy allocates a new pool and copies elements.
template <alloc::Target Tag>
class vector {
public:
    vector(size_t max_elements, size_t element_size, size_t alignment, sycl::queue &queue)
        : queue_(&queue), element_size_(element_size), alignment_(alignment), count_(0) {
        size_t total = max_elements * element_size + alignment;
        auto pool = alloc::raw::RootAllocator<Tag>().allocate(total, queue);
        allocator_ = alloc::raw::LinearAllocator<Tag>(pool);
    }

    ~vector() {
        allocator_.reset_and_free(*queue_);
    }

    vector(vector &&other) noexcept
        : queue_(other.queue_),
          allocator_(std::move(other.allocator_)),
          element_size_(other.element_size_),
          alignment_(other.alignment_),
          count_(other.count_) {
        other.queue_ = nullptr;
        other.count_ = 0;
    }

    friend void swap(vector &a, vector &b) noexcept {
        using std::swap;
        swap(a.queue_,        b.queue_);
        swap(a.allocator_,    b.allocator_);
        swap(a.element_size_, b.element_size_);
        swap(a.alignment_,    b.alignment_);
        swap(a.count_,        b.count_);
    }

    vector &operator=(vector other) noexcept {
        swap(*this, other);
        return *this;
    }

    vector(const vector &other)
        : queue_(other.queue_),
          element_size_(other.element_size_),
          alignment_(other.alignment_),
          count_(other.count_) {
        size_t total = other.max_size() * element_size_ + alignment_;
        auto pool = alloc::raw::RootAllocator<Tag>().allocate(total, *queue_);
        allocator_ = alloc::raw::LinearAllocator<Tag>(pool);
        if (count_ > 0) {
            alloc::raw::Buffer<Tag> src = other.data();
            alloc::raw::Buffer<Tag> dst = data();
            std::memcpy(dst.data, src.data, src.size);
        }
    }



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

private:
    sycl::queue                          *queue_      = nullptr;
    alloc::raw::LinearAllocator<Tag>      allocator_;
    size_t                                element_size_ = 0;
    size_t                                alignment_    = 0;
    size_t                                count_        = 0;
};

} // namespace containers::raw

namespace containers {

/// Typed vector<Tag, T> — type-safe wrapper around raw::vector.
///
/// RAII: destructor, move, copy all delegate to the raw vector.
template <alloc::Target Tag, typename T>
class vector {
public:
    vector(size_t max_elements, sycl::queue &queue)
        : impl_(max_elements, sizeof(T), alignof(T), queue) {
        static_assert(std::is_trivially_copyable_v<T>,
                      "SYCL device types must be trivially copyable");
    }

    ~vector() = default;
    vector(vector &&) = default;
    vector &operator=(vector &&) = default;
    vector(const vector &) = default;
    vector &operator=(const vector &) = default;

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

private:
    explicit vector(containers::raw::vector<Tag> &&impl) : impl_(std::move(impl)) {}
    containers::raw::vector<Tag> impl_;
};

} // namespace containers
