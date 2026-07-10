#pragma once
#include <cstring>
#include <sycl/sycl.hpp>
#include <sycl-sandbox/alloc/buffer.h>

namespace alloc::raw {

/// Tag-aware memcpy: host-pointer → buffer.
template <alloc::Target Tag>
void memcpy(Buffer<Tag> dst, const void *src, size_t bytes, sycl::queue &queue) {
    if constexpr ( Tag == alloc::Target::Host ) {
        std::memcpy(dst.data, src, bytes);
    } else {
        queue.memcpy(dst.data, src, bytes).wait();
    }
}

/// Tag-aware memcpy: same-tag buffer → buffer.
template <alloc::Target Tag>
void memcpy(Buffer<Tag> dst, Buffer<Tag> src, sycl::queue &queue) {
    size_t bytes = src.size < dst.size ? src.size : dst.size;
    if constexpr ( Tag == alloc::Target::Host ) {
        std::memcpy(dst.data, src.data, bytes);
    } else {
        queue.memcpy(dst.data, src.data, bytes).wait();
    }
}

/// Copy bytes between possibly-different-tag buffers.
template <alloc::Target From, alloc::Target To>
void transfer(Buffer<From> src, Buffer<To> dst, sycl::queue &queue) {
    size_t bytes = src.size < dst.size ? src.size : dst.size;
    if constexpr ( From == alloc::Target::Host && To == alloc::Target::Host ) {
        std::memcpy(dst.data, src.data, bytes);
    } else {
        queue.memcpy(dst.data, src.data, bytes).wait();
    }
}

} // namespace alloc::raw

namespace alloc {

/// Copy elements between typed buffers (delegates to raw::transfer).
template <alloc::Target From, alloc::Target To, typename T>
void transfer(Buffer<From, T> src, Buffer<To, T> dst, sycl::queue &queue) {
    size_t count = src.count < dst.count ? src.count : dst.count;
    alloc::raw::Buffer<From> raw_src{src.data, count * sizeof(T)};
    alloc::raw::Buffer<To>   raw_dst{dst.data, count * sizeof(T)};
    alloc::raw::transfer(raw_src, raw_dst, queue);
}

} // namespace alloc
