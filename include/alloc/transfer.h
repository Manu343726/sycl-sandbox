#pragma once
#include <cstring>
#include <sycl/sycl.hpp>
#include "buffer.h"

namespace alloc {

/// Copy data between buffers of possibly-different tags.
template <alloc::Target From, alloc::Target To>
void transfer(alloc::raw::Buffer<From> src, alloc::raw::Buffer<To> dst, sycl::queue &queue) {
    size_t bytes = src.size < dst.size ? src.size : dst.size;
    if constexpr ( From == alloc::Target::Host && To == alloc::Target::Host ) {
        std::memcpy(dst.data, src.data, bytes);
    } else {
        queue.memcpy(dst.data, src.data, bytes).wait();
    }
}

} // namespace alloc
