#pragma once
#include <cstring>
#include <sycl/sycl.hpp>
#include "buffer.h"

/// Copy data between buffers of possibly-different tags.
///
/// Uses compile-time dispatch to select the correct copy primitive:
///   Host → Host   : std::memcpy
///   Host → Device  : queue.memcpy
///   Device → Host  : queue.memcpy
///   Device → Device: queue.memcpy
template <AllocatorTag From, AllocatorTag To>
void transfer(Buffer<From> src, Buffer<To> dst, sycl::queue &queue) {
    size_t bytes = src.size < dst.size ? src.size : dst.size;
    if constexpr ( From == AllocatorTag::Host && To == AllocatorTag::Host ) {
        std::memcpy(dst.data, src.data, bytes);
    } else {
        queue.memcpy(dst.data, src.data, bytes).wait();
    }
}
