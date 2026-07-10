#pragma once
#include <sycl/sycl.hpp>
#include "buffer.h"

namespace alloc::raw {

/// Root device allocator — uses sycl::malloc_device.
struct DeviceAllocator {
    static constexpr alloc::Target tag = alloc::Target::Device;

    Buffer<alloc::Target::Device> allocate(size_t bytes, sycl::queue &queue) {
        void *p = sycl::malloc_device(bytes, queue);
        return {p, p ? bytes : 0};
    }

    void deallocate(Buffer<alloc::Target::Device> buf, sycl::queue &queue) {
        if (buf.data) sycl::free(buf.data, queue);
    }
};

} // namespace alloc::raw
