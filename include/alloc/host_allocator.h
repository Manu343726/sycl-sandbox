#pragma once
#include <sycl/sycl.hpp>
#include "buffer.h"

namespace alloc::raw {

/// Root host allocator — uses sycl::malloc_host.
struct HostAllocator {
    static constexpr AllocatorTag tag = AllocatorTag::Host;

    Buffer<AllocatorTag::Host> allocate(size_t bytes, sycl::queue &queue) {
        void *p = sycl::malloc_host(bytes, queue);
        return {p, p ? bytes : 0};
    }

    void deallocate(Buffer<AllocatorTag::Host> buf, sycl::queue &queue) {
        if (buf.data) sycl::free(buf.data, queue);
    }
};

} // namespace alloc::raw
