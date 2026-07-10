#pragma once
#include <sycl/sycl.hpp>
#include <alloc/buffer.h>

namespace alloc::raw {

/// Root host allocator — uses sycl::malloc_host.
struct HostAllocator {
    static constexpr alloc::Target tag = alloc::Target::Host;

    Buffer<alloc::Target::Host> allocate(size_t bytes, sycl::queue &queue) {
        void *p = sycl::malloc_host(bytes, queue);
        return {p, p ? bytes : 0};
    }

    void deallocate(Buffer<alloc::Target::Host> buf, sycl::queue &queue) {
        if (buf.data) sycl::free(buf.data, queue);
    }
};

} // namespace alloc::raw
