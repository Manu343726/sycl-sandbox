#pragma once
#include <variant>
#include <sycl/sycl.hpp>
#include "../variant.h"
#include "host_allocator.h"
#include "device_allocator.h"
#include "linear_allocator.h"
#include "stack_allocator.h"
#include "arena_allocator.h"

/// Composable allocator variant.
using Allocator = std::variant<
    HostAllocator,
    DeviceAllocator,
    LinearAllocator<AllocatorTag::Host>,
    LinearAllocator<AllocatorTag::Device>,
    StackAllocator<AllocatorTag::Host>,
    StackAllocator<AllocatorTag::Device>
>;

/// Convenience: allocate through any allocator variant.
inline AnyBuffer alloc(Allocator &a, size_t bytes, sycl::queue &queue) {
    AnyBuffer result;
    result.size = bytes;
    visit(a, [&](auto &impl) {
        using T = std::decay_t<decltype(impl)>;
        auto buf = impl.allocate(bytes, queue);
        result.data = buf.data;
        result.tag  = T::tag;
    });
    return result;
}

/// Convenience: deallocate through any allocator variant.
inline void dealloc(Allocator &a, AnyBuffer buf, sycl::queue &queue) {
    visit(a, [&](auto &impl) {
        // Dispatch to the correct Buffer<Tag> based on the allocator's tag
        using T = std::decay_t<decltype(impl)>;
        if constexpr ( T::tag == AllocatorTag::Host ) {
            impl.deallocate(Buffer<AllocatorTag::Host>{buf.data, buf.size}, queue);
        } else {
            impl.deallocate(Buffer<AllocatorTag::Device>{buf.data, buf.size}, queue);
        }
    });
}
