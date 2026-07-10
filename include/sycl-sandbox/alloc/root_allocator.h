#pragma once
#include <type_traits>
#include <sycl-sandbox/alloc/host_allocator.h>
#include <sycl-sandbox/alloc/device_allocator.h>
#include <sycl-sandbox/alloc/buffer.h>

namespace alloc::raw {

/// Tag-dispatched root allocator: HostAllocator when Tag=Host, DeviceAllocator when Tag=Device.
template <alloc::Target Tag>
struct RootAllocator;

template <>
struct RootAllocator<alloc::Target::Host> : HostAllocator {
    using HostAllocator::allocate;
    using HostAllocator::deallocate;
};

template <>
struct RootAllocator<alloc::Target::Device> : DeviceAllocator {
    using DeviceAllocator::allocate;
    using DeviceAllocator::deallocate;
};

} // namespace alloc::raw
