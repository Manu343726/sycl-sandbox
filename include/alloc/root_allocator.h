#pragma once
#include <type_traits>
#include "host_allocator.h"
#include "device_allocator.h"
#include "buffer.h"

/// Tag-dispatched root allocator: HostAllocator when Tag=Host, DeviceAllocator when Tag=Device.
template <AllocatorTag Tag>
struct RootAllocator;

template <>
struct RootAllocator<AllocatorTag::Host> : HostAllocator {
    using HostAllocator::allocate;
    using HostAllocator::deallocate;
};

template <>
struct RootAllocator<AllocatorTag::Device> : DeviceAllocator {
    using DeviceAllocator::allocate;
    using DeviceAllocator::deallocate;
};
