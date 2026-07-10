#pragma once
#include "math.h"
#include "types.h"
#include <alloc/tag.h>
#include <alloc/buffer.h>
#include <alloc/host_allocator.h>
#include <alloc/device_allocator.h>
#include <alloc/transfer.h>

/// Host-side scene-building helpers for raytracer kernels.
namespace rt {

/// Which axis of a 3D coordinate system.
enum class Axis : int { X = 0, Y = 1, Z = 2 };

/// Compute one corner of an axis-aligned rectangle.
inline float3 quad_corner(Axis primary_axis,
                          float axis_value,
                          float min_second_axis,
                          float max_second_axis,
                          float min_third_axis,
                          float max_third_axis,
                          int corner_index) {
    int primary = static_cast<int>(primary_axis);
    int second_axis = (primary + 1) % 3;
    int third_axis = (primary + 2) % 3;
    float bounds_second[2] = {min_second_axis, max_second_axis};
    float bounds_third[2] = {min_third_axis, max_third_axis};
    float result[3] = {0, 0, 0};
    result[primary] = axis_value;
    result[second_axis] = bounds_second[corner_index & 1];
    result[third_axis] = bounds_third[corner_index >> 1];
    return {result[0], result[1], result[2]};
}

/// A host-resident buffer that can be transferred to the device.
///
/// Grows automatically on push_back (doubles capacity).
/// transfer_to_device() allocates device memory and copies contents.
/// The destructor frees both host and device allocations.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    DeviceBuffer(sycl::queue *queue, int initial_capacity = 64)
        : queue_(queue),
          host_(alloc::raw::HostAllocator().allocate(initial_capacity * sizeof(T), *queue_)) {
    }

    ~DeviceBuffer() {
        if ( host_.data ) {
            alloc::raw::HostAllocator().deallocate(host_, *queue_);
        }
        if ( device_.data ) {
            alloc::raw::DeviceAllocator().deallocate(device_, *queue_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept {
        *this = std::move(other);
    }

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
        std::swap(queue_, other.queue_);
        std::swap(host_, other.host_);
        std::swap(device_, other.device_);
        std::swap(capacity_, other.capacity_);
        std::swap(count_, other.count_);
        return *this;
    }

    void push_back(T object) {
        if ( count_ >= capacity_ ) {
            int new_cap = capacity_ * 2;
            auto new_host = alloc::raw::HostAllocator().allocate(new_cap * sizeof(T), *queue_);
            if ( !new_host.is_valid() ) {
                return;
            }
            T *src = static_cast<T *>(host_.data);
            T *dst = static_cast<T *>(new_host.data);
            for ( int i = 0; i < count_; i++ ) {
                dst[i] = std::move(src[i]);
            }
            alloc::raw::HostAllocator().deallocate(host_, *queue_);
            host_ = new_host;
            capacity_ = new_cap;
        }
        static_cast<T *>(host_.data)[count_++] = std::move(object);
    }

    /// Allocate device memory and copy host contents.
    void transfer_to_device() {
        size_t bytes = count_ * sizeof(T);
        if ( device_.data ) {
            alloc::raw::DeviceAllocator().deallocate(device_, *queue_);
        }
        device_ = alloc::raw::DeviceAllocator().allocate(bytes, *queue_);
        alloc::transfer(alloc::raw::Buffer<AllocatorTag::Host> {host_.data, bytes},
                        alloc::raw::Buffer<AllocatorTag::Device> {device_.data, bytes},
                        *queue_);
    }

    T *device_ptr() const {
        return static_cast<T *>(device_.data);
    }
    int size() const {
        return count_;
    }

private:
    sycl::queue *queue_ = nullptr;
    alloc::raw::Buffer<AllocatorTag::Host> host_;
    alloc::raw::Buffer<AllocatorTag::Device> device_;
    int capacity_ = 0;
    int count_ = 0;
};

} // namespace rt
