#pragma once
#include "math.h"
#include "types.h"

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
        : queue_(queue), host_(sycl::malloc_host<T>(initial_capacity, *queue)),
          capacity_(initial_capacity) {
    }

    ~DeviceBuffer() {
        if ( host_ ) {
            sycl::free(host_, *queue_);
        }
        if ( device_ ) {
            sycl::free(device_, *queue_);
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
            T *new_host = sycl::malloc_host<T>(new_cap, *queue_);
            for ( int i = 0; i < count_; i++ ) {
                new_host[i] = std::move(host_[i]);
            }
            sycl::free(host_, *queue_);
            host_ = new_host;
            capacity_ = new_cap;
        }
        host_[count_++] = std::move(object);
    }

    /// Allocate device memory and copy host contents.
    void transfer_to_device() {
        if ( device_ ) {
            sycl::free(device_, *queue_);
        }
        device_ = sycl::malloc_device<T>(count_, *queue_);
        queue_->memcpy(device_, host_, count_ * sizeof(T)).wait();
    }

    T *device_ptr() const {
        return device_;
    }
    int size() const {
        return count_;
    }

private:
    sycl::queue *queue_ = nullptr;
    T *host_ = nullptr;
    T *device_ = nullptr;
    int capacity_ = 0;
    int count_ = 0;
};

} // namespace rt
