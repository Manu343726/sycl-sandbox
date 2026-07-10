#pragma once
#include <sycl-sandbox/alloc/buffer.h>

namespace containers {

/// Typed view — alias for alloc::Buffer<Tag, T>.
template <alloc::Target Tag, typename T>
using Buffer = alloc::Buffer<Tag, T>;

} // namespace containers
