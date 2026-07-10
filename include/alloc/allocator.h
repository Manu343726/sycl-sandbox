#pragma once
#include <variant>
#include "root_allocator.h"
#include "linear_allocator.h"
#include "stack_allocator.h"

namespace alloc::raw {

/// Composable allocator variant, keyed by Target.
/// Only allocators matching the given target are included.
template <alloc::Target Tgt>
using Allocator = std::variant<
    RootAllocator<Tgt>,
    LinearAllocator<Tgt>,
    StackAllocator<Tgt>
>;

} // namespace alloc::raw
