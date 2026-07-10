#pragma once
#include <variant>
#include "root_allocator.h"
#include "linear_allocator.h"
#include "stack_allocator.h"

/// Composable allocator variant, keyed by Tag.
/// Only allocators matching the given tag are included.
template <AllocatorTag Tag>
using Allocator = std::variant<
    RootAllocator<Tag>,
    LinearAllocator<Tag>,
    StackAllocator<Tag>
>;
