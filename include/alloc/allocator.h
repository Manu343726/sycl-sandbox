#pragma once
#include <variant>
#include "root_allocator.h"
#include "linear_allocator.h"
#include "stack_allocator.h"

/// Composable allocator variant, keyed by Tag.
/// Only allocators matching the given tag are included.
template <AllocatorTag Tag>
using Allocator = std::variant<
    alloc::raw::RootAllocator<Tag>,
    alloc::raw::LinearAllocator<Tag>,
    alloc::raw::StackAllocator<Tag>
>;
