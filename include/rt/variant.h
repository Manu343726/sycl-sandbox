#pragma once
#include <variant>

// SYCL-compatible visit: recursive compile-time dispatch over variant index.
// Unlike std::visit, this does NOT use function pointers — it expands to
// a chain of if/else-if at compile time, which works on any SYCL backend
// including CUDA.
template <typename V, typename Visitor, size_t I = 0>
void visit_rt(V& v, Visitor&& vis) {
    if constexpr (I < std::variant_size_v<V>) {
        if (v.index() == I)
            vis(std::get<I>(v));
        else
            visit_rt<V, Visitor, I+1>(v, std::forward<Visitor>(vis));
    }
}
