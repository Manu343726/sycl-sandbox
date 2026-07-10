#pragma once
#include <variant>

/// SYCL-compatible variant dispatch: recursive compile-time if/else-if chain.
/// Unlike std::visit, this does NOT use function pointers — it expands to
/// a chain of `if (index == 0) ... else if (index == 1) ...` at compile time,
/// which works on any SYCL backend including CUDA.
template <typename V, typename Visitor, size_t I = 0>
void visit_rt(V& v, Visitor&& vis) {
    if constexpr (I < std::variant_size_v<V>) {
        if (v.index() == I)
            vis(std::get<I>(v));
        else
            visit_rt<V, Visitor, I+1>(v, std::forward<Visitor>(vis));
    }
}
