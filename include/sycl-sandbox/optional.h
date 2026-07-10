#pragma once

/// Minimal POD optional — no exceptions, trivially copyable.
/// Drop-in for std::optional in device code where exception infrastructure is
/// unavailable (AdaptiveCpp / hipSYCL).
namespace rt {

struct nullopt_t {
    explicit constexpr nullopt_t(int) {
    }
};
inline constexpr nullopt_t nullopt {0};

template <typename T>
struct optional {
    bool has_value;
    T value;

    constexpr optional() : has_value(false), value {} {
    }
    constexpr optional(nullopt_t) : has_value(false), value {} {
    }
    constexpr optional(const T &v) : has_value(true), value(v) {
    }
    constexpr optional(T &&v) : has_value(true), value(static_cast<T &&>(v)) {
    }

    optional &operator=(nullopt_t) {
        has_value = false;
        return *this;
    }
    optional &operator=(const T &v) {
        has_value = true;
        value = v;
        return *this;
    }
    optional &operator=(T &&v) {
        has_value = true;
        value = static_cast<T &&>(v);
        return *this;
    }

    explicit constexpr operator bool() const {
        return has_value;
    }
    constexpr const T *operator->() const {
        return &value;
    }
    constexpr T *operator->() {
        return &value;
    }
    constexpr const T &operator*() const {
        return value;
    }
    constexpr T &operator*() {
        return value;
    }
};

} // namespace rt
