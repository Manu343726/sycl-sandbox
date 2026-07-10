#pragma once

/// Minimal POD optional — no constructors, no exceptions, trivially copyable.
/// Drop-in for std::optional in device code where exception infrastructure is
/// unavailable (AdaptiveCpp / hipSYCL).
namespace rt {

template <typename T>
struct Optional {
    bool has_value;
    T value;

    Optional() : has_value(false), value{} {
    }
    Optional(const T &v) : has_value(true), value(v) {
    }
    Optional(T &&v) : has_value(true), value(static_cast<T &&>(v)) {
    }

    Optional &operator=(const T &v) {
        has_value = true;
        value = v;
        return *this;
    }
    Optional &operator=(T &&v) {
        has_value = true;
        value = static_cast<T &&>(v);
        return *this;
    }

    explicit operator bool() const {
        return has_value;
    }
    const T *operator->() const {
        return &value;
    }
    T *operator->() {
        return &value;
    }
    const T &operator*() const {
        return value;
    }
    T &operator*() {
        return value;
    }
};

} // namespace rt
