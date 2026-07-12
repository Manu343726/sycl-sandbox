#pragma once

/// @file
/// Seqlock-protected statistics publication.
///
/// The render thread publishes the per-frame stat block after each frame
/// with set_data(); the UI thread polls with try_read().  The seqlock
/// guarantees the reader either gets a consistent snapshot or retries —
/// the writer (render thread) never blocks on readers.

#include <vector>
#include <atomic>
#include <cstring>
#include <cstddef>
#include <cstdint>

class PublishedStats {
public:
    /// (Re)size the stat block.  Call only while the render thread is not
    /// publishing (scene switch, under pause).
    void resize(size_t floats) {
        data_.assign(floats, 0.f);
        seq_.store(0, std::memory_order_relaxed);
    }

    size_t size() const { return data_.size(); }

    /// Render thread: publish a new stat snapshot (`floats` values).
    void set_data(const float *src, size_t floats) {
        if (floats > data_.size() || data_.empty()) return;
        seq_.fetch_add(1, std::memory_order_acquire);   // odd: write begins
        std::memcpy(data_.data(), src, floats * sizeof(float));
        seq_.fetch_add(1, std::memory_order_release);   // even: write done
    }

    /// UI thread: copy a consistent snapshot into `dst` (must hold at
    /// least size() floats).  Returns false if a write was in progress —
    /// caller keeps the previous values and retries next frame.
    bool try_read(float *dst) const {
        if (data_.empty()) return false;
        uint32_t s1 = seq_.load(std::memory_order_acquire);
        if (s1 & 1) return false;
        std::memcpy(dst, data_.data(), data_.size() * sizeof(float));
        std::atomic_thread_fence(std::memory_order_acquire);
        uint32_t s2 = seq_.load(std::memory_order_relaxed);
        return s1 == s2;
    }

private:
    std::atomic<uint32_t> seq_{0};
    std::vector<float> data_;
};
