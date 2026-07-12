#pragma once

/// @file
/// ParamStore — thread-safe parameter hand-off from UI/MCP to the render
/// thread.
///
/// The UI keeps editing its own copy of the parameter block (the scene
/// descriptor's current_buffer) and publishes a snapshot here whenever
/// something changed.  The render thread fetches at most once per frame
/// and copies the snapshot into the host-USM parameter buffer it owns
/// (d_params).  Neither side ever touches the other's memory, so no
/// unprotected sharing remains.
///
/// A publish can also request an accumulation restart (camera moved,
/// material changed, ...) which the render thread services atomically
/// with the parameter switch — the frame that first uses the new params
/// is also the first frame of the new accumulation.

#include <mutex>
#include <vector>
#include <cstdint>
#include <cstring>

class ParamStore {
public:
    /// UI/MCP thread: publish a new parameter snapshot.
    /// @param restart_accum  true if the change invalidates accumulated
    ///                       samples (most param changes do).
    void publish(const void *data, size_t bytes, bool restart_accum) {
        std::lock_guard<std::mutex> lk(mtx_);
        buffer_.assign(static_cast<const uint8_t *>(data),
                       static_cast<const uint8_t *>(data) + bytes);
        restart_ |= restart_accum;
        ++generation_;
    }

    /// Request an accumulation restart without touching params.
    void request_restart() {
        std::lock_guard<std::mutex> lk(mtx_);
        restart_ = true;
        ++generation_;
    }

    /// Render thread: if anything was published since the last fetch,
    /// copy the snapshot into `out` (resized to fit; empty when the
    /// publish was restart-only) and set `restart_accum`.  Returns false
    /// (leaving the outputs untouched) when nothing changed.
    bool fetch(std::vector<uint8_t> &out, bool &restart_accum) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (generation_ == fetched_generation_) return false;
        fetched_generation_ = generation_;
        out = buffer_;
        restart_accum = restart_;
        restart_ = false;
        return true;
    }

    /// Drop pending state (scene switch — the old snapshot's layout no
    /// longer matches).
    void reset() {
        std::lock_guard<std::mutex> lk(mtx_);
        buffer_.clear();
        restart_ = false;
        fetched_generation_ = generation_;
    }

private:
    std::mutex mtx_;
    std::vector<uint8_t> buffer_;
    uint64_t generation_ = 0;
    uint64_t fetched_generation_ = 0;
    bool restart_ = false;
};
