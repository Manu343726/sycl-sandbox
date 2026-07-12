#pragma once
#include <sycl-sandbox/profiler.h>
#include <sycl-sandbox/profiler_device.h>
#include <sycl-sandbox/kernel/execution_context.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <deque>
#include <mutex>
#include <spdlog/spdlog.h>

// ── ZoneSample ───────────────────────────────────────────────────────
/// A single zone execution interval, produced by matching one enter/exit pair.
struct ZoneSample {
    std::string name;
    uint64_t start_cycles;
    uint64_t end_cycles;
    uint32_t zone_id;
    int64_t frame_index;
    int depth;          ///< nesting depth within its frame

    uint64_t duration() const { return end_cycles - start_cycles; }
};

// ── KernelMessage ────────────────────────────────────────────────────
/// A text message emitted by the kernel during a frame.
struct KernelMessage {
    std::string text;
    uint64_t timestamp_cycles;
    int64_t frame_index;
};

// ── KernelPlot ───────────────────────────────────────────────────────
/// A named plot value emitted by the kernel during a frame.
struct KernelPlot {
    std::string name;
    float value;
    uint64_t timestamp_cycles;
    int64_t frame_index;
};

// ── ZoneAggregate ────────────────────────────────────────────────────
/// Aggregated statistics for all samples of a given zone name.
struct ZoneAggregate {
    std::string name;
    uint64_t total_duration = 0;
    uint64_t min_duration = UINT64_MAX;
    uint64_t max_duration = 0;
    uint64_t last_duration = 0;
    int64_t sample_count = 0;
    double avg_duration() const {
        return sample_count > 0 ? (double)total_duration / sample_count : 0.0;
    }
};

// ── KernelProfiler ───────────────────────────────────────────────────
class KernelProfiler {
public:
    static constexpr int MAX_SAMPLES = 100000;  // ring-buffer capacity

    KernelProfiler() = default;

    void init(rt::Runtime *rt) {
        rt_ = rt;
        size_t buf_size = sizeof(profiler::BufferHeader)
                        + profiler::MAX_RECORDS * sizeof(profiler::Record);
        if (buffer_) { rt_->dealloc(static_cast<uint8_t *>(buffer_)); buffer_ = nullptr; }
        buffer_ = rt_->alloc_host<uint8_t>(buf_size / sizeof(uint8_t));
        enabled_ = true;
        reset_frame();
    }

    void shutdown() {
        free_device_ring();
        if (buffer_ && rt_) { rt_->dealloc(buffer_); buffer_ = nullptr; }
        enabled_ = false;
        samples_.clear();
        by_name_.clear();
    }

    void reset_frame() {
        if (!buffer_ || !enabled_) return;
        auto *header = reinterpret_cast<profiler::BufferHeader *>(buffer_);
        header->write_pos = 0;
        last_read_pos_ = 0;
    }

    void *buffer() const { return static_cast<void *>(buffer_); }
    bool is_enabled() const { return enabled_; }
    void set_enabled(bool e) { enabled_ = e; }
    bool is_paused() const { return paused_; }
    void set_paused(bool p) { paused_ = p; }

    /// Clear all collected samples, messages, plots, and aggregates.
    void clear_data() {
        std::lock_guard<std::mutex> lock(mtx_);
        samples_.clear();
        messages_.clear();
        plots_.clear();
        by_name_.clear();
        total_frames_ = 0;
    }

    /// Process records after kernel execution.
    /// Matches enter/exit pairs into ZoneSamples and appends them to the timeline.
    /// Also extracts messages, plot values, and alloc/free events.
    /// When paused, records are discarded (buffer reset) but collected data is preserved.
    void collect(int64_t frame_index) {
        if (!buffer_ || !enabled_) return;
        if (paused_) { reset_frame(); return; }
        std::lock_guard<std::mutex> lock(mtx_);
        auto *header = reinterpret_cast<profiler::BufferHeader *>(buffer_);
        auto *raw = reinterpret_cast<profiler::Record *>(header + 1);

        const uint32_t wp = header->write_pos;
        const uint32_t cap = profiler::MAX_RECORDS;
        if (wp <= last_read_pos_) return;

        // Handle ring wrap: if we're more than cap behind, some records
        // were overwritten. Advance to the start of the newest cap records.
        if (wp - last_read_pos_ > cap) {
            last_read_pos_ = wp - cap;
        }

        // Stack-based enter/exit matching
        struct Frame {
            uint64_t start_cycles;
            uint32_t zone_id;
            std::string name;
            int depth;
        };
        std::vector<Frame> stack;

        for (uint32_t i = last_read_pos_; i < wp; i++) {
            uint32_t slot = i % cap;
            auto &rec = raw[slot];
            switch (rec.type) {
            case profiler::REC_ZONE_ENTER: { // enter
                Frame f;
                f.start_cycles = rec.timestamp;
                f.zone_id = rec.zone_id;
                f.name = rec.name[0] ? rec.name : "zone_" + std::to_string(rec.zone_id);
                f.depth = (int)stack.size();
                stack.push_back(std::move(f));
                break;
            }
            case profiler::REC_ZONE_EXIT: { // exit
                if (stack.empty()) break;
                Frame f = std::move(stack.back());
                stack.pop_back();

                ZoneSample s;
                s.name = std::move(f.name);
                s.start_cycles = f.start_cycles;
                s.end_cycles = rec.timestamp;
                s.zone_id = f.zone_id;
                s.frame_index = frame_index;
                s.depth = f.depth;

                if (samples_.size() >= MAX_SAMPLES)
                    samples_.pop_front();
                samples_.push_back(s);

                // Update per-name aggregates
                auto &agg = by_name_[s.name];
                agg.name = s.name;
                agg.total_duration += s.duration();
                agg.min_duration = std::min(agg.min_duration, s.duration());
                agg.max_duration = std::max(agg.max_duration, s.duration());
                agg.last_duration = s.duration();
                agg.sample_count++;
                break;
            }
            case profiler::REC_MSG: {
                KernelMessage msg;
                msg.text = rec.name[0] ? rec.name : "";
                msg.timestamp_cycles = rec.timestamp;
                msg.frame_index = frame_index;
                if (messages_.size() >= MAX_SAMPLES)
                    messages_.pop_front();
                messages_.push_back(std::move(msg));
                break;
            }
            case profiler::REC_PLOT: {
                KernelPlot plt;
                plt.name = rec.name[0] ? rec.name : "";
                plt.value = rec.plot_value();
                plt.timestamp_cycles = rec.timestamp;
                plt.frame_index = frame_index;
                if (plots_.size() >= MAX_SAMPLES)
                    plots_.pop_front();
                plots_.push_back(std::move(plt));
                break;
            }
            case profiler::REC_ALLOC:
            case profiler::REC_FREE:
                // Alloc/free tracking stored in name field — future use
                break;
            }
        }

        // ── Flush unclosed zones (ring overflow recovery) ────────────
        // When the ring wraps, some enter records may have their matching
        // exit overwritten. Close any remaining zones so the outer frames
        // are still visible in the timeline.
        if (!stack.empty()) {
            uint64_t fallback_end = 0;
            if (wp > last_read_pos_) {
                uint32_t last_slot = (wp - 1) % cap;
                fallback_end = raw[last_slot].timestamp;
            }
            // Walk from the innermost (stack top) outward
            for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
                ZoneSample s;
                s.name = it->name;
                s.start_cycles = it->start_cycles;
                s.end_cycles = fallback_end;
                s.zone_id = it->zone_id;
                s.frame_index = frame_index;
                s.depth = it->depth;
                if (samples_.size() >= MAX_SAMPLES)
                    samples_.pop_front();
                samples_.push_back(s);
                auto &agg = by_name_[s.name];
                agg.name = s.name;
                agg.total_duration += s.duration();
                agg.min_duration = std::min(agg.min_duration, s.duration());
                agg.max_duration = std::max(agg.max_duration, s.duration());
                agg.last_duration = s.duration();
                agg.sample_count++;
            }
        }

        if (!samples_.empty()) {
            total_frames_ = frame_index + 1;
        }

        last_read_pos_ = wp;
    }

    /// Thread-safe: returns a copy so the caller holds no dangling reference.
    std::deque<ZoneSample> samples() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return samples_;
    }

    /// Thread-safe: returns a copy.
    std::deque<KernelMessage> messages() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return messages_;
    }

    /// Thread-safe: returns a copy.
    std::deque<KernelPlot> plots() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return plots_;
    }

    /// Get the absolute cycle range across all samples.
    uint64_t timeline_start() const {
        std::lock_guard<std::mutex> lock(mtx_);
        if (samples_.empty()) return 0;
        return samples_.front().start_cycles;
    }
    uint64_t timeline_end() const {
        std::lock_guard<std::mutex> lock(mtx_);
        if (samples_.empty()) return 0;
        return samples_.back().end_cycles;
    }

    /// Get aggregated stats for a zone name.
    const ZoneAggregate *zone_stats(const std::string &name) const {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = by_name_.find(name);
        return it != by_name_.end() ? &it->second : nullptr;
    }

    /// Get all zone aggregates sorted by total time (descending).
    std::vector<const ZoneAggregate *> top_zones(int count = 50) const {
        std::lock_guard<std::mutex> lock(mtx_);
        std::vector<const ZoneAggregate *> result;
        for (auto &[name, agg] : by_name_)
            result.push_back(&agg);
        std::sort(result.begin(), result.end(), [](auto *a, auto *b) {
            return a->total_duration > b->total_duration;
        });
        if ((int)result.size() > count) result.resize(count);
        return result;
    }

    int64_t total_frames() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return total_frames_;
    }

    // ── Device profiler ring ─────────────────────────────────────────
    // Zones executed *inside* SYCL kernels can't reach the host ring in
    // the .so's data segment, so they write into a device-USM ring passed
    // by value through RenderContext::prof.  Once per frame — after the
    // frame's completion event — collect_device() reads the ring back,
    // matches enter/exit pairs per work-item lane, maps the device cycle
    // timestamps linearly into the frame's host time bracket, and feeds
    // the samples (named "gpu:<zone>") into the same timeline /
    // top-zones / detail views as host zones.

    /// Allocate the device ring on `q` (null → device profiling off,
    /// e.g. the software backend).  Re-call after a backend switch —
    /// free_device_ring() first, while the old queue is still alive.
    void init_device_ring(sycl::queue *q) {
        free_device_ring();
        ring_q_ = q;
        if (!q) return;
        d_ring_header_ = sycl::malloc_device<profiler::DeviceRingHeader>(1, *q);
        d_ring_records_ =
            sycl::malloc_device<profiler::DeviceRecord>(RING_CAPACITY, *q);
        q->memset(d_ring_header_, 0, sizeof(profiler::DeviceRingHeader)).wait();
        ring_staging_.resize(RING_CAPACITY);
        spdlog::debug("[profiler] device ring: {} records ({} KiB device)",
                      RING_CAPACITY,
                      RING_CAPACITY * sizeof(profiler::DeviceRecord) / 1024);
    }

    void free_device_ring() {
        if (ring_q_) {
            if (d_ring_header_) sycl::free(d_ring_header_, *ring_q_);
            if (d_ring_records_) sycl::free(d_ring_records_, *ring_q_);
        }
        d_ring_header_ = nullptr;
        d_ring_records_ = nullptr;
        ring_q_ = nullptr;
    }

    /// POD handle for RenderContext::prof.  Inactive (all-null) when the
    /// profiler is disabled/paused or no ring is allocated, which turns
    /// every device zone into a no-op.
    /// @param work_items  size of the launch — used to derive a sampling
    ///        interval that keeps roughly SAMPLED_ITEMS work items
    ///        recording per frame.
    profiler::DeviceRing device_ring(size_t work_items) const {
        if (!enabled_ || paused_ || !d_ring_header_) return {};
        profiler::DeviceRing ring;
        ring.header = d_ring_header_;
        ring.records = d_ring_records_;
        ring.capacity = RING_CAPACITY;
        ring.sample_interval =
            (uint32_t)std::max<size_t>(1, work_items / SAMPLED_ITEMS);
        return ring;
    }

    /// Resolve zone-id hashes back to names in the UI.  Register every
    /// name used with PROFILER_DEVICE_ZONE (hashes match — same FNV-1a).
    void register_device_zone(const char *name) {
        std::lock_guard<std::mutex> lock(mtx_);
        zone_names_[profiler::device_zone_hash(name)] = name;
    }

    /// Read back + reset the device ring and ingest its records.
    /// `host_t0/t1` bracket the frame in host cycles (profiler::
    /// read_timestamp() before dispatch and after the completion event);
    /// device timestamps are mapped linearly into that bracket.
    void collect_device(int64_t frame_index, uint64_t host_t0,
                        uint64_t host_t1) {
        if (!ring_q_ || !d_ring_header_ || !enabled_ || paused_) return;
        profiler::DeviceRingHeader hdr{};
        ring_q_->memcpy(&hdr, d_ring_header_, sizeof(hdr)).wait();
        uint32_t written = std::min(hdr.write_pos, RING_CAPACITY);
        if (!written) return;
        ring_q_->memcpy(ring_staging_.data(), d_ring_records_,
                        written * sizeof(profiler::DeviceRecord)).wait();
        ring_q_->memset(d_ring_header_, 0,
                        sizeof(profiler::DeviceRingHeader)).wait();

        // Device→host time mapping: linear fit of the observed device
        // timestamp range onto the frame's host bracket.
        uint64_t dev_min = UINT64_MAX, dev_max = 0;
        for (uint32_t i = 0; i < written; ++i) {
            uint64_t ts = ring_staging_[i].timestamp;
            if (!ts) continue;
            dev_min = std::min(dev_min, ts);
            dev_max = std::max(dev_max, ts);
        }
        const bool have_time = dev_max > dev_min;
        auto to_host = [&](uint64_t ts) -> uint64_t {
            if (!have_time || host_t1 <= host_t0) return host_t0;
            return host_t0 + (uint64_t)((double)(ts - dev_min) /
                                        (double)(dev_max - dev_min) *
                                        (double)(host_t1 - host_t0));
        };

        std::lock_guard<std::mutex> lock(mtx_);
        // Per-lane enter stacks: ring order preserves each lane's program
        // order (atomic slot claims are monotone per thread), so a plain
        // stack per lane reconstructs nesting.
        std::unordered_map<uint32_t, std::vector<profiler::DeviceRecord>> open;
        for (uint32_t i = 0; i < written; ++i) {
            const auto &rec = ring_staging_[i];
            uint32_t key = rec.lane;
            if (rec.type == profiler::DEV_ZONE_ENTER) {
                open[key].push_back(rec);
            } else if (rec.type == profiler::DEV_ZONE_EXIT) {
                auto it = open.find(key);
                if (it == open.end() || it->second.empty()) continue;
                profiler::DeviceRecord enter = it->second.back();
                it->second.pop_back();
                if (enter.zone_id != rec.zone_id) continue;

                ZoneSample s;
                auto nit = zone_names_.find(rec.zone_id);
                s.name = "gpu:" + (nit != zone_names_.end()
                                       ? nit->second
                                       : "zone_" + std::to_string(rec.zone_id));
                s.start_cycles = to_host(enter.timestamp);
                s.end_cycles = to_host(rec.timestamp);
                s.zone_id = rec.zone_id;
                s.frame_index = frame_index;
                s.depth = (int)it->second.size();
                if (samples_.size() >= MAX_SAMPLES) samples_.pop_front();
                samples_.push_back(s);
                auto &agg = by_name_[s.name];
                agg.name = s.name;
                agg.total_duration += s.duration();
                agg.min_duration = std::min(agg.min_duration, s.duration());
                agg.max_duration = std::max(agg.max_duration, s.duration());
                agg.last_duration = s.duration();
                agg.sample_count++;
            }
        }
    }

private:
    static constexpr uint32_t RING_CAPACITY = 1u << 15;  // 32k × 16 B = 512 KiB
    static constexpr size_t SAMPLED_ITEMS = 64;  // ~work items recording/frame

    sycl::queue *ring_q_ = nullptr;
    profiler::DeviceRingHeader *d_ring_header_ = nullptr;
    profiler::DeviceRecord *d_ring_records_ = nullptr;
    std::vector<profiler::DeviceRecord> ring_staging_;
    std::unordered_map<uint32_t, std::string> zone_names_;

    rt::Runtime *rt_ = nullptr;
    uint8_t *buffer_ = nullptr;
    bool enabled_ = false;
    bool paused_ = false;
    uint32_t last_read_pos_ = 0;

    mutable std::mutex mtx_;
    std::deque<ZoneSample> samples_;
    std::deque<KernelMessage> messages_;
    std::deque<KernelPlot> plots_;
    std::unordered_map<std::string, ZoneAggregate> by_name_;
    int64_t total_frames_ = 0;
};
