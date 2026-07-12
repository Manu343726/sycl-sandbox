#pragma once

/// @file
/// Portable staging DisplayTarget.
///
/// Each slot is a pinned host USM buffer (sycl::malloc_host) when a queue
/// is present, or plain host memory in native mode.  The tone-map kernel
/// writes RGBA8 straight into the slot (device code can write host USM);
/// the slot becomes READY once the tone-map's event completes (polled
/// non-blocking from latest_ready).  present() uploads the slot with
/// glTexSubImage2D.  Works on every backend.

#include "display_target.h"
#include "texture.h"

#include <spdlog/spdlog.h>
#include <mutex>
#include <vector>
#include <cstdint>
#include <cstring>

class StagingDisplayTarget : public DisplayTarget {
public:
    bool init(sycl::queue *q, int w, int h, int slots) override {
        q_ = q;
        w_ = w;
        h_ = h;
        tex_ = create_render_texture(w, h);
        slots_.assign((size_t)slots, Slot{});
        alloc_slots();
        spdlog::info("[display] staging target {}x{} x{} slots ({})",
                     w, h, slots, q_ ? "pinned USM" : "host memory");
        return true;
    }

    void destroy() override {
        std::lock_guard<std::mutex> lk(mtx_);
        drain();
        free_slots();
        if (tex_) { glDeleteTextures(1, &tex_); tex_ = 0; }
    }

    void resize(int w, int h) override {
        std::lock_guard<std::mutex> lk(mtx_);
        drain();
        free_slots();
        w_ = w;
        h_ = h;
        alloc_slots();
        if (tex_) {
            glBindTexture(GL_TEXTURE_2D, tex_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        }
    }

    int acquire() override {
        std::lock_guard<std::mutex> lk(mtx_);
        for (int i = 0; i < (int)slots_.size(); ++i) {
            if (slots_[i].state == FREE) {
                slots_[i].state = INFLIGHT;
                return i;
            }
        }
        return -1; // all slots pending or displayed — back-pressure
    }

    uint8_t *staging_ptr(int slot) override { return slots_[slot].pixels; }

    void publish(int slot, const FrameInfo &info, sycl::event done) override {
        std::lock_guard<std::mutex> lk(mtx_);
        slots_[slot].info = info;
        slots_[slot].done = done;
        slots_[slot].state = PENDING;
    }

    bool latest_ready(int &slot, FrameInfo &info) override {
        std::lock_guard<std::mutex> lk(mtx_);
        // Promote PENDING slots whose device writes have finished.
        for (auto &s : slots_) {
            if (s.state == PENDING && event_complete(s.done))
                s.state = READY;
        }
        int best = -1;
        for (int i = 0; i < (int)slots_.size(); ++i) {
            if (slots_[i].state == READY &&
                (best < 0 ||
                 slots_[i].info.frame_index > slots_[best].info.frame_index))
                best = i;
        }
        if (best < 0) return false;
        slot = best;
        info = slots_[best].info;
        return true;
    }

    GLuint present(int slot) override {
        glBindTexture(GL_TEXTURE_2D, tex_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_, h_,
                        GL_RGBA, GL_UNSIGNED_BYTE, slots_[slot].pixels);
        // glTexSubImage2D copies synchronously, so the slot is free now.
        // Drop any older READY slots too — their frames are superseded.
        std::lock_guard<std::mutex> lk(mtx_);
        uint64_t shown = slots_[slot].info.frame_index;
        for (auto &s : slots_) {
            if (s.state == READY && s.info.frame_index <= shown)
                s.state = FREE;
        }
        slots_[slot].state = FREE;
        return tex_;
    }

    bool read_pixels(int slot, uint8_t *dst) override {
        if (slot < 0 || slot >= (int)slots_.size() || !slots_[slot].pixels)
            return false;
        std::memcpy(dst, slots_[slot].pixels, (size_t)w_ * h_ * 4);
        return true;
    }

    GLuint texture() const override { return tex_; }
    const char *name() const override { return "staging"; }

private:
    enum State { FREE, INFLIGHT, PENDING, READY };
    struct Slot {
        uint8_t *pixels = nullptr;
        State state = FREE;
        FrameInfo info;
        sycl::event done;
    };

    static bool event_complete(sycl::event &e) {
        // Default-constructed events (software backend) report complete.
        return e.get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete;
    }

    void drain() {
        if (q_) {
            try { q_->wait(); }
            catch (const std::exception &e) {
                spdlog::error("[display] drain failed: {}", e.what());
            }
        }
    }

    void alloc_slots() {
        size_t bytes = (size_t)w_ * h_ * 4;
        for (auto &s : slots_) {
            s.pixels = q_ ? sycl::malloc_host<uint8_t>(bytes, *q_)
                          : new uint8_t[bytes];
            std::memset(s.pixels, 0, bytes);
            s.state = FREE;
            s.info = {};
            s.done = sycl::event{};
        }
    }

    void free_slots() {
        for (auto &s : slots_) {
            if (!s.pixels) continue;
            if (q_) sycl::free(s.pixels, *q_);
            else delete[] s.pixels;
            s.pixels = nullptr;
        }
    }

    sycl::queue *q_ = nullptr;
    GLuint tex_ = 0;
    std::mutex mtx_;
    std::vector<Slot> slots_;
};
