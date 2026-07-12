#pragma once

/// @file
/// DisplayTarget — abstraction over how a tone-mapped RGBA8 frame produced
/// on the device reaches the screen.
///
/// Two implementations:
///   - StagingDisplayTarget: portable.  Tone-map writes into a pinned host
///     USM slot (or plain host memory in native mode); present() uploads it
///     with glTexSubImage2D.  Works on every backend.
///   - CudaGLDisplayTarget: zero-copy.  Tone-map writes a device staging
///     buffer; present() copies it device-to-device into a CUDA-mapped
///     OpenGL PBO — the frame never touches host memory.  All CUDA code
///     stays inside src/display/cuda_*; callers only see this interface.
///
/// Threading contract:
///   - init / destroy / resize / present / latest_ready / read_pixels :
///     MAIN thread (OpenGL context current for init/destroy/resize/present)
///   - acquire / staging_ptr / publish : RENDER thread (SYCL queue owner)
/// Slots are triple-buffered so the render thread can tone-map frame N+1
/// while the main thread is still uploading frame N.
///
/// Synchronization contract: publish() receives the sycl::event of the
/// last device operation writing the slot (the tone-map).  A slot becomes
/// READY only once that event has completed — implementations must never
/// let present() observe a slot whose device writes are still in flight.

#include <cstdint>
#include <sycl/sycl.hpp>
#include <GLFW/glfw3.h>

/// Metadata published alongside each finished frame.
struct FrameInfo {
    uint64_t frame_index = 0;
    uint32_t spp = 0;
    float time_sec = 0.f;
};

class DisplayTarget {
public:
    virtual ~DisplayTarget() = default;

    /// Allocate `slots` back buffers at w×h.  Main thread, GL context
    /// current.  `q` may be null (software/native backend — synchronous
    /// host execution, default-constructed events count as complete).
    virtual bool init(sycl::queue *q, int w, int h, int slots) = 0;
    /// Release all resources.  Main thread.  Implementations drain the
    /// queue internally before freeing device-visible memory.
    virtual void destroy() = 0;
    /// Reallocate for a new resolution.  Main thread.  The caller must
    /// have stopped the render thread from acquiring new slots (see
    /// AppState::pause_pipeline); implementations additionally drain the
    /// queue before freeing.
    virtual void resize(int w, int h) = 0;

    // ── Render thread ────────────────────────────────────────────────
    /// Reserve a slot for the next frame.  Returns a slot index, or -1 if
    /// none is free yet (caller should sleep briefly and retry — this is
    /// the pipeline's back-pressure point).
    virtual int acquire() = 0;
    /// Buffer the tone-map kernel writes for the given slot (w*h*4 bytes).
    /// Device-writable: pinned host USM (staging) or device USM (CUDA-GL).
    virtual uint8_t *staging_ptr(int slot) = 0;
    /// Mark the slot as pending: it becomes READY (visible to
    /// latest_ready) once `done` — the event of the tone-map that wrote
    /// the slot — completes.  A default-constructed event is treated as
    /// already complete (software backend).
    virtual void publish(int slot, const FrameInfo &info, sycl::event done) = 0;

    // ── Main thread ──────────────────────────────────────────────────
    /// If a finished frame is available, set `slot`+`info` and return
    /// true.  Non-blocking; polls event completion internally.
    virtual bool latest_ready(int &slot, FrameInfo &info) = 0;
    /// Upload the slot to the display texture, release it, and return the
    /// texture for ImGui.  The texture's origin is bottom-left (the
    /// tone-map Y-flips), so the viewport draws it with standard
    /// (0,0)-(1,1) UVs.
    virtual GLuint present(int slot) = 0;
    /// Copy the slot's finished RGBA8 pixels (bottom-up rows, as stored)
    /// into `dst` (w*h*4 bytes).  Call between latest_ready() and
    /// present().  Used for MCP viewport capture.
    virtual bool read_pixels(int slot, uint8_t *dst) = 0;

    /// Current display texture (may be 0 before the first present()).
    virtual GLuint texture() const = 0;

    virtual const char *name() const = 0;

    int width() const { return w_; }
    int height() const { return h_; }

protected:
    int w_ = 0, h_ = 0;
};

/// Create and initialize the best display target for the queue: CUDA-GL
/// zero-copy when the queue runs on an NVIDIA device and the interop
/// self-test passes, otherwise the portable staging path.  Never returns
/// null.  Override with SYCL_SANDBOX_DISPLAY=staging|cuda.
DisplayTarget *create_display_target(sycl::queue *q, int w, int h,
                                     int slots = 3);
