#pragma once

/// @file
/// Zero-copy CUDA-GL DisplayTarget.
///
/// The tone-map kernel writes RGBA8 into a *device* staging buffer
/// (sycl::malloc_device — on the AdaptiveCpp CUDA backend this is a
/// cudaMalloc pointer in the device's primary context).  present() then:
///   1. waits the slot's tone-map event (host-side, frame is normally
///      already complete by then),
///   2. maps the slot's GL PBO into CUDA,
///   3. cuMemcpyDtoD device-staging → PBO   (never touches host memory),
///   4. unmaps (hands the buffer back to GL),
///   5. glTexSubImage2D sources the texture from the bound PBO (DMA).
///
/// All CUDA specifics live here and in cuda_api.*; the rest of the app
/// sees only the DisplayTarget interface.  init() runs a full self-test
/// (register + map + DtoD + unmap) and returns false on ANY failure, so
/// the factory can silently fall back to the portable staging target.

#include "display_target.h"
#include "cuda_api.h"
#include <mutex>
#include <vector>
#include <cstdint>

class CudaGLDisplayTarget : public DisplayTarget {
public:
    ~CudaGLDisplayTarget() override;

    bool init(sycl::queue *q, int w, int h, int slots) override;
    void destroy() override;
    void resize(int w, int h) override;

    int acquire() override;
    uint8_t *staging_ptr(int slot) override;
    void publish(int slot, const FrameInfo &info, sycl::event done) override;

    bool latest_ready(int &slot, FrameInfo &info) override;
    GLuint present(int slot) override;
    bool read_pixels(int slot, uint8_t *dst) override;

    GLuint texture() const override { return tex_; }
    const char *name() const override { return "cuda-gl zero-copy"; }

private:
    enum State { FREE, INFLIGHT, PENDING, READY };
    struct Slot {
        GLuint pbo = 0;
        cuda_api::CUgraphicsResource resource = nullptr;
        uint8_t *d_staging = nullptr;   ///< device USM, tone-map destination
        State state = FREE;
        FrameInfo info;
        sycl::event done;
    };

    bool init_cuda_context();
    bool alloc_slots(int slots);
    void free_slots();
    bool self_test();
    void drain();

    sycl::queue *q_ = nullptr;
    GLuint tex_ = 0;
    cuda_api::CUcontext cu_ctx_ = nullptr;
    cuda_api::CUdevice cu_dev_ = 0;
    bool ctx_retained_ = false;
    std::mutex mtx_;
    std::vector<Slot> slots_;
};
