#include "cuda_gl_target.h"
#include "texture.h"

#include <spdlog/spdlog.h>
#include <GLFW/glfw3.h>
#include <cstring>

// ── GL PBO entry points (loaded at runtime — core-profile safe) ───────
// glTexSubImage2D is GL 1.1 and linked directly; buffer objects are not,
// so resolve them through GLFW's loader.

#ifndef GL_PIXEL_UNPACK_BUFFER
#define GL_PIXEL_UNPACK_BUFFER 0x88EC
#endif
#ifndef GL_STREAM_DRAW
#define GL_STREAM_DRAW 0x88E0
#endif

namespace {
using glGenBuffers_fn = void (*)(GLsizei, GLuint *);
using glDeleteBuffers_fn = void (*)(GLsizei, const GLuint *);
using glBindBuffer_fn = void (*)(GLenum, GLuint);
using glBufferData_fn = void (*)(GLenum, ptrdiff_t, const void *, GLenum);

glGenBuffers_fn p_glGenBuffers = nullptr;
glDeleteBuffers_fn p_glDeleteBuffers = nullptr;
glBindBuffer_fn p_glBindBuffer = nullptr;
glBufferData_fn p_glBufferData = nullptr;

bool load_gl_functions() {
    if (p_glGenBuffers) return true;
    p_glGenBuffers = (glGenBuffers_fn)glfwGetProcAddress("glGenBuffers");
    p_glDeleteBuffers = (glDeleteBuffers_fn)glfwGetProcAddress("glDeleteBuffers");
    p_glBindBuffer = (glBindBuffer_fn)glfwGetProcAddress("glBindBuffer");
    p_glBufferData = (glBufferData_fn)glfwGetProcAddress("glBufferData");
    return p_glGenBuffers && p_glDeleteBuffers && p_glBindBuffer &&
           p_glBufferData;
}

constexpr unsigned int CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x02;

bool cu_ok(cuda_api::CUresult r, const char *what) {
    if (r == cuda_api::CUDA_SUCCESS) return true;
    spdlog::warn("[cuda_gl] {} failed: {} ({})", what,
                 cuda_api::error_name(r), r);
    return false;
}
} // namespace

CudaGLDisplayTarget::~CudaGLDisplayTarget() { destroy(); }

bool CudaGLDisplayTarget::init_cuda_context() {
    using namespace cuda_api;
    if (!cu_ok(cuInit(0), "cuInit")) return false;
    if (!cu_ok(cuDeviceGet(&cu_dev_, 0), "cuDeviceGet")) return false;
    // AdaptiveCpp's CUDA backend drives the device through the runtime
    // API, which uses the primary context — retaining it here means our
    // sycl::malloc_device pointers and the mapped PBO live in the same
    // address space, making the DtoD copy below valid.
    if (!cu_ok(cuDevicePrimaryCtxRetain(&cu_ctx_, cu_dev_),
               "cuDevicePrimaryCtxRetain"))
        return false;
    ctx_retained_ = true;
    return cu_ok(cuCtxSetCurrent(cu_ctx_), "cuCtxSetCurrent");
}

bool CudaGLDisplayTarget::alloc_slots(int slots) {
    using namespace cuda_api;
    size_t bytes = (size_t)w_ * h_ * 4;
    slots_.assign((size_t)slots, Slot{});
    for (auto &s : slots_) {
        s.d_staging = sycl::malloc_device<uint8_t>(bytes, *q_);
        if (!s.d_staging) return false;
        p_glGenBuffers(1, &s.pbo);
        p_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, s.pbo);
        p_glBufferData(GL_PIXEL_UNPACK_BUFFER, (ptrdiff_t)bytes, nullptr,
                       GL_STREAM_DRAW);
        p_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        if (!cu_ok(cuGraphicsGLRegisterBuffer(
                       &s.resource, s.pbo,
                       CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD),
                   "cuGraphicsGLRegisterBuffer"))
            return false;
    }
    return true;
}

bool CudaGLDisplayTarget::self_test() {
    using namespace cuda_api;
    // Full round-trip on slot 0: map, get pointer, DtoD from the SYCL
    // device allocation, unmap.  Any failure disqualifies the target.
    Slot &s = slots_[0];
    size_t bytes = (size_t)w_ * h_ * 4;
    q_->memset(s.d_staging, 0, bytes).wait();
    if (!cu_ok(cuGraphicsMapResources(1, &s.resource, nullptr), "map"))
        return false;
    CUdeviceptr pbo_ptr = 0;
    size_t pbo_size = 0;
    bool ok =
        cu_ok(cuGraphicsResourceGetMappedPointer(&pbo_ptr, &pbo_size,
                                                 s.resource),
              "get mapped pointer") &&
        pbo_size >= bytes &&
        cu_ok(cuMemcpyDtoD(pbo_ptr, (CUdeviceptr)(uintptr_t)s.d_staging,
                           bytes),
              "DtoD") &&
        cu_ok(cuStreamSynchronize(nullptr), "sync");
    cu_ok(cuGraphicsUnmapResources(1, &s.resource, nullptr), "unmap");
    return ok;
}

bool CudaGLDisplayTarget::init(sycl::queue *q, int w, int h, int slots) {
    if (!q || !cuda_api::load() || !load_gl_functions()) return false;
    q_ = q;
    w_ = w;
    h_ = h;
    if (!init_cuda_context()) { destroy(); return false; }
    tex_ = create_render_texture(w, h);
    if (!alloc_slots(slots) || !self_test()) {
        spdlog::warn("[cuda_gl] interop self-test failed — falling back");
        destroy();
        return false;
    }
    spdlog::info("[display] CUDA-GL zero-copy target {}x{} x{} slots", w, h,
                 slots);
    return true;
}

void CudaGLDisplayTarget::free_slots() {
    // Full teardown — drains USM, unregisters CUDA-GL resources, AND
    // deletes the GL PBO buffers.  Use from the UI thread (GL context
    // current) — release_device_buffers() is the render-thread half
    // that skips the glDeleteBuffers half.
    using namespace cuda_api;
    if (cu_ctx_) cuCtxSetCurrent(cu_ctx_);
    for (auto &s : slots_) {
        if (s.resource) {
            cuGraphicsUnregisterResource(s.resource);
            s.resource = nullptr;
        }
        if (s.pbo) { p_glDeleteBuffers(1, &s.pbo); s.pbo = 0; }
        if (s.d_staging && q_) {
            sycl::free(s.d_staging, *q_);
            s.d_staging = nullptr;
        }
    }
    slots_.clear();
}

void CudaGLDisplayTarget::release_device_buffers() {
    // Render-thread half of a backend switch: drain the queue (the
    // render thread is allowed to block on device work; the UI thread
    // is not) and free the device-visible staging buffers while the OLD
    // queue still exists.  Then null q_ so the UI-thread destroy() never
    // dereferences the OLD queue pointer (which apply_backend_switch
    // frees right after this returns).  The CUDA-GL resources
    // (s.resource) and GL PBO (s.pbo) are LEFT for destroy() on the UI
    // thread — they need the GL context.
    std::lock_guard<std::mutex> lk(mtx_);
    drain();
    using namespace cuda_api;
    if (cu_ctx_) cuCtxSetCurrent(cu_ctx_);
    for (auto &s : slots_) {
        if (s.d_staging && q_) {
            sycl::free(s.d_staging, *q_);
            s.d_staging = nullptr;
        }
    }
    q_ = nullptr;
}

void CudaGLDisplayTarget::destroy() {
    std::lock_guard<std::mutex> lk(mtx_);
    // release_device_buffers() may have freed the device staging on the
    // render thread (backend-switch path); free_slots() handles both
    // legs — UK-side CUDA-GL resources + GL PBO on the UI thread here —
    // and is a no-op for slots already fully released.  When destroy()
    // is the only call (shutdown / non-switch teardown) the slots are
    // fully live and are drained + freed here.
    if (!slots_.empty()) { drain(); free_slots(); }
    if (tex_) { glDeleteTextures(1, &tex_); tex_ = 0; }
    if (ctx_retained_) {
        cuda_api::cuDevicePrimaryCtxRelease(cu_dev_);
        ctx_retained_ = false;
        cu_ctx_ = nullptr;
    }
}

void CudaGLDisplayTarget::resize(int w, int h) {
    // Caller guarantees the render thread is not mid-frame (pause_pipeline).
    std::size_t n;
    {
        // No drain() here: the render-thread apply_resize closure already
        // drained the queue (KernelRuntime::resize() → krt_.fill →
        // q->memset().wait(), which in the in-order queue completes only
        // after every prior command — including the in-flight render
        // kernel + tone-map that wrote into the OLD staging slots — has
        // finished).  By the time the UI thread sees resize_applied and
        // calls us, those staging writes are quiescent, so we can free
        // and reallocate without blocking the UI thread on q_->wait().
        std::lock_guard<std::mutex> lk(mtx_);
        n = slots_.size();
        free_slots();
        w_ = w;
        h_ = h;
    }
    if (tex_) {
        glBindTexture(GL_TEXTURE_2D, tex_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, nullptr);
    }
    std::lock_guard<std::mutex> lk(mtx_);
    cuda_api::cuCtxSetCurrent(cu_ctx_);
    if (!alloc_slots((int)n))
        spdlog::error("[cuda_gl] slot reallocation failed after resize");
}

void CudaGLDisplayTarget::drain() {
    if (q_) {
        try { q_->wait(); }
        catch (const std::exception &e) {
            spdlog::error("[cuda_gl] drain failed: {}", e.what());
        }
    }
}

int CudaGLDisplayTarget::acquire() {
    std::lock_guard<std::mutex> lk(mtx_);
    for (int i = 0; i < (int)slots_.size(); ++i) {
        if (slots_[i].state == FREE) {
            slots_[i].state = INFLIGHT;
            return i;
        }
    }
    return -1;
}

uint8_t *CudaGLDisplayTarget::staging_ptr(int slot) {
    if (slot < 0 || slot >= (int)slots_.size()) return nullptr;
    return slots_[slot].d_staging;
}

void CudaGLDisplayTarget::publish(int slot, const FrameInfo &info,
                                  sycl::event done) {
    if (slot < 0 || slot >= (int)slots_.size()) return;
    std::lock_guard<std::mutex> lk(mtx_);
    slots_[slot].info = info;
    slots_[slot].done = done;
    slots_[slot].state = PENDING;
}

bool CudaGLDisplayTarget::latest_ready(int &slot, FrameInfo &info) {
    std::lock_guard<std::mutex> lk(mtx_);
    for (auto &s : slots_) {
        if (s.state == PENDING &&
            s.done.get_info<sycl::info::event::command_execution_status>() ==
                sycl::info::event_command_status::complete)
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

GLuint CudaGLDisplayTarget::present(int slot) {
    using namespace cuda_api;
    if (slot < 0 || slot >= (int)slots_.size()) return tex_;
    std::lock_guard<std::mutex> lk(mtx_);
    Slot &s = slots_[slot];
    size_t bytes = (size_t)w_ * h_ * 4;

    cuCtxSetCurrent(cu_ctx_);
    bool copied = false;
    if (cu_ok(cuGraphicsMapResources(1, &s.resource, nullptr), "map")) {
        CUdeviceptr pbo_ptr = 0;
        size_t pbo_size = 0;
        copied =
            cu_ok(cuGraphicsResourceGetMappedPointer(&pbo_ptr, &pbo_size,
                                                     s.resource),
                  "get mapped pointer") &&
            cu_ok(cuMemcpyDtoD(pbo_ptr, (CUdeviceptr)(uintptr_t)s.d_staging,
                               bytes),
                  "DtoD") &&
            cu_ok(cuStreamSynchronize(nullptr), "sync");
        cuGraphicsUnmapResources(1, &s.resource, nullptr);
    }

    if (copied) {
        p_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, s.pbo);
        glBindTexture(GL_TEXTURE_2D, tex_);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w_, h_, GL_RGBA,
                        GL_UNSIGNED_BYTE, nullptr /* = PBO offset 0 */);
        p_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }

    uint64_t shown = s.info.frame_index;
    for (auto &other : slots_) {
        if (other.state == READY && other.info.frame_index <= shown)
            other.state = FREE;
    }
    s.state = FREE;
    return tex_;
}

bool CudaGLDisplayTarget::read_pixels(int slot, uint8_t *dst) {
    if (slot < 0 || slot >= (int)slots_.size() || !q_) return false;
    Slot &s = slots_[slot];
    if (!s.d_staging) return false;
    q_->memcpy(dst, s.d_staging, (size_t)w_ * h_ * 4).wait();
    return true;
}
