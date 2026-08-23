// Dispatch glue between ggml-cuda's mul_mat and the FlashRT NVFP4 kernels.
// Host-only logic; the device kernels live in the sibling fr_*.cu files.

#include "fr_ggml.cuh"
#include "fr_kernels.h"

#include <mutex>
#include <unordered_map>

namespace {

// Weights repacked into the CUTLASS wire format, keyed by the ggml tensor's
// device pointer. Weight tensors are immutable and live for the process
// lifetime, so entries are never evicted.
struct repacked_weight {
    void * packed = nullptr;
    void * sf     = nullptr;
};

std::unordered_map<const void *, repacked_weight> g_repack_cache;
std::mutex g_repack_mu;

// The repack allocates with cudaMalloc, which is illegal during CUDA graph
// capture. All weights are repacked during the first (uncaptured) warmup
// evaluation of each graph, so a cache miss while capturing indicates a bug.
const repacked_weight * get_repacked(const ggml_tensor * src0, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_repack_mu);

    auto it = g_repack_cache.find(src0->data);
    if (it != g_repack_cache.end()) {
        return &it->second;
    }

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];

    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        GGML_ABORT("flashrt: weight repack for %s requested during CUDA graph capture", src0->name);
    }

    repacked_weight w;
    CUDA_CHECK(cudaMalloc(&w.packed, ggml_cuda_flashrt::packed_bytes(N, K)));
    CUDA_CHECK(cudaMalloc(&w.sf,     ggml_cuda_flashrt::sf_bytes(N, K)));

    const int rc = ggml_cuda_flashrt::repack_weight(src0->data, w.packed, w.sf, (int) N, (int) K, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: weight repack failed for %s (N=%lld K=%lld rc=%d)", src0->name, (long long) N, (long long) K, rc);
    }

    auto res = g_repack_cache.emplace(src0->data, w);
    return &res.first->second;
}

} // namespace

bool ggml_cuda_flashrt_should_use(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    static const bool disabled = getenv("GGML_CUDA_FLASHRT_DISABLE") != nullptr;
    if (disabled) {
        return false;
    }
    if (src0->type != GGML_TYPE_NVFP4 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    if (K % 64 != 0 || N % 16 != 0) {
        return false;
    }
    return true;
}

void ggml_cuda_flashrt_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int K = (int) src0->ne[0];
    const int N = (int) src0->ne[1];
    const int M = (int) src1->ne[1];

    cudaStream_t stream = ctx.stream();

    // The persistent repack cache is keyed by the weight tensor's device
    // pointer, which is only sound when weight tensors are immortal (model
    // inference). Tools that create and free tensors at recycled addresses
    // (e.g. test-backend-ops) must set GGML_CUDA_FLASHRT_NO_CACHE=1 to
    // repack into scratch memory on every call instead.
    static const bool no_cache = getenv("GGML_CUDA_FLASHRT_NO_CACHE") != nullptr;

    ggml_cuda_pool_alloc<uint8_t> b_packed_scratch(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> b_sf_scratch    (ctx.pool());

    const void * b_packed = nullptr;
    const void * b_sf     = nullptr;
    if (no_cache) {
        b_packed_scratch.alloc(ggml_cuda_flashrt::packed_bytes(N, K));
        b_sf_scratch.alloc(ggml_cuda_flashrt::sf_bytes(N, K));
        const int rrc = ggml_cuda_flashrt::repack_weight(src0->data, b_packed_scratch.get(), b_sf_scratch.get(), N, K, stream);
        if (rrc != 0) {
            GGML_ABORT("flashrt: weight repack failed (N=%d K=%d rc=%d)", N, K, rrc);
        }
        b_packed = b_packed_scratch.get();
        b_sf     = b_sf_scratch.get();
    } else {
        const repacked_weight * w = get_repacked(src0, stream);
        b_packed = w->packed;
        b_sf     = w->sf;
    }

    ggml_cuda_pool_alloc<uint8_t> a_packed(ctx.pool(), ggml_cuda_flashrt::packed_bytes(M, K));
    ggml_cuda_pool_alloc<uint8_t> a_sf    (ctx.pool(), ggml_cuda_flashrt::sf_bytes(M, K));

    int rc = ggml_cuda_flashrt::quantize_act_f32(
        (const float *) src1->data, a_packed.get(), a_sf.get(), M, K, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: activation quantize failed (M=%d K=%d rc=%d)", M, K, rc);
    }

    // ggml's block_nvfp4 scale bytes carry standard e4m3 semantics: its
    // dequant table doubles the e2m1 values but its ue4m3 decode halves the
    // scale, so the two cancel and no alpha compensation is needed.
    const float alpha = 1.0f;
    const bool  widen = N >= 8192;

    rc = ggml_cuda_flashrt::gemm_f32out(
        a_packed.get(), a_sf.get(), b_packed, b_sf,
        (float *) dst->data, M, N, K, alpha, widen, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: gemm failed (M=%d N=%d K=%d rc=%d)", M, N, K, rc);
    }
}
