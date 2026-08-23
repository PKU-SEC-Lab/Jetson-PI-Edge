// Dispatch glue between ggml-cuda's mul_mat and the FlashRT NVFP4 kernels.
// Host-only logic; the device kernels live in the sibling fr_*.cu files.

#include "fr_ggml.cuh"
#include "fr_kernels.h"
#include "fr_geglu.cuh"

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

// Interleaved gate/up weight pairs for the fused GeGLU GEMM, keyed by the
// two tensors' device pointers (same immortality caveat as above).
struct pair_key {
    const void * gate;
    const void * up;
    bool operator==(const pair_key & o) const { return gate == o.gate && up == o.up; }
};
struct pair_key_hash {
    size_t operator()(const pair_key & k) const noexcept {
        return std::hash<const void *>()(k.gate) ^ (std::hash<const void *>()(k.up) << 1);
    }
};

std::unordered_map<pair_key, repacked_weight, pair_key_hash> g_pair_cache;

const repacked_weight * get_repacked_pair(const ggml_tensor * gate_w, const ggml_tensor * up_w, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(g_repack_mu);

    pair_key key{gate_w->data, up_w->data};
    auto it = g_pair_cache.find(key);
    if (it != g_pair_cache.end()) {
        return &it->second;
    }

    const int64_t K    = gate_w->ne[0];
    const int64_t n_ff = gate_w->ne[1];
    const int64_t N_il = 2 * n_ff;

    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        GGML_ABORT("flashrt: geglu pair repack for %s requested during CUDA graph capture", gate_w->name);
    }

    repacked_weight w;
    CUDA_CHECK(cudaMalloc(&w.packed, ggml_cuda_flashrt::packed_bytes(N_il, K)));
    CUDA_CHECK(cudaMalloc(&w.sf,     ggml_cuda_flashrt::sf_bytes(N_il, K)));

    const int rc = ggml_cuda_flashrt::repack_weight_pair_interleaved(
        gate_w->data, up_w->data, w.packed, w.sf, (int) n_ff, (int) K, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: geglu pair repack failed for %s (n_ff=%lld K=%lld rc=%d)",
                   gate_w->name, (long long) n_ff, (long long) K, rc);
    }

    auto res = g_pair_cache.emplace(key, w);
    return &res.first->second;
}

// Per-evaluation quantized-activation cache. Several ops consume the same
// fp32 activation tensor (q/k/v projections, the adaLN conditioning vector
// across all layers); quantizing it once per graph evaluation removes the
// duplicate quantize launches. Keys use the ggml tensor pointer (unique
// within one evaluation) plus an evaluation counter, so recycled device
// addresses across graphs can never alias. Slot buffers are grow-only and
// never freed, which keeps addresses stable for captured CUDA graphs; a
// replayed graph rewrites any slot before its baked consumers read it.
struct act_slot {
    const ggml_tensor * key = nullptr;
    uint64_t eval_id = 0;
    void * packed = nullptr;
    size_t packed_cap = 0;
    void * sf = nullptr;
    size_t sf_cap = 0;
};

act_slot g_act_slots[4];
int      g_act_slot_rr = 0;
uint64_t g_eval_id = 1;

// Returns cached (packed, sf) for src1 quantized as [M, K], quantizing on a
// miss. Returns false when the cache cannot be used (slot growth needed
// while capturing a CUDA graph); the caller must quantize into pool memory.
bool get_quantized_act(const ggml_tensor * src1, int M, int K,
                       const void ** out_packed, const void ** out_sf,
                       cudaStream_t stream) {
    for (auto & s : g_act_slots) {
        if (s.key == src1 && s.eval_id == g_eval_id) {
            *out_packed = s.packed;
            *out_sf     = s.sf;
            return true;
        }
    }

    act_slot & s = g_act_slots[g_act_slot_rr];
    const size_t need_packed = (size_t) ggml_cuda_flashrt::packed_bytes(M, K);
    const size_t need_sf     = (size_t) ggml_cuda_flashrt::sf_bytes(M, K);

    if (need_packed > s.packed_cap || need_sf > s.sf_cap) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cap);
        if (cap != cudaStreamCaptureStatusNone) {
            return false;
        }
        if (need_packed > s.packed_cap) {
            if (s.packed != nullptr) { cudaFree(s.packed); }
            CUDA_CHECK(cudaMalloc(&s.packed, need_packed));
            s.packed_cap = need_packed;
        }
        if (need_sf > s.sf_cap) {
            if (s.sf != nullptr) { cudaFree(s.sf); }
            CUDA_CHECK(cudaMalloc(&s.sf, need_sf));
            s.sf_cap = need_sf;
        }
    }
    g_act_slot_rr = (g_act_slot_rr + 1) % 4;

    const int rc = ggml_cuda_flashrt::quantize_act_f32(
        (const float *) src1->data, s.packed, s.sf, M, K, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: activation quantize failed (M=%d K=%d rc=%d)", M, K, rc);
    }
    s.key     = src1;
    s.eval_id = g_eval_id;
    *out_packed = s.packed;
    *out_sf     = s.sf;
    return true;
}

// Grow-only device buffer for the never-written D of the no-D-store GeGLU
// variants (the host-side TMA descriptor still needs a valid allocation).
void * get_dummy_d(size_t bytes) {
    static void * buf = nullptr;
    static size_t cap = 0;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    if (bytes > cap) {
        if (buf != nullptr) {
            cudaFree(buf);
        }
        CUDA_CHECK(cudaMalloc(&buf, bytes));
        cap = bytes;
    }
    return buf;
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
    // Batched src1 with unbatched (broadcast) weights folds into a single
    // GEMM over all rows because src1/dst are fully contiguous.
    if (src0->ne[2] != 1 || src0->ne[3] != 1) {
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
    const int M = (int) ggml_nrows(src1); // batch dims fold into rows (contiguous, broadcast weights)

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

    ggml_cuda_pool_alloc<uint8_t> a_packed(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> a_sf    (ctx.pool());

    const void * q_packed = nullptr;
    const void * q_sf     = nullptr;
    if (!get_quantized_act(src1, M, K, &q_packed, &q_sf, stream)) {
        a_packed.alloc(ggml_cuda_flashrt::packed_bytes(M, K));
        a_sf.alloc(ggml_cuda_flashrt::sf_bytes(M, K));
        const int qrc = ggml_cuda_flashrt::quantize_act_f32(
            (const float *) src1->data, a_packed.get(), a_sf.get(), M, K, stream);
        if (qrc != 0) {
            GGML_ABORT("flashrt: activation quantize failed (M=%d K=%d rc=%d)", M, K, qrc);
        }
        q_packed = a_packed.get();
        q_sf     = a_sf.get();
    }

    // ggml's block_nvfp4 scale bytes carry standard e4m3 semantics: its
    // dequant table doubles the e2m1 values but its ue4m3 decode halves the
    // scale, so the two cancel and no alpha compensation is needed.
    const float alpha = 1.0f;
    // The 128x128x256 tile beats the wide-N 128x256x128 tile on Thor for
    // every production shape measured (including N=16384 prefill FFN).
    const bool  widen = false;

    const int rc = ggml_cuda_flashrt::gemm_f32out(
        q_packed, q_sf, b_packed, b_sf,
        (float *) dst->data, M, N, K, alpha, widen, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: gemm failed (M=%d N=%d K=%d rc=%d)", M, N, K, rc);
    }
}

bool ggml_cuda_flashrt_should_fuse_ada(const ggml_tensor * rms, const ggml_tensor * mm, const ggml_tensor * bias_add,
                                       const ggml_tensor * view_scale, const ggml_tensor * repeat_scale,
                                       const ggml_tensor * mul, const ggml_tensor * add1,
                                       const ggml_tensor * view_shift, const ggml_tensor * repeat_shift,
                                       const ggml_tensor * add2) {
    const ggml_tensor * x = rms != nullptr ? rms->src[0] : mul->src[0];
    const ggml_tensor * normed = rms != nullptr ? rms : x;
    const int64_t C = x->ne[0];
    const int64_t M = x->ne[1];

    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }
    // modulation projection: [3C, 1] from the shared conditioning vector
    if (!ggml_cuda_flashrt_should_use(mm->src[0], mm->src[1], mm) ||
        mm->ne[0] != 3 * C || mm->ne[1] != 1) {
        return false;
    }
    const ggml_tensor * bias = bias_add->src[1];
    if (bias_add->src[0] != mm || bias->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(bias) || bias->ne[0] != 3 * C || !ggml_is_contiguous(bias_add)) {
        return false;
    }
    // scale = mod[0:C], shift = mod[C:2C]
    if (view_scale->src[0] != bias_add || view_scale->ne[0] != C || view_scale->ne[1] != 1 ||
        view_scale->view_offs != 0) {
        return false;
    }
    if (view_shift->src[0] != bias_add || view_shift->ne[0] != C || view_shift->ne[1] != 1 ||
        view_shift->view_offs != (size_t) C * sizeof(float)) {
        return false;
    }
    if (repeat_scale->src[0] != view_scale || repeat_shift->src[0] != view_shift ||
        repeat_scale->ne[0] != C || repeat_scale->ne[1] != M) {
        return false;
    }
    // t = normed * scale; t2 = normed + t; out = t2 + shift
    if (mul->src[0] != normed || mul->src[1] != repeat_scale ||
        add1->src[0] != normed || add1->src[1] != mul ||
        add2->src[0] != add1 || add2->src[1] != repeat_shift ||
        !ggml_is_contiguous(add2) || add2->ne[0] != C || add2->ne[1] != M) {
        return false;
    }
    return true;
}

void ggml_cuda_flashrt_ada_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mm,
                                ggml_tensor * bias_add, const ggml_tensor * view_scale, const ggml_tensor * mul,
                                const ggml_tensor * view_shift, ggml_tensor * add2) {
    const ggml_tensor * x    = rms != nullptr ? rms->src[0] : mul->src[0];
    const ggml_tensor * cond = mm->src[1];

    const int C = (int) x->ne[0];
    const int M = (int) x->ne[1];
    const int K = (int) mm->src[0]->ne[0];
    const int N = (int) mm->src[0]->ne[1]; // 3C

    cudaStream_t stream = ctx.stream();

    const repacked_weight * w = get_repacked(mm->src[0], stream);

    ggml_cuda_pool_alloc<uint8_t> c_packed(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> c_sf    (ctx.pool());
    ggml_cuda_pool_alloc<float>   mod_raw (ctx.pool(), N);

    const void * q_packed = nullptr;
    const void * q_sf     = nullptr;
    int rc = 0;
    if (!get_quantized_act(cond, 1, K, &q_packed, &q_sf, stream)) {
        c_packed.alloc(ggml_cuda_flashrt::packed_bytes(1, K));
        c_sf.alloc(ggml_cuda_flashrt::sf_bytes(1, K));
        rc = ggml_cuda_flashrt::quantize_act_f32((const float *) cond->data, c_packed.get(), c_sf.get(), 1, K, stream);
        q_packed = c_packed.get();
        q_sf     = c_sf.get();
    }
    if (rc == 0) {
        rc = ggml_cuda_flashrt::gemm_f32out(q_packed, q_sf, w->packed, w->sf,
                                            mod_raw.get(), 1, N, K, 1.0f, false, stream);
    }
    if (rc == 0) {
        // materialize the biased modulation vector: the gate view reads it later
        rc = ggml_cuda_flashrt::vec_add_f32(mod_raw.get(), (const float *) bias_add->src[1]->data,
                                            (float *) bias_add->data, N, stream);
    }
    if (rc == 0) {
        const float eps = rms != nullptr ? ggml_get_op_params_f32(rms, 0) : 0.0f;
        rc = ggml_cuda_flashrt::ada_rms_mod((const float *) x->data,
                                            (const float *) view_scale->data,
                                            (const float *) view_shift->data,
                                            (float *) add2->data, M, C, eps, rms != nullptr, stream);
    }
    if (rc != 0) {
        GGML_ABORT("flashrt: fused adaLN failed (M=%d C=%d rc=%d)", M, C, rc);
    }
}

bool ggml_cuda_flashrt_should_fuse_gated_res(const ggml_tensor * view, const ggml_tensor * repeat,
                                             const ggml_tensor * mul, const ggml_tensor * add) {
    const int64_t C = view->ne[0];
    const int64_t M = mul->ne[1];
    if (view->type != GGML_TYPE_F32 || view->ne[1] != 1 ||
        view->src[0] == nullptr || view->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    if (repeat->src[0] != view || repeat->ne[0] != C || repeat->ne[1] != M) {
        return false;
    }
    const ggml_tensor * branch = mul->src[0];
    if (mul->src[1] != repeat || branch->type != GGML_TYPE_F32 || !ggml_is_contiguous(branch) ||
        branch->ne[0] != C || branch->ne[1] != M) {
        return false;
    }
    const ggml_tensor * residual = add->src[0] == mul ? add->src[1] : add->src[0];
    if ((add->src[0] != mul && add->src[1] != mul) || residual == mul ||
        residual->type != GGML_TYPE_F32 || !ggml_is_contiguous(residual) ||
        residual->ne[0] != C || residual->ne[1] != M ||
        !ggml_is_contiguous(add) || add->ne[0] != C || add->ne[1] != M) {
        return false;
    }
    return true;
}

void ggml_cuda_flashrt_gated_residual(ggml_backend_cuda_context & ctx, const ggml_tensor * view,
                                      const ggml_tensor * mul, ggml_tensor * add) {
    const ggml_tensor * branch   = mul->src[0];
    const ggml_tensor * residual = add->src[0] == mul ? add->src[1] : add->src[0];
    const int C = (int) view->ne[0];
    const int M = (int) mul->ne[1];

    const int rc = ggml_cuda_flashrt::gated_residual(
        (const float *) residual->data, (const float *) branch->data,
        (const float *) view->data, (float *) add->data, M, C, ctx.stream());
    if (rc != 0) {
        GGML_ABORT("flashrt: fused gated residual failed (M=%d C=%d rc=%d)", M, C, rc);
    }
}

bool ggml_cuda_flashrt_should_fuse_geglu(const ggml_tensor * gate_mm, const ggml_tensor * up_mm, const ggml_tensor * glu, const ggml_tensor * down_mm) {
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_GEGLU) {
        return false;
    }
    // both projections share the same input and shape
    if (gate_mm->src[1] != up_mm->src[1]) {
        return false;
    }
    const ggml_tensor * gate_w = gate_mm->src[0];
    const ggml_tensor * up_w   = up_mm->src[0];
    const ggml_tensor * down_w = down_mm->src[0];
    if (gate_w->ne[0] != up_w->ne[0] || gate_w->ne[1] != up_w->ne[1]) {
        return false;
    }
    if (!ggml_cuda_flashrt_should_use(gate_w, gate_mm->src[1], gate_mm) ||
        !ggml_cuda_flashrt_should_use(up_w,   up_mm->src[1],   up_mm)   ||
        !ggml_cuda_flashrt_should_use(down_w, glu,             down_mm)) {
        return false;
    }
    // the down projection must consume the GLU output with K = n_ff
    if (down_mm->src[1] != glu || down_w->ne[0] != gate_w->ne[1]) {
        return false;
    }
    return true;
}

void ggml_cuda_flashrt_geglu_ffn(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_mm, const ggml_tensor * up_mm, const ggml_tensor * glu, ggml_tensor * down_mm) {
    GGML_UNUSED(glu);

    const ggml_tensor * src1   = gate_mm->src[1];
    const ggml_tensor * gate_w = gate_mm->src[0];
    const ggml_tensor * down_w = down_mm->src[0];

    const int K    = (int) gate_w->ne[0];
    const int n_ff = (int) gate_w->ne[1];
    const int N_il = 2 * n_ff;
    const int M    = (int) ggml_nrows(src1);
    const int N_out = (int) down_w->ne[1];

    cudaStream_t stream = ctx.stream();

    const repacked_weight * w_il   = get_repacked_pair(gate_w, up_mm->src[0], stream);
    const repacked_weight * w_down = get_repacked(down_w, stream);

    ggml_cuda_pool_alloc<uint8_t> a_packed(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> a_sf    (ctx.pool());

    const void * q_packed = nullptr;
    const void * q_sf     = nullptr;
    int rc = 0;
    if (!get_quantized_act(src1, M, K, &q_packed, &q_sf, stream)) {
        a_packed.alloc(ggml_cuda_flashrt::packed_bytes(M, K));
        a_sf.alloc(ggml_cuda_flashrt::sf_bytes(M, K));
        rc = ggml_cuda_flashrt::quantize_act_f32(
            (const float *) src1->data, a_packed.get(), a_sf.get(), M, K, stream);
        if (rc != 0) {
            GGML_ABORT("flashrt: geglu activation quantize failed (M=%d K=%d rc=%d)", M, K, rc);
        }
        q_packed = a_packed.get();
        q_sf     = a_sf.get();
    }

    ggml_cuda_pool_alloc<uint8_t> compact_packed(ctx.pool(), ggml_cuda_flashrt::packed_bytes(M, n_ff));
    ggml_cuda_pool_alloc<uint8_t> compact_sfa   (ctx.pool(), ggml_cuda_flashrt::sf_bytes(M, n_ff));
    void * dummy_d = get_dummy_d(ggml_cuda_flashrt::packed_bytes(M, N_il));

    // skinny-M tile for decode-sized batches, default tile otherwise
    if (M < 128) {
        rc = flash_rt::fp4::cutlass_fp4_gemm_geglu_il_hw_nod_v10(
            q_packed, q_sf, w_il->packed, w_il->sf,
            dummy_d, compact_packed.get(), compact_sfa.get(), M, N_il, K, stream);
    } else {
        rc = flash_rt::fp4::cutlass_fp4_gemm_geglu_il_hw_nod(
            q_packed, q_sf, w_il->packed, w_il->sf,
            dummy_d, compact_packed.get(), compact_sfa.get(), M, N_il, K, stream);
    }
    if (rc != 0) {
        GGML_ABORT("flashrt: geglu gemm failed (M=%d N_il=%d K=%d rc=%d)", M, N_il, K, rc);
    }

    rc = ggml_cuda_flashrt::gemm_f32out(
        compact_packed.get(), compact_sfa.get(), w_down->packed, w_down->sf,
        (float *) down_mm->data, M, N_out, n_ff, /*alpha=*/1.0f, /*widen=*/false, stream);
    if (rc != 0) {
        GGML_ABORT("flashrt: geglu down gemm failed (M=%d N=%d K=%d rc=%d)", M, N_out, n_ff, rc);
    }
}

void ggml_cuda_flashrt_begin_eval() {
    g_eval_id++;
}
