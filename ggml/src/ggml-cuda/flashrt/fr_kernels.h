// FlashRT NVFP4 kernels for Jetson AGX Thor (SM110).
//
// C-style entry points implemented in the fr_*.cu translation units, which
// are compiled separately for sm_110a with CUTLASS. This header must stay
// free of CUTLASS and ggml includes so it can be consumed from the regular
// ggml-cuda translation units.
//
// Wire format (NVFP4):
//   packed:  uint8 [rows, K/2], adjacent-pair nibbles (elem 2i low, 2i+1 high)
//   scales:  e4m3 (positive/ue4m3), one per 16 elements along K, stored in
//            the CUTLASS Sm1xx block-scaled atom layout for the GEMM shape
//
// ggml's block_nvfp4 scale bytes carry standard e4m3 semantics (its dequant
// table doubles the e2m1 values and its ue4m3 decode halves the scale, which
// cancel), so they pass through to the GEMM unmodified with alpha = 1.0.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ggml_cuda_flashrt {

static inline int64_t round_up_i64(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

// Scale-factor buffer size in bytes for a [rows, K] operand: the Sm1xx atom
// layout tiles rows in chunks of 128 and K/16 scale columns in chunks of 4.
static inline int64_t sf_bytes(int64_t rows, int64_t K) {
    return round_up_i64(rows, 128) * round_up_i64(K / 16, 4);
}

static inline int64_t packed_bytes(int64_t rows, int64_t K) {
    return rows * (K / 2);
}

// Block-scaled NVFP4 x NVFP4 GEMM, D = alpha * (A x B), fp32 output.
//   A_packed [M, K/2] row-major, B_packed [N, K/2] row-major (used as
//   column-major [K, N]), D fp32 [M, N] row-major.
// Requires K % 64 == 0, N % 16 == 0, D 16-byte aligned.
// widen selects a wide-N tile (use for N >= 8192).
// Returns 0 on success.
int gemm_f32out(const void * A_packed, const void * SFA,
                const void * B_packed, const void * SFB,
                float * D, int M, int N, int K,
                float alpha, bool widen, cudaStream_t stream);

// Quantize fp32 activations [M, K] row-major (contiguous) to NVFP4 packed
// [M, K/2] plus SFA scales in the atom layout for problem (M, x, K).
// Requires K % 16 == 0. Returns 0 on success.
int quantize_act_f32(const float * src, void * dst_packed, void * dst_sfa,
                     int M, int K, cudaStream_t stream);

// Repack a ggml GGML_TYPE_NVFP4 weight tensor [N rows, K] into the GEMM's
// B-side wire format: packed [N, K/2] with adjacent-pair nibbles plus SFB
// scales (ggml half-scale bytes, unmodified) in the atom layout.
// Requires K % 64 == 0. Returns 0 on success.
int repack_weight(const void * ggml_blocks, void * dst_packed, void * dst_sf,
                  int N, int K, cudaStream_t stream);

// Pairwise-interleave two ggml NVFP4 weight tensors (gate, up; each
// [n_ff rows, K]) into one B operand for the fused GeGLU GEMM: output row 2j
// is gate row j, row 2j+1 is up row j. dst_packed holds 2*n_ff rows; dst_sf
// is sized sf_bytes(2*n_ff, K).
int repack_weight_pair_interleaved(const void * gate_blocks, const void * up_blocks,
                                   void * dst_packed, void * dst_sf,
                                   int n_ff, int K, cudaStream_t stream);

} // namespace ggml_cuda_flashrt
