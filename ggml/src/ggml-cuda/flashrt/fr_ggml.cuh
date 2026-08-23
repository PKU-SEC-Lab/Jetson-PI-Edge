// ggml-facing interface of the FlashRT NVFP4 path (Thor SM110).
// Included from ggml-cuda.cu under #ifdef GGML_CUDA_FLASHRT.
#pragma once

#include "../common.cuh"

// True when this mul_mat should be routed to the FlashRT block-scaled NVFP4
// GEMM: NVFP4 weights, fp32 contiguous activations/dst, no batch dims,
// shapes within kernel alignment. cc must already be checked by the caller.
bool ggml_cuda_flashrt_should_use(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// dst = src1 @ src0 via activation quantize + NVFP4 x NVFP4 tcgen05 GEMM.
// Weights are repacked into the CUTLASS wire format on first use and cached
// for the process lifetime.
void ggml_cuda_flashrt_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
