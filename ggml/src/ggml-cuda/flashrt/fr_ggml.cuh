// ggml-facing interface of the FlashRT NVFP4 path (Thor SM110).
// Included from ggml-cuda.cu under #ifdef GGML_CUDA_FLASHRT.
#pragma once

#include "../common.cuh"

// Called at the start of every backend graph evaluation; invalidates the
// per-evaluation quantized-activation cache.
void ggml_cuda_flashrt_begin_eval();

// True when this mul_mat should be routed to the FlashRT block-scaled NVFP4
// GEMM: NVFP4 weights, fp32 contiguous activations/dst, no batch dims,
// shapes within kernel alignment. cc must already be checked by the caller.
bool ggml_cuda_flashrt_should_use(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

// dst = src1 @ src0 via activation quantize + NVFP4 x NVFP4 tcgen05 GEMM.
// Weights are repacked into the CUTLASS wire format on first use and cached
// for the process lifetime.
void ggml_cuda_flashrt_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// True when the 4-node FFN subgraph {mul_mat gate, mul_mat up, GEGLU,
// mul_mat down} can run as one fused GeGLU GEMM (interleaved gate/up
// weights, FP4 intermediate) followed by the down GEMM.
bool ggml_cuda_flashrt_should_fuse_geglu(const ggml_tensor * gate_mm, const ggml_tensor * up_mm, const ggml_tensor * glu, const ggml_tensor * down_mm);

// Execute that fused FFN; writes the down mul_mat's dst.
void ggml_cuda_flashrt_geglu_ffn(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_mm, const ggml_tensor * up_mm, const ggml_tensor * glu, ggml_tensor * down_mm);

// pi0.5 adaLN modulate window: {rms_norm?, mul_mat mod, add bias, view,
// repeat, mul, add, view, repeat, add}. rms is null for the variant whose
// normalized input arrives from a previous graph split. Outputs written by
// the fused execution: the bias add (consumed by the gate view later) and
// the final add.
bool ggml_cuda_flashrt_should_fuse_ada(const ggml_tensor * rms, const ggml_tensor * mm, const ggml_tensor * bias_add,
                                       const ggml_tensor * view_scale, const ggml_tensor * repeat_scale,
                                       const ggml_tensor * mul, const ggml_tensor * add1,
                                       const ggml_tensor * view_shift, const ggml_tensor * repeat_shift,
                                       const ggml_tensor * add2);
void ggml_cuda_flashrt_ada_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mm,
                                ggml_tensor * bias_add, const ggml_tensor * view_scale, const ggml_tensor * mul,
                                const ggml_tensor * view_shift, ggml_tensor * add2);

// pi0.5 gated residual window: {view gate, repeat, mul, add}.
bool ggml_cuda_flashrt_should_fuse_gated_res(const ggml_tensor * view, const ggml_tensor * repeat,
                                             const ggml_tensor * mul, const ggml_tensor * add);
void ggml_cuda_flashrt_gated_residual(ggml_backend_cuda_context & ctx, const ggml_tensor * view,
                                      const ggml_tensor * mul, ggml_tensor * add);
