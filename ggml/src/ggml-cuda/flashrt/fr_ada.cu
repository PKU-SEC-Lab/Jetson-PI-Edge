// Fused adaLN kernels for the pi0.5 action expert (Thor).
//
// The ggml graph expresses each adaLN application as rms_norm + two
// broadcast repeats + mul + two adds (and the gated residual as repeat +
// mul + add), all on [M, C] tensors with M ~ 10. These kernels collapse
// each chain into one launch. The scale/shift/gate vectors are passed as
// direct pointers (the ggml view tensors' data pointers, which already
// include their byte offsets into the modulation vector).

#include "fr_kernels.h"

namespace ggml_cuda_flashrt {

namespace {

// out[m, c] = norm(x[m])[c] * (1 + scale[c]) + shift[c]
// where norm = rms-normalize when with_rms, identity otherwise.
template <bool with_rms>
__global__ void kernel_ada_rms(const float * __restrict__ x,
                               const float * __restrict__ scale,
                               const float * __restrict__ shift,
                               float * __restrict__ out,
                               int C, float eps) {
    const int m = blockIdx.x;
    const float * xr = x + (int64_t) m * C;
    float * orow = out + (int64_t) m * C;

    float inv_rms = 1.0f;
    if (with_rms) {
        float sumsq = 0.0f;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            const float v = xr[c];
            sumsq += v * v;
        }
        __shared__ float red[32];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sumsq += __shfl_xor_sync(0xffffffff, sumsq, off);
        }
        const int warp = threadIdx.x / 32;
        if (threadIdx.x % 32 == 0) {
            red[warp] = sumsq;
        }
        __syncthreads();
        if (warp == 0) {
            float v = (threadIdx.x < blockDim.x / 32) ? red[threadIdx.x] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, off);
            }
            if (threadIdx.x == 0) {
                red[0] = v;
            }
        }
        __syncthreads();
        inv_rms = rsqrtf(red[0] / C + eps);
    }

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float n = with_rms ? xr[c] * inv_rms : xr[c];
        orow[c] = n * (1.0f + scale[c]) + shift[c];
    }
}

// out[m, c] = residual[m, c] + branch[m, c] * gate[c]
__global__ void kernel_gated_residual(const float * __restrict__ residual,
                                      const float * __restrict__ branch,
                                      const float * __restrict__ gate,
                                      float * __restrict__ out,
                                      int C) {
    const int m = blockIdx.x;
    const int64_t off = (int64_t) m * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        out[off + c] = residual[off + c] + branch[off + c] * gate[c];
    }
}

// out[c] = a[c] + b[c]
__global__ void kernel_vec_add(const float * __restrict__ a,
                               const float * __restrict__ b,
                               float * __restrict__ out,
                               int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

} // namespace

int ada_rms_mod(const float * x, const float * scale, const float * shift,
                float * out, int M, int C, float eps, bool with_rms,
                cudaStream_t stream) {
    const int threads = 256;
    if (with_rms) {
        kernel_ada_rms<true><<<M, threads, 0, stream>>>(x, scale, shift, out, C, eps);
    } else {
        kernel_ada_rms<false><<<M, threads, 0, stream>>>(x, scale, shift, out, C, eps);
    }
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int gated_residual(const float * residual, const float * branch, const float * gate,
                   float * out, int M, int C, cudaStream_t stream) {
    kernel_gated_residual<<<M, 256, 0, stream>>>(residual, branch, gate, out, C);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int vec_add_f32(const float * a, const float * b, float * out, int n, cudaStream_t stream) {
    kernel_vec_add<<<(n + 255) / 256, 256, 0, stream>>>(a, b, out, n);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

} // namespace ggml_cuda_flashrt
