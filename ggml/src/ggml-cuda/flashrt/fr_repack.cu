// One-time repack of a ggml GGML_TYPE_NVFP4 weight tensor into the CUTLASS
// block-scaled GEMM's B-side wire format.
//
// ggml block_nvfp4 (36 bytes / 64 elements):
//   uint8 d[4]   - one e4m3 scale per 16-element sub-block; standard e4m3
//                  semantics (ggml's doubled dequant table and halved ue4m3
//                  decode cancel), passed through unmodified
//   uint8 qs[32] - e2m1 codes, sub-block s at qs[s*8..s*8+7], byte j holding
//                  elem[j] in the low nibble and elem[8+j] in the high nibble
//
// Output: packed uint8 [N, K/2] with adjacent-pair nibbles (elem 2i low,
// elem 2i+1 high) and the scale bytes at the Sm1xx atom-layout offsets.
// One thread handles one 16-element sub-block.

#include "fr_kernels.h"

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

namespace ggml_cuda_flashrt {

namespace {

using CfgVec = cutlass::detail::Sm1xxBlockScaledConfig<16>;

constexpr int GGML_NVFP4_BLOCK_BYTES = 36; // 4 scale bytes + 32 data bytes

template <class LayoutSF>
__global__ void kernel_repack(
        const uint8_t * __restrict__ src,  // ggml block_nvfp4 stream
        uint2 * __restrict__ dst_packed,   // [N, K/2] bytes as uint2 per sub-block
        uint8_t * __restrict__ dst_sf,
        LayoutSF layout,
        int N, int K16) {                  // K16 = K / 16 sub-blocks per row
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= N || t >= K16) return;

    const int blk = t >> 2;      // 64-element ggml block
    const int sub = t & 3;       // 16-element sub-block within it

    const uint8_t * b = src + (static_cast<int64_t>(row) * (K16 >> 2) + blk) * GGML_NVFP4_BLOCK_BYTES;
    const uint8_t scale = b[sub];
    const uint8_t * qs = b + 4 + sub * 8;

    uint2 out;
    uint8_t * ob = reinterpret_cast<uint8_t *>(&out);
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        // elements 2p, 2p+1 live in the low nibbles of qs[2p], qs[2p+1]
        ob[p]     = static_cast<uint8_t>((qs[2 * p] & 0x0F)  | ((qs[2 * p + 1] & 0x0F) << 4));
        // elements 8+2p, 8+2p+1 live in the high nibbles of the same bytes
        ob[p + 4] = static_cast<uint8_t>((qs[2 * p] >> 4)    | ((qs[2 * p + 1] & 0xF0)));
    }

    dst_packed[static_cast<int64_t>(row) * K16 + t] = out;
    dst_sf[layout(row, t * 16, 0)] = scale;
}

} // namespace

namespace {

// Pairwise-interleaved variant for the fused GeGLU GEMM: output row 2j is
// gate row j, row 2j+1 is up row j (N_il = 2 * n_ff rows total).
template <class LayoutSF>
__global__ void kernel_repack_pair(
        const uint8_t * __restrict__ gate,
        const uint8_t * __restrict__ up,
        uint2 * __restrict__ dst_packed,
        uint8_t * __restrict__ dst_sf,
        LayoutSF layout,
        int n_ff, int K16) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;             // row within gate/up
    const int which = blockIdx.z;           // 0 = gate, 1 = up
    if (row >= n_ff || t >= K16) return;

    const int blk = t >> 2;
    const int sub = t & 3;
    const uint8_t * src = which ? up : gate;
    const uint8_t * b = src + (static_cast<int64_t>(row) * (K16 >> 2) + blk) * GGML_NVFP4_BLOCK_BYTES;
    const uint8_t scale = b[sub];
    const uint8_t * qs = b + 4 + sub * 8;

    uint2 out;
    uint8_t * ob = reinterpret_cast<uint8_t *>(&out);
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        ob[p]     = static_cast<uint8_t>((qs[2 * p] & 0x0F) | ((qs[2 * p + 1] & 0x0F) << 4));
        ob[p + 4] = static_cast<uint8_t>((qs[2 * p] >> 4)   | ((qs[2 * p + 1] & 0xF0)));
    }

    const int out_row = 2 * row + which;
    dst_packed[static_cast<int64_t>(out_row) * K16 + t] = out;
    dst_sf[layout(out_row, t * 16, 0)] = scale;
}

} // namespace

int repack_weight_pair_interleaved(const void * gate_blocks, const void * up_blocks,
                                   void * dst_packed, void * dst_sf,
                                   int n_ff, int K, cudaStream_t stream) {
    if (K % 64 != 0) return -1;
    if (reinterpret_cast<uintptr_t>(dst_packed) & 7) return -1;

    const int K16  = K / 16;
    const int N_il = 2 * n_ff;
    const int threads = 128;
    dim3 grid((K16 + threads - 1) / threads, n_ff, 2);

    auto shape = cute::make_shape(1, N_il, K, 1);
    auto layout = CfgVec::tile_atom_to_shape_SFB(shape);
    if (static_cast<int64_t>(cute::cosize(layout)) > sf_bytes(N_il, K)) {
        return -3;
    }

    kernel_repack_pair<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(gate_blocks),
        reinterpret_cast<const uint8_t *>(up_blocks),
        reinterpret_cast<uint2 *>(dst_packed),
        reinterpret_cast<uint8_t *>(dst_sf),
        layout, n_ff, K16);

    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int repack_weight(const void * ggml_blocks, void * dst_packed, void * dst_sf,
                  int N, int K, cudaStream_t stream) {
    if (K % 64 != 0) return -1;
    if (reinterpret_cast<uintptr_t>(dst_packed) & 7) return -1;

    const int K16 = K / 16;
    const int threads = 128;
    dim3 grid((K16 + threads - 1) / threads, N);

    // SFB layout for a [*, N, K] problem; independent of M.
    auto shape = cute::make_shape(1, N, K, 1);
    auto layout = CfgVec::tile_atom_to_shape_SFB(shape);

    // The atom layout may address up to the padded (row, k) extents; the
    // caller allocates sf_bytes(N, K) which must cover the layout codomain.
    if (static_cast<int64_t>(cute::cosize(layout)) > sf_bytes(N, K)) {
        return -3;
    }

    kernel_repack<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(ggml_blocks),
        reinterpret_cast<uint2 *>(dst_packed),
        reinterpret_cast<uint8_t *>(dst_sf),
        layout, N, K16);

    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

} // namespace ggml_cuda_flashrt
