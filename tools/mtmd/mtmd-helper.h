#ifndef MTMD_HELPER_H
#define MTMD_HELPER_H

#include "ggml.h"
#include "llama.h"
#include "mtmd.h"
#include "pi-model.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// libmtmd helper functions
//
// Please note that these helpers are not guaranteed to be stable.
// BREAKING CHANGES are expected.
//

// Set callback for all future logging events.
// If this is not called, or NULL is supplied, everything is output on stderr.
// Note: this also call mtmd_log_set() internally
MTMD_API void mtmd_helper_log_set(ggml_log_callback log_callback, void * user_data);

// helper function to construct a mtmd_bitmap from a file
// it calls mtmd_helper_bitmap_init_from_buf() internally
// returns nullptr on failure
// this function is thread-safe
MTMD_API mtmd_bitmap * mtmd_helper_bitmap_init_from_file(mtmd_context * ctx, const char * fname);

// helper function to construct a mtmd_bitmap from a buffer containing a file
// supported formats:
//     image: formats supported by stb_image: jpg, png, bmp, gif, etc.
//     audio: formats supported by miniaudio: wav, mp3, flac
// note: audio files will be auto-detected based on magic bytes
// returns nullptr on failure
// this function is thread-safe
MTMD_API mtmd_bitmap * mtmd_helper_bitmap_init_from_buf(mtmd_context * ctx, const unsigned char * buf, size_t len);

// helper to count the total number of tokens from a list of chunks, useful to keep track of KV cache
MTMD_API size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks);

// helper to count the total position of tokens from a list of chunks, useful to keep track of n_past
// normally, n_pos is equal to n_tokens, but for M-RoPE it is different
MTMD_API llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks);

// helper function that automatically:
// 1. run llama_decode() on text chunks
// 2. run mtmd_encode() on image chunks, then mtmd_get_output_embd() and then llama_decode()
// if any of the mtmd_encode() or llama_decode() calls return non-zero, stop and forward the error
// otherwise, returns 0 on success
// this function is NOT thread-safe
MTMD_API int32_t mtmd_helper_eval_chunks(mtmd_context * ctx,
                                         struct llama_context * lctx,
                                         const mtmd_input_chunks * chunks,
                                         llama_pos n_past,
                                         llama_seq_id seq_id,
                                         int32_t n_batch,
                                         bool logits_last,
                                         llama_pos * new_n_past);

typedef struct mtmd_pi0_result {
    double vit_ms;
    double encode_ms;
    double decode_ms;
    double total_ms;
    double batch_build_ms;
    double output_extract_ms;
    double batch_free_ms;
    bool has_vit_ms;
    bool has_encode_ms;
    bool has_decode_ms;
    bool has_total_ms;
    bool has_batch_build_ms;
    bool has_output_extract_ms;
    bool has_batch_free_ms;
    int32_t state_dim;
    bool has_state;
    float * state_data;
    int32_t action_dim;
    int32_t action_steps;
    bool has_action;
    float * action_data;
    bool has_action_final;
    float * action_final_data;
    // VIT breakdown (from clip_get_last_pi0_vit_perf, LLAMA_PI0_VIT_PERF=1)
    double vit_preprocess_ms;
    double vit_graph_build_alloc_ms;
    double vit_set_inputs_ms;
    double vit_graph_compute_ms;
    double vit_output_get_ms;
    bool   vit_graph_reused;
    bool   has_vit_breakdown;
} mtmd_pi0_result;

MTMD_API void mtmd_pi0_result_free(mtmd_pi0_result * result);

// Copy last VIT perf from clip (after mtmd_encode_chunk / clip_image_batch_encode).
MTMD_API void mtmd_pi0_result_apply_vit_perf(struct mtmd_context * ctx, mtmd_pi0_result * result);

MTMD_API int32_t mtmd_helper_eval_chunks_pi0(mtmd_context * ctx,
                                        struct llama_context * lctx,
                                        const mtmd_input_chunks * chunks,
                                        llama_pos n_past,
                                        llama_seq_id seq_id,
                                        int32_t n_batch,
                                        bool logits_last,
                                        llama_pos * new_n_past,
                                        mtmd_pi0_result * pi0_result);

typedef struct mtmd_pi0_context mtmd_pi0_context;

MTMD_API int32_t mtmd_helper_prepare_chunks_pi0(
                                        mtmd_context * ctx,
                                        struct llama_context * lctx,
                                        const mtmd_input_chunks * chunks,
                                        llama_pos n_past,
                                        llama_seq_id seq_id,
                                        int32_t n_batch,
                                        bool logits_last,
                                        llama_pos * new_n_past,
                                        mtmd_pi0_context ** out_context);

MTMD_API int32_t mtmd_helper_prepare_chunks_pi0_for_model(
                                        mtmd_context * ctx,
                                        struct llama_context * lctx,
                                        const mtmd_input_chunks * chunks,
                                        pi_model_kind model_kind,
                                        llama_pos n_past,
                                        llama_seq_id seq_id,
                                        int32_t n_batch,
                                        bool logits_last,
                                        llama_pos * new_n_past,
                                        mtmd_pi0_context ** out_context);

MTMD_API int32_t mtmd_helper_decode_pi0(struct llama_context * lctx,
                                        mtmd_pi0_context * context);
MTMD_API void mtmd_helper_free_pi0_context(mtmd_pi0_context * context);

// works like mtmd_helper_eval_chunks(), but only for a single chunk
// this function is NOT thread-safe
MTMD_API int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx,
                                               struct llama_context * lctx,
                                               const mtmd_input_chunk * chunk,
                                               llama_pos n_past,
                                               llama_seq_id seq_id,
                                               int32_t n_batch,
                                               bool logits_last,
                                               llama_pos * new_n_past);

// helper function to decode an image whose embeddings have already been calculated
// this helper will handle batching and pre/post decoding setup (for ex. gemma 3 requires non-causal attention)
// ret 0 on success, -1 on chunk not being a valid image chunk, 1 on decode failure
MTMD_API int32_t mtmd_helper_decode_image_chunk(mtmd_context * ctx,
                                                struct llama_context * lctx,
                                                const mtmd_input_chunk * chunk,
                                                float * encoded_embd,
                                                llama_pos n_past,
                                                llama_seq_id seq_id,
                                                int32_t n_batch,
                                                llama_pos * new_n_past);

#ifdef __cplusplus
} // extern "C"
#endif

//
// C++ wrappers
//

#endif
