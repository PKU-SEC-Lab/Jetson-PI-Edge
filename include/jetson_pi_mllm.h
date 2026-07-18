#ifndef JETSON_PI_MLLM_H
#define JETSON_PI_MLLM_H

//
// jetson_pi_mllm - narrow multimodal LLM C API.
//
// Wraps image encoding (mtmd) + text generation (llama.cpp sample loop) behind
// one opaque handle so an embedding host (e.g. FlashRT) can drive a
// vision-language model without touching llama.cpp/mtmd/GGML internals.
//
// The host hands raw RGB images + one raw prompt (chat template applied by the
// caller) and receives one generated text blob. Each infer call clears KV
// state and runs an independent first-turn completion (no multi-turn state).
//
// This header depends only on <stddef.h>/<stdint.h>. No llama.h / mtmd.h /
// ggml.h types are exposed here.
//

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jetson_pi_mllm jetson_pi_mllm;

typedef struct jetson_pi_mllm_config {
    uint32_t struct_size;

    const char * model_path;   /* GGUF VLM (text + vision projector weights) */
    const char * mmproj_path;  /* VIT mmproj GGUF (vision encoder)            */
    const char * backend;      /* "cpu" | "cuda" | "vulkan" | "opencl" | "sycl" */

    uint32_t n_ctx;            /* KV context size; 0 = 4096 fallback          */
    int32_t  n_threads;        /* CPU threads; 0 = hardware_concurrency       */

    float    temp;             /* sampler temperature (<=0 = greedy)          */
    int32_t  top_k;            /* 0 = disabled                                */
    float    top_p;            /* 0 = disabled                                */
    uint32_t seed;             /* RNG seed; 0 = LLAMA_DEFAULT_SEED             */
    uint32_t max_tokens;       /* cap on generated tokens; 0 = 512            */
} jetson_pi_mllm_config;

enum jetson_pi_mllm_status {
    JETSON_PI_MLLM_OK               =  0,
    JETSON_PI_MLLM_INVALID          = -1, /* bad args / load failed          */
    JETSON_PI_MLLM_BUFFER_TOO_SMALL = -2, /* out_buf too small               */
    JETSON_PI_MLLM_INFER_FAILED     = -3, /* tokenize / encode / decode fail */
    JETSON_PI_MLLM_BAD_STATE        = -4, /* prefill/decode order violation  */
};

// Load model + mmproj + create context + sampler chain. On success *out is a
// handle the caller owns (release via jetson_pi_mllm_close). On failure *out
// is set to NULL; the failure is described by jetson_pi_mllm_open_error().
int32_t jetson_pi_mllm_open(const jetson_pi_mllm_config * config,
                            jetson_pi_mllm ** out);

// Release the handle. Safe to call with NULL.
void jetson_pi_mllm_close(jetson_pi_mllm * handle);

// Always returns a non-null const char * ("" when no error is recorded).
// Pointer validity: until the next jetson_pi_mllm_* call on the same handle /
// same thread.
const char * jetson_pi_mllm_last_error(const jetson_pi_mllm * handle);

// Description of the most recent open failure on the calling thread.
// Always non-null (""). Pointer validity: until the next open call on same
// thread.
const char * jetson_pi_mllm_open_error(void);

// Explicit multimodal session API. prefill always starts a fresh session,
// encodes the supplied images, evaluates image+text chunks, and leaves logits
// for the first generated token ready. decode_step samples exactly one token;
// non-EOG tokens are accepted and decoded into KV. The host may interrupt by
// stopping calls; additional decode_step calls after max_tokens return
// JETSON_PI_MLLM_BAD_STATE until reset/prefill starts a new session.
int32_t jetson_pi_mllm_reset(jetson_pi_mllm * handle);
int32_t jetson_pi_mllm_prefill(jetson_pi_mllm * handle,
                               const uint8_t * const * images_rgb,
                               uint32_t n_images,
                               uint32_t image_height, uint32_t image_width,
                               const char * prompt, size_t prompt_len);
int32_t jetson_pi_mllm_decode_step(jetson_pi_mllm * handle,
                                   int32_t * token_id, int32_t * is_eog);
int32_t jetson_pi_mllm_get_logits(jetson_pi_mllm * handle,
                                  float * logits_out, size_t logits_capacity,
                                  size_t * logits_written);
int32_t jetson_pi_mllm_token_to_piece(jetson_pi_mllm * handle,
                                      int32_t token_id,
                                      char * piece_out, size_t piece_capacity,
                                      size_t * piece_written);

// One multimodal completion. Clears KV state, encodes images via mtmd,
// tokenizes the prompt (with media markers injected), evaluates the
// image+text chunks, then runs the decode/sample loop up to max_tokens or
// EOG. Writes the generated text (UTF-8, no NUL) into out_buf.
//
//   images_rgb    array of n_images RGB row-major buffers; each buffer is
//                 image_height * image_width * 3 bytes (the engine does not
//                 know per-image dimensions - all images must share the
//                 config's image dimensions, or pass n_images=0 for a
//                 text-only completion). [Phase 4: all images share one
//                 fixed WxH; per-image sizing is a later concern.]
//   n_images      number of images; may be 0 (text-only).
//   prompt        raw prompt bytes (chat template already applied by caller).
//                 Need not be NUL-terminated. The engine injects one media
//                 marker per image.
//   out_buf       caller buffer of at least the bound returned by a size-query.
//                 Size-query (out_buf=NULL / out_capacity=0) does not run
//                 inference; it returns BUFFER_TOO_SMALL with *out_written set
//                 to max_tokens times the largest serialized token piece in
//                 the loaded vocabulary.
//
// Returns JETSON_PI_MLLM_OK or a negative status. Each call is independent and
// is implemented through prefill + repeated decode_step.
int32_t jetson_pi_mllm_infer(jetson_pi_mllm * handle,
                             const uint8_t * const * images_rgb,
                             uint32_t n_images,
                             uint32_t image_height, uint32_t image_width,
                             const char * prompt, size_t prompt_len,
                             char * out_buf, size_t out_capacity,
                             size_t * out_written);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JETSON_PI_MLLM_H
