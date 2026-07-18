#ifndef JETSON_PI_LLM_H
#define JETSON_PI_LLM_H

//
// jetson_pi_llm - narrow generic GGUF LLM C API.
//
// Wraps the standard llama.cpp generate loop (tokenize -> decode -> sample ->
// detokenize) behind one opaque handle so an embedding host (e.g. FlashRT)
// can drive a text completion without touching llama.cpp/GGML internals.
//
// The host hands one raw prompt (chat template applied by the caller) and
// receives one generated text blob. Each generate call clears KV state and
// runs an independent first-turn completion (no multi-turn state).
//
// This header depends only on <stddef.h>/<stdint.h>. No llama.h / ggml.h
// types are exposed here.
//

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jetson_pi_llm jetson_pi_llm;

typedef struct jetson_pi_llm_config {
    uint32_t struct_size;

    const char * model_path;   /* GGUF LLM                                     */
    const char * backend;      /* "cpu" | "cuda" | "vulkan" | "opencl" | "sycl" */

    uint32_t n_ctx;            /* KV context size; 0 = 4096 fallback         */
    int32_t  n_threads;        /* CPU threads; 0 = hardware_concurrency       */

    float    temp;             /* sampler temperature (0 = greedy)            */
    int32_t  top_k;            /* 0 = disabled                                */
    float    top_p;            /* 0 = disabled                                */
    uint32_t seed;             /* RNG seed; 0 = LLAMA_DEFAULT_SEED            */
    uint32_t max_tokens;       /* cap on generated tokens; 0 = 512            */
} jetson_pi_llm_config;

enum jetson_pi_llm_status {
    JETSON_PI_LLM_OK               =  0,
    JETSON_PI_LLM_INVALID          = -1, /* bad args / load failed             */
    JETSON_PI_LLM_BUFFER_TOO_SMALL = -2, /* out_buf too small                  */
    JETSON_PI_LLM_INFER_FAILED     = -3, /* tokenize / decode / sample failed  */
    JETSON_PI_LLM_BAD_STATE        = -4, /* prefill/decode ordering violation  */
};

// Load the model + create the inference context + sampler chain. On success
// *out is a handle the caller owns (release via jetson_pi_llm_close). On
// failure *out is set to NULL; the failure is described by
// jetson_pi_llm_open_error() on the calling thread.
int32_t jetson_pi_llm_open(const jetson_pi_llm_config * config,
                           jetson_pi_llm ** out);

// Release the handle. Safe to call with NULL.
void jetson_pi_llm_close(jetson_pi_llm * handle);

// Always returns a non-null const char * ("" when no error is recorded).
// Pointer validity: until the next jetson_pi_llm_* call on the same handle /
// same thread.
const char * jetson_pi_llm_last_error(const jetson_pi_llm * handle);

// Description of the most recent jetson_pi_llm_open failure on the calling
// thread (open returns NULL on failure). Always non-null ("").
// Pointer validity: until the next jetson_pi_llm_open call on the same thread.
const char * jetson_pi_llm_open_error(void);

// Explicit session API. reset clears KV and sampler state. prefill always
// starts a fresh session, tokenizes the prompt, and evaluates it so logits for
// the first generated token are available. decode_step samples exactly one
// token from the current logits; non-EOG tokens are accepted by the sampler and
// decoded into KV so the next call observes the next-token logits. The host may
// interrupt generation by simply stopping calls; additional decode_step calls
// after max_tokens return JETSON_PI_LLM_BAD_STATE until reset/prefill starts a
// new session.
int32_t jetson_pi_llm_reset(jetson_pi_llm * handle);
int32_t jetson_pi_llm_prefill(jetson_pi_llm * handle,
                              const char * prompt, size_t prompt_len);
int32_t jetson_pi_llm_prefill_tokens(jetson_pi_llm * handle,
                                     const int32_t * tokens,
                                     size_t n_tokens);
int32_t jetson_pi_llm_decode_step(jetson_pi_llm * handle,
                                  int32_t * token_id, int32_t * is_eog);

// Current next-token logits. *logits_written is measured in float elements.
// Size-query with logits_out=NULL returns BUFFER_TOO_SMALL and the vocabulary
// size. Valid after prefill and after each non-EOG decode_step.
int32_t jetson_pi_llm_get_logits(jetson_pi_llm * handle,
                                 float * logits_out, size_t logits_capacity,
                                 size_t * logits_written);

// Stateless token detokenization. *piece_written is measured in bytes and is
// populated on BUFFER_TOO_SMALL, enabling an exact size query.
int32_t jetson_pi_llm_token_to_piece(jetson_pi_llm * handle, int32_t token_id,
                                     char * piece_out, size_t piece_capacity,
                                     size_t * piece_written);

// One whole-prompt completion. Clears KV state, tokenizes the prompt, runs
// the decode/sample loop up to max_tokens or EOG, and writes the generated
// text (UTF-8, no NUL) into out_buf. *out_written is set to the byte count
// (even on BUFFER_TOO_SMALL, so callers can size-query with out_buf=NULL,
// capacity=0).
//
//   prompt        raw prompt bytes (chat template already applied by caller).
//                 Need not be NUL-terminated. The engine adds BOS (no
//                 parse_special beyond the marker); callers should NOT include
//                 a leading BOS token in the prompt.
//
//   out_buf       caller buffer of at least the bound returned by a size-query.
//                 Size-query (out_buf=NULL / out_capacity=0) does not run
//                 generation; it returns BUFFER_TOO_SMALL with *out_written
//                 set to max_tokens times the largest serialized token piece
//                 in the loaded vocabulary.
//
// Returns JETSON_PI_LLM_OK or a negative status. Each call is independent and
// is implemented through reset + prefill + repeated decode_step.
int32_t jetson_pi_llm_generate(jetson_pi_llm * handle,
                               const char * prompt, size_t prompt_len,
                               char * out_buf, size_t out_capacity,
                               size_t * out_written);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JETSON_PI_LLM_H
