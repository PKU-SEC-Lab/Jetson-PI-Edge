#ifndef JETSON_PI_PI0_H
#define JETSON_PI_PI0_H

//
// jetson_pi_pi0 - narrow Pi0 policy C API.
//
// Wraps the Pi0 whole-graph inference path (VIT -> encode -> diffusion decode
// -> action chunk) behind one opaque handle so an embedding host (e.g. FlashRT)
// can drive a Pi0 policy tick without touching llama.cpp/mtmd/GGML internals.
//
// The host hands raw RGB images, a text prompt, and an optional proprioception
// state; the engine returns an action chunk. Each infer is an independent
// first-turn tick: KV cache and position are reset before every call.
//
// This header depends only on <stddef.h>/<stdint.h>. No llama.h / mtmd.h /
// ggml.h types are exposed here.
//

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// Thread safety: a single jetson_pi_pi0 handle is NOT thread-safe - the
// caller must serialize access to one handle. Different handles are
// independent and may be used concurrently. jetson_pi_pi0_open,
// jetson_pi_pi0_open_error, and jetson_pi_pi0_default... are thread-safe
// (use thread-local error sinks).
//

typedef struct jetson_pi_pi0 jetson_pi_pi0;

typedef struct jetson_pi_pi0_config {
    uint32_t struct_size;

    const char * model_path;   /* Pi0 policy GGUF                              */
    const char * mmproj_path;  /* VIT mmproj GGUF                              */
    const char * backend;      /* "cpu" | "cuda" | "vulkan" | "opencl" | "sycl" */

    uint32_t n_views;          /* camera views per tick (images per infer)     */
    uint32_t image_height;     /* pixels, e.g. 224                             */
    uint32_t image_width;      /* pixels, e.g. 224                             */

    int32_t n_threads;         /* CPU threads; 0 = hardware_concurrency        */
} jetson_pi_pi0_config;

enum jetson_pi_pi0_status {
    JETSON_PI_PI0_OK               =  0,
    JETSON_PI_PI0_INVALID          = -1, /* bad args / not a Pi0 model       */
    JETSON_PI_PI0_LOAD_FAILED      = -2, /* model / mmproj / backend load    */
    JETSON_PI_PI0_DIM_MISMATCH     = -3, /* config dims disagree with model  */
    JETSON_PI_PI0_BUFFER_TOO_SMALL = -4, /* actions_out too small            */
    JETSON_PI_PI0_ACTION_NOT_READY = -5, /* infer did not produce an action  */
    JETSON_PI_PI0_INFER_FAILED     = -6, /* tokenize / eval failed           */
    JETSON_PI_PI0_STATE_SIZE       = -7, /* state n_values > action_dim      */
};

// Load the Pi0 model + mmproj and create the inference context. On success
// *out is a handle the caller owns (release via jetson_pi_pi0_close). On
// failure *out is set to NULL and a negative status is returned; the failure
// is described by jetson_pi_pi0_last_error on the returned handle (or, when
// the handle could not be created, by the most recent open error).
int32_t jetson_pi_pi0_open(const jetson_pi_pi0_config * config,
                           jetson_pi_pi0 ** out);

// Release the handle. Safe to call with NULL.
void jetson_pi_pi0_close(jetson_pi_pi0 * handle);

// Always returns a non-null const char * ("" when no error is recorded).
// Pointer validity: until the next call to any jetson_pi_pi0_* function on
// the same handle from the same thread.
const char * jetson_pi_pi0_last_error(const jetson_pi_pi0 * handle);

// Description of the most recent jetson_pi_pi0_open failure on the calling
// thread. open() returns NULL on failure, so there is no handle to query
// last_error from; this sink fills that gap. Always non-null ("").
// Pointer validity: until the next jetson_pi_pi0_open call on the same thread.
const char * jetson_pi_pi0_open_error(void);

// Read the model's action chunk shape: [action_steps, action_dim].
int32_t jetson_pi_pi0_action_shape(const jetson_pi_pi0 * handle,
                                   uint32_t * action_steps,
                                   uint32_t * action_dim);

// Fine-grained Pi0 execution. context resets the tick, applies state,
// tokenizes/encodes prompt+images, and retains one provider-private pending
// context. action consumes that context exactly once and writes the action
// chunk. Calling action without a successful context returns ACTION_NOT_READY.
int32_t jetson_pi_pi0_context(jetson_pi_pi0 * handle,
                              const uint8_t * const * images_rgb,
                              uint32_t n_images,
                              const char * prompt, size_t prompt_len,
                              const float * state, size_t n_state);
int32_t jetson_pi_pi0_discard_context(jetson_pi_pi0 * handle);
int32_t jetson_pi_pi0_action(jetson_pi_pi0 * handle,
                             float * actions_out, size_t action_capacity,
                             size_t * actions_written);

// One whole-graph Pi0 policy tick.
//
//   images_rgb      array of n_images RGB row-major buffers; each buffer is
//                   image_height * image_width * 3 bytes matching the config
//                   dimensions. n_images must equal config.n_views.
//   prompt          UTF-8 bytes. Must NOT contain media markers; the engine
//                   injects one marker per view. Need not be NUL-terminated.
//   state           n_state float32 proprioception values, or NULL to let the
//                   model use a zero state (Pi0 native behavior). When
//                   non-NULL and n_state < action_dim, the engine zero-pads
//                   to action_dim; n_state > action_dim is rejected.
//   actions_out     caller buffer of action_capacity float32 slots. On OK,
//                   *actions_written is set to action_steps * action_dim.
//
// Returns JETSON_PI_PI0_OK or a negative status. Implemented by context then
// action; each call clears KV state and resets position.
int32_t jetson_pi_pi0_infer(jetson_pi_pi0 * handle,
                            const uint8_t * const * images_rgb,
                            uint32_t n_images,
                            const char * prompt, size_t prompt_len,
                            const float * state, size_t n_state,
                            float * actions_out, size_t action_capacity,
                            size_t * actions_written);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // JETSON_PI_PI0_H
