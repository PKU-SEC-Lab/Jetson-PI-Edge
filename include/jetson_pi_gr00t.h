#ifndef JETSON_PI_GR00T_H
#define JETSON_PI_GR00T_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jetson_pi_gr00t jetson_pi_gr00t;

typedef struct jetson_pi_gr00t_config {
    uint32_t struct_size;
    const char * model_path;
    const char * mmproj_path;
    const char * backend;       /* cpu | cuda | vulkan | opencl | sycl */
    uint32_t n_images;          /* ordered camera/time images per policy tick */
    uint32_t image_height;      /* raw RGB input height */
    uint32_t image_width;       /* raw RGB input width */
    int32_t n_threads;          /* 0 = hardware concurrency */
    int32_t embodiment_id;      /* checkpoint embodiment table index */
} jetson_pi_gr00t_config;

enum jetson_pi_gr00t_status {
    JETSON_PI_GR00T_OK               =  0,
    JETSON_PI_GR00T_INVALID          = -1,
    JETSON_PI_GR00T_LOAD_FAILED      = -2,
    JETSON_PI_GR00T_DIM_MISMATCH     = -3,
    JETSON_PI_GR00T_BUFFER_TOO_SMALL = -4,
    JETSON_PI_GR00T_INFER_FAILED     = -5,
    JETSON_PI_GR00T_STATE_SIZE       = -6,
};

int32_t jetson_pi_gr00t_open(const jetson_pi_gr00t_config * config,
                             jetson_pi_gr00t ** out);
void jetson_pi_gr00t_close(jetson_pi_gr00t * handle);
const char * jetson_pi_gr00t_last_error(const jetson_pi_gr00t * handle);
const char * jetson_pi_gr00t_open_error(void);
int32_t jetson_pi_gr00t_action_shape(const jetson_pi_gr00t * handle,
                                     uint32_t * action_steps,
                                     uint32_t * action_dim);

// One complete GR00T policy tick. Images are RGB-interleaved raw frames with
// config dimensions, in the checkpoint's camera/time order. The API performs
// shortest-edge resize to 256, a deterministic 95% center crop, then another
// shortest-edge resize to 256 before Qwen3-VL smart resize. `instruction` is
// UTF-8 and need not be NUL-terminated. `state` is the already normalized
// policy state; shorter inputs are zero-padded to the checkpoint state width.
int32_t jetson_pi_gr00t_infer(jetson_pi_gr00t * handle,
                              const uint8_t * const * images_rgb,
                              uint32_t n_images,
                              const char * instruction,
                              size_t instruction_len,
                              const float * state,
                              size_t n_state,
                              float * actions_out,
                              size_t action_capacity,
                              size_t * actions_written);

#ifdef __cplusplus
}
#endif

#endif
