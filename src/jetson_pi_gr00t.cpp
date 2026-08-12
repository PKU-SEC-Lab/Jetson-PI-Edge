#include "jetson_pi_gr00t.h"
#include "gr00t-input.h"

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Gr00tEngine {
    mtmd_context * mtmd = nullptr;
    llama_context * lctx = nullptr;
    llama_model * model = nullptr;
    std::string backend;
    uint32_t n_images = 0;
    uint32_t image_height = 0;
    uint32_t image_width = 0;
    uint32_t action_steps = 0;
    uint32_t action_dim = 0;
    uint32_t state_dim = 0;
    int32_t embodiment_id = 0;
    std::string last_error;
};

thread_local std::string g_open_error;

int32_t reject(Gr00tEngine * e, int32_t code, const char * message) {
    if (e) e->last_error = message ? message : "";
    return code;
}

void free_bitmaps(std::vector<mtmd_bitmap *> & bitmaps) {
    for (mtmd_bitmap * bitmap : bitmaps) mtmd_bitmap_free(bitmap);
    bitmaps.clear();
}

} // namespace

extern "C" {

int32_t jetson_pi_gr00t_open(const jetson_pi_gr00t_config * config, jetson_pi_gr00t ** out) {
    g_open_error.clear();
    if (!out) return JETSON_PI_GR00T_INVALID;
    *out = nullptr;
    if (!config || config->struct_size < sizeof(*config) || !config->model_path ||
            !config->mmproj_path || !config->backend || !config->n_images ||
            !config->image_width || !config->image_height) {
        g_open_error = "invalid GR00T open configuration";
        return JETSON_PI_GR00T_INVALID;
    }
    const std::string backend = config->backend;
    if (backend != "cpu" && backend != "cuda" && backend != "vulkan" &&
            backend != "opencl" && backend != "sycl") {
        g_open_error = "unsupported GR00T backend";
        return JETSON_PI_GR00T_INVALID;
    }
    Gr00tEngine * e = new (std::nothrow) Gr00tEngine();
    if (!e) return JETSON_PI_GR00T_INVALID;
    e->backend = backend;
    e->n_images = config->n_images;
    e->image_width = config->image_width;
    e->image_height = config->image_height;
    e->embodiment_id = config->embodiment_id;
    auto fail = [&](int32_t code, const char * message) {
        g_open_error = message;
        if (e->mtmd) mtmd_free(e->mtmd);
        if (e->lctx) llama_free(e->lctx);
        if (e->model) llama_model_free(e->model);
        delete e;
        return code;
    };

    ggml_backend_load_all();
    ggml_backend_dev_t device = nullptr;
    if (backend != "cpu") {
        const char * registry = backend == "cuda" ? "CUDA" : backend == "vulkan" ? "Vulkan" :
            backend == "opencl" ? "OpenCL" : "SYCL";
        ggml_backend_reg_t reg = ggml_backend_reg_by_name(registry);
        if (!reg || ggml_backend_reg_dev_count(reg) == 0) return fail(JETSON_PI_GR00T_LOAD_FAILED, "requested backend is unavailable");
        device = ggml_backend_reg_dev_get(reg, 0);
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = device ? 9999 : 0;
    ggml_backend_dev_t devices[] = {device, nullptr};
    mp.devices = device ? devices : nullptr;
    e->model = llama_model_load_from_file(config->model_path, mp);
    if (!e->model) return fail(JETSON_PI_GR00T_LOAD_FAILED, "failed to load GR00T model");
    char architecture[64] = {};
    llama_model_meta_val_str(e->model, "general.architecture", architecture, sizeof(architecture));
    if (std::strcmp(architecture, "gr00t-n1d7") != 0) return fail(JETSON_PI_GR00T_INVALID, "model is not gr00t-n1d7");

    const int hw = (int) std::thread::hardware_concurrency();
    const int threads = config->n_threads > 0 ? config->n_threads : std::max(1, hw);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    e->lctx = llama_init_from_model(e->model, cp);
    if (!e->lctx) return fail(JETSON_PI_GR00T_LOAD_FAILED, "failed to create GR00T context");
    if (!llama_set_gr00t_embodiment(e->lctx, e->embodiment_id)) return fail(JETSON_PI_GR00T_INVALID, "invalid embodiment_id");

    mtmd_context_params mdp = mtmd_context_params_default();
    mdp.use_gpu = device != nullptr;
    mdp.n_threads = threads;
    mdp.warmup = false;
    mdp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    e->mtmd = mtmd_init_from_file(config->mmproj_path, e->model, mdp);
    if (!e->mtmd) return fail(JETSON_PI_GR00T_LOAD_FAILED, "failed to load GR00T mmproj");
    e->state_dim = (uint32_t) llama_get_gr00t_state_dim(e->lctx);
    e->action_steps = (uint32_t) llama_get_gr00t_action_steps(e->lctx);
    e->action_dim = (uint32_t) llama_get_gr00t_action_dim(e->lctx);
    if (!e->state_dim || !e->action_steps || !e->action_dim) {
        return fail(JETSON_PI_GR00T_INVALID, "GR00T state/action shape unavailable");
    }
    *out = reinterpret_cast<jetson_pi_gr00t *>(e);
    return JETSON_PI_GR00T_OK;
}

void jetson_pi_gr00t_close(jetson_pi_gr00t * handle) {
    auto * e = reinterpret_cast<Gr00tEngine *>(handle);
    if (!e) return;
    if (e->mtmd) mtmd_free(e->mtmd);
    if (e->lctx) llama_free(e->lctx);
    if (e->model) llama_model_free(e->model);
    delete e;
}

const char * jetson_pi_gr00t_last_error(const jetson_pi_gr00t * handle) {
    const auto * e = reinterpret_cast<const Gr00tEngine *>(handle);
    return e ? e->last_error.c_str() : "null jetson_pi_gr00t handle";
}
const char * jetson_pi_gr00t_open_error(void) { return g_open_error.c_str(); }

int32_t jetson_pi_gr00t_action_shape(const jetson_pi_gr00t * handle, uint32_t * steps, uint32_t * dim) {
    const auto * e = reinterpret_cast<const Gr00tEngine *>(handle);
    if (!e || !steps || !dim) return JETSON_PI_GR00T_INVALID;
    *steps = e->action_steps; *dim = e->action_dim;
    return JETSON_PI_GR00T_OK;
}

int32_t jetson_pi_gr00t_infer(jetson_pi_gr00t * handle,
        const uint8_t * const * images_rgb, uint32_t n_images,
        const char * instruction, size_t instruction_len,
        const float * state, size_t n_state,
        float * actions_out, size_t action_capacity, size_t * actions_written) {
    auto * e = reinterpret_cast<Gr00tEngine *>(handle);
    if (actions_written) *actions_written = 0;
    if (!e || !images_rgb || n_images != e->n_images || !instruction || !instruction_len ||
            !actions_out || !actions_written || (!state && n_state)) {
        return reject(e, JETSON_PI_GR00T_INVALID, "invalid GR00T infer arguments");
    }
    const size_t action_count = (size_t) e->action_steps * e->action_dim;
    if (action_capacity < action_count) { *actions_written = action_count; return reject(e, JETSON_PI_GR00T_BUFFER_TOO_SMALL, "action buffer too small"); }
    std::vector<float> padded_state;
    std::string input_error;
    if (!gr00t_input::prepare_state(state, n_state, e->state_dim, padded_state, input_error)) {
        const int32_t code = n_state > e->state_dim ? JETSON_PI_GR00T_STATE_SIZE : JETSON_PI_GR00T_INVALID;
        return reject(e, code, input_error.c_str());
    }
    llama_set_gr00t_state(e->lctx, padded_state.data(), padded_state.size());
    llama_memory_clear(llama_get_memory(e->lctx), true);

    std::vector<gr00t_input::processed_image> processed;
    std::vector<mtmd_bitmap *> bitmaps;
    std::vector<const mtmd_bitmap *> bitmap_ptrs;
    processed.reserve(n_images); bitmaps.reserve(n_images); bitmap_ptrs.reserve(n_images);
    for (uint32_t i = 0; i < n_images; ++i) {
        if (!images_rgb[i]) { free_bitmaps(bitmaps); return reject(e, JETSON_PI_GR00T_INVALID, "null RGB image"); }
        gr00t_input::processed_image image;
        if (!gr00t_input::preprocess_rgb(images_rgb[i], e->image_width, e->image_height, image, input_error)) {
            free_bitmaps(bitmaps);
            return reject(e, JETSON_PI_GR00T_INVALID, input_error.c_str());
        }
        processed.push_back(std::move(image));
        mtmd_bitmap * bitmap = mtmd_bitmap_init(processed.back().width, processed.back().height, processed.back().rgb.data());
        if (!bitmap) { free_bitmaps(bitmaps); return reject(e, JETSON_PI_GR00T_INFER_FAILED, "failed to create bitmap"); }
        bitmaps.push_back(bitmap); bitmap_ptrs.push_back(bitmap);
    }

    std::string prompt = gr00t_input::build_prompt(
            instruction, instruction_len, n_images, mtmd_default_marker());
    mtmd_input_text text{prompt.c_str(), false, true};
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (!chunks) { free_bitmaps(bitmaps); return reject(e, JETSON_PI_GR00T_INFER_FAILED, "failed to allocate chunks"); }
    const int tokenized = mtmd_tokenize(e->mtmd, chunks, &text, bitmap_ptrs.data(), bitmap_ptrs.size());
    free_bitmaps(bitmaps);
    if (tokenized != 0) { mtmd_input_chunks_free(chunks); return reject(e, JETSON_PI_GR00T_INFER_FAILED, "GR00T tokenization failed"); }
    llama_pos n_past = 0;
    const int decoded = mtmd_helper_eval_chunks_gr00t(e->mtmd, e->lctx, chunks, 0, &n_past);
    mtmd_input_chunks_free(chunks);
    if (decoded != 0 || llama_gr00t_generate_action(e->lctx) != 0) return reject(e, JETSON_PI_GR00T_INFER_FAILED, "GR00T graph execution failed");
    const float * action = llama_get_gr00t_action(e->lctx);
    if (!action) return reject(e, JETSON_PI_GR00T_INFER_FAILED, "GR00T action unavailable");
    std::memcpy(actions_out, action, action_count * sizeof(float));
    *actions_written = action_count;
    e->last_error.clear();
    return JETSON_PI_GR00T_OK;
}

} // extern "C"
