// jetson_pi_pi0 — Pi0 policy whole-graph inference wrapped behind a narrow
// C handle. See include/jetson_pi_pi0.h for the public contract.
//
// This translation unit deliberately keeps every llama.cpp/mtmd/GGML symbol
// private to the jetson_pi_pi0 shared/static library; the public header
// exposes only opaque pointers and <stdint.h> types, so an embedding host
// (FlashRT) never sees GGML types.

#include "jetson_pi_pi0.h"

#include "ggml-backend.h"   // ggml_backend_load_all
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "pi-model-detect.h"
#include "jetson_pi_pi0_prompt.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <thread>
#include <vector>
namespace {

// One Pi0 policy engine: model + inference context + multimodal (VIT) context.
// Destruction order is the reverse of construction: mtmd -> ctx -> model.
struct Pi0Engine {
    mtmd_context *   mtmd  = nullptr;
    llama_context *  lctx  = nullptr;
    llama_model *    model = nullptr;

    // Config snapshot (owned strings).
    std::string model_path;
    std::string mmproj_path;
    std::string backend;
    uint32_t n_views       = 0;
    uint32_t image_height  = 0;
    uint32_t image_width   = 0;
    int32_t  n_threads_cfg = 0;
    int32_t  n_threads_eff = 0;

    // Model-derived action chunk shape.
    uint32_t action_steps = 0;
    uint32_t action_dim   = 0;
    pi_model_kind model_kind = PI_MODEL_AUTO;

    mtmd_pi0_context * pending_context = nullptr;

    std::string last_error;
    std::atomic<long> refs{1};

    void set_error(const char * msg) { last_error = msg ? msg : ""; }
    void clear_error() { last_error.clear(); }
};

// ---- error helpers ----------------------------------------------------------

// Open errors are reported through a process sink because the contract is
// *out == NULL on failure (no handle to query last_error from).
static thread_local std::string g_open_error;

int32_t reject(Pi0Engine * e, int32_t status, const char * msg) {
    if (e) e->set_error(msg);
    return status;
}

void free_bitmaps(std::vector<mtmd_bitmap*> & bmps) {
    for (auto * b : bmps) {
        if (b) mtmd_bitmap_free(b);
    }
    bmps.clear();
}

struct PiModelKindScope {
    pi_model_kind previous;

    explicit PiModelKindScope(pi_model_kind kind)
        : previous(pi_model_set_thread_override(kind)) {}

    ~PiModelKindScope() {
        pi_model_set_thread_override(previous);
    }
};

} // namespace

extern "C" {

uint32_t jetson_pi_pi0_capabilities(void) {
    return JETSON_PI_PI0_CAP_REAL_CONTEXT_ACTION;
}

int32_t jetson_pi_pi0_open(const jetson_pi_pi0_config * config,
                           jetson_pi_pi0 ** out) {
    g_open_error.clear();
    if (!out) {
        g_open_error = "invalid Pi0 open output pointer";
        return JETSON_PI_PI0_INVALID;
    }
    *out = nullptr;
    if (!config ||
        config->struct_size < sizeof(jetson_pi_pi0_config) ||
        !config->model_path || !config->model_path[0] ||
        !config->mmproj_path || !config->mmproj_path[0] ||
        !config->backend || !config->backend[0] ||
        !config->n_views || config->n_views > 3 ||
        !config->image_height || !config->image_width ||
        static_cast<size_t>(config->image_width) >
            std::numeric_limits<size_t>::max() /
            static_cast<size_t>(config->image_height) / 3u) {
        g_open_error = "invalid Pi0 open configuration";
        return JETSON_PI_PI0_INVALID;
    }
    // The selected build must contain the requested GGML backend. Compiling a
    // backend does not guarantee every model op is supported on it — verify
    // each model×backend combination independently.
    if (std::strcmp(config->backend, "cpu") != 0 &&
        std::strcmp(config->backend, "cuda") != 0 &&
        std::strcmp(config->backend, "vulkan") != 0 &&
        std::strcmp(config->backend, "opencl") != 0 &&
        std::strcmp(config->backend, "sycl") != 0) {
        g_open_error = "unsupported Pi0 backend";
        return JETSON_PI_PI0_INVALID;
    }

    Pi0Engine * e = new (std::nothrow) Pi0Engine();
    if (!e) {
        g_open_error = "Pi0 engine allocation failed";
        return JETSON_PI_PI0_INVALID;
    }

    e->model_path   = config->model_path;
    e->mmproj_path  = config->mmproj_path;
    e->backend      = config->backend;
    e->n_views      = config->n_views;
    e->image_height = config->image_height;
    e->image_width  = config->image_width;
    e->n_threads_cfg = config->n_threads;

    const pi_model_detect_result detected = pi_model_detect_gguf_pair(
        e->model_path, e->mmproj_path);
    if (detected.kind != PI_MODEL_PI0 && detected.kind != PI_MODEL_PI05) {
        g_open_error = "could not determine a consistent Pi0/Pi0.5 model kind: ";
        g_open_error += detected.reason;
        delete e;
        return JETSON_PI_PI0_INVALID;
    }
    e->model_kind = detected.kind;
    PiModelKindScope model_kind_scope(e->model_kind);

    int32_t hw = static_cast<int32_t>(std::thread::hardware_concurrency());
    e->n_threads_eff = (e->n_threads_cfg > 0) ? e->n_threads_cfg
                                              : (hw > 0 ? hw : 1);

    // Open errors are reported through the process sink because the contract
    // is *out == NULL on failure (no handle to query last_error from).
    auto fail_open = [&](int32_t status, const char * msg) -> int32_t {
        g_open_error = msg ? msg : "";
        mtmd_helper_free_pi0_context(e->pending_context);
        if (e->mtmd)  mtmd_free(e->mtmd);
        if (e->lctx)  llama_free(e->lctx);
        if (e->model) llama_model_free(e->model);
        delete e;
        *out = nullptr;
        return status;
    };

    // Backends must be registered before llama_model_load_from_file. The
    // common_params parser does this for the CLI; we are not on that path.
    ggml_backend_load_all();

    ggml_backend_dev_t selected_device = nullptr;
    if (e->backend != "cpu") {
        const char * reg_name = e->backend == "cuda" ? "CUDA" :
                                e->backend == "vulkan" ? "Vulkan" :
                                e->backend == "opencl" ? "OpenCL" : "SYCL";
        ggml_backend_reg_t reg = ggml_backend_reg_by_name(reg_name);
        if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
            std::string msg = "requested GGML backend has no registered device: ";
            msg += e->backend;
            return fail_open(JETSON_PI_PI0_LOAD_FAILED, msg.c_str());
        }
        selected_device = ggml_backend_reg_dev_get(reg, 0);
        if (ggml_backend_dev_type(selected_device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            std::string msg = "requested GGML backend has no accelerator device: ";
            msg += e->backend;
            return fail_open(JETSON_PI_PI0_LOAD_FAILED, msg.c_str());
        }
    }

    // Any non-cpu backend offloads both the text model and MTMD/CLIP to device
    // 0 of the exact GGML backend registry selected by the backend string.
    bool use_gpu = e->backend != "cpu";

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = use_gpu ? 9999 : 0;
    ggml_backend_dev_t devices[] = { selected_device, nullptr };
    mparams.devices = selected_device ? devices : nullptr;
    e->model = llama_model_load_from_file(e->model_path.c_str(), mparams);
    if (!e->model) {
        std::string msg = "llama_model_load_from_file failed for ";
        msg += e->model_path;
        return fail_open(JETSON_PI_PI0_LOAD_FAILED, msg.c_str());
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 4096;   // >> combined token count; keeps n_batch clamp off
    cparams.n_batch     = 2048;
    cparams.n_ubatch    = 1024;
    cparams.n_seq_max   = 4;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.n_threads    = e->n_threads_eff;
    cparams.n_threads_batch = e->n_threads_eff;
    e->lctx = llama_init_from_model(e->model, cparams);
    if (!e->lctx) {
        return fail_open(JETSON_PI_PI0_LOAD_FAILED,
                         "llama_init_from_model failed");
    }

    mtmd_context_params mdp = mtmd_context_params_default();
    mdp.use_gpu          = use_gpu;
    mdp.print_timings    = false;
    mdp.n_threads        = e->n_threads_eff;
    mdp.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_AUTO;
    e->mtmd = mtmd_init_from_file_with_device(
        e->mmproj_path.c_str(), e->model, mdp, selected_device);
    if (!e->mtmd) {
        return fail_open(JETSON_PI_PI0_LOAD_FAILED,
                         "mtmd_init_from_file failed");
    }

    const int32_t steps = llama_get_pi0_action_steps(e->lctx);
    const int32_t dim = llama_get_pi0_action_dim(e->lctx);
    if (steps <= 0 || dim <= 0) {
        return fail_open(JETSON_PI_PI0_INVALID,
                         "model is not a Pi0 policy (action shape unavailable)");
    }
    e->action_steps = static_cast<uint32_t>(steps);
    e->action_dim   = static_cast<uint32_t>(dim);

    if (!mtmd_is_pi0_model(e->mtmd)) {
        return fail_open(JETSON_PI_PI0_INVALID,
                         "mmproj is not a Pi0 vision model");
    }

    e->clear_error();
    *out = reinterpret_cast<jetson_pi_pi0*>(e);
    return JETSON_PI_PI0_OK;
}

void jetson_pi_pi0_close(jetson_pi_pi0 * handle) {
    Pi0Engine * e = reinterpret_cast<Pi0Engine*>(handle);
    if (!e) return;
    if (e->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        PiModelKindScope model_kind_scope(e->model_kind);
        mtmd_helper_free_pi0_context(e->pending_context);
        if (e->mtmd)  mtmd_free(e->mtmd);
        if (e->lctx)  llama_free(e->lctx);
        if (e->model) llama_model_free(e->model);
        delete e;
    }
}

const char * jetson_pi_pi0_last_error(const jetson_pi_pi0 * handle) {
    const Pi0Engine * e = reinterpret_cast<const Pi0Engine*>(handle);
    if (!e) return "null jetson_pi_pi0 handle";
    return e->last_error.c_str();  // "" when no error, matching the header
}

const char * jetson_pi_pi0_open_error(void) {
    return g_open_error.c_str();
}

int32_t jetson_pi_pi0_action_shape(const jetson_pi_pi0 * handle,
                                   uint32_t * action_steps,
                                   uint32_t * action_dim) {
    const Pi0Engine * e = reinterpret_cast<const Pi0Engine*>(handle);
    if (!e || !action_steps || !action_dim) return JETSON_PI_PI0_INVALID;
    *action_steps = e->action_steps;
    *action_dim   = e->action_dim;
    return JETSON_PI_PI0_OK;
}

int32_t jetson_pi_pi0_infer(jetson_pi_pi0 * handle,
                            const uint8_t * const * images_rgb,
                            uint32_t n_images,
                            const char * prompt, size_t prompt_len,
                            const float * state, size_t n_state,
                            float * actions_out, size_t action_capacity,
                            size_t * actions_written) {
    Pi0Engine * e = reinterpret_cast<Pi0Engine*>(handle);
    if (actions_written) *actions_written = 0;
    if (!e) return JETSON_PI_PI0_INVALID;
    e->clear_error();
    // infer() is context()+action(); clear any prior pending context first so
    // this tick is independent (one-shot semantics, same as context()).
    mtmd_helper_free_pi0_context(e->pending_context);
    e->pending_context = nullptr;
    if (!actions_out || !actions_written) {
        return reject(e, JETSON_PI_PI0_INVALID,
                      "invalid infer output arguments");
    }
    const size_t need_elems = static_cast<size_t>(e->action_steps) *
                              static_cast<size_t>(e->action_dim);
    if (action_capacity < need_elems) {
        *actions_written = need_elems;
        return reject(e, JETSON_PI_PI0_BUFFER_TOO_SMALL,
                      "actions_out buffer too small");
    }
    int32_t status = jetson_pi_pi0_context(
        handle, images_rgb, n_images, prompt, prompt_len, state, n_state);
    if (status != JETSON_PI_PI0_OK) return status;
    return jetson_pi_pi0_action(
        handle, actions_out, action_capacity, actions_written);
}

int32_t jetson_pi_pi0_context(jetson_pi_pi0 * handle,
                              const uint8_t * const * images_rgb,
                              uint32_t n_images,
                              const char * prompt, size_t prompt_len,
                              const float * state, size_t n_state) {
    Pi0Engine * e = reinterpret_cast<Pi0Engine*>(handle);
    if (!e) return JETSON_PI_PI0_INVALID;
    PiModelKindScope model_kind_scope(e->model_kind);
    e->clear_error();
    // One-shot semantics: a new context() invalidates any prior pending
    // context so action() can never return a stale result from a previous tick.
    mtmd_helper_free_pi0_context(e->pending_context);
    e->pending_context = nullptr;
    if (!images_rgb || n_images != e->n_views || !prompt || prompt_len == 0) {
        return reject(e, JETSON_PI_PI0_INVALID, "invalid context arguments");
    }
    // PI0.5 discretizes proprioception into the text prompt and never reads the
    // legacy llama_set_pi0_state tensor, so its state is not bound to
    // action_dim (openpi proprioception is 8 dims, which can exceed the 7-dim
    // action). Legacy Pi0 consumes state via the cross.state tensor, which
    // requires exactly action_dim values (zero-padded when shorter).
    const bool is_pi05 = (e->model_kind == PI_MODEL_PI05);
    const size_t action_dim = static_cast<size_t>(e->action_dim);
    if ((!state && n_state != 0) || (state && n_state == 0)) {
        return reject(e, JETSON_PI_PI0_INVALID,
                      "state pointer and n_state are inconsistent");
    }
    const bool pi05_padded_state = is_pi05 && n_state == action_dim && n_state > 8;
    if ((!is_pi05 && n_state > action_dim) ||
        (is_pi05 && n_state > 8 && !pi05_padded_state)) {
        return reject(e, JETSON_PI_PI0_STATE_SIZE,
                      is_pi05
                          ? "PI0.5 state must have at most 8 values or be an action_dim-wide zero-padded provider tensor"
                          : "state has more values than the model action_dim");
    }
    for (size_t i = 0; i < n_state; ++i) {
        if (!std::isfinite(state[i])) {
            return reject(e, JETSON_PI_PI0_INVALID,
                          "state values must be finite");
        }
    }
    if (pi05_padded_state) {
        for (size_t i = 8; i < n_state; ++i) {
            if (state[i] != 0.0f) {
                return reject(e, JETSON_PI_PI0_STATE_SIZE,
                              "PI0.5 provider padding after the first 8 state values must be zero");
            }
        }
    }

    // Reset to a fresh first-turn tick: clear KV cache + position.
    llama_memory_clear(llama_get_memory(e->lctx), true);

    if (!is_pi05) {
        // Legacy Pi0: proprioception is a continuous tensor consumed via
        // state_proj. Zero-pad to action_dim when shorter; NULL state means
        // "use a zero state for this tick" (Pi0 native behavior). We must
        // materialize zeros explicitly: llama_memory_clear does not touch
        // cross.state, so a previous tick's state would otherwise leak in.
        if (state) {
            std::vector<float> padded(e->action_dim, 0.0f);
            std::memcpy(padded.data(), state, n_state * sizeof(float));
            llama_set_pi0_state(e->lctx, padded.data(), padded.size());
        } else {
            std::vector<float> zeros(e->action_dim, 0.0f);
            llama_set_pi0_state(e->lctx, zeros.data(), zeros.size());
        }
    }
    // PI0.5 does NOT call llama_set_pi0_state: its graph skips build_inp_state
    // and consumes proprioception only as discretized text (built below).

    // Build bitmaps from raw RGB. mtmd_bitmap_init copies nx*ny*3 bytes.
    std::vector<mtmd_bitmap*> bitmaps;
    bitmaps.reserve(e->n_views);
    const uint32_t nx = e->image_width;
    const uint32_t ny = e->image_height;
    for (uint32_t i = 0; i < e->n_views; ++i) {
        const uint8_t * rgb = images_rgb[i];
        if (!rgb) {
            free_bitmaps(bitmaps);
            return reject(e, JETSON_PI_PI0_INVALID,
                          "images_rgb contained a null view");
        }
        mtmd_bitmap * bmp = mtmd_bitmap_init(nx, ny, rgb);
        if (!bmp) {
            free_bitmaps(bitmaps);
            return reject(e, JETSON_PI_PI0_INFER_FAILED,
                          "mtmd_bitmap_init failed");
        }
        bitmaps.push_back(bmp);
    }

    // Match the foreground Pi0 path: pass the raw multimodal prefix without
    // chat-template wrappers; the multimodal helper applies the model's
    // explicit BOS/newline token contract after tokenization.
    const char * marker = mtmd_default_marker();
    const size_t marker_len = std::strlen(marker);

    // PI0.5 folds the discretized proprioception into the prompt text
    // ("Task: <prompt>, State: <bins>;\nAction: "), matching the foreground
    // server. Legacy Pi0 uses the raw prompt (state is a tensor, above).
    std::string prompt_text;
    if (is_pi05) {
        const float zero_state[8] = {};
        const float * prompt_state = state ? state : zero_state;
        const size_t prompt_state_size = state ? n_state : 8;
        prompt_text = jetson_pi_pi0_detail::format_pi05_openpi_prompt(
            std::string(prompt, prompt_len), prompt_state, prompt_state_size);
    } else {
        prompt_text.assign(prompt, prompt_len);
    }
    const size_t pi_len = prompt_text.size();

    if (marker_len != 0 &&
        static_cast<size_t>(e->n_views) >
            (std::numeric_limits<size_t>::max() - pi_len) / marker_len) {
        free_bitmaps(bitmaps);
        return reject(e, JETSON_PI_PI0_INVALID,
                      "formatted Pi0 prompt size overflows size_t");
    }
    std::string formatted;
    formatted.reserve(pi_len + static_cast<size_t>(e->n_views) * marker_len);
    for (uint32_t i = 0; i < e->n_views; ++i) formatted += marker;
    formatted.append(prompt_text);

    mtmd_input_text text;
    text.text          = formatted.c_str();
    text.add_special   = false;
    text.parse_special = true;

    std::vector<const mtmd_bitmap*> bmp_ptrs(bitmaps.begin(), bitmaps.end());
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (!chunks) {
        free_bitmaps(bitmaps);
        return reject(e, JETSON_PI_PI0_INFER_FAILED,
                      "mtmd_input_chunks_init returned null");
    }
    int32_t tr = mtmd_tokenize(e->mtmd, chunks, &text,
                               bmp_ptrs.data(), bmp_ptrs.size());
    // Bitmaps are consumed by tokenize only for preprocessing; ownership stays
    // with us, so free them now (mirrors mtmd-cli clearing its bitmap list).
    free_bitmaps(bitmaps);
    if (tr != 0) {
        mtmd_input_chunks_free(chunks);
        return reject(e, JETSON_PI_PI0_INFER_FAILED,
                      "mtmd_tokenize failed");
    }

    // Encode VIT + prompt (prefill) and retain the prepared batch as a pending
    // context; the action-producing decode runs in action(). This is a real
    // compute boundary: prepare returns right after llama_encode and before
    // llama_decode, so context() holds the encoded state and action() finishes
    // the forward pass.
    llama_pos new_n_past = 0;
    int32_t er = mtmd_helper_prepare_chunks_pi0_for_model(
        e->mtmd, e->lctx, chunks, e->model_kind,
        /*n_past=*/0, /*seq_id=*/0,
        /*n_batch=*/2048, /*logits_last=*/true, &new_n_past,
        &e->pending_context);
    mtmd_input_chunks_free(chunks);
    if (er != 0 || e->pending_context == nullptr) {
        mtmd_helper_free_pi0_context(e->pending_context);
        e->pending_context = nullptr;
        return reject(e, JETSON_PI_PI0_INFER_FAILED,
                      "mtmd_helper_prepare_chunks_pi0_for_model failed");
    }
    return JETSON_PI_PI0_OK;
}

int32_t jetson_pi_pi0_discard_context(jetson_pi_pi0 * handle) {
    Pi0Engine * e = reinterpret_cast<Pi0Engine*>(handle);
    if (!e) return JETSON_PI_PI0_INVALID;
    e->clear_error();
    mtmd_helper_free_pi0_context(e->pending_context);
    e->pending_context = nullptr;
    return JETSON_PI_PI0_OK;
}

int32_t jetson_pi_pi0_action(jetson_pi_pi0 * handle,
                             float * actions_out, size_t action_capacity,
                             size_t * actions_written) {
    Pi0Engine * e = reinterpret_cast<Pi0Engine*>(handle);
    if (actions_written) *actions_written = 0;
    if (!e || !actions_out || !actions_written) return JETSON_PI_PI0_INVALID;
    PiModelKindScope model_kind_scope(e->model_kind);
    e->clear_error();
    const size_t need_elems = static_cast<size_t>(e->action_steps) *
                              static_cast<size_t>(e->action_dim);
    if (action_capacity < need_elems) {
        *actions_written = need_elems;
        return reject(e, JETSON_PI_PI0_BUFFER_TOO_SMALL,
                      "actions_out buffer too small");
    }
    // A successful action() requires a pending prepared batch from a preceding
    // successful context().
    if (!e->pending_context) {
        return reject(e, JETSON_PI_PI0_ACTION_NOT_READY,
                      "Pi0 context not ready");
    }
    mtmd_pi0_context * context = e->pending_context;
    e->pending_context = nullptr;
    int32_t er = mtmd_helper_decode_pi0(e->lctx, context);
    // mtmd_helper_decode_pi0 only restores causal_attn on failure; restore it
    // unconditionally so a successful split decode does not leave a non-default
    // attention mode that would corrupt a subsequent tick.
    llama_set_causal_attn(e->lctx, true);
    mtmd_helper_free_pi0_context(context);
    if (er != 0) {
        return reject(e, JETSON_PI_PI0_INFER_FAILED,
                      "mtmd_helper_decode_pi0 failed");
    }

    const float * action = llama_get_pi0_action(e->lctx);
    if (action == nullptr) {
        return reject(e, JETSON_PI_PI0_ACTION_NOT_READY,
                      "Pi0 action not ready after infer");
    }
    std::memcpy(actions_out, action, need_elems * sizeof(float));
    *actions_written = need_elems;
    return JETSON_PI_PI0_OK;
}

} // extern "C"
