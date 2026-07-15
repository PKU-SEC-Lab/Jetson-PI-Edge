// jetson_pi_mllm — multimodal LLM completion wrapped behind a narrow C
// handle. See include/jetson_pi_mllm.h for the public contract.
//
// This translation unit keeps every llama.cpp/mtmd/GGML symbol private to the
// jetson_pi_mllm shared/static library; the public header exposes only opaque
// pointers and <stdint.h> types.

#include "jetson_pi_mllm.h"

#include "ggml-backend.h"   // ggml_backend_load_all
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <atomic>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace {

struct MllmEngine {
    llama_model *   model = nullptr;
    llama_context * lctx  = nullptr;
    mtmd_context *  mtmd  = nullptr;
    llama_sampler * smpl  = nullptr;
    const llama_vocab * vocab = nullptr;

    std::string model_path;
    std::string mmproj_path;
    std::string backend;
    int32_t  n_threads_eff = 0;
    uint32_t max_tokens = 0;
    uint32_t seed = 0;
    size_t max_output_bytes = 0;

    bool session_ready = false;
    bool session_eog = false;
    uint32_t generated_tokens = 0;

    std::string last_error;
    std::atomic<long> refs{1};

    void set_error(const char * msg) { last_error = msg ? msg : ""; }
    void clear_error() { last_error.clear(); }
};

static thread_local std::string g_open_error;

int32_t reject(MllmEngine * e, int32_t status, const char * msg) {
    if (e) e->set_error(msg);
    return status;
}

void free_bitmaps(std::vector<mtmd_bitmap*> & bmps) {
    for (auto * b : bmps) {
        if (b) mtmd_bitmap_free(b);
    }
    bmps.clear();
}

} // namespace

extern "C" {

int32_t jetson_pi_mllm_open(const jetson_pi_mllm_config * config,
                            jetson_pi_mllm ** out) {
    g_open_error.clear();
    if (!out) {
        g_open_error = "invalid MLLM open output pointer";
        return JETSON_PI_MLLM_INVALID;
    }
    *out = nullptr;
    if (!config ||
        config->struct_size < sizeof(jetson_pi_mllm_config) ||
        !config->model_path || !config->model_path[0] ||
        !config->mmproj_path || !config->mmproj_path[0] ||
        !config->backend || !config->backend[0]) {
        g_open_error = "invalid MLLM open configuration";
        return JETSON_PI_MLLM_INVALID;
    }
    // The selected build must contain the requested GGML backend. Compiling a
    // backend does not guarantee every model op is supported on it — verify
    // each model×backend combination independently.
    if (std::strcmp(config->backend, "cpu") != 0 &&
        std::strcmp(config->backend, "cuda") != 0 &&
        std::strcmp(config->backend, "vulkan") != 0 &&
        std::strcmp(config->backend, "opencl") != 0 &&
        std::strcmp(config->backend, "sycl") != 0) {
        g_open_error = "unsupported MLLM backend";
        return JETSON_PI_MLLM_INVALID;
    }

    MllmEngine * e = new (std::nothrow) MllmEngine();
    if (!e) {
        g_open_error = "MLLM engine allocation failed";
        return JETSON_PI_MLLM_INVALID;
    }

    e->model_path  = config->model_path;
    e->mmproj_path = config->mmproj_path;
    e->backend     = config->backend;
    e->max_tokens  = config->max_tokens ? config->max_tokens : 512;
    e->seed        = config->seed;

    int32_t hw = static_cast<int32_t>(std::thread::hardware_concurrency());
    e->n_threads_eff = (config->n_threads > 0) ? config->n_threads
                                               : (hw > 0 ? hw : 1);

    auto fail_open = [&](int32_t status, const char * msg) -> int32_t {
        g_open_error = msg ? msg : "";
        if (e->smpl)  llama_sampler_free(e->smpl);
        if (e->mtmd)  mtmd_free(e->mtmd);
        if (e->lctx)  llama_free(e->lctx);
        if (e->model) llama_model_free(e->model);
        delete e;
        *out = nullptr;
        return status;
    };

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
            return fail_open(JETSON_PI_MLLM_INVALID, msg.c_str());
        }
        selected_device = ggml_backend_reg_dev_get(reg, 0);
        if (ggml_backend_dev_type(selected_device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            std::string msg = "requested GGML backend has no accelerator device: ";
            msg += e->backend;
            return fail_open(JETSON_PI_MLLM_INVALID, msg.c_str());
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
        return fail_open(JETSON_PI_MLLM_INVALID, msg.c_str());
    }
    e->vocab = llama_model_get_vocab(e->model);
    size_t max_piece_bytes = 0;
    const int32_t n_vocab = llama_vocab_n_tokens(e->vocab);
    for (int32_t token = 0; token < n_vocab; ++token) {
        const int32_t piece_status = llama_token_to_piece(
            e->vocab, static_cast<llama_token>(token), nullptr, 0, 0, true);
        const size_t piece_bytes = piece_status < 0
            ? static_cast<size_t>(-static_cast<int64_t>(piece_status))
            : static_cast<size_t>(piece_status);
        if (piece_bytes > max_piece_bytes) max_piece_bytes = piece_bytes;
    }
    if (e->max_tokens != 0 &&
        max_piece_bytes > std::numeric_limits<size_t>::max() / e->max_tokens) {
        return fail_open(JETSON_PI_MLLM_INVALID,
                         "maximum generated text size overflows size_t");
    }
    e->max_output_bytes = max_piece_bytes * e->max_tokens;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = config->n_ctx ? config->n_ctx : 4096;
    cparams.n_batch = cparams.n_ctx;
    cparams.n_seq_max = 1;
    cparams.n_threads = e->n_threads_eff;
    cparams.n_threads_batch = e->n_threads_eff;
    e->lctx = llama_init_from_model(e->model, cparams);
    if (!e->lctx) {
        return fail_open(JETSON_PI_MLLM_INVALID,
                         "llama_init_from_model failed");
    }

    mtmd_context_params mdp = mtmd_context_params_default();
    mdp.use_gpu         = use_gpu;
    mdp.print_timings   = false;
    mdp.n_threads       = e->n_threads_eff;
    mdp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    e->mtmd = mtmd_init_from_file_with_device(
        e->mmproj_path.c_str(), e->model, mdp, selected_device);
    if (!e->mtmd) {
        return fail_open(JETSON_PI_MLLM_INVALID,
                         "mtmd_init_from_file failed");
    }

    auto sparams = llama_sampler_chain_default_params();
    e->smpl = llama_sampler_chain_init(sparams);
    if (!e->smpl) {
        return fail_open(JETSON_PI_MLLM_INVALID,
                         "llama_sampler_chain_init failed");
    }
    if (e->seed == 0) e->seed = LLAMA_DEFAULT_SEED;
    const float temp = config->temp;
    if (temp <= 0.0f) {
        llama_sampler_chain_add(e->smpl, llama_sampler_init_greedy());
    } else {
        if (config->top_k > 0) {
            llama_sampler_chain_add(e->smpl,
                                    llama_sampler_init_top_k(config->top_k));
        }
        if (config->top_p > 0.0f) {
            llama_sampler_chain_add(e->smpl,
                                    llama_sampler_init_top_p(config->top_p, 1));
        }
        llama_sampler_chain_add(e->smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(e->smpl, llama_sampler_init_dist(e->seed));
    }

    e->clear_error();
    *out = reinterpret_cast<jetson_pi_mllm*>(e);
    return JETSON_PI_MLLM_OK;
}

void jetson_pi_mllm_close(jetson_pi_mllm * handle) {
    MllmEngine * e = reinterpret_cast<MllmEngine*>(handle);
    if (!e) return;
    if (e->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (e->smpl)  llama_sampler_free(e->smpl);
        if (e->mtmd)  mtmd_free(e->mtmd);
        if (e->lctx)  llama_free(e->lctx);
        if (e->model) llama_model_free(e->model);
        delete e;
    }
}

const char * jetson_pi_mllm_last_error(const jetson_pi_mllm * handle) {
    const MllmEngine * e = reinterpret_cast<const MllmEngine*>(handle);
    if (!e) return "null jetson_pi_mllm handle";
    return e->last_error.c_str();
}

const char * jetson_pi_mllm_open_error(void) {
    return g_open_error.c_str();
}

int32_t jetson_pi_mllm_infer(jetson_pi_mllm * handle,
                             const uint8_t * const * images_rgb,
                             uint32_t n_images,
                             uint32_t image_height, uint32_t image_width,
                             const char * prompt, size_t prompt_len,
                             char * out_buf, size_t out_capacity,
                             size_t * out_written) {
    MllmEngine * e = reinterpret_cast<MllmEngine*>(handle);
    if (!e) return JETSON_PI_MLLM_INVALID;
    e->clear_error();
    if (!prompt || !out_written) {
        return reject(e, JETSON_PI_MLLM_INVALID, "invalid infer arguments");
    }
    if (n_images > 0 && !images_rgb) {
        return reject(e, JETSON_PI_MLLM_INVALID, "images_rgb is null");
    }
    if (n_images > 0 &&
        (image_height == 0 || image_width == 0 ||
         static_cast<size_t>(image_width) >
             std::numeric_limits<size_t>::max() /
             static_cast<size_t>(image_height) / 3u)) {
        return reject(e, JETSON_PI_MLLM_INVALID,
                      "invalid or overflowing image dimensions");
    }

    if (!out_buf || out_capacity < e->max_output_bytes) {
        *out_written = e->max_output_bytes;
        return reject(e, JETSON_PI_MLLM_BUFFER_TOO_SMALL,
                      "out_buf too small for the loaded vocabulary bound");
    }

    int32_t status = jetson_pi_mllm_prefill(
        handle, images_rgb, n_images, image_height, image_width,
        prompt, prompt_len);
    if (status != JETSON_PI_MLLM_OK) return status;

    std::string out;
    out.reserve(static_cast<size_t>(e->max_tokens) * 4u);
    for (uint32_t i = 0; i < e->max_tokens; ++i) {
        int32_t token_id = 0;
        int32_t is_eog = 0;
        status = jetson_pi_mllm_decode_step(handle, &token_id, &is_eog);
        if (status != JETSON_PI_MLLM_OK) return status;
        if (is_eog) break;
        size_t piece_size = 0;
        status = jetson_pi_mllm_token_to_piece(handle, token_id, nullptr, 0,
                                               &piece_size);
        if (status != JETSON_PI_MLLM_BUFFER_TOO_SMALL) return status;
        if (piece_size == 0) continue;
        std::vector<char> piece(piece_size);
        status = jetson_pi_mllm_token_to_piece(
            handle, token_id, piece.data(), piece.size(), &piece_size);
        if (status != JETSON_PI_MLLM_OK) return status;
        out.append(piece.data(), piece_size);
    }

    const size_t need = out.size();
    *out_written = need;
    if (out_capacity < need) {
        return reject(e, JETSON_PI_MLLM_INFER_FAILED,
                      "generated text exceeded the vocabulary-derived bound");
    }
    std::memcpy(out_buf, out.data(), need);
    return JETSON_PI_MLLM_OK;
}

int32_t jetson_pi_mllm_reset(jetson_pi_mllm * handle) {
    MllmEngine * e = reinterpret_cast<MllmEngine*>(handle);
    if (!e) return JETSON_PI_MLLM_INVALID;
    e->clear_error();
    llama_memory_clear(llama_get_memory(e->lctx), true);
    llama_sampler_reset(e->smpl);
    e->session_ready = false;
    e->session_eog = false;
    e->generated_tokens = 0;
    return JETSON_PI_MLLM_OK;
}

int32_t jetson_pi_mllm_prefill(jetson_pi_mllm * handle,
                               const uint8_t * const * images_rgb,
                               uint32_t n_images,
                               uint32_t image_height, uint32_t image_width,
                               const char * prompt, size_t prompt_len) {
    MllmEngine * e = reinterpret_cast<MllmEngine*>(handle);
    if (!e || !prompt) return JETSON_PI_MLLM_INVALID;
    e->clear_error();
    if (n_images > 0 && !images_rgb) {
        return reject(e, JETSON_PI_MLLM_INVALID, "images_rgb is null");
    }
    if (n_images > 0 &&
        (image_height == 0 || image_width == 0 ||
         static_cast<size_t>(image_width) >
             std::numeric_limits<size_t>::max() /
             static_cast<size_t>(image_height) / 3u)) {
        return reject(e, JETSON_PI_MLLM_INVALID,
                      "invalid or overflowing image dimensions");
    }
    llama_memory_clear(llama_get_memory(e->lctx), true);
    llama_sampler_reset(e->smpl);
    e->session_ready = false;
    e->session_eog = false;
    e->generated_tokens = 0;

    // Build bitmaps from raw RGB (n_images may be 0 for text-only).
    std::vector<mtmd_bitmap*> bitmaps;
    bitmaps.reserve(n_images);
    for (uint32_t i = 0; i < n_images; ++i) {
        const uint8_t * rgb = images_rgb[i];
        if (!rgb) {
            free_bitmaps(bitmaps);
            return reject(e, JETSON_PI_MLLM_INVALID,
                          "images_rgb contained a null view");
        }
        mtmd_bitmap * bmp = mtmd_bitmap_init(image_width, image_height, rgb);
        if (!bmp) {
            free_bitmaps(bitmaps);
            return reject(e, JETSON_PI_MLLM_INFER_FAILED,
                          "mtmd_bitmap_init failed");
        }
        bitmaps.push_back(bmp);
    }

    // Tokenize: inject one media marker per image after the prompt text.
    // The mtmd tokenizer scans for markers and replaces them with image
    // embeddings. The caller's prompt must not already contain markers.
    // Note: marker position (before vs after prompt) is model-dependent;
    // callers who need precise placement should embed markers in the prompt
    // themselves (via the chat template) and set n_images=0.
    const char * marker = mtmd_default_marker();
    const size_t marker_len = std::strlen(marker);
    if (marker_len != 0 &&
        static_cast<size_t>(n_images) >
            (std::numeric_limits<size_t>::max() - prompt_len) / marker_len) {
        free_bitmaps(bitmaps);
        return reject(e, JETSON_PI_MLLM_INVALID,
                      "formatted multimodal prompt size overflows size_t");
    }
    std::string formatted;
    formatted.reserve(prompt_len + static_cast<size_t>(n_images) * marker_len);
    formatted.append(prompt, prompt_len);
    for (uint32_t i = 0; i < n_images; ++i) formatted += marker;

    mtmd_input_text text;
    text.text          = formatted.c_str();
    text.add_special   = true;   // first turn (KV was just cleared)
    text.parse_special = true;

    std::vector<const mtmd_bitmap*> bmp_ptrs(bitmaps.begin(), bitmaps.end());
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (!chunks) {
        free_bitmaps(bitmaps);
        return reject(e, JETSON_PI_MLLM_INFER_FAILED,
                      "mtmd_input_chunks_init returned null");
    }
    int32_t tr = mtmd_tokenize(e->mtmd, chunks, &text,
                               bmp_ptrs.data(), bmp_ptrs.size());
    free_bitmaps(bitmaps);
    if (tr != 0) {
        mtmd_input_chunks_free(chunks);
        return reject(e, JETSON_PI_MLLM_INFER_FAILED, "mtmd_tokenize failed");
    }

    // Evaluate image+text chunks (standard mtmd helper, NOT the Pi0 variant).
    llama_pos new_n_past = 0;
    int32_t er = mtmd_helper_eval_chunks(e->mtmd, e->lctx, chunks,
                                         /*n_past=*/0, /*seq_id=*/0,
                                         /*n_batch=*/2048,
                                         /*logits_last=*/true, &new_n_past);
    mtmd_input_chunks_free(chunks);
    if (er != 0) {
        return reject(e, JETSON_PI_MLLM_INFER_FAILED,
                      "mtmd_helper_eval_chunks failed");
    }
    e->session_ready = true;
    return JETSON_PI_MLLM_OK;
}

int32_t jetson_pi_mllm_decode_step(jetson_pi_mllm * handle,
                                   int32_t * token_id, int32_t * is_eog) {
    MllmEngine * e = reinterpret_cast<MllmEngine*>(handle);
    if (!e || !token_id || !is_eog) return JETSON_PI_MLLM_INVALID;
    e->clear_error();
    if (!e->session_ready || e->session_eog) {
        return reject(e, JETSON_PI_MLLM_BAD_STATE,
                      "decode_step requires an active prefilled session");
    }
    if (e->generated_tokens >= e->max_tokens) {
        return reject(e, JETSON_PI_MLLM_BAD_STATE,
                      "decode_step reached the configured max_tokens limit");
    }
    const llama_token token = llama_sampler_sample(e->smpl, e->lctx, -1);
    *token_id = static_cast<int32_t>(token);
    *is_eog = llama_vocab_is_eog(e->vocab, token) ? 1 : 0;
    e->generated_tokens += 1;
    if (*is_eog) {
        e->session_eog = true;
        return JETSON_PI_MLLM_OK;
    }
    llama_sampler_accept(e->smpl, token);
    llama_token decoded = token;
    llama_batch batch = llama_batch_get_one(&decoded, 1);
    if (int32_t rc = llama_decode(e->lctx, batch); rc < 0) {
        e->session_ready = false;
        return reject(e, JETSON_PI_MLLM_INFER_FAILED,
                      "llama_decode(token) failed");
    }
    return JETSON_PI_MLLM_OK;
}

int32_t jetson_pi_mllm_get_logits(jetson_pi_mllm * handle,
                                  float * logits_out, size_t logits_capacity,
                                  size_t * logits_written) {
    MllmEngine * e = reinterpret_cast<MllmEngine*>(handle);
    if (!e || !logits_written) return JETSON_PI_MLLM_INVALID;
    e->clear_error();
    if (!e->session_ready || e->session_eog) {
        return reject(e, JETSON_PI_MLLM_BAD_STATE,
                      "get_logits requires an active prefilled session");
    }
    const size_t n_vocab = static_cast<size_t>(llama_vocab_n_tokens(e->vocab));
    *logits_written = n_vocab;
    if (!logits_out || logits_capacity < n_vocab) {
        return reject(e, JETSON_PI_MLLM_BUFFER_TOO_SMALL,
                      "logits buffer too small");
    }
    const float * logits = llama_get_logits_ith(e->lctx, -1);
    if (!logits) {
        return reject(e, JETSON_PI_MLLM_INFER_FAILED,
                      "llama_get_logits_ith returned null");
    }
    std::memcpy(logits_out, logits, n_vocab * sizeof(float));
    return JETSON_PI_MLLM_OK;
}

int32_t jetson_pi_mllm_token_to_piece(jetson_pi_mllm * handle,
                                      int32_t token_id,
                                      char * piece_out, size_t piece_capacity,
                                      size_t * piece_written) {
    MllmEngine * e = reinterpret_cast<MllmEngine*>(handle);
    if (!e || !piece_written) return JETSON_PI_MLLM_INVALID;
    e->clear_error();
    int32_t needed = llama_token_to_piece(
        e->vocab, static_cast<llama_token>(token_id), nullptr, 0, 0, true);
    if (needed < 0) needed = -needed;
    *piece_written = static_cast<size_t>(needed);
    if (!piece_out || piece_capacity < static_cast<size_t>(needed)) {
        return reject(e, JETSON_PI_MLLM_BUFFER_TOO_SMALL,
                      "token piece buffer too small");
    }
    const int32_t written = llama_token_to_piece(
        e->vocab, static_cast<llama_token>(token_id), piece_out,
        needed, 0, true);
    if (written < 0) {
        return reject(e, JETSON_PI_MLLM_INFER_FAILED,
                      "llama_token_to_piece failed");
    }
    *piece_written = static_cast<size_t>(written);
    return JETSON_PI_MLLM_OK;
}

} // extern "C"
