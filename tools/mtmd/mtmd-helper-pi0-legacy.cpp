// Legacy PI0 mtmd helper implementation copied from Jetson-PI.
// This translation unit renames public helper symbols so Jetson-PI05 can
// dispatch to the original PI0 implementation without mixing the code paths.
#define mtmd_helper_log_set mtmd_helper_log_set_pi0_legacy
#define mtmd_helper_get_n_tokens mtmd_helper_get_n_tokens_pi0_legacy
#define mtmd_helper_get_n_pos mtmd_helper_get_n_pos_pi0_legacy
#define mtmd_helper_decode_image_chunk mtmd_helper_decode_image_chunk_pi0_legacy
#define mtmd_helper_eval_chunk_single mtmd_helper_eval_chunk_single_pi0_legacy
#define mtmd_pi0_result_free mtmd_pi0_result_free_pi0_legacy
#define mtmd_helper_eval_chunks_pi0 mtmd_helper_eval_chunks_pi0_legacy
#define mtmd_helper_eval_chunks mtmd_helper_eval_chunks_pi0_legacy_eval_chunks
#define mtmd_helper_bitmap_init_from_buf mtmd_helper_bitmap_init_from_buf_pi0_legacy
#define mtmd_helper_bitmap_init_from_file mtmd_helper_bitmap_init_from_file_pi0_legacy
#define g_logger g_logger_pi0_legacy
#define print_vector print_vector_pi0_legacy
#define ma_atomic_global_lock ma_atomic_global_lock_pi0_legacy

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "llama.h"
#include "clip-impl.h"
#include "clip.h"
#include "pi-model.h"

#include <algorithm>
#include <cinttypes>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>


static bool mtmd_is_pi0_model(mtmd_context *) {
    return true;
}

static void mtmd_pi0_clear_encode_cache(mtmd_context *) {}
static int32_t mtmd_pi0_preencode_image_chunks(mtmd_context *, const mtmd_input_chunks *) { return 0; }
static bool mtmd_pi0_apply_cached_chunk_embd(mtmd_context *, int32_t) { return false; }

static void mtmd_pi0_accumulate_vit_perf(mtmd_context * ctx, mtmd_pi0_result * pi0_result) {
    mtmd_pi0_result_apply_vit_perf(ctx, pi0_result);
}

static void pi0_legacy_debug_log_line(const std::string & line) {
    const char * path = pi_model_debug_dump_file();
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    std::ofstream out(path, std::ios::app);
    if (!out) {
        return;
    }
    out << "[mtmd-legacy] " << line << "\n";
}


static void pi0_legacy_debug_dump_floats(const char * name, const float * data, int64_t count, int64_t dim0 = 0, int64_t dim1 = 0) {
    if (!pi_model_debug_dump_enabled() || data == nullptr || count <= 0) {
        return;
    }
    const int64_t limit = std::min<int64_t>(count, pi_model_debug_dump_values(32));
    std::ostringstream oss;
    oss << "name=" << name << " count=" << count;
    if (dim0 > 0) {
        oss << " shape=[" << dim0;
        if (dim1 > 0) {
            oss << ", " << dim1;
        }
        oss << "]";
    }
    oss << " values=[";
    for (int64_t i = 0; i < limit; ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << data[i];
    }
    if (limit < count) {
        oss << ",...";
    }
    oss << "]";
    pi0_legacy_debug_log_line(oss.str());
}

static void pi0_legacy_free_combined_batch(llama_batch & batch) {
    // Legacy Jetson-PI points these at mtmd-owned VIT embedding buffers.
    // Jetson-PI05's llama_batch_free_pi0 frees embedding pointers, so detach them here.
    batch.embd = nullptr;
    batch.embd2 = nullptr;
    batch.embd3 = nullptr;
    llama_batch_free_pi0(batch);
}

//#define MTMD_AUDIO_DEBUG

#define MINIAUDIO_IMPLEMENTATION
#ifndef MTMD_AUDIO_DEBUG
#   define MA_NO_ENCODING
#endif
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

//
// internal logging functions
//

struct mtmd_helper_logger {
    ggml_log_callback default_callback = [](ggml_log_level level, const char * text, void * user_data) {
        (void) level;
        (void) user_data;
        fputs(text, stderr);
        fflush(stderr);
    };

    ggml_log_callback log_callback = default_callback;
    void * log_callback_user_data;

    void log_v(enum ggml_log_level level, const char * format, va_list args) {
        if (format == NULL) {
            return;
        }
        va_list args_copy;
        va_copy(args_copy, args);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            log_callback(level, buffer, log_callback_user_data);
        } else {
            char * buffer2 = (char *) calloc(len + 1, sizeof(char));
            vsnprintf(buffer2, len + 1, format, args_copy);
            buffer2[len] = 0;
            log_callback(level, buffer2, log_callback_user_data);
            free(buffer2);
        }
        va_end(args_copy);
    }

    void log(enum ggml_log_level level, const char * format, ...) {
        va_list args;
        va_start(args, format);
        log_v(level, format, args);
        va_end(args);
    }
} g_logger;

#define LOG_INF(...) g_logger.log(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) g_logger.log(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) g_logger.log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

void mtmd_helper_log_set(ggml_log_callback log_callback, void * user_data) {
    if (log_callback == nullptr) {
        log_callback = g_logger.default_callback;
    }
    g_logger.log_callback = log_callback;
    g_logger.log_callback_user_data = user_data;
    mtmd_log_set(log_callback, user_data);
}

//
// helper functions
//

size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks) {
    size_t n_tokens = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        n_tokens += mtmd_input_chunk_get_n_tokens(chunk);
    }
    return n_tokens;
}

llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks) {
    llama_pos n_pos = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        n_pos += mtmd_input_chunk_get_n_pos(chunk);
    }
    return n_pos;
}

// helper struct to make working with embd batch easier
// note: this will be removed after llama_batch_ext refactoring
struct decode_embd_batch {
    int n_pos_per_embd;
    int n_mmproj_embd;
    std::vector<llama_pos>      pos;
    std::vector<llama_pos>      pos_view; // used by mrope
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    decode_embd_batch(float * embd, int32_t n_tokens, int n_pos_per_embd, int n_mmproj_embd) : n_pos_per_embd(n_pos_per_embd), n_mmproj_embd(n_mmproj_embd) {
        pos     .resize(n_tokens * n_pos_per_embd);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_ids [n_tokens] = nullptr;
        batch = {};
        batch.n_tokens = n_tokens;
        batch.token    = nullptr;
        batch.embd     = embd;
        batch.pos      = pos.data();
        batch.n_seq_id = n_seq_id.data();
        batch.seq_id   = seq_ids.data();
        batch.logits   = logits.data();
        batch.embd2    = nullptr;
        batch.embd3    = nullptr;
        batch.img_token_num = 0;
        batch.single_img_token_num = 0;
    }

    void set_position_normal(llama_pos pos_0, llama_seq_id seq_id) {
        seq_id_0[0] = seq_id;
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.pos     [i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    // M-RoPE for image
    void set_position_mrope_2d(llama_pos pos_0, int nx, int ny, llama_seq_id seq_id) {
        GGML_ASSERT(n_pos_per_embd == 4);
        seq_id_0[0] = seq_id;
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                int i = y * nx + x;
                pos[i                     ] = pos_0;
                pos[i + batch.n_tokens    ] = pos_0 + y;
                pos[i + batch.n_tokens * 2] = pos_0 + x;
                pos[i + batch.n_tokens * 3] = 0; // last pos dim is unused
            }
        }
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    // M-RoPE for audio
    void set_position_mrope_1d(llama_pos pos_0, llama_seq_id seq_id) {
        GGML_ASSERT(n_pos_per_embd == 4);
        seq_id_0[0] = seq_id;
        for (int i = 0; i < batch.n_tokens; i++) {
            pos[i                     ] = pos_0 + i;
            pos[i + batch.n_tokens    ] = pos_0 + i;
            pos[i + batch.n_tokens * 2] = pos_0 + i;
            pos[i + batch.n_tokens * 3] = 0; // last pos dim is unused
        }
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    llama_batch get_view(int offset, int n_tokens) {
        llama_pos * pos_ptr;
        pos_view.clear();
        pos_view.reserve(n_tokens * n_pos_per_embd);
        if (n_pos_per_embd > 1) {
            // mrope
            // for example, with layout of src: 1234...1234...1234...1234...
            //       offset 2 will give us dst: 34...34...34...34...
            for (int i = 0; i < n_pos_per_embd; i++) {
                // assume n_tokens is less than or equal to batch.n_tokens
                // batch.n_tokens is number of **total** tokens
                // n_tokens is number of viewed token
                size_t src_idx = i * batch.n_tokens + offset;
                pos_view.insert(pos_view.end(),
                    pos.data() + src_idx,
                    pos.data() + src_idx + n_tokens);
            }
            pos_ptr = pos_view.data();
        } else {
            // normal
            pos_ptr = pos.data() + offset;
        }
        llama_batch view = {};
        view.n_tokens = n_tokens;
        view.token    = nullptr;
        view.embd     = batch.embd + offset * n_mmproj_embd;
        view.pos      = pos_ptr;
        view.n_seq_id = batch.n_seq_id + offset;
        view.seq_id   = batch.seq_id + offset;
        view.logits   = batch.logits + offset;
        view.embd2    = nullptr;
        view.embd3    = nullptr;
        view.img_token_num = 0;
        view.single_img_token_num = 0;
        return view;
    }
};

// Helper function for decoding an image whose embeddings have already been calculated
int32_t mtmd_helper_decode_image_chunk(
        mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        float * encoded_embd,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        llama_pos * new_n_past) {
    
    auto chunk_type = mtmd_input_chunk_get_type(chunk);
    const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_ERR("failed to decode chunk: input chunk not of image/audio type\n");
        return -1;
    }

    const llama_model * model = llama_get_model(lctx);
    int n_mmproj_embd = llama_model_n_embd_inp(model);
    int n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;

    int32_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    int32_t i_batch = 0;
    int32_t n_img_batches = GGML_PAD(n_tokens, n_batch) / n_batch;
    // printf("Debug: n_tokens = %d, n_batch = %d, n_mmproj_embd = %d, n_pos_per_embd = %d\n",
        // n_tokens, n_batch, n_mmproj_embd, n_pos_per_embd);
    // printf("Debug: embd pointer = %p\n", (void*)encoded_embd);
    decode_embd_batch batch_embd(encoded_embd, n_tokens, n_pos_per_embd, n_mmproj_embd);

    if (mtmd_decode_use_mrope(ctx)) {
        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const auto image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            if (!image_tokens) {
                LOG_ERR("failed to decode chunk: image tokens are null\n");
                return -1;
            }
            const int nx = mtmd_image_tokens_get_nx(image_tokens);
            const int ny = mtmd_image_tokens_get_ny(image_tokens);
            batch_embd.set_position_mrope_2d(n_past, nx, ny, seq_id);
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            batch_embd.set_position_mrope_1d(n_past, seq_id);
        } else {
            GGML_ABORT("invalid chunk type for M-RoPE");
        }
    } else {
        batch_embd.set_position_normal(n_past, seq_id);
    }

    if (mtmd_decode_use_non_causal(ctx, chunk)) {
        llama_set_causal_attn(lctx, false);
        // TODO @ngxson : need to make sure only one image is processed at a time, and n_ubatch must be enough to hold the image
    }
    // batch_embd.batch.logits[n_tokens - 1] = true;
    while (i_batch < n_img_batches) { // split into batches
        int pos_offset = i_batch*n_batch;
        int n_tokens_batch = std::min(n_batch, n_tokens - pos_offset);
        llama_batch batch_embd_view = batch_embd.get_view(pos_offset, n_tokens_batch);

        LOG_INF("decoding %s batch %d/%d, n_tokens_batch = %d\n", name, i_batch+1, n_img_batches, n_tokens_batch);
        
        int64_t t1 = ggml_time_ms();


        if (mtmd_is_pi0_model(ctx)) {
            if (llama_encode(lctx, batch_embd_view)) {
                LOG_ERR("%s : failed to eval\n", __func__);
                return 1;
            }

            int32_t ret = llama_decode(lctx, batch_embd_view);

            if (ret != 0) {
                LOG_ERR("failed to decode %s\n", name);
                llama_set_causal_attn(lctx, true); // restore causal attn
                return ret;
            }
            // llama_batch batch = llama_batch_init(params.max_length, 0, 1);
            // batch.n_tokens    = params.max_length;
            // for (int32_t step = 0; step < steps_per_block; step++) {
            //     int32_t global_step = block_num * steps_per_block + step;
            //     if (params.step_callback) {
            //         if (!params.step_callback(
            //                 global_step, params.steps, output_tokens, params.max_length, params.step_callback_user_data)) {
            //             break;
            //         }
            //     }
            //     // Setup batch
            //     for (int32_t i = 0; i < params.max_length; i++) {
            //         batch.token[i]     = output_tokens[i];
            //         batch.pos[i]       = i;
            //         batch.n_seq_id[i]  = 1;
            //         batch.seq_id[i][0] = 0;
            //         batch.logits[i]    = 1;
            //     }
            //     int ret = llama_decode(ctx, batch);
            // }
        }
        else{
            int32_t ret = llama_decode(lctx, batch_embd_view);
            if (ret != 0) {
                LOG_ERR("failed to decode %s\n", name);
                llama_set_causal_attn(lctx, true); // restore causal attn
                return ret;
            }
        }

        

        LOG_INF("%s decoded (batch %d/%d) in %" PRId64 " ms\n", name, i_batch+1, n_img_batches, ggml_time_ms() - t1);

        i_batch++;
    }

    n_past += mtmd_input_chunk_get_n_pos(chunk);
    *new_n_past = n_past;

    if (mtmd_decode_use_non_causal(ctx, chunk)) {
        llama_set_causal_attn(lctx, true);
    }
    return 0;
}

int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        bool logits_last,
        llama_pos * new_n_past) {
    int32_t ret;
    llama_batch text_batch = llama_batch_init(n_batch, 0, 1);
    auto chunk_type = mtmd_input_chunk_get_type(chunk);

    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        
        size_t n_tokens;
        const auto tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        // LOG_INF("decoding text chunk, n_tokens = %zu\n", n_tokens);
        size_t i = 0;
        while (i < n_tokens) { // split into batches
            text_batch.n_tokens = 0; // clear the batch
            for (; i < n_tokens && text_batch.n_tokens < n_batch; i++) {
                int32_t j = text_batch.n_tokens;
                text_batch.token   [j]    = tokens[i];
                text_batch.pos     [j]    = n_past++;
                text_batch.n_seq_id[j]    = 1;
                text_batch.seq_id  [j][0] = seq_id;
                text_batch.logits  [j]    = false;

                text_batch.n_tokens++;
            }
            
            bool is_last_token = (i == n_tokens);
            if (logits_last && is_last_token) {
                text_batch.logits[text_batch.n_tokens - 1] = true;
            }
            ret = llama_decode(lctx, text_batch);
            if (ret != 0) {
                LOG_ERR("failed to decode text\n");
                llama_batch_free(text_batch);
                return ret;
            }
            *new_n_past += text_batch.n_tokens;
        }
        

    } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
        int64_t t0 = ggml_time_ms();

        LOG_INF("encoding %s slice...\n", name);

        ret = mtmd_encode_chunk(ctx, chunk);

        if (ret != 0) {
            LOG_ERR("failed to encode %s slice\n", name);
            llama_batch_free(text_batch);
            return ret;
        }


        float * embd = mtmd_get_output_embd(ctx);

        ret = mtmd_helper_decode_image_chunk(ctx, lctx, chunk, embd, n_past, seq_id, n_batch, new_n_past);

        if (ret != 0) {
            LOG_ERR("failed to decode %s\n", name);
            llama_batch_free(text_batch);
            return ret;
        }

    } else {
        GGML_ABORT("chunk type not supported");
    }

    llama_batch_free(text_batch);
    return 0;
}

void print_vector(const std::vector<float>& vec, const std::string& name = "Vector") {
    std::cout << name << " [" << vec.size() << "]: ";
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

void mtmd_pi0_result_free(mtmd_pi0_result * result) {
    if (result == nullptr) {
        return;
    }

    if (result->state_data != nullptr) {
        free(result->state_data);
        result->state_data = nullptr;
    }

    if (result->action_data != nullptr) {
        free(result->action_data);
        result->action_data = nullptr;
    }

    if (result->action_final_data != nullptr) {
        free(result->action_final_data);
        result->action_final_data = nullptr;
    }

    result->has_state = false;
    result->state_dim = 0;
    result->has_action = false;
    result->action_dim = 0;
    result->action_steps = 0;
    result->has_action_final = false;
}

int32_t mtmd_helper_eval_chunks_pi0(mtmd_context * ctx,
    struct llama_context * lctx,
    const mtmd_input_chunks * chunks,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * old_n_past,
    mtmd_pi0_result * pi0_result) {
    n_past = 0;
    if (pi0_result) {
        mtmd_pi0_result_free(pi0_result);
        *pi0_result = mtmd_pi0_result {
            /* vit_ms           = */ 0.0,
            /* encode_ms        = */ 0.0,
            /* decode_ms        = */ 0.0,
            /* total_ms         = */ 0.0,
            /* batch_build_ms   = */ 0.0,
            /* output_extract_ms = */ 0.0,
            /* batch_free_ms    = */ 0.0,
            /* has_vit_ms       = */ false,
            /* has_encode_ms    = */ false,
            /* has_decode_ms    = */ false,
            /* has_total_ms     = */ false,
            /* has_batch_build_ms = */ false,
            /* has_output_extract_ms = */ false,
            /* has_batch_free_ms = */ false,
            /* state_dim        = */ 0,
            /* has_state        = */ false,
            /* state_data       = */ nullptr,
            /* action_dim       = */ 0,
            /* action_steps     = */ 0,
            /* has_action       = */ false,
            /* action_data      = */ nullptr,
            /* has_action_final = */ false,
            /* action_final_data = */ nullptr,
            /* vit_preprocess_ms = */ 0.0,
            /* vit_graph_build_alloc_ms = */ 0.0,
            /* vit_set_inputs_ms = */ 0.0,
            /* vit_graph_compute_ms = */ 0.0,
            /* vit_output_get_ms = */ 0.0,
            /* vit_graph_reused = */ false,
            /* has_vit_breakdown = */ false,
        };
    }

    size_t n_chunks = mtmd_input_chunks_size(chunks);
    if (n_chunks == 0) {
        LOG_WRN("no chunks to eval\n");
        return 0;
    }

    struct mtmd_pi0_encode_cache_guard {
        mtmd_context * ctx;
        ~mtmd_pi0_encode_cache_guard() { mtmd_pi0_clear_encode_cache(ctx); }
    } pi0_cache_guard { ctx };

    if (mtmd_is_pi0_model(ctx)) {
        const int32_t pre_ret = mtmd_pi0_preencode_image_chunks(ctx, chunks);
        if (pre_ret != 0) {
            return pre_ret;
        }
    }

    llama_batch combined_batch = llama_batch_init_pi0(n_batch, 0, 1);
    combined_batch.n_tokens = 0;
    std::vector<std::vector<float>> image_embd_storage;


    for (size_t ij = 0; ij < n_chunks; ij++) {
        bool chunk_logits_last = (ij == n_chunks - 1) && logits_last;
        auto chunk = mtmd_input_chunks_get(chunks, ij);










        // int32_t res = mtmd_helper_eval_chunk_single(ctx, lctx, chunk, n_past, seq_id, n_batch, chunk_logits_last, &n_past);
        llama_pos * new_n_past = &n_past;
        int32_t ret = 0;
        llama_batch text_batch = llama_batch_init(n_batch, 0, 1);
        auto chunk_type = mtmd_input_chunk_get_type(chunk);
    
        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            
            size_t n_tokens;
            const auto tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            // LOG_INF("decoding text chunk, n_tokens = %zu\n", n_tokens);
            size_t i = 0;
            while (i < n_tokens) { // split into batches
                text_batch.n_tokens = 0; // clear the batch
                
                text_batch.token   [0]    = 2;
                text_batch.pos     [0]    = n_past++;

                text_batch.n_seq_id[0]    = 1;
                text_batch.seq_id  [0][0] = seq_id;
                text_batch.logits  [0]    = false;

                text_batch.n_tokens++;

                int32_t combined_j = combined_batch.n_tokens;
                combined_batch.token   [combined_j]    = 2;
                combined_batch.pos     [combined_j]    = text_batch.pos[0];
                combined_batch.n_seq_id[combined_j]    = 1;
                combined_batch.seq_id  [combined_j][0] = seq_id;
                combined_batch.logits  [combined_j]    = false;

                combined_batch.n_tokens++;

                for (; i < n_tokens && text_batch.n_tokens < n_batch; i++) {
                    int32_t j = text_batch.n_tokens;
                    text_batch.token   [j]    = tokens[i];
                    text_batch.pos     [j]    = n_past++;

                    text_batch.n_seq_id[j]    = 1;
                    text_batch.seq_id  [j][0] = seq_id;
                    text_batch.logits  [j]    = false;
    
                    text_batch.n_tokens++;

                    int32_t combined_j = combined_batch.n_tokens;
                    combined_batch.token   [combined_j]    = tokens[i];
                    combined_batch.pos     [combined_j]    = text_batch.pos[j];
                    combined_batch.n_seq_id[combined_j]    = 1;
                    combined_batch.seq_id  [combined_j][0] = seq_id;
                    combined_batch.logits  [combined_j]    = false;

                    combined_batch.n_tokens++;

                }
                bool is_last_token = (i == n_tokens);
                if (chunk_logits_last && is_last_token) {
                    text_batch.logits[text_batch.n_tokens - 1] = true;
                    combined_batch.logits[combined_batch.n_tokens - 1] = true;
                }
    // struct llama_batch llama_batch_get_one(
    //     llama_token * tokens,
    //         int32_t   n_tokens) {
    //        return {
    //            /*n_tokens =*/ n_tokens,
    //            /*tokens   =*/ tokens,
    //            /*embd     =*/ nullptr,
    //            /*pos      =*/ nullptr,
    //            /*n_seq_id =*/ nullptr,
    //            /*seq_id   =*/ nullptr,
    //            /*logits   =*/ nullptr,
    //        };
    //    }     
                

                if (ret != 0) {
                    LOG_ERR("failed to decode text\n");
                    llama_batch_free(text_batch);
                    return ret;
                }
            }

            const int32_t combined_j = combined_batch.n_tokens;
            combined_batch.token   [combined_j]    = 108;
            combined_batch.pos     [combined_j]    = n_past++;
            combined_batch.n_seq_id[combined_j]    = 1;
            combined_batch.seq_id  [combined_j][0] = seq_id;
            combined_batch.logits  [combined_j]    = chunk_logits_last;
            combined_batch.n_tokens++;
    
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
            int64_t t0 = ggml_time_ms();
    
            LOG_INF("encoding %s slice...\n", name);
            auto t_vit_start = std::chrono::high_resolution_clock::now();
            if (mtmd_pi0_apply_cached_chunk_embd(ctx, ij)) {
                ret = 0;
                LOG_INF("%s slice using pre-encoded embedding (chunk %zu)\n", name, ij);
            } else {
                ret = mtmd_encode_chunk(ctx, chunk);
            }
            auto t_vit_end = std::chrono::high_resolution_clock::now();
            double vit_ms = std::chrono::duration<double, std::milli>(t_vit_end - t_vit_start).count();

            if (pi0_result) {
                pi0_result->vit_ms += vit_ms;
                pi0_result->has_vit_ms = true;
                mtmd_pi0_accumulate_vit_perf(ctx, pi0_result);
            }

            if (pi_model_env_truthy(std::getenv("LLAMA_PI0_PERF"))) {
                fprintf(stderr, "Vit took: %.2f ms\n", vit_ms);
            }

            if (ret != 0) {
                LOG_ERR("failed to encode %s slice\n", name);
                llama_batch_free(text_batch);
                return ret;
            }

            LOG_INF("%s slice encoded in %" PRId64 " ms\n", name, ggml_time_ms() - t0);

            float * embd = mtmd_get_output_embd(ctx);








            // ret = mtmd_helper_decode_image_chunk(ctx, lctx, chunk, embd, n_past, seq_id, n_batch, new_n_past);


            // int32_t mtmd_helper_decode_image_chunk(
            //     mtmd_context * ctx,
            //     struct llama_context * lctx,
            //     const mtmd_input_chunk * chunk,
            //     float * encoded_embd,
            //     llama_pos n_past,
            //     llama_seq_id seq_id,
            //     int32_t n_batch,
            //     llama_pos * new_n_past) {
            float * encoded_embd = embd;
            auto chunk_type = mtmd_input_chunk_get_type(chunk);
            if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                LOG_ERR("failed to decode chunk: input chunk not of image/audio type\n");
                return -1;
            }

            const llama_model * model = llama_get_model(lctx);
            int n_mmproj_embd = llama_model_n_embd_inp(model);
            int n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;
        
            int32_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
            int32_t i_batch = 0;
            int32_t n_img_batches = GGML_PAD(n_tokens, n_batch) / n_batch;

            decode_embd_batch batch_embd(encoded_embd, n_tokens, n_pos_per_embd, n_mmproj_embd);

            if (mtmd_decode_use_mrope(ctx)) {
                if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                    const auto image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
                    if (!image_tokens) {
                        LOG_ERR("failed to decode chunk: image tokens are null\n");
                        return -1;
                    }
                    const int nx = mtmd_image_tokens_get_nx(image_tokens);
                    const int ny = mtmd_image_tokens_get_ny(image_tokens);
                    batch_embd.set_position_mrope_2d(n_past, nx, ny, seq_id);
                } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
                    batch_embd.set_position_mrope_1d(n_past, seq_id);
                } else {
                    GGML_ABORT("invalid chunk type for M-RoPE");
                }
            } else {
                batch_embd.set_position_normal(n_past, seq_id);
            }
        
            if (mtmd_decode_use_non_causal(ctx, chunk)) {
                llama_set_causal_attn(lctx, false);
                // TODO @ngxson : need to make sure only one image is processed at a time, and n_ubatch must be enough to hold the image
            }

            // batch_embd.batch.logits[n_tokens - 1] = true;
            while (i_batch < n_img_batches) { // split into batches
                int pos_offset = i_batch*n_batch;
                int n_tokens_batch = std::min(n_batch, n_tokens);
                llama_batch batch_embd_view = batch_embd.get_view(pos_offset, n_tokens_batch);
                
                for (int i_image = 0; i_image < n_tokens_batch; i_image++) {
                    int32_t combined_j = combined_batch.n_tokens + i_image;
                    combined_batch.token   [combined_j]    = 0; // placeholder, will be filled during text chunk processing
                    combined_batch.pos     [combined_j]    = batch_embd_view.pos[i_image];
                    combined_batch.n_seq_id[combined_j]    = 1;
                    combined_batch.seq_id  [combined_j][0] = seq_id;
                    combined_batch.logits  [combined_j]    = false;
                    // printf("Debug: combined_batch.pos[%d] = %d\n", combined_j, combined_batch.pos[combined_j]);
                }
                
                combined_batch.n_tokens = batch_embd_view.n_tokens+combined_batch.n_tokens;
                combined_batch.img_token_num = combined_batch.img_token_num+ batch_embd_view.n_tokens;
                combined_batch.single_img_token_num = batch_embd_view.n_tokens;
                int img_num = 0;
                if (combined_batch.img_token_num>0){
                    img_num = combined_batch.img_token_num/ combined_batch.single_img_token_num;
                }
                image_embd_storage.emplace_back(
                    batch_embd_view.embd,
                    batch_embd_view.embd + (size_t) batch_embd_view.n_tokens * (size_t) n_mmproj_embd);
                float * stored_embd = image_embd_storage.back().data();
                if (img_num==1){
                    combined_batch.embd = stored_embd;
                }
                else{
                    if (img_num==2){
                        combined_batch.embd2 = stored_embd;
                    }
                    else{
                        if (img_num==3){
                            combined_batch.embd3 = stored_embd;
                        }
                        else{
                            GGML_ABORT("Not supported more than 3 embd buffers");
                        }
                    }
                }
                
                
                LOG_INF("decoding %s batch %d/%d, n_tokens_batch = %d\n", name, i_batch+1, n_img_batches, n_tokens_batch);

                int64_t t1 = ggml_time_ms();

        

                

        
                
        
                LOG_INF("%s decoded (batch %d/%d) in %" PRId64 " ms\n", name, i_batch+1, n_img_batches, ggml_time_ms() - t1);
        
                i_batch++;
            }

            n_past += mtmd_input_chunk_get_n_pos(chunk);
            *new_n_past = n_past;
        
            if (mtmd_decode_use_non_causal(ctx, chunk)) {
                llama_set_causal_attn(lctx, true);
            }
        








            if (ret != 0) {
                LOG_ERR("failed to decode %s\n", name);
                llama_batch_free(text_batch);
                return ret;
            }

        } else {
            GGML_ABORT("chunk type not supported");
        }
    
        llama_batch_free(text_batch);















        if (ret != 0) {
            LOG_ERR("failed to eval chunk %zu\n", ij);
            return ret;
        }
        *old_n_past = n_past;
    }

    // print_vector(std::vector<float>(combined_batch.embd, combined_batch.embd + 10), "Combined Embeddings image 0 token 1");
    // print_vector(std::vector<float>(combined_batch.embd + 255 * 2048, combined_batch.embd + 10 + 255 * 2048), "Combined Embeddings");
    

    // print_vector(std::vector<float>(combined_batch.pos, combined_batch.pos + combined_batch.n_tokens), "Combined Positions");
    // //print combined_batch.token
    // print_vector(std::vector<float>(combined_batch.token, combined_batch.token + combined_batch.n_tokens), "Combined Tokens");


    {
        std::ostringstream oss;
        oss << "name=pi0_legacy_mtmd_combined_batch"
            << " n_tokens=" << combined_batch.n_tokens
            << " image_tokens=" << combined_batch.img_token_num
            << " single_img_token_num=" << combined_batch.single_img_token_num
            << " text_tokens=" << (combined_batch.n_tokens - combined_batch.img_token_num)
            << " n_past=" << n_past
            << " n_chunks=" << n_chunks;
        pi0_legacy_debug_log_line(oss.str());
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto t_encode_start = std::chrono::high_resolution_clock::now();
    if (llama_encode(lctx, combined_batch)) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return 1;
    }
    
    auto t_encode_end = std::chrono::high_resolution_clock::now();
    double encode_ms = std::chrono::duration<double, std::milli>(t_encode_end - t_encode_start).count();

    auto t_decode_start = std::chrono::high_resolution_clock::now();
    int32_t ret = llama_decode(lctx, combined_batch);

    
    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    double total_ms = std::chrono::duration<double, std::milli>(t_decode_end - start_time).count();
    if (pi0_result) {
        pi0_result->encode_ms = encode_ms;
        pi0_result->decode_ms = decode_ms;
        pi0_result->total_ms = total_ms;
        pi0_result->has_encode_ms = true;
        pi0_result->has_decode_ms = true;
        pi0_result->has_total_ms = true;
    }
    if (pi_model_env_truthy(std::getenv("LLAMA_PI0_PERF"))) {
        fprintf(stderr, "Encode took: %.2f ms\n", encode_ms);
        fprintf(stderr, "Decode took: %.2f ms\n", decode_ms);
        fprintf(stderr, "Total processing time: %.2f ms\n", total_ms);
    }


    if (ret != 0) {
        LOG_ERR("failed to decode final\n");
        llama_set_causal_attn(lctx, true); // restore causal attn
        return ret;
    }

    if (pi0_result) {
        const float * state = llama_get_pi0_state(lctx);
        const int32_t state_dim = llama_get_pi0_state_size(lctx);
        if (state != nullptr && state_dim > 0) {
            pi0_result->state_data = (float *) malloc(sizeof(float) * state_dim);
            if (pi0_result->state_data == nullptr) {
                LOG_ERR("failed to allocate pi0 state buffer\n");
                pi0_legacy_free_combined_batch(combined_batch);
                return 1;
            }

            std::copy(state, state + state_dim, pi0_result->state_data);
            pi0_result->state_dim = state_dim;
            pi0_result->has_state = true;
            pi0_legacy_debug_dump_floats("pi0_legacy_result_state", state, state_dim, state_dim);
        }

        const float * action = llama_get_pi0_action_input(lctx);
        const float * action_final = llama_get_pi0_action(lctx);
        const int32_t action_dim = llama_get_pi0_action_dim(lctx);
        const int32_t action_steps = llama_get_pi0_action_steps(lctx);
        const int64_t action_count = (int64_t) action_dim * action_steps;

        if (action != nullptr && action_dim > 0 && action_steps > 0 && action_count > 0) {
            pi0_result->action_data = (float *) malloc(sizeof(float) * action_count);
            if (pi0_result->action_data == nullptr) {
                LOG_ERR("failed to allocate pi0 action buffer\n");
                pi0_legacy_free_combined_batch(combined_batch);
                return 1;
            }

            std::copy(action, action + action_count, pi0_result->action_data);
            pi0_result->action_dim = action_dim;
            pi0_result->action_steps = action_steps;
            pi0_result->has_action = true;
            pi0_legacy_debug_dump_floats("pi0_legacy_result_action_input", action, action_count, action_steps, action_dim);
        }

        if (action_final != nullptr && action_dim > 0 && action_steps > 0 && action_count > 0) {
            pi0_result->action_final_data = (float *) malloc(sizeof(float) * action_count);
            if (pi0_result->action_final_data == nullptr) {
                LOG_ERR("failed to allocate pi0 final action buffer\n");
                pi0_legacy_free_combined_batch(combined_batch);
                return 1;
            }

            std::copy(action_final, action_final + action_count, pi0_result->action_final_data);
            pi0_result->has_action_final = true;
            pi0_legacy_debug_dump_floats("pi0_legacy_result_action_final", action_final, action_count, action_steps, action_dim);
        }
    }
    pi0_legacy_free_combined_batch(combined_batch);

    return 0;
}




int32_t mtmd_helper_eval_chunks(mtmd_context * ctx,
                                struct llama_context * lctx,
                                const mtmd_input_chunks * chunks,
                                llama_pos n_past,
                                llama_seq_id seq_id,
                                int32_t n_batch,
                                bool logits_last,
                                llama_pos * new_n_past) {
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    if (n_chunks == 0) {
        LOG_WRN("no chunks to eval\n");
        return 0;
    }

    for (size_t i = 0; i < n_chunks; i++) {
        
        bool chunk_logits_last = (i == n_chunks - 1) && logits_last;
        auto chunk = mtmd_input_chunks_get(chunks, i);

        int32_t res = mtmd_helper_eval_chunk_single(ctx, lctx, chunk, n_past, seq_id, n_batch, chunk_logits_last, &n_past);
        if (res != 0) {
            LOG_ERR("failed to eval chunk %zu\n", i);
            return res;
        }
        *new_n_past = n_past;
    }

    return 0;
}

namespace audio_helpers {

static bool is_audio_file(const char * buf, size_t len) {
    if (len < 12) {
        return false;
    }

    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    bool is_wav = memcmp(buf, "RIFF", 4) == 0 && memcmp(buf + 8, "WAVE", 4) == 0;
    bool is_mp3 = len >= 3 && (
        memcmp(buf, "ID3", 3) == 0 ||
        // Check for MPEG sync word (simplified check)
        ((unsigned char)buf[0] == 0xFF && ((unsigned char)buf[1] & 0xE0) == 0xE0)
    );
    bool is_flac = memcmp(buf, "fLaC", 4) == 0;

    return is_wav || is_mp3 || is_flac;
}

// returns true if the buffer is a valid audio file
static bool decode_audio_from_buf(const unsigned char * buf_in, size_t len, int target_sampler_rate, std::vector<float> & pcmf32_mono) {
    ma_result result;
    const int channels = 1;
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, channels, target_sampler_rate);
    ma_decoder decoder;

    result = ma_decoder_init_memory(buf_in, len, &decoder_config, &decoder);
    if (result != MA_SUCCESS) {
        return false;
    }

    ma_uint64 frame_count;
    ma_uint64 frames_read;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

    pcmf32_mono.resize(frame_count);
    result = ma_decoder_read_pcm_frames(&decoder, pcmf32_mono.data(), frame_count, &frames_read);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

#ifdef MTMD_AUDIO_DEBUG
    // save audio to wav file
    ma_encoder_config config = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, target_sampler_rate);
    ma_encoder encoder;
    ma_encoder_init_file("output.wav", &config, &encoder);
    ma_encoder_write_pcm_frames(&encoder, pcmf32_mono.data(), pcmf32_mono.size(), &frames_read);
    ma_encoder_uninit(&encoder);
#endif

    ma_decoder_uninit(&decoder);
    return true;
}

} // namespace audio_helpers

mtmd_bitmap * mtmd_helper_bitmap_init_from_buf(mtmd_context * ctx, const unsigned char * buf, size_t len) {
    if (audio_helpers::is_audio_file((const char *)buf, len)) {
        std::vector<float> pcmf32;
        int bitrate = mtmd_get_audio_sample_rate(ctx);
        if (bitrate < 0) {
            LOG_ERR("This model does not support audio input\n");
            return nullptr;
        }
        if (!audio_helpers::decode_audio_from_buf(buf, len, bitrate, pcmf32)) {
            LOG_ERR("Unable to read WAV audio file from buffer\n");
            return nullptr;
        }
        return mtmd_bitmap_init_from_audio(pcmf32.size(), pcmf32.data());
    }

    // otherwise, we assume it's an image
    mtmd_bitmap * result = nullptr;
    {
        int nx, ny, nc;
        auto * data = stbi_load_from_memory(buf, len, &nx, &ny, &nc, 3);
        if (!data) {
            LOG_ERR("%s: failed to decode image bytes\n", __func__);
            return nullptr;
        }
        result = mtmd_bitmap_init(nx, ny, data);
        stbi_image_free(data);
    }
    return result;
}

mtmd_bitmap * mtmd_helper_bitmap_init_from_file(mtmd_context * ctx, const char * fname) {
    std::vector<unsigned char> buf;
    FILE * f = fopen(fname, "rb");
    if (!f) {
        LOG_ERR("Unable to open file %s: %s\n", fname, strerror(errno));
        return nullptr;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf.resize(file_size);

    size_t n_read = fread(buf.data(), 1, file_size, f);
    fclose(f);
    if (n_read != (size_t)file_size) {
        LOG_ERR("Failed to read entire file %s", fname);
        return nullptr;
    }

    return mtmd_helper_bitmap_init_from_buf(ctx, buf.data(), buf.size());
}
