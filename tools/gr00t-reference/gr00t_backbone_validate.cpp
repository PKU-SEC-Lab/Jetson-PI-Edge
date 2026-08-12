#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

struct callback_data {
    std::string output_dir;
};

static bool capture_layers(ggml_tensor * tensor, bool ask, void * opaque) {
    const char * name = tensor->name;
    if (name == nullptr || std::string(name).rfind("l_out-", 0) != 0) {
        return false;
    }
    if (ask) return true;
    auto * data = static_cast<callback_data *>(opaque);
    const int64_t count = ggml_nelements(tensor);
    std::vector<float> output((size_t) count);
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, output.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> raw((size_t) count);
        ggml_backend_tensor_get(tensor, raw.data(), 0, ggml_nbytes(tensor));
        ggml_bf16_to_fp32_row(raw.data(), output.data(), count);
    } else {
        std::fprintf(stderr, "unsupported callback tensor type for %s: %s\n",
                name, ggml_type_name(tensor->type));
        return true;
    }
    std::ofstream file(data->output_dir + "/" + name + ".f32", std::ios::binary);
    file.write(reinterpret_cast<const char *>(output.data()), output.size() * sizeof(float));
    return true;
}

template<typename T>
static bool read_exact(const char * path, std::vector<T> & data) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != (std::streamoff) (data.size() * sizeof(T))) {
        std::fprintf(stderr, "invalid fixture file: %s\n", path);
        return false;
    }
    input.seekg(0);
    input.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(T));
    return bool(input);
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s MODEL language_input.f32 language_positions.i32 LAYER_OUTPUT_DIR\n", argv[0]);
        return 2;
    }
    constexpr int32_t n_tokens = 473;
    constexpr int32_t n_embd_input = 8192;
    std::vector<float> embeddings((size_t) n_tokens * n_embd_input);
    std::vector<int32_t> positions((size_t) n_tokens * 4);
    if (!read_exact(argv[2], embeddings) || !read_exact(argv[3], positions)) {
        return 2;
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) return 3;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    cparams.embeddings = false;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    callback_data cb_data{argv[4]};
    cparams.cb_eval = capture_layers;
    cparams.cb_eval_user_data = &cb_data;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return 4;

    std::vector<int32_t> n_seq_id(n_tokens, 1);
    std::vector<llama_seq_id> sequence(n_tokens, 0);
    std::vector<llama_seq_id *> sequence_ptr(n_tokens);
    std::vector<int8_t> logits(n_tokens, 0);
    for (int32_t i = 0; i < n_tokens; ++i) sequence_ptr[i] = &sequence[i];
    llama_batch batch = {};
    batch.n_tokens = n_tokens;
    batch.embd = embeddings.data();
    batch.pos = positions.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id = sequence_ptr.data();
    batch.logits = logits.data();
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.logits[i] = i == n_tokens - 1;
    }
    const int result = llama_decode(ctx, batch);
    const int action_result = result == 0 ? llama_gr00t_generate_action(ctx) : -1;
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return result == 0 && action_result == 0 ? 0 : 5;
}
