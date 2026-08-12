#include "llama.h"
#include "mtmd.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct input_dump {
    std::string prefix;
    std::map<std::string, int> indexes;
};

static bool capture_input(ggml_tensor * tensor, bool ask, void * opaque) {
    const std::string name = tensor->name;
    if (name != "inp_raw" && name != "patch_bias" && name != "inp_pos_emb" &&
            name.rfind("layer_out-", 0) != 0) return false;
    if (ask) return true;
    auto * dump = static_cast<input_dump *>(opaque);
    std::vector<float> values((size_t) ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    const int index = dump->indexes[name]++;
    std::ofstream output(dump->prefix + "." + name + "." + std::to_string(index) + ".f32",
            std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s MODEL MMPROJ processor_input_images.u8 OUTPUT.f32\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) return 3;
    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = true;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.warmup = false;
    input_dump dump{argv[4]};
    params.cb_eval = capture_input;
    params.cb_eval_user_data = &dump;
    mtmd_context * mtmd = mtmd_init_from_file(argv[2], model, params);
    if (!mtmd) return 4;

    constexpr int image_count = 4;
    constexpr int height = 256;
    constexpr int width = 455;
    std::vector<uint8_t> chw((size_t) image_count * 3 * height * width);
    std::ifstream input(argv[3], std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != (std::streamoff) chw.size()) return 5;
    input.seekg(0);
    input.read(reinterpret_cast<char *>(chw.data()), chw.size());
    std::vector<std::vector<uint8_t>> images(image_count,
            std::vector<uint8_t>((size_t) height * width * 3));
    for (int image = 0; image < image_count; ++image) {
        for (int channel = 0; channel < 3; ++channel) {
            for (int pixel = 0; pixel < height * width; ++pixel) {
                images[image][(size_t) pixel * 3 + channel] =
                    chw[((size_t) image * 3 + channel) * height * width + pixel];
            }
        }
    }
    std::vector<mtmd_bitmap *> bitmaps;
    std::vector<const mtmd_bitmap *> bitmap_ptrs;
    for (const auto & image : images) {
        bitmaps.push_back(mtmd_bitmap_init(width, height, image.data()));
        bitmap_ptrs.push_back(bitmaps.back());
    }
    std::string prompt;
    for (int i = 0; i < 4; ++i) {
        prompt += mtmd_default_marker();
        if (i != 3) prompt += "x"; // keep still images from being merged as video frames
    }
    mtmd_input_text text{prompt.c_str(), false, true};
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    int result = mtmd_tokenize(mtmd, chunks, &text, bitmap_ptrs.data(), bitmap_ptrs.size());
    for (auto * bitmap : bitmaps) mtmd_bitmap_free(bitmap);
    if (result != 0) return 5;

    std::ofstream output(argv[4], std::ios::binary | std::ios::trunc);
    size_t image_chunks = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
        const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        std::fprintf(stderr, "image %zu: %zu tokens\n", image_chunks, n_tokens);
        if (mtmd_encode_chunk(mtmd, chunk) != 0) return 6;
        output.write(reinterpret_cast<const char *>(mtmd_get_output_embd(mtmd)),
                (std::streamsize) (n_tokens * 8192 * sizeof(float)));
        ++image_chunks;
    }
    mtmd_input_chunks_free(chunks);
    mtmd_free(mtmd);
    llama_model_free(model);
    llama_backend_free();
    return image_chunks == 4 && output ? 0 : 7;
}
