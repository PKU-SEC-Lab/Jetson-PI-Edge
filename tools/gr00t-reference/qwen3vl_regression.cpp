#include "jetson_pi_mllm.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s MODEL MMPROJ BACKEND\n", argv[0]);
        return 2;
    }
    jetson_pi_mllm_config config{};
    config.struct_size = sizeof(config);
    config.model_path = argv[1];
    config.mmproj_path = argv[2];
    config.backend = argv[3];
    config.n_ctx = 2048;
    config.temp = 0.0f;
    config.max_tokens = 8;
    jetson_pi_mllm * engine = nullptr;
    if (jetson_pi_mllm_open(&config, &engine) != 0) {
        std::fprintf(stderr, "open: %s\n", jetson_pi_mllm_open_error());
        return 3;
    }
    constexpr uint32_t width = 320, height = 180;
    std::vector<uint8_t> image((size_t) width * height * 3);
    for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width; ++x) {
        const size_t p = ((size_t) y * width + x) * 3;
        image[p] = (uint8_t) x; image[p + 1] = (uint8_t) (2 * y); image[p + 2] = (uint8_t) (x + y);
    }
    const uint8_t * images[] = {image.data()};
    const char prompt[] = "<|im_start|>user\nDescribe the image briefly.<|im_end|>\n<|im_start|>assistant\n";
    std::vector<float> first(200000), second(200000);
    size_t n_first = 0, n_second = 0;
    int r1 = jetson_pi_mllm_prefill(engine, images, 1, height, width,
            prompt, sizeof(prompt) - 1);
    if (r1 == 0) r1 = jetson_pi_mllm_get_logits(engine, first.data(), first.size(), &n_first);
    int r2 = r1 == 0 ? jetson_pi_mllm_prefill(engine, images, 1, height, width,
            prompt, sizeof(prompt) - 1) : r1;
    if (r2 == 0) r2 = jetson_pi_mllm_get_logits(engine, second.data(), second.size(), &n_second);
    if (r1 != 0 || r2 != 0) {
        std::fprintf(stderr, "infer: %d/%d %s\n", r1, r2, jetson_pi_mllm_last_error(engine));
        jetson_pi_mllm_close(engine);
        return 4;
    }
    size_t non_finite = 0;
    double max_diff = 0.0;
    for (size_t i = 0; i < n_first; ++i) {
        non_finite += !std::isfinite(first[i]);
        max_diff = std::max(max_diff, std::abs((double) first[i] - second[i]));
    }
    const bool repeat = n_first == n_second && max_diff == 0.0;
    std::printf("logits=%zu finite=%s repeat=%s max_repeat_diff=%.9g\n",
            n_first, non_finite ? "false" : "true", repeat ? "true" : "false", max_diff);
    jetson_pi_mllm_close(engine);
    return n_first && !non_finite && repeat ? 0 : 5;
}
