#include "jetson_pi_pi0.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 8) {
        std::fprintf(stderr, "usage: %s MODEL MMPROJ BACKEND N_VIEWS WIDTH HEIGHT PROMPT\n", argv[0]);
        return 2;
    }
    const uint32_t n_views = (uint32_t) std::strtoul(argv[4], nullptr, 10);
    const uint32_t width = (uint32_t) std::strtoul(argv[5], nullptr, 10);
    const uint32_t height = (uint32_t) std::strtoul(argv[6], nullptr, 10);
    std::vector<std::vector<uint8_t>> images(n_views,
            std::vector<uint8_t>((size_t) width * height * 3));
    std::vector<const uint8_t *> ptrs;
    for (uint32_t view = 0; view < n_views; ++view) {
        for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width; ++x) {
            const size_t p = ((size_t) y * width + x) * 3;
            images[view][p + 0] = (uint8_t) (x + 13 * view);
            images[view][p + 1] = (uint8_t) (3 * y + 29 * view);
            images[view][p + 2] = (uint8_t) (x / 2 + y / 3 + 41 * view);
        }
        ptrs.push_back(images[view].data());
    }

    jetson_pi_pi0_config config{};
    config.struct_size = sizeof(config);
    config.model_path = argv[1];
    config.mmproj_path = argv[2];
    config.backend = argv[3];
    config.n_views = n_views;
    config.image_width = width;
    config.image_height = height;
    jetson_pi_pi0 * engine = nullptr;
    const int opened = jetson_pi_pi0_open(&config, &engine);
    if (opened != 0) {
        std::fprintf(stderr, "open failed (%d): %s\n", opened, jetson_pi_pi0_open_error());
        return 3;
    }
    uint32_t steps = 0, dim = 0;
    if (jetson_pi_pi0_action_shape(engine, &steps, &dim) != 0 || !steps || !dim) return 4;
    std::vector<float> state(dim, 0.0f), first((size_t) steps * dim), second(first.size());
    size_t written_first = 0, written_second = 0;
    const size_t prompt_len = std::char_traits<char>::length(argv[7]);
    const int r1 = jetson_pi_pi0_infer(engine, ptrs.data(), n_views, argv[7], prompt_len,
            state.data(), state.size(), first.data(), first.size(), &written_first);
    const int r2 = r1 == 0 ? jetson_pi_pi0_infer(engine, ptrs.data(), n_views, argv[7], prompt_len,
            state.data(), state.size(), second.data(), second.size(), &written_second) : r1;
    if (r1 != 0 || r2 != 0) {
        std::fprintf(stderr, "infer failed (%d/%d): %s\n", r1, r2, jetson_pi_pi0_last_error(engine));
        jetson_pi_pi0_close(engine);
        return 5;
    }
    size_t non_finite = 0;
    double max_repeat_diff = 0.0, mean = 0.0;
    for (size_t i = 0; i < first.size(); ++i) {
        non_finite += !std::isfinite(first[i]);
        mean += first[i];
        max_repeat_diff = std::max(max_repeat_diff, std::abs((double) first[i] - second[i]));
    }
    mean /= first.size();
    std::printf("shape=[%u,%u] written=%zu finite=%s mean=%.9g max_repeat_diff=%.9g\n",
            steps, dim, written_first, non_finite ? "false" : "true", mean, max_repeat_diff);
    jetson_pi_pi0_close(engine);
    return written_first == first.size() && written_second == second.size() && !non_finite &&
        max_repeat_diff == 0.0 ? 0 : 6;
}
