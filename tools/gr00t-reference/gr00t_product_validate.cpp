#include "jetson_pi_gr00t.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

template<class T>
static bool read_exact(const char * path, std::vector<T> & values) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != (std::streamoff) (values.size() * sizeof(T))) return false;
    input.seekg(0);
    input.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(T));
    return bool(input);
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s MODEL MMPROJ state.f32 BACKEND action.f32\n", argv[0]);
        return 2;
    }
    constexpr int width = 320, height = 180, n_images = 4;
    std::vector<std::vector<uint8_t>> images(n_images,
            std::vector<uint8_t>((size_t) width * height * 3));
    for (int camera = 0; camera < 2; ++camera) {
        for (int timestep = 0; timestep < 2; ++timestep) {
            auto & image = images[camera * 2 + timestep];
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const size_t p = ((size_t) y * width + x) * 3;
                    image[p + 0] = (uint8_t) (x + 17 * camera + 29 * timestep);
                    image[p + 1] = (uint8_t) (y * 3 + 31 * camera + 11 * timestep);
                    image[p + 2] = (uint8_t) ((x / 2) + (y / 3) + 47 * camera + 7 * timestep);
                }
            }
        }
    }
    std::vector<const uint8_t *> image_ptrs;
    for (const auto & image : images) image_ptrs.push_back(image.data());
    std::vector<float> state(132);
    if (!read_exact(argv[3], state)) return 3;

    jetson_pi_gr00t_config config{};
    config.struct_size = sizeof(config);
    config.model_path = argv[1];
    config.mmproj_path = argv[2];
    config.backend = argv[4];
    config.n_images = n_images;
    config.image_width = width;
    config.image_height = height;
    config.embodiment_id = 24; // oxe_droid_relative_eef_relative_joint
    jetson_pi_gr00t * engine = nullptr;
    if (jetson_pi_gr00t_open(&config, &engine) != 0) {
        std::fprintf(stderr, "open: %s\n", jetson_pi_gr00t_open_error());
        return 4;
    }
    const char instruction[] = "Pick up the red block, then place it beside the blue block.";
    std::vector<float> action(40 * 132);
    size_t written = 0;
    const int result = jetson_pi_gr00t_infer(engine, image_ptrs.data(), n_images,
            instruction, sizeof(instruction) - 1, state.data(), state.size(),
            action.data(), action.size(), &written);
    if (result != 0) std::fprintf(stderr, "infer: %s\n", jetson_pi_gr00t_last_error(engine));
    jetson_pi_gr00t_close(engine);
    if (result != 0 || written != action.size() ||
            !std::all_of(action.begin(), action.end(), [](float value) { return std::isfinite(value); })) {
        return 5;
    }
    std::ofstream output(argv[5], std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(action.data()), action.size() * sizeof(float));
    return output ? 0 : 6;
}
