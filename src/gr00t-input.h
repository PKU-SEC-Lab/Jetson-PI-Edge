#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gr00t_input {

struct processed_image {
    std::vector<uint8_t> rgb;
    int width = 0;
    int height = 0;
};

// Applies the GR00T camera transform before mtmd/Qwen3-VL smart resize:
// shortest-edge resize to 256, 95% center crop, shortest-edge resize to 256.
bool preprocess_rgb(const uint8_t * rgb, int width, int height,
                    processed_image & output, std::string & error);

// Matches the instruction normalization used by the upstream GR00T policy.
// ASCII punctuation is removed and ASCII letters are lower-cased. Non-ASCII
// UTF-8 bytes are preserved.
std::string formalize_instruction(const char * text, size_t size);

// Builds the exact Qwen chat prompt consumed by the GR00T multimodal path.
std::string build_prompt(const char * instruction, size_t instruction_size,
                         size_t image_count, const char * image_marker);

// Validates a caller-provided state and pads it to the model state width.
bool prepare_state(const float * state, size_t state_size, size_t state_width,
                   std::vector<float> & output, std::string & error);

} // namespace gr00t_input
