#include "gr00t-input.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>

namespace gr00t_input {
namespace {

// Pixel-center bilinear interpolation. GR00T's two SmallestMaxSize operations
// use INTER_AREA; for the normal camera path both operations upscale, where
// OpenCV uses its linear resampling path.
std::vector<uint8_t> resize_rgb(const uint8_t * src, int sw, int sh, int dw, int dh) {
    std::vector<uint8_t> dst((size_t) dw * dh * 3);
    for (int y = 0; y < dh; ++y) {
        const float sy = ((y + 0.5f) * sh / dh) - 0.5f;
        const int y0 = std::max(0, std::min(sh - 1, (int) std::floor(sy)));
        const int y1 = std::min(sh - 1, y0 + 1);
        const float fy = std::max(0.0f, sy - std::floor(sy));
        for (int x = 0; x < dw; ++x) {
            const float sx = ((x + 0.5f) * sw / dw) - 0.5f;
            const int x0 = std::max(0, std::min(sw - 1, (int) std::floor(sx)));
            const int x1 = std::min(sw - 1, x0 + 1);
            const float fx = std::max(0.0f, sx - std::floor(sx));
            for (int c = 0; c < 3; ++c) {
                const float a = src[((size_t) y0 * sw + x0) * 3 + c] * (1 - fx) +
                                src[((size_t) y0 * sw + x1) * 3 + c] * fx;
                const float b = src[((size_t) y1 * sw + x0) * 3 + c] * (1 - fx) +
                                src[((size_t) y1 * sw + x1) * 3 + c] * fx;
                dst[((size_t) y * dw + x) * 3 + c] =
                    (uint8_t) std::max(0.0f, std::min(255.0f, std::round(a * (1 - fy) + b * fy)));
            }
        }
    }
    return dst;
}

} // namespace

bool preprocess_rgb(const uint8_t * rgb, int width, int height,
                    processed_image & output, std::string & error) {
    output = {};
    error.clear();
    if (!rgb || width <= 0 || height <= 0) {
        error = "invalid RGB image";
        return false;
    }

    constexpr int shortest = 256;
    const double first_scale = (double) shortest / std::min(width, height);
    const int resized_w = (int) std::round(width * first_scale);
    const int resized_h = (int) std::round(height * first_scale);
    std::vector<uint8_t> resized = resize_rgb(rgb, width, height, resized_w, resized_h);

    const int crop_w = std::max(1, (int) (resized_w * 0.95));
    const int crop_h = std::max(1, (int) (resized_h * 0.95));
    const int x0 = (resized_w - crop_w) / 2;
    const int y0 = (resized_h - crop_h) / 2;
    std::vector<uint8_t> crop((size_t) crop_w * crop_h * 3);
    for (int y = 0; y < crop_h; ++y) {
        std::memcpy(crop.data() + (size_t) y * crop_w * 3,
                    resized.data() + ((size_t) (y + y0) * resized_w + x0) * 3,
                    (size_t) crop_w * 3);
    }

    const double second_scale = (double) shortest / std::min(crop_w, crop_h);
    output.width = (int) std::round(crop_w * second_scale);
    output.height = (int) std::round(crop_h * second_scale);
    output.rgb = resize_rgb(crop.data(), crop_w, crop_h, output.width, output.height);
    return true;
}

std::string formalize_instruction(const char * text, size_t size) {
    std::string result;
    if (!text) return result;
    result.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        const unsigned char ch = (unsigned char) text[i];
        if (ch >= 0x80) result.push_back((char) ch);
        else if (std::isalnum(ch) || ch == '_' || std::isspace(ch))
            result.push_back((char) std::tolower(ch));
    }
    return result;
}

std::string build_prompt(const char * instruction, size_t instruction_size,
                         size_t image_count, const char * image_marker) {
    std::string prompt = "<|im_start|>user\n";
    const char * marker = image_marker ? image_marker : "";
    for (size_t i = 0; i < image_count; ++i) prompt += marker;
    prompt += formalize_instruction(instruction, instruction_size);
    prompt += "<|im_end|>\n";
    return prompt;
}

bool prepare_state(const float * state, size_t state_size, size_t state_width,
                   std::vector<float> & output, std::string & error) {
    output.clear();
    error.clear();
    if (state_width == 0 || (!state && state_size)) {
        error = "invalid GR00T state";
        return false;
    }
    if (state_size > state_width) {
        error = "state exceeds GR00T state width";
        return false;
    }
    output.assign(state_width, 0.0f);
    for (size_t i = 0; i < state_size; ++i) {
        if (!std::isfinite(state[i])) {
            output.clear();
            error = "state must be finite";
            return false;
        }
        output[i] = state[i];
    }
    return true;
}

} // namespace gr00t_input
