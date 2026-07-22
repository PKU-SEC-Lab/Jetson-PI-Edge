#ifndef JETSON_PI_PI0_PROMPT_H
#define JETSON_PI_PI0_PROMPT_H

#include <cmath>
#include <cstddef>
#include <string>

namespace jetson_pi_pi0_detail {

inline std::string format_pi05_openpi_prompt(const std::string & raw_text,
                                             const float * state,
                                             size_t n_state) {
    std::string out = "Task: ";
    out += raw_text;
    out += ", State: ";

    const size_t n_prompt_state = n_state < 8 ? n_state : 8;
    for (size_t i = 0; i < n_prompt_state; ++i) {
        const float x = state[i];
        int bin;
        if (x < -1.0f) {
            bin = -1;
        } else if (x >= 1.0f) {
            bin = 255;
        } else {
            bin = static_cast<int>(std::floor((x + 1.0f) * 128.0f));
            if (bin < 0) bin = 0;
            if (bin > 255) bin = 255;
        }
        if (i != 0) out += ' ';
        out += std::to_string(bin);
    }
    for (size_t i = n_prompt_state; i < 8; ++i) {
        if (i != 0) out += ' ';
        out += "128";
    }
    out += ";\nAction: ";
    return out;
}

} // namespace jetson_pi_pi0_detail

#endif
