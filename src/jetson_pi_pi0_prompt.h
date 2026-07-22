#ifndef JETSON_PI_PI0_PROMPT_H
#define JETSON_PI_PI0_PROMPT_H

#include <cmath>
#include <cstddef>
#include <string>

namespace jetson_pi_pi0_detail {

// state must contain n_state finite values. The public C entry point validates
// that contract before calling this formatter.
inline std::string format_pi05_openpi_prompt(const std::string & raw_text,
                                             const float * state,
                                             size_t n_state) {
    std::string out;
    out.reserve(raw_text.size() + 64);
    out += "Task: ";
    out += raw_text;
    out += ", State: ";

    for (size_t i = 0; i < n_state; ++i) {
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
        if (i > 0) out += ' ';
        out += std::to_string(bin);
    }

    out += ";\nAction: ";
    return out;
}

} // namespace jetson_pi_pi0_detail

#endif // JETSON_PI_PI0_PROMPT_H
