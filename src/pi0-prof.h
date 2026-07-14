#pragma once

#include "pi0-build-config.h"

#if PI0_ENABLE_REUSE_LOG

#include <fstream>
#include <sstream>
#include <string>

inline void pi0_prof_append_log_line(const std::string & line) {
    std::ofstream ofs("pi0_ae_reuse.log", std::ios::app);
    if (!ofs.is_open()) {
        return;
    }
    ofs << line << '\n';
}

#define PI0_REUSE_LOG_LINE(line) \
    do { \
        ::pi0_prof_append_log_line((line)); \
    } while (0)

#define PI0_REUSE_LOG_STREAM(expr) \
    do { \
        std::ostringstream pi0_prof_oss__; \
        pi0_prof_oss__ << expr; \
        ::pi0_prof_append_log_line(pi0_prof_oss__.str()); \
    } while (0)

#else

#define PI0_REUSE_LOG_LINE(line) \
    do { \
    } while (0)

#define PI0_REUSE_LOG_STREAM(expr) \
    do { \
    } while (0)

#endif
