#pragma once

#if defined(__has_include)
#  if __has_include("pi0-build-config.h")
#    include "pi0-build-config.h"
#  else
#    define PI0_ENABLE_NVTX 0
#  endif
#else
#  define PI0_ENABLE_NVTX 0
#endif

#include <string>

#if PI0_ENABLE_NVTX && defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define LLAMA_NVTX_AVAILABLE 1
#  elif __has_include(<nvToolsExt.h>)
#    include <nvToolsExt.h>
#    define LLAMA_NVTX_AVAILABLE 1
#  else
#    define LLAMA_NVTX_AVAILABLE 0
#  endif
#else
#  define LLAMA_NVTX_AVAILABLE 0
#endif

#if LLAMA_NVTX_AVAILABLE

struct llama_nvtx_range {
    explicit llama_nvtx_range(const char * name) {
        nvtxRangePushA(name);
    }

    explicit llama_nvtx_range(const std::string & name) : llama_nvtx_range(name.c_str()) {
    }

    ~llama_nvtx_range() {
        nvtxRangePop();
    }

    llama_nvtx_range(const llama_nvtx_range &) = delete;
    llama_nvtx_range & operator=(const llama_nvtx_range &) = delete;
};

#define PI0_NVTX_SCOPE(name) \
    ::llama_nvtx_range pi0_nvtx_range_##__LINE__ (name)

#else

struct llama_nvtx_range {
    explicit llama_nvtx_range(const char * name) {
        (void) name;
    }

    explicit llama_nvtx_range(const std::string & name) {
        (void) name;
    }
};

#define PI0_NVTX_SCOPE(name) \
    do { \
        (void) sizeof(name); \
    } while (0)

#endif
