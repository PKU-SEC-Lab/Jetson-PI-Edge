#pragma once

#include "pi-model.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <vector>

enum pi_model_file_kind {
    PI_MODEL_FILE_UNKNOWN = 0,
    PI_MODEL_FILE_LLM,
    PI_MODEL_FILE_MMPROJ,
};

struct pi_model_detect_result {
    pi_model_kind kind = PI_MODEL_AUTO;
    pi_model_file_kind file_kind = PI_MODEL_FILE_UNKNOWN;
    std::string reason;
    std::string general_name;
    std::string general_architecture;
};

namespace pi_model_detect_detail {

static constexpr uint32_t GGUF_TYPE_UINT8   = 0;
static constexpr uint32_t GGUF_TYPE_INT8    = 1;
static constexpr uint32_t GGUF_TYPE_UINT16  = 2;
static constexpr uint32_t GGUF_TYPE_INT16   = 3;
static constexpr uint32_t GGUF_TYPE_UINT32  = 4;
static constexpr uint32_t GGUF_TYPE_INT32   = 5;
static constexpr uint32_t GGUF_TYPE_FLOAT32 = 6;
static constexpr uint32_t GGUF_TYPE_BOOL    = 7;
static constexpr uint32_t GGUF_TYPE_STRING  = 8;
static constexpr uint32_t GGUF_TYPE_ARRAY   = 9;
static constexpr uint32_t GGUF_TYPE_UINT64  = 10;
static constexpr uint32_t GGUF_TYPE_INT64   = 11;
static constexpr uint32_t GGUF_TYPE_FLOAT64 = 12;

struct gguf_probe {
    std::string general_name;
    std::string general_architecture;
    bool has_clip_vision = false;
    bool clip_use_gelu_known = false;
    bool clip_use_gelu = false;
    std::vector<std::string> tensor_names;
};

static inline std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return value;
}

static inline bool contains_ci(const std::string & value, const std::string & needle) {
    return lower_ascii(value).find(lower_ascii(needle)) != std::string::npos;
}

static inline bool is_pi_llm_architecture(const std::string & arch) {
    const std::string lower = lower_ascii(arch);
    return lower == "pi0" || lower == "pi05" || lower == "pi0.5";
}

template <typename T>
static inline bool read_pod(std::ifstream & in, T & value) {
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    return bool(in);
}

static inline bool read_string(std::ifstream & in, std::string & out) {
    uint64_t size = 0;
    if (!read_pod(in, size)) {
        return false;
    }
    out.assign((size_t) size, '\0');
    if (size > 0) {
        in.read(&out[0], (std::streamsize) size);
    }
    return bool(in);
}

static inline bool skip_bytes(std::ifstream & in, uint64_t size) {
    in.seekg((std::streamoff) size, std::ios::cur);
    return bool(in);
}

static inline uint64_t scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return 8;
        default:
            return 0;
    }
}

static inline bool skip_scalar(std::ifstream & in, uint32_t type);

static inline bool skip_value(std::ifstream & in, uint32_t type) {
    if (type == GGUF_TYPE_ARRAY) {
        uint32_t item_type = 0;
        uint64_t count = 0;
        if (!read_pod(in, item_type) || !read_pod(in, count)) {
            return false;
        }
        for (uint64_t i = 0; i < count; ++i) {
            if (!skip_scalar(in, item_type)) {
                return false;
            }
        }
        return true;
    }
    return skip_scalar(in, type);
}

static inline bool skip_scalar(std::ifstream & in, uint32_t type) {
    if (type == GGUF_TYPE_STRING) {
        std::string ignored;
        return read_string(in, ignored);
    }
    const uint64_t size = scalar_size(type);
    return size > 0 && skip_bytes(in, size);
}

static inline bool read_bool_value(std::ifstream & in, uint32_t type, bool & out) {
    if (type != GGUF_TYPE_BOOL) {
        return skip_value(in, type);
    }
    uint8_t value = 0;
    if (!read_pod(in, value)) {
        return false;
    }
    out = value != 0;
    return true;
}

static inline bool read_string_value(std::ifstream & in, uint32_t type, std::string & out) {
    if (type != GGUF_TYPE_STRING) {
        return skip_value(in, type);
    }
    return read_string(in, out);
}

static inline bool read_gguf_probe(const std::string & path, gguf_probe & out, std::string & error) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        error = "failed to open GGUF";
        return false;
    }

    char magic[4] = {};
    in.read(magic, sizeof(magic));
    if (!in || std::string(magic, magic + 4) != "GGUF") {
        error = "not a GGUF file";
        return false;
    }

    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    if (!read_pod(in, version) || !read_pod(in, tensor_count) || !read_pod(in, kv_count)) {
        error = "failed to read GGUF header";
        return false;
    }

    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!read_string(in, key) || !read_pod(in, type)) {
            error = "failed to read GGUF metadata";
            return false;
        }

        if (key == "general.name") {
            if (!read_string_value(in, type, out.general_name)) {
                error = "failed to read general.name";
                return false;
            }
        } else if (key == "general.architecture") {
            if (!read_string_value(in, type, out.general_architecture)) {
                error = "failed to read general.architecture";
                return false;
            }
        } else if (key == "clip.has_vision_encoder") {
            if (!read_bool_value(in, type, out.has_clip_vision)) {
                error = "failed to read clip.has_vision_encoder";
                return false;
            }
        } else if (key == "clip.use_gelu") {
            if (!read_bool_value(in, type, out.clip_use_gelu)) {
                error = "failed to read clip.use_gelu";
                return false;
            }
            out.clip_use_gelu_known = true;
        } else if (!skip_value(in, type)) {
            error = "failed to skip GGUF metadata";
            return false;
        }
    }

    out.tensor_names.reserve((size_t) tensor_count);
    for (uint64_t i = 0; i < tensor_count; ++i) {
        std::string name;
        uint32_t n_dims = 0;
        if (!read_string(in, name) || !read_pod(in, n_dims)) {
            error = "failed to read tensor info";
            return false;
        }
        if (!skip_bytes(in, 8ull * n_dims) || !skip_bytes(in, 4) || !skip_bytes(in, 8)) {
            error = "failed to skip tensor info";
            return false;
        }
        out.tensor_names.push_back(std::move(name));
    }

    return true;
}

} // namespace pi_model_detect_detail

static inline const char * pi_model_file_kind_name(pi_model_file_kind kind) {
    switch (kind) {
        case PI_MODEL_FILE_LLM:    return "llm";
        case PI_MODEL_FILE_MMPROJ: return "mmproj";
        case PI_MODEL_FILE_UNKNOWN:
        default:                   return "unknown";
    }
}

static inline pi_model_detect_result pi_model_detect_gguf_file(const std::string & path) {
    using namespace pi_model_detect_detail;

    pi_model_detect_result result;
    gguf_probe probe;
    std::string error;
    if (!read_gguf_probe(path, probe, error)) {
        result.reason = error;
        return result;
    }

    result.general_name = probe.general_name;
    result.general_architecture = probe.general_architecture;

    const std::string arch = lower_ascii(probe.general_architecture);
    const bool is_mmproj = arch == "clip" || probe.has_clip_vision;
    result.file_kind = is_mmproj ? PI_MODEL_FILE_MMPROJ : PI_MODEL_FILE_LLM;

    if (is_mmproj) {
        result.kind = PI_MODEL_AUTO;
        result.reason = "mmproj PI model detection is ignored; LLM GGUF decides pi0/pi05";
        return result;
    }

    if (!is_pi_llm_architecture(arch)) {
        result.kind = PI_MODEL_AUTO;
        result.reason = probe.general_architecture.empty()
            ? "missing general.architecture; not probing llm tensor names"
            : "general.architecture is not a pi llm; not probing llm tensor names";
        return result;
    }

    const std::set<std::string> names(probe.tensor_names.begin(), probe.tensor_names.end());
    if (names.count("output_norm_dense.weight") != 0) {
        result.kind = PI_MODEL_PI05;
        result.reason = "llm tensor output_norm_dense.weight exists";
    } else if (std::any_of(probe.tensor_names.begin(), probe.tensor_names.end(), [](const std::string & name) {
        return name.find(".attn_norm_dense.") != std::string::npos ||
               name.find(".ffn_norm_dense.")  != std::string::npos;
    })) {
        result.kind = PI_MODEL_PI05;
        result.reason = "llm dense norm tensors exist";
    } else if (names.count("time_mlp_in.weight") != 0 || names.count("time_mlp_out.weight") != 0) {
        result.kind = PI_MODEL_PI05;
        result.reason = "llm time_mlp_in/out tensors exist";
    } else if (names.count("state_proj.weight") != 0) {
        result.kind = PI_MODEL_PI0;
        result.reason = "llm tensor state_proj.weight exists";
    } else if (names.count("action_time_mlp_in.weight") != 0 || names.count("action_time_mlp_out.weight") != 0) {
        result.kind = PI_MODEL_PI0;
        result.reason = "llm action_time_mlp_in/out tensors exist";
    } else if (contains_ci(probe.general_name, "pi05") || contains_ci(probe.general_name, "pi0.5")) {
        result.kind = PI_MODEL_PI05;
        result.reason = "pi llm general.name contains pi05 fallback";
    } else {
        result.kind = PI_MODEL_AUTO;
        result.reason = "no strong pi llm tensor feature matched";
    }

    return result;
}

static inline pi_model_detect_result pi_model_detect_gguf_pair(
        const std::string & llm_path,
        const std::string & mmproj_path) {
    const pi_model_detect_result llm = llm_path.empty()
        ? pi_model_detect_result{}
        : pi_model_detect_gguf_file(llm_path);
    (void) mmproj_path;

    if (llm.kind != PI_MODEL_AUTO) {
        return llm;
    }

    pi_model_detect_result unknown;
    unknown.reason = "no strong llm feature matched";
    return unknown;
}
