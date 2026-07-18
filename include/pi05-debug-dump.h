#ifndef PI05_DEBUG_DUMP_H
#define PI05_DEBUG_DUMP_H

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

static inline const char * pi05_debug_binary_dir() {
    const char * dir = std::getenv("PI05_DEBUG_DIR");
    return dir != nullptr && dir[0] != '\0' ? dir : nullptr;
}

static inline bool pi05_debug_binary_enabled() {
    return pi05_debug_binary_dir() != nullptr;
}

static inline bool pi05_debug_ensure_dir(const std::string & dir) {
    if (dir.empty()) {
        return false;
    }

    std::string current;
    current.reserve(dir.size());
    for (size_t i = 0; i < dir.size(); ++i) {
        current.push_back(dir[i]);
        if (dir[i] != '/' || current.size() == 1) {
            continue;
        }
        if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
            return false;
        }
    }
    return ::mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST;
}

static inline std::string pi05_debug_safe_name(const char * name) {
    std::string result = name != nullptr ? name : "unnamed";
    for (char & ch : result) {
        const bool ok = (ch >= 'a' && ch <= 'z') ||
                        (ch >= 'A' && ch <= 'Z') ||
                        (ch >= '0' && ch <= '9') || ch == '_' || ch == '-';
        if (!ok) {
            ch = '_';
        }
    }
    return result;
}

static inline bool pi05_debug_dump_binary(
        const char * name,
        const void * data,
        size_t nbytes,
        const char * dtype,
        std::initializer_list<int64_t> shape) {
    const char * dir_raw = pi05_debug_binary_dir();
    if (dir_raw == nullptr || data == nullptr || nbytes == 0) {
        return false;
    }

    const std::string dir(dir_raw);
    if (!pi05_debug_ensure_dir(dir)) {
        std::fprintf(stderr, "[PI05_DUMP] failed to create directory: %s\n", dir.c_str());
        return false;
    }

    const std::string safe = pi05_debug_safe_name(name);
    const std::string bin_path  = dir + "/" + safe + ".bin";
    const std::string meta_path = dir + "/" + safe + ".json";

    std::ofstream bin(bin_path, std::ios::binary | std::ios::trunc);
    if (!bin.is_open()) {
        std::fprintf(stderr, "[PI05_DUMP] failed to open: %s\n", bin_path.c_str());
        return false;
    }
    bin.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(nbytes));
    bin.close();
    if (!bin) {
        std::fprintf(stderr, "[PI05_DUMP] failed to write: %s\n", bin_path.c_str());
        return false;
    }

    std::ofstream meta(meta_path, std::ios::trunc);
    if (!meta.is_open()) {
        std::fprintf(stderr, "[PI05_DUMP] failed to open: %s\n", meta_path.c_str());
        return false;
    }
    meta << "{\n"
         << "  \"version\": 1,\n"
         << "  \"name\": \"" << safe << "\",\n"
         << "  \"dtype\": \"" << (dtype != nullptr ? dtype : "unknown") << "\",\n"
         << "  \"shape\": [";
    size_t index = 0;
    for (const int64_t dim : shape) {
        if (index++ > 0) {
            meta << ", ";
        }
        meta << dim;
    }
    meta << "],\n"
         << "  \"layout\": \"ggml-ne0-contiguous\",\n"
         << "  \"nbytes\": " << nbytes << ",\n"
         << "  \"data_file\": \"" << safe << ".bin\"\n"
         << "}\n";
    meta.close();

    std::fprintf(stderr, "[PI05_DUMP] wrote %s (%zu bytes)\n", safe.c_str(), nbytes);
    return true;
}

static inline bool pi05_debug_dump_f32(
        const char * name,
        const float * data,
        size_t count,
        std::initializer_list<int64_t> shape) {
    return pi05_debug_dump_binary(name, data, count * sizeof(float), "f32", shape);
}

static inline bool pi05_debug_dump_i32(
        const char * name,
        const int32_t * data,
        size_t count,
        std::initializer_list<int64_t> shape) {
    return pi05_debug_dump_binary(name, data, count * sizeof(int32_t), "i32", shape);
}

#endif // PI05_DEBUG_DUMP_H
