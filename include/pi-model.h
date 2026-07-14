#pragma once

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum pi_model_kind {
    PI_MODEL_AUTO = 0,
    PI_MODEL_PI0,
    PI_MODEL_PI05,
} pi_model_kind;

static inline char pi_model_ascii_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? (char) (c - 'A' + 'a') : c;
}

static inline int pi_model_streq_ci(const char * lhs, const char * rhs) {
    if (lhs == NULL || rhs == NULL) {
        return 0;
    }
    while (*lhs != '\0' && *rhs != '\0') {
        if (pi_model_ascii_lower(*lhs) != pi_model_ascii_lower(*rhs)) {
            return 0;
        }
        ++lhs;
        ++rhs;
    }
    return *lhs == '\0' && *rhs == '\0';
}

static inline int pi_model_is_empty(const char * value) {
    if (value == NULL) {
        return 1;
    }
    while (*value == ' ' || *value == '\t' || *value == '\n' || *value == '\r') {
        ++value;
    }
    return *value == '\0';
}

static inline pi_model_kind pi_model_kind_from_string(const char * value) {
    if (pi_model_is_empty(value) ||
        pi_model_streq_ci(value, "auto") ||
        pi_model_streq_ci(value, "detect")) {
        return PI_MODEL_AUTO;
    }

    if (pi_model_streq_ci(value, "pi0") ||
        pi_model_streq_ci(value, "p0") ||
        pi_model_streq_ci(value, "0")) {
        return PI_MODEL_PI0;
    }

    if (pi_model_streq_ci(value, "pi05") ||
        pi_model_streq_ci(value, "pi0.5") ||
        pi_model_streq_ci(value, "pi0_5") ||
        pi_model_streq_ci(value, "p05") ||
        pi_model_streq_ci(value, "05")) {
        return PI_MODEL_PI05;
    }

    return PI_MODEL_AUTO;
}

static inline pi_model_kind pi_model_kind_from_env(void) {
    return pi_model_kind_from_string(getenv("PI_MODEL"));
}

static inline const char * pi_model_kind_name(pi_model_kind kind) {
    switch (kind) {
        case PI_MODEL_PI0:  return "pi0";
        case PI_MODEL_PI05: return "pi05";
        case PI_MODEL_AUTO:
        default:            return "auto";
    }
}

static inline int pi_model_kind_is_pi0(pi_model_kind kind) {
    return kind == PI_MODEL_PI0;
}

static inline int pi_model_kind_is_pi05(pi_model_kind kind) {
    return kind == PI_MODEL_PI05;
}

// In the current Jetson-PI05 fusion base, AUTO keeps the existing PI05-biased
// behavior. Set PI_MODEL=pi0 to explicitly select legacy PI0 adapters.
static inline int pi_model_use_pi05_adapters_by_default(pi_model_kind kind) {
    return kind != PI_MODEL_PI0;
}

static inline int pi_model_env_truthy(const char * value) {
    if (pi_model_is_empty(value)) {
        return 0;
    }
    return !(pi_model_streq_ci(value, "0") ||
             pi_model_streq_ci(value, "false") ||
             pi_model_streq_ci(value, "off") ||
             pi_model_streq_ci(value, "no"));
}

static inline const char * pi_model_debug_dump_file(void) {
    const char * path = getenv("PI_MODEL_DEBUG_DUMP_FILE");
    if (!pi_model_is_empty(path)) {
        return path;
    }
    return getenv("PI05_DEBUG_DUMP_FILE");
}

static inline int pi_model_debug_dump_enabled(void) {
    return !pi_model_is_empty(pi_model_debug_dump_file());
}

static inline int pi_model_debug_enabled(void) {
    return pi_model_env_truthy(getenv("PI_MODEL_DEBUG")) || pi_model_debug_dump_enabled();
}

static inline int pi_model_debug_dump_values(int default_value) {
    const char * raw = getenv("PI_MODEL_DEBUG_DUMP_VALUES");
    if (pi_model_is_empty(raw)) {
        raw = getenv("PI05_DEBUG_DUMP_VALUES");
    }
    const int parsed = raw != NULL ? atoi(raw) : 0;
    if (parsed > 0) {
        return parsed;
    }
    return default_value;
}

#ifdef __cplusplus
}
#endif
