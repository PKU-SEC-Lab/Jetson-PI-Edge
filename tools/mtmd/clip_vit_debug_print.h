#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <inttypes.h>
#include "ggml.h"
#include "ggml-backend.h"

static void pi05_clip_print_vit_debug_tensors(ggml_backend_sched_t sched, ggml_cgraph * gf) {
    const char * dbg_env = std::getenv("PI05_DEBUG_PREFIX");
    if (!dbg_env || dbg_env[0] == 0 || dbg_env[0] == 0) return;

    int preview_n = 256;
    const char * pv_env = std::getenv("PI05_DEBUG_VALUES");
    if (pv_env) preview_n = std::atoi(pv_env);

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (!node || !node->name) continue;
        if (strncmp(node->name, "pi05_dbg_vit_", 13) != 0) continue;

        size_t nbytes = ggml_nbytes(node);
        std::vector<uint8_t> buf(nbytes);
        ggml_backend_t be = ggml_backend_sched_get_tensor_backend(sched, node);
        if (be) ggml_backend_tensor_get(node, buf.data(), 0, nbytes);

        const float * fp = (const float *)buf.data();
        size_t n_elem = nbytes / sizeof(float);
        size_t n = (size_t)preview_n < n_elem ? (size_t)preview_n : n_elem;

        const char * tag = node->name + 9; // skip "pi05_dbg_"
        fprintf(stderr, "[DBG][JETSON][%s] type=%s shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
            tag, ggml_type_name(node->type),
            node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
        fprintf(stderr, "[DBG][JETSON][%s] first=[", tag);
        for (size_t j = 0; j < n; ++j) {
            if (j > 0) fprintf(stderr, ", ");
            fprintf(stderr, "%.9g", fp[j]);
        }
        fprintf(stderr, "]\n");
    }
}
