#include "llama-context.h"

#include "ggml-alloc.h"

#include "llama-arch.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-nvtx.h"
#include "pi0-build-config.h"
#include "pi0-perf.h"
#include "pi-model.h"
#include "pi0-prof.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <chrono>
#include <sstream>

#include <random>
#include <vector>
#include <functional>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>

//
// llama_context
//

bool llama_pi0_perf_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("LLAMA_PI0_PERF");
        return env != nullptr && env[0] != '\0' && std::atoi(env) != 0;
    }();
    return enabled;
}

int32_t llama_pi0_decode_unroll_steps(bool is_pi05, int32_t inference_steps) {
    if (!is_pi05 || inference_steps < 2) {
        return 1;
    }
    const char * env = std::getenv("PI0_DECODE_UNROLL");
    if (env == nullptr || env[0] == '\0') {
        return 1;
    }
    const int32_t requested = std::atoi(env);
    if (requested < 2) {
        return 1;
    }
    // Fuse at most inference_steps per graph_compute; tail uses min(remaining, cfg).
    return std::min(requested, inference_steps);
}

namespace {

using pi0_perf_clock = std::chrono::high_resolution_clock;

double pi0_perf_elapsed_ms(const pi0_perf_clock::time_point & t0, const pi0_perf_clock::time_point & t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static bool pi0_log_encode_kv_shapes_enabled() {
    return llama_pi0_perf_enabled() || std::getenv("PI0_LOG_ENCODE_KV") != nullptr;
}

static bool pi05_debug_pre_noise_enabled() {
    const char * env = std::getenv("PI05_DEBUG_PRE_NOISE");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

static bool pi05_debug_dump_file_enabled() {
    return pi_model_debug_dump_enabled();
}

static int pi05_debug_preview_count() {
    const char * env = std::getenv("PI05_DEBUG_VALUES");
    if (env != nullptr && env[0] != '\0') {
        const int n = std::atoi(env);
        if (n > 0) {
            return n;
        }
    }
    return 8;
}

static void pi05_debug_print_tensor_preview(const char * tag, ggml_tensor * tensor) {
    if (!pi05_debug_pre_noise_enabled() || tensor == nullptr) {
        return;
    }

    fprintf(stderr,
        "[DBG][JETSON][%s] type=%s shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] nb=[%zu,%zu,%zu,%zu]\n",
        tag,
        ggml_type_name(tensor->type),
        tensor->ne[0],
        tensor->ne[1],
        tensor->ne[2],
        tensor->ne[3],
        tensor->nb[0],
        tensor->nb[1],
        tensor->nb[2],
        tensor->nb[3]);

    if (tensor->type != GGML_TYPE_F32) {
        fprintf(stderr, "[DBG][JETSON][%s] preview skipped: non-F32 tensor\n", tag);
        return;
    }

    const size_t n_total = ggml_nelements(tensor);
    const size_t n = std::min<size_t>((size_t) pi05_debug_preview_count(), n_total);
    std::vector<float> values(n);
    ggml_backend_tensor_get(tensor, values.data(), 0, n * sizeof(float));

    fprintf(stderr, "[DBG][JETSON][%s] first=[", tag);
    for (size_t i = 0; i < n; ++i) {
        fprintf(stderr, "%s%.9g", i == 0 ? "" : ", ", values[i]);
    }
    fprintf(stderr, "]\n");
}

static void pi05_debug_print_float_preview(const char * tag, const float * data, size_t count) {
    if (!pi05_debug_pre_noise_enabled() || data == nullptr) {
        return;
    }

    const size_t n = std::min<size_t>((size_t) pi05_debug_preview_count(), count);
    fprintf(stderr, "[DBG][JETSON][%s] first=[", tag);
    for (size_t i = 0; i < n; ++i) {
        fprintf(stderr, "%s%.9g", i == 0 ? "" : ", ", data[i]);
    }
    fprintf(stderr, "]\n");
}

struct pi0_decode_ubatch_perf_log {
    double mem_apply_ms     = 0;
    double graph_build_ms   = 0;
    double graph_alloc_ms   = 0;
    double set_inputs_ms    = 0;
    double set_static_ms    = 0;
    double set_dynamic_ms   = 0;
    double set_kv_ms        = 0;
    bool   set_kv_skipped   = false;
    double graph_compute_ms = 0;
    bool   graph_reused     = false;
};

// Fetch action tensor to host: one CUDA stream sync + D2H (same pattern as logits in decode).
static void pi0_fetch_action_to_host(
        ggml_backend_sched_t sched,
        ggml_tensor * action,
        void * dst,
        size_t nbytes) {
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, action);
    if (backend != nullptr) {
        ggml_backend_tensor_get_async(backend, action, dst, 0, nbytes);
        ggml_backend_synchronize(backend);
    } else {
        ggml_backend_tensor_get(action, dst, 0, nbytes);
    }
}

void pi0_perf_log_decode_step(
        int32_t step_idx,
        int32_t n_steps,
        float time_step,
        bool kv_skip,
        bool gpu_kv,
        double step_total_ms,
        double precompute_ms,
        const pi0_decode_ubatch_perf_log & ub,
        double gpu_wait_action_ms,
        double update_state_ms,
        double update_action_ms,
        bool mask_skip,
        int32_t reused_delta) {
    fprintf(stderr,
        "[PI0 step %d/%d] time_step=%.4f total=%.2f ms\n",
        step_idx + 1,
        n_steps,
        time_step,
        step_total_ms);

    if (precompute_ms > 0.0) {
        fprintf(stderr, "  precompute_time_emb=%.2f ms (once per decode)\n", precompute_ms);
    }

    fprintf(stderr,
        "  process_ubatch=%.2f ms (reused=%d reused_delta=%d graph_build=%.2f alloc_graph=%.2f "
        "set_inputs=%.2f set_static=%.2f set_dynamic=%.2f set_kv=%.2f graph_submit=%.2f mem_apply=%.2f)\n",
        ub.mem_apply_ms + ub.graph_build_ms + ub.graph_alloc_ms + ub.set_inputs_ms + ub.graph_compute_ms,
        ub.graph_reused ? 1 : 0,
        reused_delta,
        ub.graph_build_ms,
        ub.graph_alloc_ms,
        ub.set_inputs_ms,
        ub.set_static_ms,
        ub.set_dynamic_ms,
        ub.set_kv_ms,
        ub.graph_compute_ms,
        ub.mem_apply_ms);

    fprintf(stderr,
        "  gpu_wait_action=%.2f ms (stream sync + D2H; includes real kernel time)\n"
        "  update_state=%.2f ms\n"
        "  update_action=%.2f ms\n"
        "  kv_skip=%d set_kv_skipped=%d mask_skip=%d gpu_kv=%d\n",
        gpu_wait_action_ms,
        update_state_ms,
        update_action_ms,
        kv_skip ? 1 : 0,
        ub.set_kv_skipped ? 1 : 0,
        mask_skip ? 1 : 0,
        gpu_kv ? 1 : 0);
}

} // namespace

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_threads        = params.n_threads;
    cparams.n_threads_batch  = params.n_threads_batch;
    cparams.yarn_ext_factor  = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast   = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow   = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings       = params.embeddings;
    cparams.offload_kqv      = params.offload_kqv;
    cparams.no_perf          = params.no_perf;
    cparams.pooling_type     = params.pooling_type;
    cparams.warmup           = false;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    // the batch has to be at least GGML_KQ_MASK_PAD because we will be padding the KQ_mask
    // this is required by GPU kernels in order to avoid out-of-bounds accesses (e.g. ggml_flash_attn_ext)
    // ref: https://github.com/ggerganov/llama.cpp/pull/5021
    // TODO: this padding is not needed for the cache-less context so we should probably move it to llama_memory
    if (cparams.n_batch < GGML_KQ_MASK_PAD) {
        LLAMA_LOG_WARN("%s: n_batch is less than GGML_KQ_MASK_PAD - increasing to %d\n", __func__, GGML_KQ_MASK_PAD);
        cparams.n_batch = GGML_KQ_MASK_PAD;
    }
#if PI0_ENABLE_AE_STEP_DEBUG
    // printf("Debug: cparams.n_batch=%d,params.n_ubatch=%d\n", cparams.n_batch, params.n_ubatch);
#endif
    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq     = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn   = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn    = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified    = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (auto * dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            // resized during inference when a batch uses more outputs
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }
    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k   =*/ params.type_k,
            /*.type_v   =*/ params.type_v,
            /*.swa_full =*/ params.swa_full,
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                auto * dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        const size_t max_nodes = this->graph_max_nodes();

        LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

        gf_res_prev.reset(new llm_graph_result(max_nodes));
        gf_res_reserve.reset(new llm_graph_result(max_nodes));
        if (model.arch == LLM_ARCH_PI0) {
            gf_res_pi0_enc.reset(new llm_graph_result(max_nodes));
            gf_res_pi0_dec.reset(new llm_graph_result(max_nodes));
            LLAMA_LOG_INFO("%s: PI0 dual graph cache enabled (separate encoder/decoder slots)\n", __func__);
        }
        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.params.n_gpu_layers > (int) model.hparams.n_layer &&
            model.params.split_mode == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }
        sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, pipeline_parallel, cparams.op_offload));

        if (pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled (n_copies=%d)\n", __func__, ggml_backend_sched_get_n_copies(sched.get()));
        }

        llama_memory_context_ptr mctx;
        if (memory) {
            LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
            mctx = memory->init_full();
            if (!mctx) {
                throw std::runtime_error("failed to initialize memory module");
            }
        }
        cross.v_embd.clear();

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        // avoid reserving graphs with zero outputs - assume one output per sequence
        n_outputs = n_seqs;

        LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

        // resolve automatic Flash Attention use
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to split graph for Flash Attention check");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FATTN) + 1;
            bool fa_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_FLASH_ATTN_EXT) {
                    continue;
                }
                ggml_backend_dev_t device_fa = ggml_backend_get_device(
                    ggml_backend_sched_get_tensor_backend(sched.get(), n));

                // TODO: instead of the tensor names, use a map to keep track of which (FA) tensors belong to which layer
                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FATTN "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                if (device_fa != device_kv) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the Flash Attention tensor "
                        "is assigned to device %s (usually due to missing support)\n",
                        __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_fa));
                    // FIXME: fa_device_mismatch logic is wrong for --no-kv-offload, but this is broken anyways
                    fa_device_mismatch = true;
                    break;
                }
            }
            if (fa_device_mismatch) {
                cparams.flash_attn = false;
                LLAMA_LOG_WARN("%s: Flash Attention was auto, set to disabled\n", __func__);
                if (ggml_is_quantized(params.type_v)) {
                    throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
                }
            } else {
                cparams.flash_attn = true;
                LLAMA_LOG_INFO("%s: Flash Attention was auto, set to enabled\n", __func__);
            }
        }

        // reserve worst-case graph
        int n_splits_pp = -1;
        int n_nodes_pp  = -1;

        int n_splits_tg = -1;
        int n_nodes_tg  = -1;

        // reserve pp (prompt processing) graph first so that buffers are only allocated once
        {
            auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get());

            if (!gf) {
                if (pipeline_parallel) {
                    LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                    gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get());
                }
                if (!gf) {
                    throw std::runtime_error("failed to allocate compute pp buffers");
                }
            }


            n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
            n_nodes_pp  = ggml_graph_n_nodes(gf);
        }

        // reserve with tg (token generation) graph to get the number of splits and nodes
        {
            auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get());
            if (!gf) {
                throw std::runtime_error("failed to allocate compute tg buffers");
            }

            n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
            n_nodes_tg  = ggml_graph_n_nodes(gf);
        }

        // reserve again with pp graph to avoid ggml-alloc reallocations during inference
        {
            // TODO: not sure if the following graph would be worster case for multi-stream KV caches:
            //
            // auto * gf = graph_reserve(n_tokens, 1, n_tokens, mctx.get());
            //
            auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get());
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            size_t size = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size > 1) {
                LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(buft),
                        size / 1024.0 / 1024.0);
            }
        }

        if (n_nodes_pp == n_nodes_tg) {
            LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
        } else {
            LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
        }

        if (n_splits_pp == n_splits_tg) {
            LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
        } else {
            LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
        }

        if (model.arch == LLM_ARCH_PI0) {
            sched_pi0_enc.reset(ggml_backend_sched_new(
                    backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                    max_nodes, pipeline_parallel, cparams.op_offload));
            sched_pi0_dec.reset(ggml_backend_sched_new(
                    backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                    max_nodes, pipeline_parallel, cparams.op_offload));

            if (!sched_pi0_enc || !sched_pi0_dec) {
                throw std::runtime_error("failed to create PI0 persistent graph schedulers");
            }

            LLAMA_LOG_INFO("%s: PI0 persistent encoder/decoder schedulers enabled\n", __func__);
        }
    }
}

llama_context::~llama_context() {
    ggml_opt_free(opt_ctx);
}


void print_vector(const std::vector<float>& vec, const std::string& name = "Vector") {
    GGML_UNUSED(vec);
    GGML_UNUSED(name);
}

static std::string pi05_debug_base_tensor_name(const char * name) {
    if (name == nullptr) {
        return {};
    }

    std::string tensor_name(name);
    const size_t annotation_pos = tensor_name.find(' ');
    if (annotation_pos != std::string::npos) {
        tensor_name.resize(annotation_pos);
    }

    const size_t dash_pos = tensor_name.find_last_of('-');
    if (dash_pos != std::string::npos && dash_pos + 1 < tensor_name.size()) {
        bool numeric_suffix = true;
        for (size_t i = dash_pos + 1; i < tensor_name.size(); ++i) {
            if (!std::isdigit((unsigned char) tensor_name[i])) {
                numeric_suffix = false;
                break;
            }
        }
        if (numeric_suffix) {
            tensor_name.resize(dash_pos);
        }
    }

    return tensor_name;
}

static bool pi05_debug_should_dump_tensor(const char * name) {
    if (name == nullptr) {
        return false;
    }

    const std::string tensor_name = pi05_debug_base_tensor_name(name);
    return tensor_name.find("pi05_dbg_") == 0 ||
           tensor_name == "pi05_time_cond" ||
           tensor_name == "pi05_adarms_cond" ||
           tensor_name == "pi05_adarms_cond_raw" ||
           tensor_name == "pi05_attn_mod" ||
           tensor_name == "pi05_ffn_mod" ||
           tensor_name == "pi05_out_mod" ||
           tensor_name == "pi05_normed" ||
           tensor_name == "pi05_scale" ||
           tensor_name == "pi05_shift" ||
           tensor_name == "pi05_gate" ||
            tensor_name == "pi05_scale_broadcast" ||
            tensor_name == "pi05_shift_broadcast" ||
            tensor_name == "pi05_gate_broadcast" ||
            tensor_name == "pi05_attn_gated" ||
            tensor_name == "pi05_attn_residual" ||
            tensor_name == "pi05_ffn_gated" ||
            tensor_name == "pi05_ffn_residual" ||
            tensor_name == "action_in" ||
            tensor_name == "action_final" ||
            tensor_name == "kqv_out" ||
            tensor_name == "pi05_dbg_l0_encoded" ||
            tensor_name == "pi05_dbg_l0_k_prefix" ||
             tensor_name == "pi05_dbg_l0_v_prefix";
}

static bool pi05_debug_should_dump_prefix_tensor(const char * name) {
    if (name == nullptr) {
        return false;
    }

    const std::string tensor_name = pi05_debug_base_tensor_name(name);
    return tensor_name.find("pi05_dbg_prefix_l0_") == 0 ||
           tensor_name.find("pi05_dbg_prefix_image_tokens_") == 0 ||
           tensor_name.find("pi05_dbg_siglip_") == 0 ||
           tensor_name == "pi05_dbg_prefix_text_tokens";
}

static void pi05_debug_log_tensor_preview_to_stream(std::ofstream & ofs, ggml_tensor * tensor, const char * prefix = nullptr) {
    if (tensor == nullptr) {
        ofs << (prefix ? prefix : "") << "tensor=<null>\n";
        return;
    }

    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne2 = tensor->ne[2];
    const int64_t ne3 = tensor->ne[3];
    const int64_t n = ggml_nelements(tensor);
    const size_t nbytes = ggml_nbytes(tensor);
    const size_t elem_size = ggml_element_size(tensor);

    ofs << (prefix ? prefix : "")
        << "name=" << tensor->name
        << " type=" << ggml_type_name(tensor->type)
        << " shape=[" << ne0 << ", " << ne1 << ", " << ne2 << ", " << ne3 << "]"
        << " n=" << n
        << " nbytes=" << nbytes
        << " elem_size=" << elem_size
        << " data=" << tensor->data
        << "\n";

    if (n <= 0 || nbytes == 0) {
        ofs << (prefix ? prefix : "") << "empty_tensor\n";
        return;
    }

    if (tensor->type != GGML_TYPE_F32) {
        ofs << (prefix ? prefix : "") << "preview_skipped_non_f32\n";
        return;
    }

    size_t requested_preview_count = (size_t) pi_model_debug_dump_values(32);
    const size_t preview_count = std::min<size_t>((size_t) n, requested_preview_count);
    const size_t preview_bytes = std::min(nbytes, preview_count * sizeof(float));
    const size_t safe_count = preview_bytes / sizeof(float);

    if (safe_count == 0) {
        ofs << (prefix ? prefix : "") << "preview_skipped_zero_safe_count\n";
        return;
    }

    std::vector<float> values(safe_count);
    ggml_backend_tensor_get(tensor, values.data(), 0, safe_count * sizeof(float));

    size_t finite_count = 0;
    size_t non_finite_count = 0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();

    for (float v : values) {
        if (std::isfinite(v)) {
            ++finite_count;
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        } else {
            ++non_finite_count;
        }
    }

    ofs << (prefix ? prefix : "")
        << "preview_count=" << safe_count
        << " finite=" << finite_count
        << " non_finite=" << non_finite_count;
    if (finite_count > 0) {
        ofs << " min=" << min_v << " max=" << max_v;
    }
    ofs << "\n" << (prefix ? prefix : "") << "values=";

    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ofs << ' ';
        }
        ofs << values[i];
    }
    if ((size_t) n > values.size()) {
        ofs << " ...";
    }
    ofs << "\n";
}

static void pi05_debug_dump_tensor_to_stream(std::ofstream & ofs, ggml_tensor * tensor) {
    pi05_debug_log_tensor_preview_to_stream(ofs, tensor, nullptr);
}

static void pi05_debug_dump_host_vector_to_file(
        const char * name,
        const std::vector<float> & values,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    const char * dump_path = pi_model_debug_dump_file();
    if (dump_path == nullptr || dump_path[0] == '\0' || name == nullptr) {
        return;
    }

    std::ofstream ofs(dump_path, std::ios::app);
    if (!ofs.is_open()) {
        LLAMA_LOG_WARN("%s: failed to open PI05_DEBUG_DUMP_FILE=%s\n", __func__, dump_path);
        return;
    }

    size_t requested_preview_count = (size_t) pi_model_debug_dump_values(32);

    const size_t preview_count = std::min(values.size(), requested_preview_count);
    size_t finite_count = 0;
    size_t non_finite_count = 0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < preview_count; ++i) {
        const float v = values[i];
        if (std::isfinite(v)) {
            ++finite_count;
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        } else {
            ++non_finite_count;
        }
    }

    ofs << "name=" << name
        << " type=f32 shape=[" << ne0 << ", " << ne1 << ", " << ne2 << ", " << ne3 << "]"
        << " n=" << values.size()
        << " nbytes=" << values.size() * sizeof(float)
        << " elem_size=4 data=<host>\n";
    ofs << "preview_count=" << preview_count
        << " finite=" << finite_count
        << " non_finite=" << non_finite_count;
    if (finite_count > 0) {
        ofs << " min=" << min_v << " max=" << max_v;
    }
    ofs << "\nvalues=";
    for (size_t i = 0; i < preview_count; ++i) {
        if (i > 0) {
            ofs << ' ';
        }
        ofs << values[i];
    }
    if (values.size() > preview_count) {
        ofs << " ...";
    }
    ofs << "\n";
}

static std::ofstream & pi05_debug_host_log_stream() {
    static std::ofstream ofs;
    return ofs;
#if 0
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        const char * dump_path = getenv("PI05_DEBUG_HOST_LOG_FILE");
        if (dump_path != nullptr && dump_path[0] != '\0') {
            ofs.open(dump_path, std::ios::out | std::ios::app);
        }
    }
    return ofs;
#endif
}

static void pi05_debug_host_log_line(const std::string & line) {
    GGML_UNUSED(line);
    return;
#if 0
    auto & ofs = pi05_debug_host_log_stream();
    if (!ofs.is_open()) {
        return;
    }
    ofs << line << '\n';
    ofs.flush();
#endif
}

static void pi05_debug_dump_graph_tensors_to_file(ggml_backend_sched_t sched, ggml_cgraph * gf) {
    const char * dump_path = pi_model_debug_dump_file();
    if (dump_path == nullptr || dump_path[0] == '\0' || gf == nullptr) {
        return;
    }

    std::ofstream ofs(dump_path, std::ios::app);
    if (!ofs.is_open()) {
        LLAMA_LOG_WARN("%s: failed to open PI05_DEBUG_DUMP_FILE=%s\n", __func__, dump_path);
        return;
    }

    int matched = 0;
    ofs << "===== pi05 tensor dump begin =====\n";
    ofs << "gf_n_nodes=" << ggml_graph_n_nodes(gf) << "\n";

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node == nullptr || !pi05_debug_should_dump_tensor(node->name)) {
            continue;
        }

        ++matched;
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, node);
        ofs << "node_index=" << i
            << " backend=" << (backend ? ggml_backend_name(backend) : "<null>")
            << "\n";
        pi05_debug_dump_tensor_to_stream(ofs, node);
    }

    ofs << "matched_nodes=" << matched << "\n";
    ofs << "===== pi05 tensor dump end =====\n";
}

static void pi05_debug_dump_prefix_graph_tensors_to_file(ggml_backend_sched_t sched, ggml_cgraph * gf) {
    const char * dump_path = pi_model_debug_dump_file();
    if (dump_path == nullptr || dump_path[0] == '\0' || gf == nullptr) {
        return;
    }

    std::ofstream ofs(dump_path, std::ios::app);
    if (!ofs.is_open()) {
        LLAMA_LOG_WARN("%s: failed to open PI05_DEBUG_DUMP_FILE=%s\n", __func__, dump_path);
        return;
    }

    int matched = 0;
    ofs << "===== pi05 prefix tensor dump begin =====\n";
    ofs << "gf_n_nodes=" << ggml_graph_n_nodes(gf) << "\n";

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node == nullptr || !pi05_debug_should_dump_prefix_tensor(node->name)) {
            continue;
        }

        ++matched;
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, node);
        ofs << "node_index=" << i
            << " backend=" << (backend ? ggml_backend_name(backend) : "<null>")
            << "\n";
        pi05_debug_dump_tensor_to_stream(ofs, node);
    }

    ofs << "matched_nodes=" << matched << "\n";
    ofs << "===== pi05 prefix tensor dump end =====\n";
}

void llama_context::synchronize() {
    ggml_backend_sched_synchronize(sched_active ? sched_active : sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched_active ? sched_active : sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits;
}

float * llama_context::get_logits_ith(int32_t i) {
    int64_t j = -1;

    output_reorder();

    try {
        if (logits == nullptr) {
            throw std::runtime_error("no logits");
        }

        if (i < 0) {
            j = n_outputs + i;
            if (j < 0) {
                throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
            }
        } else if ((size_t) i >= output_ids.size()) {
            throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
        } else {
            j = output_ids[i];
        }

        if (j < 0) {
            throw std::runtime_error(format("batch.logits[%d] != true", i));
        }
        if (j >= n_outputs) {
            // This should not happen
            throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
        }

        return logits + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    int64_t j = -1;

    output_reorder();

    try {
        if (embd == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        if (i < 0) {
            j = n_outputs + i;
            if (j < 0) {
                throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
            }
        } else if ((size_t) i >= output_ids.size()) {
            throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
        } else {
            j = output_ids[i];
        }

        if (j < 0) {
            throw std::runtime_error(format("batch.logits[%d] != true", i));
        }
        if (j >= n_outputs) {
            // This should not happen
            throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
        }

        return embd + j*model.hparams.n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
        if (set_abort_callback_fn) {
            set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.causal_attn = value;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.warmup = value;
}

void llama_context::set_adapter_lora(
            llama_adapter_lora * adapter,
            float scale) {
    LLAMA_LOG_DEBUG("%s: adapter = %p, scale = %f\n", __func__, (void *) adapter, scale);

    loras[adapter] = scale;
}

bool llama_context::rm_adapter_lora(
            llama_adapter_lora * adapter) {
    LLAMA_LOG_DEBUG("%s: adapter = %p\n", __func__, (void *) adapter);

    auto pos = loras.find(adapter);
    if (pos != loras.end()) {
        loras.erase(pos);
        return true;
    }

    return false;
}

void llama_context::clear_adapter_lora() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    loras.clear();
}

// void llama_context::set_state(std::vector<float> state_array) {
//     cross.state = std::move(state_array);
// }
// void llama_context::set_state(std::vector<float>&& state_array) {
//     cross.state = std::move(state_array);
// }


void llama_context::set_state(const std::vector<float>& state_array) {
    cross.state = state_array;
}

const float * llama_context::get_pi0_state() const {
    if (model.arch != LLM_ARCH_PI0 || cross.pi0_state_result.empty()) {
        return nullptr;
    }

    return cross.pi0_state_result.data();
}

int32_t llama_context::get_pi0_state_size() const {
    if (model.arch != LLM_ARCH_PI0) {
        return 0;
    }

    return (int32_t) cross.pi0_state_result.size();
}

const float * llama_context::get_pi0_action_input() const {
    if (model.arch != LLM_ARCH_PI0 || cross.pi0_action_input.empty()) {
        return nullptr;
    }

    return cross.pi0_action_input.data();
}

const float * llama_context::get_pi0_action() const {
    if (model.arch != LLM_ARCH_PI0 || cross.action.empty()) {
        return nullptr;
    }

    return cross.action.data();
}

int32_t llama_context::get_pi0_action_dim() const {
    if (model.arch != LLM_ARCH_PI0) {
        return 0;
    }

    return model.hparams.action_dim;
}

int32_t llama_context::get_pi0_action_steps() const {
    if (model.arch != LLM_ARCH_PI0) {
        return 0;
    }

    return model.hparams.action_steps;
}

llm_graph_result * llama_context::pi0_graph_result(llm_graph_type gtype) const {
    switch (gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return gf_res_pi0_enc ? gf_res_pi0_enc.get() : gf_res_prev.get();
        case LLM_GRAPH_TYPE_DECODER:
            return gf_res_pi0_dec ? gf_res_pi0_dec.get() : gf_res_prev.get();
        default:
            return gf_res_prev.get();
    }
}

ggml_backend_sched_t llama_context::pi0_sched_for_gtype(llm_graph_type gtype) const {
    switch (gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return sched_pi0_enc ? sched_pi0_enc.get() : sched.get();
        case LLM_GRAPH_TYPE_DECODER:
            return sched_pi0_dec ? sched_pi0_dec.get() : sched.get();
        default:
            return sched.get();
    }
}

// std::vector<float> create_normal_noise_cpp11(size_t count, unsigned int seed = 42) {
//     std::vector<float> noise(count);
    
//     // Create a random generator similar to a JAX RNG.
//     std::mt19937_64 generator(seed);  // 64-bit Mersenne Twister 19937.
//     std::normal_distribution<float> distribution(0.0f, 1.0f);  // Mean 0, standard deviation 1.
    
//     // Generate standard normal random values.
//     for (auto& value : noise) {
//         value = distribution(generator);
//     }
    
//     return noise;
// }

std::vector<float> create_normal_noise_cpp11(size_t count,unsigned int seed = 42) {
    std::vector<float> noise(count);
    
    // 1. Create a random device for OS/hardware entropy.
    std::random_device rd; 
    
    // 2. Seed the Mersenne Twister generator from the random device.
    std::mt19937_64 generator(rd());  
    
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    
    // 3. Generate standard normal random values.
    for (auto& value : noise) {
        value = distribution(generator);
    }
    
    return noise;
}

static bool try_load_pi0_action_noise(std::vector<float> & noise) {
    const char * path = std::getenv("PI0_ACTION_NOISE_BIN");
    if (path == nullptr || path[0] == '\0') {
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);

    const std::streamoff expected = (std::streamoff) (noise.size() * sizeof(float));
    if (size != expected) {
        LLAMA_LOG_WARN(
            "%s: PI0_ACTION_NOISE_BIN=%s has %lld bytes, expected %lld; falling back to C++ RNG\n",
            __func__, path, (long long) size, (long long) expected);
        return false;
    }

    in.read(reinterpret_cast<char *>(noise.data()), expected);
    if (!in) {
        LLAMA_LOG_WARN("%s: failed to read PI0 action noise from %s; falling back to C++ RNG\n", __func__, path);
        return false;
    }

    return true;
}

static void maybe_dump_pi0_action_noise(const std::vector<float> & noise) {
    const char * path = std::getenv("PI0_ACTION_NOISE_BIN");
    if (path == nullptr || path[0] == '\0') {
        return;
    }

    std::ifstream existing(path, std::ios::binary);
    if (existing) {
        return;
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        LLAMA_LOG_WARN("%s: failed to open PI0_ACTION_NOISE_BIN=%s for writing\n", __func__, path);
        return;
    }

    out.write(reinterpret_cast<const char *>(noise.data()), (std::streamsize) (noise.size() * sizeof(float)));
    if (!out) {
        LLAMA_LOG_WARN("%s: failed to write PI0 action noise to %s\n", __func__, path);
        return;
    }

}


bool llama_context::apply_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    return cvec.apply(model, data, len, n_embd, il_start, il_end);
}

void llama_context::pi0_clear_cross_kv() {
    pi0_enc_kv_gpu.ctx.reset();
    pi0_enc_kv_gpu.buf.reset();
    pi0_enc_kv_gpu.tensors.clear();
    pi0_enc_kv_gpu.kv_tokens = 0;

    cross.encoded_kv_gpu.clear();
    cross.pi0_use_gpu_kv             = false;
    cross.pi0_cross_kv_inputs_ready  = false;
    cross.pi0_decode_attn_mask_ready = false;
    cross.encoded_kv_dirty           = true;
    cross.pi0_decode_unroll          = 1;
    cross.pi0_action_out_accumulated = false;

    for (auto & kv : cross.encoded_kv_data) {
        kv.clear();
    }
}

void llama_context::pi0_log_encoded_kv_shapes(const std::vector<ggml_tensor *> & t_encoded_kv) const {
    if (!pi0_log_encode_kv_shapes_enabled()) {
        return;
    }

    const auto & hparams = model.hparams;
    const int n_enc_layers = hparams.n_layer / 2;
    const int n_embd_head  = hparams.n_embd_head_v;
    const int64_t n_prefix = cross.n_token;

    fprintf(stderr,
            "[PI0 encode->decode KV] summary: prefix_seq_len=%" PRId64 " n_enc_layers=%d n_dec_layers=%d "
            "n_layer_total=%d action_steps=%d action_dim=%d\n",
            n_prefix,
            n_enc_layers,
            n_enc_layers,
            hparams.n_layer,
            hparams.action_steps,
            hparams.action_dim);
    fprintf(stderr,
            "[PI0 encode->decode KV] tensor layout per K/V: ne0=n_embd_head=%d, ne1=n_head_kv(layer), "
            "ne2=prefix_seq_len; float_count=ne0*ne1*ne2\n",
            n_embd_head);

    int n_kv_tensors = 0;
    for (size_t i = 0; i < t_encoded_kv.size(); ++i) {
        const ggml_tensor * t = t_encoded_kv[i];
        if (t == nullptr) {
            continue;
        }
        ++n_kv_tensors;
        const int enc_layer = (int) i / 2;
        const char * kv_kind = (i % 2 == 0) ? "K" : "V";
        const bool on_device = t->buffer != nullptr && !ggml_backend_buffer_is_host(t->buffer);
        fprintf(stderr,
                "[PI0 encode->decode KV] enc[%02d] %s slot=%02zu: ne=[%" PRId64 ", %" PRId64 ", %" PRId64 "] "
                "floats=%" PRId64 " nbytes=%zu backend=%s\n",
                enc_layer,
                kv_kind,
                i,
                t->ne[0],
                t->ne[1],
                t->ne[2],
                ggml_nelements(t),
                (size_t) ggml_nbytes(t),
                on_device ? "device" : "host");
        fprintf(stderr,
                "[PI0 encode->decode KV]   -> decoder layer %02d cross-attn uses this %s "
                "(slots [%d,%d])\n",
                enc_layer + n_enc_layers,
                kv_kind,
                2 * enc_layer,
                2 * enc_layer + 1);
    }

    fprintf(stderr,
            "[PI0 encode->decode KV] encoder K/V tensors: %d (expect %d = 2 * n_enc_layers)\n",
            n_kv_tensors,
            2 * n_enc_layers);

    if (cross.pi0_use_gpu_kv) {
        fprintf(stderr,
                "[PI0 encode->decode KV] gpu staging: pi0_use_gpu_kv=1 cached_kv_tokens=%" PRId64
                " gpu_tensors=%zu\n",
                (int64_t) pi0_enc_kv_gpu.kv_tokens,
                pi0_enc_kv_gpu.tensors.size());
        for (size_t i = 0; i < pi0_enc_kv_gpu.tensors.size(); ++i) {
            const ggml_tensor * t = pi0_enc_kv_gpu.tensors[i];
            if (t == nullptr) {
                continue;
            }
            fprintf(stderr,
                    "[PI0 encode->decode KV] gpu_staging[%02zu]: ne=[%" PRId64 ", %" PRId64 ", %" PRId64 "] "
                    "floats=%" PRId64 "\n",
                    i,
                    t->ne[0],
                    t->ne[1],
                    t->ne[2],
                    ggml_nelements(t));
        }
    } else {
        fprintf(stderr, "[PI0 encode->decode KV] gpu staging: pi0_use_gpu_kv=0 (host encoded_kv_data)\n");
        int n_host_slots = 0;
        for (size_t i = 0; i < cross.encoded_kv_data.size(); ++i) {
            const size_t nfloats = cross.encoded_kv_data[i].size();
            if (nfloats == 0) {
                continue;
            }
            ++n_host_slots;
            fprintf(stderr,
                    "[PI0 encode->decode KV] host_data[%02zu]: floats=%zu (%.2f MiB) "
                    "expected=ne0*ne1*prefix=%d*n_head_kv(%zu)*%" PRId64 "\n",
                    i,
                    nfloats,
                    (double) nfloats * sizeof(float) / (1024.0 * 1024.0),
                    n_embd_head,
                    i,
                    n_prefix);
        }
        fprintf(stderr, "[PI0 encode->decode KV] host_data slots filled: %d\n", n_host_slots);
    }

    const int64_t suffix_len = hparams.action_steps;
    const int64_t attn_kv_len = n_prefix + suffix_len;
    fprintf(stderr,
            "[PI0 encode->decode KV] decoder cross-attn mask KV length: prefix=%" PRId64 " + suffix=%" PRId64
            " = %" PRId64 " (PI0.5 suffix_len=action_steps)\n",
            n_prefix,
            suffix_len,
            attn_kv_len);
}

void llama_context::pi0_refresh_encoded_kv_gpu(const std::vector<ggml_tensor *> & src, int n_layer) {
    const auto & hparams = model.hparams;

    ggml_tensor * src_ref = nullptr;
    for (int i = 0; i < n_layer && i < (int) src.size(); ++i) {
        if (src[i] != nullptr) {
            src_ref = src[i];
            break;
        }
    }

    if (src_ref == nullptr || src_ref->buffer == nullptr) {
        cross.pi0_use_gpu_kv = false;
        return;
    }

    if (ggml_backend_buffer_is_host(src_ref->buffer)) {
        cross.pi0_use_gpu_kv = false;
        return;
    }

    const int64_t kv_tokens = src_ref->ne[2];
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(src_ref->buffer);
    if (buft == nullptr) {
        cross.pi0_use_gpu_kv = false;
        return;
    }

    const bool need_realloc =
        !pi0_enc_kv_gpu.buf ||
        pi0_enc_kv_gpu.kv_tokens != kv_tokens ||
        (int) pi0_enc_kv_gpu.tensors.size() != n_layer;

    if (need_realloc) {
        if (!pi0_enc_kv_gpu.tensors.empty()) {
            if (sched_pi0_dec) {
                ggml_backend_sched_reset(sched_pi0_dec.get());
            }
            if (gf_res_pi0_dec) {
                gf_res_pi0_dec->reset();
            }
        }

        pi0_enc_kv_gpu.ctx.reset();
        pi0_enc_kv_gpu.buf.reset();
        pi0_enc_kv_gpu.tensors.clear();

        const size_t ctx_size = size_t(n_layer + 1) * ggml_tensor_overhead();
        ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        pi0_enc_kv_gpu.ctx.reset(ggml_init(params));
        if (!pi0_enc_kv_gpu.ctx) {
            cross.pi0_use_gpu_kv = false;
            return;
        }

        pi0_enc_kv_gpu.tensors.resize(n_layer, nullptr);
        const int64_t n_embd_head = hparams.n_embd_head_v;
        for (int i = 0; i < n_layer; ++i) {
            const int n_head_kv_layer = hparams.n_head_kv(i);
            pi0_enc_kv_gpu.tensors[i] = ggml_new_tensor_3d(
                pi0_enc_kv_gpu.ctx.get(), GGML_TYPE_F32, n_embd_head, n_head_kv_layer, kv_tokens);
            ggml_format_name(pi0_enc_kv_gpu.tensors[i], "pi0_enc_kv_%d", i);
        }

        pi0_enc_kv_gpu.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(pi0_enc_kv_gpu.ctx.get(), buft));
        if (!pi0_enc_kv_gpu.buf) {
            pi0_enc_kv_gpu.tensors.clear();
            cross.pi0_use_gpu_kv = false;
            return;
        }
        pi0_enc_kv_gpu.kv_tokens = kv_tokens;
    }

    cross.encoded_kv_gpu.assign(n_layer, nullptr);
    for (int i = 0; i < n_layer && i < (int) src.size(); ++i) {
        if (src[i] == nullptr || pi0_enc_kv_gpu.tensors[i] == nullptr) {
            continue;
        }
        llama_backend_tensor_copy_compat(src[i], pi0_enc_kv_gpu.tensors[i]);
        cross.encoded_kv_gpu[i] = pi0_enc_kv_gpu.tensors[i];
    }

    cross.pi0_use_gpu_kv            = true;
    cross.pi0_cross_kv_inputs_ready = false;
    cross.encoded_kv_dirty          = true;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    const bool pi0_decode_perf =
        llama_pi0_perf_enabled() && model.arch == LLM_ARCH_PI0 && gtype == LLM_GRAPH_TYPE_DECODER;

    pi0_decode_ubatch_perf ub_perf = {};

    auto t_mem_apply_begin = pi0_perf_clock::now();
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    auto t_mem_apply_end = pi0_perf_clock::now();
    if (pi0_decode_perf) {
        ub_perf.mem_apply_ms = pi0_perf_elapsed_ms(t_mem_apply_begin, t_mem_apply_end);
    }

    const bool pi0_dual_slot = model.arch == LLM_ARCH_PI0 &&
        (gtype == LLM_GRAPH_TYPE_ENCODER || gtype == LLM_GRAPH_TYPE_DECODER);

    auto * res = pi0_dual_slot ? pi0_graph_result(gtype) : gf_res_prev.get();
    auto * gf  = res->get_gf();
    ggml_backend_sched_t graph_sched = pi0_dual_slot ? pi0_sched_for_gtype(gtype) : sched.get();
    sched_active = graph_sched;

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters

    const auto gparams = graph_params(res, ubatch, mctx, gtype, graph_sched);

    if (!graph_reuse_disable && res->can_reuse(gparams)) {
        PI0_NVTX_SCOPE("pi0/reuse_hit");
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        n_reused++;
        if (pi0_decode_perf) {
            ub_perf.graph_reused = true;
        }
    } else {
        PI0_NVTX_SCOPE("pi0/reuse_miss");
        if (model.arch == LLM_ARCH_PI0 && gtype == LLM_GRAPH_TYPE_DECODER) {
            cross.pi0_cross_kv_inputs_ready  = false;
            cross.pi0_decode_attn_mask_ready = false;
        }
        res->reset();

        ggml_backend_sched_reset(graph_sched);
        ggml_backend_sched_set_eval_callback(graph_sched, cparams.cb_eval, cparams.cb_eval_user_data);

        auto t_build_begin = pi0_perf_clock::now();
        {
            PI0_NVTX_SCOPE("pi0/build_graph");
            gf = model.build_graph(gparams);
        }
        auto t_build_end = pi0_perf_clock::now();

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        auto t_alloc_begin = pi0_perf_clock::now();
        {
            PI0_NVTX_SCOPE("pi0/alloc_graph");
            if (!ggml_backend_sched_alloc_graph(graph_sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
                ret = GGML_STATUS_ALLOC_FAILED;
                return nullptr;
            }
        }
        auto t_alloc_end = pi0_perf_clock::now();

        if (pi0_decode_perf) {
            ub_perf.graph_build_ms = pi0_perf_elapsed_ms(t_build_begin, t_build_end);
            ub_perf.graph_alloc_ms = pi0_perf_elapsed_ms(t_alloc_begin, t_alloc_end);
        }
    }

    // set the input data for the input tensors
    {
        PI0_NVTX_SCOPE("pi0/set_inputs");
        if (model.arch == LLM_ARCH_PI0 && gtype == LLM_GRAPH_TYPE_DECODER) {
            auto t_set_inputs_begin = pi0_decode_perf ? pi0_perf_clock::now() : pi0_perf_clock::time_point{};

            if (pi0_decode_perf) {
                cross.pi0_perf_set_kv_ms     = 0;
                cross.pi0_perf_set_kv_skipped = false;
            }

            // Static inputs every step; cross-KV and AE mask early-return via pi0_*_ready flags.
            {
                auto t_static_begin = pi0_perf_clock::now();
                res->set_static_inputs(&ubatch);
                if (pi0_decode_perf) {
                    auto t_static_end = pi0_perf_clock::now();
                    ub_perf.set_static_ms = pi0_perf_elapsed_ms(t_static_begin, t_static_end);
                    ub_perf.set_kv_ms       = cross.pi0_perf_set_kv_ms;
                    ub_perf.set_kv_skipped  = cross.pi0_perf_set_kv_skipped;
                }
            }

            auto t_dynamic_begin = pi0_perf_clock::now();
            res->set_dynamic_inputs(&ubatch);
            if (pi0_decode_perf) {
                auto t_dynamic_end = pi0_perf_clock::now();
                ub_perf.set_dynamic_ms = pi0_perf_elapsed_ms(t_dynamic_begin, t_dynamic_end);
                if (ub_perf.set_kv_ms == 0 && cross.pi0_cross_kv_inputs_ready) {
                    ub_perf.set_kv_skipped = true;
                }
                ub_perf.set_inputs_ms = pi0_perf_elapsed_ms(t_set_inputs_begin, t_dynamic_end);
            }
        } else {
            res->set_inputs(&ubatch);
        }
    }

    auto t_compute_begin = pi0_perf_clock::now();
    PI0_NVTX_SCOPE("pi0/graph_compute");
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1, graph_sched);
    if (pi0_decode_perf) {
        auto t_compute_end = pi0_perf_clock::now();
        ub_perf.graph_compute_ms = pi0_perf_elapsed_ms(t_compute_begin, t_compute_end);
        pi0_last_decode_ubatch_perf = ub_perf;
    }

    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    if (model.arch == LLM_ARCH_PI0 && pi05_debug_dump_file_enabled()) {
        ggml_backend_sched_synchronize(graph_sched);
        if (gtype == LLM_GRAPH_TYPE_ENCODER) {
            pi05_debug_dump_prefix_graph_tensors_to_file(graph_sched, res->get_gf());
        } else if (gtype == LLM_GRAPH_TYPE_DECODER) {
            pi05_debug_dump_graph_tensors_to_file(graph_sched, res->get_gf());
        }
    }

    // Debug: dump all pi05_dbg tensors after graph compute
    if (pi05_debug_pre_noise_enabled()) {
        ggml_cgraph * cgf = res->get_gf();
        for (int i = 0; i < ggml_graph_n_nodes(cgf); ++i) {
            ggml_tensor * node = ggml_graph_node(cgf, i);
            if (node != nullptr && node->name[0] != 0 && strncmp(node->name, "pi05_dbg_", 9) == 0) {
                // Strip "pi05_dbg_" prefix so tags match PyTorch side
                pi05_debug_print_tensor_preview(node->name + 9, node);
            }
        }
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    // GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    if (model.arch == LLM_ARCH_PI0) {
        pi0_clear_cross_kv();
    }

    const int64_t n_embd  = hparams.n_embd_inp();
    const int64_t n_vocab = model.vocab.n_tokens();

    // PI0: pad language tokens to 200 slots to match PyTorch/JAX tokenizer behavior.
    llama_batch batch_padded = {};
    std::vector<llama_token>  _pad_tokens;
    std::vector<llama_pos>    _pad_pos;
    std::vector<int32_t>      _pad_n_seq_id;
    std::vector<llama_seq_id> _pad_seq_id_val;
    std::vector<llama_seq_id*> _pad_seq_id;
    std::vector<int8_t>       _pad_logits;
    const llama_batch & batch_to_use = [&]() -> const llama_batch & {
        // NOTE: PyTorch pads language to 200 tokens but masks them in attention,
        // so pad tokens don't affect non-pad KV values. llama.cpp doesn't mask,
        // so we skip padding entirely. The rope offset uses batch_inp.n_tokens (779)
        // which matches PyTorch's sum(prefix_pad_masks).
        return batch_inp;
        if (model.arch != LLM_ARCH_PI0 || !batch_inp.token || batch_inp.img_token_num == 0) {
            return batch_inp;
        }
        const int32_t n_text = batch_inp.n_tokens - batch_inp.img_token_num;
        const int32_t n_text_padded = 200;
        if (n_text >= n_text_padded) {
            return batch_inp;
        }
        const int32_t n_pad = n_text_padded - n_text;
        const int32_t n_total = batch_inp.img_token_num + n_text_padded;

        _pad_tokens.resize(n_total);
        _pad_pos.resize(n_total);
        _pad_n_seq_id.resize(n_total);
        _pad_seq_id_val.resize(n_total, 0);
        _pad_seq_id.resize(n_total);
        _pad_logits.resize(n_total, 1);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            _pad_tokens[i] = batch_inp.token[i];
            _pad_pos[i]    = batch_inp.pos[i];
            _pad_n_seq_id[i] = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;
            _pad_seq_id_val[i] = (batch_inp.seq_id && batch_inp.seq_id[i]) ? batch_inp.seq_id[i][0] : 0;
            _pad_seq_id[i] = &_pad_seq_id_val[i];
        }
        llama_pos last_pos = batch_inp.pos[batch_inp.n_tokens - 1];
        for (int32_t i = 0; i < n_pad; ++i) {
            int32_t idx = batch_inp.n_tokens + i;
            _pad_tokens[idx] = 0;
            _pad_pos[idx]    = last_pos + 1 + i;
            _pad_n_seq_id[idx] = 1;
            _pad_seq_id_val[idx] = 0;
            _pad_seq_id[idx] = &_pad_seq_id_val[idx];
            _pad_logits[idx] = 1;
        }

        batch_padded.n_tokens = n_total;
        batch_padded.token    = _pad_tokens.data();
        batch_padded.embd     = batch_inp.embd;
        batch_padded.embd2    = batch_inp.embd2;
        batch_padded.embd3    = batch_inp.embd3;
        batch_padded.img_token_num = batch_inp.img_token_num;
        batch_padded.single_img_token_num = batch_inp.single_img_token_num;
        batch_padded.pos      = _pad_pos.data();
        batch_padded.n_seq_id = _pad_n_seq_id.data();
        batch_padded.seq_id   = _pad_seq_id.data();
        batch_padded.logits   = _pad_logits.data();
        return batch_padded;
    }();

    if (model.arch == LLM_ARCH_PI0 && batch_to_use.token && batch_to_use.img_token_num > 0) {
        // PyTorch computes AE suffix RoPE positions from sum(prefix_pad_masks),
        // which counts only non-pad tokens (e.g. 768 image + 11 real language = 779).
        // Use batch_inp.n_tokens (pre-padding) to match.
        cross.pi0_rope_prefix_offset = batch_inp.n_tokens;
    }

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_to_use, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true

    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot

    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;
    // if (model.arch == LLM_ARCH_PI0) {
    //     for (int i = 0; i < max_layers; ++i) {
    //         // Create persistent tensors in a dedicated context or the main context.
    //         res->encoded_kv[i] = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd_head, n_head_kv, n_tokens);
    //     }
    // }
    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits = res->get_logits();
    auto * t_embd = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    

    // extract logits
   if (logits && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched_active ? sched_active : sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched_active ? sched_active : sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd != nullptr);

                    GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd, 0, n_tokens*n_embd*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_embd);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd*seq_idx)*sizeof(float), n_embd*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    if (model.arch == LLM_ARCH_PI0) {
        // cross.t_embd = t_embd;
        // auto * t_logits = res->get_logits();
        // auto * t_embd = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        
        auto & t_encoded_kv = res->get_encoded_kv();
        GGML_ASSERT(!t_encoded_kv.empty());
        cross.n_token = t_encoded_kv[0]->ne[2]; // encoder sequence length

        double encode_kv_refresh_ms = 0;
        double encode_kv_d2h_ms       = 0;
        if (llama_pi0_perf_enabled()) {
            auto t_kv_refresh_begin = pi0_perf_clock::now();
            pi0_refresh_encoded_kv_gpu(t_encoded_kv, hparams.n_layer);
            encode_kv_refresh_ms = pi0_perf_elapsed_ms(t_kv_refresh_begin, pi0_perf_clock::now());
        } else {
            pi0_refresh_encoded_kv_gpu(t_encoded_kv, hparams.n_layer);
        }

        if (!cross.pi0_use_gpu_kv) {
            const int64_t n_embd_head = hparams.n_embd_head_v;
            auto t_kv_d2h_begin = llama_pi0_perf_enabled() ? pi0_perf_clock::now() : pi0_perf_clock::time_point{};
            for (uint32_t i = 0; i < hparams.n_layer; i++) {
                if (i >= t_encoded_kv.size() || t_encoded_kv[i] == nullptr) {
                    continue;
                }
                GGML_ASSERT(i < cross.encoded_kv_data.size());
                cross.encoded_kv_data[i].resize(n_embd_head * hparams.n_head_kv(i) * cross.n_token);
                ggml_backend_tensor_get(t_encoded_kv[i], cross.encoded_kv_data[i].data(), 0, ggml_nbytes(t_encoded_kv[i]));
            }
            cross.encoded_kv_dirty = true;
            if (llama_pi0_perf_enabled()) {
                encode_kv_d2h_ms = pi0_perf_elapsed_ms(t_kv_d2h_begin, pi0_perf_clock::now());
            }
        }

        if (llama_pi0_perf_enabled()) {
            fprintf(stderr,
                "[PI0 encode] kv_refresh_gpu=%.2f ms kv_d2h=%.2f ms gpu_kv=%d n_layer=%u n_token=%" PRId64 "\n",
                encode_kv_refresh_ms,
                encode_kv_d2h_ms,
                cross.pi0_use_gpu_kv ? 1 : 0,
                hparams.n_layer,
                cross.n_token);
        }

        pi0_log_encoded_kv_shapes(t_encoded_kv);

        if (pi_model_debug_dump_enabled()) {
            const char * dump_path = pi_model_debug_dump_file();
            std::ofstream ofs(dump_path, std::ios::app);
            if (ofs.is_open()) {
                ofs << "===== pi0 encoded kv tensor dump begin =====\n";
                ofs << "count=" << t_encoded_kv.size()
                    << " n_layer=" << hparams.n_layer
                    << " n_token=" << cross.n_token
                    << " gpu_kv=" << (cross.pi0_use_gpu_kv ? 1 : 0)
                    << "\n";

                size_t mid_even = 0;
                size_t last_even = 0;
                if (!t_encoded_kv.empty()) {
                    mid_even = (t_encoded_kv.size() / 2) & ~(size_t) 1;
                    if (mid_even >= t_encoded_kv.size()) {
                        mid_even = 0;
                    }
                    last_even = (t_encoded_kv.size() - 1) & ~(size_t) 1;
                }

                const size_t slots[] = {0, 1, 2, 3, mid_even, mid_even + 1, last_even, last_even + 1};
                size_t dumped[8] = {};
                size_t n_dumped = 0;
                for (size_t raw_slot : slots) {
                    if (raw_slot >= t_encoded_kv.size() || t_encoded_kv[raw_slot] == nullptr) {
                        continue;
                    }
                    bool seen = false;
                    for (size_t i = 0; i < n_dumped; ++i) {
                        if (dumped[i] == raw_slot) {
                            seen = true;
                            break;
                        }
                    }
                    if (seen) {
                        continue;
                    }
                    dumped[n_dumped++] = raw_slot;
                    ofs << "[llama] name=pi0_dbg_encoded_kv"
                        << " slot=" << raw_slot
                        << " enc_layer=" << (raw_slot / 2)
                        << " kind=" << ((raw_slot % 2) == 0 ? "K" : "V")
                        << "\n";
                    pi05_debug_dump_tensor_to_stream(ofs, t_encoded_kv[raw_slot]);
                }
                ofs << "===== pi0 encoded kv tensor dump end =====\n";
            }
        }

        if (pi05_debug_dump_file_enabled() && cross.encoded_kv_data.size() >= 2) {
            const int64_t kv_ne0 = hparams.n_embd_head_v;
            const int64_t kv_ne1 = hparams.n_head_kv(0);
            const int64_t kv_ne2 = cross.n_token;
            if (!cross.encoded_kv_data[0].empty()) {
                pi05_debug_dump_host_vector_to_file(
                    "pi05_dbg_l0_k_prefix",
                    cross.encoded_kv_data[0],
                    kv_ne0,
                    kv_ne1,
                    kv_ne2,
                    1);
            }
            if (!cross.encoded_kv_data[1].empty()) {
                pi05_debug_dump_host_vector_to_file(
                    "pi05_dbg_l0_v_prefix",
                    cross.encoded_kv_data[1],
                    kv_ne0,
                    kv_ne1,
                    kv_ne2,
                    1);
            }
        }

        if (pi05_debug_pre_noise_enabled()) {
            fprintf(stderr,
                "[DBG][JETSON][pre_noise_prefix] cross.n_token=%" PRId64 " cross.n_embd=%" PRId64 " cross.n_enc=%" PRId64
                " n_layer=%u action_steps=%u action_dim=%u gpu_kv=%d\n",
                cross.n_token,
                cross.n_embd,
                cross.n_enc,
                hparams.n_layer,
                hparams.action_steps,
                hparams.action_dim,
                cross.pi0_use_gpu_kv ? 1 : 0);

            if (!t_encoded_kv.empty()) {
                pi05_debug_print_tensor_preview("pre_noise_encoded_kv_l0_k", t_encoded_kv[0]);
                if (t_encoded_kv.size() > 1) {
                    pi05_debug_print_tensor_preview("pre_noise_encoded_kv_l0_v", t_encoded_kv[1]);
                }
            }

            if (cross.encoded_kv_data.size() > 0 && !cross.encoded_kv_data[0].empty()) {
                pi05_debug_print_float_preview(
                    "pre_noise_cross_encoded_kv_data_l0_k",
                    cross.encoded_kv_data[0].data(),
                    cross.encoded_kv_data[0].size());
            }
            if (cross.encoded_kv_data.size() > 1 && !cross.encoded_kv_data[1].empty()) {
                pi05_debug_print_float_preview(
                    "pre_noise_cross_encoded_kv_data_l0_v",
                    cross.encoded_kv_data[1].data(),
                    cross.encoded_kv_data[1].size());
            }
        }

        const int64_t action_num = hparams.action_steps;
        const int64_t action_dim = hparams.action_dim;
        const int64_t total_elements = action_num * action_dim;
        cross.action = create_normal_noise_cpp11(total_elements,41);
        const bool loaded_action_noise = try_load_pi0_action_noise(cross.action);
        if (!loaded_action_noise) {
            maybe_dump_pi0_action_noise(cross.action);
        }
        if (pi05_debug_dump_file_enabled()) {
            const char * dump_path = pi_model_debug_dump_file();
            std::ofstream ofs(dump_path, std::ios::app);
            if (ofs.is_open()) {
                ofs << "[llama] pi_model=" << pi_model_kind_name(pi_model_kind_from_env())
                    << " action_noise_source=" << (loaded_action_noise ? "PI0_ACTION_NOISE_BIN" : "cpp_rng_seed41")
                    << " action_steps=" << action_num
                    << " action_dim=" << action_dim
                    << " total_elements=" << total_elements << "\n";
            }
            pi05_debug_dump_host_vector_to_file(
                "pi_model_llama_initial_action_noise",
                cross.action,
                action_dim,
                action_num,
                1,
                1);
        }
        cross.pi0_action_input = cross.action;
        cross.pi0_state_result.clear();
        if (cross.state.empty()) {
            cross.state = std::vector<float>(action_dim, 0.0f);  // Default to a zero state vector.
        }
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

int llama_context::decode(const llama_batch & batch_inp) {
    // GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();
    const int64_t n_embd  = hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all = cparams.embeddings;

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();
    output_swaps.clear();

    bool did_optimize = false;

    llama_memory_context_ptr mctx;

    // PI0 decoder uses cross-KV on GPU, not llama_kv_cache — skip slot search in init_batch.
    if (model.arch != LLM_ARCH_PI0) {
        // handle any pending shifts/copies
        memory_update(false);

        while (true) {
            mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
            if (!mctx) {
                return -2;
            }

            switch (mctx->get_status()) {
                case LLAMA_MEMORY_STATUS_SUCCESS:
                    {
                    } break;
                case LLAMA_MEMORY_STATUS_NO_UPDATE:
                    {
                        LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                        return -2;
                    }
                case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                    {
                        if (!did_optimize) {
                            did_optimize = true;

                            if (memory_update(true)) {
                                LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                                continue;
                            }
                        }

                        LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                        return 1;
                    }
                case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                    {
                        LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                        return -2;
                    }
            }

            break;
        }
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    int64_t n_outputs_prev = 0;
    if (model.arch == LLM_ARCH_PI0) {
        double memory_init_batch_ms = 0.0;
        auto t_memory_init_begin = llama_pi0_perf_enabled() ? pi0_perf_clock::now() : pi0_perf_clock::time_point{};
        const llama_ubatch ubatch = balloc->split_simple(n_tokens_all);
        if (llama_pi0_perf_enabled()) {
            memory_init_batch_ms = pi0_perf_elapsed_ms(t_memory_init_begin, pi0_perf_clock::now());
        }

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }
#if PI0_ENABLE_AE_STEP_DEBUG
        {
            PI0_NVTX_SCOPE("pi0/ae/init/print_action");
            print_vector(cross.action, "action");
        }
#endif
        cross.pi0_action_input = cross.action;
        const int64_t action_num = hparams.action_steps;
        const int64_t action_dim = hparams.action_dim;
        const bool weights_are_pi05 = model.time_mlp_in != nullptr &&
                                      model.time_mlp_out != nullptr &&
                                      model.ae_output_norm_dense != nullptr;
        const pi_model_kind pi_model = pi_model_kind_from_env();
        const bool is_pi05 = weights_are_pi05 && !pi_model_kind_is_pi0(pi_model);
        const int64_t total_action_elements = (is_pi05 ? action_num : (action_num + 1)) * action_dim;
        const float dt = -1.0f / (float) hparams.inference_steps;
        const bool pi0_perf = llama_pi0_perf_enabled();
        double precompute_ms = 0.0;
        {
            PI0_NVTX_SCOPE("pi0/ae/init/precompute_time_embeddings");
            auto t_precompute_begin = pi0_perf_clock::now();
            const int64_t half_dim = hparams.n_embd_ae / 2;
            const int64_t embd_ae = hparams.n_embd_ae;
            const int64_t total_elements = action_num * embd_ae;
            const double min_period = 0.004f;
            const double max_period = 4.0f;

            const bool cache_valid =
                cross.ae_time_cache_inference_steps == hparams.inference_steps &&
                cross.ae_time_cache_action_steps == action_num &&
                cross.ae_time_cache_n_embd_ae == embd_ae &&
                cross.ae_time_embeddings.size() == (size_t) hparams.inference_steps * total_elements &&
                cross.ae_time_cond.size() == (size_t) hparams.inference_steps * embd_ae;

            if (!cache_valid) {
                cross.ae_time_embeddings.resize((size_t) hparams.inference_steps * total_elements);
                cross.ae_time_cond.resize((size_t) hparams.inference_steps * embd_ae);

                std::vector<double> scaling_factors(half_dim);
                for (int32_t step = 0; step < hparams.inference_steps; ++step) {
                    const float time_step = float(1 - step/float(hparams.inference_steps));
                    for (int64_t i = 0; i < half_dim; ++i) {
                        const double fraction = (half_dim > 1) ? (double) i / (half_dim - 1) : 0.0f;
                        const double period = min_period * std::pow(max_period / min_period, fraction);
                        scaling_factors[i] = (1.0f / period) * 2.0f * M_PI * time_step;
                    }

                    float * cond_data = cross.ae_time_cond.data() + (size_t) step * embd_ae;
                    for (int64_t i = 0; i < half_dim; ++i) {
                        const double scaled_time = scaling_factors[i];
                        cond_data[i] = std::sin(scaled_time);
                        cond_data[half_dim + i] = std::cos(scaled_time);
                    }

                    float * time_data = cross.ae_time_embeddings.data() + (size_t) step * total_elements;
                    for (int64_t t = 0; t < action_num; ++t) {
                        std::memcpy(time_data + t * embd_ae, cond_data, sizeof(float) * embd_ae);
                    }
                }

                cross.ae_time_cache_inference_steps = hparams.inference_steps;
                cross.ae_time_cache_action_steps = action_num;
                cross.ae_time_cache_n_embd_ae = embd_ae;
            }
            if (pi0_perf) {
                precompute_ms = pi0_perf_elapsed_ms(t_precompute_begin, pi0_perf_clock::now());
            }
        }
        const int32_t unroll_cfg = llama_pi0_decode_unroll_steps(is_pi05, hparams.inference_steps);
        if (pi0_perf) {
            fprintf(stderr,
                "[PI0 decode] inference_steps=%d unroll_cfg=%d memory_init_batch=%.2f ms precompute_time_emb=%.2f ms gpu_kv=%d n_tokens=%u\n",
                hparams.inference_steps,
                unroll_cfg,
                memory_init_batch_ms,
                precompute_ms,
                cross.pi0_use_gpu_kv ? 1 : 0,
                n_tokens_all);
        }
        PI0_REUSE_LOG_LINE("=== pi0 ae decode begin ===");
        cross.pi0_cross_kv_inputs_ready  = false;
        cross.pi0_decode_attn_mask_ready = false;
        std::vector<float> action_data(total_action_elements);
        if (is_pi05) {
            cross.pi0_state_result.clear();
        }

        double perf_sum_total_ms = 0.0;
        double perf_sum_ubatch_ms = 0.0;
        double perf_sum_set_inputs_ms = 0.0;
        double perf_sum_compute_ms = 0.0;

        for (int32_t i = 0; i < hparams.inference_steps; ) {
            const int32_t steps_left = hparams.inference_steps - i;
            const int32_t unroll = (unroll_cfg >= 2 && steps_left >= 2)
                ? std::min(unroll_cfg, steps_left)
                : 1;
            cross.pi0_decode_unroll          = unroll;
            cross.pi0_action_out_accumulated = (is_pi05 && unroll >= 2);

            PI0_NVTX_SCOPE("pi0/ae/step_total");
            cross.time_step = float(1 - i / float(hparams.inference_steps));
            cross.time_step_index = i;
            const bool kv_skip   = cross.pi0_cross_kv_inputs_ready;
            const bool mask_skip = cross.pi0_decode_attn_mask_ready;
#if PI0_ENABLE_REUSE_LOG
            const auto reused_before = n_reused;
            auto step_decode_start = std::chrono::high_resolution_clock::now();
#endif
            const int32_t reused_before_perf = n_reused;

            auto t_step_begin = pi0_perf_clock::now();

            if (is_pi05 && pi05_debug_dump_file_enabled()) {
                pi05_debug_dump_host_vector_to_file(
                    "action_in",
                    cross.action,
                    action_dim,
                    action_num,
                    1,
                    1);
            }

            ggml_status status;
            const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, nullptr, status);
            if (!res) {
                switch (status) {
                    case GGML_STATUS_ABORTED:      return  2;
                    case GGML_STATUS_ALLOC_FAILED: return -2;
                    case GGML_STATUS_FAILED:       return -3;
                    case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
                }
            }

            auto * action = res->get_action();
            const size_t action_nbytes = (size_t) total_action_elements * ggml_element_size(action);

            auto t_gpu_wait_begin = pi0_perf_clock::now();
            {
                PI0_NVTX_SCOPE("pi0/ae/post_compute/gpu_wait_action");
                pi0_fetch_action_to_host(sched_active ? sched_active : sched.get(), action, action_data.data(), action_nbytes);
            }
            auto t_gpu_wait_end = pi0_perf_clock::now();

            if (is_pi05 && pi05_debug_dump_file_enabled()) {
                pi05_debug_dump_host_vector_to_file(
                    "pi05_dbg_action_out_proj",
                    action_data,
                    action_dim,
                    action_num,
                    1,
                    1);
            }
            if (!is_pi05 && pi05_debug_dump_file_enabled()) {
                const int64_t pi0_rows = action_num + 1;
                std::ofstream ofs(pi_model_debug_dump_file(), std::ios::app);
                if (ofs.is_open()) {
                    ofs << "[llama] name=pi0_dbg_decode_step step=" << i
                        << " time_step=" << cross.time_step
                        << " action_rows=" << pi0_rows
                        << " action_dim=" << action_dim
                        << " values_are=[state,delta_actions]\n";
                }
                std::string action_data_name = "pi0_dbg_decode_action_data_step" + std::to_string(i);
                pi05_debug_dump_host_vector_to_file(
                    action_data_name.c_str(),
                    action_data,
                    action_dim,
                    pi0_rows,
                    1,
                    1);
            }

            double update_state_ms = 0.0;
            if (!is_pi05) {
                auto t_update_state_begin = pi0_perf_clock::now();
                PI0_NVTX_SCOPE("pi0/ae/post_compute/update_state_result");
                cross.pi0_state_result.assign(action_data.begin(), action_data.begin() + action_dim);
                update_state_ms = pi0_perf_elapsed_ms(t_update_state_begin, pi0_perf_clock::now());
            }

#if PI0_ENABLE_AE_STEP_DEBUG
            {
                PI0_NVTX_SCOPE("pi0/ae/post_compute/print_state_action");
                if (is_pi05) {
                    print_vector(std::vector<float>(action_data.begin(), action_data.begin() + std::min<int64_t>(action_dim, action_data.size())), "action_step0");
                } else {
                    print_vector(std::vector<float>(action_data.begin(), action_data.begin()+32), "state");
                    print_vector(std::vector<float>(action_data.begin()+32, action_data.end()), "action");
                }
            }
#endif

            auto t_update_action_begin = pi0_perf_clock::now();
            {
                PI0_NVTX_SCOPE("pi0/ae/post_compute/update_action");
                GGML_ASSERT((int64_t) cross.action.size() == action_dim * action_num);
                if (is_pi05) {
                    if (cross.pi0_action_out_accumulated) {
                        GGML_ASSERT((int64_t) action_data.size() == action_dim * action_num);
                        pi0_fetch_action_to_host(sched_active ? sched_active : sched.get(), action, action_data.data(), action_nbytes);
                        std::memcpy(cross.action.data(), action_data.data(), action_nbytes);
                    } else {
                    const size_t action_tensor_nbytes = ggml_nbytes(action);
                    const size_t action_tensor_elem_size = ggml_element_size(action);
                    const size_t expected_action_bytes = (size_t) total_action_elements * action_tensor_elem_size;
                    GGML_ASSERT(action_tensor_nbytes >= expected_action_bytes);

#if 0
                    int first_non_finite_action_data = -1;
                    float action_data_min = std::numeric_limits<float>::infinity();
                    float action_data_max = -std::numeric_limits<float>::infinity();
                    for (int j = 0; j < action_dim * action_num; ++j) {
                        action_data_min = std::min(action_data_min, action_data[j]);
                        action_data_max = std::max(action_data_max, action_data[j]);
                        if (!std::isfinite(action_data[j])) {
                            first_non_finite_action_data = j;
                            break;
                        }
                    }
                    if (first_non_finite_action_data >= 0) {
                        GGML_UNUSED(first_non_finite_action_data);
                    } else {
                        const int preview = std::min<int64_t>(8, action_dim * action_num);
                        // printf("pi05_debug: action_data finite range=[%f,%f] preview=", action_data_min, action_data_max);
                        for (int j = 0; j < preview; ++j) {
                            // printf(j + 1 == preview ? "%f\n" : "%f,", action_data[j]);
                        }
                    }

                    int first_non_finite_prev_action = -1;
                    float prev_action_min = std::numeric_limits<float>::infinity();
                    float prev_action_max = -std::numeric_limits<float>::infinity();
                    for (int j = 0; j < action_dim * action_num; ++j) {
                        prev_action_min = std::min(prev_action_min, cross.action[j]);
                        prev_action_max = std::max(prev_action_max, cross.action[j]);
                        if (!std::isfinite(cross.action[j])) {
                            first_non_finite_prev_action = j;
                            break;
                        }
                    }
                    if (first_non_finite_prev_action >= 0) {
                        GGML_UNUSED(first_non_finite_prev_action);
                    } else {
                        const int preview = std::min<int64_t>(8, action_dim * action_num);
                        // printf("pi05_debug: cross.action before update finite range=[%f,%f] preview=", prev_action_min, prev_action_max);
                        for (int j = 0; j < preview; ++j) {
                            // printf(j + 1 == preview ? "%f\n" : "%f,", cross.action[j]);
                        }
                    }
#endif

                    for (int j = 0; j < action_dim * action_num; ++j) {
                        cross.action[j] = cross.action[j] + dt * action_data[j];
                    }
                    if (pi05_debug_dump_file_enabled()) {
                        pi05_debug_dump_host_vector_to_file(
                            "action_after_update",
                            cross.action,
                            action_dim,
                            action_num,
                            1,
                            1);
                    }
                    }
#if 0
                    int first_non_finite_updated_action = -1;
                    float updated_action_min = std::numeric_limits<float>::infinity();
                    float updated_action_max = -std::numeric_limits<float>::infinity();
                    if (first_non_finite_updated_action >= 0) {
                        GGML_UNUSED(first_non_finite_updated_action);
                    } else {
                        const int preview = std::min<int64_t>(8, action_dim * action_num);
                        // printf("pi05_debug: cross.action after update finite range=[%f,%f] preview=", updated_action_min, updated_action_max);
                        for (int j = 0; j < preview; ++j) {
                            // printf(j + 1 == preview ? "%f\n" : "%f,", cross.action[j]);
                        }
                    }
                    if (i + 1 == hparams.inference_steps) {
                        pi05_debug_dump_host_vector_to_file(
                            "action_final",
                            cross.action,
                            action_num,
                            action_dim,
                            1,
                            1);
                    }
#endif
                } else {
                    for (int j = 0; j < action_dim*action_num; j++) {
                        cross.action[j] = cross.action[j] - action_data[action_dim+j];
                    }
                    if (pi05_debug_dump_file_enabled()) {
                        std::string action_after_name = "pi0_dbg_action_after_step" + std::to_string(i);
                        pi05_debug_dump_host_vector_to_file(
                            action_after_name.c_str(),
                            cross.action,
                            action_dim,
                            action_num,
                            1,
                            1);
                    }
                }
            }
            const double update_action_ms = pi0_perf_elapsed_ms(t_update_action_begin, pi0_perf_clock::now());

            const double step_total_ms = pi0_perf_elapsed_ms(t_step_begin, pi0_perf_clock::now());
            const double gpu_wait_action_ms = pi0_perf_elapsed_ms(t_gpu_wait_begin, t_gpu_wait_end);

            if (pi0_perf) {
                const auto & src = pi0_last_decode_ubatch_perf;
                pi0_decode_ubatch_perf_log ub = {
                    src.mem_apply_ms,
                    src.graph_build_ms,
                    src.graph_alloc_ms,
                    src.set_inputs_ms,
                    src.set_static_ms,
                    src.set_dynamic_ms,
                    src.set_kv_ms,
                    src.set_kv_skipped,
                    src.graph_compute_ms,
                    src.graph_reused,
                };
                const double ubatch_ms = ub.mem_apply_ms + ub.graph_build_ms + ub.graph_alloc_ms +
                    ub.set_inputs_ms + ub.graph_compute_ms;

                perf_sum_total_ms += step_total_ms;
                perf_sum_ubatch_ms += ubatch_ms;
                perf_sum_set_inputs_ms += ub.set_inputs_ms;
                perf_sum_compute_ms += ub.graph_compute_ms;

                pi0_perf_log_decode_step(
                    i,
                    hparams.inference_steps,
                    cross.time_step,
                    kv_skip,
                    cross.pi0_use_gpu_kv,
                    step_total_ms,
                    i == 0 ? precompute_ms : 0.0,
                    ub,
                    gpu_wait_action_ms,
                    update_state_ms,
                    update_action_ms,
                    mask_skip,
                    n_reused - reused_before_perf);
            }

            i += unroll;

#if PI0_ENABLE_AE_STEP_DEBUG
            {
                PI0_NVTX_SCOPE("pi0/ae/post_compute/print_action_final");
                print_vector(cross.action, "action_final");
            }
#endif

#if PI0_ENABLE_REUSE_LOG
            const auto reused_after = n_reused;
            auto step_decode_end = std::chrono::high_resolution_clock::now();
            double step_decode_ms = std::chrono::duration<double, std::milli>(step_decode_end - step_decode_start).count();
            PI0_REUSE_LOG_STREAM(
                "step=" << i <<
                " reused_delta=" << (reused_after - reused_before) <<
                " reused_total=" << n_reused <<
                " step_ms=" << step_decode_ms);
#endif
        }

        if (pi0_perf) {
            fprintf(stderr,
                "[PI0 decode summary] steps=%d total=%.2f ms ubatch_sum=%.2f ms set_inputs_sum=%.2f ms "
                "graph_compute_sum=%.2f ms graphs_reused_total=%d\n",
                hparams.inference_steps,
                perf_sum_total_ms,
                perf_sum_ubatch_ms,
                perf_sum_set_inputs_ms,
                perf_sum_compute_ms,
                n_reused);
        }

        pi0_clear_cross_kv();
        t_p_eval_us += ggml_time_us() - t_compute_start_us;
        // break;
        
    }
    else{
        do {
            const auto & ubatch = mctx->get_ubatch();
            
            // count the outputs in this ubatch
            {
                int32_t n_outputs_new = 0;

                if (n_outputs_all == n_tokens_all) {
                    n_outputs_new = ubatch.n_tokens;
                } else {
                    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                        n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                    }
                }

                // needs to happen before the graph is built
                n_outputs = n_outputs_new;
            }

            // auto start = std::chrono::high_resolution_clock::now();
            ggml_status status;
            const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, mctx.get(), status);

            if (!res) {
                // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
                llama_pos pos_min[LLAMA_MAX_SEQ];
                for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                    pos_min[s] = std::numeric_limits<llama_pos>::max();
                }

                for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                    const auto & seq_id = ubatch.seq_id[i][0];

                    pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
                }

                for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                    if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                        continue;
                    }

                    LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                    memory->seq_rm(s, pos_min[s], -1);
                }

                switch (status) {
                    case GGML_STATUS_ABORTED:      return  2;
                    case GGML_STATUS_ALLOC_FAILED: return -2;
                    case GGML_STATUS_FAILED:       return -3;
                    case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
                }
            }

            // plot the computation graph in dot format (for debugging purposes)
            //if (n_past%100 == 0) {
            //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
            //}

            auto * t_logits = res->get_logits();
            auto * t_embd   = cparams.embeddings ? res->get_embd() : nullptr;

            if (t_embd && res->get_embd_pooled()) {
                t_embd = res->get_embd_pooled();
            }

            // extract logits
            if (t_logits && n_outputs > 0) {
                ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched_active ? sched_active : sched.get(), t_logits);
                GGML_ASSERT(backend_res != nullptr);
                GGML_ASSERT(logits != nullptr);

                float * logits_out = logits + n_outputs_prev*n_vocab;

                if (n_outputs) {
                    GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                    GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits_size);
                    ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
                }
            }

            // extract embeddings
            if (t_embd && n_outputs > 0) {
                ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched_active ? sched_active : sched.get(), t_embd);
                GGML_ASSERT(backend_embd != nullptr);

                switch (cparams.pooling_type) {
                    case LLAMA_POOLING_TYPE_NONE:
                        {
                            // extract token embeddings
                            GGML_ASSERT(embd != nullptr);
                            float * embd_out = embd + n_outputs_prev*n_embd;

                            if (n_outputs) {
                                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                                GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd <= (int64_t) embd_size);
                                ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd*sizeof(float));
                            }
                        } break;
                    case LLAMA_POOLING_TYPE_MEAN:
                    case LLAMA_POOLING_TYPE_CLS:
                    case LLAMA_POOLING_TYPE_LAST:
                        {
                            // extract sequence embeddings (cleared before processing each batch)
                            auto & embd_seq_out = embd_seq;

                            for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                                const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                                const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                                embd_seq_out[seq_id].resize(n_embd);
                                ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd*seq_idx)*sizeof(float), n_embd*sizeof(float));
                            }
                        } break;
                    case LLAMA_POOLING_TYPE_RANK:
                        {
                            // extract the rerank score - n_cls_out floats per sequence
                            auto & embd_seq_out = embd_seq;

                            const uint32_t n_cls_out = hparams.n_cls_out;

                            for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                                const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                                const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                                embd_seq_out[seq_id].resize(n_cls_out);
                                ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                            }
                        } break;
                    case LLAMA_POOLING_TYPE_UNSPECIFIED:
                        {
                            GGML_ABORT("unknown pooling type");
                        }
                }
            }

            n_outputs_prev += n_outputs;
        } while (mctx->next());
    }
    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch = cparams.n_batch;
    const auto n_vocab = vocab.n_tokens();
    const auto n_embd  = hparams.n_embd;

    bool has_logits = true;
    bool has_embd   = cparams.embeddings;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    logits_size = has_logits ? n_vocab*n_outputs_max : 0;
    embd_size   = has_embd   ?  n_embd*n_outputs_max : 0;

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  = (logits_size + embd_size) * sizeof(float);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_INFO("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            buf_output = nullptr;
            logits = nullptr;
            embd = nullptr;
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    logits = has_logits ? output_base               : nullptr;
    embd   = has_embd   ? output_base + logits_size : nullptr;

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    return n_outputs_max;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab = model.vocab.n_tokens();
    const uint64_t n_embd  = model.hparams.n_embd;

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits_size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits[i0*n_vocab + k], logits[i1*n_vocab + k]);
            }
        }

        if (embd_size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd[i0*n_embd + k], embd[i1*n_embd + k]);
            }
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes() const {
    if (model.arch == LLM_ARCH_QWEN3NEXT) {
        return std::max<uint32_t>(8192u, 32u*model.n_tensors());
    }
    return std::max<uint32_t>(65536u, 32u*model.n_tensors());
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

ggml_cgraph * llama_context::graph_reserve(uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only) {

    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    // printf("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);

    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        n_outputs = std::min(n_outputs, n_tokens);

        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannnot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, LLM_GRAPH_TYPE_DEFAULT);

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    // During graph reservation.

    if (split_only) {

        ggml_backend_sched_split_graph(sched.get(), gf);
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {

        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
            llm_graph_type   gtype,
        ggml_backend_sched_t graph_sched) const {
    if (!graph_sched) {
        graph_sched = sched.get();
    }
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ graph_sched,
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ &cvec,
        /*.loras       =*/ &loras,
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
        /*.pi0_decode_unroll =*/ (model.arch == LLM_ARCH_PI0 && gtype == LLM_GRAPH_TYPE_DECODER ? cross.pi0_decode_unroll : 1),
    };
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched,
   ggml_backend_sched_t   graph_sched) {
    last_graph_compute = gf;
    if (!graph_sched) {
        graph_sched = sched.get();
    }

    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    {
        PI0_NVTX_SCOPE("pi0/graph_compute/set_threadpool");
        if (backend_cpu != nullptr) {
            auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
            auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
            if (set_threadpool_fn) {
                set_threadpool_fn(backend_cpu, tp);
            }
        }
    }

    {
        PI0_NVTX_SCOPE("pi0/graph_compute/set_n_threads");
        for (const auto & set_n_threads_fn : set_n_threads_fns) {
            set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
        }
    }

    {
        PI0_NVTX_SCOPE("pi0/graph_compute/validate_inputs");
        if (!graph_sched) {
            LLAMA_LOG_ERROR("error: backend scheduler is null\n");
        }

        ggml_backend_sched_t sched_ptr = graph_sched;
        if (sched_ptr == nullptr) {
            LLAMA_LOG_ERROR("error: failed to get raw pointer from sched\n");
        }

        if (gf == nullptr) {
            LLAMA_LOG_ERROR("error: compute graph (gf) is null\n");
        }
    }

#if PI0_ENABLE_AE_STEP_DEBUG
    {
        PI0_NVTX_SCOPE("pi0/graph_compute/debug_scan_nodes");
        // printf("gf_n_nodes: %d\n", ggml_graph_n_nodes(gf));
        // printf("Checking graph nodes...\n");
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            struct ggml_tensor * node = ggml_graph_node(gf, i);
            if (node == nullptr) {
                // printf("Node %d is NULL!\n", i);
                continue;
            }
            if (node->data != nullptr && (uintptr_t)node->data < 0x1000) {
                // printf("CRITICAL: Node %d (%s) has corrupt data pointer: %p\n", i, node->name, node->data);
            }
        }
    }
#endif

    PI0_NVTX_SCOPE("pi0/graph_compute/submit_async");
    sched_active = graph_sched;
    auto status = ggml_backend_sched_graph_compute_async(graph_sched, gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }
    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        if (!cparams.offload_kqv) {
            if (strcmp(name, "kqv_merged_cont") == 0) {
                // all nodes between the KV store and the attention output are run on the CPU
                ggml_backend_sched_set_tensor_backend(sched_active ? sched_active : sched.get(), cur, backend_cpu);
            }
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.params.n_gpu_layers > (int) model.hparams.n_layer;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched_active ? sched_active : sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy() = default;

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(const ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    size_t size_written = 0;
};

class llama_io_write_buffer : public llama_io_write_i {
public:
    llama_io_write_buffer(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ggml_backend_tensor_get(tensor, ptr, offset, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;
};

class llama_io_read_buffer : public llama_io_read_i {
public:
    llama_io_read_buffer(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    const uint8_t * read(size_t size) override {
        const uint8_t * base_ptr = ptr;
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ptr += size;
        size_read += size;
        buf_size -= size;
        return base_ptr;
    }

    void read_to(void * dst, size_t size) override {
        memcpy(dst, read(size), size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read_to(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    const uint8_t * read(size_t size) override {
        temp_buffer.resize(size);
        read_to(temp_buffer.data(), size);
        return temp_buffer.data();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io;
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_buffer io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io;
    try {
        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    llama_io_read_buffer io(src, size);
    try {
        return state_seq_read_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    // write output ids
    {
        LLAMA_LOG_DEBUG("%s: - writing output ids\n", __func__);

        const auto n_outputs    = this->n_outputs;
        const auto & output_ids = this->output_ids;

        std::vector<int32_t> w_output_pos;

        w_output_pos.resize(n_outputs);

        // build a more compact representation of the output ids
        for (size_t i = 0; i < n_batch(); ++i) {
            // map an output id to a position in the batch
            int64_t pos = output_ids[i];
            if (pos >= 0) {
                GGML_ASSERT(pos < n_outputs);
                w_output_pos[pos] = i;
            }
        }

        io.write(&n_outputs, sizeof(n_outputs));

        if (n_outputs) {
            io.write(w_output_pos.data(), n_outputs * sizeof(int32_t));
        }
    }

    // write logits
    {
        LLAMA_LOG_DEBUG("%s: - writing logits\n", __func__);

        const uint64_t logits_size = std::min((uint64_t) this->logits_size, (uint64_t) n_outputs * model.vocab.n_tokens());

        io.write(&logits_size, sizeof(logits_size));

        if (logits_size) {
            io.write(logits, logits_size * sizeof(float));
        }
    }

    // write embeddings
    {
        LLAMA_LOG_DEBUG("%s: - writing embeddings\n", __func__);

        const uint64_t embd_size = std::min((uint64_t) this->embd_size, (uint64_t) n_outputs * model.hparams.n_embd);

        io.write(&embd_size, sizeof(embd_size));

        if (embd_size) {
            io.write(embd, embd_size * sizeof(float));
        }
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    // read output ids
    {
        LLAMA_LOG_DEBUG("%s: - reading output ids\n", __func__);

        auto n_outputs = this->n_outputs;
        io.read_to(&n_outputs, sizeof(n_outputs));

        if (n_outputs > output_reserve(n_outputs)) {
            throw std::runtime_error("could not reserve outputs");
        }

        std::vector<int32_t> output_pos;

        if (n_outputs) {
            output_pos.resize(n_outputs);
            io.read_to(output_pos.data(), n_outputs * sizeof(int32_t));

            for (int32_t i = 0; i < (int32_t) output_pos.size(); ++i) {
                int32_t id = output_pos[i];
                if ((uint32_t) id >= n_batch()) {
                    throw std::runtime_error(format("invalid output id, %d does not fit in batch size of %u", id, n_batch()));
                }
                this->output_ids[id] = i;
            }

            this->n_outputs = n_outputs;
        }
    }

    // read logits
    {
        LLAMA_LOG_DEBUG("%s: - reading logits\n", __func__);

        uint64_t logits_size;
        io.read_to(&logits_size, sizeof(logits_size));

        if (this->logits_size < logits_size) {
            throw std::runtime_error("logits buffer too small");
        }

        if (logits_size) {
            io.read_to(this->logits, logits_size * sizeof(float));
        }
    }

    // read embeddings
    {
        LLAMA_LOG_DEBUG("%s: - reading embeddings\n", __func__);

        uint64_t embd_size;
        io.read_to(&embd_size, sizeof(embd_size));

        if (this->embd_size < embd_size) {
            throw std::runtime_error("embeddings buffer too small");
        }

        if (embd_size) {
            io.read_to(this->embd, embd_size * sizeof(float));
        }
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & buft_size : model.memory_breakdown()) {
        ret[buft_size.first].model += buft_size.second;
    }
    for (const auto & buft_size : memory->memory_breakdown()) {
        ret[buft_size.first].context += buft_size.second;
    }
    auto add_sched_buffers = [&](ggml_backend_sched_t graph_sched) {
        if (!graph_sched) {
            return;
        }
        for (const auto & backend_ptr : backends) {
            ggml_backend_t backend = backend_ptr.get();
            ret[ggml_backend_sched_get_buffer_type(graph_sched, backend)].compute +=
                ggml_backend_sched_get_buffer_size(graph_sched, backend);
        }
    };

    add_sched_buffers(sched.get());
    add_sched_buffers(sched_pi0_enc.get());
    add_sched_buffers(sched_pi0_dec.get());
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(res, ubatch, mctx.get(), LLM_GRAPH_TYPE_DEFAULT);

            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 2048,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 1024,
        /*.n_seq_max                   =*/ 4096,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        if (model->hparams.n_embd_head_k % blck_size != 0) {
            LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k);
            return nullptr;
        }
    }

    if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        if (model->hparams.n_embd_head_v % blck_size != 0) {
            LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v);
            return nullptr;
        }
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_logits_ith(i);
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

const float * llama_get_pi0_action(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_pi0_action();
}

const float * llama_get_pi0_state(llama_context * ctx) {
    return ctx->get_pi0_state();
}

int32_t llama_get_pi0_state_size(llama_context * ctx) {
    return ctx->get_pi0_state_size();
}

const float * llama_get_pi0_action_input(llama_context * ctx) {
    return ctx->get_pi0_action_input();
}

int32_t llama_get_pi0_action_dim(llama_context * ctx) {
    return ctx->get_pi0_action_dim();
}

int32_t llama_get_pi0_action_steps(llama_context * ctx) {
    return ctx->get_pi0_action_steps();
}

void llama_set_pi0_state(llama_context * ctx, const float * state_array, size_t size) {
    if (ctx == nullptr) {
        return;
    }

    if (state_array == nullptr || size == 0) {
        ctx->set_state(std::vector<float>());
        return;
    }

    ctx->set_state(std::vector<float>(state_array, state_array + size));
}

// llama adapter API

int32_t llama_set_adapter_lora(
            llama_context * ctx,
            llama_adapter_lora * adapter,
            float scale) {
    ctx->set_adapter_lora(adapter, scale);

    return 0;
}

int32_t llama_rm_adapter_lora(
            llama_context * ctx,
            llama_adapter_lora * adapter) {
    bool res = ctx->rm_adapter_lora(adapter);

    return res ? 0 : -1;
}

void llama_clear_adapter_lora(llama_context * ctx) {
    ctx->clear_adapter_lora();
}

int32_t llama_apply_adapter_cvec(
        llama_context * ctx,
                 const float * data,
                      size_t   len,
                     int32_t   n_embd,
                     int32_t   il_start,
                     int32_t   il_end) {
    bool res = ctx->apply_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    return ctx->get_memory();
}

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}

size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

void llama_memory_breakdown_print(const struct llama_context * ctx) {
    const std::vector<ggml_backend_dev_t> & devices = ctx->get_model().devices;

    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> memory_breakdown = ctx->memory_breakdown();

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t          dev = devices[i];
        llama_memory_breakdown_data mb  = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const size_t unaccounted = total - self - free;

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / MiB)});
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted
        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LLAMA_LOG_INFO(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}
