#include "models.h"

void llama_model_gr00t_n1d7::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_NUM_DEEPSTACK_LAYERS, hparams.n_deepstack_layers, false);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key("gr00t-n1d7.state_dimension",              hparams.gr00t_state_dim);
    ml.get_key("gr00t-n1d7.action_dimension",             hparams.action_dim);
    ml.get_key("gr00t-n1d7.action_horizon",               hparams.action_steps);
    ml.get_key("gr00t-n1d7.inference_steps",              hparams.inference_steps);
    ml.get_key("gr00t-n1d7.timestep_buckets",             hparams.gr00t_timestep_buckets);
    ml.get_key("gr00t-n1d7.embodiment_count",             hparams.gr00t_embodiment_count);
    ml.get_key("gr00t-n1d7.dit_block_count",              hparams.gr00t_dit_blocks);
    ml.get_key("gr00t-n1d7.dit_head_count",               hparams.gr00t_dit_heads);
    ml.get_key("gr00t-n1d7.dit_head_dimension",           hparams.gr00t_dit_head_dim);
    ml.get_key("gr00t-n1d7.dit_output_dimension",         hparams.gr00t_dit_output_dim);
    ml.get_key("gr00t-n1d7.vl_self_attention_block_count", hparams.gr00t_vl_blocks);

    hparams.n_embd_ae = hparams.gr00t_dit_heads * hparams.gr00t_dit_head_dim;
    type = LLM_TYPE_3B;
}

void llama_model_gr00t_n1d7::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    // Qwen3-VL language backbone (the vision projector remains the companion mmproj GGUF).
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, n_ff}, 0);
    }

    const int64_t ae = hparams.n_embd_ae;
    const int64_t hidden = hparams.gr00t_dit_output_dim;
    const int64_t state = hparams.gr00t_state_dim;
    const int64_t action = hparams.action_dim;
    const int64_t embodiments = hparams.gr00t_embodiment_count;
    auto add = [&](llm_tensor id, const char * suffix, std::initializer_list<int64_t> shape, int bid = -1, int xid = -1) {
        action_tensors.push_back(create_tensor(tn(id, suffix, bid, xid), shape, 0));
    };
    auto linear = [&](llm_tensor id, int64_t in, int64_t out, int bid = -1, int xid = -1) {
        add(id, "weight", {in, out}, bid, xid);
        add(id, "bias", {out}, bid, xid);
    };

    add(LLM_TENSOR_GR00T_VLLN, "weight", {n_embd});
    add(LLM_TENSOR_GR00T_VLLN, "bias", {n_embd});
    for (uint32_t i = 0; i < hparams.gr00t_vl_blocks; ++i) {
        const int bid = i % n_layer;
        linear(LLM_TENSOR_GR00T_VL_ATTN_Q, n_embd, n_embd, bid, i);
        linear(LLM_TENSOR_GR00T_VL_ATTN_K, n_embd, n_embd, bid, i);
        linear(LLM_TENSOR_GR00T_VL_ATTN_V, n_embd, n_embd, bid, i);
        linear(LLM_TENSOR_GR00T_VL_ATTN_OUT, n_embd, n_embd, bid, i);
        add(LLM_TENSOR_GR00T_VL_NORM1, "weight", {n_embd}, bid, i);
        add(LLM_TENSOR_GR00T_VL_NORM1, "bias", {n_embd}, bid, i);
        add(LLM_TENSOR_GR00T_VL_NORM3, "weight", {n_embd}, bid, i);
        add(LLM_TENSOR_GR00T_VL_NORM3, "bias", {n_embd}, bid, i);
        linear(LLM_TENSOR_GR00T_VL_FFN_IN, n_embd, 4*n_embd, bid, i);
        linear(LLM_TENSOR_GR00T_VL_FFN_OUT, 4*n_embd, n_embd, bid, i);
    }

    add(LLM_TENSOR_GR00T_STATE_1, "W", {hidden, state, embodiments});
    add(LLM_TENSOR_GR00T_STATE_1, "b", {hidden, embodiments});
    add(LLM_TENSOR_GR00T_STATE_2, "W", {ae, hidden, embodiments});
    add(LLM_TENSOR_GR00T_STATE_2, "b", {ae, embodiments});
    add(LLM_TENSOR_GR00T_ACTION_1, "W", {ae, action, embodiments});
    add(LLM_TENSOR_GR00T_ACTION_1, "b", {ae, embodiments});
    add(LLM_TENSOR_GR00T_ACTION_2, "W", {ae, 2*ae, embodiments});
    add(LLM_TENSOR_GR00T_ACTION_2, "b", {ae, embodiments});
    add(LLM_TENSOR_GR00T_ACTION_3, "W", {ae, ae, embodiments});
    add(LLM_TENSOR_GR00T_ACTION_3, "b", {ae, embodiments});
    add(LLM_TENSOR_GR00T_POSITION, "weight", {ae, hidden});
    linear(LLM_TENSOR_GR00T_TIME_1, 256, ae);
    linear(LLM_TENSOR_GR00T_TIME_2, ae, ae);

    for (uint32_t i = 0; i < hparams.gr00t_dit_blocks; ++i) {
        const int bid = i % n_layer;
        linear(LLM_TENSOR_GR00T_DIT_Q, ae, ae, bid, i);
        linear(LLM_TENSOR_GR00T_DIT_K, i % 2 == 0 ? n_embd : ae, ae, bid, i);
        linear(LLM_TENSOR_GR00T_DIT_V, i % 2 == 0 ? n_embd : ae, ae, bid, i);
        linear(LLM_TENSOR_GR00T_DIT_OUT, ae, ae, bid, i);
        linear(LLM_TENSOR_GR00T_DIT_NORM, ae, 2*ae, bid, i);
        linear(LLM_TENSOR_GR00T_DIT_FFN_IN, ae, 4*ae, bid, i);
        linear(LLM_TENSOR_GR00T_DIT_FFN_OUT, 4*ae, ae, bid, i);
    }
    linear(LLM_TENSOR_GR00T_PROJ_OUT_1, ae, 2*ae);
    linear(LLM_TENSOR_GR00T_PROJ_OUT_2, ae, hidden);
    add(LLM_TENSOR_GR00T_DECODER_1, "W", {hidden, hidden, embodiments});
    add(LLM_TENSOR_GR00T_DECODER_1, "b", {hidden, embodiments});
    add(LLM_TENSOR_GR00T_DECODER_2, "W", {action, hidden, embodiments});
    add(LLM_TENSOR_GR00T_DECODER_2, "b", {action, embodiments});
}

std::unique_ptr<llm_graph_context> llama_model_gr00t_n1d7::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER) {
        return std::make_unique<action_graph>(*this, params);
    }
    return std::make_unique<llama_model_qwen3vl::graph>(*this, params);
}

llama_model_gr00t_n1d7::action_graph::action_graph(
        const llama_model_gr00t_n1d7 & model,
        const llm_graph_params & params) : llm_graph_context(params) {
    GGML_ASSERT(model.action_tensors.size() == 537);
    GGML_ASSERT(cross != nullptr);

    constexpr int64_t vl_head_dim = 64;
    constexpr int64_t vl_heads    = 32;
    GGML_ASSERT(hparams.n_embd == vl_head_dim * vl_heads);

    auto linear = [&](ggml_tensor * input, size_t wi, size_t bi, const char * name, int il) {
        ggml_tensor * output = ggml_mul_mat(ctx0, model.action_tensors.at(wi), input);
        ggml_tensor * bias = ggml_cast(ctx0, model.action_tensors.at(bi), output->type);
        output = ggml_add(ctx0, output, bias);
        cb(output, name, il);
        return output;
    };
    auto layer_norm = [&](ggml_tensor * input, size_t wi, size_t bi, const char * name, int il) {
        ggml_tensor * output = ggml_norm(ctx0, input, 1e-5f);
        ggml_tensor * weight = ggml_cast(ctx0, model.action_tensors.at(wi), output->type);
        ggml_tensor * bias = ggml_cast(ctx0, model.action_tensors.at(bi), output->type);
        output = ggml_mul(ctx0, output, weight);
        output = ggml_add(ctx0, output, bias);
        cb(output, name, il);
        return output;
    };

    // Official process_backbone_output(): first LayerNorm, then four
    // non-causal SelfAttentionTransformer blocks.  Tensor layout here is
    // [channel, token], the transpose of PyTorch's [batch, token, channel].
    ggml_tensor * vl = build_inp_cross_embd();
    vl = layer_norm(vl, 0, 1, "gr00t_vlln", -1);

    for (uint32_t il = 0; il < hparams.gr00t_vl_blocks; ++il) {
        const size_t base = 2 + (size_t) il * 16;
        ggml_tensor * residual = vl;
        ggml_tensor * norm = layer_norm(vl, base + 8, base + 9,
                "gr00t_vl_attn_norm", (int) il);

        ggml_tensor * q = linear(norm, base + 0, base + 1, "gr00t_vl_q", (int) il);
        ggml_tensor * k = linear(norm, base + 2, base + 3, "gr00t_vl_k", (int) il);
        ggml_tensor * v = linear(norm, base + 4, base + 5, "gr00t_vl_v", (int) il);
        q = ggml_reshape_3d(ctx0, q, vl_head_dim, vl_heads, q->ne[1]);
        k = ggml_reshape_3d(ctx0, k, vl_head_dim, vl_heads, k->ne[1]);
        v = ggml_reshape_3d(ctx0, v, vl_head_dim, vl_heads, v->ne[1]);

        ggml_tensor * attn = build_attn_mha(q, k, v,
                nullptr, nullptr, nullptr, nullptr,
                1.0f / sqrtf((float) vl_head_dim), (int) il);
        attn = linear(attn, base + 6, base + 7, "gr00t_vl_attn_out", (int) il);
        vl = ggml_add(ctx0, residual, attn);
        cb(vl, "gr00t_vl_attn_residual", (int) il);

        residual = vl;
        norm = layer_norm(vl, base + 10, base + 11,
                "gr00t_vl_ffn_norm", (int) il);
        ggml_tensor * ffn = linear(norm, base + 12, base + 13,
                "gr00t_vl_ffn_in", (int) il);
        // diffusers FeedForward activation_fn="gelu-approximate".
        ffn = ggml_gelu(ctx0, ffn);
        cb(ffn, "gr00t_vl_ffn_gelu", (int) il);
        ffn = linear(ffn, base + 14, base + 15,
                "gr00t_vl_ffn_out", (int) il);
        vl = ggml_add(ctx0, residual, ffn);
        cb(vl, "gr00t_vl_block_out", (int) il);
    }

    // Deterministic positions in action_tensors are established by
    // load_arch_tensors().  Keep these constants beside the graph until all
    // stages have typed storage of their own.
    constexpr size_t state_1_w  = 66;
    constexpr size_t state_1_b  = 67;
    constexpr size_t state_2_w  = 68;
    constexpr size_t state_2_b  = 69;
    constexpr size_t action_1_w = 70;
    constexpr size_t action_1_b = 71;
    constexpr size_t action_2_w = 72;
    constexpr size_t action_2_b = 73;
    constexpr size_t action_3_w = 74;
    constexpr size_t action_3_b = 75;
    constexpr size_t position_w = 76;

    const int embodiment = cross->gr00t_embodiment_id;
    GGML_ASSERT(embodiment >= 0 && embodiment < (int) hparams.gr00t_embodiment_count);

    auto category_weight = [&](size_t index) {
        ggml_tensor * weight = model.action_tensors.at(index);
        ggml_tensor * selected = ggml_view_2d(ctx0, weight,
                weight->ne[0], weight->ne[1], weight->nb[1],
                (size_t) embodiment * weight->nb[2]);
        // CategorySpecificLinear stores [category, input, output] in PyTorch;
        // after GGUF conversion a selected view is [output, input] in ne[].
        // GGML linear weights require [input, output].
        return ggml_cont(ctx0, ggml_transpose(ctx0, selected));
    };
    auto category_bias = [&](size_t index) {
        ggml_tensor * bias = model.action_tensors.at(index);
        return ggml_view_1d(ctx0, bias, bias->ne[0], (size_t) embodiment * bias->nb[1]);
    };
    auto category_linear = [&](ggml_tensor * input, size_t wi, size_t bi, const char * name) {
        ggml_tensor * output = ggml_mul_mat(ctx0, category_weight(wi), input);
        ggml_tensor * bias = ggml_cast(ctx0, category_bias(bi), output->type);
        output = ggml_add(ctx0, output, bias);
        cb(output, name, -1);
        return output;
    };

    // CategorySpecificMLP: state [132, 1] -> ReLU [1024, 1] -> [1536, 1].
    ggml_tensor * state = build_inp_state();
    state = category_linear(state, state_1_w, state_1_b, "gr00t_state_linear_1");
    state = ggml_relu(ctx0, state);
    cb(state, "gr00t_state_relu", -1);
    state = category_linear(state, state_2_w, state_2_b, "gr00t_state_features");

    // MultiEmbodimentActionEncoder.  The timestep input is the official
    // discrete 1536-wide sinusoidal encoding cached by llama_context.
    ggml_tensor * action = build_inp_action();
    action = category_linear(action, action_1_w, action_1_b, "gr00t_action_linear_1");
    ggml_tensor * timestep = build_inp_sinusoidal_embedding();
    action = ggml_concat(ctx0, action, timestep, 0);
    cb(action, "gr00t_action_time_concat", -1);
    action = category_linear(action, action_2_w, action_2_b, "gr00t_action_linear_2");
    action = ggml_silu(ctx0, action);
    cb(action, "gr00t_action_swish", -1);
    action = category_linear(action, action_3_w, action_3_b, "gr00t_action_features_no_pos");

    ggml_tensor * position = model.action_tensors.at(position_w);
    position = ggml_view_2d(ctx0, position, position->ne[0], hparams.action_steps,
            position->nb[1], 0);
    position = ggml_cast(ctx0, position, action->type);
    action = ggml_add(ctx0, action, position);
    cb(action, "gr00t_action_features", -1);

    ggml_tensor * state_action = ggml_concat(ctx0, state, action, 1);
    cb(state_action, "gr00t_state_action", -1);

    // diffusers TimestepEncoder:
    // Timesteps(256) -> Linear(256, 1536) -> SiLU -> Linear(1536, 1536).
    constexpr size_t time_1_w = 77;
    constexpr size_t time_1_b = 78;
    constexpr size_t time_2_w = 79;
    constexpr size_t time_2_b = 80;
    ggml_tensor * dit_time = build_inp_gr00t_dit_time();
    ggml_tensor * temb = linear(dit_time, time_1_w, time_1_b,
            "gr00t_dit_time_linear_1", -1);
    temb = ggml_silu(ctx0, temb);
    cb(temb, "gr00t_dit_time_silu", -1);
    temb = linear(temb, time_2_w, time_2_b,
            "gr00t_dit_time_embedding", -1);

    const int64_t ae = hparams.n_embd_ae;
    constexpr int64_t dit_head_dim = 48;
    constexpr int64_t dit_heads    = 32;
    GGML_ASSERT(ae == dit_head_dim * dit_heads);
    ggml_tensor * dit_hidden = state_action;
    ggml_tensor * non_image_mask = build_inp_gr00t_vl_mask(false);
    ggml_tensor * image_mask     = build_inp_gr00t_vl_mask(true);

    // The converted checkpoint contains 32 BasicTransformerBlocks.  Even
    // blocks cross-attend to VL features; odd blocks self-attend over the 41
    // state/action tokens.  AlternateVLDiT image/text masks are connected in a
    // later input step; all other inference-time block operations are here.
    for (uint32_t il = 0; il < hparams.gr00t_dit_blocks; ++il) {
        const size_t base = 81 + (size_t) il * 14;

        ggml_tensor * modulation = ggml_silu(ctx0, temb);
        modulation = linear(modulation, base + 8, base + 9,
                "gr00t_dit_adaln_modulation", (int) il);
        ggml_tensor * scale = ggml_view_1d(ctx0, modulation, ae, 0);
        ggml_tensor * shift = ggml_view_1d(ctx0, modulation, ae,
                (size_t) ae * modulation->nb[0]);
        ggml_tensor * dit_norm = ggml_norm(ctx0, dit_hidden, 1e-5f);
        dit_norm = ggml_mul(ctx0, dit_norm,
                ggml_scale_bias(ctx0, scale, 1.0f, 1.0f));
        dit_norm = ggml_add(ctx0, dit_norm, shift);
        cb(dit_norm, "gr00t_dit_adaln", (int) il);

        ggml_tensor * kv_input = il % 2 == 0 ? vl : dit_norm;
        ggml_tensor * dit_q = linear(dit_norm, base + 0, base + 1,
                "gr00t_dit_q", (int) il);
        ggml_tensor * dit_k = linear(kv_input, base + 2, base + 3,
                "gr00t_dit_k", (int) il);
        ggml_tensor * dit_v = linear(kv_input, base + 4, base + 5,
                "gr00t_dit_v", (int) il);
        dit_q = ggml_reshape_3d(ctx0, dit_q, dit_head_dim, dit_heads, dit_q->ne[1]);
        dit_k = ggml_reshape_3d(ctx0, dit_k, dit_head_dim, dit_heads, dit_k->ne[1]);
        dit_v = ggml_reshape_3d(ctx0, dit_v, dit_head_dim, dit_heads, dit_v->ne[1]);

        // attend_text_every_n_blocks=2: cross blocks 0,4,8,... attend
        // non-image tokens; cross blocks 2,6,10,... attend image tokens.
        ggml_tensor * kq_mask = nullptr;
        if (il % 2 == 0) {
            kq_mask = il % 4 == 0 ? non_image_mask : image_mask;
        }
        ggml_tensor * dit_attn = build_attn_mha(dit_q, dit_k, dit_v,
                nullptr, kq_mask, nullptr, nullptr,
                1.0f / sqrtf((float) dit_head_dim), (int) il);
        dit_attn = linear(dit_attn, base + 6, base + 7,
                "gr00t_dit_attn_out", (int) il);
        dit_hidden = ggml_add(ctx0, dit_hidden, dit_attn);
        cb(dit_hidden, "gr00t_dit_attn_residual", (int) il);

        // norm3 has no affine parameters in this model.  Inference dropout is
        // disabled.  The checkpoint confirms FFN 1536 -> 6144 -> 1536.
        ggml_tensor * dit_ffn = ggml_norm(ctx0, dit_hidden, 1e-5f);
        cb(dit_ffn, "gr00t_dit_ffn_norm", (int) il);
        dit_ffn = linear(dit_ffn, base + 10, base + 11,
                "gr00t_dit_ffn_in", (int) il);
        dit_ffn = ggml_gelu(ctx0, dit_ffn);
        cb(dit_ffn, "gr00t_dit_ffn_gelu", (int) il);
        dit_ffn = linear(dit_ffn, base + 12, base + 13,
                "gr00t_dit_ffn_out", (int) il);
        dit_hidden = ggml_add(ctx0, dit_hidden, dit_ffn);
        cb(dit_hidden, "gr00t_dit_block_out", (int) il);
    }

    // DiT output conditioning.  Unlike AdaLayerNorm, the official forward
    // names the first proj_out_1 half "shift" and the second half "scale".
    constexpr size_t proj_out_1_w = 529;
    constexpr size_t proj_out_1_b = 530;
    constexpr size_t proj_out_2_w = 531;
    constexpr size_t proj_out_2_b = 532;
    ggml_tensor * output_mod = ggml_silu(ctx0, temb);
    output_mod = linear(output_mod, proj_out_1_w, proj_out_1_b,
            "gr00t_dit_output_modulation", -1);
    ggml_tensor * output_shift = ggml_view_1d(ctx0, output_mod, ae, 0);
    ggml_tensor * output_scale = ggml_view_1d(ctx0, output_mod, ae,
            (size_t) ae * output_mod->nb[0]);
    ggml_tensor * model_output = ggml_norm(ctx0, dit_hidden, 1e-6f);
    model_output = ggml_mul(ctx0, model_output,
            ggml_scale_bias(ctx0, output_scale, 1.0f, 1.0f));
    model_output = ggml_add(ctx0, model_output, output_shift);
    model_output = linear(model_output, proj_out_2_w, proj_out_2_b,
            "gr00t_dit_output", -1);

    // CategorySpecificMLP action decoder: 1024 -> ReLU -> 1024 -> 132.
    // The first token is the state token, so expose only the final 40 tokens.
    constexpr size_t decoder_1_w = 533;
    constexpr size_t decoder_1_b = 534;
    constexpr size_t decoder_2_w = 535;
    constexpr size_t decoder_2_b = 536;
    ggml_tensor * decoded = category_linear(model_output,
            decoder_1_w, decoder_1_b, "gr00t_action_decoder_1");
    decoded = ggml_relu(ctx0, decoded);
    cb(decoded, "gr00t_action_decoder_relu", -1);
    decoded = category_linear(decoded,
            decoder_2_w, decoder_2_b, "gr00t_action_velocity_all_tokens");
    decoded = ggml_view_2d(ctx0, decoded,
            hparams.action_dim, hparams.action_steps,
            decoded->nb[1], decoded->nb[1]);
    cb(decoded, "gr00t_action_velocity", -1);

    res->action = decoded;
    ggml_build_forward_expand(gf, res->action);
}
