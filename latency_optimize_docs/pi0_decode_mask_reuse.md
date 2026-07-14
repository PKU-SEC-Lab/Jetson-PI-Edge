# PI0 Decode 优化（第三步）：复用 Attention Mask

本文档记录 **decode 侧 AE self-attention mask** 在 4 步 diffusion 内只计算一次的方案。

- 第一步（图复用）：[pi0_decode_graph_reuse.md](./pi0_decode_graph_reuse.md)  
- 第二步（KV GPU）：[pi0_decode_kv_gpu.md](./pi0_decode_kv_gpu.md)  

**不要**与 encoder mask、KV 优化混在同一补丁里做数值对比。

---

## 1. 目标与范围

| 项 | 本步做 | 本步不做 |
|----|--------|----------|
| decode mask | `attn_no_cache_ae` 第 1 步填充，第 2–4 步早退 | encoder 的 `attn_no_cache_pi0` |
| 依赖 | 第一步图复用（`reused=1`） | 无图复用时 mask 每步仍会重填 |
| KV / time / action | 仍按第二步/每步逻辑更新 | — |

**预期收益**：mask 填充本身较轻（`std::fill` + 小循环），单步约 **0.5–1.5 ms**；4 步合计约 **1.5–4 ms**。主要价值是与 KV 早退一起压低 `set_inputs`，而非改变 `graph_compute`。

---

## 2. 问题背景

Decoder 使用 `llm_graph_input_attn_no_cache_ae`。每步 diffusion 原先在 `set_input` 中：

1. `std::fill` 整块 mask 为 `0`  
2. 对部分位置写 `-INFINITY`（屏蔽 future action tokens）

4 步内 **mask 数值与 `time_step`、`action` 无关**（只与 `action_steps`、encoder+action 的 token 布局有关），在 **同一张复用图** 上可只写一次。

---

## 3. 核心思路

### 3.1 独立 flag：`pi0_decode_attn_mask_ready`

| flag | 用途 |
|------|------|
| `pi0_cross_kv_inputs_ready` | cross-KV 已写入 graph |
| `pi0_decode_attn_mask_ready` | **仅** decode AE self-attn mask 已写入 graph |

**禁止**与 encoder 共用同一 flag：encoder 用 `attn_no_cache_pi0`，若在 encode 后置位会导致 decode mask **未初始化** 却早退。

### 3.2 与 `can_reuse` 的关系

- `llm_graph_input_attn_no_cache_ae::can_reuse`：mask tensor 形状与 `kv_token_num`、`action_steps` 一致 → 允许 **图拓扑** 复用  
- `pi0_decode_attn_mask_ready`：图复用后 **跳过 mask 的 host 填充**

---

## 4. 数据流

```
decode()  diffusion 循环前
  └─ pi0_cross_kv_inputs_ready  = false
  └─ pi0_decode_attn_mask_ready = false   # 不清 GPU KV

decode()  step 0（新建 decode 图）
  └─ set_input: attn_no_cache_ae 填 mask → pi0_decode_attn_mask_ready = true
  └─ cross_kv 写入 → pi0_cross_kv_inputs_ready = true

decode()  step 1–3（reused=1）
  └─ attn_no_cache_ae::set_input 早退
  └─ cross_kv::set_input 早退
  └─ time / action / sinusoidal / state 仍每步更新

decode() 结束
  └─ pi0_clear_cross_kv()  # 含 mask flag
```

---

## 5. 修改清单

| 文件 | 改动 |
|------|------|
| `src/llama-graph.h` | `llama_cross::pi0_decode_attn_mask_ready`；`attn_no_cache_ae` 持有 `const llama_cross *` |
| `src/llama-graph.cpp` | `set_input` 早退 + 填完后置位；`build_attn_inp_no_cache_ae` 传入 `cross` |
| `src/llama-context.cpp` | `pi0_clear_cross_kv`、重建图、diffusion 循环前重置 flag |

### 5.1 `set_input` 逻辑（`llama-graph.cpp`）

```cpp
if (cross && cross->pi0_decode_attn_mask_ready) {
    return;
}
// ... 原有 fill ...
if (cross) {
    cross->pi0_decode_attn_mask_ready = true;
}
```

### 5.2 flag 重置时机

| 时机 | 重置 `pi0_decode_attn_mask_ready` |
|------|-----------------------------------|
| `pi0_clear_cross_kv()` | 是 |
| diffusion 循环开始前 | 是（与 KV flag 一起，**不清** GPU KV） |
| `process_ubatch` 重建 decode 图（`can_reuse` 失败） | 是 |
| `reused=1` 路径 | **否** |

重建图时必须重置：新 graph 的 mask buffer 地址已变，若 flag 仍为 true 会跳过填充 → 数值错误。

---

## 6. 验证

1. 编译并 **重启** `llama-server`  
2. `export LLAMA_PI0_PERF=1`  
3. `./run_foreground_pipeline.sh`

### 数值

- 第 4 步 `action_final` 与优化前（图复用 + KV GPU）一致  

### 性能

- 第 1 步 `set_inputs` 与之前相近  
- 第 2–4 步 `set_inputs` 可能再降约 **0.5–1 ms/步**（取决于 mask 大小）  
- `reused=1` 仍应出现  

---

## 7. 注意点

1. **不要**在 `attn_no_cache_pi0::set_input` 使用 `pi0_decode_attn_mask_ready`。  
2. **不要**在 `reused=1` 时重置 mask flag。  
3. mask 收益小于 KV；若 `set_inputs` 几乎不变，属正常。  
4. 本步 **不** 改 mask 计算公式，只减少重复写入。  

---

## 8. 参考

- [pi0_decode_graph_reuse.md](./pi0_decode_graph_reuse.md)  
- [pi0_decode_kv_gpu.md](./pi0_decode_kv_gpu.md)  
- `docs/pi0_decode_optimization.md`（总览）
