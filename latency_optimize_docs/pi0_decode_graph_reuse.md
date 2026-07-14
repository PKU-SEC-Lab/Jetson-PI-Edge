# PI0 Decode 优化（第一步）：复用计算图

本文档记录已在 `Jetson-PI-2` 落地、且与 `Jetson-PI-2-new` 数值基线对齐的 **decode 计算图复用** 方案。

- 第二步（KV GPU）：[pi0_decode_kv_gpu.md](./pi0_decode_kv_gpu.md)  
- 第三步（mask 复用）：[pi0_decode_mask_reuse.md](./pi0_decode_mask_reuse.md)  
- 完整路线图：`pi0_decode_optimization.md`  

**不要**把多步优化混在同一补丁里验证。

---

## 1. 目标与范围

| 项 | 本步做 | 本步不做 |
|----|--------|----------|
| 计算图 | 4 步 diffusion 内第 2–4 步复用第 1 步 decode 图拓扑 | — |
| 输入数值 | 每步仍 `set_input` 全量更新 | 不对 `time`/`action` 等做早退 |
| KV | 每步仍从 CPU `encoded_kv_data` `tensor_set` | GPU KV、跳过 KV 写入 |
| Mask | 每步仍填充 host mask | 持久 mask、`pi0_*_ready` 早退 |
| encode | 不变（仍 `tensor_get` → CPU） | `pi0_refresh_encoded_kv_gpu` |

**收益**：diffusion 第 2–4 步跳过 `build_graph`，`graph_build` 时间显著下降（`LLAMA_PI0_PERF=1` 时 `reused=1`）。

---

## 2. 核心概念：`can_reuse` ≠ 跳过输入更新

这是本步最容易误解的一点。

```
can_reuse == true  →  图拓扑（算子、tensor 形状）与上一步相同，可跳过 build_graph
set_input()        →  每步仍会执行，把当前 step 的数值写入 graph input buffer
```

因此 **`time` / `sinusoidal` / `action` / `state` 每步数值都会变，仍然可以 `can_reuse=true`**：

- `can_reuse` 只检查形状、`action_steps`、`n_embd_ae` 等是否一致；
- `set_input` 里用最新的 `cross->time_step`、`cross->action`、`cross->state` 做 `ggml_backend_tensor_set`。

`llm_graph_result::can_reuse()` 要求 **所有** input 的 `can_reuse` 均为 true 才复用整图；任一 false 则本步重建图（与基线每步重建等价）。

---

## 3. 数据流（单次 encode → decode）

```
encode()
  └─ cross.n_token = t_encoded_kv[0]->ne[2]
  └─ ggml_backend_tensor_get → cross.encoded_kv_data[i]   # 仍在 CPU，本步未改

decode()  for i in 0..inference_steps-1
  └─ cross.time_step = 1 - i/steps
  └─ process_ubatch(..., LLM_GRAPH_TYPE_DECODER)
        ├─ can_reuse?  →  i==0: 否（新建图）; i>=1: 是（复用）
        ├─ set_inputs() →  time / sinusoidal / action / state / KV / mask 全写
        └─ graph_compute()
  └─ 更新 cross.action（diffusion）
```

---

## 4. 修改文件清单

| 文件 | 改动类型 |
|------|----------|
| `src/llama-graph.h` | `allow_reuse` 修复；若干 input 增加 `can_reuse`；`attn_no_cache_ae` 增加 `kv_token_num` |
| `src/llama-graph.cpp` | 实现上述 `can_reuse`；`build_attn_inp_no_cache_ae` 传入 `kv_token_num` |
| `src/llama-context.cpp` | `process_ubatch` 复用路径注释（**不**在复用时二次 `alloc_graph`） |

---

## 5. 具体修改说明

### 5.1 `llm_graph_params::allow_reuse`（`llama-graph.h`）

PI0 的 batch 可能 **同时** 带 `token` 与 `embd`。旧逻辑：

```cpp
(!ubatch.token && !other.ubatch.token) || (!ubatch.embd && !other.ubatch.embd)
```

在两边都有 token+embd 时恒为 false，decode 图 **永远无法** 进入复用分支。

**修复**：模态必须完全一致：

```cpp
u_has_token == o_has_token && u_has_embd == o_has_embd
```

### 5.2 各 graph input 的 `can_reuse`（`llama-graph.cpp`）

| 类 | `can_reuse` 条件要点 | 每步 `set_input` |
|----|----------------------|------------------|
| `llm_graph_input_time` | `nelements` 与 `action_steps`、`n_embd_ae` 一致 | 用 `cross->time_step` 填充 |
| `llm_graph_input_sinusoidal_embedding` | 同上 | 按 `time_step` 重算 sin/cos 再写入 |
| `llm_graph_input_action` | `action_steps * action_dim` 与 tensor 一致 | `cross->action` |
| `llm_graph_input_state` | `action_dim` 一致 | `cross->state` |
| `llm_graph_input_cross_kv_pi0` | 各层 `kv->ne[2] == cross->n_token` | `encoded_kv_data` → `tensor_set` |
| `llm_graph_input_attn_no_cache_pi0` | mask `ne[0/1]` 与 `ubatch.n_tokens`、pad 一致 | 填 self-attn mask |
| `llm_graph_input_attn_no_cache_ae` | `ne[0]==kv_token_num`，`ne[1]==pad(action_steps+1)` | 填 AE mask |

`llm_graph_input_pos_ae` 等本就 `can_reuse=true` 的 input 保持不变。

**`llm_graph_input_cross_kv_pi0::set_input`**：仅增加 `kv_array[i]` 空指针跳过；**无** `pi0_cross_kv_inputs_ready` 早退。

### 5.3 `attn_no_cache_ae` 与 `kv_token_num`（`llama-graph.h` / `.cpp`）

`build_attn_inp_no_cache_ae(token_num, kv_token_num)` 建图时 mask 的 KV 维是固定的 `kv_token_num`（PI0 AE 里为 `encoder_tokens + action_tokens`）。`can_reuse` 需要记住该维：

```cpp
llm_graph_input_attn_no_cache_ae(hparams, cparams, kv_token_num);
```

### 5.4 `process_ubatch` 复用路径（`llama-context.cpp`）

**正确做法**（与 `Jetson-PI-2-new` 一致，且已验证数值）：

```cpp
if (res->can_reuse(gparams)) {
    n_reused++;
    ub_timing.graph_reused = true;
    // 不调用 build_graph / sched_alloc_graph
} else {
    res->reset();
    sched_reset + build_graph + alloc_graph;
}
res->set_inputs(&ubatch);   // 复用与新建都会执行
graph_compute(...);
```

**错误做法（已踩坑）**：在 `reused=1` 时再 `sched_reset` + `alloc_graph`，意图“重绑 GPU buffer”。会导致 input 写入地址与本次 `graph_compute` 使用的 buffer 不一致，**第 2 步起 diffusion 数值错误**。性能优化文档里曾建议该路径，**本步验证后不应使用**。

`graph_compute` 仍使用：

```cpp
graph_compute(res->get_gf(), ubatch.n_tokens > 1);
```

与当前正确基线一致；勿在未单独验证的情况下改为按 `action_steps+1` 选 batched 参数。

---

## 6. 验证方法

1. 编译并重启 `llama-server`（必须加载新二进制）。
2. `export LLAMA_PI0_PERF=1`，运行 `run_foreground_pipeline.sh`。
3. 日志期望：
   - `[PI0 step 1/4] ... reused=0`
   - `[PI0 step 2/4]` ~ `[PI0 step 4/4] ... reused=1`
   - 第 2–4 步 `graph_build` 应接近 0 ms（无 `build_graph`）
4. 第 4 步 `action_final` 与未开图复用的 `Jetson-PI-2-new` 一致（或仅浮点误差）。

环境变量：

| 变量 | 作用 |
|------|------|
| `LLAMA_PI0_PERF=1` | 打印每步 `graph_build` / `set_inputs` / `reused` |
| `LLAMA_GRAPH_REUSE_DISABLE=1` | 关闭图复用，用于 A/B 对比 |

---

## 7. 注意点与常见坑

### 7.1 可变输入与图复用

- **可以复用图**：`time`、`sinusoidal`、`action`、`state` 每步值不同，但形状不变。
- **不能省略 `set_input`**：除非在后续独立步骤中引入“mask/KV 早退”，且保证 `alloc` 后 buffer 仍有效（本步未做）。

### 7.2 复用时不要二次 `alloc_graph`

复用 = 保留同一 `ggml_cgraph` 与 input 张量句柄；只做 `set_inputs` + `graph_compute`。二次 alloc 是此前数值错误的主因。

### 7.3 `cross.n_token` 与 cross-KV 形状

- encode 后必须设置：`cross.n_token = t_encoded_kv[0]->ne[2]`。
- `cross_kv_pi0::can_reuse` 用 `cross->n_token` 与各层 `kv->ne[2]` 比较。
- 若建图第三维误用 decode 的 `action_steps` 而非 encoder 长度，会导致 `can_reuse` 失败或拷贝 shape 断言（属 KV 优化阶段问题，本步未改 `build_inp_cross_kv_pi0` 的 `n_tokens` 逻辑，与当前正确基线一致）。

### 7.4 分步落地

建议顺序：

1. **本步**：仅图复用（本文档）→ 确认 `action_final` 正确。  
2. mask 复用（持久 buffer + 第 1 步写入、2–4 步早退）。  
3. KV GPU 缓存（encode 后不落 CPU / 少拷贝）。  

每步单独合并、单独对比 `Jetson-PI-2-new`，避免多个优化叠加后难以定位回归。

### 7.5 与 encoder 图的关系

单次请求内：encode 使用 `LLM_GRAPH_TYPE` encoder 图；第 1 次 decode 因 `gtype` 不同会 **重建** decode 图。第 2–4 次 decode 才复用 decode 图。encoder 图复用不受本步 decode `can_reuse` 影响。

---

## 8. 后续步骤

| 步骤 | 文档 | 状态 |
|------|------|------|
| KV GPU + decode 复用 KV | [pi0_decode_kv_gpu.md](./pi0_decode_kv_gpu.md) | 已落地 |
| mask 早退（decode AE） | [pi0_decode_mask_reuse.md](./pi0_decode_mask_reuse.md) | 已落地 |
| 跳过 memory_init_batch | [pi0_decode_memory_init.md](./pi0_decode_memory_init.md) | 已落地 |
| 复用路径 `sched_reset` + `alloc_graph` | — | 不推荐（数值错误） |

---

## 9. 参考

- 数值基线：`Jetson-PI-2-new`（每步重建图 + CPU KV）
- 完整优化路线图：`docs/pi0_decode_optimization.md`
- 旧实现清单：`Jetson-PI-2-old/docs/pi0_decode_opt_implementation_guide.md`
