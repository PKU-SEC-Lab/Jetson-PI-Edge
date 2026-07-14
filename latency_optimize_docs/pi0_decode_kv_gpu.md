# PI0 Decode 优化（第二步）：Encode KV 留 GPU + Decode 复用 KV

本文档记录已在 `Jetson-PI-2` 落地、且与数值基线对齐的 **Cross-KV GPU 缓存** 方案。  
**前置依赖**：第一步「计算图复用」见 [pi0_decode_graph_reuse.md](./pi0_decode_graph_reuse.md)。本步与之叠加使用，须分步验证。

---

## 1. 目标与范围

| 项 | 本步做 | 本步不做 |
|----|--------|----------|
| encode 后 KV | `pi0_refresh_encoded_kv_gpu()`：GPU→GPU 写入持久 buffer，**不再** `tensor_get` 到 CPU | — |
| decode 第 1 步 | 绑定/拷贝 KV 到 graph input，置 `pi0_cross_kv_inputs_ready` | — |
| decode 第 2–4 步 | `set_input` 对 cross-KV **早退**（内容未变） | — |
| 建图 | `build_inp_cross_kv_pi0` 第三维用 `cross.n_token`；可直绑 `encoded_kv_gpu` | — |
| 计算图 | 依赖第一步 `can_reuse` + `reused=1` | 单独重建图而不复用 |
| mask | 每步仍填充 AE mask | mask 早退（已单独试验并回滚） |
| time/action 等 | 每步仍 `tensor_set` | 对 diffusion 变量早退 |

**预期收益**（`LLAMA_PI0_PERF=1`）：

- encode：`kv_extract` 去掉 D2H，略降延迟  
- decode：第 2–4 步 `set_inputs` 明显下降（原先约 30 ms 中 KV 占大头）  
- `decode_ms`（HTTP 汇总）可能有数 ms～十余 ms 量级改善，仍受 `graph_compute` 主导

---

## 2. 问题背景：基线路径为何慢

基线（`Jetson-PI-2-new`）每步 diffusion 都执行：

```cpp
ggml_backend_tensor_get(t_encoded_kv[i], cross.encoded_kv_data[i].data(), ...);  // encode：D2H
// ...
ggml_backend_tensor_set(kv_array[i], cross.encoded_kv_data[i].data(), ...);       // decode 每步：H2D
```

4 步 decode × `n_layer` 层 KV ≈ **重复 4 次大体积 H2D**。  
在 perf 日志里体现为：**`set_inputs` 占 decode ~44%**，而 `graph_build` 仅 ~1%（图复用后）。

因此本步优化的是 **KV 数据路径**，不是计算图拓扑。

---

## 3. 核心思路

### 3.1 单次请求内的不变性

对固定一次 encode→decode：

- encoder 输出的 cross-attention KV **在 4 步 diffusion 内不变**（只随 `time_step` / `action` 变，KV 来自 encoder 前向）  
- 图复用（第一步）保证第 2–4 步 **不重建** decode 图  

于是可以：

1. encode 结束：KV 留在 GPU 持久区 `pi0_enc_kv_gpu`  
2. 第 1 步 decode：写入 graph input（或 alias 到同一 GPU tensor）  
3. 第 2–4 步：`pi0_cross_kv_inputs_ready` 早退，跳过 KV `set_input`

### 3.2 `pi0_cross_kv_inputs_ready` ≠ `can_reuse`

| 机制 | 含义 |
|------|------|
| `can_reuse`（graph input 类） | 图**拓扑**不变，允许跳过 `build_graph` |
| `pi0_cross_kv_inputs_ready` | 本轮 diffusion 内 KV **已写入 graph**，允许跳过 KV 的 `set_input` |

二者独立：图复用解决 `graph_build`；KV flag 解决 `set_inputs` 里的 KV 拷贝。

### 3.3 与 mask 早退的区别（本步未采用 mask 方案）

mask 早退曾试验：第 2–4 步跳过 mask 填充，收益仅 ~1 ms/步（mask 本身很轻）。  
KV 早退收益大得多（36 层 `tensor_set`），故本步只做 KV。

---

## 4. 数据流

```
encode() 开始
  └─ pi0_clear_cross_kv()                    # 清上一轮请求状态

encode() 结束（synchronize 之后）
  └─ cross.n_token = t_encoded_kv[0]->ne[2]
  └─ pi0_refresh_encoded_kv_gpu(t_encoded_kv) # GPU→GPU，无 D2H
        ├─ cross.encoded_kv_gpu[i] = pi0_enc_kv_gpu.tensors[i]
        ├─ cross.pi0_use_gpu_kv = true
        └─ cross.pi0_cross_kv_inputs_ready = false

decode()  for step i in 0..3
  └─ process_ubatch(DECODER)
        ├─ 图复用?  step0: 新建图; step1-3: reused=1
        ├─ 新建图时: pi0_cross_kv_inputs_ready = false
        └─ set_inputs()
              ├─ cross_kv: ready? 早退 : 写 KV → ready=true
              ├─ time / action / sinusoidal / state: 每步更新
              └─ mask: 每步填充

decode() 结束
  └─ pi0_clear_cross_kv()
```

---

## 5. 修改文件清单

| 文件 | 改动 |
|------|------|
| `src/llama-graph.h` | `llama_cross` 增加 GPU KV 字段；`llama_backend_tensor_copy_compat` 声明 |
| `src/llama-graph.cpp` | 兼容拷贝；`cross_kv_pi0::set_input` / `build_inp_cross_kv_pi0` |
| `src/llama-context.h` | `pi0_encoded_kv_gpu_storage`；`pi0_clear` / `pi0_refresh` 声明 |
| `src/llama-context.cpp` | 实现与 encode/decode/process_ubatch 挂钩 |

---

## 6. 具体修改说明

### 6.1 `llama_backend_tensor_copy_compat`（`llama-graph.cpp`）

GPU tensor 之间拷贝时，stride 可能不一致，不能直接 `ggml_backend_tensor_copy`：

```cpp
// shape 一致且 stride 一致 → ggml_backend_tensor_copy
// 否则 → tensor_get + tensor_set（经 thread_local buffer）
```

用于：

- encode 后：`t_encoded_kv[i]` → `pi0_enc_kv_gpu.tensors[i]`  
- decode 第 1 步：`encoded_kv_gpu[i]` → `kv_array[i]`（未 alias 时）

### 6.2 `struct llama_cross` 新字段

```cpp
std::vector<ggml_tensor *> encoded_kv_gpu;
bool                         pi0_use_gpu_kv            = false;
mutable bool                 pi0_cross_kv_inputs_ready = false;
```

- `encoded_kv_gpu`：指向本轮 encode 缓存在 GPU 上的 KV  
- `pi0_use_gpu_kv`：`pi0_refresh` 成功后为 true  
- `pi0_cross_kv_inputs_ready`：当前 diffusion 轮内是否已写入 decode graph input  

`encoded_kv_data` **保留**作 CPU 回退（正常路径不再填充）。

### 6.3 `pi0_refresh_encoded_kv_gpu`（`llama-context.cpp`）

1. 选择非 CPU backend（与模型一致，通常 CUDA）  
2. 按 `n_layer` 与 `src[0]->ne[2]` 决定是否重分配 `pi0_enc_kv_gpu`  
3. 每层 `llama_backend_tensor_copy_compat(src[i], pi0_enc_kv_gpu.tensors[i])`  
4. `cross.encoded_kv_gpu[i]` 指向持久 tensor  
5. **`pi0_cross_kv_inputs_ready = false`**（新一轮 diffusion 必须重写 graph input）

### 6.4 `pi0_clear_cross_kv`（请求边界）

| 调用位置 | 原因 |
|----------|------|
| `encode()` 开头 | 避免上一轮 KV 残留 |
| `decode()` PI0 分支结束 | 避免影响下一次请求 |

释放 `pi0_enc_kv_gpu` buffer，清空 `encoded_kv_gpu` / flag / `encoded_kv_data`。

### 6.5 `llm_graph_input_cross_kv_pi0::set_input`

```cpp
if (cross->pi0_cross_kv_inputs_ready) {
    return;  // 第 2–4 步早退
}
for (每层 i) {
    if (kv_array[i] == encoded_kv_gpu[i]) continue;           // 已 alias，无需拷贝
    if (pi0_use_gpu_kv) llama_backend_tensor_copy_compat(...); // GPU→GPU
    else if (!encoded_kv_data[i].empty()) tensor_set(...);     // CPU 回退
}
cross->pi0_cross_kv_inputs_ready = true;
```

### 6.6 `build_inp_cross_kv_pi0`（形状与直绑）

**正确性关键**：第三维必须是 encoder 序列长度，不能用 decode ubatch 的 `n_tokens`（action 步数）：

```cpp
const int64_t n_kv_tokens = (cross && cross->n_token > 0) ? cross->n_token : n_tokens;
```

直绑（避免第 1 步多余拷贝）：

```cpp
if (cross->pi0_use_gpu_kv && encoded_kv_gpu[i]->ne[2] == n_kv_tokens)
    cur_layer_kv = cross->encoded_kv_gpu[i];
else
    cur_layer_kv = ggml_new_tensor_3d(..., n_kv_tokens);
```

`cross_kv_pi0::can_reuse` 检查各层 `kv->ne[2] == cross->n_token`（与图复用配合）。

### 6.7 `process_ubatch`：何时重置 `pi0_cross_kv_inputs_ready`

| 路径 | 行为 |
|------|------|
| `can_reuse` 成功（`reused=1`） | **不**重置 ready；第 2–4 步 KV 早退 |
| 新建图（PI0 + `LLM_GRAPH_TYPE_DECODER`） | `pi0_cross_kv_inputs_ready = false`，第 1 步会写 KV |

**禁止**在 `reused=1` 时再 `sched_reset` + `alloc_graph`（曾导致数值错误：写入 buffer 与 compute 使用 buffer 不一致）。

### 6.8 encode 路径替换

**删除**：

```cpp
for (i) {
    encoded_kv_data[i].resize(...);
    ggml_backend_tensor_get(t_encoded_kv[i], encoded_kv_data[i].data(), ...);
}
```

**改为**：

```cpp
cross.n_token = t_encoded_kv[0]->ne[2];
pi0_refresh_encoded_kv_gpu(t_encoded_kv, hparams.n_layer);
```

---

## 7. 与第一步（图复用）的配合

| diffusion 步 | 图 | KV `set_input` |
|--------------|-----|----------------|
| 1 | 新建（`reused=0`），`ready=false` | 写入/绑定 |
| 2 | 复用（`reused=1`） | 早退 |
| 3 | 复用 | 早退 |
| 4 | 复用 | 早退 |

若关闭图复用（`LLAMA_GRAPH_REUSE_DISABLE=1`）但保留本步：每步会重建图并重置 `ready`，KV 每步重传，**本步收益丧失**。

---

## 8. 验证方法

1. 编译并**重启** `llama-server`  
2. `export LLAMA_PI0_PERF=1`  
3. `./run_foreground_pipeline.sh`

### 8.1 数值

- 第 4 步 `action_final` 与 `Jetson-PI-2-new` 或「仅图复用」一致  

### 8.2 性能（stderr）

期望类似：

```
[PI0 step 1/4] ... set_inputs=~8-9 ms  reused=0
[PI0 step 2/4] ... set_inputs=~1-3 ms  reused=1   # 相对基线 ~7 ms 明显下降
...
set_inputs (sum)  由 ~30 ms 降至 ~15-20 ms 量级（视硬件而定）
```

### 8.3 不应出现的现象

- 第 2 步起 `action_final` 与基线分叉 → 检查是否误加 `alloc_graph` on reuse、或 `n_kv_tokens` 用错  
- `set_inputs` 无下降但 `reused=1` → 检查 `pi0_cross_kv_inputs_ready` 是否被错误重置  
- `GGML_ASSERT(src->ne[i] == dst->ne[i])` → cross-KV 第三维与 `cross.n_token` 不一致  

---

## 9. 注意点与踩坑记录

### 9.1 Cross-KV 第三维必须用 `cross.n_token`

`build_inp_cross_kv_pi0` 若误用 decode ubatch 的 `n_tokens`（如 `action_steps+1`），会导致：

- 与 encoder KV shape 不一致 → `llama_backend_tensor_copy_compat` 断言失败  
- 或 silently 错误（若未断言）

### 9.2 复用路径不要二次 `alloc_graph`

在 `reused=1` 时做 `sched_reset` + `alloc_graph` 会重绑 input buffer，若仍 `pi0_cross_kv_inputs_ready` 早退，则 **未向新 buffer 写 KV** → diffusion 第 2 步起数值错误。  
图复用步应：**只** `set_inputs` + `graph_compute`（见 [pi0_decode_graph_reuse.md](./pi0_decode_graph_reuse.md) 7.2 节）。

### 9.3 仅在新图时重置 `pi0_cross_kv_inputs_ready`

- `pi0_refresh` 末尾：`ready = false`（新 diffusion）  
- `process_ubatch` 的 **else**（重建图）分支：`ready = false`  
- **不要**在 `reused=1` 分支重置  

### 9.4 Alias 与拷贝

建图直绑 `encoded_kv_gpu[i]` 时，第 1 步 `set_input` 检测 `kv_array[i] == encoded_kv_gpu[i]` 则 **skip 拷贝**。  
若 sched 分配导致 graph input 为新 tensor（未 alias），则第 1 步走 **一次** GPU→GPU 拷贝，第 2–4 步仍早退（数据已在 graph 侧 buffer）。

### 9.5 不要与 mask 早退混在同一补丁验证

mask 早退收益小；KV 路径独立。分步合并、分步对比 `action_final` 与 perf。

### 9.6 HTTP `decode_ms` 为何仍不明显

- 瓶颈仍是 `graph_compute`（~30 ms）  
- 本步主要压缩 `set_inputs`（~30 ms → 更低）  
- 总 `decode_ms` ~68 ms，省 10–15 ms 约 **15–20%**，可能被测量噪声淹没  

看 stderr 分阶段比只看 JSON 更准确。

### 9.7 Encoder 与 Decoder 的 mask 不要共用 flag

encoder 用 `attn_no_cache_pi0`，decoder AE 用 `attn_no_cache_ae`。  
若共用 `pi0_self_attn_mask_ready` 且 encoder 先置位，会导致 decode mask 未初始化。本步 **未** 做 mask 早退，无此问题。

---

## 10. 环境变量

| 变量 | 作用 |
|------|------|
| `LLAMA_PI0_PERF=1` | 打印 encode/decode 分阶段与每步 `reused` / `set_inputs` |
| `LLAMA_GRAPH_REUSE_DISABLE=1` | 关闭图复用（对照实验：KV 优化收益依赖图复用） |

---

## 11. 参考

- 第一步图复用：[pi0_decode_graph_reuse.md](./pi0_decode_graph_reuse.md)  
- 第三步 mask 复用：[pi0_decode_mask_reuse.md](./pi0_decode_mask_reuse.md)  
- 完整路线图：`pi0_decode_optimization.md`  
- 旧实现清单：`Jetson-PI-2-old/docs/pi0_decode_opt_implementation_guide.md`  
- 数值基线：`Jetson-PI-2-new`
