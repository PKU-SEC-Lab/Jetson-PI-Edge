# PI0 VIT 延迟优化经验总结

本文档记录 Jetson-PI-2 上对 PI0 **视觉编码（VIT / mmproj）** 阶段的性能分析与工程优化，包括 profiling 方法、两项核心优化（固定分辨率图复用、preprocess 直写 `inp_raw` 布局）、验证方式与踩坑说明。可与 `docs/pi0_decode_optimization.md`（LLM decode）、`docs/pi0_server_changes.md`（Server 集成）配合阅读。

---

## 1. 背景与瓶颈

PI0 端到端推理中，VIT 负责将相机图像编码为 token embedding，再送入 LLM encoder。典型 profiling（`LLAMA_PI0_VIT_PERF=1`）如下：

```
========== PI0 VIT encode latency ==========
  preprocess slice:        ~0.05 ms
  preprocess normalize:    ~0.74 ms
  preprocess total:        ~0.79 ms  (~5%)
  graph_build+alloc:         ~1.69 ms (~12%)
  set_inputs:              ~1.19 ms (~8%)
  graph_compute:            ~10.92 ms (~76%)  ← 主瓶颈
  output_get:               ~0.65 ms (~5%)
  encode total:             ~14.4 ms
```

**结论**：

- **`graph_compute`** 占绝大部分时间，属于 GPU 前向硬算力，工程上只能通过 Flash Attention、GPU 后端、环境调频等加速，**不能**靠跳过计算来「优化」。
- **`graph_build+alloc`**、**`set_inputs`** 在固定相机分辨率、连续推理场景下存在**重复且结果不变**的工作，适合在**不改变 VIT 数值语义**的前提下优化。

**设计约束**：

1. 不改变 VIT 图拓扑与算子（同一输入应得到同一 embedding）。
2. 固定分辨率下可复用计算图；分辨率变化时必须安全重建。
3. 不依赖减少 `graph_compute` 步数来「虚快」。

---

## 2. 优化方案概览

### 方案 1：固定分辨率 VIT 图复用

**问题（优化前）**

每次 `clip_image_batch_encode()` 都执行：

```text
ggml_backend_sched_reset → clip_image_build_graph → ggml_backend_sched_alloc_graph → set_inputs → graph_compute
```

机器人/机位场景下 **图像 `nx×ny` 通常固定**，图拓扑不变，重复 build+alloc 约 **1.5–2 ms/帧**。

**做法**

1. 在 `clip_ctx` 中缓存 PI0 的 `ggml_context` 与 `ggml_cgraph`（`pi0_ctx0_ptr`、`pi0_gf`），并记录 `pi0_cached_nx`、`pi0_cached_ny`。
2. 当本次 `nx、ny` 与缓存一致：跳过 `sched_reset`、建图、`alloc_graph`，直接 `graph_compute`。
3. 当分辨率变化：调用 `clip_pi0_invalidate_graph_cache()`，再按原流程重建并持久化图上下文（避免 `clip_graph` 栈对象析构后图失效）。

**持久化要点**

`clip_image_build_graph(..., persist_pi0_cache=true)` 在 PI0 路径将 `clip_graph::ctx0_ptr` **move** 到 `clip_ctx::pi0_ctx0_ptr`，防止局部 `clip_graph` 析构释放 meta buffer 上的图。

**日志**

```
graph_build+alloc:    0.xxx ms  (reused=1)
```

首次或分辨率变化后为 `reused=0`。

**关闭复用（调试用）**

```bash
export CLIP_PI0_VIT_GRAPH_REUSE_DISABLE=1
```

### 方案 2：preprocess 直写 `inp_raw` 布局 + buffer 复用

**问题（优化前）**

1. **preprocess**：`normalize_image_u8_to_f32_pi0` 写出 **RGB 交错**布局（`RGBRGB...`）。
2. **set_inputs**：`clip_image_batch_encode` 内再循环做 **平面重排**（R/G/B 分三块），然后 `ggml_backend_tensor_set` 上传。

同一帧像素被布局转换两次，且每帧 `vector` 分配 `inp_raw`。

**`inp_raw` 目标布局（与 graph 输入一致）**

```text
┌── W ──┐
│   H   │  channel = R
├───────┤
│   H   │  channel = G
├───────┤
│   H   │  channel = B
└───────┘
```

**做法**

1. 新增 `normalize_image_u8_to_f32_pi0_inp_raw()`：normalize 时直接写入上述平面布局。
2. `clip_image_f32` 增加 `inp_raw_layout = true` 标记。
3. `set_inputs`：PI0 且 `inp_raw_layout` 时，直接 `tensor_set(img->buf)`，跳过重排循环。
4. **`positions_data`**（`0 .. n_patches-1`）缓存在 `clip_ctx::pi0_positions_buf`；仅在 **图未复用**（新建图或分辨率变化）时上传；图复用时跳过（数值不变）。

---

## 3. 涉及文件

| 文件 | 改动要点 |
|------|----------|
| `tools/mtmd/clip.cpp` | 图缓存、`clip_image_build_graph` 持久化、`clip_image_batch_encode` 复用分支、preprocess 平面布局 |
| `tools/mtmd/clip-impl.h` | `clip_image_f32::inp_raw_layout` |
| `tools/mtmd/clip.h` | `clip_pi0_vit_perf::graph_reused` |
| `tools/mtmd/mtmd-helper.cpp` | 汇总 VIT breakdown 到 `mtmd_pi0_result`（只读 perf，无逻辑变更） |

---

## 4. 数据流（优化后）

```text
clip_image_preprocess (PI0)
  └─ slice + normalize_image_u8_to_f32_pi0_inp_raw  →  buf 为 inp_raw 平面布局

clip_image_batch_encode
  ├─ [nx,ny 相同] 复用 pi0_gf，跳过 build+alloc
  ├─ tensor_set(inp_raw)  ← 直接来自 img->buf
  ├─ [图未复用] set positions_data ← pi0_positions_buf
  ├─ graph_compute
  └─ output_get → embedding
```

---

## 5. Profiling 使用方式

### 环境变量

```bash
export LLAMA_PI0_VIT_PERF=1    # 开启 VIT 分阶段日志
export LLAMA_PI0_PERF=1        # pipeline 总览（含 vit_ms）
```

### 启动 server 后测试

```bash
./run_foreground_pipeline.sh
```

### 日志解读

| 字段 | 含义 |
|------|------|
| `preprocess slice/normalize` | `clip_image_preprocess` 内 UHD slice + 归一化 |
| `graph_build+alloc` | 建图 + scheduler alloc；优化后复用帧应 ≈0 ms |
| `(reused=1)` | 命中 PI0 图缓存 |
| `set_inputs` | 填 `inp_raw`、`positions_data` 等 |
| `graph_compute` | GPU ViT 前向（主瓶颈） |
| `output_get` | embedding D2H |

**预期（固定分辨率、第 2 帧起）**：

- `graph_build+alloc` 明显下降，`reused=1`
- `set_inputs` 略有下降（无重排、positions 跳过）
- `graph_compute` 与优化前同量级

---

## 6. 踩坑与说明

### 6.1 图缓存不能与 warmup 混用

模型加载时 `alloc_compute_meta` / warmup 会调用 `clip_image_build_graph`，但 **不** 设置 `persist_pi0_cache`。仅在 `clip_image_batch_encode` 中、且 `persist_pi0_cache=true` 时写入缓存，避免 warmup 的假分辨率污染真实推理。

### 6.2 分辨率变化必须 invalidate

`nx` 或 `ny` 变化时会 `clip_pi0_invalidate_graph_cache()` 并完整重建，否则 patch 数、tensor shape 与图不一致。

### 6.3 `graph_compute` 不应因优化而「变快」

若通过错误 input shape 减少计算量，属于正确性问题。本优化 **只减 build/set_inputs**，不改 ViT 层数与 attention 逻辑。

### 6.4 Flash Attention 与 GPU 后端

`graph_compute` 仍依赖：

- `CLIP using CUDA`（或指定 `MTMD_BACKEND_DEVICE`）
- 日志中 `flash attention is enabled`

与本次工程优化正交，但应优先确认已开启。

### 6.5 仅 PI0 projector

图复用、`inp_raw_layout` 路径仅在 `PROJECTOR_TYPE_PI0` 启用；其他 mmproj 类型保持原逻辑。

---

## 7. 验证步骤

1. 编译并重启服务（须加载新 `libmtmd.so`）：

   ```bash
   cd build && cmake --build . --target mtmd llama-server
   # 重启 llama-server
   ```

2. 固定分辨率连续请求两次以上，查看 VIT 日志：
   - 第 1 次：`reused=0`
   - 第 2 次起：`reused=1`，`graph_build+alloc` 接近 0

3. 更换图像尺寸（若 pipeline 支持），应自动 `reused=0` 并重建图。

4. 对比优化前后 **总 vit_ms** 与 **embedding 数值**（可选：同一帧两次 encode 差分应仅浮点噪声级）。

---

## 8. 设计原则（可复用）

1. **先 profiling**：区分 `graph_compute` 与 build/set_inputs，避免误优化 GPU 主路径。
2. **固定拓扑 → 复用图**：与 LLM decode 的 `can_reuse` 思路一致；VIT 每帧只跑一次，收益来自省略重复 build/alloc。
3. **布局一次到位**：preprocess 输出与 graph input 布局一致，避免 CPU 二次重排。
4. **静态输入只填一次**：`positions_data` 随图生命周期上传，图复用时跳过。
5. **显式失效**：分辨率变化、禁用 env 时走完整路径，保证正确性。

---

## 9. 后续可探索（未实施）

| 方向 | 说明 |
|------|------|
| CUDA Graph capture | 拓扑固定时对 `graph_compute` 做 capture，降低 launch 开销 |
| embedding 留 GPU | `output_get` 后直接供 LLM，省 D2H（需改 mtmd/llama 接口） |
| pinned host buffer | 加速 `output_get`，数值不变 |
| Q8 mmproj | 换权重，需精度验证 |

---

## 10. API / 环境变量速查

| 名称 | 作用 |
|------|------|
| `LLAMA_PI0_VIT_PERF` | 打印 VIT preprocess / encode 分阶段耗时 |
| `CLIP_PI0_VIT_GRAPH_REUSE_DISABLE` | 设为非 0 关闭固定分辨率图复用 |
| `MTMD_BACKEND_DEVICE` | 指定 CLIP/VIT 使用的 backend 设备名 |
| `clip_image_f32::inp_raw_layout` | 为 true 表示 `buf` 已是 inp_raw 平面布局 |
| `clip_pi0_vit_perf::graph_reused` | 本次 encode 是否复用缓存图 |

---

*文档版本：2026-05 Jetson-PI-2 分支 PI0 VIT 优化（图复用 + inp_raw 布局）。*
