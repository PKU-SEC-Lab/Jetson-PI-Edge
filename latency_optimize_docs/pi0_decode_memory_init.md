# PI0 Decode 优化（第四步）：跳过 KV cache 的 memory_init_batch

## 1. 问题

Profiling 显示 decode 约 **29 ms** 中 `memory_init_batch` 占 **~3.7 ms（13%）**，而 diffusion 的 `set_inputs` 已压到 **~0.6 ms**。

根因：PI0 **decoder 不使用** `llama_kv_cache`（cross-attn 走 `build_inp_cross_kv_pi0` + GPU KV），但 `decode()` 仍对整批 encoder token（如 **773**）调用：

```cpp
memory->init_batch(*balloc, n_ubatch, ...)  // → prepare() 做 KV slot 搜索
```

这是对无用工作的 **O(n_tokens)** 开销。

## 2. 做法

对 `LLM_ARCH_PI0` 的 `decode()`：

1. **跳过** `memory_update()` 与 `memory->init_batch()`
2. 与 `encode()` 一致，用 `balloc->split_simple(n_tokens_all)` 得到 `ubatch`
3. `process_ubatch(..., nullptr)` — 不调用 `mctx->apply()`，不写 KV cache

非 PI0 模型仍走原有 `init_batch` 路径。

## 3. 修改位置

`src/llama-context.cpp` — `llama_context::decode()`：在 `output_reserve` 之后对 PI0 提前分支并 `return`；通用路径保留原 `memory_update` + `init_batch` 循环。

## 4. 预期收益

- `memory_init_batch`：**~3.7 ms → 接近 0**（仅 `split_simple`）
- **TOTAL decode**：约 **29 ms → ~25 ms**（约 15%）
- 数值：与优化前一致（不改变 diffusion / cross-KV 语义）

## 5. 验证

```bash
export LLAMA_PI0_PERF=1
# 重启 llama-server
./run_foreground_pipeline.sh
```

关注 stderr：

```
memory_init_batch    < 0.1 ms
TOTAL decode         ~ 25 ms 量级
action_final         与之前一致
```

## 6. 注意点

- **不要**对 PI0 去掉 `memory` 对象本身：`graph_reserve` 等仍可能用到 KV cache 缓冲区布局。
- `ubatch` 仍来自完整 `combined_batch`（与旧路径 `mctx->get_ubatch()` 首包一致）。
- 若将来 PI0 decoder 改用标准 KV cache，需恢复 `init_batch`。

## 7. 参考

- [pi0_decode_graph_reuse.md](./pi0_decode_graph_reuse.md)
- [pi0_decode_kv_gpu.md](./pi0_decode_kv_gpu.md)
- [pi0_decode_mask_reuse.md](./pi0_decode_mask_reuse.md)
