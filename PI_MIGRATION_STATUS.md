# PI0/PI05 迁移工作记录

更新时间：2026-07-13

## 背景

`${PI_LEGACY_ROOT}/Jetson-PImerge_newestcpp/llama.cpp-master` 是已经融合过 PI0/PI05 的旧版 llama.cpp。当前目标是把 PI0/PI05 可靠迁移到最新版 llama.cpp：

文档使用以下环境变量表示机器相关路径：

```text
PI_REPO, PI_LEGACY_ROOT, PI_PIPELINE_SCRIPT, ACTION_FINAL_OUT
PI0_MODEL_FILE, PI0_MMPROJ_FILE, PI05_MODEL_FILE, PI05_MMPROJ_FILE
```

要求是在不破坏新版 llama.cpp 原始功能的前提下，完成 PI0/PI05 的迁移、CUDA build、GPU server 启动和参考结果验证。

## 一开始评估的两种方案

### 方案 1：在新版 llama.cpp 中实现 PI0/PI05

参考目录：

```text
${PI_LEGACY_ROOT}/Jetson-PImerge_beifen
${PI_LEGACY_ROOT}/Jetson-PImerge_newestcpp/llama.cpp-master
```

目标目录：

```text
${PI_REPO}
```

思路：

- 以新版 llama.cpp 为基底。
- 从旧版 PI0/PI05 实现中识别必要修改。
- 将 PI0/PI05 的模型结构、mtmd 多模态路径、server foreground API、动作输出等能力移植到新版。
- 尽量只在 PI 模型路径生效，避免影响新版 llama.cpp 的普通 LLM/VLM 功能。

优点：

- 最终代码基于最新版 llama.cpp。
- 更容易保持新版原始功能和后续同步能力。
- 风险集中在 PI0/PI05 的新增路径上。

难点：

- 新旧 llama.cpp 的 mtmd、server、graph、backend 接口变化较大。
- PI0 和 PI05 在 prompt、media marker、prefix KV、action decoder、state/action 后处理上有差异，需要逐项适配。

### 方案 2：在旧版融合代码上替换新版差异

参考目录：

```text
${PI_LEGACY_ROOT}/Jetson-PImerge_newestcpp/
```

思路：

- 以旧版已融合 PI0/PI05 的代码为基底。
- 将新版 llama.cpp 和旧版差异较大的部分替换/补齐。

优点：

- PI0/PI05 旧实现已经能跑，初期看起来迁移量较小。

缺点：

- 等于把新版 llama.cpp 的大量变化反向合并进旧仓库。
- 更容易破坏新版功能。
- 后续维护难度更高，也更难判断哪些变化是新版必要行为、哪些是旧版私有修改。

## 当前选择的方案

当前执行的是 **方案 1**：

```text
以 ${PI_REPO} 为目标，在最新版 llama.cpp 中移植 PI0/PI05。
```

选择原因：

- 用户要求最终落到最新版 llama.cpp。
- 该方案更符合“不要影响新版原始功能”的目标。
- 虽然移植过程更细，但边界更清晰：PI 相关逻辑尽量只在 PI 模型自动识别后启用。

## 当前工作目录


目标代码目录：

```text
${PI_REPO}
```

旧版参考目录：

```text
${PI_LEGACY_ROOT}/Jetson-PImerge_beifen
${PI_LEGACY_ROOT}/Jetson-PImerge_newestcpp
```

验证脚本：

```text
${PI_PIPELINE_SCRIPT}
```

主要参考输出：

```text
${PI_LEGACY_ROOT}/Jetson-PImerge_beifen/action_final_pi0_merge.txt
${PI_LEGACY_ROOT}/Jetson-PImerge_beifen/action_final_pi05_merge.txt
```

pipeline 当前保存输出：

```text
${ACTION_FINAL_OUT}
```

## 已经完成的部分

### Build

在目标目录完成过 CUDA 配置和完整 build：

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --parallel
```

`llama-server` 目标也能单独 build：

```bash
cmake --build build --target llama-server --parallel
```

### Server 启动

PI0 server 已能用 GPU 按指定方式启动：

```bash
./build/bin/llama-server \
  -m ${PI0_MODEL_FILE} \
  --mmproj ${PI0_MMPROJ_FILE} \
  -ngl 37 \
  --chat-template vicuna \
  --host 0.0.0.0 \
  --port 8080
```

PI05 server 已能用 GPU 按指定方式启动：

```bash
./build/bin/llama-server \
  -m ${PI05_MODEL_FILE} \
  --mmproj ${PI05_MMPROJ_FILE} \
  -ngl 37 \
  --chat-template vicuna \
  --host 0.0.0.0 \
  --port 8080
```

### 已移植/修复的关键点

- PI0/PI05 GGUF 自动识别，并自动设置 `PI_MODEL=pi0` 或 `PI_MODEL=pi05`。
- PI 模型 foreground server 路径。
- PI0/PI05 action 输出字段：
  - `action`
  - `action_final`
  - `action_final_raw`
  - `joint_command`
  - `joint_command_final`
- PI05 OpenPI 风格 prompt 包装：
  - `Task: <text>, State: <bins>;\nAction: `
- PI05 foreground text token 处理，避免 BOS 导致尾部 text chunk 被跳过。
- PI0/PI05 media marker 对齐，foreground image 使用固定 marker。
- PI05 prefix KV shape 适配。
- CUDA scheduler split input 上限调整，避免 PI 路径 split-input 崩溃。
- PI0 action noise 文件兼容逻辑。
- PI0 action norm stats response 兼容逻辑。
- 清理了主要临时 debug stderr 输出。

### 当前验证结果

PI05 已通过当前参考余弦验证：

```text
cosine(action_final, action_final_pi05_merge.txt) = 0.9753304220
```

PI0 当前 GPU server 和 pipeline 能跑通，但参考余弦仍未达到“确认通过”的程度：

```text
cosine(action_final, action_final_pi0_merge.txt) = 0.8735884120
```

重要发现：

- 用旧版 beifen 目录按当前 `run_foreground_pipeline.sh` 输入重新跑 PI0 时，也不能很好匹配 `action_final_pi0_merge.txt`。
- 因此 PI0 的参考文件可能不是由当前 foreground pipeline 的同一组输入生成。
- 这意味着 PI0 还需要继续确认参考生成路径，或者继续定位当前新版 PI0 输出和参考之间的剩余差异。

## 最后要达到的目标

最终完成条件如下。

### 1. 新版 llama.cpp CUDA build 通过

目标目录：

```text
${PI_REPO}
```

命令：

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --parallel
```

要求：

- build 成功。
- `build/bin/llama-server` 存在且可运行。

### 2. PI0 GPU server 跑通并通过参考验证

server 使用用户指定命令启动。

然后运行：

```bash
ACTION_FINAL_OUT=${ACTION_FINAL_OUT} \
bash ${PI_PIPELINE_SCRIPT}
```

要求：

```text
${ACTION_FINAL_OUT}
```

与：

```text
${PI_LEGACY_ROOT}/Jetson-PImerge_beifen/action_final_pi0_merge.txt
```

余弦相似度大于0.999，能够确认 PI0 移植正确。

### 3. PI05 GPU server 跑通并通过参考验证

server 使用用户指定命令启动。

然后运行同一个 foreground pipeline。

要求：

```text
${ACTION_FINAL_OUT}
```

与：

```text
${PI_LEGACY_ROOT}/Jetson-PImerge_beifen/action_final_pi05_merge.txt
```

余弦相似度大于0.999，能够确认 PI05 移植正确。


### 4. 保持新版 llama.cpp 原始功能

迁移代码应尽量满足：

- PI0/PI05 逻辑只在识别到 PI 模型时启用。
- 普通 llama.cpp server、普通文本模型、普通 mtmd/VLM 路径不被 PI 私有逻辑污染。
- 不用旧版整体覆盖新版文件。
- 保留新版已有接口和构建方式。

## 后续继续工作的重点

下一步应优先解决 PI0/PI05 余弦未完全对齐的问题：

1. 继续追踪 `action_final_pi0_merge.txt` 的真实生成输入：
   - 图片路径
   - state 符号和值
   - infer text
   - image count
   - 是否使用旧 request 脚本而不是当前 pipeline
2. 对比旧版 PI0/PI05 同输入和新版 PI0/PI05 同输入的中间结果：
   - token/chunk 顺序
   - image embedding
   - encoded KV
   - action noise
   - AE decoder output
3. 若确认参考文件来自不同输入，应更新验证基准或明确生成方式；若确认输入相同，则继续修正新版 PI0/PI05 路径直到余弦通过。

## 2026-07-16：PI05 ViT 计算图向旧版对齐

同输入 dump 已确认 token、position、text token 和 10x32 action noise 一致，
首个显著差异位于 image mmproj。进一步审计发现新版 PI05 adapter 曾将
batched ViT 硬编码关闭，并用占位函数替代旧版批量编码接口。

本轮修改：

1. `tools/mtmd/mtmd-helper-pi05.cpp`
   - 使用新版 `mtmd_batch_*` API 恢复多图单次 ViT 编码。
   - 默认启用 batched ViT。
   - `PI0_DISABLE_VIT_BATCH=1` 仅保留为诊断回退开关。
   - 按输入 chunk 顺序复制每张图的 embedding，保持 prefix 中两张图的顺序不变。
2. `tools/mtmd/models/models.h`
   - 仅为 `PROJECTOR_TYPE_PI0` 开启 SigLIP batch capability。
   - 其他新版 SigLIP/VLM 路径不受影响。
3. `tools/mtmd/models/siglip.cpp`
   - PI05 patch embedding 恢复旧版的 F16 kernel -> F32 显式转换。
   - 多图采用逐图 convolution，再沿 batch 维拼接，避免新版单次 batched convolution 改变 kernel/舍入。
   - PI05 transformer 恢复旧版 3D/4D batch tensor 流。
   - attention 恢复旧版“先 scale Q，再 QK，softmax scale=1”的计算顺序。
   - FFN 沿用新版中已有的 PI0 contiguous 处理，其运算顺序与旧版专用 FFN 一致。

验证状态：

- `cmake --build build --target llama-server -j4` 已通过。
- 尚需用固定输入和
  `PI0_ACTION_NOISE_BIN=${PI05_NOISE_FILE}`
  重跑 server dump，重点比较 `image_0_mmproj`、`image_1_mmproj`、prefix KV 和最终 action。
