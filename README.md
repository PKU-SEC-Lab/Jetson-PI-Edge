# Jetson-PI

Jetson-PI is the codebase for the paper "Jetson-PI: Towards Onboard Real-Time Robot Control via Foresight-Aligned Asynchronous Inference." It serves PI0 and PI0.5 models through a llama.cpp-based HTTP server and supports multiple deployment backends for embedded robot inference, including CPU-only builds, NVIDIA Jetson Orin/Thor, and NPU-oriented integrations.

The main runtime path is the foreground server used by robot control loops. Images, robot state, and instruction text are submitted to a persistent server session, and the server returns action tensors with timing breakdowns. This keeps the model interface simple for robot applications: send sensor inputs, run one foreground inference call, and consume the final action output.

This project is based on [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Features

- Efficient VLA inference engine built on llama.cpp.
- Automatic PI model detection from GGUF metadata and PI-specific tensor names.
- Foreground HTTP APIs for persistent image/state/instruction sessions.
- Optimized foreground scheduling and graph reuse for PI encoder and action expert paths.
- Backend-flexible builds for Jetson devices and other edge devices.
- PI0.5 preprocessing and action expert support.
- GGUF conversion scripts for PI language and vision components.

## Build

For a CPU-only build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

For NVIDIA Jetson Orin or other CUDA-capable targets:

```bash
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

Other ggml/llama.cpp backends can be selected with the corresponding CMake options for the target platform. Keep the foreground server target the same; only the backend configuration changes.

Use the same build directory for later rebuilds:

```bash
cmake --build build --target llama-server -j
```

## Model Preparation

See [docs/model_conversion.md](docs/model_conversion.md) for model download, config preparation, PI surgery, and GGUF conversion steps.

## Start The Foreground Server

The server command is the same across backends. On GPU builds, `-ngl 99` offloads model layers to the accelerator. On CPU-only builds, omit `-ngl` or set it to `0`.

```bash
PI_MODEL=auto \
PI0_ACTION_NOISE_BIN=/path/to/noise_10x32_or_50x32.bin \
./build/bin/llama-server \
  -m /path/to/pi_llm.gguf \
  --mmproj /path/to/mmproj.gguf \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8080
```

For PI0.5, use the PI0.5 LLM and matching PI0.5 vision GGUF. For PI0, use the PI0 LLM and matching PI0 vision GGUF.

## Minimal Foreground Example

Reset the persistent foreground session:

```bash
curl -X POST http://127.0.0.1:8080/foreground/reset
```

Submit image inputs:

```bash
curl -X POST http://127.0.0.1:8080/foreground/image \
  -H 'Content-Type: application/json' \
  -d '{"path":"/path/to/image_1.png"}'

curl -X POST http://127.0.0.1:8080/foreground/image \
  -H 'Content-Type: application/json' \
  -d '{"path":"/path/to/image_2.png"}'
```

Submit the robot state:

```bash
curl -X PUT http://127.0.0.1:8080/foreground/state \
  -H 'Content-Type: application/json' \
  -d '{"state":"1.8731,-1.0370,1.9652,7.0876,0.2546,-9.1432,-0.0147,-0.5037,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"}'
```

Run inference:

```bash
curl -X POST http://127.0.0.1:8080/foreground/infer \
  -H 'Content-Type: application/json' \
  -d '{"text":"pick up the object and place it into the tray"}'
```

Inspect the session:

```bash
curl http://127.0.0.1:8080/foreground/session
```

See [docs/foreground_server_usage.md](docs/foreground_server_usage.md) for endpoint details and response fields.

## Response Fields

PI foreground responses commonly include:

- `is_pi0`: whether the PI action path was used.
- `pi_model`: active model family, such as `pi0` or `pi05`.
- `state`: state vector used or returned by the action path.
- `action`: flattened action buffer.
- `action_final`: final flattened action output.
- `encode_ms`: image/text/state encoding latency.
- `decode_ms`: action expert latency.
- `total_ms`: model-side total latency.
- `timing_breakdown_ms`: server-side request timing details.

## Jetson Orin Latency

Latency is reported in milliseconds on Jetson Orin.

### PI0 Latency on Jetson Orin

| Method | ViT (ms) | LLM (ms) | Action Expert (ms) | Total (ms) |
|---|---:|---:|---:|---:|
| Naive PI0 | 143.5 | 601.9 | 505.5 | 1250.9 |
| + Schedule opt. | 147.1 | 603.1 | 501.3 | 1251.5 |
| + Graph reuse | 76.8 | 200.6 | 167.0 | 444.4 |
| + Intermediate Buffer & Unroll| 75.4 | 200.3 | 118.8 | 394.5 |


### PI0.5 Latency on Jetson Orin

| Method | ViT (ms) | LLM (ms) | Action Expert (ms) | Total (ms) |
|---|---:|---:|---:|---:|
| Naive PI0.5 | 152.3 | 631.0 | 536.8 | 1420.8 |
| + Schedule opt. | 152.3 | 631.0 | 536.8 | 1420.8 |
| + Graph reuse | 79.5 | 212.6 | 184.0 | 476.1 |
| + Intermediate Buffer & Unroll | 79.5 | 210.3 | 123.1 | 412.9 |

## Notes

- The foreground server is the supported runtime path in this repository.
- Models natively supported by llama.cpp remain supported through the standard llama.cpp runtime path.
- Ordinary llama GGUF files still load through llama.cpp's standard `llama` architecture path, not the PI foreground action path.
- PI0.5 supports its own image preprocessing, prompt construction, and action expert tensor layout.
- PI0 and PI0.5 conversion utilities are kept under the llama.cpp/GGUF toolchain in this repository.
