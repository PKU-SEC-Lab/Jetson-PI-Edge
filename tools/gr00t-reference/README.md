# GR00T N1.7 validation tools

This directory contains reference and regression utilities for the Jetson-PI GR00T
implementation. The exporter reads an NVIDIA Isaac-GR00T checkout and official
checkpoints without modifying them. Generated checkpoints, fixtures, GGUF files, and
binary outputs must not be committed.

The examples below assume these paths are configured for the local machine:

```bash
export EDGE_DIR=/path/to/Jetson-PI-Edge
export GR00T_SOURCE=/path/to/Isaac-GR00T
export GR00T_CHECKPOINT=/path/to/GR00T-N1.7-3B
export COSMOS_CHECKPOINT=/path/to/Cosmos-Reason2-2B
export REFERENCE_DIR=/path/to/gr00t-n1d7-reference
export GGUF_DIR=/path/to/gguf
```

## Export the PyTorch reference

Create the official Isaac-GR00T Python environment, then run:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
NO_ALBUMENTATIONS_UPDATE=1 \
python "$EDGE_DIR/tools/gr00t-reference/export_gr00t_n1d7_reference.py" \
  --gr00t-source "$GR00T_SOURCE" \
  --checkpoint "$GR00T_CHECKPOINT" \
  --cosmos "$COSMOS_CHECKPOINT" \
  --output "$REFERENCE_DIR" \
  --device cuda:0
```

The fixture uses deterministic synthetic DROID images and state, so it does not require
a Git LFS dataset. Append `--processor-only` for a processor-only check. A complete run
writes `reference.safetensors` and a `manifest.json` containing source hashes, tensor
metadata, and environment versions.

## Convert and inspect GGUF files

Run from the Jetson-PI-Edge checkout:

```bash
python convert_hf_to_gguf.py "$GR00T_CHECKPOINT" \
  --gr00t-cosmos "$COSMOS_CHECKPOINT" \
  --outtype bf16 \
  --outfile "$GGUF_DIR/gr00t-n1d7-bf16.gguf"

python convert_hf_to_gguf.py "$GR00T_CHECKPOINT" \
  --gr00t-cosmos "$COSMOS_CHECKPOINT" \
  --mmproj \
  --outtype bf16 \
  --outfile "$GGUF_DIR/mmproj-gr00t-n1d7-bf16.gguf"

PYTHONPATH=gguf-py python tools/gr00t-reference/validate_gr00t_n1d7_gguf.py \
  --checkpoint "$GR00T_CHECKPOINT" \
  --model "$GGUF_DIR/gr00t-n1d7-bf16.gguf" \
  --mmproj "$GGUF_DIR/mmproj-gr00t-n1d7-bf16.gguf"
```

The main GGUF contains the Qwen3-VL text backbone and GR00T Action Head. The mmproj
contains the Qwen3-VL vision tensors used by the multimodal runtime.

## Runtime validation utilities

- `gr00t_product_validate.cpp` exercises the public `libjetson_pi_gr00t` API from raw
  RGB frames, instruction, and state through a complete `[40, 132]` action result.
- `gr00t_backbone_validate.cpp` accepts prepared language embeddings and M-RoPE
  positions and can capture decoder layer outputs through the evaluation callback.
- `gr00t_vision_validate.cpp` runs four deterministic images through the Qwen3-VL
  vision path and captures selected graph tensors through the evaluation callback.
- `pi_policy_regression.cpp` checks the existing Pi policy APIs for finite, correctly
  shaped, repeatable output.
- `qwen3vl_regression.cpp` checks ordinary non-GR00T Qwen3-VL prefill for finite,
  repeatable logits.
- `prepare_runtime_fixture.py` extracts binary inputs from `reference.safetensors`.
- `compare_runtime_action.py` compares two `[40, 132]` float action files.

For example, build the public API smoke validator after building the project:

```bash
g++ -std=c++17 -O2 tools/gr00t-reference/gr00t_product_validate.cpp \
  -Iinclude -Lbuild/bin -ljetson_pi_gr00t \
  -Wl,-rpath,"$PWD/build/bin" \
  -o build/bin/gr00t-product-validate
```

The validators intentionally use public APIs or evaluation callbacks. Production
inference does not provide environment-variable hooks for replacing model inputs or
intermediate tensors.
