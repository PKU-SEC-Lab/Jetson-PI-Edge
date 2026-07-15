# Model Preparation

This guide prepares PI0 and PI0.5 checkpoints for the Jetson-PI foreground server. The server needs two GGUF files:

- LLM/action GGUF: converted with `convert_hf_to_gguf.py`
- vision encoder/projector GGUF: converted with `tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py`

The conversion commands are the same for PI0 and PI0.5. The only difference is the source checkpoint path. The examples below use:

- `PATH/TO/PI_MODEL`: the downloaded PI0 or PI0.5 checkpoint
- `PATH/TO/VIT`: the temporary vision encoder/projector directory

## 1. Install Conversion Dependencies

```bash
python3 -m pip install -r requirements/requirements-convert_hf_to_gguf.txt
python3 -m pip install -r tools/mtmd/requirements.txt
```

## 2. Download The Source Model

Download the PI checkpoint from Hugging Face:

```bash
huggingface-cli download PI_REPO \
  --local-dir PATH/TO/PI_MODEL \
  --local-dir-use-symlinks False
```

Common source repositories:

| Model | `PI_REPO` |
|---|---|
| PI0 | `lerobot/pi0_base` |
| PI0.5 | `lerobot/pi05_base` |

Use the same remaining commands for both model families.

## 3. Patch `config.json` For GGUF Conversion

Before conversion, make sure the source model `config.json` contains the PI architecture fields needed by the converter and runtime auto-detection.

Merge the following fields into `PATH/TO/PI_MODEL/config.json`:

```json
{
  "architectures": [
    "pi0"
  ],
  "hidden_act": "gelu",
  "hidden_size": 2048,
  "intermediate_size": 16384,
  "hidden_size_ae": 1024,
  "intermediate_size_ae": 4096,
  "model_type": "pi0",
  "num_attention_heads": 8,
  "num_hidden_layers": 36,
  "num_key_value_heads": 1,
  "use_cache": true,
  "head_dim": 256,
  "rms_norm_eps": 1e-05,
  "vocab_size": 257152,
  "max_position_embeddings": 131072
}
```

Do not keep duplicate JSON keys. If the downloaded config already has `max_position_embeddings`, replace it with the final value above.

If your model architecture differs from the base PI0/PI0.5 architecture, use the matching config values for that model instead of the defaults above.

You can patch the config with:

```bash
python3 - <<'PY'
import json
from pathlib import Path

cfg_path = Path("PATH/TO/PI_MODEL/config.json")
cfg = json.loads(cfg_path.read_text())
cfg.update({
    "architectures": ["pi0"],
    "hidden_act": "gelu",
    "hidden_size": 2048,
    "intermediate_size": 16384,
    "hidden_size_ae": 1024,
    "intermediate_size_ae": 4096,
    "model_type": "pi0",
    "num_attention_heads": 8,
    "num_hidden_layers": 36,
    "num_key_value_heads": 1,
    "use_cache": True,
    "head_dim": 256,
    "rms_norm_eps": 1e-05,
    "vocab_size": 257152,
    "max_position_embeddings": 131072,
})
cfg_path.write_text(json.dumps(cfg, indent=4) + "\n")
PY
```

## 4. Split The PI Model

Run the PI surgery helper to split the checkpoint into language and vision/projector components:

```bash
python3 tools/mtmd/legacy-models/pi0_surgery.py \
  -C \
  -m PATH/TO/PI_MODEL
```

The `-C` option removes the embedded vision tower from the language model files after extracting the vision weights. This keeps the LLM GGUF focused on the PI language/action path.

## 5. Prepare The Vision Directory

Create a vision directory and copy the extracted vision files into it. Jetson-PI ships separate vision config templates in the repository root: use `vit_config.json` for PI0 and `vit_config_pi05.json` for PI0.5.

```bash
mkdir -p PATH/TO/VIT
cp PATH/TO/PI_MODEL/llava.clip PATH/TO/VIT/pytorch_model.bin
cp PATH/TO/PI_MODEL/llava.projector PATH/TO/VIT/
cp ./vit_config.json PATH/TO/VIT/config.json       # PI0
# or
cp ./vit_config_pi05.json PATH/TO/VIT/config.json  # PI0.5
```

If your model architecture differs from the base PI0/PI0.5 architecture, copy the matching vision config for that model instead of these base templates.

## 6. Convert The Vision Encoder

Convert the PI vision encoder/projector to GGUF:

```bash
python3 tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py \
  -m PATH/TO/VIT \
  --llava-projector PATH/TO/VIT/llava.projector \
  --output-dir PATH/TO/VIT \
  --clip-model-is-siglip \
  --vision-only \
  --projector-type pi0
```

The generated file is the `--mmproj` argument used by `llama-server`.

## 7. Convert The Language/Action Model

Convert the PI language/action model to GGUF:

```bash
python3 convert_hf_to_gguf.py \
  PATH/TO/PI_MODEL \
  --outfile PATH/TO/PI_MODEL/pi_llm.gguf \
  --outtype f16
```

PI0 and PI0.5 use the same conversion entry points. At serving time, Jetson-PI first checks GGUF architecture metadata and then inspects PI-specific tensor names to select the PI0 or PI0.5 runtime branch.

## 8. Check The Converted Files

After conversion, you should have:

```text
PATH/TO/PI_MODEL/pi_llm.gguf
PATH/TO/VIT/mmproj-model-f16.gguf
```

Use these files with the foreground server:

```bash
PI_MODEL=auto \
./build/bin/llama-server \
  -m PATH/TO/PI_MODEL/pi_llm.gguf \
  --mmproj PATH/TO/VIT/mmproj-model-f16.gguf \
  --host 0.0.0.0 \
  --port 8080
```
