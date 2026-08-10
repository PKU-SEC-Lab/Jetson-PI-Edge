#!/usr/bin/env python3
"""Export deterministic GR00T N1.7 tensors for Jetson-PI validation.

This tool intentionally imports NVIDIA's reference implementation from a separate,
read-only checkout. It uses a synthetic DROID observation so the fixture is small,
deterministic, and independent of Git LFS demo data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file


EMBODIMENT_NAME = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
EMBODIMENT_VALUE = "oxe_droid_relative_eef_relative_joint"
LANGUAGE = "Pick up the red block, then place it beside the blue block."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cosmos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--processor-only",
        action="store_true",
        help="Export deterministic observation and processor tensors without loading the model.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_image(camera: int, timestep: int, height: int = 180, width: int = 320) -> np.ndarray:
    """Generate an RGB pattern with no RNG or image codec dependency."""
    y, x = np.indices((height, width), dtype=np.uint16)
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = (x + 17 * camera + 29 * timestep) % 256
    image[..., 1] = (y * 3 + 31 * camera + 11 * timestep) % 256
    image[..., 2] = ((x // 2) + (y // 3) + 47 * camera + 7 * timestep) % 256
    return image


def make_observation(processor: Any) -> dict[str, Any]:
    stats = processor.state_action_processor.statistics[EMBODIMENT_VALUE]["state"]
    observation: dict[str, Any] = {
        "annotation.language.language_instruction": [LANGUAGE],
    }
    cameras = ("exterior_image_1_left", "wrist_image_left")
    for camera_index, key in enumerate(cameras):
        frames = [make_image(camera_index, timestep) for timestep in range(2)]
        observation[f"video.{key}"] = np.stack(frames, axis=0)[None, ...]
    for key in ("eef_9d", "gripper_position", "joint_position"):
        # Statistics means are valid physical-space values and normalize away from
        # clipping boundaries. Add a small deterministic offset to avoid an all-zero
        # normalized state while remaining well inside the observed distribution.
        mean = np.asarray(stats[key]["mean"], dtype=np.float32)
        std = np.asarray(stats[key]["std"], dtype=np.float32)
        value = mean + np.float32(0.125) * np.maximum(std, np.float32(1e-4))
        observation[f"state.{key}"] = value.reshape(1, 1, -1)
    return observation


def as_cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().cpu()


def add_tensor(tensors: dict[str, torch.Tensor], name: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensors[name] = as_cpu_tensor(value)


def add_tree(tensors: dict[str, torch.Tensor], prefix: str, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        add_tensor(tensors, prefix, value)
    elif isinstance(value, dict) or hasattr(value, "items"):
        for key, item in value.items():
            add_tree(tensors, f"{prefix}.{key}", item)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            add_tree(tensors, f"{prefix}.{index}", item)


def install_hooks(model: Any, tensors: dict[str, torch.Tensor]):
    handles = []
    calls: defaultdict[str, int] = defaultdict(int)

    def hook(name: str):
        def capture(_module, _inputs, output):
            call = calls[name]
            calls[name] += 1
            add_tree(tensors, f"hook.{name}.call_{call}", output)

        return capture

    def pre_hook(name: str):
        def capture(_module, inputs):
            add_tree(tensors, f"hook.{name}.call_0", inputs)

        return capture

    visual = model.backbone.model.visual
    handles.append(visual.register_forward_hook(hook("vision.output")))
    handles.append(visual.patch_embed.register_forward_hook(hook("vision.patch_embed")))
    handles.append(visual.blocks[0].register_forward_pre_hook(pre_hook("vision.block_0_input")))
    for index, block in enumerate(visual.blocks):
        handles.append(block.register_forward_hook(hook(f"vision.block_{index}")))
    def capture_language_input(_module, inputs, kwargs):
        add_tree(tensors, "hook.language.input.args", inputs)
        add_tree(tensors, "hook.language.input.kwargs", kwargs)

    handles.append(
        model.backbone.model.language_model.register_forward_pre_hook(
            capture_language_input, with_kwargs=True
        )
    )
    for index, layer in enumerate(model.backbone.model.language_model.layers):
        handles.append(layer.register_forward_hook(hook(f"language.layer_{index}")))
    for index, block in enumerate(model.action_head.vl_self_attention.transformer_blocks):
        handles.append(block.register_forward_hook(hook(f"vl_self.block_{index}")))
    for index, block in enumerate(model.action_head.model.transformer_blocks):
        handles.append(block.register_forward_hook(hook(f"dit.block_{index}")))
    return handles


def export_processor_tensors(
    processor: Any,
    embodiment: Any,
    observation: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            add_tensor(tensors, f"observation.{key}", torch.from_numpy(value))
    # Capture the exact uint8 frames handed to Qwen3-VL after GR00T's own
    # deterministic evaluation crop/resize. Order is timestep-major, then view.
    if processor.use_albumentations:
        from PIL import Image
        from gr00t.model.gr00t_n1d7.image_augmentations import apply_with_replay

        camera_keys = processor.modality_configs[EMBODIMENT_VALUE]["video"].modality_keys
        raw = torch.stack(
            [torch.from_numpy(observation[f"video.{key}"]) for key in camera_keys], dim=2
        )
        _, timesteps, views, height, width, channels = raw.shape
        flat = raw.reshape(timesteps * views, height, width, channels)
        pil_images = [Image.fromarray(frame.numpy()) for frame in flat]
        transformed, _ = apply_with_replay(processor.eval_image_transform, pil_images)
        add_tensor(tensors, "processor_input.images", torch.stack(transformed))
    processed = processor.process_observation(observation, embodiment)
    add_tree(tensors, "processor", processed)
    return dict(processed)


@torch.inference_mode()
def export_model_tensors(
    model: Any,
    processed: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    seed: int,
) -> torch.Tensor:
    add_tensor(
        tensors,
        "model_weight.language_norm",
        model.backbone.model.language_model.norm.weight,
    )
    backbone_input, action_input = model.prepare_input(dict(processed))
    add_tree(tensors, "model_input.backbone", backbone_input)
    add_tree(tensors, "model_input.action", action_input)

    backbone_output = model.backbone(backbone_input)
    add_tree(tensors, "backbone", backbone_output)

    vl_after_norm = model.action_head.vlln(backbone_output.backbone_features)
    add_tensor(tensors, "action_head.vlln", vl_after_norm)
    vl_features = model.action_head.vl_self_attention(vl_after_norm)
    add_tensor(tensors, "action_head.vl_self_attention", vl_features)

    state = action_input.state.view(action_input.state.shape[0], 1, -1)
    state_features = model.action_head.state_encoder(state, action_input.embodiment_id)
    add_tensor(tensors, "action_head.state_features", state_features)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    actions = torch.randn(
        (vl_features.shape[0], model.config.action_horizon, model.action_head.action_dim),
        dtype=vl_features.dtype,
        device=vl_features.device,
    )
    add_tensor(tensors, "denoise.initial_noise", actions)

    dt = 1.0 / model.action_head.num_inference_timesteps
    for step in range(model.action_head.num_inference_timesteps):
        timestep = int(step / model.action_head.num_inference_timesteps * model.config.num_timestep_buckets)
        timestep_tensor = torch.full(
            (vl_features.shape[0],), timestep, dtype=torch.long, device=vl_features.device
        )
        add_tensor(tensors, f"denoise.step_{step}.action_in", actions)
        add_tensor(tensors, f"denoise.step_{step}.timestep", timestep_tensor)

        action_features = model.action_head.action_encoder(
            actions, timestep_tensor, action_input.embodiment_id
        )
        if model.config.add_pos_embed:
            position_ids = torch.arange(action_features.shape[1], device=vl_features.device)
            position = model.action_head.position_embedding(position_ids).unsqueeze(0)
            add_tensor(tensors, f"denoise.step_{step}.position", position)
            action_features = action_features + position
        add_tensor(tensors, f"denoise.step_{step}.action_features", action_features)

        state_action = torch.cat((state_features, action_features), dim=1)
        add_tensor(tensors, f"denoise.step_{step}.state_action", state_action)
        dit_output = model.action_head.model(
            hidden_states=state_action,
            encoder_hidden_states=vl_features,
            timestep=timestep_tensor,
            image_mask=backbone_output.image_mask,
            backbone_attention_mask=backbone_output.backbone_attention_mask,
        )
        add_tensor(tensors, f"denoise.step_{step}.dit_output", dit_output)
        decoded = model.action_head.action_decoder(dit_output, action_input.embodiment_id)
        velocity = decoded[:, -model.config.action_horizon :]
        add_tensor(tensors, f"denoise.step_{step}.decoded", decoded)
        add_tensor(tensors, f"denoise.step_{step}.velocity", velocity)
        actions = actions + dt * velocity
        add_tensor(tensors, f"denoise.step_{step}.action_out", actions)

    add_tensor(tensors, "result.normalized_action", actions)
    return actions


def main() -> None:
    args = parse_args()
    source = args.gr00t_source.resolve()
    checkpoint = args.checkpoint.resolve()
    # Keep the user-visible path instead of resolving symlinks. Upstream selects the
    # backbone by matching the literal "nvidia/Cosmos-Reason2" substring in model_name.
    cosmos = args.cosmos.absolute()
    output = args.output.resolve()
    sys.path.insert(0, str(source))

    import gr00t.model  # noqa: F401
    from gr00t.data.embodiment_tags import EmbodimentTag
    from transformers import AutoConfig, AutoModel, AutoProcessor

    embodiment = EmbodimentTag.resolve(EMBODIMENT_NAME)
    loading_kwargs = {"local_files_only": True}
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        model_name=str(cosmos),
        transformers_loading_kwargs=loading_kwargs,
        local_files_only=True,
    )
    processor.eval()
    observation = make_observation(processor)
    tensors: dict[str, torch.Tensor] = {}
    processed = export_processor_tensors(processor, embodiment, observation, tensors)

    manifest: dict[str, Any] = {
        "format": "jetson-pi.gr00t-n1d7-reference.v1",
        "seed": args.seed,
        "embodiment_name": EMBODIMENT_NAME,
        "embodiment_value": EMBODIMENT_VALUE,
        "embodiment_id": processor.embodiment_id_mapping[EMBODIMENT_VALUE],
        "language": LANGUAGE,
        "checkpoint": str(checkpoint),
        "cosmos": str(cosmos),
        "checkpoint_config_sha256": sha256(checkpoint / "config.json"),
        "cosmos_config_sha256": sha256(cosmos / "config.json"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "device": args.device,
        "processor_only": args.processor_only,
    }

    handles = []
    if not args.processor_only:
        if not torch.cuda.is_available() or not args.device.startswith("cuda"):
            raise RuntimeError("Full reference export requires a CUDA device")
        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        config.model_name = str(cosmos)
        config.use_flash_attention = False
        config.backbone_trainable_params_fp32 = False
        model = AutoModel.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        model.eval().to(args.device)
        handles = install_hooks(model, tensors)
        normalized_action = export_model_tensors(model, processed, tensors, args.seed)
        state = {
            key.removeprefix("state."): value
            for key, value in observation.items()
            if key.startswith("state.")
        }
        decoded = processor.decode_action(
            normalized_action.float().cpu().numpy(), embodiment, state=state
        )
        for key, value in decoded.items():
            add_tensor(tensors, f"result.decoded_action.{key}", torch.from_numpy(value))
        manifest["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
        manifest["model_dtype"] = str(model.dtype)
        manifest["model_layers"] = {
            "vision": len(model.backbone.model.visual.blocks),
            "language": len(model.backbone.model.language_model.layers),
            "vl_self_attention": len(
                model.action_head.vl_self_attention.transformer_blocks
            ),
            "dit": len(model.action_head.model.transformer_blocks),
        }

    for handle in handles:
        handle.remove()

    output.mkdir(parents=True, exist_ok=True)
    tensor_path = output / "reference.safetensors"
    save_file(tensors, tensor_path)
    manifest["tensor_count"] = len(tensors)
    manifest["tensor_file"] = tensor_path.name
    manifest["tensor_file_sha256"] = sha256(tensor_path)
    manifest["tensors"] = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(tensors.items())
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(tensors)} tensors to {tensor_path}")
    print(f"SHA-256: {manifest['tensor_file_sha256']}")


if __name__ == "__main__":
    main()
