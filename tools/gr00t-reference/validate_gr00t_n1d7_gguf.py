#!/usr/bin/env python3
"""Validate the structural fidelity of GR00T N1.7 GGUF conversion outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gguf import GGUFReader
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    args = parse_args()
    model = GGUFReader(args.model)
    mmproj = GGUFReader(args.mmproj)
    model_tensors = {tensor.name: tensor for tensor in model.tensors}
    mmproj_tensors = {tensor.name: tensor for tensor in mmproj.tensors}

    require(model.fields["general.architecture"].contents() == "gr00t-n1d7", "bad arch")
    require(len(model_tensors) == 715, f"expected 715 main tensors, got {len(model_tensors)}")
    require(len(mmproj_tensors) == 316, f"expected 316 mmproj tensors, got {len(mmproj_tensors)}")

    expected_metadata = {
        "gr00t-n1d7.state_dimension": 132,
        "gr00t-n1d7.action_dimension": 132,
        "gr00t-n1d7.action_horizon": 40,
        "gr00t-n1d7.inference_steps": 4,
        "gr00t-n1d7.timestep_buckets": 1000,
        "gr00t-n1d7.embodiment_count": 32,
        "gr00t-n1d7.backbone_layer_count": 16,
        "gr00t-n1d7.dit_block_count": 32,
        "gr00t-n1d7.dit_head_count": 32,
        "gr00t-n1d7.dit_head_dimension": 48,
        "gr00t-n1d7.dit_output_dimension": 1024,
        "gr00t-n1d7.vl_self_attention_block_count": 4,
    }
    for key, expected in expected_metadata.items():
        actual = model.fields[key].contents()
        require(actual == expected, f"{key}: expected {expected}, got {actual}")

    processor = json.loads(model.fields["gr00t-n1d7.processor_config"].contents())
    statistics = json.loads(model.fields["gr00t-n1d7.statistics"].contents())
    embodiment_ids = json.loads(model.fields["gr00t-n1d7.embodiment_ids"].contents())
    require(len(processor["processor_kwargs"]["modality_configs"]) == 8, "bad modality count")
    require(len(statistics) == 8, "bad statistics count")
    require(len(embodiment_ids) == 52, "bad embodiment name count")

    source_actions: dict[str, tuple[int, ...]] = {}
    for part in args.checkpoint.glob("model-*.safetensors"):
        with safe_open(part, framework="pt", device="cpu") as source:
            for name in source.keys():
                if name.startswith("action_head."):
                    source_actions[f"gr00t.{name}"] = tuple(source.get_slice(name).get_shape())
    converted_actions = {
        name for name in model_tensors if name.startswith("gr00t.action_head.")
    }
    require(set(source_actions) == converted_actions, "action tensor name set mismatch")
    for name, source_shape in source_actions.items():
        converted_shape = tuple(model_tensors[name].shape)
        require(
            converted_shape == tuple(reversed(source_shape)),
            f"{name}: source {source_shape}, GGUF {converted_shape}",
        )

    backbone_names = [name for name in model_tensors if not name.startswith("gr00t.")]
    require(len(backbone_names) == 178, f"expected 178 backbone tensors, got {len(backbone_names)}")
    block_ids = {
        int(name.split(".")[1]) for name in backbone_names if name.startswith("blk.")
    }
    require(block_ids == set(range(16)), f"unexpected backbone blocks: {sorted(block_ids)}")
    require("output.weight" not in model_tensors, "unused LM head must not be exported")

    required_vision = {
        "v.patch_embd.weight",
        "v.patch_embd.weight.1",
        "v.deepstack.5.fc1.weight",
        "v.deepstack.11.fc1.weight",
        "v.deepstack.17.fc1.weight",
        "mm.2.weight",
    }
    missing_vision = required_vision.difference(mmproj_tensors)
    require(not missing_vision, f"missing mmproj tensors: {sorted(missing_vision)}")

    print(f"OK: {args.model} ({len(model_tensors)} tensors)")
    print(f"OK: {args.mmproj} ({len(mmproj_tensors)} tensors)")
    print("OK: metadata, action names/shapes, 16 backbone blocks, and vision split")


if __name__ == "__main__":
    main()
