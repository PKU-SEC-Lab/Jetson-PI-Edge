#!/usr/bin/env python3
"""Prepare exact GR00T reference tensors for the GGML runtime validator."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


TENSORS = {
    "backbone_features.f32": ("backbone.backbone_features", np.float32),
    "state.f32": ("model_input.action.state", np.float32),
    "initial_noise.f32": ("denoise.initial_noise", np.float32),
    "image_mask.u8": ("backbone.image_mask", np.uint8),
    "expected_action.f32": ("result.normalized_action", np.float32),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = {"source": str(args.reference.resolve()), "files": {}}
    with safe_open(args.reference, framework="pt", device="cpu") as archive:
        for filename, (tensor_name, dtype) in TENSORS.items():
            if tensor_name not in archive.keys():
                continue
            tensor = archive.get_tensor(tensor_name)
            array = tensor.float().cpu().numpy().astype(dtype, copy=False)
            path = args.output / filename
            array.tofile(path)
            manifest["files"][filename] = {
                "tensor": tensor_name,
                "shape": list(tensor.shape),
                "dtype": str(dtype),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        language_input = "hook.language.input.kwargs.inputs_embeds"
        extra = {}
        if language_input in archive.keys():
            base = archive.get_tensor(language_input).float().cpu().numpy()[0]
            mask = archive.get_tensor(
                "hook.language.input.kwargs.visual_pos_masks"
            ).cpu().numpy()[0].astype(bool)
            slab = np.zeros((base.shape[0], base.shape[1] * 4), dtype=np.float32)
            slab[:, : base.shape[1]] = base
            for index in range(3):
                deepstack = archive.get_tensor(
                    f"hook.language.input.kwargs.deepstack_visual_embeds.{index}"
                ).float().cpu().numpy()
                slab[mask, (index + 1) * base.shape[1] : (index + 2) * base.shape[1]] = deepstack
            positions3 = archive.get_tensor(
                "hook.language.input.kwargs.position_ids"
            ).cpu().numpy()[:, 0]
            positions4 = np.zeros((4, base.shape[0]), dtype=np.int32)
            positions4[:3] = positions3.astype(np.int32)
            extra.update({
                "language_input.f32": slab,
                "language_positions.i32": positions4,
                "expected_backbone.f32": archive.get_tensor(
                    "backbone.backbone_features"
                ).float().cpu().numpy(),
            })
            for index in range(16):
                key = f"hook.language.layer_{index}.call_0"
                if key in archive.keys():
                    extra[f"expected_l_out-{index}.f32"] = (
                        archive.get_tensor(key).float().cpu().numpy()
                    )
            vision_main = "hook.vision.output.call_0.0"
            if vision_main in archive.keys():
                main = archive.get_tensor(vision_main).float().cpu().numpy()
                vision = np.zeros((main.shape[0], main.shape[1] * 4), dtype=np.float32)
                vision[:, : main.shape[1]] = main
                for index in range(3):
                    vision[:, (index + 1) * main.shape[1] : (index + 2) * main.shape[1]] = (
                        archive.get_tensor(
                            f"hook.vision.output.call_0.1.{index}"
                        ).float().cpu().numpy()
                    )
                extra["expected_vision_embeddings.f32"] = vision
        if "processor_input.images" in archive.keys():
            images = archive.get_tensor("processor_input.images").cpu().numpy()
            extra["processor_input_images.u8"] = images.astype(np.uint8, copy=False)
        pixel_key = "model_input.backbone.pixel_values"
        if pixel_key in archive.keys():
            pixels = archive.get_tensor(pixel_key).float().cpu().numpy().reshape(
                4, 448, 1536
            )
            inverse = np.argsort([0, 3, 6, 4, 7, 2, 1, 5, 8])
            raw = []
            for image in pixels:
                temporal = image.reshape(1, 8, 14, 2, 2, 3, 2, 16, 16)
                temporal = temporal.transpose(inverse).reshape(2, 3, 256, 448)
                raw.append(temporal[0])
            extra["expected_vision_input.f32"] = np.stack(raw).astype(np.float32)
        for filename, array in extra.items():
            path = args.output / filename
            array.tofile(path)
            manifest["files"][filename] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

    (args.output / "fixture.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output / "fixture.json")


if __name__ == "__main__":
    main()
