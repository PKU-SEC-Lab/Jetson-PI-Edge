#!/usr/bin/env python3
"""Standalone PI model detector for GGUF LLM/mmproj files."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any, BinaryIO

GGUF_STRING = 8
GGUF_ARRAY = 9

SCALAR_FORMATS = {
    0: "B",   # uint8
    1: "b",   # int8
    2: "H",   # uint16
    3: "h",   # int16
    4: "I",   # uint32
    5: "i",   # int32
    6: "f",   # float32
    7: "?",   # bool
    10: "Q",  # uint64
    11: "q",  # int64
    12: "d",  # float64
}

INTERESTING_META = {
    "general.architecture",
    "general.name",
    "general.basename",
    "general.finetune",
    "clip.projector_type",
    "clip.use_gelu",
    "clip.has_vision_encoder",
    "clip.has_llava_projector",
}


def read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise EOFError("unexpected EOF while reading GGUF")
    return data


def read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(f, 4))[0]


def read_u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(f, 8))[0]


def read_string(f: BinaryIO) -> str:
    n = read_u64(f)
    return read_exact(f, n).decode("utf-8", "replace")


def scalar_size(value_type: int) -> int | None:
    fmt = SCALAR_FORMATS.get(value_type)
    return struct.calcsize("<" + fmt) if fmt else None


def read_scalar(f: BinaryIO, value_type: int) -> Any:
    if value_type == GGUF_STRING:
        return read_string(f)
    fmt = SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise ValueError(f"unsupported GGUF scalar type {value_type}")
    return struct.unpack("<" + fmt, read_exact(f, struct.calcsize("<" + fmt)))[0]


def skip_scalar(f: BinaryIO, value_type: int) -> None:
    if value_type == GGUF_STRING:
        f.seek(read_u64(f), 1)
        return
    n = scalar_size(value_type)
    if n is None:
        raise ValueError(f"unsupported GGUF scalar type {value_type}")
    f.seek(n, 1)


def read_or_skip_value(f: BinaryIO, value_type: int, keep: bool) -> Any:
    if value_type == GGUF_ARRAY:
        item_type = read_u32(f)
        count = read_u64(f)
        if keep and count <= 64:
            return [read_scalar(f, item_type) for _ in range(count)]
        for _ in range(count):
            skip_scalar(f, item_type)
        return f"<array type={item_type} count={count}>"

    if keep:
        return read_scalar(f, value_type)
    skip_scalar(f, value_type)
    return None


def read_gguf_header(path: Path) -> tuple[dict[str, Any], list[str]]:
    metadata: dict[str, Any] = {}
    tensor_names: list[str] = []

    with path.open("rb") as f:
        if read_exact(f, 4) != b"GGUF":
            raise ValueError("not a GGUF file")
        metadata["GGUF.version"] = read_u32(f)
        tensor_count = read_u64(f)
        kv_count = read_u64(f)
        metadata["GGUF.tensor_count"] = tensor_count
        metadata["GGUF.kv_count"] = kv_count

        for _ in range(kv_count):
            key = read_string(f)
            value_type = read_u32(f)
            keep = key in INTERESTING_META
            value = read_or_skip_value(f, value_type, keep)
            if keep:
                metadata[key] = value

        for _ in range(tensor_count):
            name = read_string(f)
            n_dims = read_u32(f)
            f.seek(8 * n_dims, 1)  # dimensions
            f.seek(4, 1)           # tensor type
            f.seek(8, 1)           # data offset
            tensor_names.append(name)

    return metadata, tensor_names


def lower(value: Any) -> str:
    return str(value or "").lower()


def is_pi_llm_architecture(arch: str) -> bool:
    return arch in {"pi0", "pi05", "pi0.5"}


def detect_pi_model(metadata: dict[str, Any], tensor_names: list[str]) -> tuple[str, str, str]:
    names = set(tensor_names)
    arch = lower(metadata.get("general.architecture"))
    general_name = lower(metadata.get("general.name"))
    is_mmproj = arch == "clip" or "clip.has_vision_encoder" in metadata

    if is_mmproj:
        if "pi05" in general_name or "pi0.5" in general_name:
            return "pi05", "mmproj", "general.name contains pi05"
        if metadata.get("clip.use_gelu") is False:
            return "pi05", "mmproj", "clip.use_gelu is false"
        if general_name == "vit" and metadata.get("clip.use_gelu") is True:
            return "pi0", "mmproj", "general.name=vit and clip.use_gelu is true"
        return "unknown", "mmproj", "no strong mmproj feature matched"

    if not is_pi_llm_architecture(arch):
        reason = (
            "missing general.architecture; not probing llm tensor names"
            if not arch
            else "general.architecture is not a pi llm; not probing llm tensor names"
        )
        return "unknown", "llm", reason

    if "output_norm_dense.weight" in names:
        return "pi05", "llm", "tensor output_norm_dense.weight exists"
    if any(".attn_norm_dense." in n or ".ffn_norm_dense." in n for n in names):
        return "pi05", "llm", "dense norm tensors exist"
    if "time_mlp_in.weight" in names or "time_mlp_out.weight" in names:
        return "pi05", "llm", "time_mlp_in/out tensors exist"

    if "state_proj.weight" in names:
        return "pi0", "llm", "tensor state_proj.weight exists"
    if "action_time_mlp_in.weight" in names or "action_time_mlp_out.weight" in names:
        return "pi0", "llm", "action_time_mlp_in/out tensors exist"
    if "pi05" in general_name or "pi0.5" in general_name:
        return "pi05", "llm", "pi llm general.name contains pi05 fallback"
    return "unknown", "llm", "no strong pi llm tensor feature matched"


def detect_file(path: Path) -> dict[str, Any]:
    metadata, tensor_names = read_gguf_header(path)
    model, file_kind, reason = detect_pi_model(metadata, tensor_names)
    return {
        "path": str(path),
        "file_kind": file_kind,
        "pi_model": model,
        "reason": reason,
        "metadata": metadata,
        "tensor_count_read": len(tensor_names),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect pi0/pi05 from GGUF metadata/tensors")
    parser.add_argument("paths", nargs="+", help="GGUF file path(s)")
    parser.add_argument("--json", action="store_true", help="print JSON output")
    args = parser.parse_args()

    results = [detect_file(Path(p)) for p in args.paths]
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0

    for result in results:
        print(f"path: {result['path']}")
        print(f"file_kind: {result['file_kind']}")
        print(f"pi_model: {result['pi_model']}")
        print(f"reason: {result['reason']}")
        print(f"general.name: {result['metadata'].get('general.name', '')}")
        print(f"general.architecture: {result['metadata'].get('general.architecture', '')}")
        if result["file_kind"] == "mmproj":
            print(f"clip.use_gelu: {result['metadata'].get('clip.use_gelu', '')}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
