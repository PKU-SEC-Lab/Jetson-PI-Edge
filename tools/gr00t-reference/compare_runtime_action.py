#!/usr/bin/env python3
"""Compare a GGML GR00T action dump with the prepared official reference."""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()
    expected = np.fromfile(args.expected, dtype=np.float32)
    actual = np.fromfile(args.actual, dtype=np.float32)
    if expected.shape != actual.shape or expected.size == 0:
        raise SystemExit(f"shape mismatch: expected={expected.shape}, actual={actual.shape}")
    difference = actual - expected
    cosine = np.dot(actual, expected) / (
        np.linalg.norm(actual) * np.linalg.norm(expected)
    )
    print(f"elements: {actual.size}")
    print(f"all_finite: {bool(np.isfinite(actual).all())}")
    print(f"mae: {np.abs(difference).mean():.9g}")
    print(f"rmse: {np.sqrt(np.mean(difference * difference)):.9g}")
    print(f"max_abs: {np.abs(difference).max():.9g}")
    print(f"cosine: {cosine:.9g}")


if __name__ == "__main__":
    main()
