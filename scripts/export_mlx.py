#!/usr/bin/env python3
"""Convert the bf16 Greyscope checkpoint to a native MLX 4-bit classifier.

Run on Apple Silicon with the optional Mac dependencies installed:

    python scripts/export_mlx.py --output outputs/greyscope-v2-mlx-4bit
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

from greyscope.mlx_export import stage_model

BF16_MODEL = "yaoandy107/greyscope-v2-qwen3.5-4b"


def _source_path(source: str) -> Path:
    local = Path(source)
    if local.exists():
        return local.resolve()
    # A small worker count is slower for tiny files but avoids large-model Hub
    # downloads stalling while several shard requests compete on one connection.
    return Path(snapshot_download(repo_id=source, max_workers=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=BF16_MODEL, help="bf16 Hub ID or local model directory")
    parser.add_argument("--output", required=True, type=Path, help="new MLX model directory")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    from mlx_lm.convert import convert

    with tempfile.TemporaryDirectory(prefix="greyscope-mlx-stage-") as tmp:
        stage = Path(tmp)
        source_path = _source_path(args.source)
        stage_model(source_path, stage)
        convert(
            str(stage),
            str(args.output),
            quantize=True,
            q_bits=args.bits,
            q_group_size=args.group_size,
            dtype="bfloat16",
        )

    source_calibration = source_path / "calibration.json"
    shutil.copy2(source_calibration, args.output / "calibration.json")
    print(f"MLX artifact written to {args.output}")


if __name__ == "__main__":
    main()
