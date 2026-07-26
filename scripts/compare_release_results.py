#!/usr/bin/env python3
"""Compare two detectors on identical frozen rows with paired uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from greyscope.release_stats import paired_bootstrap_differences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("model_a")
    parser.add_argument("model_b")
    parser.add_argument("--unit", choices=["row", "source"], required=True)
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()

    results_dir = Path("benchmarks/results/release") / args.snapshot
    rows = pd.read_csv(Path("data/release") / f"{args.snapshot}.csv")
    predictions_a = pd.read_json(results_dir / f"{args.model_a}.jsonl", lines=True)
    predictions_b = pd.read_json(results_dir / f"{args.model_b}.jsonl", lines=True)
    result = paired_bootstrap_differences(
        rows,
        predictions_a,
        predictions_b,
        unit=args.unit,
        samples=args.samples,
    )
    result.update({"model_a": args.model_a, "model_b": args.model_b})
    output_path = results_dir / f"compare-{args.model_a}-vs-{args.model_b}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
