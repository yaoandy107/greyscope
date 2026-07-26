#!/usr/bin/env python3
"""Add reproducible uncertainty estimates to downloaded release results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from greyscope.release_stats import bootstrap_intervals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument("--unit", choices=["row", "source"], required=True)
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()

    data_path = Path("data/release") / f"{args.snapshot}.csv"
    results_dir = Path("benchmarks/results/release") / args.snapshot
    rows = pd.read_csv(data_path)
    for report_path in sorted(results_dir.glob("*-metrics.json")):
        if "-smoke-" in report_path.name:
            continue
        model_id = report_path.name.removesuffix("-metrics.json")
        prediction_path = results_dir / f"{model_id}.jsonl"
        if not prediction_path.is_file():
            continue
        report = json.loads(report_path.read_text())
        metrics = report["metrics"]
        ternary = metrics.get("ternary")
        binary = metrics.get("binary", {}).get("at_shipped_threshold")
        predictions = pd.read_json(prediction_path, lines=True)
        uncertainty = bootstrap_intervals(
            rows,
            predictions,
            unit=args.unit,
            samples=args.samples,
            ternary_thresholds=(ternary["h_threshold"], ternary["ai_threshold"])
            if ternary
            else None,
            binary_threshold=binary["threshold"] if binary else None,
        )
        output_path = results_dir / f"{model_id}-uncertainty.json"
        output_path.write_text(json.dumps(uncertainty, indent=2) + "\n")
        print(output_path)


if __name__ == "__main__":
    main()
