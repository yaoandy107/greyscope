#!/usr/bin/env python3
"""Run one pinned release model on an existing snapshot without cloud compute."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from greyscope.detectors import (
    make_editlens_llama_scorer,
    make_greyscope_scorer,
    make_transformers_scorer,
)
from greyscope.release_manifest import load_release_manifest, release_models_for
from greyscope.release_runner import evaluate_release_scores, score_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot")
    parser.add_argument("model_id")
    parser.add_argument("--benchmark")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    benchmark = args.benchmark or args.snapshot.removesuffix("-sample")
    manifest = load_release_manifest("configs/release_eval.json")
    models = {model["id"]: model for model in release_models_for(manifest, benchmark)}
    if args.model_id not in models:
        raise ValueError(f"{args.model_id} is not selected for {benchmark}")
    model = models[args.model_id]
    batch_size = args.batch_size or model.get("batch_size", 32)

    scorer_factory = {
        "greyscope": make_greyscope_scorer,
        "editlens-llama": make_editlens_llama_scorer,
    }.get(model["adapter"], make_transformers_scorer)

    rows = pd.read_csv(Path("data/release") / f"{args.snapshot}.csv")
    snapshot = json.loads(
        (Path("benchmarks/manifests") / f"{args.snapshot}.json").read_text()
    )
    output_dir = Path("benchmarks/results/release") / args.snapshot
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    score_fn, _tokenizer, loaded_model = scorer_factory(
        model,
        device=args.device,
        batch_size=batch_size,
    )
    load_seconds = time.perf_counter() - started
    predictions = score_snapshot(
        rows,
        score_fn,
        output_dir / f"{args.model_id}.jsonl",
        model=model,
        snapshot=snapshot,
        chunk_size=args.chunk_size,
    )
    elapsed = time.perf_counter() - started
    calibration = getattr(loaded_model, "greyscope_calibration", None)
    metrics = evaluate_release_scores(
        rows,
        predictions,
        ternary_thresholds=(calibration["h_thresh"], calibration["ai_thresh"])
        if calibration
        else None,
        binary_threshold=calibration["binary_threshold"] if calibration else None,
    )
    report = {
        "model": model,
        "snapshot": snapshot,
        "runtime": {
            "device": args.device,
            "batch_size": batch_size,
        },
        "load_seconds": load_seconds,
        "elapsed_seconds": elapsed,
        "rows_per_second": len(rows) / max(elapsed - load_seconds, 1e-9),
        "metrics": metrics,
    }
    report_path = output_dir / f"{args.model_id}-metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
