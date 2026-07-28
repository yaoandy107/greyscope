#!/usr/bin/env python3
"""Compare MLX Q4 with saved bf16 predictions on matched external subsets."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from greyscope.corn import corn_scalar_score
from greyscope.preprocess import clean_text
from greyscope.release_data import RELEASE_COLUMNS, snapshot_metadata, validate_release_rows
from greyscope.release_runner import evaluate_release_scores, score_snapshot
from greyscope.release_stats import (
    bootstrap_intervals,
    paired_bootstrap_differences,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "release"
RESULTS_DIR = ROOT / "benchmarks" / "results" / "mac_mlx_external_2026-07-27"


def _proportional_sample(frame: pd.DataFrame, n: int, strata: list[str]) -> pd.DataFrame:
    """Deterministic proportional allocation using stable row IDs."""
    if n >= len(frame):
        return frame.copy()
    keyed = frame.copy()
    keyed["_stratum"] = keyed[strata].fillna("<null>").astype(str).agg("\0".join, axis=1)
    sizes = keyed.groupby("_stratum").size().sort_index()
    exact = sizes * (n / len(keyed))
    quotas = exact.astype(int)
    remainder = n - int(quotas.sum())
    order = (exact - quotas).sort_values(ascending=False, kind="stable").index
    for name in order[:remainder]:
        quotas[name] += 1
    pieces = [
        group.sort_values("row_id").head(int(quotas[name]))
        for name, group in keyed.groupby("_stratum", sort=True)
        if quotas[name]
    ]
    return pd.concat(pieces, ignore_index=True).drop(columns="_stratum")


def _document_sample(
    frame: pd.DataFrame, *, n_documents: int, strata: list[str]
) -> pd.DataFrame:
    documents = frame.sort_values("row_id").drop_duplicates("source_id")
    selected = _proportional_sample(documents, n_documents, strata)["source_id"]
    return frame[frame["source_id"].isin(selected)].copy()


def _normalize_raid(frame: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame({
        "row_id": frame["id"].astype(str),
        "text_sha256": frame["generation"].fillna("").astype(str).map(
            lambda text: hashlib.sha256(text.encode()).hexdigest()
        ),
        "benchmark": "raid",
        "source_id": frame["source_id"].astype(str),
        "variant": frame["attack"].astype(str),
        "text": frame["generation"].fillna("").astype(str),
        "language": frame["domain"].astype(str),
        "label_binary": frame["model"].ne("human").astype(int),
        "label_ternary": np.nan,
        "edit_target": np.nan,
        "domain": frame["domain"].astype(str),
        "generator": frame["model"].where(frame["model"].ne("human")),
    })
    return validate_release_rows(rows[RELEASE_COLUMNS])


def select_subset(benchmark: str) -> tuple[pd.DataFrame, dict]:
    """Return the frozen matched subset and its sampling description."""
    if benchmark == "apt-eval":
        parent = pd.read_csv(DATA_DIR / "apt-eval-sample.csv")
        human = _proportional_sample(
            parent[parent["variant"].eq("human")], 100, ["domain"]
        )
        polished = _proportional_sample(
            parent[~parent["variant"].eq("human")],
            900,
            ["domain", "generator", "variant"],
        )
        rows = validate_release_rows(pd.concat([human, polished], ignore_index=True))
        sampling = {
            "method": "100 proportional human rows plus 900 proportional polished rows",
            "parent": "apt-eval-sample",
            "parent_n": len(parent),
        }
    elif benchmark == "beemo":
        parent = pd.read_csv(DATA_DIR / "beemo-sample.csv")
        rows = validate_release_rows(
            _document_sample(parent, n_documents=111, strata=["domain"])
        )
        sampling = {
            "method": "111 proportional source documents with all nine variants",
            "parent": "beemo-sample",
            "parent_n": len(parent),
        }
    elif benchmark == "raid-extra":
        parent = _normalize_raid(pd.read_csv(DATA_DIR / "raid-adversarial-4968.csv"))
        rows = validate_release_rows(
            _document_sample(parent, n_documents=42, strata=["domain"])
        )
        sampling = {
            "method": "42 proportional source groups with clean text and all 11 attacks",
            "parent": "raid-adversarial-4968",
            "parent_n": len(parent),
        }
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")
    return rows, sampling


def _parent_manifest(benchmark: str) -> dict:
    name = {
        "apt-eval": "apt-eval-sample",
        "beemo": "beemo-sample",
        "raid-extra": "raid-adversarial-4968",
    }[benchmark]
    return json.loads((ROOT / "benchmarks" / "manifests" / f"{name}.json").read_text())


def _bf16_predictions(benchmark: str, rows: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    directory = ROOT / "benchmarks" / "results" / "release"
    path = {
        "apt-eval": directory / "apt-eval-sample" / "greyscope-v2-bf16.jsonl",
        "beemo": directory / "beemo-sample" / "greyscope-v2-bf16.jsonl",
        "raid-extra": directory / "raid-adversarial-4968" / "greyscope-v2-bf16.jsonl",
    }[benchmark]
    predictions = pd.read_json(path, lines=True)
    if "id" in predictions:
        predictions = predictions.rename(columns={"id": "row_id"})
    predictions["row_id"] = predictions["row_id"].astype(str)
    selected = predictions[predictions["row_id"].isin(rows["row_id"])].copy()
    if len(selected) != len(rows):
        raise ValueError(
            f"bf16 coverage mismatch for {benchmark}: {len(selected)}/{len(rows)}"
        )
    return selected[["row_id", "score"]], path


def _load_mlx_scorer(model_path: Path):
    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(str(model_path), lazy=False)
    calibration = json.loads((model_path / "calibration.json").read_text())

    def score(texts: list[str]) -> np.ndarray:
        logits = []
        for text in texts:
            body = clean_text(str(text)) if calibration["lowercase"] else str(text)
            prompt = calibration["prompt_template"].format(text=body)
            tokens = tokenizer.encode(
                prompt, add_special_tokens=False
            )[: calibration["max_length"]]
            output = model(mx.array([tokens]))
            mx.eval(output)
            logits.append([float(value) for value in output[0].tolist()])
            del output
            mx.clear_cache()
        raw = corn_scalar_score(np.asarray(logits))
        oriented = -raw if calibration["flip"] else raw
        scaled = (oriented - calibration["score_min"]) / (
            calibration["score_max"] - calibration["score_min"]
        )
        return np.clip(scaled, 0.0, 1.0)

    return score, calibration


def run_benchmark(
    benchmark: str,
    *,
    model_path: Path,
    score_fn,
    calibration: dict,
    output_root: Path,
) -> dict:
    rows, sampling = select_subset(benchmark)
    parent = _parent_manifest(benchmark)
    snapshot = snapshot_metadata(
        rows,
        source=parent["source"],
        revision=parent.get("revision", parent.get("csv_sha256", "pinned")),
    )
    snapshot["sampling"] = sampling
    model_spec = {
        "id": "greyscope-v2-mlx-q4",
        "source": str(model_path),
        "revision": "production-r2-mlx-q4-g64",
        "adapter": "greyscope",
        "max_length": calibration["max_length"],
        "head_type": calibration["head_type"],
        "normalize_unicode": True,
    }
    output_dir = output_root / benchmark
    prediction_path = output_dir / "greyscope-v2-mlx-q4.jsonl"
    started = time.perf_counter()
    q4_predictions = score_snapshot(
        rows,
        score_fn,
        prediction_path,
        model=model_spec,
        snapshot=snapshot,
        chunk_size=8,
    )
    elapsed = time.perf_counter() - started
    bf16_predictions, bf16_path = _bf16_predictions(benchmark, rows)
    thresholds = (calibration["h_thresh"], calibration["ai_thresh"])
    q4_metrics = evaluate_release_scores(
        rows,
        q4_predictions,
        ternary_thresholds=thresholds,
        binary_threshold=calibration["binary_threshold"],
    )
    bf16_metrics = evaluate_release_scores(
        rows,
        bf16_predictions,
        ternary_thresholds=thresholds,
        binary_threshold=calibration["binary_threshold"],
    )
    unit = "row" if benchmark == "apt-eval" else "source"
    q4_uncertainty = bootstrap_intervals(
        rows,
        q4_predictions,
        unit=unit,
        samples=500,
        ternary_thresholds=thresholds,
        binary_threshold=calibration["binary_threshold"],
    )
    difference = paired_bootstrap_differences(
        rows,
        q4_predictions,
        bf16_predictions,
        unit=unit,
        samples=500,
    )
    report = {
        "benchmark": benchmark,
        "model": model_spec,
        "snapshot": snapshot,
        "bf16_predictions": str(bf16_path.relative_to(ROOT)),
        "elapsed_seconds": elapsed,
        "rows_per_second": len(rows) / max(elapsed, 1e-9),
        "metrics": {"mlx_q4": q4_metrics, "bf16": bf16_metrics},
        "mlx_q4_uncertainty": q4_uncertainty,
        "mlx_q4_minus_bf16": difference,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "outputs" / "export_production-r2" / "mlx-4bit",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=("apt-eval", "beemo", "raid-extra"),
        default=("apt-eval", "beemo", "raid-extra"),
    )
    parser.add_argument("--output-root", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    score_fn, calibration = _load_mlx_scorer(args.model)
    for benchmark in args.benchmarks:
        run_benchmark(
            benchmark,
            model_path=args.model,
            score_fn=score_fn,
            calibration=calibration,
            output_root=args.output_root,
        )


if __name__ == "__main__":
    main()
