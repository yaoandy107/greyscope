"""Analyze same-row adversarial RAID predictions with source-grouped uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_IDS = [
    "greyscope-v2-bf16",
    "greyscope-v1-bf16",
    "editlens-llama-3.2-3b",
    "meld",
    "desklib-v1.01",
    "binoculars",
]


def _metrics(rows: pd.DataFrame, model_id: str) -> dict[str, float]:
    labels = rows["label"].to_numpy(dtype=np.int8, copy=False)
    scores = rows[model_id].to_numpy(dtype=np.float64, copy=False)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("binary metrics require both positive and negative rows")

    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    rank_ends = np.cumsum(counts)
    average_ranks = rank_ends - (counts - 1) / 2
    positive_rank_sum = float(average_ranks[inverse][labels == 1].sum())
    auroc = (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)

    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    true_positives = np.cumsum(sorted_labels)[group_ends]
    false_positives = group_ends + 1 - true_positives
    eligible = false_positives / negatives <= 0.01
    tpr_at_1pct_fpr = float(
        true_positives[eligible].max() / positives if eligible.any() else 0.0
    )

    return {
        "auroc": float(auroc),
        "tpr_at_1pct_fpr": tpr_at_1pct_fpr,
    }


def _interval(values: list[float]) -> list[float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def analyze(
    sample_path: Path,
    predictions_dir: Path,
    *,
    model_ids: list[str],
    bootstrap_samples: int,
    seed: int,
) -> dict:
    required_pair = {"greyscope-v2-bf16", "greyscope-v1-bf16"}
    if not required_pair.issubset(model_ids):
        raise ValueError("models must include both Greyscope v2 and v1")
    rows = pd.read_csv(sample_path, keep_default_na=False)
    rows["label"] = rows["model"].ne("human").astype(int)
    for model_id in model_ids:
        predictions = pd.read_json(predictions_dir / f"{model_id}.jsonl", lines=True)
        predictions = predictions.rename(columns={"score": model_id})
        rows = rows.merge(predictions, on="id", validate="one_to_one")

    slices = {
        "overall": rows,
        "attacked": rows[rows["attack"].ne("none")],
    }
    slices.update({
        f"attack:{attack}": group
        for attack, group in rows.groupby("attack", sort=True)
    })
    slices.update({
        f"domain:{domain}": group
        for domain, group in rows.groupby("domain", sort=True)
    })
    point = {
        slice_id: {model_id: _metrics(group, model_id) for model_id in model_ids}
        for slice_id, group in slices.items()
    }

    rng = np.random.default_rng(seed)
    source_ids = np.asarray(sorted(rows["source_id"].unique()))
    source_groups = {
        source_id: rows.index[rows["source_id"].eq(source_id)].to_numpy()
        for source_id in source_ids
    }
    bootstrap = {
        slice_id: {
            model_id: {"auroc": [], "tpr_at_1pct_fpr": []}
            for model_id in model_ids
        }
        for slice_id in slices
    }
    differences = {
        slice_id: {"auroc": [], "tpr_at_1pct_fpr": []}
        for slice_id in slices
    }
    pairwise_differences = {
        slice_id: {
            model_id: {"auroc": [], "tpr_at_1pct_fpr": []}
            for model_id in model_ids
            if model_id != "greyscope-v2-bf16"
        }
        for slice_id in slices
    }
    for _ in range(bootstrap_samples):
        drawn_sources = rng.choice(source_ids, size=len(source_ids), replace=True)
        indices = np.concatenate([source_groups[source_id] for source_id in drawn_sources])
        replicate = rows.loc[indices]
        replicate_slices = {
            "overall": replicate,
            "attacked": replicate[replicate["attack"].ne("none")],
        }
        replicate_slices.update({
            f"attack:{attack}": group
            for attack, group in replicate.groupby("attack", sort=True)
        })
        replicate_slices.update({
            f"domain:{domain}": group
            for domain, group in replicate.groupby("domain", sort=True)
        })
        for slice_id, group in replicate_slices.items():
            values = {}
            for model_id in model_ids:
                values[model_id] = _metrics(group, model_id)
                for metric, value in values[model_id].items():
                    bootstrap[slice_id][model_id][metric].append(value)
            for metric in differences[slice_id]:
                differences[slice_id][metric].append(
                    values["greyscope-v2-bf16"][metric]
                    - values["greyscope-v1-bf16"][metric]
                )
            for model_id, metrics in pairwise_differences[slice_id].items():
                for metric in metrics:
                    metrics[metric].append(
                        values["greyscope-v2-bf16"][metric]
                        - values[model_id][metric]
                    )

    results = {}
    for slice_id in slices:
        results[slice_id] = {
            "rows": len(slices[slice_id]),
            "models": {},
            "greyscope_v2_minus_v1": {},
            "greyscope_v2_minus": {},
        }
        for model_id in model_ids:
            results[slice_id]["models"][model_id] = {
                metric: {
                    "estimate": point[slice_id][model_id][metric],
                    "interval_95": _interval(bootstrap[slice_id][model_id][metric]),
                }
                for metric in point[slice_id][model_id]
            }
        for metric, values in differences[slice_id].items():
            results[slice_id]["greyscope_v2_minus_v1"][metric] = {
                "estimate": (
                    point[slice_id]["greyscope-v2-bf16"][metric]
                    - point[slice_id]["greyscope-v1-bf16"][metric]
                ),
                "interval_95": _interval(values),
            }
        for model_id, metrics in pairwise_differences[slice_id].items():
            results[slice_id]["greyscope_v2_minus"][model_id] = {}
            for metric, values in metrics.items():
                results[slice_id]["greyscope_v2_minus"][model_id][metric] = {
                    "estimate": (
                        point[slice_id]["greyscope-v2-bf16"][metric]
                        - point[slice_id][model_id][metric]
                    ),
                    "interval_95": _interval(values),
                }

    return {
        "method": "source-document grouped percentile bootstrap",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "rows": len(rows),
        "source_documents": rows["source_id"].nunique(),
        "models": model_ids,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path)
    parser.add_argument("predictions_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--models", nargs="+", default=MODEL_IDS)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    result = analyze(
        args.sample,
        args.predictions_dir,
        model_ids=args.models,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
