"""Confidence intervals for frozen release-benchmark predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from greyscope.release_runner import evaluate_release_scores


def _flatten_metrics(metrics: dict) -> dict[str, float]:
    shipped = metrics.get("binary", {}).get("at_shipped_threshold", {})
    selected = {
        "binary.auroc": metrics.get("binary", {}).get("auroc"),
        "binary.tpr@fpr1": metrics.get("binary", {}).get("tpr@fpr1"),
        "binary.tpr@fpr5": metrics.get("binary", {}).get("tpr@fpr5"),
        "binary.shipped_threshold_fpr": shipped.get("fpr"),
        "binary.shipped_threshold_tpr": shipped.get("tpr"),
        "ternary.macro_f1": metrics.get("ternary", {}).get("macro_f1"),
        "edit_correlation.pearson": metrics.get("edit_correlation", {}).get("pearson"),
        "edit_correlation.spearman": metrics.get("edit_correlation", {}).get("spearman"),
    }
    return {
        name: float(value)
        for name, value in selected.items()
        if value is not None and np.isfinite(value)
    }


def bootstrap_intervals(
    rows: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    unit: str,
    samples: int = 500,
    seed: int = 20260722,
    ternary_thresholds: tuple[float, float] | None = None,
    binary_threshold: float | None = None,
) -> dict:
    """Return deterministic percentile intervals using rows or source documents.

    Source bootstrap keeps every variant of a sampled document together. APT and
    NLPCC use row bootstrap because their public releases do not expose reliable
    parent-document links across variants.
    """
    if unit not in {"row", "source"}:
        raise ValueError("bootstrap unit must be 'row' or 'source'")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")

    merged = rows.merge(predictions, on="row_id", how="left", validate="one_to_one")
    if merged["score"].isna().any():
        raise ValueError("predictions do not cover every release row")

    rng = np.random.default_rng(seed)
    if unit == "source":
        source_ids = merged["source_id"].drop_duplicates().to_numpy()
        grouped_indices = {
            source_id: indices.to_numpy()
            for source_id, indices in merged.groupby("source_id", sort=False).groups.items()
        }

        def draw() -> pd.DataFrame:
            chosen = rng.choice(source_ids, size=len(source_ids), replace=True)
            indices = np.concatenate([grouped_indices[source_id] for source_id in chosen])
            return merged.iloc[indices].reset_index(drop=True)

    else:

        def draw() -> pd.DataFrame:
            indices = rng.integers(0, len(merged), size=len(merged))
            return merged.iloc[indices].reset_index(drop=True)

    values: dict[str, list[float]] = {}
    for _ in range(samples):
        sampled = draw().copy()
        sampled["row_id"] = np.arange(len(sampled)).astype(str)
        sampled_predictions = sampled[["row_id", "score"]].copy()
        sampled_rows = sampled.drop(columns="score")
        try:
            metrics = evaluate_release_scores(
                sampled_rows,
                sampled_predictions,
                ternary_thresholds=ternary_thresholds,
                binary_threshold=binary_threshold,
            )
        except ValueError:
            continue
        for name, value in _flatten_metrics(metrics).items():
            values.setdefault(name, []).append(value)

    point = _flatten_metrics(
        evaluate_release_scores(
            rows,
            predictions,
            ternary_thresholds=ternary_thresholds,
            binary_threshold=binary_threshold,
        )
    )
    intervals = {}
    for name, estimate in point.items():
        observed = np.asarray(values.get(name, []), dtype=float)
        if not len(observed):
            continue
        low, high = np.percentile(observed, [2.5, 97.5])
        intervals[name] = {
            "estimate": estimate,
            "low": float(low),
            "high": float(high),
            "successful_samples": int(len(observed)),
        }
    return {
        "method": "percentile bootstrap",
        "unit": unit,
        "samples": samples,
        "seed": seed,
        "confidence": 0.95,
        "metrics": intervals,
    }


def paired_bootstrap_differences(
    rows: pd.DataFrame,
    predictions_a: pd.DataFrame,
    predictions_b: pd.DataFrame,
    *,
    unit: str,
    samples: int = 500,
    seed: int = 20260722,
) -> dict:
    """Estimate metric A-minus-B intervals from identical bootstrap draws."""
    if unit not in {"row", "source"}:
        raise ValueError("bootstrap unit must be 'row' or 'source'")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    merged = rows.merge(
        predictions_a.rename(columns={"score": "score_a"}), on="row_id", validate="one_to_one"
    ).merge(
        predictions_b.rename(columns={"score": "score_b"}), on="row_id", validate="one_to_one"
    )
    if len(merged) != len(rows) or merged[["score_a", "score_b"]].isna().any().any():
        raise ValueError("both prediction sets must cover every release row")

    rng = np.random.default_rng(seed)
    if unit == "source":
        source_ids = merged["source_id"].drop_duplicates().to_numpy()
        groups = {
            source_id: indices.to_numpy()
            for source_id, indices in merged.groupby("source_id", sort=False).groups.items()
        }

        def draw_indices() -> np.ndarray:
            chosen = rng.choice(source_ids, size=len(source_ids), replace=True)
            return np.concatenate([groups[source_id] for source_id in chosen])

    else:

        def draw_indices() -> np.ndarray:
            return rng.integers(0, len(merged), size=len(merged))

    def metrics_for(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
        sampled = frame.copy()
        sampled["row_id"] = np.arange(len(sampled)).astype(str)
        predictions = sampled[["row_id", score_column]].rename(columns={score_column: "score"})
        return _flatten_metrics(
            evaluate_release_scores(sampled.drop(columns=["score_a", "score_b"]), predictions)
        )

    point_a = metrics_for(merged, "score_a")
    point_b = metrics_for(merged, "score_b")
    values: dict[str, list[float]] = {}
    for _ in range(samples):
        sampled = merged.iloc[draw_indices()].reset_index(drop=True)
        try:
            metrics_a = metrics_for(sampled, "score_a")
            metrics_b = metrics_for(sampled, "score_b")
        except ValueError:
            continue
        for name in metrics_a.keys() & metrics_b.keys():
            values.setdefault(name, []).append(metrics_a[name] - metrics_b[name])

    differences = {}
    for name in point_a.keys() & point_b.keys():
        observed = np.asarray(values.get(name, []), dtype=float)
        if not len(observed):
            continue
        low, high = np.percentile(observed, [2.5, 97.5])
        differences[name] = {
            "estimate": point_a[name] - point_b[name],
            "low": float(low),
            "high": float(high),
            "successful_samples": int(len(observed)),
        }
    return {
        "method": "paired percentile bootstrap",
        "direction": "model_a minus model_b",
        "unit": unit,
        "samples": samples,
        "seed": seed,
        "confidence": 0.95,
        "metrics": differences,
    }
