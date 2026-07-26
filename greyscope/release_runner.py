"""Resumable prediction storage and metrics for frozen release snapshots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from greyscope.eval import roc_auc, tpr_at_fpr


def score_snapshot(
    rows: pd.DataFrame,
    score_fn: Callable[[list[str]], np.ndarray],
    output_path: str | Path,
    *,
    model: dict,
    snapshot: dict,
    chunk_size: int = 256,
    on_chunk: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Score missing rows in chunks and checkpoint every chunk as JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    expected_metadata = {
        "model_id": model["id"],
        "model_source": model["source"],
        "model_revision": model["revision"],
        "adapter": model["adapter"],
        "max_length": model["max_length"],
        "benchmark": snapshot["benchmark"],
        "snapshot_rows_sha256": snapshot["rows_sha256"],
        "snapshot_texts_sha256": snapshot["texts_sha256"],
        "snapshot_records_sha256": snapshot.get("records_sha256"),
        "base_source": model.get("base_source"),
        "base_revision": model.get("base_revision"),
    }
    if model["adapter"] == "greyscope":
        expected_metadata.update({
            "head_type": model["head_type"],
            "normalize_unicode": model["normalize_unicode"],
        })
    if metadata_path.is_file():
        existing_metadata = json.loads(metadata_path.read_text())
        if existing_metadata != expected_metadata:
            raise ValueError("prediction metadata differs from the requested model or snapshot")
    else:
        metadata_path.write_text(json.dumps(expected_metadata, indent=2) + "\n")

    completed = {}
    if output_path.is_file():
        for line in output_path.read_text().splitlines():
            record = json.loads(line)
            completed[record["row_id"]] = float(record["score"])

    missing = rows[~rows["row_id"].isin(completed)]
    for start in range(0, len(missing), chunk_size):
        chunk = missing.iloc[start : start + chunk_size]
        scores = np.asarray(score_fn(chunk["text"].astype(str).tolist()), dtype=float)
        if scores.shape != (len(chunk),):
            raise ValueError(f"score function returned {scores.shape}, expected {(len(chunk),)}")
        if not np.isfinite(scores).all():
            raise ValueError("score function returned non-finite values")
        with output_path.open("a") as handle:
            for row_id, score in zip(chunk["row_id"], scores):
                record = {"row_id": row_id, "score": float(score)}
                handle.write(json.dumps(record) + "\n")
                completed[row_id] = float(score)
        if on_chunk:
            on_chunk()
        print(f"[{model['id']}/{snapshot['benchmark']}] {len(completed)}/{len(rows)}", flush=True)

    unknown = set(completed) - set(rows["row_id"])
    missing_ids = set(rows["row_id"]) - set(completed)
    if unknown or missing_ids:
        raise ValueError(f"prediction coverage mismatch: unknown={len(unknown)} missing={len(missing_ids)}")
    return pd.DataFrame({
        "row_id": rows["row_id"],
        "score": [completed[row_id] for row_id in rows["row_id"]],
    })


def evaluate_release_scores(
    rows: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    ternary_thresholds: tuple[float, float] | None = None,
    binary_threshold: float | None = None,
) -> dict:
    """Threshold-free binary, edit-correlation, and provenance summaries."""
    merged = rows.merge(predictions, on="row_id", how="left", validate="one_to_one")
    if merged["score"].isna().any():
        raise ValueError("predictions do not cover every release row")
    scores = merged["score"].to_numpy(dtype=float)
    output: dict = {"n": len(merged)}

    def binary_metrics(frame: pd.DataFrame) -> dict | None:
        binary_rows = frame[frame["label_binary"].notna()]
        if binary_rows.empty:
            return None
        labels = binary_rows["label_binary"].astype(int).to_numpy()
        if len(np.unique(labels)) < 2:
            return None
        binary_scores = binary_rows["score"].to_numpy(dtype=float)
        result = {
            "auroc": roc_auc(labels, binary_scores),
            "tpr@fpr1": tpr_at_fpr(labels, binary_scores, 0.01),
            "tpr@fpr5": tpr_at_fpr(labels, binary_scores, 0.05),
            "n": len(binary_rows),
        }
        if binary_threshold is not None:
            predicted = binary_scores > binary_threshold
            negatives = labels == 0
            positives = labels == 1
            result["at_shipped_threshold"] = {
                "threshold": float(binary_threshold),
                "fpr": float(predicted[negatives].mean()),
                "tpr": float(predicted[positives].mean()),
            }
        return result

    binary = binary_metrics(merged)
    if binary:
        output["binary"] = binary

    ternary_labels = merged.get("label_ternary", pd.Series(np.nan, index=merged.index))
    ternary_mask = ternary_labels.notna().to_numpy()
    if ternary_thresholds is not None and ternary_mask.any():
        labels = merged.loc[ternary_mask, "label_ternary"].astype(int).to_numpy()
        if set(labels) == {0, 1, 2}:
            from sklearn.metrics import f1_score

            h_threshold, ai_threshold = ternary_thresholds
            ternary_scores = scores[ternary_mask]
            predicted = np.where(
                ternary_scores < h_threshold,
                0,
                np.where(ternary_scores > ai_threshold, 1, 2),
            )
            output["ternary"] = {
                "macro_f1": float(f1_score(labels, predicted, average="macro")),
                "f1_human": float(f1_score(labels, predicted, labels=[0], average="macro")),
                "f1_generated": float(f1_score(labels, predicted, labels=[1], average="macro")),
                "f1_edited": float(f1_score(labels, predicted, labels=[2], average="macro")),
                "h_threshold": float(h_threshold),
                "ai_threshold": float(ai_threshold),
                "n": int(ternary_mask.sum()),
            }

    target_mask = merged["edit_target"].notna().to_numpy()
    if target_mask.sum() >= 2:
        from scipy.stats import pearsonr, spearmanr

        target = merged.loc[target_mask, "edit_target"].to_numpy(dtype=float)
        target_scores = scores[target_mask]
        output["edit_correlation"] = {
            "pearson": float(pearsonr(target, target_scores).statistic),
            "spearman": float(spearmanr(target, target_scores).statistic),
            "n": int(target_mask.sum()),
        }

    output["by_variant"] = {
        str(variant): {
            "mean": float(group["score"].mean()),
            "median": float(group["score"].median()),
            "n": len(group),
        }
        for variant, group in merged.groupby("variant", sort=True)
    }
    if "domain" in merged:
        output["by_domain"] = {
            str(domain): metrics
            for domain, group in merged.groupby("domain", dropna=False, sort=True)
            if (metrics := binary_metrics(group)) is not None
        }
    return output
