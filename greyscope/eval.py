"""Detector evaluation: ternary + binary metrics, calibrate-on-val protocol.

Ternary core (two thresholds on val, oriented so higher = more AI) is ported from
EditLens scripts/eval/. The binary + OOD reporting on top (AUROC, TPR@fixed-FPR,
val-frozen thresholds, per-detector benchmarking) is ours.

Source: https://github.com/pangramlabs/EditLens/blob/main/scripts/eval/
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


LABEL_TO_ID = {"human_written": 0, "ai_generated": 1, "ai_edited": 2}


def compute_scalar_score(bucket_logits: np.ndarray, n_buckets: int) -> np.ndarray:
    """Collapse [N, n_buckets] logits to a [N] scalar in [0, 1] (higher = more AI):
    softmax, then expected bucket index normalized by (n_buckets - 1). Matches
    EditLens's decode."""
    from scipy.special import softmax

    probs = softmax(bucket_logits, axis=1)
    bucket_index = np.arange(n_buckets, dtype=np.float32)
    return (probs @ bucket_index) / (n_buckets - 1)


def find_optimal_threshold(
    preds: np.ndarray, labels: np.ndarray, num_thresholds: int = 1000
) -> tuple[float, float]:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    thresholds = np.linspace(0, 1, num_thresholds)

    best_threshold = 0.0
    best_f1 = 0.0

    for threshold in thresholds:
        pred_labels = (preds >= threshold).astype(int)
        tp = np.sum((pred_labels == 1) & (labels == 1))
        fp = np.sum((pred_labels == 1) & (labels == 0))
        fn = np.sum((pred_labels == 0) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def minmax_scale(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def orient_scores(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, bool]:
    """Ensure higher score = more AI. Returns (oriented_scores, was_flipped)."""
    human_mean = scores[labels == 0].mean()
    ai_mean = scores[labels == 1].mean()
    if human_mean > ai_mean:
        return -scores, True
    return scores, False


def calibrate_thresholds(
    labels: np.ndarray, scaled_scores: np.ndarray
) -> tuple[float, float, float, float]:
    """Find two thresholds on val. Assumes higher score = more AI.

    Returns (human_thresh, ai_thresh, f1_human, f1_ai).
    """
    binary_human = (labels > 0).astype(int)
    h_thresh, h_f1 = find_optimal_threshold(scaled_scores, binary_human)

    binary_ai = (labels == 1).astype(int)
    ai_thresh, ai_f1 = find_optimal_threshold(scaled_scores, binary_ai)

    return h_thresh, ai_thresh, h_f1, ai_f1


def predict_ternary(
    scaled_scores: np.ndarray, h_thresh: float, ai_thresh: float
) -> np.ndarray:
    """Assign ternary labels based on two thresholds. Assumes higher score = more AI."""
    preds = np.full(len(scaled_scores), 2, dtype=int)
    preds[scaled_scores < h_thresh] = 0
    preds[scaled_scores > ai_thresh] = 1
    return preds


def evaluate(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    per_class = f1_score(true_labels, pred_labels, average=None, labels=[0, 1, 2])
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_human": per_class[0],
        "f1_ai_generated": per_class[1],
        "f1_ai_edited": per_class[2],
        "confusion_matrix": cm,
    }


def _maybe_float(x) -> float | None:
    return float(x) if x is not None else None


def binary_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(row_mask, y01) for human-vs-AI binary detection.

    EditLens splits (have `text_type`): keep human_written + ai_generated and drop
    ai_edited, matching EditLens's binary composition; y=1 for ai_generated.
    Third-party splits: use the integer `label` column (0=human, 1=AI), all rows.
    """
    if "text_type" in df.columns:
        mask = df["text_type"].isin(["human_written", "ai_generated"]).to_numpy()
        y = (df["text_type"].to_numpy()[mask] == "ai_generated").astype(int)
        return mask, y
    y = df["label"].astype(int).to_numpy()
    return np.ones(len(df), dtype=bool), y


def _orient_binary(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, bool]:
    """Flip so higher = more AI, using the val human(0) vs AI(1) means. Single-class
    inputs can't orient themselves → caller orients on val and reuses the flag."""
    if (y == 0).sum() == 0 or (y == 1).sum() == 0:
        return scores, False
    if scores[y == 0].mean() > scores[y == 1].mean():
        return -scores, True
    return scores, False


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Binary macro-F1 / FPR / FNR. macro_f1 is None when only one class is present."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    fnr = fn / (fn + tp) if (fn + tp) > 0 else None
    macro = f1_score(y_true, y_pred, average="macro") if len(set(y_true.tolist())) > 1 else None
    return {"macro_f1": _maybe_float(macro), "fpr": _maybe_float(fpr),
            "fnr": _maybe_float(fnr), "n": int(len(y_true))}


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Threshold-free separability (higher score = more AI). None if one class only."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def threshold_for_fpr(human_scores: np.ndarray, target_fpr: float) -> float:
    """Score threshold whose false-positive rate on `human_scores` equals `target_fpr`."""
    return float(np.quantile(human_scores, 1.0 - target_fpr))


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float | None:
    """Detection rate (TPR) at a threshold calibrated to `target_fpr` on this set's own
    human rows, RAID's reporting protocol. None unless both classes are present."""
    if (y_true == 0).sum() == 0 or (y_true == 1).sum() == 0:
        return None
    thr = threshold_for_fpr(scores[y_true == 0], target_fpr)
    return float((scores[y_true == 1] > thr).mean())


def detection_from_scalar(scores: np.ndarray, ternary_labels: np.ndarray,
                          target_fprs: tuple[float, ...] = (0.01, 0.05)) -> dict:
    """Human(0) vs ANY-AI(>0) detection from an oriented scalar (higher = more AI) — the product
    boundary "is this AI?" asks, and what the CORN ranking loss optimizes. Threshold-free AUROC +
    TPR at each self-calibrated FPR (RAID's protocol). `ternary_labels` are LABEL_TO_ID ints
    (human=0, generated/edited>0); None on a single-class slice."""
    y = (np.asarray(ternary_labels) > 0).astype(int)
    out: dict = {"auroc": roc_auc(y, scores), "n": int(len(y))}
    for fpr in target_fprs:
        out[f"tpr@fpr{int(round(fpr * 100))}"] = tpr_at_fpr(y, scores, fpr)
    return out


def conformal_threshold_for_fpr(human_scores: np.ndarray, target_fpr: float) -> float:
    """Split-conformal score threshold with *finite-sample* false-positive control.

    Uses the ceil((n+1)(1-fpr))-th order statistic, so at most `target_fpr` of future human
    texts are expected to exceed it — a distribution-free guarantee, unlike the plain
    empirical quantile (`threshold_for_fpr`), which it meets or exceeds. Returns +inf when
    there are too few humans to certify the rate (n < 1/fpr - 1), i.e. flag nothing rather
    than over-promise.
    """
    s = np.sort(np.asarray(human_scores, dtype=float))
    n = len(s)
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * (1.0 - target_fpr)))  # 1-based rank of the threshold value
    if rank > n:
        return float("inf")  # not enough calibration humans to certify this FPR
    return float(s[rank - 1])


def grouped_conformal_threshold(human_scores: np.ndarray, groups, target_fpr: float) -> float:
    """Max conformal threshold across human subgroups, so the FPR target holds for the
    *hardest* group (non-native writers / each language) — not just the easy majority.
    Pooling would let the in-domain majority dominate the quantile and inflate the rate
    on the vulnerable group (v1's false-accusation failure mode)."""
    scores = np.asarray(human_scores, dtype=float)
    groups = np.asarray(groups)
    thresholds = [conformal_threshold_for_fpr(scores[groups == g], target_fpr)
                  for g in np.unique(groups)]
    return float(max(thresholds)) if thresholds else float("inf")


def eval_detector_ternary(val_df: pd.DataFrame, split_df: pd.DataFrame, col: str) -> dict:
    """Ternary metrics for one detector `col`: calibrate two thresholds on val
    (per-split min-max), apply to the split."""
    vy = val_df["text_type"].map(LABEL_TO_ID).to_numpy()
    vs, flipped = orient_scores(val_df[col].to_numpy(dtype=float), vy)
    h, ai, _, _ = calibrate_thresholds(vy, minmax_scale(vs))
    ty = split_df["text_type"].map(LABEL_TO_ID).to_numpy()
    ts = split_df[col].to_numpy(dtype=float)
    preds = predict_ternary(minmax_scale(-ts if flipped else ts), h, ai)
    m = evaluate(ty, preds)
    return {"accuracy": _maybe_float(m["accuracy"]), "macro_f1": _maybe_float(m["macro_f1"]),
            "f1_human": _maybe_float(m["f1_human"]), "f1_ai_generated": _maybe_float(m["f1_ai_generated"]),
            "f1_ai_edited": _maybe_float(m["f1_ai_edited"]), "n": int(len(ty))}


def eval_detector_binary(val_df: pd.DataFrame, split_df: pd.DataFrame, col: str) -> dict:
    """Binary metrics for one detector `col`: fit orientation, min-max range, and a
    human-vs-AI threshold on val, then apply them to the split. Val-based (not
    per-split) scaling keeps single-class and OOD splits on the calibration scale."""
    vmask, vy = binary_labels(val_df)
    vs, flipped = _orient_binary(val_df.loc[vmask, col].to_numpy(dtype=float), vy)
    lo, hi = float(vs.min()), float(vs.max())
    span = (hi - lo) or 1.0
    thr, _ = find_optimal_threshold((vs - lo) / span, vy)

    smask, sy = binary_labels(split_df)
    ss = split_df.loc[smask, col].to_numpy(dtype=float)
    ss = -ss if flipped else ss
    pred = (np.clip((ss - lo) / span, 0.0, 1.0) >= thr).astype(int)
    rec = evaluate_binary(sy, pred)
    # AUROC and TPR@FPR are scale-invariant, so they use the oriented raw scores
    # rather than the val-frozen min-max threshold above.
    rec["auroc"] = roc_auc(sy, ss)
    rec["tpr@fpr1"] = tpr_at_fpr(sy, ss, 0.01)
    rec["tpr@fpr5"] = tpr_at_fpr(sy, ss, 0.05)
    return rec


def benchmark_split(
    val_df: pd.DataFrame, split_df: pd.DataFrame, score_cols: Iterable[str], ternary: bool
) -> dict:
    """Per-detector metrics for one split. Always computes binary; adds ternary when
    the split carries a `text_type` column. Skips detector columns absent from either df."""
    out: dict = {}
    for col in score_cols:
        if col not in split_df.columns or col not in val_df.columns:
            continue
        rec = {"binary": eval_detector_binary(val_df, split_df, col)}
        if ternary and "text_type" in split_df.columns:
            rec["ternary"] = eval_detector_ternary(val_df, split_df, col)
        out[col] = rec
    return out


def run_ternary_eval(
    trainer,
    val_dataset,
    test_dataset,
    n_buckets: int,
    head: str = "seqcls",
) -> dict:
    """Calibrate-on-val, evaluate-on-test ternary macro-F1, the headline metric.
    Datasets need a `text_type` column. `head="corn"` decodes the ordinal cumulative
    probabilities to the scalar score instead of the seq-cls softmax-expectation.
    """
    val_logits = _predict_bucket_logits(trainer, val_dataset)
    test_logits = _predict_bucket_logits(trainer, test_dataset)

    if head == "corn":
        from greyscope.corn import corn_scalar_score
        val_scores = corn_scalar_score(val_logits)
        test_scores = corn_scalar_score(test_logits)
    else:
        val_scores = compute_scalar_score(val_logits, n_buckets)
        test_scores = compute_scalar_score(test_logits, n_buckets)

    val_labels = np.asarray([LABEL_TO_ID[t] for t in val_dataset["text_type"]])
    test_labels = np.asarray([LABEL_TO_ID[t] for t in test_dataset["text_type"]])

    val_oriented, flipped = orient_scores(val_scores, val_labels)
    val_scaled = minmax_scale(val_oriented)
    h_thresh, ai_thresh, h_f1, ai_f1 = calibrate_thresholds(val_labels, val_scaled)

    test_oriented = -test_scores if flipped else test_scores
    test_scaled = minmax_scale(test_oriented)
    preds = predict_ternary(test_scaled, h_thresh, ai_thresh)
    metrics = evaluate(test_labels, preds)

    out = {
        "metrics": metrics,
        "detection": detection_from_scalar(test_oriented, test_labels),
        "h_thresh": float(h_thresh),
        "ai_thresh": float(ai_thresh),
        "val_h_f1": float(h_f1),
        "val_ai_f1": float(ai_f1),
        "score_flipped": flipped,
    }

    # Per-language breakdown (v2 trilingual): same global thresholds, F1 + human-vs-AI
    # detection (AUROC / TPR@FPR self-calibrated per language, so the FPR target holds for
    # the hardest language, not just the pooled majority). Absent for v1 (no language).
    cols = getattr(test_dataset, "column_names", [])
    if "language" in cols:
        langs = np.asarray(test_dataset["language"])
        per_language = {}
        for g in sorted(set(langs.tolist())):
            gi = langs == g
            mg = evaluate(test_labels[gi], preds[gi])
            per_language[g] = {"macro_f1": float(mg["macro_f1"]), "f1_human": float(mg["f1_human"]),
                               "f1_ai_generated": float(mg["f1_ai_generated"]),
                               "f1_ai_edited": float(mg["f1_ai_edited"]), "n": int(gi.sum()),
                               "detection": detection_from_scalar(test_oriented[gi], test_labels[gi])}
        out["per_language"] = per_language

    return out


def _predict_bucket_logits(trainer, dataset) -> np.ndarray:
    pred_output = trainer.predict(dataset)
    logits = pred_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits


class StandaloneScorer:
    """Adapts a plain (model, tokenizer) pair to the `trainer.predict` interface that
    run_ternary_eval / eval_ood_splits consume, so exported artifacts — including the
    quantized int4/fp8 ones B0 judges — run the exact eval protocol of a training run."""

    def __init__(self, model, tokenizer, *, max_length: int = 2048, batch_size: int = 16):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

    def predict(self, dataset):
        from types import SimpleNamespace

        from greyscope.scoring import batch_logits

        logits = batch_logits(self.model, self.tokenizer, dataset["prompt"],
                              max_length=self.max_length, batch_size=self.batch_size)
        return SimpleNamespace(predictions=logits)


def eval_ood_splits(trainer, splits_dir, split_names, n_buckets, *, head, flip,
                    h_thresh, ai_thresh, apply_clean=True, limit=None,
                    use_prompt_template=True) -> dict:
    """Score held-out OOD splits with the *val-calibrated* thresholds → ternary macro-F1 +
    per-language, the generalization check the in-domain test can't give (v1's selection flaw).

    Per-split min-max scaling + the frozen val thresholds, matching the OOD protocol. Missing
    split files are skipped. `head="corn"` uses the ordinal cumulative decode. `limit`
    subsamples each split (the eval is at the small no-OOM batch, so full splits are slow; a
    stratified subsample is enough for a directional recipe comparison).
    """
    import os

    from datasets import load_dataset

    from greyscope.data import _prepare_v2_split

    results: dict = {}
    for name in split_names:
        path = os.path.join(splits_dir, f"{name}.csv")
        if not os.path.exists(path):
            continue
        raw = load_dataset("csv", data_files=path,
                           usecols=["text", "language", "text_type", "bucket"])["train"]
        ds = _prepare_v2_split(raw, apply_clean=apply_clean, subset=limit,
                               use_prompt_template=use_prompt_template)
        logits = _predict_bucket_logits(trainer, ds)
        if head == "corn":
            from greyscope.corn import corn_scalar_score
            scores = corn_scalar_score(logits)
        else:
            scores = compute_scalar_score(logits, n_buckets)
        oriented = -scores if flip else scores
        preds = predict_ternary(minmax_scale(oriented), h_thresh, ai_thresh)
        labels = np.asarray([LABEL_TO_ID[t] for t in ds["text_type"]])
        m = evaluate(labels, preds)
        langs = np.asarray(ds["language"])
        per_language = {}
        for g in sorted(set(langs.tolist())):
            gi = langs == g
            mg = evaluate(labels[gi], preds[gi])
            per_language[g] = {"macro_f1": float(mg["macro_f1"]), "n": int(gi.sum()),
                               "detection": detection_from_scalar(oriented[gi], labels[gi])}
        results[name] = {"macro_f1": float(m["macro_f1"]), "f1_human": float(m["f1_human"]),
                         "f1_ai_generated": float(m["f1_ai_generated"]),
                         "f1_ai_edited": float(m["f1_ai_edited"]),
                         "detection": detection_from_scalar(oriented, labels),
                         "per_language": per_language, "n": int(len(ds))}
    return results
