"""Trainer extensions for the seq-cls head: class-weighted cross-entropy + macro-F1 metrics."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _detection_auroc(gold: np.ndarray, ai_prob: np.ndarray) -> float:
    """Human(0) vs any-AI(>0) AUROC from P(y>0). The checkpoint-selection metric: aligned
    with the product's detection boundary, unlike 4-bucket macro-F1 (which the thin middle
    buckets make noisy). 0.5 on a degenerate single-class eval slice."""
    from sklearn.metrics import roc_auc_score

    y = (gold > 0).astype(int)
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, ai_prob))


def make_compute_metrics(n_buckets: int) -> Callable[[Any], dict[str, float]]:
    """compute_metrics for the sequence-classification head.

    The head emits `[N, n_buckets]` logits and `label_ids` is the `[N]` integer
    bucket, so no last-token gather is needed. Returns raw keys; HF Trainer adds
    the "eval_" prefix → "eval_macro_f1" / "eval_detection_auroc".
    """
    from scipy.special import softmax
    from sklearn.metrics import f1_score, recall_score

    def _compute(eval_pred) -> dict[str, float]:
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        gold = np.asarray(eval_pred.label_ids).astype(int)
        preds = logits.argmax(axis=-1)

        macro = f1_score(gold, preds, average="macro", labels=list(range(n_buckets)), zero_division=0)
        per_class = f1_score(gold, preds, average=None, labels=list(range(n_buckets)), zero_division=0)
        # per-bucket recall too: F1 can look healthy while the model quietly *ignores* the thin
        # middle buckets (low recall masked by high precision) — the graded middle is the product.
        per_recall = recall_score(gold, preds, average=None, labels=list(range(n_buckets)), zero_division=0)
        acc = float((preds == gold).mean())

        ai_prob = 1.0 - softmax(logits, axis=1)[:, 0]  # P(bucket > 0)
        out = {"macro_f1": float(macro), "accuracy": acc, "detection_auroc": _detection_auroc(gold, ai_prob)}
        for i, (f, r) in enumerate(zip(per_class, per_recall)):
            out[f"f1_bucket_{i}"] = float(f)
            out[f"recall_bucket_{i}"] = float(r)
        return out

    return _compute


def make_corn_compute_metrics(n_buckets: int) -> Callable[[Any], dict[str, float]]:
    """compute_metrics for the CORN head: decode the [N, K−1] conditional logits to a
    bucket via the cumulative-product rule, then the same macro-F1 keys as seq-cls."""
    from sklearn.metrics import f1_score, recall_score

    from greyscope.corn import corn_cumulative_probs, corn_predict_buckets

    def _compute(eval_pred) -> dict[str, float]:
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        gold = np.asarray(eval_pred.label_ids).astype(int)
        preds = corn_predict_buckets(logits)

        macro = f1_score(gold, preds, average="macro", labels=list(range(n_buckets)), zero_division=0)
        per_class = f1_score(gold, preds, average=None, labels=list(range(n_buckets)), zero_division=0)
        # per-bucket recall too: F1 can mask a model that ignores the thin middle buckets (the graded
        # middle is the product) — watch recall_bucket_1/2 during training.
        per_recall = recall_score(gold, preds, average=None, labels=list(range(n_buckets)), zero_division=0)
        acc = float((preds == gold).mean())

        ai_prob = corn_cumulative_probs(logits)[:, 0]  # P(y > 0), the CORN detection head
        out = {"macro_f1": float(macro), "accuracy": acc, "detection_auroc": _detection_auroc(gold, ai_prob)}
        for i, (f, r) in enumerate(zip(per_class, per_recall)):
            out[f"f1_bucket_{i}"] = float(f)
            out[f"recall_bucket_{i}"] = float(r)
        return out

    return _compute


def build_weighted_trainer_class(class_weights: list[float] | None):
    """Trainer subclass replacing the model's built-in loss with class-weighted CE."""
    import torch
    from torch import nn
    from transformers import Trainer

    bucket_weights = (
        torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
    )

    class WeightedSeqClsTrainer(Trainer):
        _bucket_weights = bucket_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits  # [batch, n_buckets]

            weight = None
            if self._bucket_weights is not None:
                weight = self._bucket_weights.to(logits.device, dtype=logits.dtype)

            loss_fct = nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits, labels.to(logits.device))
            return (loss, outputs) if return_outputs else loss

    return WeightedSeqClsTrainer


def build_sampler_trainer_class(sample_weights: list[float] | None):
    """Trainer subclass that draws training examples via a WeightedRandomSampler from
    per-sample weights (joint language+bucket balancing, from data.compute_sample_weights).

    Loss stays plain CE: the sampler already balances buckets, so class-weighting on top
    would double-count. Use this OR build_weighted_trainer_class, not both. For v2 the
    sampler is preferred because it balances *language* too, which loss class-weights can't.
    """
    import torch
    from torch.utils.data import WeightedRandomSampler
    from transformers import Trainer

    weights = torch.as_tensor(sample_weights, dtype=torch.double) if sample_weights else None

    class SampledSeqClsTrainer(Trainer):
        _sample_weights = weights

        def _get_train_sampler(self, *args, **kwargs):
            if self._sample_weights is None:
                return super()._get_train_sampler(*args, **kwargs)
            return WeightedRandomSampler(
                self._sample_weights, num_samples=len(self._sample_weights), replacement=True)

    return SampledSeqClsTrainer


def build_corn_trainer_class(sample_weights: list[float] | None,
                             ranking_weight: float = 0.0, ranking_margin: float = 0.25):
    """Trainer for the CORN ordinal head: the joint language+bucket WeightedRandomSampler
    (as in the seq-cls v2 path) plus the CORN conditional loss instead of cross-entropy.

    `ranking_weight > 0` adds the MELD hard-negative ranking loss at the human/AI boundary
    (corn.corn_ranking_loss) — the TPR@low-FPR lever. 0 (default) = the plain conditional loss."""
    import torch
    from torch.utils.data import WeightedRandomSampler
    from transformers import Trainer

    from greyscope.corn import corn_loss, corn_ranking_loss

    weights = torch.as_tensor(sample_weights, dtype=torch.double) if sample_weights else None

    class CornTrainer(Trainer):
        _sample_weights = weights
        _ranking_weight = ranking_weight
        _ranking_margin = ranking_margin

        def _get_train_sampler(self, *args, **kwargs):
            if self._sample_weights is None:
                return super()._get_train_sampler(*args, **kwargs)
            return WeightedRandomSampler(
                self._sample_weights, num_samples=len(self._sample_weights), replacement=True)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            labels = labels.to(logits.device)
            loss = corn_loss(logits, labels)
            if self._ranking_weight > 0:
                loss = loss + self._ranking_weight * corn_ranking_loss(logits, labels, self._ranking_margin)
            return (loss, outputs) if return_outputs else loss

    return CornTrainer
