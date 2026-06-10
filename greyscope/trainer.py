"""Trainer extensions for the seq-cls head: class-weighted cross-entropy + macro-F1 metrics."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def make_compute_metrics(n_buckets: int) -> Callable[[Any], dict[str, float]]:
    """compute_metrics for the sequence-classification head.

    The head emits `[N, n_buckets]` logits and `label_ids` is the `[N]` integer
    bucket, so no last-token gather is needed. Returns raw keys; HF Trainer adds
    the "eval_" prefix → "eval_macro_f1".
    """
    from sklearn.metrics import f1_score

    def _compute(eval_pred) -> dict[str, float]:
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        gold = np.asarray(eval_pred.label_ids).astype(int)
        preds = logits.argmax(axis=-1)

        macro = f1_score(gold, preds, average="macro", labels=list(range(n_buckets)), zero_division=0)
        per_class = f1_score(gold, preds, average=None, labels=list(range(n_buckets)), zero_division=0)
        acc = float((preds == gold).mean())

        out = {"macro_f1": float(macro), "accuracy": acc}
        for i, f in enumerate(per_class):
            out[f"f1_bucket_{i}"] = float(f)
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
