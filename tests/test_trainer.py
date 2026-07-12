"""greyscope.trainer: the WeightedRandomSampler wiring.

Only the sampler branch is unit-tested — instantiated without HF Trainer's heavy __init__
(which needs a model). compute_loss / metrics are integration-tested by the smoke run.
"""
import pytest

from greyscope.trainer import build_sampler_trainer_class


def test_sampler_trainer_uses_weighted_random_sampler():
    pytest.importorskip("transformers")
    from torch.utils.data import WeightedRandomSampler

    cls = build_sampler_trainer_class([1.0, 2.0, 3.0])
    inst = object.__new__(cls)  # bypass Trainer.__init__; only exercise _get_train_sampler
    sampler = inst._get_train_sampler()

    assert isinstance(sampler, WeightedRandomSampler)
    assert sampler.num_samples == 3
    assert sampler.replacement is True


def test_sampler_trainer_none_falls_back_to_default():
    pytest.importorskip("transformers")
    cls = build_sampler_trainer_class(None)
    assert cls._sample_weights is None  # falls through to super()._get_train_sampler


def test_corn_compute_metrics_emits_detection_auroc():
    pytest.importorskip("scipy")
    import numpy as np
    from types import SimpleNamespace

    from greyscope.trainer import make_corn_compute_metrics

    # K=4 → 3 CORN conditional logits; head-0 = P(y>0). Separable humans (0) vs AI (>0).
    logits = np.array([[-5.0, -5, -5], [-4.0, -5, -5], [5.0, 0, -5], [6.0, 4, 2]])
    labels = np.array([0, 0, 1, 3])
    out = make_corn_compute_metrics(4)(SimpleNamespace(predictions=logits, label_ids=labels))
    assert out["detection_auroc"] == 1.0        # the checkpoint-selection metric is emitted + separable
    assert out["tpr_fpr1"] == 1.0 and out["tpr_fpr5"] == 1.0  # ship metric emitted; separable → full TPR
    assert 0.0 <= out["macro_f1"] <= 1.0
    # per-bucket recall is emitted for every bucket (watches the thin middle at train time)
    for i in range(4):
        assert 0.0 <= out[f"recall_bucket_{i}"] <= 1.0
    assert out["recall_bucket_0"] == 1.0        # both true humans predicted human


def test_seqcls_compute_metrics_emits_per_bucket_recall():
    pytest.importorskip("scipy")
    import numpy as np
    from types import SimpleNamespace

    from greyscope.trainer import make_compute_metrics

    # [N, n_buckets] logits; argmax over buckets. Separable across all 4 buckets.
    logits = np.array([[5.0, 0, 0, 0], [4.0, 0, 0, 0], [0, 5.0, 0, 0], [0, 0, 0, 5.0]])
    labels = np.array([0, 0, 1, 3])
    out = make_compute_metrics(4)(SimpleNamespace(predictions=logits, label_ids=labels))
    for i in range(4):
        assert f"recall_bucket_{i}" in out and 0.0 <= out[f"recall_bucket_{i}"] <= 1.0
    assert out["recall_bucket_1"] == 1.0        # the lone middle example is recalled


def test_tpr_at_fpr_thresholds_on_human_negatives():
    import numpy as np

    from greyscope.trainer import _tpr_at_fpr

    gold = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 4 human, 4 any-AI
    # Perfect separation → TPR is 1.0 at any FPR.
    assert _tpr_at_fpr(gold, np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]), 0.01) == 1.0
    # One AI buried inside the human score band → it's missed at a strict FPR.
    partial = _tpr_at_fpr(gold, np.array([0.1, 0.2, 0.3, 0.4, 0.35, 0.9, 0.9, 0.9]), 0.01)
    assert partial == 0.75
    # Single-class slice is undefined → 0.0, never a crash.
    assert _tpr_at_fpr(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]), 0.01) == 0.0
