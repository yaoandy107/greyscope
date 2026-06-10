"""Batched scoring helpers (greyscope.scoring) — stubbed model, no weights/GPU."""
import numpy as np
import torch

from greyscope.scoring import batch_logits, score_prompts


class _Enc(dict):
    def to(self, _device):
        return self


class _Tok:
    def __call__(self, prompts, **_kw):
        n = len(prompts)
        return _Enc(input_ids=torch.zeros(n, 4, dtype=torch.long))


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _Model:
    """Returns deterministic logits = bucket index per row, so scores are predictable."""

    device = torch.device("cpu")

    def __call__(self, input_ids=None, **_kw):
        n = input_ids.shape[0]
        # row r → logits favoring bucket (r % 4): one-hot-ish
        rows = []
        for r in range(n):
            v = [-10.0, -10.0, -10.0, -10.0]
            v[r % 4] = 10.0
            rows.append(v)
        return _Out(torch.tensor(rows))


def test_batch_logits_shape_and_batching():
    prompts = [f"p{i}" for i in range(10)]  # 10 prompts, batch_size 4 → 3 batches
    logits = batch_logits(_Model(), _Tok(), prompts, batch_size=4)
    assert logits.shape == (10, 4)
    assert logits.dtype == np.float32 or logits.dtype == np.float64
    assert logits.argmax(1).tolist() == [i % 4 for i in range(10)]  # order preserved across batches


def test_score_prompts_monotone():
    # bucket 0 → score 0.0, bucket 3 → score 1.0 (compute_scalar_score normalizes by n-1)
    s = score_prompts(_Model(), _Tok(), ["a", "b", "c", "d"], n_buckets=4, batch_size=4)
    assert s.shape == (4,)
    assert s[0] < s[1] < s[2] < s[3]
    assert abs(s[0]) < 1e-3 and abs(s[3] - 1.0) < 1e-3
