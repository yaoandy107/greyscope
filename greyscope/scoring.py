"""Batched model forward passes for scoring prompts (shared by eval + export).

Kept out of eval.py so its metrics stay pure numpy; these need torch and a loaded model.
"""
from __future__ import annotations

import numpy as np


def batch_logits(model, tok, prompts, *, max_length: int = 2048, batch_size: int = 16) -> np.ndarray:
    """Forward already-formatted `prompts` through `model` in batches → [N, n_labels] float logits.

    Right padding (the seq-cls head reads the last non-pad token); no grad; logits
    returned on CPU as numpy. `prompts` is a list of strings.
    """
    import torch

    rows = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i : i + batch_size], padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt",
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            rows.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def score_prompts(model, tok, prompts, n_buckets: int, *, head: str = "seqcls", **kw) -> np.ndarray:
    """`batch_logits` collapsed to scalar AI-ness scores in [0, 1]. `head="corn"` decodes the
    K−1 ordinal logits via the cumulative-sigmoid product; else the seq-cls softmax-expectation.
    Both callers (calibration, RAID flip) must pass the model's head so the scalar matches the
    one the thresholds were fit on."""
    logits = batch_logits(model, tok, prompts, **kw)
    if head == "corn":
        from greyscope.corn import corn_scalar_score

        return corn_scalar_score(logits)
    from greyscope.eval import compute_scalar_score

    return compute_scalar_score(logits, n_buckets)
