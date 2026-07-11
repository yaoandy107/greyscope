"""CORN ordinal head (Cao, Mirjalili & Raschka, 2023) for the graded bucket target.

K buckets → K−1 conditional logits, head k models P(y > k | y ≥ k). Unconditional P(y > k)
is the cumprod of sigmoids — non-increasing in k, so the decoded rank is consistent by
construction (no weight-sharing, unlike CORAL).
"""
from __future__ import annotations

import numpy as np


def corn_loss(logits, labels):
    """CORN conditional training loss. `logits` [N, K−1], `labels` [N] in 0..K−1 (torch).

    For task k, only samples with y ≥ k contribute, and the target is 1[y > k]; the per-task
    conditional BCE is summed and averaged over the total contributing examples.
    """
    import torch
    import torch.nn.functional as F

    n_tasks = logits.shape[1]
    total_loss = logits.new_zeros(())
    n = 0
    for k in range(n_tasks):
        mask = labels > (k - 1)  # samples with y ≥ k
        m = int(mask.sum())
        if m == 0:
            continue
        target = (labels[mask] > k).to(logits.dtype)  # 1[y > k]
        pred = logits[mask, k]
        # numerically stable: logsigmoid(p) - p == log(1 - sigmoid(p))
        total_loss = total_loss - torch.sum(
            F.logsigmoid(pred) * target + (F.logsigmoid(pred) - pred) * (1.0 - target)
        )
        n += m
    return total_loss / max(n, 1)


def corn_ranking_loss(logits, labels, margin: float = 0.25):
    """Hard-negative pairwise margin loss at the human/AI boundary (MELD, arXiv:2605.06903), a
    TPR@low-FPR lever added to corn_loss. Operates on P(y>0) ("any AI present"), not the graded
    scalar, so a lightly-edited row sits on the AI side without being forced a full magnitude-margin
    above human. Penalizes every (non-human, human) pair with a P(y>0) gap under `margin`; returns 0
    for a single-class batch."""
    import torch

    detect = torch.sigmoid(logits[:, 0])       # P(y > 0)
    human = detect[labels == 0]
    nonhuman = detect[labels > 0]
    if human.numel() == 0 or nonhuman.numel() == 0:
        return logits.new_zeros(())
    gap = nonhuman[:, None] - human[None, :]   # [n_ai, n_human]; want each ≥ margin
    return torch.clamp(margin - gap, min=0.0).mean()


def corn_cumulative_probs(logits: np.ndarray) -> np.ndarray:
    """[N, K−1] conditional logits → [N, K−1] unconditional P(y > k) (cumprod of sigmoids)."""
    from scipy.special import expit

    return np.cumprod(expit(np.asarray(logits, dtype=float)), axis=1)


def corn_bucket_probs(logits: np.ndarray) -> np.ndarray:
    """[N, K−1] logits → [N, K] discrete P(y = k), from the cumulative P(y > k):
    P(y=k) = P(y>k−1) − P(y>k), with P(y>−1)=1 and P(y>K−1)=0. Rows sum to 1 and stay
    non-negative because P(y>k) is non-increasing (the CORN rank-consistency guarantee)."""
    cum = corn_cumulative_probs(logits)  # [N, K−1] = P(y>0)…P(y>K−2)
    n = cum.shape[0]
    upper = np.concatenate([np.ones((n, 1)), cum], axis=1)   # P(y>k−1), k=0…K−1
    lower = np.concatenate([cum, np.zeros((n, 1))], axis=1)  # P(y>k),   k=0…K−1
    return upper - lower


def corn_scalar_score(logits: np.ndarray) -> np.ndarray:
    """[N, K−1] logits → scalar AI-ness in [0, 1]: expected rank Σ_k P(y > k) over (K−1)."""
    cum = corn_cumulative_probs(logits)
    return cum.sum(axis=1) / cum.shape[1]


def corn_predict_buckets(logits: np.ndarray) -> np.ndarray:
    """[N, K−1] logits → predicted bucket 0..K−1: count of P(y > k) > 0.5 (rank-consistent)."""
    return (corn_cumulative_probs(logits) > 0.5).sum(axis=1).astype(int)
