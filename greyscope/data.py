"""Dataset loading and prompt formatting.

Rows bucket on `cosine_score` (lo=0.03, hi=0.15, n_buckets=4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from greyscope.preprocess import clean_text, count_words, score_to_bucket

PROMPT_TEMPLATE = (
    "You are an expert AI-text detector. Classify the passage below.\n"
    "0 = no AI involvement (entirely human-written)\n"
    "1 = lightly AI-edited\n"
    "2 = moderately AI-edited\n"
    "3 = heavily AI-edited or fully AI-generated\n"
    "Respond with a single digit (0, 1, 2, or 3) and nothing else.\n\n"
    'Passage:\n"""\n{text}\n"""\n\nAnswer: '
)


def compute_class_weights(labels: list[int], n_buckets: int) -> list[float]:
    """Inverse-frequency class weights with mean ≈ 1 (sklearn 'balanced')."""
    arr = np.asarray(labels)
    counts = np.bincount(arr, minlength=n_buckets).astype(float)
    counts = np.where(counts == 0, 1.0, counts)  # avoid div-by-zero on absent classes
    weights = len(arr) / (n_buckets * counts)
    return weights.tolist()


@dataclass
class PreparedData:
    train: Any
    val: Any
    test: Any
    class_weights: list[float]
    n_buckets: int


def _prepare_split(split, cfg, subset: int | None):
    if subset:  # subsample before the maps; 3x buffer covers rows the min_words filter drops
        split = split.shuffle(seed=cfg.seed).select(range(min(subset * 3, len(split))))
    split = split.filter(lambda ex: count_words(ex["text"]) >= cfg.min_words)
    if cfg.apply_clean_text:
        split = split.map(lambda ex: {"text": clean_text(ex["text"])})
    split = split.map(lambda ex: {"label": score_to_bucket(
        float(ex[cfg.label_field]), cfg.n_buckets,
        cfg.bucket_lo_threshold, cfg.bucket_hi_threshold)})
    if subset:  # exact size after filtering
        split = split.select(range(min(subset, len(split))))
    return split.map(lambda ex: {"prompt": PROMPT_TEMPLATE.format(text=ex["text"])})


def prepare_data(cfg) -> PreparedData:
    """Load pangram/editlens_iclr and return formatted train/val/test + class weights."""
    from datasets import load_dataset

    raw = load_dataset(cfg.dataset)
    train = _prepare_split(raw["train"], cfg, cfg.train_subset)
    return PreparedData(
        train=train,
        val=_prepare_split(raw["val"], cfg, cfg.val_subset),
        test=_prepare_split(raw["test"], cfg, cfg.test_subset),
        class_weights=compute_class_weights(train["label"], cfg.n_buckets),
        n_buckets=cfg.n_buckets,
    )
