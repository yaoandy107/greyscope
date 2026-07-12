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


def compute_sample_weights(
    languages: list[str], labels: list[int], temperature: float = 1.0
) -> list[float]:
    """Per-sample weights that balance language AND bucket jointly: smoothed inverse
    (language, bucket) frequency, normalized to mean 1.

    The trilingual mix is mildly language-skewed (EN ~39% / ja ~32% / zh-tw ~29%) and heavily
    bucket-skewed — the graded middle (buckets 1–2) is only ~6% of rows, since it comes
    solely from edits. Bucket class-weights alone can't rebalance *within* a language; joint
    (language, bucket) inverse-frequency lifts the thin middle cells without letting any
    language or minority bucket drown. Feed to a WeightedRandomSampler or a per-sample
    weighted loss. Returns [] for empty input.

    `temperature` τ smooths the balancing: weight ∝ count**(−τ). τ=1 is full inverse-frequency
    (equal sampling mass per (language, bucket) cell — most aggressive); τ=0 is natural
    frequency (no balancing); τ≈0.3–0.5 is the multilingual default (sqrt-smoothed) that lifts
    the thin ja/zh-tw cells without oversampling them so hard the LoRA memorizes them.
    """
    from collections import Counter

    pairs = list(zip(languages, labels))
    if not pairs:
        return []
    counts = Counter(pairs)
    raw = np.asarray([counts[p] ** (-temperature) for p in pairs])
    return (raw * (len(raw) / raw.sum())).tolist()


@dataclass
class PreparedData:
    train: Any
    val: Any
    test: Any
    class_weights: list[float]
    n_buckets: int
    sample_weights: list[float] | None = None  # per-train-row; language+bucket balancing


def _prepare_editlens_split(split, cfg, subset: int | None):
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


def prepare_editlens_data(cfg) -> PreparedData:
    """Load pangram/editlens_iclr and return formatted train/val/test + class weights.

    EditLens is EVAL-only in this repo (the EN benchmark, RAID flip calibration, and the
    deploy calibration bundle); the shipped model trains on the trilingual splits below.
    """
    from datasets import load_dataset

    raw = load_dataset(cfg.dataset)
    train = _prepare_editlens_split(raw["train"], cfg, cfg.train_subset)
    return PreparedData(
        train=train,
        val=_prepare_editlens_split(raw["val"], cfg, cfg.val_subset),
        test=_prepare_editlens_split(raw["test"], cfg, cfg.test_subset),
        class_weights=compute_class_weights(train["label"], cfg.n_buckets),
        n_buckets=cfg.n_buckets,
    )


SPLITS_DIR = "data/v2/splits"


def _prepare_split(split, *, apply_clean: bool, subset: int | None = None, seed: int = 42,
                      use_prompt_template: bool = True):
    """Format a training split. The precomputed per-language `bucket` is the label.

    No English word-count filter: `count_words` treats a CJK run as ~one word, so a
    `min_words` gate would delete nearly every ja/zh-tw row — the build gates already
    enforce length per language. `clean_text` is CJK-safe (lower/whitespace are no-ops there).
    `subset` (smoke runs) samples after shuffle.
    """
    if subset:
        split = split.shuffle(seed=seed).select(range(min(subset, len(split))))
    if apply_clean:
        split = split.map(lambda ex: {"text": clean_text(ex["text"])})
    return split.map(lambda ex: {"label": int(ex["bucket"]),
                                 "prompt": (PROMPT_TEMPLATE.format(text=ex["text"])
                                            if use_prompt_template else ex["text"])})


def prepare_data(cfg, splits_dir: str = SPLITS_DIR) -> PreparedData:
    """Load the local trilingual splits for training. Buckets are precomputed with
    per-language cuts (used directly); returns both bucket class-weights and joint
    language+bucket per-sample weights so the EN-heavy mix can be balanced at train time.
    """
    from datasets import load_dataset

    files: dict = {s: f"{splits_dir}/{s}.csv" for s in ("train", "val", "test")}
    extra = list(getattr(cfg, "train_extra_files", ()) or ())
    if extra:  # train-only augmentation files (e.g. the paraphrase-invariance slice)
        files["train"] = [files["train"], *extra]
    # Load only the columns training needs. Other columns (e.g. `model`) are empty on human
    # rows but strings on AI rows, so CSV type-inference picks `double` then fails to cast
    # "openai/..." — restricting columns sidesteps the mixed null/string inference entirely.
    raw = load_dataset("csv", data_files=files,
                       usecols=["text", "language", "text_type", "bucket"])
    use_prompt = getattr(cfg, "use_prompt_template", True)
    train = _prepare_split(raw["train"], apply_clean=cfg.apply_clean_text,
                              subset=cfg.train_subset, seed=cfg.seed, use_prompt_template=use_prompt)
    return PreparedData(
        train=train,
        val=_prepare_split(raw["val"], apply_clean=cfg.apply_clean_text,
                              subset=cfg.val_subset, seed=cfg.seed, use_prompt_template=use_prompt),
        test=_prepare_split(raw["test"], apply_clean=cfg.apply_clean_text,
                               subset=cfg.test_subset, seed=cfg.seed, use_prompt_template=use_prompt),
        class_weights=compute_class_weights(train["label"], cfg.n_buckets),
        sample_weights=compute_sample_weights(
            train["language"], train["label"],
            temperature=getattr(cfg, "sample_weight_temperature", 0.5)),
        n_buckets=cfg.n_buckets,
    )
