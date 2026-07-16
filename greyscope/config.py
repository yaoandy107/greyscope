"""Programmatic data config for `prepare_data`. Training binds plain YAML via Hydra;
this dataclass is the equivalent object the standalone eval/benchmark paths construct."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    dataset: str = "pangram/editlens_iclr"
    n_buckets: int = 4
    bucket_lo_threshold: float = 0.03
    bucket_hi_threshold: float = 0.15
    min_words: int = 75
    train_subset: Optional[int] = None
    val_subset: Optional[int] = None
    test_subset: Optional[int] = None
    seed: int = 42
    apply_clean_text: bool = True
    label_field: str = "cosine_score"
    sample_weight_temperature: float = 0.5  # τ for the joint language+bucket sampler (0=natural,
    #                                         1=full balance); design τ≈0.3–0.5 (smoothed inverse)
    boundary_margin: float = 0.0  # drop train rows with |cosine_score − bucket cut| < margin
    #                               (label noise: humans agree only α≈0.5 on bucketed edit
    #                               magnitude, EditLens Table 3). 0 = off.
    train_extra_files: tuple[str, ...] = ()  # extra train-only CSVs (e.g. the paraphrase aug)
