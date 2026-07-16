"""Official RAID benchmark (raid-bench) evaluation for a Greyscope detector.

A separate module so the optional `raid` dependency never touches benchmark.py or
the training path. RAID's headline metric is TPR@FPR (default 5%), aggregated over
domain x generator x decoding x adversarial attack; the detector contract is
``list[str] -> list[float]`` with higher = more AI.

    pip install raid-bench   # provides the `raid` package

Splits: ``train``/``extra`` carry labels (local dev + gap-decomposition); ``test``
is held out — score it, dump predictions.json, and submit to the leaderboard.
"""
from __future__ import annotations

import json
import os
from typing import Callable

import numpy as np

ScoreFn = Callable[[list[str]], np.ndarray]


def oriented_detector(score_fn: ScoreFn, *, flip: bool) -> Callable[[list[str]], list[float]]:
    """Adapt a raw Greyscope `score_fn` to RAID's ``list[str] -> list[float]`` contract,
    flipping so higher = more AI. `flip` is resolved once on in-domain val because
    single-class RAID strata can't orient themselves."""
    def detector(texts: list[str]) -> list[float]:
        s = np.asarray(score_fn(texts), dtype=float)
        return (-s if flip else s).tolist()

    return detector


def resolve_flip(model, tok, *, n_buckets: int = 4, max_length: int = 2048) -> bool:
    """Resolve the orientation flag on in-domain val so RAID scores read higher = more AI."""
    from greyscope.config import DataConfig
    from greyscope.data import prepare_editlens_data
    from greyscope.eval import LABEL_TO_ID, orient_scores
    from greyscope.scoring import score_prompts

    head = getattr(model.config, "head_type", "seqcls")
    val = prepare_editlens_data(DataConfig(n_buckets=n_buckets, train_subset=100)).val
    vs = score_prompts(model, tok, val["prompt"], n_buckets, head=head, max_length=max_length)
    vlab = np.asarray([LABEL_TO_ID[t] for t in val["text_type"]])
    _, flip = orient_scores(vs, vlab)
    return bool(flip)


def leaderboard_metadata(name: str, **fields) -> dict:
    """Minimal leaderboard metadata; override/extend any field before submitting."""
    meta = {
        "name": name,
        "detector_type": "metric-based",
        "organization": "",
        "description": "Greyscope graded AI-involvement detector (Qwen3.5 LoRA seq-cls).",
        "open_source": True,
        "url": "",
    }
    meta.update(fields)
    return meta


def run_raid(score_fn: ScoreFn, *, flip: bool, out_dir: str, split: str = "extra",
             detector_name: str = "greyscope", include_adversarial: bool = True,
             target_fpr: float = 0.05, run_eval: bool | None = None,
             metadata: dict | None = None, limit: int | None = None) -> dict | None:
    """Run the official RAID harness over `split` and write predictions.json (+ metadata.json).

    `run_eval` defaults to on for labeled splits (train/extra) and off for the held-out
    `test` split. Returns the run_evaluation dict (TPR@FPR per stratum) when evaluated,
    else None. `limit` subsamples for a cheap pipeline sanity check — NOT an official score.
    """
    from raid import run_detection, run_evaluation
    from raid.utils import load_data

    os.makedirs(out_dir, exist_ok=True)
    detector = oriented_detector(score_fn, flip=flip)

    df = load_data(split=split, include_adversarial=include_adversarial)
    if limit is not None:
        # Random (not head) so the subsample keeps RAID's domain/generator/attack mix —
        # head() would slice one ordered stratum and skew TPR@FPR. Still an approximate
        # estimate, NOT the official leaderboard score (which needs the full split).
        df = df.sample(n=min(limit, len(df)), random_state=42).reset_index(drop=True)
        print(f"[raid] approximate subsample: {len(df)} random rows "
              f"(representative estimate, NOT the official leaderboard score)", flush=True)

    predictions = run_detection(detector, df)
    with open(os.path.join(out_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    meta = metadata or leaderboard_metadata(detector_name)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if run_eval is None:
        run_eval = split != "test"
    if not run_eval:
        print(f"[raid] wrote predictions for split={split} (held-out; submit to leaderboard)",
              flush=True)
        return None

    results = run_evaluation(predictions, df, target_fpr=target_fpr)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[raid] split={split} evaluated at FPR={target_fpr} -> results.json", flush=True)
    return results
