"""OOD and benchmark-suite evaluation of a merged detector (the logic behind
modal/evaluate.py): EditLens OOD splits with val-calibrated thresholds, and the
OpenPangram benchmark CSVs scored through one harness.

Kept out of eval.py so its metrics stay pure numpy/pandas; these need a loaded
model (or a score_fn closed over one) and touch the network and filesystem.
"""
from __future__ import annotations

import os
import urllib.request
from typing import Callable

import numpy as np
import pandas as pd

EDITLENS_DATA_URL = "https://raw.githubusercontent.com/pangramlabs/EditLens/main/data"
OOD_SPLITS = ["test_enron", "test_llama"]
BENCHMARK_SPLITS = ["test", "test_enron", "test_llama", "raid_10k", "human_detectors",
                    "nonnative_english"]
# Label sources in the benchmark CSVs, not detectors under test.
NON_DETECTOR_SCORE_COLS = {"cosine_score", "soft_ngrams_score"}

ScoreFn = Callable[[list[str]], np.ndarray]


def fetch_editlens_csv(name: str, data_dir: str) -> str:
    """Download data/<name>.csv from the EditLens repo into `data_dir` (cached);
    returns the local path."""
    os.makedirs(data_dir, exist_ok=True)
    dst = os.path.join(data_dir, f"{name}.csv")
    if not os.path.exists(dst):
        urllib.request.urlretrieve(f"{EDITLENS_DATA_URL}/{name}.csv", dst)
        print(f"[bench] fetched {name}.csv", flush=True)
    return dst


def greyscope_score_fn(model, tok, *, n_buckets: int = 4, max_length: int = 2048,
                       normalize: bool = True) -> ScoreFn:
    """Raw texts → scalar AI-ness scores: clean_text + prompt template + score_prompts.

    `normalize=False` reproduces the EditLens-faithful preprocessing (no Unicode
    hardening) — the baseline arm for measuring the homoglyph/attack-axis recovery.
    """
    from greyscope.data import PROMPT_TEMPLATE
    from greyscope.preprocess import clean_text
    from greyscope.scoring import score_prompts

    head = getattr(getattr(model, "config", None), "head_type", "seqcls")

    def score(texts: list[str]) -> np.ndarray:
        prompts = [PROMPT_TEMPLATE.format(text=clean_text(str(t), normalize=normalize)) for t in texts]
        return score_prompts(model, tok, prompts, n_buckets, head=head, max_length=max_length)

    return score


def run_ood_eval(model, tok, *, in_domain: float, n_buckets: int = 4, max_length: int = 2048,
                 on_split: Callable[[dict], None] | None = None) -> dict:
    """OOD ternary macro-F1 on test_enron (domain shift) and test_llama (generator shift),
    with thresholds calibrated on in-domain val.

    `in_domain` is the training run's test macro-F1, quoted to report the OOD drop.
    `on_split(results_so_far)` fires after each split for incremental persistence.
    """
    from datasets import load_dataset

    from greyscope.config import DataConfig
    from greyscope.data import _prepare_editlens_split, prepare_editlens_data
    from greyscope.eval import (
        LABEL_TO_ID, calibrate_thresholds, evaluate, minmax_scale, orient_scores, predict_ternary,
    )
    from greyscope.scoring import score_prompts

    cfg = DataConfig(n_buckets=n_buckets)
    head = getattr(model.config, "head_type", "seqcls")
    val = prepare_editlens_data(DataConfig(n_buckets=n_buckets, train_subset=100)).val  # full val
    vs = score_prompts(model, tok, val["prompt"], n_buckets, head=head, max_length=max_length)
    vlab = np.asarray([LABEL_TO_ID[t] for t in val["text_type"]])
    v_or, flipped = orient_scores(vs, vlab)
    h_thresh, ai_thresh, _, _ = calibrate_thresholds(vlab, minmax_scale(v_or))
    print(f"[ood] calibrated on in-domain val: h={h_thresh:.3f} ai={ai_thresh:.3f} "
          f"flipped={flipped} (n={len(val)})")

    raw = load_dataset(cfg.dataset)
    results: dict = {
        "in_domain": in_domain,
        "thresholds": {"h": float(h_thresh), "ai": float(ai_thresh), "flipped": bool(flipped)},
    }
    for name in OOD_SPLITS:
        ds = _prepare_editlens_split(raw[name], cfg, None)
        sc = score_prompts(model, tok, ds["prompt"], n_buckets, head=head, max_length=max_length)
        labs = np.asarray([LABEL_TO_ID[t] for t in ds["text_type"]])
        preds = predict_ternary(minmax_scale(-sc if flipped else sc), h_thresh, ai_thresh)
        m = evaluate(labs, preds)
        results[name] = {"macro_f1": float(m["macro_f1"]), "f1_human": float(m["f1_human"]),
                         "f1_ai_generated": float(m["f1_ai_generated"]),
                         "f1_ai_edited": float(m["f1_ai_edited"]), "n": len(ds)}
        print(f"[ood] {name:11s} macro-F1={m['macro_f1']:.4f}  "
              f"(h {m['f1_human']:.3f} / ai {m['f1_ai_generated']:.3f} / edit {m['f1_ai_edited']:.3f}; n={len(ds)})  "
              f"drop vs in-domain {in_domain:.4f}: {m['macro_f1'] - in_domain:+.4f}")
        if on_split:
            on_split(results)
    return results


def run_benchmark_suite(score_fn: ScoreFn, *, data_dir: str, scores_dir: str,
                        on_split: Callable[[dict], None] | None = None) -> dict:
    """Score `score_fn` (as "greyscope_score") plus every baseline `*_score` column in the
    OpenPangram CSVs through eval.benchmark_split, calibrated on in-domain val.

    Dumps per-split score CSVs into `scores_dir`; `on_split(results_so_far)` fires after
    each split for incremental persistence.
    """
    from greyscope.eval import benchmark_split, raid_protocol_split

    def score_and_dump(name: str) -> pd.DataFrame:
        df = pd.read_csv(fetch_editlens_csv(name, data_dir))
        df["greyscope_score"] = score_fn(df["text"].tolist())
        cols = [c for c in ("text_type", "label", "model", "domain") if c in df.columns]
        df[cols + ["greyscope_score"]].to_csv(
            os.path.join(scores_dir, f"scores_{name}.csv"), index=False)
        return df

    val = score_and_dump("val")
    print(f"[bench] scored val (n={len(val)})", flush=True)
    score_cols = [c for c in val.columns if c.endswith("_score") and c not in NON_DETECTOR_SCORE_COLS]
    print(f"[bench] detectors: {score_cols}", flush=True)

    results: dict = {"score_cols": score_cols, "splits": {}}
    for name in BENCHMARK_SPLITS:
        df = score_and_dump(name)
        ternary = "text_type" in df.columns
        split: dict = {"n": int(len(df)), "ternary": ternary,
                       "detectors": benchmark_split(val, df, score_cols, ternary)}
        # RAID's fair protocol (per-domain threshold @ fixed FPR → TPR) applies to any
        # split with per-domain structure and both classes — currently just raid_10k.
        if "domain" in df.columns and "label" in df.columns and df["label"].nunique() > 1:
            split["raid_protocol"] = raid_protocol_split(df, score_cols)
        results["splits"][name] = split

        g = split["detectors"].get("greyscope_score", {})
        gb, gt = g.get("binary", {}), g.get("ternary")
        msg = f"[bench] {name:18s} greyscope binF1={gb.get('macro_f1')} fpr={gb.get('fpr')}"
        gr = split.get("raid_protocol", {}).get("greyscope_score")
        if gr and gr["tpr"] is not None:
            msg += f"  RAID tpr@fpr{int(gr['target_fpr'] * 100)}={gr['tpr']:.4f} (fpr={gr['fpr']:.4f})"
        if gt:
            msg += (f"  terF1={gt['macro_f1']:.4f} acc={gt['accuracy']:.4f} "
                    f"(h{gt['f1_human']:.3f}/ai{gt['f1_ai_generated']:.3f}/ed{gt['f1_ai_edited']:.3f})")
        print(msg, flush=True)
        if on_split:
            on_split(results)
    return results
