"""Calibrated AI-text detection with the merged Greyscope model (ternary or binary).

Loads with plain `transformers` (no Unsloth/FLA) and applies the thresholds
shipped in the model's `calibration.json`. Runs on CUDA, Apple MPS, or CPU,
picking the fastest available.

Usage:
    python -m greyscope.inference "..."
    echo "..." | python -m greyscope.inference
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from typing import Literal, TypedDict

import numpy as np
import torch

from .preprocess import clean_text

DEFAULT_MODEL = "yaoandy107/greyscope-qwen3.5-4b"


Mode = Literal["ternary", "binary"]


class DetectionResult(TypedDict):
    """Output of `detect`: a `label`, a 0-1 `ai_involvement` score, and the
    per-bucket probabilities. In ternary mode `label` is human / AI-edited /
    AI-generated; in binary mode it is human / AI at the accusation-safe
    threshold. `ai_involvement` and `bucket_probs` are identical in both modes."""

    label: str
    ai_involvement: float
    bucket_probs: dict[str, float]


def load_seqcls_model(source: str, *, dtype, device: str | None = None):
    """Load a merged seq-cls model + tokenizer (HF Hub id or local dir) with plain
    transformers, applying the deploy conventions: pad-token fallback, right padding
    (the head reads the last non-pad token), eval mode. Returns (tokenizer, model)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(source)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(source, dtype=dtype).eval()
    model.config.pad_token_id = tok.pad_token_id
    if device:
        model = model.to(device)
    return tok, model


@lru_cache(maxsize=1)
def _load():
    from huggingface_hub import hf_hub_download

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # bf16 matches the calibrated thresholds; fp16 overflows the GDN recurrence on MPS.
    tok, model = load_seqcls_model(DEFAULT_MODEL, dtype=torch.bfloat16, device=device)

    calib_path = hf_hub_download(repo_id=DEFAULT_MODEL, filename="calibration.json")
    with open(calib_path) as fh:
        return model, tok, json.load(fh)


@torch.no_grad()
def detect(text: str, mode: Mode = "ternary") -> DetectionResult:
    """Classify one passage and return a 0-1 AI-involvement score plus a label.

    Applies the training-time preprocessing, decodes the 4-bucket logits to a
    continuous score, and turns it into a `label` with the calibrated thresholds
    from calibration.json:

    - "ternary" (default): human / AI-edited / AI-generated.
    - "binary": human / AI at the accusation-safe operating point — the
      threshold is pinned to <=1% false accusations on the hardest human
      subgroup (non-native English; see calibration.json binary_fpr_target).
      Tune it per deployment rather than reading it as ground truth.
    """
    if mode not in ("ternary", "binary"):
        raise ValueError(f"mode must be 'ternary' or 'binary', got {mode!r}")
    model, tok, calib = _load()
    n_buckets = calib["n_buckets"]
    body = clean_text(text) if calib["lowercase"] else text
    prompt = calib["prompt_template"].format(text=body)
    enc = tok(prompt, return_tensors="pt", truncation=True,
              max_length=calib["max_length"], add_special_tokens=False).to(model.device)

    raw = model(**enc).logits[0].float()
    # CORN emits K−1 conditional logits (decode via the cumulative-sigmoid product); seq-cls
    # emits K bucket logits (softmax-expectation). head_type rides in calibration.json.
    if calib.get("head_type", "seqcls") == "corn":
        from .corn import corn_bucket_probs, corn_scalar_score

        arr = raw.cpu().numpy()[None, :]  # [1, K−1]
        probs = corn_bucket_probs(arr)[0]
        scalar = float(corn_scalar_score(arr)[0])
    else:
        probs = torch.softmax(raw, dim=-1).cpu().numpy()
        scalar = float((probs * np.arange(n_buckets)).sum() / (n_buckets - 1))
    oriented = -scalar if calib["flip"] else scalar
    lo, hi = calib["score_min"], calib["score_max"]
    scaled = min(max((oriented - lo) / (hi - lo), 0.0), 1.0)

    if mode == "binary":
        label = "AI" if scaled > calib["binary_threshold"] else "human"
    else:
        idx = 0 if scaled < calib["h_thresh"] else 1 if scaled > calib["ai_thresh"] else 2
        label = calib["label_names"][idx]

    return {
        "label": label,
        "ai_involvement": round(scaled, 3),
        "bucket_probs": {d: round(float(p), 3) for d, p in zip(calib["bucket_descriptions"], probs)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="AI-text detection (Greyscope, seq-cls head).")
    ap.add_argument("text", nargs="?", help="Text to classify (reads stdin if omitted).")
    ap.add_argument("--mode", choices=("ternary", "binary"), default="ternary",
                    help="ternary = human/AI-edited/AI-generated (default); binary = accusation-safe human/AI.")
    args = ap.parse_args()

    if args.text is None and sys.stdin.isatty():
        ap.error("no text given (pass it as an argument or pipe it on stdin)")
    text = args.text if args.text is not None else sys.stdin.read()
    print(json.dumps(detect(text, mode=args.mode), ensure_ascii=False))


if __name__ == "__main__":
    main()
