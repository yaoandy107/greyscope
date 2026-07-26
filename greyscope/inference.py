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
import importlib.util
import json
import platform
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch

from .preprocess import clean_text

BF16_MODEL = "yaoandy107/greyscope-v2-qwen3.5-4b"
INT4_MODEL = "yaoandy107/greyscope-v2-qwen3.5-4b-int4"
DEFAULT_MODEL = BF16_MODEL


Mode = Literal["ternary", "binary"]


class DetectionResult(TypedDict):
    """Output of `detect`: a `label`, a 0-1 `ai_involvement` score, and the
    per-bucket probabilities. In ternary mode `label` is human / AI-edited /
    AI-generated; in binary mode it is human / AI at the calibrated
    threshold. `ai_involvement` and `bucket_probs` are identical in both modes."""

    label: str
    ai_involvement: float
    bucket_probs: dict[str, float]


def decode_logits(raw_logits, calib: dict, mode: Mode = "ternary") -> DetectionResult:
    """Decode one model output with the thresholds in ``calibration.json``.

    Kept backend-neutral so Transformers and native MLX inference cannot drift in
    score orientation, scaling, or label semantics.
    """
    if mode not in ("ternary", "binary"):
        raise ValueError(f"mode must be 'ternary' or 'binary', got {mode!r}")
    raw = np.asarray(raw_logits, dtype=float).reshape(1, -1)
    n_buckets = calib["n_buckets"]
    if calib.get("head_type", "seqcls") == "corn":
        from .corn import corn_bucket_probs, corn_scalar_score

        probs = corn_bucket_probs(raw)[0]
        scalar = float(corn_scalar_score(raw)[0])
    else:
        shifted = raw[0] - raw[0].max()
        probs = np.exp(shifted) / np.exp(shifted).sum()
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
        "bucket_probs": {
            description: round(float(probability), 3)
            for description, probability in zip(calib["bucket_descriptions"], probs)
        },
    }


def _available_device(requested: str = "auto") -> str:
    """Resolve a user-facing device name without silently changing explicit choices."""
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model(model: str = "auto") -> str:
    """Resolve the friendly ``auto``/``bf16``/``int4`` aliases to a Hub ID."""
    if model == "bf16":
        return BF16_MODEL
    if model == "int4":
        return INT4_MODEL
    return BF16_MODEL if model == "auto" else model


def _mlx_available() -> bool:
    return importlib.util.find_spec("mlx_lm") is not None


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _is_local_mlx_model(model: str) -> bool:
    config_path = Path(model) / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("model_file") == "mlx_model.py"


def _should_use_mlx(model: str, device: str) -> bool:
    """Select native MLX without importing it on non-Mac installations."""
    if model == "q4" or "-mlx-" in model or _is_local_mlx_model(model):
        if device != "auto":
            raise ValueError("--device applies to Transformers models, not MLX")
        return True
    return model == "auto" and device == "auto" and _is_apple_silicon() and _mlx_available()


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


def _load_calibration(source: str) -> dict:
    from huggingface_hub import hf_hub_download

    local = Path(source) / "calibration.json"
    calib_path = local if local.is_file() else Path(
        hf_hub_download(repo_id=source, filename="calibration.json")
    )
    return json.loads(calib_path.read_text())


@lru_cache(maxsize=4)
def _load(source: str = DEFAULT_MODEL, device: str = "auto"):
    resolved_device = _available_device(device)
    # The unquantized artifact was trained, calibrated, and validated in bf16.
    tok, loaded_model = load_seqcls_model(
        source, dtype=torch.bfloat16, device=resolved_device
    )
    return loaded_model, tok, _load_calibration(source)


@torch.no_grad()
def detect(
    text: str,
    mode: Mode = "ternary",
    *,
    model: str = "auto",
    device: str = "auto",
) -> DetectionResult:
    """Classify one passage and return a 0-1 AI-involvement score plus a label.

    Applies the training-time preprocessing, decodes the 4-bucket logits to a
    continuous score, and turns it into a `label` with the calibrated thresholds
    from calibration.json:

    - "ternary" (default): human / AI-edited / AI-generated.
    - "binary": human / AI at the calibrated operating point — the
      threshold is pinned to <=1% false accusations on the hardest human
      subgroup (non-native English; see calibration.json binary_fpr_target).
      Tune it per deployment rather than reading it as ground truth.
    """
    if _should_use_mlx(model, device):
        from .mlx_inference import detect_mlx

        return detect_mlx(text, mode=mode, model="q4" if model == "auto" else model)

    loaded_model, tok, calib = _load(resolve_model(model), device)
    body = clean_text(text) if calib["lowercase"] else text
    prompt = calib["prompt_template"].format(text=body)
    enc = tok(prompt, return_tensors="pt", truncation=True,
              max_length=calib["max_length"], add_special_tokens=False).to(loaded_model.device)

    raw = loaded_model(**enc).logits[0].float().cpu().numpy()
    return decode_logits(raw, calib, mode)


def main() -> None:
    ap = argparse.ArgumentParser(description="AI-text detection with Greyscope.")
    ap.add_argument("text", nargs="?", help="Text to classify (reads stdin if omitted).")
    ap.add_argument("--mode", choices=("ternary", "binary"), default="ternary",
                    help="ternary = human/AI-edited/AI-generated (default); binary = human/AI.")
    ap.add_argument(
        "--model", default="auto", metavar="MODEL",
        help="auto (default), q4, bf16, int4, a Hugging Face model ID, or a local model directory.",
    )
    ap.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto",
        help="Transformers device override; auto uses MLX Q4 on Apple Silicon when installed.",
    )
    args = ap.parse_args()

    if args.text is None and sys.stdin.isatty():
        ap.error("no text given (pass it as an argument or pipe it on stdin)")
    text = args.text if args.text is not None else sys.stdin.read()
    print(json.dumps(
        detect(text, mode=args.mode, model=args.model, device=args.device), ensure_ascii=False
    ))


if __name__ == "__main__":
    main()
