"""Native Apple-Silicon inference for the MLX Greyscope artifact."""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache

from .inference import DetectionResult, Mode, _load_calibration, decode_logits
from .preprocess import clean_text

MLX_Q4_MODEL = "yaoandy107/greyscope-v2-qwen3.5-4b-mlx-4bit"
MLX_INT4_MODEL = MLX_Q4_MODEL  # Backward-compatible name.
MLX_MODELS = {"q4": MLX_Q4_MODEL}


def _resolve_mlx_model(source: str) -> str:
    return MLX_MODELS.get(source, source)


@lru_cache(maxsize=2)
def _load_mlx(source: str = MLX_INT4_MODEL):
    try:
        from mlx_lm import load
    except ImportError as exc:
        raise RuntimeError(
            "Native Mac inference requires the optional dependencies: "
            "pip install 'greyscope[mac]'"
        ) from exc

    model, tokenizer = load(source, lazy=False)
    return model, tokenizer, _load_calibration(source)


def detect_mlx(
    text: str,
    mode: Mode = "ternary",
    *,
    model: str = MLX_INT4_MODEL,
) -> DetectionResult:
    """Classify one passage with native MLX on Apple Silicon."""
    import mlx.core as mx

    loaded_model, tokenizer, calib = _load_mlx(_resolve_mlx_model(model))
    body = clean_text(text) if calib["lowercase"] else text
    prompt = calib["prompt_template"].format(text=body)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)[: calib["max_length"]]
    raw = loaded_model(mx.array([token_ids]))
    mx.eval(raw)
    return decode_logits(raw[0].tolist(), calib, mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-text detection (Greyscope MLX).")
    parser.add_argument("text", nargs="?", help="Text to classify (reads stdin if omitted).")
    parser.add_argument("--mode", choices=("ternary", "binary"), default="ternary")
    parser.add_argument(
        "--model",
        default="q4",
        help="q4 (default), a Hugging Face model ID, or a local MLX artifact directory.",
    )
    args = parser.parse_args()
    if args.text is None and sys.stdin.isatty():
        parser.error("no text given (pass it as an argument or pipe it on stdin)")
    text = args.text if args.text is not None else sys.stdin.read()
    print(json.dumps(detect_mlx(text, mode=args.mode, model=args.model), ensure_ascii=False))


if __name__ == "__main__":
    main()
