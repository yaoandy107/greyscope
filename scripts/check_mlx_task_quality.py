#!/usr/bin/env python3
"""Evaluate native MLX inference on a deterministic trilingual test sample."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from greyscope.corn import corn_scalar_score
from greyscope.equivalence import select_task_probe
from greyscope.eval import LABEL_TO_ID, detection_from_scalar, evaluate, predict_ternary
from greyscope.inference import _load_calibration
from greyscope.preprocess import clean_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-csv", type=Path, default=Path("data/v2/splits/test.csv"))
    parser.add_argument("--per-group", type=int, default=20)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(args.model, lazy=False)
    calibration = _load_calibration(args.model)
    sample = select_task_probe(args.test_csv, per_group=args.per_group)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    cached = {}
    if checkpoint_path.is_file():
        for line in checkpoint_path.read_text().splitlines():
            row = json.loads(line)
            cached[row["text_id"]] = row["logits"]
        print(f"resuming with {len(cached)} cached rows from {checkpoint_path}", flush=True)

    logits = []
    started = time.perf_counter()
    for index, row in sample.iterrows():
        row_logits = cached.get(row["text_id"])
        if row_logits is None:
            body = clean_text(row["text"]) if calibration["lowercase"] else row["text"]
            prompt = calibration["prompt_template"].format(text=body)
            tokens = tokenizer.encode(prompt, add_special_tokens=False)[: calibration["max_length"]]
            output = model(mx.array([tokens]))
            mx.eval(output)
            row_logits = [float(value) for value in output[0].tolist()]
            with checkpoint_path.open("a") as checkpoint:
                checkpoint.write(json.dumps({"text_id": row["text_id"], "logits": row_logits}) + "\n")
        logits.append(row_logits)
        print(
            f"[{index + 1:03d}/{len(sample)}] {row['language']} {row['text_type']}",
            flush=True,
        )

    scores = corn_scalar_score(np.asarray(logits))
    oriented = -scores if calibration["flip"] else scores
    scaled = np.clip(
        (oriented - calibration["score_min"])
        / (calibration["score_max"] - calibration["score_min"]),
        0.0,
        1.0,
    )
    labels = sample["text_type"].map(LABEL_TO_ID).to_numpy()
    predictions = predict_ternary(scaled, calibration["h_thresh"], calibration["ai_thresh"])
    metrics = evaluate(labels, predictions)
    metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    detection = detection_from_scalar(oriented, labels)

    per_language = {}
    for language in sorted(sample["language"].unique()):
        mask = sample["language"].to_numpy() == language
        language_metrics = evaluate(labels[mask], predictions[mask])
        language_metrics["confusion_matrix"] = language_metrics["confusion_matrix"].tolist()
        per_language[language] = {
            "metrics": language_metrics,
            "detection": detection_from_scalar(oriented[mask], labels[mask]),
        }

    elapsed = time.perf_counter() - started
    payload = {
        "artifact": args.model,
        "test_csv": str(args.test_csv),
        "selection": {
            "method": "deterministic stratified language x text_type",
            "per_group": args.per_group,
            "n": len(sample),
        },
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "detection": detection,
        "per_language": per_language,
        "rows": [
            {
                "text_id": row["text_id"],
                "language": row["language"],
                "text_type": row["text_type"],
                "score": float(scores[index]),
                "scaled_score": float(scaled[index]),
                "prediction": int(predictions[index]),
                "logits": logits[index],
            }
            for index, row in sample.iterrows()
        ],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps({key: payload[key] for key in ("elapsed_seconds", "metrics", "detection", "per_language")}, indent=2))


if __name__ == "__main__":
    main()
