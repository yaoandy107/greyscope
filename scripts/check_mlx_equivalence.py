#!/usr/bin/env python3
"""Compare the native MLX artifact with saved bf16 predictions on the fixed probe."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local MLX artifact directory")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--val-csv", default="data/v2/splits/val.csv")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    from greyscope.corn import corn_predict_buckets, corn_scalar_score
    from greyscope.equivalence import select_equivalence_probe
    from greyscope.export import quantization_equivalence
    from greyscope.mlx_export import stage_model

    reference = json.loads(args.reference.read_text())
    probe = select_equivalence_probe(args.val_csv)
    if len(probe) != reference["n"]:
        raise ValueError(f"probe size changed: local={len(probe)} reference={reference['n']}")

    ref_rows = {row["text_id"]: row for row in reference["rows"]}
    model_source = Path(args.model)
    stage_tmp = None
    source_config = json.loads((model_source / "config.json").read_text())
    if "model_file" not in source_config:
        stage_tmp = tempfile.TemporaryDirectory(prefix="greyscope-mlx-bf16-")
        staged = Path(stage_tmp.name)
        stage_model(model_source, staged)
        model_source = staged
    model, tokenizer, config = load(str(model_source), lazy=False, return_config=True)
    max_length = int(reference["max_length"])
    logits = []
    rows = []
    for index, row in probe.iterrows():
        ref = ref_rows.get(row["text_id"])
        if ref is None or ref["prompt_sha256"] != row["prompt_sha256"]:
            raise ValueError(f"bf16 reference does not match local prompt: {row['text_id']}")
        token_ids = tokenizer.encode(
            row["prompt"], add_special_tokens=False
        )[:max_length]
        output = model(mx.array([token_ids]))
        mx.eval(output)
        values = [float(value) for value in output[0].tolist()]
        logits.append(values)
        print(f"[{index + 1:02d}/{len(probe)}] {row['language']} bucket={row['bucket']}", flush=True)

    logits_array = np.asarray(logits)
    mlx_scores = corn_scalar_score(logits_array)
    mlx_buckets = corn_predict_buckets(logits_array)
    ref_scores = np.asarray([ref_rows[text_id]["score"] for text_id in probe["text_id"]])
    ref_buckets = np.asarray([
        ref_rows[text_id]["predicted_bucket"] for text_id in probe["text_id"]
    ])
    metrics = quantization_equivalence(ref_scores, mlx_scores, ref_buckets, mlx_buckets)

    per_language = {}
    for language in sorted(probe["language"].unique()):
        mask = probe["language"].to_numpy() == language
        per_language[language] = quantization_equivalence(
            ref_scores[mask], mlx_scores[mask], ref_buckets[mask], mlx_buckets[mask]
        )

    for index, row in probe.iterrows():
        rows.append({
            "text_id": row["text_id"],
            "language": row["language"],
            "bucket": int(row["bucket"]),
            "prompt_sha256": row["prompt_sha256"],
            "bf16_score": float(ref_scores[index]),
            "mlx_score": float(mlx_scores[index]),
            "bf16_bucket": int(ref_buckets[index]),
            "mlx_bucket": int(mlx_buckets[index]),
            "mlx_logits": logits[index],
        })

    passed = (
        metrics["score_pearson"] >= 0.99
        and metrics["bucket_agreement"] >= 0.95
        and metrics["score_maxdiff"] <= 0.55
    )
    payload = {
        "artifact": args.model,
        "quantization": config.get("quantization"),
        "reference": str(args.reference),
        "max_length": max_length,
        "metrics": metrics,
        "per_language": per_language,
        "release_criteria": {
            "score_pearson_min": 0.99,
            "bucket_agreement_min": 0.95,
            "score_maxdiff_max": 0.55,
        },
        "passed_release_probe": passed,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if stage_tmp is not None:
        stage_tmp.cleanup()
    print(json.dumps({"metrics": metrics, "per_language": per_language, "passed": passed}, indent=2))
    if not passed:
        raise SystemExit("MLX artifact failed the bf16 equivalence probe")


if __name__ == "__main__":
    main()
