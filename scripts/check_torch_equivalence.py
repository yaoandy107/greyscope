#!/usr/bin/env python3
"""Compare a local Transformers artifact with saved reference predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--val-csv", default="data/v2/splits/val.csv")
    parser.add_argument("--device", default="mps", choices=("mps", "cuda", "cpu"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import numpy as np
    import torch

    from greyscope.corn import corn_predict_buckets, corn_scalar_score
    from greyscope.equivalence import select_equivalence_probe
    from greyscope.export import quantization_equivalence
    from greyscope.inference import load_seqcls_model

    reference = json.loads(args.reference.read_text())
    ref_rows = {row["text_id"]: row for row in reference["rows"]}
    probe = select_equivalence_probe(args.val_csv)
    tokenizer, model = load_seqcls_model(
        args.model, dtype=torch.bfloat16, device=args.device
    )
    logits = []
    with torch.no_grad():
        for index, row in probe.iterrows():
            ref = ref_rows[row["text_id"]]
            if ref["prompt_sha256"] != row["prompt_sha256"]:
                raise ValueError(f"reference prompt mismatch: {row['text_id']}")
            encoded = tokenizer(
                row["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=reference["max_length"],
                add_special_tokens=False,
            ).to(args.device)
            output = model(**encoded).logits[0].float().cpu().tolist()
            logits.append(output)
            print(f"[{index + 1:02d}/{len(probe)}] {row['language']} bucket={row['bucket']}", flush=True)

    logits_array = np.asarray(logits)
    scores = corn_scalar_score(logits_array)
    buckets = corn_predict_buckets(logits_array)
    ref_scores = np.asarray([ref_rows[text_id]["score"] for text_id in probe["text_id"]])
    ref_buckets = np.asarray([
        ref_rows[text_id]["predicted_bucket"] for text_id in probe["text_id"]
    ])
    metrics = quantization_equivalence(ref_scores, scores, ref_buckets, buckets)
    payload = {
        "artifact": args.model,
        "backend": f"torch-{args.device}-bfloat16",
        "reference": str(args.reference),
        "metrics": metrics,
        "rows": [
            {
                "text_id": row["text_id"],
                "language": row["language"],
                "prompt_sha256": row["prompt_sha256"],
                "reference_score": float(ref_scores[index]),
                "local_score": float(scores[index]),
                "reference_bucket": int(ref_buckets[index]),
                "local_bucket": int(buckets[index]),
                "local_logits": logits[index],
            }
            for index, row in probe.iterrows()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
