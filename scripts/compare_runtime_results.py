#!/usr/bin/env python3
"""Compare two saved runtime-equivalence result files by text ID."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from greyscope.export import quantization_equivalence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    reference_rows = {row["text_id"]: row for row in reference["rows"]}
    candidate_rows = {row["text_id"]: row for row in candidate["rows"]}
    if reference_rows.keys() != candidate_rows.keys():
        raise ValueError("reference and candidate text IDs differ")

    text_ids = list(reference_rows)
    for text_id in text_ids:
        if reference_rows[text_id]["prompt_sha256"] != candidate_rows[text_id]["prompt_sha256"]:
            raise ValueError(f"prompt mismatch: {text_id}")

    reference_scores = np.asarray([reference_rows[text_id]["local_score"] for text_id in text_ids])
    reference_buckets = np.asarray([reference_rows[text_id]["local_bucket"] for text_id in text_ids])
    candidate_scores = np.asarray([candidate_rows[text_id]["mlx_score"] for text_id in text_ids])
    candidate_buckets = np.asarray([candidate_rows[text_id]["mlx_bucket"] for text_id in text_ids])
    metrics = quantization_equivalence(
        reference_scores, candidate_scores, reference_buckets, candidate_buckets
    )

    payload = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "metrics": metrics,
        "rows": [
            {
                "text_id": text_id,
                "language": reference_rows[text_id]["language"],
                "prompt_sha256": reference_rows[text_id]["prompt_sha256"],
                "reference_score": float(reference_scores[index]),
                "candidate_score": float(candidate_scores[index]),
                "reference_bucket": int(reference_buckets[index]),
                "candidate_bucket": int(candidate_buckets[index]),
            }
            for index, text_id in enumerate(text_ids)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
