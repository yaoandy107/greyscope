"""Edit-magnitude scoring (design §7, plan §8): Qwen3-Embedding-8B cosine.

Only the EDITED class needs a score (human=0, generated=1 are fixed by class). For an
edit, `score = 1 - cos(embed(source), embed(edited))` — EditLens's validated proxy in
the locked embedder's scale (Qwen3-Embedding-8B beat gemini-embedding-001, §7/§8). The
LLM-judge was dropped on cost-ROI (the edited class is a label-noise ceiling).

Bucketing is deliberately NOT done here: the pilot reads the score DISTRIBUTION to
re-derive per-language thresholds in Qwen's scale (v1's Linq-scale cuts don't transfer),
so this stage just fills `score`.

Networked (cached embeddings → resumable). The cosine + pairing logic is pure and is
unit-tested through an injectable `embed_fn`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from greyscope.v2 import openrouter

SCORER_MODEL = "qwen/qwen3-embedding-8b"


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def edit_magnitude(source_vec, edited_vec) -> float:
    """1 - cos: 0 = no change, → 1 = wholesale rewrite (design §7)."""
    return 1.0 - cosine(source_vec, edited_vec)


def score_edited(
    rows: list[dict],
    *,
    model: str = SCORER_MODEL,
    embed_fn: Callable = openrouter.embed,
) -> list[dict]:
    """Fill `score = 1 - cos(source, edited)` for every ai_edited row (in place);
    human/generated rows keep their by-class score. One batched, de-duplicated embed
    call over all source + edited texts (identical strings embed once → cheaper + the
    cache stays warm)."""
    edited = [
        r for r in rows
        if r["text_type"] == "ai_edited" and r.get("source_text") and r.get("text")
    ]
    if not edited:
        return rows

    unique: dict[str, None] = {}  # insertion-ordered set of texts to embed
    for row in edited:
        unique.setdefault(row["source_text"], None)
        unique.setdefault(row["text"], None)
    texts = list(unique)
    position = {text: i for i, text in enumerate(texts)}

    vectors = embed_fn(texts, model=model)
    for row in edited:
        row["cosine_score"] = edit_magnitude(
            vectors[position[row["source_text"]]], vectors[position[row["text"]]]
        )
    return rows
