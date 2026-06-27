"""Edit-magnitude scoring: Qwen3-Embedding-8B cosine.

Only the EDITED class needs a score (human=0, generated=1 are fixed by class). For an edit,
`score = 1 - cos(embed(source), embed(edited))` — EditLens's validated proxy.

Bucketing is deliberately NOT done here: the build reads the score DISTRIBUTION to re-derive
per-language thresholds, so this stage just fills `score`.

Networked (cached embeddings → resumable); the cosine + pairing logic is pure and unit-tested
through an injectable `embed_fn`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from greyscope.v2 import openrouter

SCORER_MODEL = "qwen/qwen3-embedding-8b"
# Cap embed inputs: a few fineweb/gutenberg sources run to 80–150k chars and the embedding provider
# rejects them (HTTP 400 "parameter invalid"). Truncating BOTH sides to the same window keeps 1−cos a
# valid edit-magnitude proxy (the edit signal is dense early), and source_text never ships (assemble
# drops it), so only the score sees the truncation. p95 of real edits is ~4.7k chars → most are untouched.
EMBED_MAX_CHARS = 8000


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def edit_magnitude(source_vec, edited_vec) -> float:
    """1 - cos: 0 = no change, → 1 = wholesale rewrite."""
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

    def _clip(text: str) -> str:
        return text[:EMBED_MAX_CHARS]

    unique: dict[str, None] = {}  # insertion-ordered set of (clipped) texts to embed
    for row in edited:
        unique.setdefault(_clip(row["source_text"]), None)
        unique.setdefault(_clip(row["text"]), None)
    texts = list(unique)
    position = {text: i for i, text in enumerate(texts)}

    vectors = embed_fn(texts, model=model)
    for row in edited:
        row["cosine_score"] = edit_magnitude(
            vectors[position[_clip(row["source_text"])]], vectors[position[_clip(row["text"])]]
        )
    return rows
