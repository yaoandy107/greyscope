"""Validate gemini-embedding-001 as the edit-magnitude scorer against EditLens's
Linq-Embed-Mistral `cosine_score`, on the EN ai_edited rows we already have.

Decision rule (derived from the data, EditLens-style — no human ratings needed
for the safety call):
  - Spearman(gemini 1-cos, Linq cosine) >= ~0.84, the in-dataset agreement
    between EditLens's *own* two proxies (cosine vs soft_ngrams). A same-method
    embedder should clear it comfortably; below it is the red flag.
  - gemini's bucket-band means stay monotonic (no moderate-band compression —
    the documented failure mode of retrieval-style embeddings).
It also dumps the largest-disagreement pairs so the few that matter can be
eyeballed (the efficient version of EditLens's human-agreement study).

Run from the repo root:
    OPENROUTER_API_KEY=... .venv/bin/python scripts/v2_scorer_validation.py [per_bucket]
Optional: GEMINI_TASK_TYPE=SEMANTIC_SIMILARITY to test the STS task hint.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from greyscope.v2.openrouter import embed  # noqa: E402

LO, HI, N_BUCKETS = 0.03, 0.15, 4
DEFAULT_PER_BUCKET = 100
MODEL = os.environ.get("EMBED_MODEL", "google/gemini-embedding-001")  # any OpenRouter embedder
TASK_TYPE = os.environ.get("GEMINI_TASK_TYPE")  # default: endpoint default
AGREEMENT_BAR = 0.84  # Spearman(cosine, soft_ngrams) measured on the full EN set


def score_to_bucket(score: float) -> int:
    """Mirror of greyscope.preprocess.score_to_bucket (inlined to keep this
    experiment free of the clean_text/emoji import chain)."""
    if score <= LO:
        return 0
    if score >= HI:
        return N_BUCKETS - 1
    return 1 + int((score - LO) / (HI - LO) * (N_BUCKETS - 2))


def load_ai_edited() -> pa.Table:
    base = os.path.expanduser("~/.cache/huggingface/datasets/pangram___editlens_iclr")
    path = sorted(glob.glob(base + "/**/editlens_iclr-train.arrow", recursive=True))[0]
    src = pa.memory_map(path, "r")
    try:
        table = ipc.open_file(src).read_all()
    except Exception:
        src.seek(0)
        table = ipc.open_stream(src).read_all()
    return table.filter(pc.equal(table.column("text_type"), "ai_edited"))


def stratified_indices(buckets: np.ndarray, per_bucket: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for b in range(N_BUCKETS):
        idx = np.where(buckets == b)[0]
        rng.shuffle(idx)
        picked.extend(idx[:per_bucket].tolist())
    return sorted(picked)


def cosine_distance(a: list, b: list) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - np.sum(a * b, axis=1)


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x))
    return ranks


def main() -> None:
    per_bucket = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PER_BUCKET
    table = load_ai_edited()
    linq = np.array(table.column("cosine_score").to_pylist(), float)
    sources = table.column("source_text").to_pylist()
    edits = table.column("text").to_pylist()
    buckets_all = np.array([score_to_bucket(s) for s in linq])

    idx = stratified_indices(buckets_all, per_bucket)
    print(f"model={MODEL}  sampled {len(idx)} ai_edited pairs ({per_bucket}/bucket); "
          f"task_type={TASK_TYPE or 'default'}; {2 * len(idx)} embeddings")

    src_emb = embed([sources[i] for i in idx], model=MODEL, task_type=TASK_TYPE)
    edit_emb = embed([edits[i] for i in idx], model=MODEL, task_type=TASK_TYPE)
    gem = cosine_distance(src_emb, edit_emb)
    linq_s = linq[idx]
    bk = buckets_all[idx]

    rho, _ = spearmanr(gem, linq_s)
    print(f"\nSpearman(gemini 1-cos, Linq cosine) = {rho:.3f}   (bar >= {AGREEMENT_BAR})")

    print("\ngemini score by Linq bucket (want monotonic increase):")
    means = []
    for b in range(N_BUCKETS):
        m = gem[bk == b]
        mean = float(m.mean()) if len(m) else float("nan")
        means.append(mean)
        sd = float(m.std()) if len(m) else float("nan")
        print(f"  bucket {b}: n={len(m):4d}  gemini mean={mean:.4f}  sd={sd:.4f}")
    monotonic = all(
        means[i] < means[i + 1]
        for i in range(N_BUCKETS - 1)
        if not (np.isnan(means[i]) or np.isnan(means[i + 1]))
    )
    print(f"  monotonic increasing: {monotonic}")

    diff = np.abs(rankdata(gem) - rankdata(linq_s))
    print("\n=== top disagreements (eyeball: which score matches the real edit size?) ===")
    for j in np.argsort(-diff)[:8]:
        i = idx[j]
        print(f"\nLinq={linq_s[j]:.3f} (b{bk[j]})  gemini={gem[j]:.3f}")
        print(f"  SOURCE: {sources[i][:160]!r}")
        print(f"  EDITED: {edits[i][:160]!r}")

    verdict = "MIGRATE" if (rho >= AGREEMENT_BAR and monotonic) else "INVESTIGATE"
    print(f"\nVERDICT: {verdict}  (rho={rho:.3f}, monotonic={monotonic})")


if __name__ == "__main__":
    main()
