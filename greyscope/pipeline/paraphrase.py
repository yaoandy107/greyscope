"""Paraphrase-invariance augmentation + the paraphrase-attack eval slice.

The one measured capability gap (RAID: paraphrase 0.57 / synonym 0.63 AUROC) is semantic
rewriting — paraphrased AI reads as human. The literature fix (DAMAGE arXiv:2501.03437,
RADAR, Macko arXiv:2503.15128) is invariance data-aug: paraphrase existing AI rows once
with a STRONG paraphraser and keep the ORIGINAL label (weak paraphrasers RAISE FPR).

Two artifacts, split-safe by construction:
- `train_aug_paraphrase.csv` — train AI rows paraphrased by AUG_MODEL; training-only extra
  file (never assembled into the shipped dataset).
- `attack_paraphrase.csv` — val/test AI rows paraphrased by ATTACK_MODEL, a DIFFERENT
  family, so the eval measures transfer to an unseen attacker, not memorization.

Selection/prompt/row logic is pure and seeded (unit-tested, cache-stable); only `run()`
touches the network via the cached openrouter client.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from greyscope.preprocess import score_to_bucket
from greyscope.pipeline import openrouter, score
from greyscope.pipeline.assemble import BUCKET_CUTS, N_BUCKETS, SPLITS_DIR, _COLUMNS
from greyscope.pipeline.gates import SIMPLIFIED_DROP_RATIO, simplified_ratio
from greyscope.pipeline.generate import (
    GEN_CONCURRENCY, MAX_COMPLETION_TOKENS, _seeded_index, _strip_ai_header, _strip_think,
)

# Strong paraphrasers only. AUG = train-side pool (seeded per-row pick → style diversity);
# ATTACK = eval-side, a family OUTSIDE the aug pool so the slice measures transfer.
AUG_MODELS = [
    {"slug": "google/gemini-3-flash-preview", "flex": True, "reasoning": {"effort": "minimal"}},
    {"slug": "openai/gpt-5.6-luna", "flex": True, "reasoning": {"effort": "minimal"}},
    {"slug": "x-ai/grok-4.3", "flex": False, "reasoning": {"enabled": False}},
]
ATTACK_MODELS = [
    {"slug": "moonshotai/kimi-k2.6", "flex": False, "reasoning": {"enabled": False}},
]

_PROMPT = {
    "en": ("Rewrite the following text completely in your own words. Preserve the meaning, "
           "tone, and approximate length, but change the wording and sentence structure "
           "throughout. Output only the rewritten text — no preamble, no commentary."),
    "ja": ("次の文章を、意味・トーン・長さを保ったまま、語彙と文の構造を全面的に変えて書き直して"
           "ください。書き直した本文だけを出力し、前置きや説明は書かないでください。"),
    "zh-tw": ("請將下面的文字完全改寫：保留原意、語氣與大致篇幅，但全面改變用詞與句子結構。"
              "請只輸出改寫後的正文，不要加任何開場白或說明。"),
}

_AI_TYPES = ("ai_generated", "ai_edited")
# Keep-gate: an empty/echoed reply or a wild length change is a failed paraphrase, not data.
_LEN_RATIO = (0.5, 2.0)


# --- split-CSV I/O ------------------------------------------------------------
def read_split(name: str, splits_dir: Path = SPLITS_DIR) -> list[dict]:
    with (splits_dir / f"{name}.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["meta"] = json.loads(row.get("meta") or "{}")
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "meta": json.dumps(row["meta"], ensure_ascii=False)})


# --- selection (pure, seeded) --------------------------------------------------
def select_ai_rows(rows: list[dict], n_per_language: int) -> list[dict]:
    """Seeded stratified pick of AI rows: per language, round-robin across
    (text_type, bucket) cells so the slice covers the graded range, not just mirrors."""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["text_type"] in _AI_TYPES and row.get("text"):
            cells[(row["language"], row["text_type"], row["bucket"])].append(row)
    for cell in cells.values():
        cell.sort(key=lambda r: hashlib.sha256(r["text_id"].encode()).hexdigest())
    picked: list[dict] = []
    for lang in sorted({key[0] for key in cells}):
        lang_cells = [cells[k] for k in sorted(cells, key=str) if k[0] == lang]
        got: list[dict] = []
        i = 0
        while len(got) < n_per_language and any(lang_cells):
            cell = lang_cells[i % len(lang_cells)]
            if cell:
                got.append(cell.pop())
            i += 1
        picked += got
    return picked


def model_for(row: dict, models: list[dict]) -> dict:
    """Seeded per-row pick from the pool (cache-stable across re-runs)."""
    return models[_seeded_index(models, row["text_id"], "para")]


def sample_humans(split_name: str, n_per_language: int,
                  splits_dir: Path = SPLITS_DIR) -> list[dict]:
    """Seeded, per-language-balanced human negatives from a split. An attack slice needs BOTH
    classes: detection AUROC / TPR@FPR are undefined on attacked-AI alone (single class), and
    the question — at a threshold that keeps humans safe, how many paraphrased-AI slip through —
    is only measurable against real human negatives."""
    humans = [r for r in read_split(split_name, splits_dir) if r["text_type"] == "human_written"]
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for row in humans:
        by_lang[row["language"]].append(row)
    picked: list[dict] = []
    for lang in sorted(by_lang):
        rows = sorted(by_lang[lang], key=lambda r: hashlib.sha256(r["text_id"].encode()).hexdigest())
        picked += rows[:n_per_language]
    for row in picked:  # CSV-read rows carry strings; match the int schema of the AI rows
        row["bucket"] = int(row["bucket"])
        row["cosine_score"] = float(row["cosine_score"]) if row.get("cosine_score") else None
    return picked


# --- request/row shaping (pure) -------------------------------------------------
def build_messages(row: dict) -> list[dict]:
    return [{"role": "system", "content": _PROMPT[row["language"]]},
            {"role": "user", "content": row["text"]}]


def paraphrase_row(row: dict, paraphraser: dict, text: str) -> dict | None:
    """Shape the augmented row: original label kept (invariance), provenance in meta.
    None = failed paraphrase (empty, echo, or out-of-band length)."""
    clean, _ = _strip_ai_header(_strip_think(text), row["language"])
    clean = clean.strip()
    ratio = len(clean) / max(1, len(row["text"]))
    if not clean or clean == row["text"] or not (_LEN_RATIO[0] <= ratio <= _LEN_RATIO[1]):
        return None
    if row["language"] == "zh-tw" and simplified_ratio(clean) > SIMPLIFIED_DROP_RATIO:
        return None  # wholesale-Simplified output is the wrong variety for zh-TW
    return {**{k: row[k] for k in _COLUMNS if k != "meta"},
            "text_id": f"{row['text_id']}/para",
            "text": clean,
            "bucket": int(row["bucket"]),  # CSV-read rows carry strings
            "cosine_score": float(row["cosine_score"]) if row.get("cosine_score") else None,
            "meta": {**row["meta"], "augmentation": "paraphrase",
                     "paraphrased_by": paraphraser["slug"]}}


def human_sources(*split_names, splits_dir: Path = SPLITS_DIR) -> dict[tuple, str]:
    """(language, source, source_id) → human text, for re-scoring paraphrased edits.
    Derivatives co-locate with their human doc (assemble splits by source doc), so
    reading the same splits the rows came from always finds the source."""
    sources: dict[tuple, str] = {}
    for name in split_names:
        for row in read_split(name, splits_dir):
            if row["text_type"] == "human_written":
                sources[(row["language"], row["source"], str(row["source_id"]))] = row["text"]
    return sources


def rescore_edited(rows: list[dict], sources: dict[tuple, str], *,
                   embed_fn=openrouter.embed) -> list[dict]:
    """Honest buckets for paraphrased ai_edited rows. Keep-original-label is only valid for
    bucket-3 rows (fully-AI stays fully-AI under rewording); a paraphrase ON TOP of a light
    edit is a bigger edit, so re-score it as one: cosine vs the ORIGINAL human doc,
    re-bucketed with the standard per-language cuts. ai_generated rows stay bucket 3 by
    class. Edited rows whose human source can't be found are dropped (label unverifiable)."""
    kept: list[dict] = []
    for row in rows:
        if row["text_type"] == "ai_edited":
            src = sources.get((row["language"], row["source"], str(row["source_id"])))
            if src is None:
                continue
            row["source_text"] = src
        kept.append(row)
    score.score_edited(kept, embed_fn=embed_fn)
    for row in kept:
        if row.pop("source_text", None) is not None:
            lo, hi = BUCKET_CUTS[row["language"]]
            row["bucket"] = score_to_bucket(float(row["cosine_score"]), N_BUCKETS, lo, hi)
    return kept


# --- cost estimate (pure; list-price, chars→tokens heuristic) --------------------
def estimate_cost(rows: list[dict], models: list[dict], prices: dict) -> dict:
    """List-price estimate with the seeded per-row model assignment: tokens ≈ chars/4 (EN)
    or chars/1.5 (CJK), output ≈ input body, flex halved. Actual `usage.cost` is the authority."""
    total = {"rows": len(rows), "tokens_in": 0, "tokens_out": 0, "cost": 0.0}
    for row in rows:
        model = model_for(row, models)
        pin, pout = prices.get(model["slug"], (0.0, 0.0))
        body = len(row["text"]) / (4.0 if row["language"] == "en" else 1.5)
        tok_in = body + len(_PROMPT[row["language"]]) / 3.0
        cost = tok_in * pin + body * pout
        total["tokens_in"] += int(tok_in)
        total["tokens_out"] += int(body)
        total["cost"] += cost * (0.5 if model["flex"] else 1.0)
    return total


# --- networked run ---------------------------------------------------------------
def run(rows: list[dict], models: list[dict], out_path: Path) -> tuple[list[dict], float]:
    """Paraphrase `rows` (cached → resumable), write the kept rows, return (kept, cost)."""
    def one(row: dict) -> tuple[dict | None, float]:
        paraphraser = model_for(row, models)
        result = openrouter.chat(
            build_messages(row),
            model=paraphraser["slug"],
            service_tier="flex" if paraphraser["flex"] else None,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            extra={"reasoning": paraphraser["reasoning"]},
        )
        return paraphrase_row(row, paraphraser, result.text), openrouter.cost_of(result.usage)

    kept: list[dict] = []
    cost = 0.0
    dropped = 0
    with ThreadPoolExecutor(max_workers=GEN_CONCURRENCY) as pool:
        futures = [pool.submit(one, row) for row in rows]
        for fut in as_completed(futures):
            shaped, row_cost = fut.result()
            cost += row_cost
            if shaped is None:
                dropped += 1
            else:
                kept.append(shaped)
    kept.sort(key=lambda r: r["text_id"])  # deterministic file order across re-runs
    write_rows(kept, out_path)
    print(f"  kept {len(kept)} / dropped {dropped} → {out_path} (${cost:.2f})")
    return kept, cost
