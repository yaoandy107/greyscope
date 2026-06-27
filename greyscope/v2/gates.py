"""Quality gates for generated/edited samples.

Each per-sample gate returns keep/drop + reason; `run_gates` applies them, then a
within-`text_type` near-dedup, logging every drop with its reason. Gates are
**asymmetric**: human rows are real and bypass the AI-output checks; length-match +
regurgitation apply only to MIRRORS (an edit is *meant* to track its source).
**Formatting is NOT gated** — it is signal, not noise.

The zh-TW script check uses a lightweight unambiguous-Simplified-character ratio —
enough to catch the "wholesale wrong variety" this gate drops and to report the rate
the build needs. The finer *lexical* mainland-vocab MONITOR (OpenCC s2twp) is a
full-build addition — deferred until that feature exists (the build needs
the character rate, not the word-choice lexicon).
"""

from __future__ import annotations

from dataclasses import dataclass

from greyscope.preprocess import count_words
from greyscope.v2.corpora import count_cjk_chars

# Unambiguous Simplified-only characters (their Traditional form differs AND the
# Simplified glyph is not itself common Traditional) — a high-signal subset, not a
# full table. Wholesale-Simplified zh-TW output is saturated with these.
_SIMPLIFIED_ONLY = frozenset(
    "们这国说时会个学发经应样见关门问还实现对长书东车马鸟鱼龙头买卖区医华团图块坏备够妈"
    "来过进远爱难钱银铁错间阳阴队际边达运选适网罗习乐飞风馆饭饮验体历厂听单严临为产亲从"
    "价众优传伤软络资讯计划轮铜钢铝镇页顺颗领题骨鸡鸭鹅"
)
SIMPLIFIED_DROP_RATIO = 0.05  # >5% Simplified-only chars ⇒ wrong variety (drop)

# Length-match band for a mirror vs its source: generous — the mirror is
# length-guided, not length-clamped.
LENGTH_MIN_RATIO, LENGTH_MAX_RATIO = 0.33, 3.0

# Regurgitation: a shared exact n-gram this long ⇒ pretraining-memorized copy.
_REGURGITATION_N = {"en": 15, "ja": 30, "zh-tw": 30}  # words (EN) / chars (CJK)
# Near-dedup within text_type: shingle size + Jaccard cutoff.
_DEDUP_N = {"en": 5, "ja": 10, "zh-tw": 10}
DEDUP_THRESHOLD = 0.8

# A refusal = the model declining the task (drop the row). VERB-anchored so positive content
# ("I can't recommend it enough", "我慢できません", "抱歉這麼晚才推薦") is NOT a false hit.
# Leading chat-wrappers are stripped at row shaping (generate._strip_ai_header), not gated here.
_REFUSALS = (
    "i can't help", "i cannot help", "i can't assist", "i cannot assist",
    "i can't provide", "i cannot provide", "i can't comply", "i cannot comply",
    "i can't fulfill", "i cannot fulfill", "i won't be able to",
    "i'm not able to", "i am not able to", "i'm unable to", "i am unable to", "as an ai",
    "申し訳ありませんが", "申し訳ございませんが", "お手伝いできません", "対応できません",
    "我無法協助", "我無法提供", "我不能提供", "抱歉，我無法", "对不起，我无法",
)


@dataclass
class GateResult:
    keep: bool
    reason: str | None = None


_KEEP = GateResult(True)


def _is_ai(row: dict) -> bool:
    return row["text_type"] in ("ai_generated", "ai_edited")


def _length(text: str, language: str) -> int:
    return count_words(text) if language == "en" else count_cjk_chars(text)


def _units(text: str, language: str) -> list[str]:
    return text.split() if language == "en" else list(text)


def _shingles(text: str, language: str, n: int) -> set[str]:
    units = _units(text, language)
    return {" ".join(units[i:i + n]) for i in range(len(units) - n + 1)}


# --- per-sample gates -------------------------------------------------------
def gate_nonempty(row: dict) -> GateResult:
    if not _is_ai(row):
        return _KEEP
    return _KEEP if row["text"].strip() else GateResult(False, "empty")


def gate_truncated(row: dict) -> GateResult:
    if _is_ai(row) and row["meta"].get("finish_reason") == "length":
        return GateResult(False, "truncated")
    return _KEEP


def gate_refusal(row: dict) -> GateResult:
    if not _is_ai(row):
        return _KEEP
    head = row["text"][:200].lower()
    return GateResult(False, "refusal") if any(p in head for p in _REFUSALS) else _KEEP


def simplified_ratio(text: str) -> float:
    cjk = count_cjk_chars(text)
    if not cjk:
        return 0.0
    return sum(ch in _SIMPLIFIED_ONLY for ch in text) / cjk


def gate_script(row: dict) -> GateResult:
    """zh-TW AI output in wholesale-Simplified is the wrong variety → drop. Human
    zh-TW is kept as-is (natural Simplified mixing happens)."""
    if not (_is_ai(row) and row["language"] == "zh-tw"):
        return _KEEP
    if simplified_ratio(row["text"]) > SIMPLIFIED_DROP_RATIO:
        return GateResult(False, "wrong_script_simplified")
    return _KEEP


def gate_length_match(row: dict) -> GateResult:
    """Mirror only: its length must sit within a band of the source (an edit may
    legitimately expand/condense, so it is exempt)."""
    if row["text_type"] != "ai_generated" or not row.get("source_text"):
        return _KEEP
    src = _length(row["source_text"], row["language"])
    if src == 0:
        return _KEEP
    ratio = _length(row["text"], row["language"]) / src
    if ratio < LENGTH_MIN_RATIO or ratio > LENGTH_MAX_RATIO:
        return GateResult(False, "length_mismatch")
    return _KEEP


def gate_regurgitation(row: dict) -> GateResult:
    """Mirror only: a long EXACT n-gram shared with the source is a memorized copy
    that reads as human. Distinct from near-dedup; topical overlap is
    fine. An edit derives from its source by design, so it is exempt."""
    if row["text_type"] != "ai_generated" or not row.get("source_text"):
        return _KEEP
    n = _REGURGITATION_N[row["language"]]
    src = _shingles(row["source_text"], row["language"], n)
    if src and src & _shingles(row["text"], row["language"], n):
        return GateResult(False, "regurgitation")
    return _KEEP


def gate_noop_edit(row: dict) -> GateResult:
    """Edit only: an edit that returned the source verbatim did no editing — a wasted
    generation that would land as a bucket-0 `ai_edited` twin of the very human row it
    derives from. Drop it (that text already ships as the human row, at bucket 0)."""
    if row["text_type"] != "ai_edited" or not row.get("source_text"):
        return _KEEP
    if row["text"].strip() == row["source_text"].strip():
        return GateResult(False, "noop_edit")
    return _KEEP


PER_SAMPLE_GATES = (
    gate_nonempty, gate_truncated, gate_refusal,
    gate_script, gate_length_match, gate_regurgitation, gate_noop_edit,
)


# --- pool-level near-dedup (within text_type) -------------------------------
def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def near_dedup(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop near-duplicates WITHIN (language, text_type) — never across types, which
    would flag the deliberate human↔edited near-paraphrase pairs. Keeps
    the first occurrence; greedy shingle-Jaccard (fine at this scale). Texts too short
    to shingle fall back to exact-match (else two identical short outputs both survive)."""
    kept: list[dict] = []
    dropped: list[dict] = []
    seen_shingles: dict[tuple, list[set]] = {}
    seen_exact: dict[tuple, set[str]] = {}
    for row in rows:
        key = (row["language"], row["text_type"])
        text = row["text"].strip()
        shingles = _shingles(text, row["language"], _DEDUP_N[row["language"]])
        if shingles:
            duplicate = any(_jaccard(shingles, prev) >= DEDUP_THRESHOLD for prev in seen_shingles.get(key, []))
        else:
            duplicate = text in seen_exact.get(key, set())
        if duplicate:
            dropped.append({**row, "drop_reason": "near_duplicate"})
        else:
            seen_shingles.setdefault(key, []).append(shingles)
            seen_exact.setdefault(key, set()).add(text)
            kept.append(row)
    return kept, dropped


def run_gates(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Apply per-sample gates then near-dedup. Returns (kept, dropped); each dropped
    row carries a `drop_reason` for `drops.jsonl` + the build's gate-firing report."""
    survivors: list[dict] = []
    dropped: list[dict] = []
    for row in rows:
        for gate in PER_SAMPLE_GATES:
            result = gate(row)
            if not result.keep:
                dropped.append({**row, "drop_reason": result.reason})
                break
        else:
            survivors.append(row)
    kept, dedup_dropped = near_dedup(survivors)
    return kept, dropped + dedup_dropped
