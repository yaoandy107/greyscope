"""Human-corpus loaders for the v2 trilingual build (design §4, plan §5).

One function per source → normalized `HumanRecord`s (`text_type=human_written`) with a
**stable `source_id`** for the rebuild-from-IDs release (design §13) and `meta` for the
mirror/edit prompts.

**Source-artifact normalization runs HERE, before anything else** (design §8.7): each
corpus carries markup the AI side never produces — wiki40b structural tokens, Aozora
ruby/gaiji — so if it leaked, "human" would be trivially separable from the AI mirror.
This is source-specific and lives in the loader; `preprocess.clean_text` is the later,
source-agnostic pass.

The canonical record is a superset of EditLens's columns (text_id / text / text_type /
model / source / source_id / source_text + cosine_score), plus the v2 fields `language`,
`register`, `markdown_mode`, `bucket`, and a `meta.text_register` for per-register FPR
eval (PTT & EditLens are multi-register, so `source` alone no longer fixes register).
"""

from __future__ import annotations

import ast
import glob
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from greyscope.preprocess import count_words

# --- register taxonomy (design §4 matrix) -----------------------------------
CASUAL, REVIEWS, CREATIVE, FORMAL, JOURNALISTIC = (
    "casual", "reviews", "creative", "formal", "journalistic")

# Per-source length floors (design §10): EN by words, CJK by characters, since
# `count_words` (\b\w+\b) matches ~no CJK.
EN_WORD_FLOOR = 75
CJK_CHAR_FLOOR = 150

HUMAN_DIR = Path("data/v2/human")
_EDITLENS_CACHE = "~/.cache/huggingface/datasets/pangram___editlens_iclr"

# EditLens sub-source → register (verified against the cached arrow, 2026-06-15).
_EDITLENS_REGISTER = {
    "reddit_writing_prompts": CREATIVE,
    "news": JOURNALISTIC,
    "fineweb_edu": FORMAL,
    "amazon_reviews": REVIEWS,
    "google_reviews": REVIEWS,
}

# HF dataset coordinates per source (path, config). Kept here so a mirror swap is
# a one-line change.
_WIKI40B = {"ja": ("google/wiki40b", "ja"), "zh-tw": ("google/wiki40b", "zh-tw")}
_MARC_JA = ("shunk031/JGLUE", "MARC-ja")
_OPEN2CH = ("p1atdev/open2ch", "all-corpus")

# PTT board → register (design §4: PTT is one multi-register source).
PTT_BOARDS = {
    "Gossiping": CASUAL,
    "Food": REVIEWS,
    "MobileComm": REVIEWS,
    "marvel": CREATIVE,
    "eWriter": CREATIVE,
}


# --- canonical record --------------------------------------------------------
@dataclass
class HumanRecord:
    """One human document. `to_row()` emits the canonical pipeline schema."""

    text: str
    language: str  # "en" | "ja" | "zh-tw"
    source: str  # "editlens" | "wiki40b-ja" | "ptt" | ...
    source_id: str  # stable addressable id for rebuild (design §13)
    text_register: str  # casual | reviews | creative | formal | journalistic
    meta: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "text_id": f"{self.language}/{self.source}/{self.source_id}/human_written",
            "text": self.text,
            "language": self.language,
            "text_type": "human_written",
            "source": self.source,
            "source_id": self.source_id,
            "source_text": None,  # human IS the source; mirror/edit fill this later
            "model": None,
            "prompt_id": None,  # generation style id (design §6); n/a for human
            "markdown_mode": None,
            "cosine_score": 0.0,  # human = 0 by class (design §7)
            "bucket": 0,
            "meta": {**self.meta, "text_register": self.text_register},
        }


# --- source-artifact normalization (design §8.7) -----------------------------
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ]")
_AOZORA_RUBY = re.compile(r"《[^》]*》")  # furigana readings
_AOZORA_NOTE = re.compile(r"［＃[^］]*］")  # input notes / gaiji directives
_WIKI40B_HAS_MARKERS = "_START_"
_WIKI40B_SPLIT = re.compile(r"_START_ARTICLE_|_START_SECTION_|_START_PARAGRAPH_")


def count_cjk_chars(text: str) -> int:
    return len(_CJK_RE.findall(text))


def passes_floor(text: str, language: str) -> bool:
    if language == "en":
        return count_words(text) >= EN_WORD_FLOOR
    return count_cjk_chars(text) >= CJK_CHAR_FLOOR


def parse_wiki40b(text: str) -> tuple[str, str]:
    """Strip wiki40b structural tokens → (title, body). Article title feeds the
    mirror prompt's topic slot; `_NEWLINE_` becomes a real newline."""
    text = text.replace("_NEWLINE_", "\n")
    if _WIKI40B_HAS_MARKERS not in text:
        return "", text.strip()
    segments = [s.strip() for s in _WIKI40B_SPLIT.split(text) if s.strip()]
    if not segments:
        return "", ""
    title = segments[0]
    body = "\n".join(segments[1:]) if len(segments) > 1 else segments[0]
    return title, body


def strip_aozora_markup(text: str) -> str:
    """Remove Aozora ruby (《…》), the ｜ ruby-base delimiter, and ［＃…］ input
    notes/gaiji directives. (Deep 旧仮名/旧字体 normalization is a later, optional
    pass; the markup strip is the load-bearing anti-confound step.)"""
    text = _AOZORA_RUBY.sub("", text)
    text = text.replace("｜", "")
    text = _AOZORA_NOTE.sub("", text)
    return text


# --- loaders -----------------------------------------------------------------
def _read_editlens_arrow(split: str):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    base = os.path.expanduser(_EDITLENS_CACHE)
    matches = sorted(glob.glob(f"{base}/**/editlens_iclr-{split}.arrow", recursive=True))
    if not matches:
        raise FileNotFoundError(
            f"editlens {split} arrow not found under {base} — load it once via "
            "`datasets.load_dataset('pangram/editlens_iclr')`."
        )
    src = pa.memory_map(matches[0], "r")
    try:
        return ipc.open_file(src).read_all()
    except Exception:
        src.seek(0)
        return ipc.open_stream(src).read_all()


def load_editlens(*, split: str = "train", limit: int | None = None) -> list[HumanRecord]:
    """EN backbone (reused, free). Human rows from the cached EditLens arrow; the
    sub-`source` maps to register (design §4 matrix)."""
    import pyarrow.compute as pc

    table = _read_editlens_arrow(split)
    table = table.filter(pc.equal(table.column("text_type"), "human_written"))
    texts = table.column("text").to_pylist()
    sources = table.column("source").to_pylist()
    source_ids = table.column("source_id").to_pylist()

    out: list[HumanRecord] = []
    for text, sub_source, sid in zip(texts, sources, source_ids):
        if not passes_floor(text, "en"):
            continue
        out.append(HumanRecord(
            text=text,
            language="en",
            source="editlens",
            source_id=str(sid),
            text_register=_EDITLENS_REGISTER.get(sub_source, "mixed"),
            meta={"editlens_source": sub_source, "length": len(text)},
        ))
        if limit is not None and len(out) >= limit:
            break
    return out


def _iter_hf(path: str, config: str | None, split: str) -> Iterator[dict]:
    # datasets >=4 dropped loading scripts → parquet/data-only sources only.
    from datasets import load_dataset

    yield from load_dataset(path, config, split=split, streaming=True)


def _as_str(value) -> str:
    """Decode corpus fields to text. google/wiki40b's parquet conversion stored the
    original `bytes` as their str() repr (e.g. `'b"\\n...\\xe6\\x97\\xa5"'`), so the
    CJK arrives as literal \\xNN escape-text — un-repr it back to bytes, then decode."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, str) and value[:2] in ("b'", 'b"'):
        try:
            return ast.literal_eval(value).decode("utf-8", "replace")
        except (ValueError, SyntaxError):
            return value
    return value or ""


def load_wiki40b(language: str, *, limit: int | None = None) -> list[HumanRecord]:
    """Formal-factual ja / zh-TW (pre-2020 snapshot). Strips wiki40b markers; pins
    the revision via `version_id` (design §4 date-pin, §13 rebuild)."""
    path, config = _WIKI40B[language]
    source = f"wiki40b-{language}"
    out: list[HumanRecord] = []
    for ex in _iter_hf(path, config, "train"):
        title, body = parse_wiki40b(_as_str(ex["text"]))
        if not passes_floor(body, language):
            continue
        wid = _as_str(ex.get("wikidata_id"))
        version = _as_str(ex.get("version_id"))
        out.append(HumanRecord(
            text=body,
            language=language,
            source=source,
            source_id=f"{wid}@{version}",
            text_register=FORMAL,
            meta={"topic": title, "length": len(body)},
        ))
        if limit is not None and len(out) >= limit:
            break
    return out


def load_marc_ja(*, limit: int | None = None) -> list[HumanRecord]:
    """ja reviews (templated), a minor slice (design §4). NOTE: `shunk031/JGLUE` is a
    loading SCRIPT → dead under datasets≥4; pending a parquet ja-reviews mirror. Not
    used in the pilot (wiki40b + open2ch already give ja its ≥2 registers)."""
    out: list[HumanRecord] = []
    for i, ex in enumerate(_iter_hf(*_MARC_JA, "train")):
        text = ex.get("sentence") or ex.get("text") or ""
        if not passes_floor(text, "ja"):
            continue
        out.append(HumanRecord(
            text=text,
            language="ja",
            source="marc-ja",
            source_id=str(ex.get("idx", i)),
            text_register=REVIEWS,
            meta={"rating": ex.get("label"), "length": len(text)},
        ))
        if limit is not None and len(out) >= limit:
            break
    return out


def load_open2ch(*, limit: int | None = None) -> list[HumanRecord]:
    """ja casual/forum (Apache-2.0) — the ja analog to PTT, pilot go/no-go on
    yield + naturalness (design §4). 2ch-derived → short turns; the 150-char floor
    drops most, so a thread's turns are joined into a passage before the floor."""
    out: list[HumanRecord] = []
    for i, ex in enumerate(_iter_hf(*_OPEN2CH, "train")):
        text = _open2ch_passage(ex)
        if not passes_floor(text, "ja"):
            continue
        out.append(HumanRecord(
            text=text,
            language="ja",
            source="open2ch",
            source_id=str(ex.get("id", i)),
            text_register=CASUAL,
            meta={"board": ex.get("board"), "length": len(text)},
        ))
        if limit is not None and len(out) >= limit:
            break
    return out


def _open2ch_passage(ex: dict) -> str:
    """Join a thread's turns into one passage. open2ch rows are
    `dialogue={speaker:[...], content:[...]}` — typically just 2 short turns, so most
    fall under the 150-char floor (the pilot go/no-go signal, design §4)."""
    dialogue = ex.get("dialogue")
    if isinstance(dialogue, dict) and isinstance(dialogue.get("content"), list):
        return "\n".join(c for c in dialogue["content"] if c)
    return ex.get("text") or ex.get("body") or ""


def load_ptt(
    boards: dict[str, str] | None = None,
    *,
    limit_per_board: int = 50,
    year_range: tuple[int, int] = (2015, 2019),
) -> list[HumanRecord]:
    """zh-TW casual + reviews + creative from PTT (design §4). One source, one
    license/ID story across boards; pre-2022 only. Lazy-imports the scraper so the
    EN/HF loaders stay free of httpx/bs4."""
    from greyscope.v2 import ptt

    boards = boards or PTT_BOARDS
    out: list[HumanRecord] = []
    for board, register in boards.items():
        for art in ptt.fetch_board(board, year_range=year_range, limit=limit_per_board):
            if not passes_floor(art["body"], "zh-tw"):
                continue
            out.append(HumanRecord(
                text=art["body"],
                language="zh-tw",
                source="ptt",
                source_id=f"{board}/{art['post_id']}",
                text_register=register,
                meta={"board": board, "topic": art["title"], "length": len(art["body"])},
            ))
    return out


def write_jsonl(records: list[HumanRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_row(), ensure_ascii=False) + "\n")


if __name__ == "__main__":  # zero-dep smoke: the EN loader runs off the cached arrow
    rows = load_editlens(limit=5)
    print(f"load_editlens → {len(rows)} records")
    for r in rows[:3]:
        row = r.to_row()
        print(f"  [{row['meta']['text_register']:<11}] {row['text_id']}")
        print(f"    {row['text'][:90]!r}")
