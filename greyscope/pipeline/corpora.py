"""Human-corpus loaders for the trilingual build.

One function per source → normalized `HumanRecord`s (`text_type=human_written`) with a
**stable `source_id`** for the rebuild-from-IDs release and `meta` for the
mirror/edit prompts.

**Source-artifact normalization runs HERE, before anything else**: each
corpus carries markup the AI side never produces — wiki40b structural tokens, Aozora
ruby/gaiji — so if it leaked, "human" would be trivially separable from the AI mirror.
This is source-specific and lives in the loader; `preprocess.clean_text` is the later,
source-agnostic pass.

The canonical record is a superset of EditLens's columns (text_id / text / text_type /
model / source / source_id / source_text + cosine_score), plus the fields `language`,
`register`, `markdown_mode`, `bucket`, and a `meta.text_register` for per-register FPR
eval (PTT & EditLens are multi-register, so `source` alone no longer fixes register).
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from greyscope.preprocess import count_words

# --- register taxonomy -------------------------------------------------------
CASUAL, REVIEWS, CREATIVE, FORMAL, JOURNALISTIC = (
    "casual", "reviews", "creative", "formal", "journalistic")

# Per-source length floors: EN by words, CJK by characters, since
# `count_words` (\b\w+\b) matches ~no CJK.
EN_WORD_FLOOR = 75
CJK_CHAR_FLOOR = 150

HUMAN_DIR = Path("data/v2/human")
_EDITLENS_CACHE = "~/.cache/huggingface/datasets/pangram___editlens_iclr"

# EditLens sub-source → register (verified against the cached arrow).
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
_AMAZON_JA = ("SetFit/amazon_reviews_multi_ja", None)
_OPEN2CH = ("p1atdev/open2ch", "all-corpus")
_AOZORA = ("globis-university/aozorabunko-clean", None)

# Aozora works are whole novels (up to ~840k chars) → chunked into passages. Target is
# comfortably above the 150-char floor (mirror-able length); the per-work cap keeps one
# long novel (or prolific author) from dominating the creative register.
AOZORA_TARGET_CHARS = 450
AOZORA_MAX_PER_WORK = 3

# PTT board → register (PTT is one multi-register source).
PTT_BOARDS = {
    "Gossiping": CASUAL,
    "Food": REVIEWS,
    "MobileComm": REVIEWS,
    "marvel": CREATIVE,
    "eWriter": CREATIVE,
}

# --- EN permissive sources (the EditLens-free backbone: Apache / CC0 / CC BY / PD / pragmatic) ---
# EditLens is CC BY-NC-SA → dropped from TRAINING so the model can ship Apache-2.0 (it stays an EVAL
# set via load_editlens_split). These replace it. Pre-2022 CommonCrawl dumps only → the "human" text
# predates wide LLM contamination (the EN analog of the ja/zh pre-2022 rule); FineWeb exposes each
# dump as a directly-loadable config.
_FINEWEB = "HuggingFaceFW/fineweb"
_FINEWEB_PRE2022_DUMPS = ("CC-MAIN-2019-18", "CC-MAIN-2020-16", "CC-MAIN-2021-17")
_ARXIV_ABSTRACTS = ("common-pile/arxiv_abstracts", None)     # CC0
_WIKINEWS_EN = ("Fumika/Wikinews-multilingual", None)        # CC BY 2.5 (filter lang=="en")
_GUTENBERG = ("sedthh/gutenberg_english", None)              # public domain (residual boilerplate strip)
_STACKEXCHANGE = ("donfu/oa-stackexchange", None)            # CC BY-SA (accepted answers <1000 chars)
_AMAZON_EN_URL = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
                  "resolve/main/raw/review_categories/{cat}.jsonl")  # loader script is v4-dead → raw jsonl
_AMAZON_EN_CATEGORIES = ("Books", "Movies_and_TV", "Kindle_Store")
# Free permissive AI for the ai_generated bucket — NO proprietary-model provenance (both avoid it
# by construction), so no vendor-ToS shadow. Replaces load_editlens_ai's EditLens backbone.
_COSMOPEDIA = ("HuggingFaceTB/cosmopedia", "web_samples_v2")  # Apache-2.0, Mixtral-generated
_HELPSTEER2 = ("nvidia/HelpSteer2", None)                     # CC BY 4.0, NVIDIA in-house models
_FREE_AI = (("cosmopedia", _COSMOPEDIA, "text", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
            ("helpsteer2", _HELPSTEER2, "response", "nvidia/helpsteer2-mix"))


# --- canonical record --------------------------------------------------------
@dataclass
class HumanRecord:
    """One human document. `to_row()` emits the canonical pipeline schema."""

    text: str
    language: str  # "en" | "ja" | "zh-tw"
    source: str  # "editlens" | "wiki40b-ja" | "ptt" | ...
    source_id: str  # stable addressable id for rebuild
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
            "prompt_id": None,  # generation style id; n/a for human
            "markdown_mode": None,
            "cosine_score": 0.0,  # human = 0 by class
            "bucket": 0,
            "meta": {**self.meta, "text_register": self.text_register},
        }


# --- source-artifact normalization -------------------------------------------
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


def chunk_passages(text: str, target_chars: int, max_chunks: int) -> list[str]:
    """Whole-work text → up to `max_chunks` passages of ~`target_chars` CJK each. Accumulates
    non-blank lines (paragraph units) until the target so passages break at line boundaries, not
    mid-sentence; emits a final sub-target passage (short works survive — the floor filters it)."""
    passages: list[str] = []
    buffer: list[str] = []
    chars = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:  # blank line = paragraph separator, not content
            continue
        buffer.append(line)
        chars += count_cjk_chars(line)
        if chars >= target_chars:
            passages.append("\n".join(buffer))
            buffer, chars = [], 0
            if len(passages) >= max_chunks:
                return passages
    if buffer and len(passages) < max_chunks:
        passages.append("\n".join(buffer))
    return passages


# --- EN source-artifact normalization (pure) ---------------------------------
_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", re.I)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Cut the PG license header/footer the AI mirror never produces (anti-confound). sedthh already
    best-effort strips; this removes the residual `*** START/END OF … PROJECT GUTENBERG …***` frame."""
    m = _GUTENBERG_START.search(text)
    if m:
        text = text[m.end():]
    m = _GUTENBERG_END.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


def first_passage_en(text: str, target_words: int = 300, skip_paragraphs: int = 3,
                     max_chars: int = 4000) -> str:
    """One representative passage from a whole book: skip front matter, accumulate paragraphs to
    ~`target_words`. One passage per book (source_id-stable) — 48k books far exceed the row need,
    so multi-chunking a single work (Aozora's problem) is unnecessary here. The hard `max_chars`
    cap is load-bearing: a book with no blank-line paragraph breaks collapses to one giant block, so
    without it a whole 150k-char work would leak through as a single 'passage'."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()][skip_paragraphs:]
    buffer: list[str] = []
    words = 0
    for para in paras:
        buffer.append(para)
        words += len(para.split())
        if words >= target_words:
            break
    return "\n\n".join(buffer)[:max_chars]


def join_paragraphs(value) -> str:
    """Wikinews `text` arrives as a paragraph sequence (list) or a flat string → one body."""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(p) for p in value if p).strip()
    return (value or "").strip()


def parse_blob_meta(raw) -> dict:
    """Best-effort parse of a serialized-dict metadata blob (Gutenberg METADATA): dict → as-is,
    else try JSON then Python-literal, else {}. Never raises (feeds source_id/title, not correctness)."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def _text_hash(text: str) -> str:
    """Stable 16-hex content id for sources that expose no row id (SE, free-AI corpora)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


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
    sub-`source` maps to register."""
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


def load_editlens_split(split: str) -> list[dict]:
    """Ingest an EditLens eval split AS-IS (all 3 classes) for the EN OOD slice:
    `test_llama` = held-out generator, `test_enron` = held-out domain. Keeps EditLens's own
    labels (text_type, cosine_score) and tags `split` so assembly reserves it untouched.
    (cosine is Linq-scale; an optional Qwen re-score would harmonize it — measured ≈ parity.)"""
    table = _read_editlens_arrow(split)
    cols = {c: table.column(c).to_pylist()
            for c in ("text", "text_type", "source", "source_id", "model", "cosine_score")}
    out: list[dict] = []
    for i, text in enumerate(cols["text"]):
        if not passes_floor(text, "en"):
            continue
        text_type = cols["text_type"][i]
        out.append({
            "text_id": f"en/editlens/{cols['source_id'][i]}/{text_type}/{split}",
            "text": text, "language": "en", "text_type": text_type,
            "source": "editlens", "source_id": str(cols["source_id"][i]), "source_text": None,
            "model": None if text_type == "human_written" else cols["model"][i],
            "prompt_id": None, "markdown_mode": None,
            "cosine_score": cols["cosine_score"][i], "bucket": None, "split": split,
            "meta": {"editlens_source": cols["source"][i], "editlens_split": split},
        })
    return out


def load_editlens_ai(*, limit: int | None = None, split: str = "train") -> list[dict]:
    """EN backbone AI (free, cached): EditLens's own ai_generated + ai_edited rows. `load_editlens`
    loads only humans, which leaves EN AI-starved vs the EditLens-trained baseline (its full,
    AI-heavy set); this restores it. `limit=None` takes all (~41k); an int caps to `limit//2` per class.
    ai_generated → cosine 1.0 by class; ai_edited keeps EditLens's cosine (Linq-scale, which the EN cut
    matches). No split tag → assembled into train/val/test by source-doc."""
    table = _read_editlens_arrow(split)
    cols = {c: table.column(c).to_pylist()
            for c in ("text", "text_type", "source", "source_id", "model", "cosine_score")}
    by_type: dict[str, list[dict]] = {"ai_generated": [], "ai_edited": []}
    for i, text in enumerate(cols["text"]):
        text_type = cols["text_type"][i]
        if text_type not in by_type or not passes_floor(text, "en"):
            continue
        by_type[text_type].append({
            "text_id": f"en/editlens/{cols['source_id'][i]}/{text_type}/backbone",
            "text": text, "language": "en", "text_type": text_type,
            "source": "editlens", "source_id": str(cols["source_id"][i]), "source_text": None,
            "model": cols["model"][i], "prompt_id": None, "markdown_mode": None,
            "cosine_score": 1.0 if text_type == "ai_generated" else cols["cosine_score"][i],
            "bucket": None,
            "meta": {"editlens_source": cols["source"][i], "backbone": True},
        })
    if limit is None:
        return by_type["ai_generated"] + by_type["ai_edited"]
    per_type = limit // 2
    return by_type["ai_generated"][:per_type] + by_type["ai_edited"][:per_type]


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
    the revision via `version_id` (date-pin for rebuild)."""
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


def load_amazon_reviews_ja(*, limit: int | None = None) -> list[HumanRecord]:
    """ja reviews (`SetFit/amazon_reviews_multi_ja`, 2015–2019 → pre-2022; replaces the dead
    JGLUE MARC-ja loading script). A minor slice — reviews skew short, so only ~12% clear the
    150-CJK floor, but the 200k corpus still yields plenty. Amazon withdrew MARC redistribution
    → treat as **mirror-only**, NOT edited. `source_id` = the review id."""
    out: list[HumanRecord] = []
    for i, ex in enumerate(_iter_hf(*_AMAZON_JA, "train")):
        text = ex.get("text") or ""
        if not passes_floor(text, "ja"):
            continue
        out.append(HumanRecord(
            text=text,
            language="ja",
            source="amazon-reviews-ja",
            source_id=str(ex.get("id", i)),
            text_register=REVIEWS,
            meta={"rating": ex.get("label"), "length": count_cjk_chars(text)},
        ))
        if limit is not None and len(out) >= limit:
            break
    return out


def load_open2ch(*, limit: int | None = None) -> list[HumanRecord]:
    """ja casual/forum (Apache-2.0) — the ja analog to PTT. 2ch-derived → short turns; the 150-char floor
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
    fall under the 150-char floor."""
    dialogue = ex.get("dialogue")
    if isinstance(dialogue, dict) and isinstance(dialogue.get("content"), list):
        return "\n".join(c for c in dialogue["content"] if c)
    return ex.get("text") or ex.get("body") or ""


def load_aozora(
    *,
    limit: int | None = None,
    target_chars: int = AOZORA_TARGET_CHARS,
    max_per_work: int = AOZORA_MAX_PER_WORK,
) -> list[HumanRecord]:
    """ja creative (Aozora Bunko, public-domain; `globis-university/aozorabunko-clean` is
    CC BY 4.0 → edited-OK). Modern orthography only (新字新仮名 —
    classical 旧字旧仮名 is the wrong distribution for a modern detector). Whole works are
    chunked into passages (the source pre-strips ruby; `strip_aozora_markup` runs as a safety
    no-op). `source_id = {作品ID}#p{n}` is stable for the rebuild-from-IDs release."""
    out: list[HumanRecord] = []
    for ex in _iter_hf(*_AOZORA, "train"):
        meta = ex.get("meta") or {}
        if meta.get("文字遣い種別") != "新字新仮名":
            continue
        work_id = _as_str(meta.get("作品ID"))
        body = strip_aozora_markup(_as_str(ex.get("text")))
        published = meta.get("公開日")
        for i, passage in enumerate(chunk_passages(body, target_chars, max_per_work)):
            if not passes_floor(passage, "ja"):
                continue
            out.append(HumanRecord(
                text=passage,
                language="ja",
                source="aozora",
                source_id=f"{work_id}#p{i}",
                text_register=CREATIVE,
                meta={
                    "topic": _as_str(meta.get("作品名")),
                    "person_id": _as_str(meta.get("人物ID")),
                    "pub_date": str(published)[:10] if published else "",
                    "length": count_cjk_chars(passage),
                },
            ))
            if limit is not None and len(out) >= limit:
                return out
    return out


def load_wikinews_ja(*, limit: int | None = None) -> list[HumanRecord]:
    """ja journalistic (Japanese Wikinews, CC BY 4.0 → edited-OK — the permissive ja
    journalistic source). Pre-2022 articles only; the 【…】 dateline is
    stripped at the scraper (anti-confound). `source_id = {pageid}@{revid}` for the
    rebuild. Lazy-imports the scraper so the EN/HF loaders stay free of httpx/bs4."""
    from greyscope.pipeline import wikinews

    out: list[HumanRecord] = []
    for art in wikinews.fetch_articles(limit=limit):
        if not passes_floor(art["body"], "ja"):
            continue
        out.append(HumanRecord(
            text=art["body"],
            language="ja",
            source="wikinews-ja",
            source_id=f"{art['pageid']}@{art['revid']}",
            text_register=JOURNALISTIC,
            meta={
                "topic": art["title"],
                "pub_date": art["pub_date"],
                "rev_timestamp": art["timestamp"],
                "length": count_cjk_chars(art["body"]),
            },
        ))
    return out


def load_ptt(
    boards: dict[str, str] | None = None,
    *,
    limit_per_board: int = 50,
    year_range: tuple[int, int] = (2015, 2019),
) -> list[HumanRecord]:
    """zh-TW casual + reviews + creative from PTT. One source, one
    license/ID story across boards; pre-2022 only. Lazy-imports the scraper so the
    EN/HF loaders stay free of httpx/bs4."""
    from greyscope.pipeline import ptt

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


def load_twgov(*, limit: int | None = None) -> list[HumanRecord]:
    """zh-TW journalistic (Taipei City gov press-release archive, OGDL/公眾授權 → edited-OK —
    the native Taiwan-Traditional journalistic source). Pre-2022 only (the
    open-data feeds are rolling/current, so this scrapes the dated archive). `source_id` = the
    article id for the rebuild. Lazy-imports the scraper so the HF loaders stay httpx-free."""
    from greyscope.pipeline import twgov

    out: list[HumanRecord] = []
    for art in twgov.fetch_news(limit=limit):
        if not passes_floor(art["body"], "zh-tw"):
            continue
        out.append(HumanRecord(
            text=art["body"],
            language="zh-tw",
            source="tw-gov",
            source_id=art["s"],
            text_register=JOURNALISTIC,
            meta={
                "topic": art["title"],
                "pub_date": art["pub_date"],
                "unit": art["unit"],
                "length": count_cjk_chars(art["body"]),
            },
        ))
    return out


# --- EN permissive human loaders (EditLens-free backbone) --------------------
def load_fineweb_en(*, limit: int | None = None,
                    dumps: tuple[str, ...] = _FINEWEB_PRE2022_DUMPS) -> list[HumanRecord]:
    """EN formal-web (ODC-BY). Pre-2022 CommonCrawl snapshots ONLY (each `dump` is a config) so the
    'human' text predates wide LLM contamination. Round-robins the dumps for temporal spread;
    `source_id` = the CC record id (stable for rebuild)."""
    out: list[HumanRecord] = []
    per_dump = None if limit is None else max(1, limit // len(dumps))
    for dump in dumps:
        got = 0
        for ex in _iter_hf(_FINEWEB, dump, "train"):
            # Full pages run to ~80k chars; cap to a passage so EN human docs stay comparable to the
            # other sources (and never blow the embedder input at edit-scoring time).
            body = (ex.get("text") or "").strip()[:6000]
            if not passes_floor(body, "en"):
                continue
            out.append(HumanRecord(
                text=body, language="en", source="fineweb", source_id=str(ex.get("id")),
                text_register=FORMAL,
                meta={"dump": dump, "url": ex.get("url"), "date": str(ex.get("date"))[:10],
                      "length": len(body)},
            ))
            got += 1
            if limit is not None and len(out) >= limit:
                return out
            if per_dump is not None and got >= per_dump:
                break
    return out


def load_arxiv_abstracts_en(*, limit: int | None = None) -> list[HumanRecord]:
    """EN formal-academic (CC0 — the cleanest license in the set). arXiv abstracts via common-pile
    (parquet-native). `source_id` = the arXiv id."""
    out: list[HumanRecord] = []
    for ex in _iter_hf(*_ARXIV_ABSTRACTS, "train"):
        body = (ex.get("text") or "").strip()
        if not passes_floor(body, "en"):
            continue
        out.append(HumanRecord(
            text=body, language="en", source="arxiv-abstracts", source_id=str(ex.get("id")),
            text_register=FORMAL,
            meta={"created": str(ex.get("created"))[:10], "length": len(body)},
        ))
        if limit is not None and len(out) >= limit:
            break
    return out


def load_wikinews_en(*, limit: int | None = None) -> list[HumanRecord]:
    """EN journalistic (CC BY 2.5 → edited-OK — the permissive EN journalistic source). Fumika's
    multilingual Wikinews, `lang=="en"`; `text` is a paragraph sequence → joined. Pre-LLM vintage
    (guaranteed human). `source_id` = pageid."""
    out: list[HumanRecord] = []
    for ex in _iter_hf(*_WIKINEWS_EN, "train"):
        if ex.get("lang") != "en":
            continue
        body = join_paragraphs(ex.get("text"))
        if not passes_floor(body, "en"):
            continue
        out.append(HumanRecord(
            text=body, language="en", source="wikinews-en", source_id=str(ex.get("pageid")),
            text_register=JOURNALISTIC,
            meta={"topic": ex.get("title"), "date": str(ex.get("date"))[:10], "length": len(body)},
        ))
        if limit is not None and len(out) >= limit:
            break
    return out


def load_gutenberg_en(*, limit: int | None = None, target_words: int = 300) -> list[HumanRecord]:
    """EN creative (public domain → edited-OK). sedthh/gutenberg_english; strips the residual PG
    boilerplate (anti-confound) then takes one passage per book. `source_id` = the parsed Gutenberg
    id (falls back to a content hash). Columns are UPPERCASE (TEXT/METADATA)."""
    out: list[HumanRecord] = []
    for ex in _iter_hf(*_GUTENBERG, "train"):
        meta = parse_blob_meta(ex.get("METADATA"))
        if meta.get("language") not in (None, "en"):  # gutenberg_english is EN, but guard defensively
            continue
        body = strip_gutenberg_boilerplate(_as_str(ex.get("TEXT")))
        passage = first_passage_en(body, target_words)
        if not passes_floor(passage, "en"):
            continue
        gid = meta.get("text_id")  # the Gutenberg book id (stable for the rebuild)
        out.append(HumanRecord(
            text=passage, language="en", source="gutenberg",
            source_id=str(gid) if gid is not None else _text_hash(passage), text_register=CREATIVE,
            meta={"topic": meta.get("title", ""), "issued": str(meta.get("issued"))[:10],
                  "length": len(passage)},
        ))
        if limit is not None and len(out) >= limit:
            break
    return out


def _iter_amazon_en(category: str) -> Iterator[dict]:
    from datasets import load_dataset

    yield from load_dataset("json", data_files=_AMAZON_EN_URL.format(cat=category),
                            split="train", streaming=True)


def load_amazon_reviews_en(*, limit: int | None = None,
                           categories: tuple[str, ...] = _AMAZON_EN_CATEGORIES) -> list[HumanRecord]:
    """EN reviews (Amazon Reviews 2023, McAuley Lab; no redistribution grant → mirror-only, like
    amazon-reviews-ja). Streams the raw per-category JSONL (the HF loader script is v4-dead).
    `source_id` = parent_asin/user_id."""
    out: list[HumanRecord] = []
    per_cat = None if limit is None else max(1, limit // len(categories))
    for category in categories:
        got = 0
        for ex in _iter_amazon_en(category):
            body = (ex.get("text") or "").strip()
            if not passes_floor(body, "en"):
                continue
            out.append(HumanRecord(
                text=body, language="en", source="amazon-reviews-en",
                source_id=f"{ex.get('parent_asin')}/{ex.get('user_id')}", text_register=REVIEWS,
                meta={"rating": ex.get("rating"), "category": category, "length": len(body)},
            ))
            got += 1
            if limit is not None and len(out) >= limit:
                return out
            if per_cat is not None and got >= per_cat:
                break
    return out


def load_stackexchange_en(*, limit: int | None = None) -> list[HumanRecord]:
    """EN casual/Q&A (Stack Exchange, CC BY-SA → mirror-only like the other share-alike sources).
    donfu/oa-stackexchange: accepted answers <1000 chars; the ANSWER (RESPONSE) is the human text.
    `source_id` = a stable content hash (the set exposes no row id). Columns are UPPERCASE."""
    out: list[HumanRecord] = []
    for ex in _iter_hf(*_STACKEXCHANGE, "train"):
        body = (ex.get("RESPONSE") or "").strip()
        if not passes_floor(body, "en"):
            continue
        out.append(HumanRecord(
            text=body, language="en", source="stackexchange", source_id=_text_hash(body),
            text_register=CASUAL,
            meta={"site": ex.get("SOURCE"), "question": (ex.get("INSTRUCTION") or "")[:200],
                  "length": len(body)},
        ))
        if limit is not None and len(out) >= limit:
            break
    return out


def load_reddit_l2_en(*, limit: int | None = 3000) -> list[HumanRecord]:
    """EN casual, NON-NATIVE-SPEAKER (L2 English) — European-flaired r/AskEurope (+ overflow
    r/AskEuropeans / r/europe) authors writing English selftext, pre-2022. A large TRAINABLE pool
    of EditLens's `nonnative_english` calibration phenomenon, so the model sees L2-English humans in
    training too. Reddit is unlicensed → **human + mirror only**, never an edited derivative (like
    PTT). `source_id = {subreddit}/{id}`. Lazy-imports the scraper to keep HF/EN loaders httpx-free."""
    from greyscope.pipeline import redditl2

    out: list[HumanRecord] = []
    for doc in redditl2.fetch_l2_humans(target=limit or 3000):
        out.append(HumanRecord(
            text=doc["body"],
            language="en",
            source="reddit-l2",
            source_id=f"{doc['subreddit']}/{doc['id']}",
            text_register=CASUAL,
            meta={"topic": doc["title"], "country": doc["country"],
                  "length": len(doc["body"])},
        ))
        if limit is not None and len(out) >= limit:
            break
    return out


def load_free_ai_en(*, limit: int | None = None) -> list[dict]:
    """EN ai_generated rows from PERMISSIVE synthetic corpora — Cosmopedia (Apache-2.0, Mixtral) +
    HelpSteer2 (CC BY 4.0, NVIDIA models). Neither carries proprietary-model provenance. These are
    standalone AI text → cosine 1.0 / bucket 3, straight into assembly (bypass gen/gate/score), the
    permissive replacement for `load_editlens_ai`. `limit=None` takes all; an int caps `limit//2`
    per source."""
    out: list[dict] = []
    per_source = None if limit is None else max(1, limit // len(_FREE_AI))
    for source, (path, config), text_col, model in _FREE_AI:
        got = 0
        for ex in _iter_hf(path, config, "train"):
            body = (ex.get(text_col) or "").strip()
            if not passes_floor(body, "en"):
                continue
            sid = _text_hash(body)
            out.append({
                "text_id": f"en/{source}/{sid}/ai_generated/backbone",
                "text": body, "language": "en", "text_type": "ai_generated",
                "source": source, "source_id": sid, "source_text": None,
                "model": model, "prompt_id": None, "markdown_mode": None,
                "cosine_score": 1.0, "bucket": None,
                "meta": {"backbone": True, "free_ai": True, "length": len(body)},
            })
            got += 1
            if limit is not None and len(out) >= limit:
                return out
            if per_source is not None and got >= per_source:
                break
    return out


if __name__ == "__main__":  # zero-dep smoke: the EN loader runs off the cached arrow
    rows = load_editlens(limit=5)
    print(f"load_editlens → {len(rows)} records")
    for r in rows[:3]:
        row = r.to_row()
        print(f"  [{row['meta']['text_register']:<11}] {row['text_id']}")
        print(f"    {row['text'][:90]!r}")
