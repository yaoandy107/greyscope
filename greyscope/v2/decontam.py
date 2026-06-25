"""Decontamination: scrub EN training humans of overlap with the benchmarks we report on
(design §14.2). Runs as a **pre-generation filter on the human pool** — a contaminated human
poisons its mirror+edit too, so dropping it before generation keeps the ~$50 spend on clean
docs and makes the splits leak-free by construction.

**EN only, by design.** Our zh-TW (Taiwan Wikipedia / PTT / gov.taipei) and ja (Wikipedia /
Aozora / Wikinews / open2ch) sources are disjoint from the public MGT benchmarks by construction
— RAID is EN/cs/de, and M4/M4GT's only Chinese data is Simplified QA (no Chinese Wikipedia). No
public eval shares those sources, so there is nothing external to scrub; zh/ja eval integrity
rests on the internal held-out generator/domain slices (§11), not external decontam. EN is the
exception: EditLens humans (news/reviews/reddit/web) plausibly share English web sources with
RAID, AND we submit to RAID at release — so train∩RAID would inflate the headline number.

Method = canonical word-n-gram containment (GPT-3/Pile style). The reference is the union of
RAID human texts + EditLens's held-out test humans; a candidate sharing ≥`MIN_SHARED` distinct
13-grams with it is dropped. Two distinct 13-grams (~14–26 verbatim words) effectively never
coincide between independent texts, and humans are the free FPR base, so we favor dropping.

The pure core (normalize → grams → containment → filter) unit-tests with no network. The reference
loaders download RAID's non-adversarial CSVs once (resumable, so a flaky connection can't kill a
mid-stream read), filter humans locally, and cache the extracted texts; every later run reads the cache.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import httpx

DECONTAM_DIR = Path("data/v2/decontam")
csv.field_size_limit(2**31 - 1)  # RAID `generation` cells can be long — lift the default 128k cap

K_EN = 13          # the canonical decontamination shingle: 13 words (GPT-3/The Pile)
MIN_SHARED = 2     # ≥2 distinct shared 13-grams ⇒ real reuse, not coincidence (favor dropping)

# RAID's NON-ADVERSARIAL train_none.csv from its CDN (~0.76GB) vs the 11.8GB adversarial train.csv on
# HF: ~15× smaller (no 12× attack bloat → humans are dense) AND fetched to disk WITH RESUME — the HF
# range-stream died on a flaky-WiFi mid-read with no recovery; a download resumes from the bytes
# already saved, then filtering runs locally with no network. Only `train` is usable: RAID hides the
# test labels for its leaderboard (test_none.csv is just id+generation, no `model`), and train humans
# cover the same domains/sources anyway.
_RAID_CDN = "https://dataset.raid-bench.xyz"
_RAID_NONE_FILES = {"train": "train_none.csv"}
# Czech/German/code domains live only in RAID's `extra` split, so train is EN-only; the denylist is a
# cheap safety net (a stray non-EN row can't false-match English prose anyway).
_RAID_NON_EN_DOMAINS = frozenset({"czech_news", "german_news", "code"})
_RAID_SPLITS = ("train",)
_EDITLENS_TEST_SPLITS = ("test_llama", "test_enron")

_WORD = re.compile(r"\w+", re.UNICODE)


# --- pure core (no network) -------------------------------------------------
def _normalize(text: str) -> list[str]:
    """Lowercase word tokens — the unit n-grams are built from (punctuation/spacing dropped)."""
    return _WORD.findall(text.lower())


def _hash_gram(tokens: list[str]) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(tokens).encode("utf-8"), digest_size=8).digest(), "big")


def text_grams(text: str, k: int = K_EN) -> set[int]:
    """Hashed word-`k`-grams of `text`. A doc shorter than `k` collapses to one whole-doc gram
    (so short texts stay comparable instead of silently matching nothing)."""
    tokens = _normalize(text)
    if len(tokens) < k:
        return {_hash_gram(tokens)} if tokens else set()
    return {_hash_gram(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def build_reference(texts: Iterable[str], k: int = K_EN) -> set[int]:
    reference: set[int] = set()
    for text in texts:
        reference |= text_grams(text, k)
    return reference


def overlap_count(text: str, reference: set[int], k: int = K_EN) -> int:
    """How many of `text`'s distinct k-grams appear in the reference."""
    return len(text_grams(text, k) & reference)


def filter_english(rows: list[dict], reference: set[int], *, k: int = K_EN,
                   min_shared: int = MIN_SHARED) -> tuple[list[dict], list[dict]]:
    """Drop EN rows whose text overlaps the reference by ≥`min_shared` distinct k-grams.
    Non-EN rows pass through untouched (they have no external target — see module doc).
    Returns (clean, dropped); each dropped row carries `drop_reason` + `meta.contam_overlap`."""
    clean: list[dict] = []
    dropped: list[dict] = []
    for row in rows:
        if row["language"] != "en":
            clean.append(row)
            continue
        shared = overlap_count(row["text"], reference, k)
        if shared >= min_shared:
            dropped.append({**row, "drop_reason": "contaminated",
                            "meta": {**row["meta"], "contam_overlap": shared}})
        else:
            clean.append(row)
    return clean, dropped


# --- reference loaders (resumable download → local filter, cached to disk) --
def _content_total(headers, have: int) -> int | None:
    """Total file size from a (possibly partial) response: prefer Content-Range's total, else
    Content-Length (+ what's already on disk for a 206). None if the server advertises neither."""
    cr = headers.get("content-range")  # e.g. "bytes 100-11233/11234"
    if cr and "/" in cr:
        tail = cr.rsplit("/", 1)[-1]
        return int(tail) if tail.isdigit() else None
    cl = headers.get("content-length")
    return have + int(cl) if cl is not None else None


def _download_resumable(url: str, dest: Path, *, max_attempts: int = 30, timeout: float = 60.0) -> Path:
    """Download `url` → `dest`, resuming from the partial file on each retry (HTTP Range). Built for
    a flaky connection: a dropped read waits and continues from the bytes already on disk instead of
    restarting. Returns once the local file matches the server's advertised total size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_attempts):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with httpx.stream("GET", url, headers=headers, timeout=timeout, follow_redirects=True) as r:
                if r.status_code == 416:  # requested range past EOF ⇒ already complete
                    return dest
                if have and r.status_code == 200:  # server ignored Range → restart clean
                    have = 0
                r.raise_for_status()
                total = _content_total(r.headers, have)
                with dest.open("ab" if have else "wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
            if total is None or dest.stat().st_size >= total:  # clean end or size matched ⇒ done
                return dest
        except httpx.HTTPError as exc:
            now = dest.stat().st_size if dest.exists() else 0
            print(f"    [decontam] download blip ({type(exc).__name__}); resuming from {now:,} B")
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"failed to download {url} after {max_attempts} attempts")


def _download_raid_csv(split: str) -> Path:
    fname = _RAID_NONE_FILES[split]
    dest = DECONTAM_DIR / "raw" / fname
    print(f"    [decontam] raid/{split}: fetching {fname} (resumable) …")
    return _download_resumable(f"{_RAID_CDN}/{fname}", dest)


def _humans_from_csv(path: Path, *, limit: int | None = None) -> list[str]:
    """Human texts from a RAID CSV: `model=="human"` rows (the non-adversarial file has no attack
    variants), EN domains only, deduped by source id, taking the `generation` column."""
    texts: list[str] = []
    seen: set = set()
    with path.open(newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i and i % 100_000 == 0:
                print(f"    [decontam] {path.name}: read {i:,} rows, {len(texts):,} humans")
            if row.get("model") != "human" or row.get("domain") in _RAID_NON_EN_DOMAINS:
                continue
            sid = row.get("source_id") or row.get("id")
            if sid in seen:
                continue
            seen.add(sid)
            text = (row.get("generation") or "").strip()
            if text:
                texts.append(text)
            if limit is not None and len(texts) >= limit:
                break
    return texts


def _raid_cache(split: str) -> Path:
    return DECONTAM_DIR / f"raid_humans_{split}.jsonl"


def _read_text_cache(path: Path) -> list[str]:
    # JSONL: split on '\n' ONLY. json.dumps(ensure_ascii=False) can emit U+2028/U+2029/U+0085
    # inside a string, and str.splitlines() would wrongly break a record at those.
    return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").split("\n") if line]


def _write_text_cache(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in texts) + "\n", encoding="utf-8")


def extract_raid_humans(split: str, *, limit: int | None = None, refresh: bool = False) -> list[str]:
    """RAID human texts for one split → cached to disk. Downloads the non-adversarial CSV (resumable)
    then filters locally, so a flaky connection can't kill a mid-stream read. A partial (smoke) pull
    never reads/writes the disk cache → can't poison the full reference."""
    cache = _raid_cache(split)
    use_cache = limit is None  # a partial (smoke/validation) pull must not poison the full reference
    if use_cache and cache.exists() and not refresh:
        return _read_text_cache(cache)

    texts = _humans_from_csv(_download_raid_csv(split), limit=limit)
    if not texts:  # a labeled split must yield humans — 0 ⇒ schema/label problem; never cache empty
        raise RuntimeError(f"raid/{split}: 0 humans extracted — does the CSV have a 'model' column?")
    print(f"    [decontam] raid/{split}: {len(texts):,} human texts")
    if use_cache:
        _write_text_cache(cache, texts)
    return texts


def _editlens_test_humans() -> list[str]:
    from greyscope.v2 import corpora

    out: list[str] = []
    for split in _EDITLENS_TEST_SPLITS:
        out += [r["text"] for r in corpora.load_editlens_split(split) if r["text_type"] == "human_written"]
    return out


@lru_cache(maxsize=1)
def english_reference(*, raid_limit: int | None = None) -> frozenset:
    """Union n-gram reference of {RAID train humans + EditLens held-out test humans}.
    Cached per process; the RAID extraction is itself disk-cached, so re-runs are cheap."""
    texts = list(_editlens_test_humans())
    raid = 0
    for split in _RAID_SPLITS:
        split_texts = extract_raid_humans(split, limit=raid_limit)
        texts += split_texts
        raid += len(split_texts)
    print(f"    [decontam] EN reference: {raid:,} RAID + {len(texts) - raid:,} EditLens-test humans")
    return frozenset(build_reference(texts))
