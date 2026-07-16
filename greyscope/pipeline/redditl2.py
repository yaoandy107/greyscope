"""Arctic Shift scraper for the EN non-native-English (L2) human corpus — a large TRAINABLE pool
of the same phenomenon as EditLens's `nonnative_english` calibration slice (European-flaired
r/AskEurope authors writing English prose). Submissions are pre-2022 (`CUTOFF_EPOCH`), the build's
contamination cutoff. Every API page is cached by request hash and paging is resumable: each page's
`before` cursor is derived from the previous page's oldest timestamp, so a killed run cache-hits
back to the frontier before new I/O. Pure parsers are unit-tested; only `_get_json` touches the net.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterator

import httpx

API_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"
CACHE_DIR = Path("data/v2/cache/reddit-l2")
_HEADERS = {"User-Agent": "greyscope dataset build (research; contact yaoandy107@gmail.com)"}
_RATE_LIMIT_S = 0.5  # polite delay between live API calls
_FIELDS = "id,created_utc,author,author_flair_text,selftext,title,subreddit"
_PAGE_LIMIT = 100  # Arctic Shift's practical per-request cap

CUTOFF_EPOCH = 1640995200  # 2022-01-01T00:00:00Z — contamination-defense cutoff
SUBREDDITS = ("AskEurope", "AskEuropeans", "europe")
MIN_CHARS = 400

# Non-English-speaking European countries, matched as a substring of `author_flair_text` (plain
# "Germany" or emoji-coded ":flag-de: Germany"), longest-name-first so "Czech Republic" wins over
# a shorter partial.
FLAIR_COUNTRY_MAP: dict[str, str] = {
    "Germany": "Germany",
    "France": "France",
    "Spain": "Spain",
    "Italy": "Italy",
    "Poland": "Poland",
    "Netherlands": "Netherlands",
    "Sweden": "Sweden",
    "Finland": "Finland",
    "Norway": "Norway",
    "Denmark": "Denmark",
    "Portugal": "Portugal",
    "Greece": "Greece",
    "Austria": "Austria",
    "Czech Republic": "Czechia",
    "Czechia": "Czechia",
    "Hungary": "Hungary",
    "Romania": "Romania",
    "Croatia": "Croatia",
    "Belgium": "Belgium",
    "Switzerland": "Switzerland",
    "Bulgaria": "Bulgaria",
    "Serbia": "Serbia",
    "Slovakia": "Slovakia",
    "Slovenia": "Slovenia",
    "Lithuania": "Lithuania",
    "Latvia": "Latvia",
    "Estonia": "Estonia",
    "Russia": "Russia",
    "Ukraine": "Ukraine",
}
# Anglophone flairs — never accepted. Checked FIRST so a future map addition can't un-exclude one.
EXCLUDED_FLAIRS: frozenset[str] = frozenset({
    "United Kingdom", "England", "Scotland", "Wales", "Northern Ireland",
    "Ireland", "United States", "United States of America", "Canada",
    "Australia", "New Zealand", "Malta",
})

_URL_RE = re.compile(r"https?://\S+")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # [text](url) → text
_MD_EMPHASIS_RE = re.compile(r"(\*\*\*|\*\*|\*|__|_|~~)")  # bold/italic/strike markers
_MD_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_QUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)
_MD_CODE_RE = re.compile(r"`{1,3}")
_DELETED_BODIES = frozenset({"[deleted]", "[removed]"})
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
_ALPHA_RE = re.compile(r"\w", re.UNICODE)
_MIN_ASCII_LETTER_RATIO = 0.9


class RedditL2Error(RuntimeError):
    """An Arctic Shift API call failed after retries, or a response could not be parsed."""


def country_from_flair(flair_text: str | None) -> str | None:
    """`author_flair_text` → mapped non-English-speaking European country, or None if absent,
    anglophone/excluded, or unmatched (ambiguous → reject, never guess)."""
    if not flair_text:
        return None
    text = flair_text.strip()
    if not text:
        return None
    if any(excluded in text for excluded in EXCLUDED_FLAIRS):
        return None
    for pattern in sorted(FLAIR_COUNTRY_MAP, key=len, reverse=True):
        if pattern in text:
            return FLAIR_COUNTRY_MAP[pattern]
    return None


def clean_selftext(text: str) -> str:
    """Strip URLs and markdown down to plain prose, so the length floor measures written content."""
    text = _URL_RE.sub("", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_HEADER_RE.sub("", text)
    text = _MD_QUOTE_RE.sub("", text)
    text = _MD_BULLET_RE.sub("", text)
    text = _MD_CODE_RE.sub("", text)
    text = _MD_EMPHASIS_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_deleted_or_removed(text: str) -> bool:
    """A `[deleted]`/`[removed]` selftext (Reddit's tombstone bodies), or empty/whitespace-only."""
    stripped = (text or "").strip()
    return not stripped or stripped in _DELETED_BODIES


def looks_english(text: str) -> bool:
    """Most word-characters must be ASCII letters — rejects non-Latin scripts without a lang-ID model."""
    alpha = _ALPHA_RE.findall(text)
    if not alpha:
        return False
    ascii_letters = sum(1 for ch in alpha if _ASCII_LETTER_RE.match(ch))
    return (ascii_letters / len(alpha)) >= _MIN_ASCII_LETTER_RATIO


def passes_filters(cleaned_body: str, *, min_chars: int = MIN_CHARS) -> bool:
    """Length + English gate on an already-cleaned body (deleted/removed is checked earlier on the
    RAW body, since cleaning can obscure a tombstone)."""
    return len(cleaned_body) >= min_chars and looks_english(cleaned_body)


def parse_submission(raw: dict) -> dict | None:
    """One submission row → {id, subreddit, country, title, body}, or None if it fails any filter
    (deleted/removed, too short, non-English, or flair not an accepted non-native country)."""
    raw_body = raw.get("selftext") or ""
    if is_deleted_or_removed(raw_body):
        return None
    country = country_from_flair(raw.get("author_flair_text"))
    if country is None:
        return None
    cleaned = clean_selftext(raw_body)
    if not passes_filters(cleaned):
        return None
    return {
        "id": raw["id"],
        "subreddit": raw.get("subreddit", ""),
        "country": country,
        "title": raw.get("title", ""),
        "created_utc": raw.get("created_utc"),
        "body": cleaned,
    }


def _get_json(params: dict, *, max_retries: int = 4, timeout: float = 30.0) -> dict:
    key = json.dumps(params, sort_keys=True, ensure_ascii=False)
    cached = CACHE_DIR / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))

    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            time.sleep(_RATE_LIMIT_S)
            response = httpx.get(API_URL, params=params, headers=_HEADERS, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            last_error = exc
        else:
            if payload.get("error") is None:  # success → cache + return
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                return payload
            last_error = RedditL2Error(f"API error: {payload['error']}")  # never cache an error body
        if attempt < max_retries - 1:
            time.sleep(delay)
            delay *= 2
    raise RedditL2Error(f"GET {params} failed after {max_retries} attempts: {last_error}")


def iter_pages(subreddit: str, *, before: int = CUTOFF_EPOCH,
              page_limit: int = _PAGE_LIMIT) -> Iterator[list[dict]]:
    """Raw submission dicts, paged newest→oldest from `before`, strictly backward in time. The
    page N+1 cursor is page N's oldest timestamp, so a re-run cache-hits every fetched page."""
    cursor = before
    while True:
        params = {"subreddit": subreddit, "before": cursor, "limit": page_limit,
                  "sort": "desc", "fields": _FIELDS}
        data = _get_json(params)
        page = data.get("data") or []
        if not page:
            return
        yield page
        oldest = min(row["created_utc"] for row in page if row.get("created_utc") is not None)
        if oldest >= cursor:  # no forward progress (pathological/duplicate page) → stop
            return
        cursor = oldest


def fetch_subreddit(subreddit: str, *, target: int, before: int = CUTOFF_EPOCH,
                    max_pages: int = 2000) -> list[dict]:
    """Accepted L2-English docs from one subreddit, paging back until `target` docs or `max_pages`
    (the hard stop so a low-yield subreddit can't run forever)."""
    out: list[dict] = []
    for page_no, page in enumerate(iter_pages(subreddit, before=before)):
        for raw in page:
            parsed = parse_submission(raw)
            if parsed is not None:
                out.append(parsed)
        if len(out) >= target or page_no + 1 >= max_pages:
            break
    return out


def fetch_l2_humans(*, target: int = 2000, subreddits: tuple[str, ...] = SUBREDDITS,
                    before: int = CUTOFF_EPOCH) -> list[dict]:
    """Accepted non-native-English-European docs up to `target`. Pulls r/AskEurope first, spilling
    into r/AskEuropeans then r/europe only while still under target."""
    out: list[dict] = []
    for subreddit in subreddits:
        remaining = target - len(out)
        if remaining <= 0:
            break
        out.extend(fetch_subreddit(subreddit, target=remaining, before=before))
    return out
