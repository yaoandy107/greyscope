"""PTT (批踢踢實業坊) scraper for the v2 zh-TW human corpus (design §4, plan §5).

PTT is one MULTI-REGISTER source: casual (Gossiping), reviews (Food / MobileComm),
creative (marvel / eWriter / Fiction). Same fetch/parse/ID story for every board, so
`corpora.load_ptt` just maps board → register.

Reproducibility + contamination defense:
- The post id `M.<unixtime>.A.<hash>` embeds the exact post time → we keep **pre-2022**
  only (the contamination defense, design §4) and **binary-search** the index by date
  instead of walking thousands of pages back from today.
- The id is the **stable addressable id** for the rebuild-from-IDs release (design §13);
  PTT is unlicensed user text → **human + mirror only**, never an edited derivative.
- Every page is cached to disk by URL hash → re-runs never re-fetch, and the scrape is
  resumable (the design's "cache every response" principle).

The pure parsers (`parse_index`, `parse_article`, `post_unixtime`) take HTML/text and are
unit-tested; only `_get` and the `fetch_*` wrappers touch the network.
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

BASE_URL = "https://www.ptt.cc"
CACHE_DIR = Path("data/v2/cache/ptt")
_HEADERS = {"User-Agent": "Mozilla/5.0 (greyscope-v2 dataset build; research)"}
_COOKIES = {"over18": "1"}  # bypass the age gate (Gossiping etc.); harmless elsewhere
_RATE_LIMIT_S = 0.4  # polite delay between live fetches
_SIGNATURE = "※ 發信站"  # body ends here; everything after is sig + edit notes
_POST_ID_RE = re.compile(r"M\.(\d+)\.A\.")


class PTTError(RuntimeError):
    """A PTT fetch failed after retries, or a page could not be parsed."""


# --- pure parsers (unit-tested, no network) ----------------------------------
def post_unixtime(post_id: str) -> int | None:
    """`M.1528881296.A.0AA` → 1528881296 (the post's unix time), or None."""
    match = _POST_ID_RE.search(post_id)
    return int(match.group(1)) if match else None


def post_year(post_id: str) -> int | None:
    ts = post_unixtime(post_id)
    return datetime.fromtimestamp(ts, timezone.utc).year if ts is not None else None


def parse_index(html: str) -> tuple[list[str], str | None]:
    """An index page → (post_ids on it, href of the OLDER '‹ 上頁' page)."""
    soup = BeautifulSoup(html, "html.parser")
    post_ids: list[str] = []
    for link in soup.select("div.r-ent div.title a[href]"):
        stem = link["href"].rsplit("/", 1)[-1].removesuffix(".html")
        if _POST_ID_RE.search(stem):
            post_ids.append(stem)
    older = None
    for btn in soup.select("a.btn.wide"):
        if "上頁" in btn.get_text():
            older = btn.get("href")
    return post_ids, older


def parse_article(html: str, post_id: str) -> dict | None:
    """An article page → {post_id, title, body}, or None if deleted/empty. Strips the
    metalines, push comments, and the `※ 發信站` signature/edit trailer so `body` is the
    human-written passage only (source-artifact normalization, design §8.7)."""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("#main-content")
    if main is None:
        return None

    title = None
    for line in main.select("div.article-metaline"):
        tag = line.select_one(".article-meta-tag")
        value = line.select_one(".article-meta-value")
        if tag and value and tag.get_text(strip=True) == "標題":
            title = value.get_text(strip=True)

    for junk in main.select(
        "div.article-metaline, div.article-metaline-right, div.push, span.f2"
    ):
        junk.decompose()

    text = main.get_text()
    cut = text.find(_SIGNATURE)
    if cut != -1:
        text = text[:cut]
    body = text.strip().strip("-").strip()
    if not body:
        return None
    return {"post_id": post_id, "title": title, "body": body}


# --- network (cached, polite) ------------------------------------------------
def _get(url: str, *, max_retries: int = 4, timeout: float = 20.0) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cached = CACHE_DIR / f"{digest}.html"
    if cached.exists():
        return cached.read_text(encoding="utf-8")

    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            time.sleep(_RATE_LIMIT_S)
            response = httpx.get(
                url, headers=_HEADERS, cookies=_COOKIES, timeout=timeout,
                follow_redirects=True,
            )
            if response.status_code == 404:
                return ""  # deleted board/page — caller skips
            response.raise_for_status()
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
        else:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cached.write_text(response.text, encoding="utf-8")
            return response.text
    raise PTTError(f"GET {url} failed after {max_retries} attempts: {last_error}")


def _index_url(board: str, page: int | None = None) -> str:
    suffix = "index.html" if page is None else f"index{page}.html"
    return f"{BASE_URL}/bbs/{board}/{suffix}"


def _latest_index(board: str) -> int:
    """Highest index page number (newest)."""
    _, older = parse_index(_get(_index_url(board)))
    if not older:
        raise PTTError(f"no paging on {board} index — board missing or layout changed")
    return int(re.search(r"index(\d+)\.html", older).group(1)) + 1


def _page_median_unixtime(board: str, page: int) -> int | None:
    post_ids, _ = parse_index(_get(_index_url(board, page)))
    stamps = sorted(t for t in (post_unixtime(p) for p in post_ids) if t)
    return stamps[len(stamps) // 2] if stamps else None


def _bisect_index_for(board: str, target_unixtime: int, *, max_steps: int = 20) -> int:
    """Index page whose posts are nearest to (and not after) `target_unixtime`."""
    lo, hi = 1, _latest_index(board)
    while lo < hi and max_steps > 0:
        max_steps -= 1
        mid = (lo + hi) // 2
        median = _page_median_unixtime(board, mid)
        if median is None:
            lo = mid + 1
        elif median < target_unixtime:
            lo = mid + 1
        else:
            hi = mid
    return max(1, lo)


def fetch_board(
    board: str, *, year_range: tuple[int, int] = (2015, 2019), limit: int = 50
) -> list[dict]:
    """Up to `limit` pre-2022 article bodies from `board`, walking newer from the
    bisected start page. Each dict = {post_id, title, body} (parse_article)."""
    start_ts = int(datetime(year_range[0], 1, 1, tzinfo=timezone.utc).timestamp())
    page = _bisect_index_for(board, start_ts)
    latest = _latest_index(board)

    out: list[dict] = []
    while page <= latest and len(out) < limit:
        post_ids, _ = parse_index(_get(_index_url(board, page)))
        for post_id in post_ids:
            year = post_year(post_id)
            if year is None or not (year_range[0] <= year <= year_range[1]):
                continue
            article = parse_article(
                _get(f"{BASE_URL}/bbs/{board}/{post_id}.html"), post_id
            )
            if article and article["body"]:
                out.append(article)
                if len(out) >= limit:
                    break
        page += 1
    return out
