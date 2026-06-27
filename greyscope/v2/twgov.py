"""Taiwan government press-release scraper for the v2 zh-TW journalistic corpus.

zh-TW journalistic needs NATIVE Taiwan Traditional + pre-2022 + a permissive (edited-OK)
license. The data.gov.tw open-data feeds are rolling "latest-N" CURRENT releases → they fail
the pre-2022 contamination defense, so this scrapes the gov.taipei news ARCHIVE directly
(server-rendered, dated, paginated ~2977 pages back to 2016) the way ptt.py scrapes PTT.

Reproducibility + contamination defense:
- List rows carry a 民國 (ROC) publish date → keep pre-2022 only (ROC ≤ 110 = 2021) and
  **binary-search** the index by date instead of walking thousands of pages back from today.
- `source_id` = the article's `s=` id (stable addressable id). Taipei City gov
  content is OGDL (公眾授權) → permissive, so it feeds human + mirror + **edited**.
- Every page is cached by URL hash → re-runs never re-fetch (resumable scrape).

The pure parsers (`roc_to_year`, `parse_list`, `parse_article`) take HTML and are unit-tested;
only `_get` and the `fetch_*` wrappers touch the network.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup

BASE_URL = "https://www.gov.taipei"
NEWS_NODE = "F0DDAF49B89E9413"  # 市府新聞稿 (integrated city press-release archive)
SMS = "72544237BBE4C5F6"        # the category's own self-link s — never an article
CACHE_DIR = Path("data/v2/cache/twgov")
_HEADERS = {"User-Agent": "Mozilla/5.0 (greyscope-v2 dataset build; research)"}
_RATE_LIMIT_S = 0.4
_ROC_YEAR_OFFSET = 1911  # 民國 year + 1911 = Gregorian (民國111 = 2022)
_ROC_DATE = re.compile(r"(\d{3})-(\d{1,2})-(\d{1,2})")  # list-row publish date, e.g. 115-06-22
# Leading press-release metadata labels (issuing unit / date / contact / phone) — some bodies
# open with this block; the AI mirror never produces it, so it is a source-artifact.
_HEADER_LABELS = ("發稿單位", "發稿日期", "發布單位", "發布日期", "聯絡人", "新聞聯絡", "業務聯絡",
                  "聯絡電話", "聯絡資訊", "主辦單位", "承辦單位", "資料來源", "發言人")


class TwGovError(RuntimeError):
    """A gov.taipei fetch failed after retries, or a page could not be parsed."""


# --- pure parsers (unit-tested, no network) ----------------------------------
def roc_to_year(roc_year: int) -> int:
    return roc_year + _ROC_YEAR_OFFSET


def parse_list(html: str) -> list[dict]:
    """A news index page → [{s, title, date: (y, m, d), unit}] for the real articles only
    (links on the news node, excluding the category's own self-link)."""
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict] = []
    for link in soup.select("a[href*=News_Content]"):
        query = parse_qs(urlparse(link["href"]).query)
        sid = query.get("s", [""])[0]
        if query.get("n", [""])[0] != NEWS_NODE or sid in ("", SMS):
            continue
        row = link.find_parent(["li", "tr", "div"])
        match = _ROC_DATE.search(row.get_text(" ", strip=True)) if row else None
        if not match:
            continue
        roc, month, day = (int(g) for g in match.groups())
        unit = row.get_text(" ", strip=True).split(match.group(0))[-1].strip()
        out.append({"s": sid, "title": link.get_text(strip=True),
                    "date": (roc_to_year(roc), month, day), "unit": unit})
    return out


def _strip_press_header(body: str) -> str:
    """Drop a leading 發稿單位/聯絡電話/… metadata block (anti-confound)."""
    lines = body.split("\n")
    cut = 0
    for line in lines:
        compact = line.replace(" ", "").replace("　", "")
        head = compact[:12]
        if any(compact.startswith(lbl) for lbl in _HEADER_LABELS) and ("：" in head or ":" in head):
            cut += 1
        else:
            break
    return "\n".join(lines[cut:]).strip()


def parse_article(html: str) -> str | None:
    """An article page → the body prose (the `.area-essay` block, minus any leading press-release
    metadata header), or None if empty."""
    soup = BeautifulSoup(html, "html.parser")
    essay = soup.select_one(".area-essay")
    if essay is None:
        return None
    body = _strip_press_header(essay.get_text("\n", strip=True))
    return body or None


# --- network (cached, polite) ------------------------------------------------
def _get(url: str, *, max_retries: int = 4, timeout: float = 25.0) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cached = CACHE_DIR / f"{digest}.html"
    if cached.exists():
        return cached.read_text(encoding="utf-8")

    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            time.sleep(_RATE_LIMIT_S)
            response = httpx.get(url, headers=_HEADERS, timeout=timeout, follow_redirects=True)
            if response.status_code == 404:
                return ""
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
    raise TwGovError(f"GET {url} failed after {max_retries} attempts: {last_error}")


def _list_url(page: int) -> str:
    return f"{BASE_URL}/News.aspx?n={NEWS_NODE}&sms={SMS}&page={page}"


def _article_url(sid: str) -> str:
    return f"{BASE_URL}/News_Content.aspx?n={NEWS_NODE}&s={sid}"


def _max_page() -> int:
    pages = [int(m) for m in re.findall(r"[?&]page=(\d+)", _get(_list_url(1)))]
    if not pages:
        raise TwGovError("no pagination on gov.taipei news index — layout changed")
    return max(pages)


def _page_year(page: int) -> int | None:
    rows = parse_list(_get(_list_url(page)))
    return rows[0]["date"][0] if rows else None


def _bisect_page_for(before_year: int, *, max_steps: int = 20) -> int:
    """Smallest page index whose newest article is already older than `before_year` (the index
    runs newest→oldest, so dates fall as the page number rises). Mirrors ptt._bisect_index_for."""
    lo, hi = 1, _max_page()
    while lo < hi and max_steps > 0:
        max_steps -= 1
        mid = (lo + hi) // 2
        year = _page_year(mid)
        if year is None or year >= before_year:
            lo = mid + 1
        else:
            hi = mid
    return max(1, lo)


def fetch_news(*, before_year: int = 2022, limit: int | None = None) -> list[dict]:
    """Pre-`before_year` gov.taipei press releases. Each dict = {s, title, pub_date, unit, body}.
    Walks from the date-bisected start page toward older pages, keeping only pre-cutoff rows."""
    page = _bisect_page_for(before_year)
    last = _max_page()
    out: list[dict] = []
    while page <= last and (limit is None or len(out) < limit):
        for row in parse_list(_get(_list_url(page))):
            year, month, day = row["date"]
            if year >= before_year:
                continue
            body = parse_article(_get(_article_url(row["s"])))
            if not body:
                continue
            out.append({"s": row["s"], "title": row["title"], "unit": row["unit"],
                        "pub_date": f"{year:04d}-{month:02d}-{day:02d}", "body": body})
            if limit is not None and len(out) >= limit:
                break
        page += 1
    return out
