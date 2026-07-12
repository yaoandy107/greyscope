"""Japanese Wikinews scraper for the ja journalistic corpus.

ja has no clean HF full-article Wikinews source (`malteos/wikinews` omits ja), so this pulls
from the MediaWiki API the way `ptt.py` pulls PTT: pure parsers + a cached, polite `_get_json`.

Reproducibility + contamination defense:
- Each article is pinned to its **last revision before 2022** (the `rvstart` cutoff → AI-free),
  and the 【YYYY年MM月DD日】 dateline (true publication date) must also be pre-2022.
- `source_id = {pageid}@{revid}` is the stable rebuild id. Wikinews is CC BY → the
  permissive ja journalistic source, so it feeds human + mirror + **edited**.
- The 【…】 dateline is a Wikinews source-artifact the AI side never produces → stripped from the
  body after the date is read off it (anti-confound).
- Every API response is cached by request hash → re-runs never re-fetch (resumable scrape).

The pure parsers (`parse_dateline`, `extract_body`) take text/HTML and are unit-tested; only
`_get_json` and the `fetch_*` wrappers touch the network.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterator

import httpx
from bs4 import BeautifulSoup

API_URL = "https://ja.wikinews.org/w/api.php"
CACHE_DIR = Path("data/v2/cache/wikinews-ja")
_HEADERS = {"User-Agent": "greyscope dataset build (research; contact yaoandy107@gmail.com)"}
_RATE_LIMIT_S = 0.3  # polite delay between live API calls
_CUTOFF_ISO = "2022-01-01T00:00:00Z"  # pre-2022 contamination defense
_DATELINE_RE = re.compile(r"【\s*(20\d{2})年(\d{1,2})月(\d{1,2})日\s*】")


class WikinewsError(RuntimeError):
    """A Wikinews API call failed after retries, or a response could not be parsed."""


# --- pure parsers (unit-tested, no network) ----------------------------------
def parse_dateline(body: str) -> tuple[int, int, int] | None:
    """`【2008年2月3日】…` → (2008, 2, 3) — the article's publication date, or None."""
    match = _DATELINE_RE.search(body)
    if not match:
        return None
    return tuple(int(g) for g in match.groups())  # type: ignore[return-value]


def extract_body(parse_html: str) -> str:
    """Rendered article HTML → the human prose only: join `<p>` paragraphs (the sources /
    related / category boilerplate renders as headings + lists, not `<p>`, so it drops out)
    and strip the 【…】 dateline source-artifact."""
    root = BeautifulSoup(parse_html, "html.parser").select_one("div.mw-parser-output")
    if root is None:
        return ""
    paragraphs = [p.get_text().strip() for p in root.select("p") if p.get_text().strip()]
    body = "\n".join(paragraphs)
    body = _DATELINE_RE.sub("", body)
    return body.strip()


# --- network (cached, polite) ------------------------------------------------
def _get_json(params: dict, *, max_retries: int = 4, timeout: float = 30.0) -> dict:
    params = {**params, "format": "json"}
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
            if "error" not in payload:  # success → cache + return
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                return payload
            last_error = WikinewsError(f"API error: {payload['error']}")  # never cache an error body
        if attempt < max_retries - 1:
            time.sleep(delay)
            delay *= 2
    raise WikinewsError(f"API {params} failed after {max_retries} attempts: {last_error}")


def iter_titles(*, batch: int = 500) -> Iterator[str]:
    """Every content-namespace, non-redirect article title (paginated allpages)."""
    apcontinue: str | None = None
    while True:
        params = {"action": "query", "list": "allpages", "apnamespace": 0,
                  "apfilterredir": "nonredirects", "aplimit": batch}
        if apcontinue:
            params["apcontinue"] = apcontinue
        data = _get_json(params)
        for page in data["query"]["allpages"]:
            yield page["title"]
        apcontinue = data.get("continue", {}).get("apcontinue")
        if not apcontinue:
            return


def _last_pre_cutoff_rev(title: str) -> tuple[int, int, str] | None:
    """(pageid, revid, timestamp) of the last revision before the 2022 cutoff, or None if
    the page did not exist yet then (→ a post-2022 article, skipped)."""
    data = _get_json({"action": "query", "prop": "revisions", "titles": title,
                      "rvprop": "ids|timestamp", "rvlimit": 1, "rvdir": "older",
                      "rvstart": _CUTOFF_ISO})
    page = next(iter(data["query"]["pages"].values()))
    revisions = page.get("revisions")
    if not revisions:
        return None
    return page["pageid"], revisions[0]["revid"], revisions[0]["timestamp"]


def _render(revid: int) -> str | None:
    """Rendered article HTML for a revision, or None if it can't be parsed — a missing/deleted
    revision or an API error that survived retries → the caller skips that article rather than
    letting one bad revision abort the whole build."""
    try:
        data = _get_json({"action": "parse", "oldid": revid, "prop": "text", "disabletoc": 1})
    except WikinewsError:
        return None
    parse = data.get("parse")
    if not parse or "text" not in parse:
        return None
    return parse["text"]["*"]


def fetch_articles(*, limit: int | None = None) -> list[dict]:
    """Pre-2022 ja Wikinews articles. Each dict = {pageid, revid, timestamp, title, pub_date,
    body} — `body` is dateline-stripped human prose, `pub_date` is ISO from the dateline (or the
    revision date if the article carries none). Articles dated 2022+ are dropped."""
    out: list[dict] = []
    for title in iter_titles():
        pinned = _last_pre_cutoff_rev(title)
        if pinned is None:
            continue
        pageid, revid, timestamp = pinned
        html = _render(revid)
        if html is None:  # revision wouldn't render (missing/deleted/API error) → skip, don't abort
            continue
        dateline = parse_dateline(html)  # read off the raw render; extract_body then strips it
        year = dateline[0] if dateline else int(timestamp[:4])
        if year >= 2022:  # post-2022 publication slipped through → contamination risk
            continue
        pub_date = (f"{dateline[0]:04d}-{dateline[1]:02d}-{dateline[2]:02d}"
                    if dateline else timestamp[:10])
        out.append({"pageid": pageid, "revid": revid, "timestamp": timestamp,
                    "title": title, "pub_date": pub_date, "body": extract_body(html)})
        if limit is not None and len(out) >= limit:
            break
    return out
