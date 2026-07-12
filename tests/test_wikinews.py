"""Tests for the Japanese Wikinews scraper (greyscope/pipeline/wikinews.py): the pure parsers
(dateline + anti-confound body extraction) and `_render`'s skip-don't-crash handling of an
unparseable revision (the live network paths stay un-mocked — same policy as test_ptt)."""

from greyscope.pipeline import wikinews
from greyscope.pipeline.wikinews import WikinewsError, _render, extract_body, parse_dateline


def test_parse_dateline_reads_publication_date():
    assert parse_dateline("【2008年2月3日】岩波書店は…") == (2008, 2, 3)
    assert parse_dateline("【 2006年11月28日 】NNNによると…") == (2006, 11, 28)
    assert parse_dateline("本文にしか日付がない") is None


def test_extract_body_joins_paragraphs_and_strips_dateline():
    html = (
        '<div class="mw-parser-output">'
        "<p>【2008年2月3日】</p>"
        "<p>岩波書店は第六版を発売した。</p>"
        "<p>累計1100万部を誇る。</p>"
        "<h2>出典</h2><ul><li>J-CAST</li></ul>"  # sources render as heading+list, not <p>
        "</div>"
    )
    body = extract_body(html)
    assert "【2008年2月3日】" not in body          # dateline source-artifact stripped
    assert "岩波書店は第六版を発売した。" in body
    assert "累計1100万部を誇る。" in body
    assert "J-CAST" not in body and "出典" not in body  # sources boilerplate excluded


def test_extract_body_returns_empty_when_no_content_div():
    assert extract_body("<div>no mw-parser-output here</div>") == ""


def test_render_skips_unparseable_revision(monkeypatch):
    # the KeyError that killed the full build: a response with no 'parse' must yield None (skip),
    # whether it's an API error body, an empty dict, or _get_json raising after retries.
    monkeypatch.setattr(wikinews, "_get_json", lambda *a, **k: {"error": {"code": "nosuchrevid"}})
    assert _render(1) is None
    monkeypatch.setattr(wikinews, "_get_json", lambda *a, **k: {})
    assert _render(2) is None

    def _raise(*a, **k):
        raise WikinewsError("failed after retries")
    monkeypatch.setattr(wikinews, "_get_json", _raise)
    assert _render(3) is None


def test_render_returns_html_for_good_revision(monkeypatch):
    monkeypatch.setattr(wikinews, "_get_json", lambda *a, **k: {"parse": {"text": {"*": "<div>ok</div>"}}})
    assert _render(4) == "<div>ok</div>"
