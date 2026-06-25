"""Tests for the Taiwan-gov news scraper (greyscope/v2/twgov.py): the pure parsers only
(ROC date math, news-row extraction with boilerplate/self-link exclusion, body extraction).
The cached network paths (bisect, fetch_news) are not unit-tested — same policy as test_v2_ptt."""

from greyscope.v2 import twgov
from greyscope.v2.twgov import parse_article, parse_list, roc_to_year


def test_roc_to_year_converts_minguo_to_gregorian():
    assert roc_to_year(111) == 2022  # the contamination cutoff boundary
    assert roc_to_year(110) == 2021
    assert roc_to_year(115) == 2026


def test_parse_list_keeps_news_rows_drops_boilerplate_and_self_link():
    html = f"""
    <ul>
      <li><span><a href="News_Content.aspx?n={twgov.NEWS_NODE}&sms={twgov.SMS}&s=ABC123">
        標題測試</a></span> 110-12-31 臺北市政府測試局</li>
      <li><a href="News_Content.aspx?n=10FDEA7683714512&s=3B6C92FD22C01611">政府網站資料開放宣告</a></li>
      <li><a href="News_Content.aspx?n={twgov.NEWS_NODE}&s={twgov.SMS}">分類自連結</a></li>
    </ul>
    """
    rows = parse_list(html)
    assert len(rows) == 1                                   # boilerplate node + self-link excluded
    row = rows[0]
    assert row["s"] == "ABC123" and row["title"] == "標題測試"
    assert row["date"] == (2021, 12, 31)                    # 民國110 → 2021 (pre-2022)
    assert row["unit"] == "臺北市政府測試局"


def test_parse_article_extracts_essay_body():
    html = '<div class="area-essay">臺北市政府表示，這是測試內容。\n第二段內容。</div>'
    assert parse_article(html) == "臺北市政府表示，這是測試內容。\n第二段內容。"
    assert parse_article("<div>no essay block here</div>") is None


def test_parse_article_strips_leading_press_metadata_header():
    html = (
        '<div class="area-essay">'
        "發稿單位：綜合企劃科\n發稿日期：110年12月30日\n聯 絡 人：陳科長\n聯絡電話：1999分機6975\n"
        "臺北市政府為更瞭解本市民眾居住現況，特此說明。\n第二段正文內容。"
        "</div>"
    )
    body = parse_article(html)
    assert body.startswith("臺北市政府為更瞭解")   # metadata header (incl. spaced 聯 絡 人) stripped
    assert "發稿單位" not in body and "1999分機6975" not in body
    assert "第二段正文內容。" in body                # real body preserved
