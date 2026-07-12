"""Tests for the PTT pure parsers (greyscope/pipeline/ptt.py).

Fixture-driven — no network. Guards body extraction (metalines / pushes / signature
stripped → human passage only) and the post-id date decode that powers the pre-2022
contamination filter.
"""

from greyscope.pipeline.ptt import parse_article, parse_index, post_unixtime, post_year

_ARTICLE_HTML = """
<div id="main-content">
<div class="article-metaline"><span class="article-meta-tag">作者</span>
  <span class="article-meta-value">user (暱稱)</span></div>
<div class="article-metaline-right"><span class="article-meta-tag">看板</span>
  <span class="article-meta-value">marvel</span></div>
<div class="article-metaline"><span class="article-meta-tag">標題</span>
  <span class="article-meta-value">[經驗] 半夜的腳步聲</span></div>
<div class="article-metaline"><span class="article-meta-tag">時間</span>
  <span class="article-meta-value">Sat Jun 13 12:34:56 2018</span></div>
那是一個很普通的夜晚，我一個人住在老舊的公寓裡。半夜時分，走廊傳來規律的腳步聲。
我屏住呼吸，腳步聲卻在門口停了下來……
--
※ 發信站: 批踢踢實業坊(ptt.cc), 來自: 1.2.3.4 (臺灣)
※ 文章網址: https://www.ptt.cc/bbs/marvel/M.1528864496.A.0AA.html
<span class="f2">※ 編輯: user (1.2.3.4 臺灣), 06/13/2018 12:40:00</span>
<div class="push"><span class="push-tag">推 </span>
  <span class="push-userid">someone</span><span class="push-content">: 好可怕</span></div>
</div>
"""

_INDEX_HTML = """
<div class="r-ent"><div class="title">
  <a href="/bbs/marvel/M.1528864496.A.0AA.html">[經驗] 半夜的腳步聲</a></div></div>
<div class="r-ent"><div class="title">(本文已被刪除)</div></div>
<div class="btn-group btn-group-paging">
  <a class="btn wide" href="/bbs/marvel/index2519.html">‹ 上頁</a>
  <a class="btn wide" href="/bbs/marvel/index2521.html">下頁 ›</a></div>
"""


def test_post_unixtime_and_year():
    assert post_unixtime("M.1528864496.A.0AA") == 1528864496
    assert post_year("M.1528864496.A.0AA") == 2018
    assert post_unixtime("not-a-post-id") is None


def test_parse_article_extracts_body_only():
    art = parse_article(_ARTICLE_HTML, "M.1528864496.A.0AA")
    assert art is not None
    assert art["title"] == "[經驗] 半夜的腳步聲"
    body = art["body"]
    assert "腳步聲" in body and "停了下來" in body
    # metalines / signature / push / edit-note all stripped:
    assert "作者" not in body and "看板" not in body
    assert "發信站" not in body and "文章網址" not in body
    assert "好可怕" not in body and "編輯" not in body
    assert not body.endswith("-")


def test_parse_article_returns_none_when_no_main_content():
    assert parse_article("<html><body>deleted</body></html>", "M.1.A.1") is None


def test_parse_index_collects_posts_and_older_page():
    post_ids, older = parse_index(_INDEX_HTML)
    assert post_ids == ["M.1528864496.A.0AA"]  # the deleted entry has no link
    assert older == "/bbs/marvel/index2519.html"
