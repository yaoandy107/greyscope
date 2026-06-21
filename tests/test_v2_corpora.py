"""Tests for v2 human-corpus normalization (greyscope/v2/corpora.py).

These guard the source-artifact anti-confound logic (design §8.7): if wiki40b
tokens or Aozora ruby leaked into "human" text, the detector would learn the
markup instead of authorship. Pure functions — no network, no `datasets`.
"""

from greyscope.v2.corpora import (
    HumanRecord,
    _as_str,
    count_cjk_chars,
    parse_wiki40b,
    passes_floor,
    strip_aozora_markup,
)


def test_count_cjk_chars_counts_kana_and_kanji_not_ascii():
    assert count_cjk_chars("ひらがなカタカナ漢字") == 10
    assert count_cjk_chars("繁體中文") == 4
    assert count_cjk_chars("plain ascii only") == 0


def test_passes_floor_en_by_words_cjk_by_chars():
    assert passes_floor("word " * 75, "en")
    assert not passes_floor("word " * 74, "en")
    assert passes_floor("漢" * 150, "ja")
    assert not passes_floor("漢" * 149, "zh-tw")


def test_parse_wiki40b_strips_markers_and_extracts_title():
    raw = (
        "_START_ARTICLE_\n東京\n_START_SECTION_\n概要\n"
        "_START_PARAGRAPH_\n東京は首都。_NEWLINE_人口が多い。"
    )
    title, body = parse_wiki40b(raw)
    assert title == "東京"
    assert "_START_" not in body and "_NEWLINE_" not in body
    assert "東京は首都。" in body and "人口が多い。" in body


def test_parse_wiki40b_handles_text_without_markers():
    title, body = parse_wiki40b("plain text _NEWLINE_ second line")
    assert title == ""
    assert body == "plain text \n second line"


def test_strip_aozora_markup_removes_ruby_and_notes():
    cleaned = strip_aozora_markup("吾輩《わがはい》は｜猫《ねこ》である［＃改行］。")
    assert cleaned == "吾輩は猫である。"


def test_as_str_decodes_bytes():
    assert _as_str(b"Q677915") == "Q677915"
    assert _as_str("already str") == "already str"
    assert _as_str(None) == ""


def test_human_record_to_row_schema_and_human_defaults():
    row = HumanRecord(
        text="本文テキスト",
        language="ja",
        source="wiki40b-ja",
        source_id="Q1@v2",
        text_register="formal",
        meta={"topic": "東京", "length": 6},
    ).to_row()
    assert row["text_id"] == "ja/wiki40b-ja/Q1@v2/human_written"
    assert row["text_type"] == "human_written"
    assert row["cosine_score"] == 0.0 and row["bucket"] == 0
    assert row["source_text"] is None and row["model"] is None
    assert row["meta"]["text_register"] == "formal"
    assert row["meta"]["topic"] == "東京"
