"""Tests for v2 human-corpus normalization (greyscope/v2/corpora.py).

These guard the source-artifact anti-confound logic: if wiki40b
tokens or Aozora ruby leaked into "human" text, the detector would learn the
markup instead of authorship. Pure functions — no network, no `datasets`.
"""

from greyscope.v2 import corpora
from greyscope.v2.corpora import (
    HumanRecord,
    _as_str,
    chunk_passages,
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


def test_chunk_passages_caps_per_work_and_keeps_short_tail():
    para = "漢" * 200
    # Four 200-char paragraphs, target 450 → 450+ then a sub-target tail (proves short works survive).
    passages = chunk_passages("\n".join([para] * 4), target_chars=450, max_chunks=5)
    assert len(passages) == 2
    assert count_cjk_chars(passages[0]) == 600  # 3 paras accumulate past the target
    assert count_cjk_chars(passages[1]) == 200  # trailing partial is emitted, not dropped
    # The per-work cap halts emission early (one long novel can't flood the register).
    capped = chunk_passages("\n".join([para] * 20), target_chars=450, max_chunks=3)
    assert len(capped) == 3


def test_load_aozora_filters_orthography_chunks_and_stamps_ids(monkeypatch):
    modern = {"text": "\n".join(["漢" * 200] * 4),
              "meta": {"作品ID": "000123", "人物ID": "000045", "作品名": "テスト作品",
                       "文字遣い種別": "新字新仮名", "公開日": "2020-01-02 00:00:00"}}
    classical = {"text": "旧" * 800,
                 "meta": {"作品ID": "000999", "文字遣い種別": "旧字旧仮名"}}
    monkeypatch.setattr(corpora, "_iter_hf", lambda *a, **k: iter([modern, classical]))

    rows = [r.to_row() for r in corpora.load_aozora()]
    # Only the modern work survives, chunked into two passages with stable #p{n} ids.
    assert {r["source_id"] for r in rows} == {"000123#p0", "000123#p1"}
    assert all(r["language"] == "ja" and r["source"] == "aozora" for r in rows)
    assert all(r["meta"]["text_register"] == "creative" for r in rows)
    assert rows[0]["text_id"] == "ja/aozora/000123#p0/human_written"
    assert rows[0]["meta"]["topic"] == "テスト作品" and rows[0]["meta"]["person_id"] == "000045"
    assert rows[0]["meta"]["pub_date"] == "2020-01-02"                        # datetime trimmed to date


def test_load_wikinews_ja_maps_records_and_filters_floor(monkeypatch):
    from greyscope.v2 import wikinews

    arts = [
        {"pageid": 11479, "revid": 74204, "timestamp": "2009-08-18T09:22:24Z",
         "title": "広辞苑が大改訂", "pub_date": "2008-02-03", "body": "記" * 200},  # clears floor
        {"pageid": 1, "revid": 2, "timestamp": "2007-01-01T00:00:00Z",
         "title": "短信", "pub_date": "2007-01-01", "body": "記" * 10},            # under floor
    ]
    monkeypatch.setattr(wikinews, "fetch_articles", lambda *, limit=None: arts)
    rows = [r.to_row() for r in corpora.load_wikinews_ja()]
    assert len(rows) == 1                                              # short article dropped
    assert rows[0]["text_id"] == "ja/wikinews-ja/11479@74204/human_written"  # pageid@revid id
    assert rows[0]["language"] == "ja" and rows[0]["meta"]["text_register"] == "journalistic"
    assert rows[0]["meta"]["pub_date"] == "2008-02-03"                # dateline, not rev date
    assert rows[0]["meta"]["rev_timestamp"].startswith("2009")


def test_load_amazon_reviews_ja_maps_records_and_filters_floor(monkeypatch):
    exs = [
        {"id": 555, "text": "本" * 200, "label": 4, "label_text": "5 stars"},  # clears floor
        {"id": 7, "text": "短評", "label": 0},                                  # under floor
    ]
    monkeypatch.setattr(corpora, "_iter_hf", lambda *a, **k: iter(exs))
    rows = [r.to_row() for r in corpora.load_amazon_reviews_ja()]
    assert len(rows) == 1                                                      # short review dropped
    assert rows[0]["text_id"] == "ja/amazon-reviews-ja/555/human_written"
    assert rows[0]["language"] == "ja" and rows[0]["meta"]["text_register"] == "reviews"
    assert rows[0]["meta"]["rating"] == 4 and rows[0]["meta"]["length"] == 200


def test_load_twgov_maps_records_and_filters_floor(monkeypatch):
    from greyscope.v2 import twgov

    arts = [
        {"s": "ABC123", "title": "市政測試新聞", "unit": "臺北市政府測試局",
         "pub_date": "2021-12-31", "body": "測" * 200},   # clears the 150-CJK floor
        {"s": "X1", "title": "短", "unit": "局", "pub_date": "2020-01-01", "body": "測" * 10},
    ]
    monkeypatch.setattr(twgov, "fetch_news", lambda *, limit=None: arts)
    rows = [r.to_row() for r in corpora.load_twgov()]
    assert len(rows) == 1                                              # short article dropped
    assert rows[0]["text_id"] == "zh-tw/tw-gov/ABC123/human_written"
    assert rows[0]["language"] == "zh-tw" and rows[0]["meta"]["text_register"] == "journalistic"
    assert rows[0]["meta"]["pub_date"] == "2021-12-31" and rows[0]["meta"]["unit"] == "臺北市政府測試局"


def test_load_editlens_split_ingests_all_classes_split_tagged(monkeypatch):
    import pyarrow as pa

    long = "lorem ipsum dolor sit amet consectetur " * 16  # > the 75-word EN floor
    fake = pa.table({
        "text": [long, long, long],
        "text_type": ["human_written", "ai_generated", "ai_edited"],
        "source": ["news", "news", "enron"],
        "source_id": [1, 2, 3],
        "model": ["human", "x-ai/grok", "x-ai/grok"],
        "cosine_score": [0.0, 1.0, 0.2],
    })
    monkeypatch.setattr(corpora, "_read_editlens_arrow", lambda split: fake)
    rows = corpora.load_editlens_split("test_enron")
    assert {r["text_type"] for r in rows} == {"human_written", "ai_generated", "ai_edited"}
    assert all(r["split"] == "test_enron" and r["language"] == "en" and r["source"] == "editlens" for r in rows)
    assert rows[0]["model"] is None         # human → model null
    assert rows[1]["model"] == "x-ai/grok"  # AI keeps its generator
    assert rows[2]["cosine_score"] == 0.2   # EditLens's own label kept


def test_load_editlens_ai_balances_classes_and_buckets_by_class(monkeypatch):
    import pyarrow as pa

    long = "lorem ipsum dolor sit amet consectetur " * 16
    fake = pa.table({
        "text": [long] * 6,
        "text_type": ["human_written", "ai_generated", "ai_generated", "ai_edited", "ai_edited", "ai_edited"],
        "source": ["news"] * 6,
        "source_id": list(range(6)),
        "model": ["human", "g1", "g2", "g1", "g2", "g3"],
        "cosine_score": [0.0, 0.9, 0.8, 0.2, 0.3, 0.4],
    })
    monkeypatch.setattr(corpora, "_read_editlens_arrow", lambda split: fake)
    rows = corpora.load_editlens_ai(limit=2)  # per_type = 1 → 1 generated + 1 edited, no humans
    assert [r["text_type"] for r in rows] == ["ai_generated", "ai_edited"]
    assert all(r["language"] == "en" and r["source"] == "editlens" and r["meta"]["backbone"] for r in rows)
    assert rows[0]["cosine_score"] == 1.0      # generated → 1.0 by class (not EditLens's 0.9)
    assert rows[1]["cosine_score"] == 0.2      # edited keeps EditLens's Linq cosine
    assert all("split" not in r for r in rows)  # no split tag → assembled by source-doc fallback


# --- EN permissive backbone (pure helpers + loaders) -------------------------
def test_strip_gutenberg_boilerplate_removes_license_frame():
    raw = ("Header junk\n*** START OF THE PROJECT GUTENBERG EBOOK FOO ***\n"
           "Real body text here.\n*** END OF THE PROJECT GUTENBERG EBOOK FOO ***\nLicense tail")
    assert corpora.strip_gutenberg_boilerplate(raw) == "Real body text here."


def test_first_passage_en_skips_front_matter_and_accumulates_to_target():
    body = "\n\n".join(["Title Page", "Contents", "Preface"] + ["word " * 150] * 3)
    passage = corpora.first_passage_en(body, target_words=250, skip_paragraphs=3)
    assert "Title Page" not in passage and "Preface" not in passage  # front matter skipped
    assert passage.count("word") == 300  # two 150-word paragraphs accumulate past the 250 target


def test_first_passage_en_caps_giant_block_without_paragraph_breaks():
    # A book with no blank-line breaks collapses to one giant block; the max_chars cap must bound it
    # (the gutenberg 151k-char "passage" bug).
    text = "\n\n".join(["front1", "front2", "front3", "word " * 40000])  # 3 skipped + one giant block
    passage = corpora.first_passage_en(text, target_words=300, max_chars=4000)
    assert len(passage) == 4000  # capped, not the ~200k giant


def test_join_paragraphs_handles_sequence_string_and_none():
    assert corpora.join_paragraphs(["a", "", "b"]) == "a\nb"
    assert corpora.join_paragraphs("flat") == "flat"
    assert corpora.join_paragraphs(None) == ""


def test_parse_blob_meta_json_literal_dict_and_garbage():
    assert corpora.parse_blob_meta('{"id": 42, "title": "X"}') == {"id": 42, "title": "X"}
    assert corpora.parse_blob_meta("{'id': 7}") == {"id": 7}  # python-literal fallback
    assert corpora.parse_blob_meta("not a dict") == {}
    assert corpora.parse_blob_meta({"already": "dict"}) == {"already": "dict"}
    assert corpora.parse_blob_meta(None) == {}


def test_load_fineweb_en_maps_records_and_filters_floor(monkeypatch):
    exs = [
        {"id": "cc-1", "text": "word " * 80, "url": "http://x", "date": "2019-05-01T00:00:00Z"},
        {"id": "cc-2", "text": "short", "url": "http://y", "date": "2019-05-02"},  # under floor
    ]
    monkeypatch.setattr(corpora, "_iter_hf", lambda *a, **k: iter(exs))
    rows = [r.to_row() for r in corpora.load_fineweb_en(dumps=("CC-MAIN-2019-18",))]
    assert len(rows) == 1
    assert rows[0]["text_id"] == "en/fineweb/cc-1/human_written"
    assert rows[0]["language"] == "en" and rows[0]["meta"]["text_register"] == "formal"
    assert rows[0]["meta"]["dump"] == "CC-MAIN-2019-18" and rows[0]["meta"]["date"] == "2019-05-01"


def test_load_arxiv_abstracts_en_maps_records(monkeypatch):
    exs = [{"id": "2401.00001", "text": "word " * 80, "created": "2024-01-01T00:00:00"},
           {"id": "x", "text": "tiny", "created": "2024-01-02"}]
    monkeypatch.setattr(corpora, "_iter_hf", lambda *a, **k: iter(exs))
    rows = [r.to_row() for r in corpora.load_arxiv_abstracts_en()]
    assert len(rows) == 1
    assert rows[0]["text_id"] == "en/arxiv-abstracts/2401.00001/human_written"
    assert rows[0]["meta"]["text_register"] == "formal" and rows[0]["meta"]["created"] == "2024-01-01"


def test_load_wikinews_en_filters_lang_and_joins_paragraph_sequence(monkeypatch):
    exs = [
        {"pageid": 100, "lang": "en", "title": "Big Story", "date": "2015-06-01",
         "text": ["word " * 40, "word " * 40]},                # en, joined clears the 75-word floor
        {"pageid": 200, "lang": "fr", "title": "Histoire", "date": "2015-06-02",
         "text": ["mot " * 100]},                               # non-en dropped
    ]
    monkeypatch.setattr(corpora, "_iter_hf", lambda *a, **k: iter(exs))
    rows = [r.to_row() for r in corpora.load_wikinews_en()]
    assert len(rows) == 1
    assert rows[0]["text_id"] == "en/wikinews-en/100/human_written"
    assert rows[0]["meta"]["text_register"] == "journalistic" and rows[0]["meta"]["topic"] == "Big Story"


def test_load_gutenberg_en_strips_boilerplate_takes_passage_and_id(monkeypatch):
    body = ("*** START OF THE PROJECT GUTENBERG EBOOK T ***\n\n"
            "Front\n\nMatter\n\nSkip\n\n" + "word " * 200 + "\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK T ***")
    exs = [{"TEXT": body, "SOURCE": "gutenberg",
            "METADATA": '{"language": "en", "text_id": 1342, "title": "Pride", "issued": "1813-01-28"}'}]
    monkeypatch.setattr(corpora, "_iter_hf", lambda *a, **k: iter(exs))
    rows = [r.to_row() for r in corpora.load_gutenberg_en(target_words=100)]
    assert len(rows) == 1
    assert rows[0]["text_id"] == "en/gutenberg/1342/human_written"       # text_id parsed from METADATA
    assert rows[0]["meta"]["text_register"] == "creative" and rows[0]["meta"]["topic"] == "Pride"
    assert "PROJECT GUTENBERG" not in rows[0]["text"]                    # frame stripped


def test_load_stackexchange_en_uses_answer_text_and_hashes_id(monkeypatch):
    exs = [{"INSTRUCTION": "How do I X?", "RESPONSE": "word " * 80, "SOURCE": "stackoverflow"},
           {"INSTRUCTION": "Q", "RESPONSE": "short", "SOURCE": "serverfault"}]  # under floor
    monkeypatch.setattr(corpora, "_iter_hf", lambda *a, **k: iter(exs))
    rows = [r.to_row() for r in corpora.load_stackexchange_en()]
    assert len(rows) == 1
    assert rows[0]["source"] == "stackexchange" and rows[0]["language"] == "en"
    assert rows[0]["meta"]["text_register"] == "casual" and rows[0]["meta"]["site"] == "stackoverflow"
    assert len(rows[0]["source_id"]) == 16  # content hash (no row id in the source)


def test_load_amazon_reviews_en_maps_records_and_filters_floor(monkeypatch):
    exs = [{"text": "word " * 80, "rating": 5.0, "parent_asin": "B01", "user_id": "U1"},
           {"text": "meh", "rating": 1.0, "parent_asin": "B02", "user_id": "U2"}]  # under floor
    monkeypatch.setattr(corpora, "_iter_amazon_en", lambda cat: iter(exs))
    rows = [r.to_row() for r in corpora.load_amazon_reviews_en(categories=("Books",))]
    assert len(rows) == 1
    assert rows[0]["text_id"] == "en/amazon-reviews-en/B01/U1/human_written"
    assert rows[0]["meta"]["text_register"] == "reviews" and rows[0]["meta"]["rating"] == 5.0
    assert rows[0]["meta"]["category"] == "Books"


def test_load_free_ai_en_emits_ai_generated_backbone_rows(monkeypatch):
    pools = {"cosmopedia": [{"text": "word " * 80}], "helpsteer2": [{"response": "word " * 80}]}

    def fake_iter(path, config, split):
        return iter(pools["cosmopedia" if "cosmopedia" in path else "helpsteer2"])

    monkeypatch.setattr(corpora, "_iter_hf", fake_iter)
    rows = corpora.load_free_ai_en()
    assert {r["source"] for r in rows} == {"cosmopedia", "helpsteer2"}
    assert all(r["text_type"] == "ai_generated" and r["cosine_score"] == 1.0 for r in rows)
    assert all(r["language"] == "en" and r["meta"]["free_ai"] and r["source_text"] is None for r in rows)
    cosmo = next(r for r in rows if r["source"] == "cosmopedia")
    assert cosmo["model"] == "mistralai/Mixtral-8x7B-Instruct-v0.1"
    assert cosmo["text_id"].startswith("en/cosmopedia/") and cosmo["text_id"].endswith("/ai_generated/backbone")
