"""Tests for EN decontamination (greyscope/pipeline/decontam.py): word-n-gram containment,
the pre-generation filter (drop contaminated EN humans, keep clean, pass non-EN through),
RAID-CSV human extraction, the resumable-download size math, and the drop-reason shape.
The network download itself is exercised by the build's extraction step, not here."""

import csv

from greyscope.pipeline import decontam


def _row(text, *, language="en", source="editlens", sid="d1"):
    return {
        "text_id": f"{language}/{source}/{sid}/human_written", "text": text,
        "language": language, "text_type": "human_written", "source": source,
        "source_id": sid, "meta": {"length": len(text)},
    }


def test_overlap_distinguishes_reuse_from_unrelated():
    ref = decontam.build_reference(["the quick brown fox jumps over the lazy dog"], k=4)
    reused = "a story: the quick brown fox jumps over the lazy dog, allegedly"
    unrelated = "completely different words appear in this particular sentence here now"
    assert decontam.overlap_count(reused, ref, k=4) >= 3   # shares the whole verbatim span
    assert decontam.overlap_count(unrelated, ref, k=4) == 0


def test_filter_drops_contaminated_keeps_clean():
    eval_text = "machine generated text detection benchmark with many distinct tokens inside"
    ref = decontam.build_reference([eval_text], k=4)
    rows = [_row(eval_text, sid="dup"),
            _row("an entirely unrelated human paragraph about gardening and soil health", sid="clean")]
    clean, dropped = decontam.filter_english(rows, ref, k=4, min_shared=2)
    assert {r["source_id"] for r in dropped} == {"dup"}
    assert {r["source_id"] for r in clean} == {"clean"}
    assert dropped[0]["drop_reason"] == "contaminated"
    assert dropped[0]["meta"]["contam_overlap"] >= 2       # provenance: how strong the match was


def test_filter_keeps_incidental_single_phrase_overlap():
    # one shared 4-gram (a coincidental quote), then divergence → below min_shared → kept
    ref = decontam.build_reference(["alpha beta gamma delta epsilon zeta eta theta"], k=4)
    row = _row("alpha beta gamma delta then a totally different unrelated continuation here")
    clean, dropped = decontam.filter_english([row], ref, k=4, min_shared=2)
    assert clean and not dropped


def test_filter_passes_non_english_untouched():
    ref = decontam.build_reference(["whatever english reference text goes here for matching"], k=4)
    ja = _row("これは日本語の文章であり英語の参照とは無関係です", language="ja", source="wiki40b-ja")
    clean, dropped = decontam.filter_english([ja], ref, k=4, min_shared=2)
    assert clean == [ja] and not dropped                   # non-EN has no external target (zh/ja: internal)


def test_humans_from_csv_filters_and_dedupes(tmp_path):
    p = tmp_path / "raid_none.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["source_id", "model", "domain", "generation"])
        w.writeheader()
        w.writerow({"source_id": "s1", "model": "human", "domain": "news", "generation": "a human news article"})
        w.writerow({"source_id": "s1", "model": "human", "domain": "news", "generation": "same source id again"})
        w.writerow({"source_id": "s2", "model": "chatgpt", "domain": "news", "generation": "machine generated"})
        w.writerow({"source_id": "s3", "model": "human", "domain": "code", "generation": "int x = 0;"})
        w.writerow({"source_id": "s4", "model": "human", "domain": "reviews", "generation": "loved this product"})
    # s1 once (dedup), s2 dropped (not human), s3 dropped (non-EN code domain), s4 kept
    assert decontam._humans_from_csv(p) == ["a human news article", "loved this product"]


def test_text_cache_roundtrips_unicode_line_separators(tmp_path):
    # chr(0x2028)/chr(0x2029)/chr(0x85): str.splitlines() breaks on these but json.dumps emits them
    # raw, so the cache must split on newline only. This is the bug that crashed the first --full run.
    p = tmp_path / "c.jsonl"
    texts = ["plain", "a" + chr(0x2028) + "b", "c" + chr(10) + "d", "e" + chr(0x85) + "f",
             "g" + chr(0x2029) + "h"]
    decontam._write_text_cache(p, texts)
    assert decontam._read_text_cache(p) == texts


def test_content_total_prefers_content_range_then_length():
    assert decontam._content_total({"content-range": "bytes 100-11233/11234"}, 100) == 11234
    assert decontam._content_total({"content-length": "500"}, 0) == 500
    assert decontam._content_total({"content-length": "400"}, 100) == 500  # resume: have + remaining
    assert decontam._content_total({}, 0) is None
