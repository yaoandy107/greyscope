"""Tests for assembly (greyscope/pipeline/assemble.py): per-language bucketing, split-by-source-doc
(derivatives co-locate, the edit's tag drives the split), the build-only field strip, and
CSV + prompt-manifest emission."""

import csv
import json

from greyscope.pipeline import assemble


def _doc(lang, source, sid, *, edit_cosine, split_tag):
    def row(text_type, cosine, **extra):
        meta = {"length": 100, "served_tier": "flex", "finish_reason": "stop", "usage": {"x": 1}, **extra}
        return {
            "text_id": f"{lang}/{source}/{sid}/{text_type}", "text": f"{text_type} text",
            "language": lang, "text_type": text_type, "source": source, "source_id": sid,
            "source_text": "the human original",
            "model": None if text_type == "human_written" else "m",
            "prompt_id": None if text_type == "human_written" else "plain",
            "markdown_mode": "default", "cosine_score": cosine, "bucket": None, "meta": meta,
        }

    return [
        row("human_written", 0.0),
        row("ai_generated", 1.0),
        row("ai_edited", edit_cosine, split_tag=split_tag, edit_prompt_id="e1", edit_category="paraphrase"),
    ]


def test_assign_buckets_by_class_and_cosine():
    rows = _doc("en", "editlens", "d1", edit_cosine=0.09, split_tag="train")
    assemble.assign_buckets(rows)
    by_type = {r["text_type"]: r["bucket"] for r in rows}
    assert by_type["human_written"] == 0       # cosine 0 → bucket 0
    assert by_type["ai_generated"] == 3        # cosine 1 → bucket 3
    assert 1 <= by_type["ai_edited"] <= 2      # 0.09 ∈ (0.030, 0.150) → middle


def test_split_by_source_doc_co_locates_derivatives():
    rows = (_doc("en", "editlens", "d1", edit_cosine=0.05, split_tag="val")
            + _doc("en", "editlens", "d2", edit_cosine=0.05, split_tag="test"))
    assemble.assign_splits(rows)
    assert {r["split"] for r in rows if r["source_id"] == "d1"} == {"val"}   # whole doc follows its edit
    assert {r["split"] for r in rows if r["source_id"] == "d2"} == {"test"}


def test_to_split_row_strips_build_only_fields():
    edit = _doc("en", "editlens", "d1", edit_cosine=0.05, split_tag="train")[2]
    edit["meta"]["shippable_edit"] = True
    shipped = assemble.to_split_row(edit)
    assert "source_text" not in shipped
    for k in ("served_tier", "finish_reason", "usage", "shippable_edit"):  # licensing tag is build-only
        assert k not in shipped["meta"]
    assert shipped["meta"]["edit_category"] == "paraphrase"   # provenance kept


def test_dedupe_text_ids_keeps_first_when_a_source_reemits_an_id():
    # gov.taipei re-emits a few `s` ids with an edited body → same text_id, DIFFERENT text
    # (text-based dedupe_splits can't see it); the id guard must keep the first, drop the rest.
    rows = [
        {"text_id": "zh-tw/tw-gov/A/human_written", "text": "body one"},
        {"text_id": "zh-tw/tw-gov/A/human_written", "text": "body two (edited re-list)"},
        {"text_id": "zh-tw/tw-gov/B/human_written", "text": "unique"},
    ]
    kept, dropped = assemble.dedupe_text_ids(rows)
    assert dropped == 1
    assert [r["text_id"] for r in kept] == ["zh-tw/tw-gov/A/human_written", "zh-tw/tw-gov/B/human_written"]
    assert kept[0]["text"] == "body one"                 # first occurrence wins
    assert len({r["text_id"] for r in kept}) == len(kept)  # invariant: globally unique text_id


def test_drop_unshippable_edits_removes_mirror_only_source_edits():
    rows = (_doc("ja", "wiki40b-ja", "d1", edit_cosine=0.05, split_tag="train")
            + _doc("ja", "aozora", "d2", edit_cosine=0.05, split_tag="train"))
    for r in rows:  # tag edits as generate.edit_row would: wiki40b CC BY-SA mirror-only, aozora ships
        if r["text_type"] == "ai_edited":
            r["meta"]["shippable_edit"] = (r["source"] == "aozora")
    kept = assemble.drop_unshippable_edits(rows)
    wiki = {r["text_type"] for r in kept if r["source"] == "wiki40b-ja"}
    assert wiki == {"human_written", "ai_generated"}      # restricted edit dropped, human+mirror stay
    aozora = {r["text_type"] for r in kept if r["source"] == "aozora"}
    assert "ai_edited" in aozora                          # permissive edit kept


def test_write_splits_and_manifest(tmp_path):
    rows = (_doc("en", "editlens", "d1", edit_cosine=0.05, split_tag="train")
            + _doc("en", "editlens", "d2", edit_cosine=0.05, split_tag="val"))
    assemble.assign_buckets(rows)
    assemble.assign_splits(rows)
    assert assemble.write_splits(rows, tmp_path) == {"train": 3, "val": 3}  # empty splits get no file

    with (tmp_path / "train.csv").open() as fh:
        train = list(csv.DictReader(fh))
    assert len(train) == 3 and "source_text" not in train[0]
    assert json.loads(train[0]["meta"])  # meta is JSON-encoded in its column

    manifest = json.loads(assemble.write_prompt_manifest(tmp_path).read_text())
    ids = {(e["type"], e["language"], e["id"]) for e in manifest}
    assert ("edit", "en", "en-paraphrase-1") in ids   # edit prompts resolved
    assert ("system", "en", "humanizer") in ids       # styles resolved
    assert any(e["type"] == "mirror" for e in manifest)


def test_held_out_generator_goes_to_ood_split():
    rows = (_doc("ja", "wiki40b-ja", "d1", edit_cosine=0.05, split_tag="train")
            + _doc("ja", "wiki40b-ja", "d2", edit_cosine=0.05, split_tag="train"))
    for r in rows:  # make d1 the held-out generator's doc
        if r["source_id"] == "d1" and r["model"]:
            r["model"] = assemble.OOD_GENERATOR["ja"]
    assemble.assign_splits(rows)
    assert {r["split"] for r in rows if r["source_id"] == "d1"} == {"ood_generator"}  # whole doc reserved
    assert {r["split"] for r in rows if r["source_id"] == "d2"} == {"train"}          # normal doc → edit tag


def test_dedupe_splits_removes_cross_and_within_split_exact_texts():
    def r(split, text, text_type="human_written"):
        return {"text": text, "text_type": text_type, "language": "en", "split": split}

    rows = [
        r("train", "shared boilerplate line"),   # kept (highest priority)
        r("val", "shared boilerplate line"),      # dropped — same text, lower-priority split
        r("test", "shared boilerplate line"),     # dropped
        r("train", "shared boilerplate line", "ai_generated"),  # dropped — within-split exact dup
        r("val", "unique to val"),                # kept
        r("test_llama", "shared boilerplate line"),  # external eval split — NEVER touched
    ]
    kept, dropped = assemble.dedupe_splits(rows)
    assert dropped == 3
    by_split = {}
    for row in kept:
        by_split.setdefault(row["split"], []).append(row["text"])
    assert by_split["train"] == ["shared boilerplate line"]   # one copy, in train
    assert "shared boilerplate line" not in by_split.get("val", [])
    assert by_split["val"] == ["unique to val"]
    assert "test" not in by_split                              # its only row was a dup
    assert by_split["test_llama"] == ["shared boilerplate line"]  # external split preserved verbatim


def test_inherited_split_is_respected():
    rows = _doc("en", "editlens", "d1", edit_cosine=0.05, split_tag="train")
    for r in rows:
        r["split"] = "test_llama"   # as if ingested from EditLens's held-out split
    assemble.assign_splits(rows)
    assert {r["split"] for r in rows} == {"test_llama"}
