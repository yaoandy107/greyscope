import csv
import json

from greyscope.v2 import paraphrase


def _row(i, lang="en", text_type="ai_generated", bucket=3, text="Some AI text here."):
    return {"text_id": f"{lang}/src/{i}/{text_type}/m/p", "text": text, "language": lang,
            "text_type": text_type, "source": "src", "source_id": str(i), "model": "m",
            "prompt_id": "p", "markdown_mode": "default", "cosine_score": 1.0,
            "bucket": bucket, "meta": {}}


def test_select_ai_rows_skips_humans_and_is_deterministic():
    rows = [_row(i) for i in range(10)] + [_row(99, text_type="human_written", bucket=0)]
    picked = paraphrase.select_ai_rows(rows, 5)
    assert len(picked) == 5
    assert all(r["text_type"] != "human_written" for r in picked)
    again = paraphrase.select_ai_rows(list(reversed(rows)), 5)
    assert {r["text_id"] for r in picked} == {r["text_id"] for r in again}


def test_select_ai_rows_stratifies_across_cells():
    rows = ([_row(i, text_type="ai_generated", bucket=3) for i in range(20)]
            + [_row(100 + i, text_type="ai_edited", bucket=1) for i in range(20)])
    picked = paraphrase.select_ai_rows(rows, 10)
    types = {r["text_type"] for r in picked}
    assert types == {"ai_generated", "ai_edited"}


def test_select_ai_rows_caps_at_pool_size():
    assert len(paraphrase.select_ai_rows([_row(1), _row(2)], 100)) == 2


def test_build_messages_uses_language_prompt():
    msgs = paraphrase.build_messages(_row(1, lang="ja"))
    assert msgs[0]["content"] == paraphrase._PROMPT["ja"]
    assert msgs[1]["content"] == "Some AI text here."


def test_paraphrase_row_keeps_label_and_records_provenance():
    row = _row(1, text_type="ai_edited", bucket=2)
    shaped = paraphrase.paraphrase_row(row, paraphrase.AUG_MODELS[0], "Rewritten AI text here!")
    assert shaped["bucket"] == 2 and shaped["text_type"] == "ai_edited"
    assert shaped["text_id"] == row["text_id"] + "/para"
    assert shaped["meta"]["paraphrased_by"] == paraphrase.AUG_MODELS[0]["slug"]
    assert shaped["meta"]["augmentation"] == "paraphrase"


def test_paraphrase_row_drops_echo_empty_and_bad_length():
    row = _row(1)
    assert paraphrase.paraphrase_row(row, paraphrase.AUG_MODELS[0], row["text"]) is None
    assert paraphrase.paraphrase_row(row, paraphrase.AUG_MODELS[0], "   ") is None
    assert paraphrase.paraphrase_row(row, paraphrase.AUG_MODELS[0], "x" * 500) is None


def test_paraphrase_row_strips_chat_wrapper():
    shaped = paraphrase.paraphrase_row(
        _row(1), paraphrase.AUG_MODELS[0], "Here is the rewritten text:\nA fresh AI text now.")
    assert shaped["text"] == "A fresh AI text now."


def test_estimate_cost_scales_with_text_and_halves_flex():
    rows = [_row(i, text="word " * 200) for i in range(10)]
    prices = {paraphrase.AUG_MODELS[0]["slug"]: (1e-6, 4e-6),
              paraphrase.ATTACK_MODELS[0]["slug"]: (1e-6, 4e-6)}
    aug = paraphrase.estimate_cost(rows, [paraphrase.AUG_MODELS[0]], prices)
    attack = paraphrase.estimate_cost(rows, [paraphrase.ATTACK_MODELS[0]], prices)
    assert aug["rows"] == 10 and aug["tokens_in"] > aug["tokens_out"] > 0
    assert abs(attack["cost"] - 2 * aug["cost"]) < 1e-9  # flex halves


def test_write_and_read_roundtrip(tmp_path):
    rows = [paraphrase.paraphrase_row(_row(1), paraphrase.AUG_MODELS[0], "Alt AI text body.")]
    paraphrase.write_rows(rows, tmp_path / "aug.csv")
    with (tmp_path / "aug.csv").open() as fh:
        back = list(csv.DictReader(fh))
    assert back[0]["text"] == "Alt AI text body."
    assert json.loads(back[0]["meta"])["augmentation"] == "paraphrase"


def test_model_for_is_seeded_and_covers_pool():
    rows = [_row(i) for i in range(60)]
    picks = {paraphrase.model_for(r, paraphrase.AUG_MODELS)["slug"] for r in rows}
    assert picks == {m["slug"] for m in paraphrase.AUG_MODELS}
    assert paraphrase.model_for(rows[0], paraphrase.AUG_MODELS) is paraphrase.model_for(
        rows[0], paraphrase.AUG_MODELS)


def test_paraphrase_row_drops_simplified_zh():
    row = _row(1, lang="zh-tw", text="這是一段繁體中文的測試文字，內容關於天氣與生活。")
    bad = "这是一段简体中文的输出，关于天气与生活的说明文字。"
    assert paraphrase.paraphrase_row(row, paraphrase.AUG_MODELS[0], bad) is None


def test_rescore_edited_rebuckets_and_drops_orphans():
    src = "The original human paragraph about weather patterns."
    sources = {("en", "src", "1"): src}
    edited = _row(1, text_type="ai_edited", bucket=1, text="A big rewrite of it all.")
    orphan = _row(2, text_type="ai_edited", bucket=1)
    generated = _row(3, text_type="ai_generated", bucket=3)

    def fake_embed(texts, model=None):
        return [[1.0, 0.0] if t.startswith("The original") else [0.0, 1.0] for t in texts]

    kept = paraphrase.rescore_edited([edited, orphan, generated], sources, embed_fn=fake_embed)
    assert {r["text_id"] for r in kept} == {edited["text_id"], generated["text_id"]}
    re_edited = next(r for r in kept if r["text_type"] == "ai_edited")
    assert re_edited["cosine_score"] == 1.0  # orthogonal embeddings → full-rewrite magnitude
    assert re_edited["bucket"] == 3
    assert next(r for r in kept if r["text_type"] == "ai_generated")["bucket"] == 3


def test_sample_humans_balanced_and_seeded(tmp_path, monkeypatch):
    # CSV-read rows carry strings for bucket/cosine_score
    rows = ([{**_row(i, lang="en", text_type="human_written"), "bucket": "0", "cosine_score": ""}
             for i in range(10)]
            + [{**_row(i, lang="ja", text_type="human_written"), "bucket": "0", "cosine_score": ""}
               for i in range(10, 18)]
            + [_row(i, text_type="ai_generated") for i in range(20, 25)])
    monkeypatch.setattr(paraphrase, "read_split", lambda name, splits_dir=None: rows)
    picked = paraphrase.sample_humans("val", 5)
    assert all(r["text_type"] == "human_written" for r in picked)
    assert all(r["bucket"] == 0 and r["cosine_score"] is None for r in picked)  # coerced
    from collections import Counter
    assert Counter(r["language"] for r in picked) == {"en": 5, "ja": 5}
    assert {r["text_id"] for r in picked} == {r["text_id"] for r in paraphrase.sample_humans("val", 5)}
