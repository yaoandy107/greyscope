"""Tests for the quality gates (greyscope/pipeline/gates.py).

Covers each per-sample gate's asymmetry (human rows bypass AI checks; edits exempt
from length/regurgitation), the zh-TW Simplified script gate, within-text_type
near-dedup, and the run_gates orchestration + drop reasons.
"""

from greyscope.pipeline import gates


def _row(text, text_type="ai_generated", language="en", source_text=None,
         finish_reason="stop", source="editlens"):
    return {
        "text": text, "text_type": text_type, "language": language,
        "source": source, "source_id": "x", "source_text": source_text,
        "meta": {"finish_reason": finish_reason},
    }


def test_gate_refusal_drops_decline_keeps_positive_content():
    assert not gates.gate_refusal(_row("I'm sorry, I can't help with that.")).keep
    assert not gates.gate_refusal(_row("申し訳ありませんが、お手伝いできません。", language="ja")).keep
    # verb-anchored → positive content that merely contains "can't"/"sorry"/CJK negatives survives
    assert gates.gate_refusal(_row("I can't recommend this bakery enough — my favorite.")).keep
    assert gates.gate_refusal(_row("I am sorry to say it, but this was the best show of my life.")).keep
    assert gates.gate_refusal(_row("悔しくて我慢できません。最高でした。", language="ja")).keep
    assert gates.gate_refusal(_row("I cannot help", text_type="human_written")).keep  # human bypass


def test_gate_truncated():
    assert not gates.gate_truncated(_row("cut off mid", finish_reason="length")).keep
    assert gates.gate_truncated(_row("a complete thought.", finish_reason="stop")).keep


def test_gate_script_simplified_dropped_traditional_kept():
    simp = _row("这是一个测试，国家发展很快。" * 5, language="zh-tw")
    trad = _row("這是一個測試，國家發展很快。" * 5, language="zh-tw")
    assert not gates.gate_script(simp).keep
    assert gates.gate_script(trad).keep
    assert gates.gate_script(_row("这是简体的人类文字", text_type="human_written", language="zh-tw")).keep


def test_gate_length_match():
    src = "word " * 100
    assert gates.gate_length_match(_row("word " * 100, source_text=src)).keep
    assert not gates.gate_length_match(_row("word " * 10, source_text=src)).keep
    assert not gates.gate_length_match(_row("word " * 400, source_text=src)).keep
    assert gates.gate_length_match(_row("word " * 10, text_type="ai_edited", source_text=src)).keep  # edit exempt


def test_gate_regurgitation():
    src = " ".join(f"w{i}" for i in range(50))
    assert not gates.gate_regurgitation(_row(src + " tail", source_text=src)).keep  # verbatim run
    fresh = _row("w0 w1 w2 then wholly different prose on the same general subject matter", source_text=src)
    assert gates.gate_regurgitation(fresh).keep  # only short topical overlap
    assert gates.gate_regurgitation(_row(src, text_type="ai_edited", source_text=src)).keep  # edit exempt


def test_gate_noop_edit_drops_verbatim_keeps_real_edit():
    src = "The harvest festival runs every autumn in the village square."
    assert not gates.gate_noop_edit(_row(src, text_type="ai_edited", source_text=src)).keep
    assert not gates.gate_noop_edit(  # whitespace-only diff still counts as no edit
        _row(f"  {src}\n", text_type="ai_edited", source_text=src)).keep
    assert gates.gate_noop_edit(  # a real change survives
        _row("Every fall the village square hosts a harvest festival.",
             text_type="ai_edited", source_text=src)).keep
    assert gates.gate_noop_edit(_row(src, text_type="ai_generated", source_text=src)).keep  # mirror exempt


def test_near_dedup_within_type_only():
    a = _row("the quick brown fox jumps over the lazy dog every single morning")
    b = _row("the quick brown fox jumps over the lazy dog every single morning")
    kept, dropped = gates.near_dedup([a, b])
    assert len(kept) == 1 and len(dropped) == 1
    assert dropped[0]["drop_reason"] == "near_duplicate"
    # identical text across DIFFERENT text_types (a human↔edited pair) is NOT deduped
    pair = "the same paraphrased sentence content lives here unchanged"
    kept2, dropped2 = gates.near_dedup([
        _row(pair, text_type="human_written"), _row(pair, text_type="ai_edited")])
    assert len(kept2) == 2 and not dropped2


def test_near_dedup_short_text_exact_fallback():
    a, b = _row("hi there friend"), _row("hi there friend")  # 3 words < EN shingle window
    c = _row("a totally different short line")
    kept, dropped = gates.near_dedup([a, b, c])
    assert len(kept) == 2 and len(dropped) == 1
    assert dropped[0]["drop_reason"] == "near_duplicate"


def test_run_gates_end_to_end():
    rows = [
        _row("A perfectly normal generated paragraph about gardening tips and tools."),
        _row("I'm sorry, I can't help with that."),
        _row("cut off", finish_reason="length"),
        _row("a normal human passage about the harvest", text_type="human_written"),
    ]
    kept, dropped = gates.run_gates(rows)
    assert {"refusal", "truncated"} <= {d["drop_reason"] for d in dropped}
    assert "a normal human passage about the harvest" in {r["text"] for r in kept}
