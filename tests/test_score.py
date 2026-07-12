"""Tests for edit-magnitude scoring (greyscope/pipeline/score.py).

Pure: cosine/edit-magnitude math + the pairing logic via an injected fake embedder
(no network). Guards that only ai_edited rows get scored and that identical texts
embed once.
"""

import pytest

from greyscope.pipeline import score


def test_cosine_identity_orthogonal_and_zero_guard():
    assert score.cosine([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert score.cosine([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
    assert score.cosine([0, 0, 0], [1, 0, 0]) == 0.0  # zero-norm guard


def test_edit_magnitude_bounds():
    assert score.edit_magnitude([1, 0], [1, 0]) == pytest.approx(0.0)  # no change
    assert score.edit_magnitude([1, 0], [0, 1]) == pytest.approx(1.0)  # wholesale


def test_score_edited_fills_only_edited_class():
    rows = [
        {"text_type": "ai_edited", "source_text": "abc", "text": "abd"},
        {"text_type": "ai_generated", "source_text": "abc", "text": "xyz", "cosine_score": 1.0},
        {"text_type": "human_written", "text": "abc", "cosine_score": 0.0},
    ]
    table = {"abc": [1.0, 0.0], "abd": [0.9, 0.1], "xyz": [0.0, 1.0]}

    def fake_embed(texts, model=None):
        return [table[t] for t in texts]

    score.score_edited(rows, embed_fn=fake_embed)
    assert rows[0]["cosine_score"] == pytest.approx(1 - score.cosine([1, 0], [0.9, 0.1]))
    assert rows[1]["cosine_score"] == 1.0  # generated untouched
    assert rows[2]["cosine_score"] == 0.0  # human untouched


def test_score_edited_dedups_texts_before_embedding():
    seen = {}

    def fake_embed(texts, model=None):
        seen["texts"] = texts
        return [[1.0, 0.0]] * len(texts)

    rows = [
        {"text_type": "ai_edited", "source_text": "same", "text": "same"},
        {"text_type": "ai_edited", "source_text": "same", "text": "other"},
    ]
    score.score_edited(rows, embed_fn=fake_embed)
    assert seen["texts"].count("same") == 1  # 3 references → embedded once
    assert sorted(seen["texts"]) == ["other", "same"]


def test_score_edited_truncates_oversized_inputs_before_embedding():
    # A few fineweb/gutenberg sources exceed the provider's embed input limit → truncate before send.
    long_src = "a" * (score.EMBED_MAX_CHARS + 500)
    seen = {}

    def fake_embed(texts, model=None):
        seen["texts"] = texts
        return [[1.0, 0.0]] * len(texts)

    rows = [{"text_type": "ai_edited", "source_text": long_src, "text": "short edit"}]
    score.score_edited(rows, embed_fn=fake_embed)
    assert all(len(t) <= score.EMBED_MAX_CHARS for t in seen["texts"])  # capped before the provider
    assert rows[0]["cosine_score"] is not None  # still scored, no crash
