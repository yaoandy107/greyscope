"""Tests for the verbatim OpenPangram preprocess port.

Source of truth: https://github.com/pangramlabs/EditLens/blob/main/scripts/preprocess.py
If these tests fail after upstream changes, update both the source and the tests.
"""

from greyscope.preprocess import (
    clean_text,
    count_words,
    normalize_emoji,
    normalize_whitespace,
    remove_ai_header,
    remove_think_tag,
    score_to_bucket,
)


def test_normalize_whitespace_collapses_runs():
    assert normalize_whitespace("a  b\t\tc\n\nd") == "a b c d"
    assert normalize_whitespace("  hello  ") == "hello"


def test_normalize_emoji_demojizes():
    assert normalize_emoji("hi :)") == "hi :)"
    assert "thumbs_up" in normalize_emoji("ok 👍")


def test_remove_think_tag_strips_pre_think_content():
    assert remove_think_tag("scratch</think>final answer") == "final answer"
    assert remove_think_tag("no tag here") == "no tag here"


def test_remove_ai_header_strips_boilerplate_first_paragraph():
    text = "Sure! Here is the answer.\n\nThe actual content."
    assert remove_ai_header(text) == "The actual content."


def test_remove_ai_header_keeps_text_without_boilerplate():
    text = "Real content paragraph.\n\nMore content."
    assert remove_ai_header(text) == text


def test_remove_ai_header_handles_empty_input():
    assert remove_ai_header("") == ""
    assert remove_ai_header("   \n  \n") == "   \n  \n"


def test_remove_ai_header_handles_single_boilerplate_paragraph():
    text = "Sure! Here is the answer."
    assert remove_ai_header(text) == text


def test_clean_text_composes_all_steps():
    text = "🤔 thinking...</think>Sure! Here you go.\n\nThe Real Answer  Has  Spaces."
    result = clean_text(text)
    assert "</think>" not in result
    assert "sure" not in result.split("\n", 1)[0]
    assert result == result.lower()
    assert "  " not in result


def test_clean_text_lowercases():
    assert clean_text("HELLO World") == "hello world"


def test_count_words_basic():
    assert count_words("hello world") == 2
    assert count_words("a-b c d") == 4
    assert count_words("") == 0


def test_score_to_bucket_boundaries():
    assert score_to_bucket(0.0, 4, 0.03, 0.15) == 0
    assert score_to_bucket(0.03, 4, 0.03, 0.15) == 0
    assert score_to_bucket(0.15, 4, 0.03, 0.15) == 3
    assert score_to_bucket(1.0, 4, 0.03, 0.15) == 3


def test_score_to_bucket_middle_range():
    bucket_at_low = score_to_bucket(0.04, 4, 0.03, 0.15)
    bucket_at_high = score_to_bucket(0.14, 4, 0.03, 0.15)
    assert bucket_at_low in (1, 2)
    assert bucket_at_high in (1, 2)
    assert bucket_at_low <= bucket_at_high
