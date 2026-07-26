import numpy as np
import pytest

from greyscope.detectors import (
    fakespot_clean_text,
    graded_scores_from_logits,
    scores_from_logits,
)


def test_scores_from_binary_logits_uses_declared_ai_class():
    logits = np.array([[4.0, 0.0], [0.0, 4.0]])
    ai_zero = scores_from_logits(logits, ai_label_id=0)
    ai_one = scores_from_logits(logits, ai_label_id=1)
    assert ai_zero[0] > ai_zero[1]
    assert ai_one[1] > ai_one[0]


def test_scores_from_single_logit_uses_sigmoid():
    scores = scores_from_logits([[-2.0], [2.0]])
    assert scores[0] < 0.5 < scores[1]


def test_scores_from_graded_logits_is_expected_rank():
    scores = scores_from_logits([[10.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 10.0]])
    assert scores[0] == pytest.approx(0.0, abs=0.001)
    assert scores[1] == pytest.approx(1.0, abs=0.001)


def test_graded_scores_decode_v1_and_v2_heads():
    seqcls = graded_scores_from_logits(
        [[4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]],
        head_type="seqcls",
    )
    corn = graded_scores_from_logits(
        [[-8.0, -8.0, -8.0], [8.0, 8.0, 8.0]],
        head_type="corn",
    )
    assert seqcls[0] < 0.1 < 0.9 < seqcls[1]
    assert corn[0] < 0.1 < 0.9 < corn[1]


def test_fakespot_cleanup_matches_published_contract():
    source = "# Heading\n> quote\nA **bold** [link](https://example.com) &amp; `code` |x|"
    assert fakespot_clean_text(source) == "Heading A bold link & "
