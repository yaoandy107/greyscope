"""Tests for the CORN ordinal head decode + loss (greyscope/corn.py)."""

import numpy as np
import pytest

from greyscope.corn import (
    corn_bucket_probs,
    corn_cumulative_probs,
    corn_predict_buckets,
    corn_scalar_score,
)

# K=4 buckets -> 3 conditional logits per row.
BIG = 10.0


def test_scalar_score_in_unit_range():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(50, 3)) * 3
    s = corn_scalar_score(logits)
    assert s.shape == (50,)
    assert np.all((s >= 0.0) & (s <= 1.0))


def test_extremes_decode_to_end_buckets():
    logits = np.array([[BIG, BIG, BIG], [-BIG, -BIG, -BIG]])
    assert corn_predict_buckets(logits).tolist() == [3, 0]
    s = corn_scalar_score(logits)
    assert s[0] == pytest.approx(1.0, abs=1e-3)
    assert s[1] == pytest.approx(0.0, abs=1e-3)


def test_cumulative_probs_are_rank_consistent_monotone():
    # P(y>k) must be non-increasing in k for every row (the CORN guarantee).
    rng = np.random.default_rng(1)
    cum = corn_cumulative_probs(rng.normal(size=(100, 3)) * 2)
    assert np.all(np.diff(cum, axis=1) <= 1e-12)


def test_predicted_bucket_matches_threshold_count():
    # one positive then negative tasks -> exactly one P(y>k) above 0.5 -> bucket 1
    logits = np.array([[BIG, -BIG, -BIG], [BIG, BIG, -BIG]])
    assert corn_predict_buckets(logits).tolist() == [1, 2]


def test_bucket_probs_sum_to_one_and_nonnegative():
    rng = np.random.default_rng(2)
    p = corn_bucket_probs(rng.normal(size=(50, 3)) * 3)
    assert p.shape == (50, 4)  # K−1=3 logits decode to K=4 discrete bucket probs
    assert np.all(p >= -1e-12)  # non-negative by rank consistency
    assert np.allclose(p.sum(axis=1), 1.0)  # telescoping sum


def test_bucket_probs_argmax_matches_predicted_bucket_at_extremes():
    logits = np.array([[BIG, BIG, BIG], [-BIG, -BIG, -BIG], [BIG, -BIG, -BIG]])
    assert corn_bucket_probs(logits).argmax(axis=1).tolist() == [3, 0, 1]
    assert corn_predict_buckets(logits).tolist() == [3, 0, 1]


def test_scalar_score_monotonic_in_logits():
    lo = corn_scalar_score(np.array([[0.0, -1.0, -2.0]]))
    hi = corn_scalar_score(np.array([[2.0, 1.0, 0.0]]))
    assert hi[0] > lo[0]


def test_corn_loss_lower_when_logits_match_labels():
    torch = pytest.importorskip("torch")
    from greyscope.corn import corn_loss

    labels = torch.tensor([0, 3])
    good = torch.tensor([[-BIG, -BIG, -BIG], [BIG, BIG, BIG]])  # matches 0 and 3
    bad = torch.tensor([[BIG, BIG, BIG], [-BIG, -BIG, -BIG]])   # inverted
    assert corn_loss(good, labels).item() < corn_loss(bad, labels).item()
    assert corn_loss(good, labels).item() == pytest.approx(0.0, abs=1e-3)


def test_ranking_loss_zero_when_separated_positive_when_violated():
    torch = pytest.importorskip("torch")
    from greyscope.corn import corn_ranking_loss

    labels = torch.tensor([0, 3])  # human, ai_generated
    # head-0 logit drives P(y>0): human ≪ 0, AI ≫ 0 → gap ≈ 1 ≥ margin → no penalty
    separated = torch.tensor([[-BIG, 0.0, 0.0], [BIG, 0.0, 0.0]])
    assert corn_ranking_loss(separated, labels, margin=0.25).item() == pytest.approx(0.0, abs=1e-4)
    # inverted (human scores high, AI low) → large margin violation → positive loss
    violated = torch.tensor([[BIG, 0.0, 0.0], [-BIG, 0.0, 0.0]])
    assert corn_ranking_loss(violated, labels, margin=0.25).item() > 0.5


def test_ranking_loss_zero_when_batch_is_single_class():
    torch = pytest.importorskip("torch")
    from greyscope.corn import corn_ranking_loss

    logits = torch.tensor([[0.5, 0.0, 0.0], [0.3, 0.0, 0.0]])
    assert corn_ranking_loss(logits, torch.tensor([0, 0])).item() == 0.0  # all human
    assert corn_ranking_loss(logits, torch.tensor([1, 2])).item() == 0.0  # all non-human


def test_ranking_loss_uses_detection_boundary_not_magnitude():
    torch = pytest.importorskip("torch")
    from greyscope.corn import corn_ranking_loss

    # A lightly-edited bucket-1 row with high P(y>0) but low upper logits is a SATISFIED non-human:
    # only head-0 matters, so it needs no magnitude margin over human (the graded-design guardrail).
    labels = torch.tensor([0, 1])  # human, ai_edited (bucket 1)
    logits = torch.tensor([[-BIG, -BIG, -BIG], [BIG, -BIG, -BIG]])  # edited: P(y>0) high, P(y>1) low
    assert corn_ranking_loss(logits, labels, margin=0.25).item() == pytest.approx(0.0, abs=1e-4)


def test_ranking_loss_gradient_flows_to_head0():
    torch = pytest.importorskip("torch")
    from greyscope.corn import corn_ranking_loss

    logits = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]], requires_grad=True)
    corn_ranking_loss(logits, torch.tensor([0, 3]), margin=0.5).backward()  # human > AI → violation
    assert logits.grad is not None and logits.grad[:, 0].abs().sum().item() > 0
