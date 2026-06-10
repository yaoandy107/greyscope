"""Tests for the ternary evaluation port."""

import numpy as np

from greyscope.eval import (
    LABEL_TO_ID,
    calibrate_thresholds,
    compute_scalar_score,
    evaluate,
    find_optimal_threshold,
    minmax_scale,
    orient_scores,
    predict_ternary,
    roc_auc,
    threshold_for_fpr,
    tpr_at_fpr,
)


def test_label_id_mapping_matches_openpangram():
    assert LABEL_TO_ID == {"human_written": 0, "ai_generated": 1, "ai_edited": 2}


def test_find_optimal_threshold_perfectly_separable():
    preds = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    thresh, f1 = find_optimal_threshold(preds, labels)
    assert f1 == 1.0
    assert 0.2 < thresh <= 0.8


def test_find_optimal_threshold_all_negative():
    preds = np.array([0.1, 0.2, 0.3])
    labels = np.array([0, 0, 0])
    thresh, f1 = find_optimal_threshold(preds, labels)
    assert f1 == 0.0


def test_minmax_scale_basic():
    out = minmax_scale(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_array_almost_equal(out, [0.0, 0.5, 1.0])


def test_minmax_scale_constant_input():
    out = minmax_scale(np.array([0.5, 0.5, 0.5]))
    np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])


def test_orient_scores_flips_when_humans_score_higher():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([0, 0, 1, 1])
    oriented, flipped = orient_scores(scores, labels)
    assert flipped is True
    np.testing.assert_array_almost_equal(oriented, -scores)


def test_orient_scores_keeps_when_ai_scores_higher():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    oriented, flipped = orient_scores(scores, labels)
    assert flipped is False
    np.testing.assert_array_equal(oriented, scores)


def test_calibrate_thresholds_well_separated():
    labels = np.array([0, 0, 1, 1, 2, 2])
    scaled = np.array([0.05, 0.10, 0.95, 0.90, 0.50, 0.55])
    h_thresh, ai_thresh, h_f1, ai_f1 = calibrate_thresholds(labels, scaled)
    assert h_thresh < ai_thresh
    assert h_f1 > 0.5
    assert ai_f1 > 0.5


def test_predict_ternary_assigns_correctly():
    scaled = np.array([0.05, 0.50, 0.95])
    preds = predict_ternary(scaled, h_thresh=0.20, ai_thresh=0.80)
    np.testing.assert_array_equal(preds, [0, 2, 1])


def test_evaluate_returns_expected_keys():
    true = np.array([0, 1, 2, 0, 1, 2])
    pred = np.array([0, 1, 2, 0, 1, 2])
    m = evaluate(true, pred)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["f1_human"] == 1.0
    assert m["f1_ai_generated"] == 1.0
    assert m["f1_ai_edited"] == 1.0
    assert m["confusion_matrix"].shape == (3, 3)


def test_evaluate_imperfect_predictions():
    true = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    m = evaluate(true, pred)
    assert 0.0 < m["accuracy"] < 1.0
    assert 0.0 < m["macro_f1"] < 1.0


def test_compute_scalar_score_pure_human_logit():
    # Bucket 0 logit dominant → score ≈ 0
    logits = np.array([[100.0, 0.0, 0.0, 0.0]])
    score = compute_scalar_score(logits, n_buckets=4)
    np.testing.assert_array_almost_equal(score, [0.0])


def test_compute_scalar_score_pure_ai_logit():
    # Last-bucket logit dominant → score ≈ 1
    logits = np.array([[0.0, 0.0, 0.0, 100.0]])
    score = compute_scalar_score(logits, n_buckets=4)
    np.testing.assert_array_almost_equal(score, [1.0])


def test_compute_scalar_score_uniform_is_midpoint():
    # All equal logits → softmax uniform → expected bucket = (0+1+2+3)/4 = 1.5
    # Normalized by (n-1)=3 → 0.5
    logits = np.zeros((1, 4))
    score = compute_scalar_score(logits, n_buckets=4)
    np.testing.assert_array_almost_equal(score, [0.5])


def test_compute_scalar_score_batch_and_monotonic():
    # Increasing tilt toward bucket 3 should produce monotonically increasing scores.
    logits = np.array([
        [10.0, 0.0, 0.0, 0.0],   # → ~0
        [0.0, 10.0, 0.0, 0.0],   # → ~1/3
        [0.0, 0.0, 10.0, 0.0],   # → ~2/3
        [0.0, 0.0, 0.0, 10.0],   # → ~1
    ])
    scores = compute_scalar_score(logits, n_buckets=4)
    assert scores.shape == (4,)
    assert np.all(np.diff(scores) > 0)
    np.testing.assert_array_almost_equal(scores, [0.0, 1/3, 2/3, 1.0], decimal=4)


def test_roc_auc_perfect_and_single_class():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    # higher human scores → AUROC 0 (caller is responsible for orientation)
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    # one class only → undefined
    assert roc_auc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) is None


def test_threshold_for_fpr_quantile():
    human = np.arange(0.0, 1.0001, 0.01)  # 101 evenly-spaced human scores
    # threshold at the 99th percentile → ~1% of humans exceed it
    thr = threshold_for_fpr(human, 0.01)
    assert (human > thr).mean() <= 0.02
    # a stricter target sets a higher threshold
    assert threshold_for_fpr(human, 0.01) >= threshold_for_fpr(human, 0.05)


def test_tpr_at_fpr_separable_and_single_class():
    # humans in [0,0.5), AI in (0.5,1] → at any low FPR, every AI is detected
    scores = np.concatenate([np.linspace(0, 0.49, 50), np.linspace(0.51, 1.0, 50)])
    y = np.array([0] * 50 + [1] * 50)
    assert tpr_at_fpr(y, scores, 0.05) == 1.0
    # human-only split → TPR undefined
    assert tpr_at_fpr(np.zeros(10, dtype=int), np.linspace(0, 1, 10), 0.05) is None
