"""Tests for the data prep module; only the small deterministic pieces.

The full `prepare_data` is integration-tested via the smoke run; we don't
hit the HF Hub from unit tests.
"""

from greyscope.data import PROMPT_TEMPLATE, compute_class_weights


def test_prompt_template_is_4_bucket():
    assert "0, 1, 2, or 3" in PROMPT_TEMPLATE
    assert PROMPT_TEMPLATE.format(text="passage body").rstrip().endswith("Answer:")


def test_compute_class_weights_balanced_distribution():
    weights = compute_class_weights([0, 0, 1, 1, 2, 2, 3, 3], 4)
    assert all(abs(w - 1.0) < 1e-9 for w in weights)


def test_compute_class_weights_imbalanced_distribution():
    weights = compute_class_weights([0, 0, 0, 0, 1, 1, 2, 2, 3, 3], 4)
    assert weights[0] < weights[1]
    assert weights[1] == weights[2] == weights[3]


def test_compute_class_weights_handles_missing_class():
    weights = compute_class_weights([0, 1, 2, 0, 1, 2], 4)
    assert len(weights) == 4
    assert all(w > 0 for w in weights)
