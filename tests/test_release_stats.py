import pandas as pd
import pytest

from greyscope.release_stats import bootstrap_intervals, paired_bootstrap_differences


def _rows() -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": [str(i) for i in range(8)],
        "source_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
        "variant": ["human", "generated"] * 4,
        "label_binary": [0, 1] * 4,
        "label_ternary": [0, 1] * 4,
        "edit_target": [float("nan")] * 8,
    })


def test_bootstrap_is_deterministic_and_clusters_sources():
    rows = _rows()
    predictions = pd.DataFrame({
        "row_id": rows["row_id"],
        "score": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
    })
    first = bootstrap_intervals(rows, predictions, unit="source", samples=20, seed=7)
    second = bootstrap_intervals(rows, predictions, unit="source", samples=20, seed=7)

    assert first == second
    assert first["unit"] == "source"
    assert first["metrics"]["binary.auroc"]["estimate"] == pytest.approx(1.0)
    assert first["metrics"]["binary.auroc"]["successful_samples"] == 20


def test_bootstrap_rejects_unknown_unit():
    rows = _rows()
    predictions = pd.DataFrame({"row_id": rows["row_id"], "score": [0.5] * len(rows)})

    with pytest.raises(ValueError, match="bootstrap unit"):
        bootstrap_intervals(rows, predictions, unit="document")


def test_paired_bootstrap_reports_a_minus_b():
    rows = _rows()
    better = pd.DataFrame({
        "row_id": rows["row_id"],
        "score": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
    })
    worse = pd.DataFrame({
        "row_id": rows["row_id"],
        "score": [0.45, 0.55, 0.4, 0.6, 0.35, 0.65, 0.3, 0.7],
    })

    result = paired_bootstrap_differences(
        rows, better, worse, unit="source", samples=20, seed=7
    )

    assert result["direction"] == "model_a minus model_b"
    assert result["metrics"]["binary.auroc"]["estimate"] >= 0
