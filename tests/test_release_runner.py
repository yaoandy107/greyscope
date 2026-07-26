import json

import numpy as np
import pandas as pd
import pytest

from greyscope.release_runner import evaluate_release_scores, score_snapshot


def _rows():
    return pd.DataFrame({
        "row_id": ["a", "b", "c", "d"],
        "text": ["a", "bb", "ccc", "dddd"],
        "label_binary": [0, 0, 1, 1],
        "edit_target": [None, None, 0.25, 0.75],
        "variant": ["human", "human", "minor", "major"],
        "domain": ["one", "two", "one", "two"],
    })


MODEL = {
    "id": "model",
    "source": "source",
    "revision": "a" * 40,
    "adapter": "transformers",
    "max_length": 512,
}
SNAPSHOT = {
    "benchmark": "test",
    "rows_sha256": "b" * 64,
    "texts_sha256": "c" * 64,
}


def test_score_snapshot_resumes_without_rescoring(tmp_path):
    output = tmp_path / "predictions.jsonl"
    calls = []

    def score(texts):
        calls.append(texts)
        return np.asarray([len(text) for text in texts], dtype=float)

    first = score_snapshot(_rows(), score, output, model=MODEL, snapshot=SNAPSHOT, chunk_size=2)
    second = score_snapshot(_rows(), score, output, model=MODEL, snapshot=SNAPSHOT, chunk_size=2)
    assert first.equals(second)
    assert calls == [["a", "bb"], ["ccc", "dddd"]]
    assert len(output.read_text().splitlines()) == 4


def test_score_snapshot_checkpoints_each_chunk(tmp_path):
    checkpoints = []
    score_snapshot(
        _rows(),
        lambda texts: np.zeros(len(texts)),
        tmp_path / "predictions.jsonl",
        model=MODEL,
        snapshot=SNAPSHOT,
        chunk_size=2,
        on_chunk=lambda: checkpoints.append(True),
    )
    assert checkpoints == [True, True]


def test_score_snapshot_rejects_metadata_drift(tmp_path):
    output = tmp_path / "predictions.jsonl"
    score_snapshot(_rows(), lambda texts: np.zeros(len(texts)), output, model=MODEL, snapshot=SNAPSHOT)
    changed = MODEL | {"revision": "d" * 40}
    with pytest.raises(ValueError, match="metadata differs"):
        score_snapshot(_rows(), lambda texts: np.zeros(len(texts)), output, model=changed, snapshot=SNAPSHOT)


def test_evaluate_release_scores_reports_binary_and_edit_metrics():
    predictions = pd.DataFrame({"row_id": ["a", "b", "c", "d"], "score": [0.1, 0.2, 0.7, 0.9]})
    metrics = evaluate_release_scores(_rows(), predictions)
    assert metrics["binary"]["auroc"] == 1.0
    assert metrics["edit_correlation"]["pearson"] == pytest.approx(1.0)
    assert metrics["by_variant"]["human"]["n"] == 2
    assert metrics["by_domain"]["one"]["auroc"] == 1.0


def test_evaluate_release_scores_uses_frozen_ternary_thresholds():
    rows = _rows().copy()
    rows["label_ternary"] = [0, 0, 2, 1]
    predictions = pd.DataFrame({"row_id": ["a", "b", "c", "d"], "score": [0.1, 0.2, 0.6, 0.9]})
    metrics = evaluate_release_scores(
        rows,
        predictions,
        ternary_thresholds=(0.3, 0.8),
        binary_threshold=0.75,
    )
    assert metrics["ternary"]["macro_f1"] == 1.0
    assert metrics["binary"]["at_shipped_threshold"]["fpr"] == 0.0


def test_prediction_file_is_plain_jsonl(tmp_path):
    output = tmp_path / "predictions.jsonl"
    score_snapshot(_rows(), lambda texts: np.zeros(len(texts)), output, model=MODEL, snapshot=SNAPSHOT)
    assert all(set(json.loads(line)) == {"row_id", "score"} for line in output.read_text().splitlines())
