"""greyscope.raid_eval: the official RAID harness wiring, exercised fully offline.

raid-bench is not a test dependency, so a fake `raid` package is injected into
sys.modules and the model/score_fn is a deterministic stub. The real run (a full
RAID split through a loaded model on a GPU) is integration-only — covered by the
Modal run, not here.
"""
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

import greyscope.raid_eval as raid_eval


def _fake_score(texts):
    """higher = more human, so a flip is needed to read higher = more AI."""
    return np.asarray([0.1 if "ai" in t else 0.9 for t in texts])


def test_oriented_detector_no_flip():
    det = raid_eval.oriented_detector(lambda t: np.array([0.2, 0.8]), flip=False)
    assert det(["a", "b"]) == [0.2, 0.8]


def test_oriented_detector_flips_and_returns_plain_floats():
    det = raid_eval.oriented_detector(lambda t: np.array([0.2, 0.8]), flip=True)
    out = det(["a", "b"])
    assert out == [-0.2, -0.8]
    assert isinstance(out, list) and isinstance(out[0], float)  # RAID's list[float] contract


def test_leaderboard_metadata_defaults_and_override():
    meta = raid_eval.leaderboard_metadata("greyscope-9b", organization="acme", url="http://x")
    assert meta["name"] == "greyscope-9b"
    assert meta["open_source"] is True
    assert meta["detector_type"] == "metric-based"
    assert meta["organization"] == "acme" and meta["url"] == "http://x"


@pytest.fixture
def fake_raid(monkeypatch):
    """Inject a minimal fake `raid` + `raid.utils` so run_raid runs without raid-bench."""
    calls = {}

    def load_data(split, include_adversarial=True):
        calls["load_data"] = {"split": split, "include_adversarial": include_adversarial}
        n = 6
        return pd.DataFrame({
            "id": list(range(n)),
            "text": [f"ai sample {i}" if i % 2 else f"human sample {i}" for i in range(n)],
        })

    def run_detection(detector, df):
        scores = detector(df["text"].tolist())
        calls["run_detection_n"] = len(scores)
        return {str(i): s for i, s in zip(df["id"], scores)}

    def run_evaluation(predictions, df, target_fpr=0.05):
        calls["run_evaluation"] = {"target_fpr": target_fpr, "n": len(predictions)}
        return {"scores": {"tpr_at_fpr": 0.87}, "target_fpr": target_fpr}

    raid = types.ModuleType("raid")
    raid.run_detection = run_detection
    raid.run_evaluation = run_evaluation
    utils = types.ModuleType("raid.utils")
    utils.load_data = load_data
    raid.utils = utils
    monkeypatch.setitem(sys.modules, "raid", raid)
    monkeypatch.setitem(sys.modules, "raid.utils", utils)
    return calls


def test_run_raid_labeled_split_writes_and_evaluates(tmp_path, fake_raid):
    results = raid_eval.run_raid(_fake_score, flip=True, out_dir=str(tmp_path), split="extra",
                                 detector_name="greyscope-9b", target_fpr=0.05)

    assert results == {"scores": {"tpr_at_fpr": 0.87}, "target_fpr": 0.05}
    assert fake_raid["load_data"] == {"split": "extra", "include_adversarial": True}
    assert fake_raid["run_detection_n"] == 6
    assert fake_raid["run_evaluation"]["target_fpr"] == 0.05

    assert len(json.loads((tmp_path / "predictions.json").read_text())) == 6
    assert json.loads((tmp_path / "metadata.json").read_text())["name"] == "greyscope-9b"
    assert json.loads((tmp_path / "results.json").read_text())["scores"]["tpr_at_fpr"] == 0.87


def test_run_raid_test_split_holds_out_eval(tmp_path, fake_raid):
    results = raid_eval.run_raid(_fake_score, flip=False, out_dir=str(tmp_path), split="test")

    assert results is None  # test labels are held out -> no local eval
    assert "run_evaluation" not in fake_raid
    assert (tmp_path / "predictions.json").exists()
    assert (tmp_path / "metadata.json").exists()
    assert not (tmp_path / "results.json").exists()


def test_run_raid_limit_subsamples(tmp_path, fake_raid):
    raid_eval.run_raid(_fake_score, flip=False, out_dir=str(tmp_path), split="extra", limit=2)
    assert len(json.loads((tmp_path / "predictions.json").read_text())) == 2
