"""greyscope.benchmark: the CSV harness and score_fn plumbing, exercised offline.

All EditLens CSVs are pre-seeded into tmp_path so fetch_editlens_csv's cache hits and
nothing touches the network; the model is replaced by a deterministic score_fn.
run_ood_eval is integration-only (needs the HF dataset + a GPU) and is covered by the
Modal run, not here.
"""
import numpy as np
import pandas as pd
import pytest

import greyscope.benchmark as bench
import greyscope.scoring
from greyscope.preprocess import clean_text


def _fake_score(texts: list[str]) -> np.ndarray:
    """Deterministic, perfectly separating: human < edited < generated."""

    def score(t: str) -> float:
        if "edited" in t:
            return 0.5
        return 0.9 if "ai" in t else 0.1

    return np.asarray([score(t) for t in texts])


def _ternary_df(n_per_class: int = 8) -> pd.DataFrame:
    rows = []
    for i in range(n_per_class):
        rows.append({"text": f"human sample {i}", "text_type": "human_written",
                     "cosine_score": 0.0, "foo_score": 0.1})
        rows.append({"text": f"ai sample {i}", "text_type": "ai_generated",
                     "cosine_score": 0.9, "foo_score": 0.9})
        rows.append({"text": f"ai edited sample {i}", "text_type": "ai_edited",
                     "cosine_score": 0.4, "foo_score": 0.5})
    return pd.DataFrame(rows)


def _label_only_df(n_per_class: int = 8) -> pd.DataFrame:
    rows = []
    for i in range(n_per_class):
        rows.append({"text": f"human sample {i}", "label": 0, "foo_score": 0.1})
        rows.append({"text": f"ai sample {i}", "label": 1, "foo_score": 0.9})
    return pd.DataFrame(rows)


@pytest.fixture
def seeded_dirs(tmp_path):
    """(data_dir, scores_dir) with every benchmark CSV pre-seeded (no network)."""
    data_dir, scores_dir = tmp_path / "data", tmp_path / "scores"
    data_dir.mkdir()
    scores_dir.mkdir()
    _ternary_df().to_csv(data_dir / "val.csv", index=False)
    for name in bench.BENCHMARK_SPLITS:
        df = _label_only_df() if name == "raid_10k" else _ternary_df()
        df.to_csv(data_dir / f"{name}.csv", index=False)
    return str(data_dir), str(scores_dir)


def test_fetch_is_cached(tmp_path):
    (tmp_path / "val.csv").write_text("text\nhello\n")
    path = bench.fetch_editlens_csv("val", str(tmp_path))  # must not hit the network
    assert pd.read_csv(path)["text"].tolist() == ["hello"]


def test_greyscope_score_fn_applies_clean_and_template(monkeypatch):
    captured = {}

    def fake_score_prompts(model, tok, prompts, n_buckets, head="seqcls", max_length=2048):
        captured.update(prompts=prompts, n_buckets=n_buckets, head=head, max_length=max_length)
        return np.zeros(len(prompts))

    monkeypatch.setattr(greyscope.scoring, "score_prompts", fake_score_prompts)
    fn = bench.greyscope_score_fn(model=None, tok=None)
    out = fn(["  Some RAW   Text  "])

    assert out.shape == (1,)
    assert captured["n_buckets"] == 4 and captured["max_length"] == 2048
    assert captured["head"] == "seqcls"  # no model.config → seq-cls default; a CORN merged model sets head_type
    prompt = captured["prompts"][0]
    assert clean_text("  Some RAW   Text  ") in prompt  # lowercased + whitespace-normalized
    assert prompt.endswith("Answer: ")  # full template applied


def test_run_benchmark_suite(seeded_dirs):
    data_dir, scores_dir = seeded_dirs
    seen = []
    results = bench.run_benchmark_suite(
        _fake_score, data_dir=data_dir, scores_dir=scores_dir,
        on_split=lambda r: seen.append(len(r["splits"])))

    # cosine_score is a label source, not a detector; greyscope_score is appended.
    assert results["score_cols"] == ["foo_score", "greyscope_score"]
    assert set(results["splits"]) == set(bench.BENCHMARK_SPLITS)
    assert seen == list(range(1, len(bench.BENCHMARK_SPLITS) + 1))  # fired per split

    # Perfect separation (binary drops ai_edited rows) → perfect binary metrics.
    g = results["splits"]["test"]["detectors"]["greyscope_score"]
    assert g["binary"]["macro_f1"] == 1.0
    assert g["binary"]["fpr"] == 0.0
    assert g["binary"]["auroc"] == 1.0
    assert "ternary" in g  # text_type present on this split

    # raid_10k has only an integer `label` column → binary-only path.
    raid = results["splits"]["raid_10k"]
    assert raid["ternary"] is False
    assert "ternary" not in raid["detectors"]["greyscope_score"]
    assert raid["detectors"]["greyscope_score"]["binary"]["macro_f1"] == 1.0

    # Per-split score CSVs are dumped for every split plus val.
    for name in ["val", *bench.BENCHMARK_SPLITS]:
        dumped = pd.read_csv(f"{scores_dir}/scores_{name}.csv")
        assert "greyscope_score" in dumped.columns
        assert "text" not in dumped.columns  # raw text is not re-exported
