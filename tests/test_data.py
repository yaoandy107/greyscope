"""Tests for the data prep module; only the small deterministic pieces.

The full `prepare_data` is integration-tested via the smoke run; we don't
hit the HF Hub from unit tests.
"""

from greyscope.data import PROMPT_TEMPLATE, compute_class_weights, compute_sample_weights


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


def test_compute_sample_weights_equalizes_language_groups():
    # 4 en vs 2 ja, same bucket: each language's total mass is equal after balancing.
    w = compute_sample_weights(["en"] * 4 + ["ja"] * 2, [0] * 6)
    assert abs(sum(w[:4]) - sum(w[4:])) < 1e-9
    assert abs(sum(w) / len(w) - 1.0) < 1e-9  # mean 1


def test_compute_sample_weights_upweights_rarest_language_bucket_cell():
    # (ja, 3) is the single rarest (language, bucket) cell -> largest weight.
    w = compute_sample_weights(["en", "en", "en", "ja"], [0, 0, 0, 3])
    assert w[3] == max(w)


def test_compute_sample_weights_uniform_cells_are_all_ones():
    w = compute_sample_weights(["en", "ja", "zh-tw", "en", "ja", "zh-tw"], [0, 1, 2, 0, 1, 2])
    assert all(abs(x - 1.0) < 1e-9 for x in w)


def test_compute_sample_weights_empty():
    assert compute_sample_weights([], []) == []


def test_compute_sample_weights_default_is_full_inverse():
    # default τ=1.0 preserves the original inverse-frequency behavior (backward compatible).
    langs, labels = ["en"] * 4 + ["ja"] * 2, [0] * 6
    assert compute_sample_weights(langs, labels) == compute_sample_weights(langs, labels, temperature=1.0)


def test_compute_sample_weights_temperature_softens_balancing():
    # 8 en vs 2 ja (same bucket): the rare ja cell is up-weighted less as τ shrinks, and
    # τ=0 is natural frequency (all weights equal). Every setting stays mean-1 normalized.
    langs, labels = ["en"] * 8 + ["ja"] * 2, [0] * 10
    full = compute_sample_weights(langs, labels, temperature=1.0)
    soft = compute_sample_weights(langs, labels, temperature=0.5)
    natural = compute_sample_weights(langs, labels, temperature=0.0)
    assert full[-1] > soft[-1] > natural[-1]  # smoothing lifts the thin cell less aggressively
    assert all(abs(x - 1.0) < 1e-9 for x in natural)  # τ=0 → no balancing
    for w in (full, soft, natural):
        assert abs(sum(w) / len(w) - 1.0) < 1e-9


def test_prepare_v2_data_uses_buckets_and_keeps_cjk(tmp_path):
    import pandas as pd

    from greyscope.config import DataConfig
    from greyscope.data import prepare_v2_data

    # `model` is empty on the human row but a string on AI rows — the mixed null/string
    # column that broke CSV type-inference on Modal; the loader must not choke on it.
    rows = [
        {"text_id": "en/1", "text": "This is an English sample.", "language": "en",
         "text_type": "human_written", "bucket": 0, "model": ""},
        {"text_id": "ja/1", "text": "これは日本語のテキストです。", "language": "ja",
         "text_type": "ai_generated", "bucket": 3, "model": "openai/gpt-5.5"},
        {"text_id": "zh/1", "text": "這是繁體中文文字。", "language": "zh-tw",
         "text_type": "ai_edited", "bucket": 2, "model": "anthropic/claude-sonnet-4.6"},
    ]
    for split in ("train", "val", "test"):
        pd.DataFrame(rows).to_csv(tmp_path / f"{split}.csv", index=False)

    data = prepare_v2_data(DataConfig(n_buckets=4), splits_dir=str(tmp_path))

    assert data.train["label"] == [0, 3, 2]  # precomputed per-language bucket used directly
    assert len(data.train) == 3  # CJK rows survive (no English word-count filter)
    assert all("Answer:" in p for p in data.train["prompt"])
    assert len(data.sample_weights) == 3
    assert abs(sum(data.sample_weights) / 3 - 1.0) < 1e-9  # mean-1 balancing weights
    assert len(data.class_weights) == 4


def test_prepare_v2_data_raw_text_when_prompt_template_disabled(tmp_path):
    import pandas as pd

    from greyscope.config import DataConfig
    from greyscope.data import prepare_v2_data
    from greyscope.preprocess import clean_text

    rows = [{"text_id": "en/1", "text": "An English sample.", "language": "en",
             "text_type": "human_written", "bucket": 0, "model": ""}]
    for split in ("train", "val", "test"):
        pd.DataFrame(rows).to_csv(tmp_path / f"{split}.csv", index=False)

    # encoder arm: prompt = raw (cleaned) text, no instruction wrapper eating context budget
    data = prepare_v2_data(DataConfig(n_buckets=4, use_prompt_template=False), splits_dir=str(tmp_path))
    assert data.train["prompt"] == [clean_text("An English sample.")]
    assert "Answer:" not in data.train["prompt"][0]
