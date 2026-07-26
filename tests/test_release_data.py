import pytest
import pandas as pd

from greyscope.release_data import (
    normalize_apt,
    normalize_beemo,
    normalize_cred,
    normalize_nlpcc,
    normalize_raid,
    snapshot_metadata,
)


def test_normalize_apt_maps_original_and_polished_rows():
    originals = [{"id": 1, "generation": "Human text", "domain": "blog"}]
    polished = [{
        "id": 2,
        "generation": "Polished text",
        "polish_type": "percentage-based",
        "polishing_degree": None,
        "polishing_percent": 25.0,
        "polisher": "GPT-4o",
        "domain": "blog",
    }]
    df = normalize_apt(polished, originals)
    assert set(df["label_ternary"]) == {0, 2}
    assert df["label_binary"].isna().all()
    edited = df[df["label_ternary"] == 2].iloc[0]
    assert edited["edit_target"] == 0.25
    assert edited["generator"] == "GPT-4o"


def test_normalize_beemo_expands_all_edit_variants():
    source = [{
        "id": 7,
        "category": "QA",
        "model": "model-a",
        "human_output": "Human",
        "model_output": "Generated",
        "human_edits": "Edited by person",
        "llama-3.1-70b_edits": "[{'P1': 'Llama edit'}]",
        "gpt-4o_edits": "[{'P1': 'GPT edit one', 'P2': 'GPT edit two'}]",
    }]
    df = normalize_beemo(source)
    assert len(df) == 6
    assert (df["label_ternary"] == 2).sum() == 4
    assert df["row_id"].is_unique


def test_normalize_raid_maps_binary_labels_and_groups_prompts():
    rows = [
        {
            "model": "human",
            "domain": "news",
            "prompt": "Write a story",
            "text": "Human text",
            "label": 0,
        },
        {
            "model": "gpt4",
            "domain": "news",
            "prompt": "Write a story",
            "text": "Generated text",
            "label": 1,
        },
    ]
    df = normalize_raid(rows)
    assert sorted(df["label_binary"]) == [0, 1]
    assert df["source_id"].nunique() == 1
    assert set(df["variant"]) == {"human", "generated:gpt4"}


def test_normalize_raid_preserves_identical_rows():
    row = {
        "model": "human",
        "domain": "news",
        "prompt": "Write a story",
        "text": "Duplicate text",
        "label": 0,
    }
    df = normalize_raid([row, row.copy()])
    assert len(df) == 2
    assert df["row_id"].is_unique


def test_snapshot_metadata_changes_when_text_changes():
    original = normalize_apt([], [{"id": 1, "generation": "Human", "domain": "blog"}])
    changed = normalize_apt([], [{"id": 1, "generation": "Changed", "domain": "blog"}])
    first = snapshot_metadata(original, source="source", revision="a" * 40)
    second = snapshot_metadata(changed, source="source", revision="a" * 40)
    assert first["rows_sha256"] != second["rows_sha256"]
    assert first["texts_sha256"] != second["texts_sha256"]


def test_snapshot_metadata_changes_when_label_changes():
    rows = [{"id": "x", "text": "Same text"}]
    original = normalize_nlpcc(rows, [{"id": "x", "label": 0}], phase="testp1")
    changed = normalize_nlpcc(rows, [{"id": "x", "label": 1}], phase="testp1")
    first = snapshot_metadata(original, source="source", revision="a" * 40)
    second = snapshot_metadata(changed, source="source", revision="a" * 40)
    assert first["texts_sha256"] == second["texts_sha256"]
    assert first["records_sha256"] != second["records_sha256"]


def test_normalize_nlpcc_maps_three_way_labels():
    rows = [
        {"id": "a", "text": "Human"},
        {"id": "b", "text": "Generated"},
        {"id": "c", "text": "Refined"},
    ]
    labels = [{"id": "a", "label": 0}, {"id": "b", "label": 1}, {"id": "c", "label": 2}]
    df = normalize_nlpcc(rows, labels, phase="testp1")
    assert set(df["label_ternary"]) == {0, 1, 2}
    assert sorted(df["label_binary"]) == [0, 1, 1]


def test_normalize_cred_groups_generator_with_original_document(tmp_path):
    domain = tmp_path / "benchmark data" / "news"
    domain.mkdir(parents=True)
    pd.DataFrame([{"id": 7, "text": "Human", "label": 1}]).to_csv(
        domain / "CReD_news_human.csv", index=False
    )
    pd.DataFrame([
        {"id": 99, "original_id": 7, "text": "Generated", "label": 0}
    ]).to_csv(domain / "CReD_news_model-a.csv", index=False)
    df = normalize_cred(tmp_path)
    assert df["source_id"].nunique() == 1
    assert set(df["variant"]) == {"human", "generated:model-a"}


def test_duplicate_rows_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        normalize_apt([], [
            {"id": 1, "generation": "Same", "domain": "blog"},
            {"id": 1, "generation": "Same", "domain": "blog"},
        ])
