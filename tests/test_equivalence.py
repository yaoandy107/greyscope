import pandas as pd

from greyscope.equivalence import select_equivalence_probe, select_task_probe


def test_equivalence_probe_covers_groups_and_is_deterministic(tmp_path):
    rows = []
    for language in ("en", "ja", "zh-tw"):
        for bucket in range(4):
            for i in range(7):
                rows.append({
                    "text_id": f"{language}-{bucket}-{i}",
                    "text": "text " * (i + 1),
                    "language": language,
                    "bucket": bucket,
                })
    path = tmp_path / "val.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    first = select_equivalence_probe(path, random_per_group=2)
    second = select_equivalence_probe(path, random_per_group=2)

    assert first["text_id"].tolist() == second["text_id"].tolist()
    assert len(first) == 3 * 4 * 3
    assert first.groupby(["language", "bucket"]).size().eq(3).all()
    assert first.groupby(["language", "bucket"])["text_id"].apply(
        lambda ids: any(text_id.endswith("-6") for text_id in ids)
    ).all()
    assert first["prompt_sha256"].str.len().eq(64).all()


def test_task_probe_is_deterministic_and_balanced(tmp_path):
    rows = []
    for language in ("en", "ja"):
        for text_type in ("human_written", "ai_generated", "ai_edited"):
            for index in range(5):
                rows.append({
                    "text_id": f"{language}/{text_type}/{index}",
                    "text": f"text {index}",
                    "language": language,
                    "text_type": text_type,
                })
    source = tmp_path / "test.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    first = select_task_probe(source, per_group=2)
    second = select_task_probe(source, per_group=2)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 12
    assert first.groupby(["language", "text_type"]).size().eq(2).all()
