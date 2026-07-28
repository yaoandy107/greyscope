import pandas as pd
import pytest

from scripts.check_mlx_release_quality import (
    DATA_DIR,
    _document_sample,
    _proportional_sample,
    select_subset,
)


RELEASE_SNAPSHOTS = [
    DATA_DIR / "apt-eval-sample.csv",
    DATA_DIR / "beemo-sample.csv",
    DATA_DIR / "raid-adversarial-4968.csv",
]


def test_proportional_sample_is_stable_and_exact():
    rows = pd.DataFrame({
        "row_id": [f"{index:02d}" for index in range(10)],
        "group": ["a"] * 7 + ["b"] * 3,
    })
    selected = _proportional_sample(rows, 5, ["group"])
    assert len(selected) == 5
    assert selected["group"].value_counts().to_dict() == {"a": 4, "b": 1}
    assert selected["row_id"].tolist() == ["00", "01", "02", "03", "07"]


def test_document_sample_keeps_complete_groups():
    rows = pd.DataFrame({
        "row_id": [f"{document}-{variant}" for document in range(4) for variant in range(3)],
        "source_id": [document for document in range(4) for _ in range(3)],
        "domain": ["a"] * 6 + ["b"] * 6,
    })
    selected = _document_sample(rows, n_documents=2, strata=["domain"])
    assert selected["source_id"].nunique() == 2
    assert selected.groupby("source_id").size().eq(3).all()


@pytest.mark.skipif(
    not all(path.exists() for path in RELEASE_SNAPSHOTS),
    reason="release benchmark snapshots are generated locally",
)
def test_release_subsets_preserve_benchmark_units():
    apt, _ = select_subset("apt-eval")
    assert len(apt) == 1000
    assert apt["variant"].eq("human").sum() == 100

    beemo, _ = select_subset("beemo")
    assert len(beemo) == 999
    assert beemo["source_id"].nunique() == 111
    assert beemo.groupby("source_id").size().eq(9).all()

    raid, _ = select_subset("raid-extra")
    assert len(raid) == 1008
    assert raid["source_id"].nunique() == 42
    assert raid.groupby("source_id").size().eq(24).all()
