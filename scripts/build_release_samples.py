#!/usr/bin/env python3
"""Freeze the small, deterministic release samples used for paid comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from greyscope.release_data import validate_release_rows, write_snapshot

DATA_DIR = Path("data/release")
MANIFEST_DIR = Path("benchmarks/manifests")


def _proportional_sample(frame: pd.DataFrame, n: int, strata: list[str]) -> pd.DataFrame:
    """Deterministic proportional allocation, using hash-like row IDs as ordering."""
    if n >= len(frame):
        return frame.copy()
    keyed = frame.copy()
    keyed["_stratum"] = keyed[strata].fillna("<null>").astype(str).agg("\0".join, axis=1)
    sizes = keyed.groupby("_stratum").size().sort_index()
    exact = sizes * (n / len(keyed))
    quotas = exact.astype(int)
    remaining = n - int(quotas.sum())
    order = (exact - quotas).sort_values(ascending=False, kind="stable").index
    for name in order[:remaining]:
        quotas[name] += 1
    pieces = [
        group.sort_values("row_id").head(int(quotas[name]))
        for name, group in keyed.groupby("_stratum", sort=True)
        if quotas[name]
    ]
    return pd.concat(pieces, ignore_index=True).drop(columns="_stratum")


def _document_sample(
    frame: pd.DataFrame,
    *,
    strata: list[str],
    n_documents: int | None = None,
    documents_per_stratum: int | None = None,
) -> pd.DataFrame:
    document_rows = frame.sort_values("row_id").drop_duplicates("source_id")
    if (n_documents is None) == (documents_per_stratum is None):
        raise ValueError("set exactly one document sampling size")
    if n_documents is not None:
        chosen = _proportional_sample(document_rows, n_documents, strata)["source_id"]
    else:
        chosen = []
        group_key = strata[0] if len(strata) == 1 else strata
        for _, group in document_rows.groupby(group_key, dropna=False, sort=True):
            chosen.extend(
                group.sort_values("source_id").head(documents_per_stratum)["source_id"]
            )
    return frame[frame["source_id"].isin(chosen)].copy()


def _write(name: str, sample: pd.DataFrame, *, method: str) -> None:
    parent_manifest = json.loads((MANIFEST_DIR / f"{name}.json").read_text())
    write_snapshot(
        validate_release_rows(sample),
        DATA_DIR / f"{name}-sample.csv",
        MANIFEST_DIR / f"{name}-sample.json",
        source=parent_manifest["source"],
        revision=parent_manifest["revision"],
        extra_metadata={
            "sampling": {
                "method": method,
                "parent_n": parent_manifest["n"],
                "parent_rows_sha256": parent_manifest["rows_sha256"],
                "parent_records_sha256": parent_manifest["records_sha256"],
            }
        },
    )
    print(f"{name}: {len(sample)} rows ({method})")


def main() -> None:
    apt = pd.read_csv(DATA_DIR / "apt-eval.csv")
    human = apt[apt["variant"] == "human"]
    polished = _proportional_sample(
        apt[apt["variant"] != "human"], 2700, ["domain", "generator", "variant"]
    )
    _write(
        "apt-eval",
        pd.concat([human, polished], ignore_index=True),
        method="all 300 human rows plus 2700 proportionally stratified polished rows",
    )

    beemo = pd.read_csv(DATA_DIR / "beemo.csv")
    _write(
        "beemo",
        _document_sample(beemo, n_documents=333, strata=["domain"]),
        method="333 proportionally stratified source documents with all nine variants retained",
    )

    cred = pd.read_csv(DATA_DIR / "c-red.csv")
    _write(
        "c-red",
        _document_sample(cred, documents_per_stratum=80, strata=["domain"]),
        method="80 source documents per domain with available human and generator variants",
    )

if __name__ == "__main__":
    main()
