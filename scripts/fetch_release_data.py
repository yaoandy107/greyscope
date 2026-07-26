#!/usr/bin/env python3
"""Fetch and freeze normalized public release-benchmark snapshots."""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from greyscope.release_data import (
    normalize_apt,
    normalize_beemo,
    normalize_cred,
    normalize_nlpcc,
    normalize_raid,
    validate_release_rows,
    write_snapshot,
)

SOURCES = {
    "apt-eval": ("smksaha/apt-eval", "1a183126ec24791c14c3a3254e8cdb5c58935d27"),
    "beemo": ("toloka/beemo", "9c014107fe9b85c4c784c1ce3a43b0b7b0a6d162"),
    "nlpcc-2026-task6": (
        "NLP2CT/NLPCC-2026-Task6-Detection",
        "297c0dd504be7fedfbaa297f1c5ec5fd1b837fdb",
    ),
    "c-red": (
        "HeraldofLight/C-ReD",
        "b90072cd218b6ebdbd1d1478ce6e439677f18192",
    ),
    "raid-10k": (
        "pangramlabs/EditLens/data/raid_10k.csv",
        "05a588f15d792330ccaf91be8ee4fdb54ce26835",
    ),
}


def fetch(name: str, *, repo_path: Path | None = None):
    source, revision = SOURCES[name]
    if repo_path is not None:
        actual_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual_revision != revision:
            raise ValueError(
                f"{name} checkout is {actual_revision}, expected pinned revision {revision}"
            )
    if name == "apt-eval":
        polished = load_dataset(source, revision=revision, split="test")
        originals = load_dataset(
            source,
            revision=revision,
            data_files={"original": "original.csv"},
            split="original",
        )
        rows = normalize_apt(polished, originals)
    elif name == "beemo":
        rows = normalize_beemo(load_dataset(source, revision=revision, split="train"))
    elif name == "nlpcc-2026-task6":
        if repo_path is None:
            raise ValueError("nlpcc-2026-task6 requires --repo-path at its pinned revision")
        data_dir = repo_path / "data"
        phases = []
        for phase in ("testp1", "testp2"):
            rows_path = data_dir / f"{phase}.json"
            labels_path = data_dir / f"{phase}_testing_label.json"
            phases.append(normalize_nlpcc(
                json.loads(rows_path.read_text()),
                json.loads(labels_path.read_text()),
                phase=phase,
            ))
        rows = validate_release_rows(pd.concat(phases, ignore_index=True))
    elif name == "c-red":
        if repo_path is None:
            raise ValueError("c-red requires --repo-path at its pinned revision")
        rows = normalize_cred(repo_path)
    elif name == "raid-10k":
        url = (
            "https://raw.githubusercontent.com/pangramlabs/EditLens/"
            f"{revision}/data/raid_10k.csv"
        )
        with urllib.request.urlopen(url) as response:
            rows = normalize_raid(pd.read_csv(response).to_dict("records"))
    else:
        raise KeyError(name)
    return rows, source, revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=sorted(SOURCES))
    parser.add_argument("--data-dir", type=Path, default=Path("data/release"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("benchmarks/manifests"))
    parser.add_argument(
        "--repo-path",
        type=Path,
        help="local checkout for GitHub-hosted datasets, at the pinned revision",
    )
    args = parser.parse_args()
    rows, source, revision = fetch(args.benchmark, repo_path=args.repo_path)
    write_snapshot(
        rows,
        args.data_dir / f"{args.benchmark}.csv",
        args.manifest_dir / f"{args.benchmark}.json",
        source=source,
        revision=revision,
        extra_metadata=(
            {"selection": "10,000-row non-adversarial RAID subset released with EditLens"}
            if args.benchmark == "raid-10k"
            else None
        ),
    )
    print(f"{args.benchmark}: froze {len(rows)} rows at revision {revision}")


if __name__ == "__main__":
    main()
