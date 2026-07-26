"""Deterministic cross-runtime probe selection for release equivalence gates."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from greyscope.data import PROMPT_TEMPLATE
from greyscope.preprocess import clean_text


def select_equivalence_probe(csv_path: str | Path, *, random_per_group: int = 4) -> pd.DataFrame:
    """Select random rows plus the longest row from every language x bucket group."""
    df = pd.read_csv(csv_path)
    required = {"text_id", "text", "language", "bucket"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"equivalence source is missing columns: {sorted(missing)}")

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["_text_length"] = df["text"].str.len()
    selected = []
    for (language, bucket), group in df.groupby(["language", "bucket"], sort=True):
        group = group.sort_values("text_id")
        longest_idx = group.sort_values(
            ["_text_length", "text_id"], ascending=[False, True]
        ).index[0]
        longest = group.loc[[longest_idx]]
        remaining = group.drop(index=longest_idx)
        seed_text = f"{language}:{bucket}:greyscope-mlx-v1"
        seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:4], "big")
        random_rows = remaining.sample(n=min(random_per_group, len(remaining)), random_state=seed)
        selected.extend([longest, random_rows])

    probe = pd.concat(selected, ignore_index=True)
    probe = probe.sort_values(["language", "bucket", "text_id"]).reset_index(drop=True)
    probe["prompt"] = probe["text"].map(lambda text: PROMPT_TEMPLATE.format(text=clean_text(text)))
    probe["prompt_sha256"] = probe["prompt"].map(
        lambda prompt: hashlib.sha256(prompt.encode()).hexdigest()
    )
    return probe.drop(columns=["_text_length"])


def select_task_probe(csv_path: str | Path, *, per_group: int = 20) -> pd.DataFrame:
    """Select a deterministic balanced language x ternary-label task probe."""
    df = pd.read_csv(csv_path)
    required = {"text_id", "text", "text_type", "language"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"task source is missing columns: {sorted(missing)}")
    selected = []
    for (language, text_type), group in df.groupby(["language", "text_type"], sort=True):
        seed = int.from_bytes(
            hashlib.sha256(f"{language}:{text_type}:mlx-task-v1".encode()).digest()[:4],
            "big",
        )
        selected.append(group.sample(n=min(per_group, len(group)), random_state=seed))
    return (
        pd.concat(selected, ignore_index=True)
        .sort_values(["language", "text_type", "text_id"])
        .reset_index(drop=True)
    )
