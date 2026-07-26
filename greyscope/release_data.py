"""Normalize public release benchmarks into stable, hash-addressed rows."""
from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


RELEASE_COLUMNS = [
    "row_id",
    "text_sha256",
    "benchmark",
    "source_id",
    "variant",
    "text",
    "language",
    "label_binary",
    "label_ternary",
    "edit_target",
    "domain",
    "generator",
]


def _normalized_row(*, benchmark: str, source_id: str, variant: str, text: str, **fields) -> dict:
    text = str(text).strip()
    if not text:
        raise ValueError(f"empty text for {benchmark}:{source_id}:{variant}")
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    identity = (
        f"{benchmark}\0{source_id}\0{variant}\0{text_hash}\0{fields.get('identity_salt', '')}"
    )
    return {
        "row_id": hashlib.sha256(identity.encode()).hexdigest(),
        "text_sha256": text_hash,
        "benchmark": benchmark,
        "source_id": str(source_id),
        "variant": variant,
        "text": text,
        "language": fields.get("language", "en"),
        "label_binary": fields.get("label_binary"),
        "label_ternary": fields.get("label_ternary"),
        "edit_target": fields.get("edit_target"),
        "domain": fields.get("domain"),
        "generator": fields.get("generator"),
    }


def normalize_apt(polished: Iterable[dict], originals: Iterable[dict]) -> pd.DataFrame:
    """APT-Eval originals + polished texts with declared continuous edit targets."""
    rows = []
    for row in originals:
        rows.append(_normalized_row(
            benchmark="apt-eval",
            source_id=row["id"],
            variant="human",
            text=row["generation"],
            label_binary=None,
            label_ternary=0,
            edit_target=0.0,
            domain=row.get("domain"),
        ))
    for row in polished:
        target = row.get("polishing_percent")
        rows.append(_normalized_row(
            benchmark="apt-eval",
            source_id=row["id"],
            variant=f"polished:{row['polish_type']}:{row.get('polishing_degree') or target}",
            text=row["generation"],
            label_binary=None,
            label_ternary=2,
            edit_target=(float(target) / 100.0 if target is not None else None),
            domain=row.get("domain"),
            generator=row.get("polisher"),
        ))
    return validate_release_rows(pd.DataFrame(rows, columns=RELEASE_COLUMNS))


def _edit_values(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value
    else:
        parsed = value
    if isinstance(parsed, str):
        return [parsed] if parsed.strip() else []
    if isinstance(parsed, dict):
        parsed = [parsed]
    values = []
    for item in parsed:
        if isinstance(item, dict):
            values.extend(str(text) for text in item.values() if str(text).strip())
        elif str(item).strip():
            values.append(str(item))
    return values


def normalize_beemo(source_rows: Iterable[dict]) -> pd.DataFrame:
    """Expand Beemo's human, generated, and edited variants into document rows."""
    rows = []
    for row in source_rows:
        common = {"benchmark": "beemo", "source_id": row["id"], "domain": row.get("category")}
        rows.append(_normalized_row(
            **common,
            variant="human",
            text=row["human_output"],
            label_binary=0,
            label_ternary=0,
        ))
        rows.append(_normalized_row(
            **common,
            variant="generated",
            text=row["model_output"],
            label_binary=1,
            label_ternary=1,
            generator=row.get("model"),
        ))
        variants = [
            ("human_edit", row.get("human_edits"), "human"),
            ("llama_edit", row.get("llama-3.1-70b_edits"), "meta-llama/Llama-3.1-70B"),
            ("gpt4o_edit", row.get("gpt-4o_edits"), "openai/gpt-4o"),
        ]
        for variant, value, generator in variants:
            for index, text in enumerate(_edit_values(value)):
                rows.append(_normalized_row(
                    **common,
                    variant=f"{variant}:{index}",
                    text=text,
                    label_binary=1,
                    label_ternary=2,
                    generator=generator,
                ))
    return validate_release_rows(pd.DataFrame(rows, columns=RELEASE_COLUMNS))


def normalize_raid(source_rows: Iterable[dict]) -> pd.DataFrame:
    """Normalize the pinned non-adversarial RAID subset released with EditLens."""
    rows = []
    for index, row in enumerate(source_rows):
        label = int(row["label"])
        if label not in {0, 1}:
            raise ValueError(f"invalid RAID label: {label}")
        prompt = row.get("prompt")
        source_text = row["text"] if pd.isna(prompt) else prompt
        source_hash = hashlib.sha256(str(source_text).encode()).hexdigest()
        model = str(row["model"])
        rows.append(_normalized_row(
            benchmark="raid",
            source_id=f"{row['domain']}:{source_hash}",
            variant="human" if label == 0 else f"generated:{model}",
            text=row["text"],
            label_binary=label,
            label_ternary=None,
            domain=row["domain"],
            generator=None if label == 0 else model,
            identity_salt=index,
        ))
    return validate_release_rows(pd.DataFrame(rows, columns=RELEASE_COLUMNS))


def normalize_nlpcc(rows: Iterable[dict], labels: Iterable[dict], *, phase: str) -> pd.DataFrame:
    """Join an NLPCC 2026 test phase with its released three-way labels."""
    label_by_id = {str(row["id"]): int(row["label"]) for row in labels}
    normalized = []
    for row in rows:
        source_id = str(row["id"])
        if source_id not in label_by_id:
            raise ValueError(f"missing NLPCC label for {source_id}")
        label = label_by_id[source_id]
        if label not in {0, 1, 2}:
            raise ValueError(f"invalid NLPCC label for {source_id}: {label}")
        normalized.append(_normalized_row(
            benchmark="nlpcc-2026-task6",
            source_id=f"{phase}:{source_id}",
            variant={0: "human", 1: "generated", 2: "refined"}[label],
            text=row["text"],
            language="zh-CN",
            label_binary=0 if label == 0 else 1,
            label_ternary=label,
            domain=phase,
        ))
    if len(normalized) != len(label_by_id):
        raise ValueError(
            f"NLPCC row/label count mismatch: {len(normalized)} rows, {len(label_by_id)} labels"
        )
    return validate_release_rows(pd.DataFrame(normalized, columns=RELEASE_COLUMNS))


def normalize_cred(repo_path: str | Path) -> pd.DataFrame:
    """Normalize the complete C-ReD benchmark across domains and generators."""
    benchmark_root = Path(repo_path) / "benchmark data"
    rows = []
    paths = sorted(benchmark_root.glob("*/*.csv"))
    if not paths:
        raise ValueError(f"no C-ReD CSV files under {benchmark_root}")
    for path in paths:
        domain = path.parent.name
        prefix = f"CReD_{domain.replace(' ', '_')}_"
        if not path.stem.startswith(prefix):
            raise ValueError(f"unexpected C-ReD filename: {path.name}")
        generator = path.stem[len(prefix):]
        frame = pd.read_csv(path)
        for row in frame.to_dict("records"):
            is_human = int(row["label"]) == 1
            if is_human != (generator == "human"):
                raise ValueError(f"C-ReD filename/label mismatch in {path}")
            original_id = row.get("original_id")
            document_id = row["id"] if is_human or pd.isna(original_id) else original_id
            rows.append(_normalized_row(
                benchmark="c-red",
                source_id=f"{domain}:{document_id}",
                variant="human" if is_human else f"generated:{generator}",
                text=row["text"],
                language="zh-CN",
                label_binary=0 if is_human else 1,
                label_ternary=None,
                domain=domain,
                generator=None if is_human else generator,
            ))
    return validate_release_rows(pd.DataFrame(rows, columns=RELEASE_COLUMNS))


def validate_release_rows(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(RELEASE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"release rows missing columns: {sorted(missing)}")
    if df["row_id"].duplicated().any():
        duplicates = df.loc[df["row_id"].duplicated(), "row_id"].tolist()
        raise ValueError(f"duplicate release row IDs: {duplicates[:3]}")
    for row in df.itertuples():
        if hashlib.sha256(str(row.text).encode()).hexdigest() != row.text_sha256:
            raise ValueError(f"text hash mismatch: {row.row_id}")
    return df.sort_values("row_id").reset_index(drop=True)


def snapshot_metadata(df: pd.DataFrame, *, source: str, revision: str) -> dict:
    """Compact manifest binding a normalized snapshot to all ordered row hashes."""
    df = validate_release_rows(df)
    rows_digest = hashlib.sha256("\n".join(df["row_id"]).encode()).hexdigest()
    text_digest = hashlib.sha256("\n".join(df["text_sha256"]).encode()).hexdigest()
    records = df.drop(columns=["text"]).where(pd.notna(df.drop(columns=["text"])), None)
    records_digest = hashlib.sha256(
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for row in records.to_dict("records")
        ).encode()
    ).hexdigest()
    return {
        "benchmark": str(df["benchmark"].iloc[0]),
        "source": source,
        "revision": revision,
        "n": len(df),
        "rows_sha256": rows_digest,
        "texts_sha256": text_digest,
        "records_sha256": records_digest,
        "columns": RELEASE_COLUMNS,
        "label_mapping": {
            "label_binary": {"0": "human", "1": "AI-involved"},
            "label_ternary": {"0": "human", "1": "AI-generated", "2": "AI-edited"},
        },
    }


def write_snapshot(
    df: pd.DataFrame,
    data_path,
    manifest_path,
    *,
    source: str,
    revision: str,
    extra_metadata: dict | None = None,
) -> None:
    """Write normalized CSV plus a compact integrity manifest."""
    df = validate_release_rows(df)
    data_path = Path(data_path)
    manifest_path = Path(manifest_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    metadata = snapshot_metadata(df, source=source, revision=revision)
    if extra_metadata:
        overlap = set(metadata).intersection(extra_metadata)
        if overlap:
            raise ValueError(f"extra snapshot metadata replaces reserved fields: {sorted(overlap)}")
        metadata.update(extra_metadata)
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n")
