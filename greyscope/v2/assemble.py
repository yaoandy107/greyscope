"""Assembly (design §14, plan §9): gated + scored rows → bucketed, source-doc-split,
training-shaped CSV splits + a prompt manifest.

Pure functions + a thin orchestrator; the upstream load→gate→score stages prepare the rows
(the build driver wires them, like the pilot). Two design steps that need inputs not yet
available are explicit TODOs in `assemble()`: the held-out generator/domain OOD slice
(design §11) and decontamination vs external benchmarks (RAID/COLING/long-form, §14).
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import yaml

from greyscope.preprocess import score_to_bucket

SPLITS_DIR = Path("data/v2/splits")
PROMPTS_DIR = Path(__file__).parent / "prompts"
N_BUCKETS = 4

# Per-language 4-bucket cuts, DERIVED from the full-build edit-cosine distribution (lo=p22, hi=p80
# per language → consistent ~22/35/22/20 spread) + perceptually cross-checked (2026-06-25, EXPERIMENTS).
# zh-tw edits run cold (median 0.037 vs en 0.073), so its cuts sit well below en's — not a bug.
BUCKET_CUTS = {"en": (0.030, 0.149), "ja": (0.019, 0.111), "zh-tw": (0.015, 0.069)}

# Dropped from the training/shipped view (design §4/§13): the human original the scorer
# needed, the build-only API telemetry, the cleanup review field, and the licensing tag.
_DROP_META = ("served_tier", "finish_reason", "usage", "stripped_header", "shippable_edit")

# Fallback split for a doc whose edit was gated out (most docs split by their edit's tag).
SPLIT_RATIO = {"train": 0.8, "val": 0.1, "test": 0.1}

# Held-out OOD slices (design §11) so a strong in-domain number means transfer, not memorization.
# EN inherits EditLens's own test_llama (unseen generator) + test_enron (unseen domain). ja/zh-tw
# hold out a LESS-IMPORTANT, CHEAP family — one the detector can afford to never train on (NOT a
# key app-channel player like deepseek/qwen): ja=ling (cheapest, distinct, non-reasoning), zh-tw=
# gemma (the only open model left after the mainland exclusion). A held-out *domain* for ja/zh-tw
# costs register coverage (≈ one source per register), so it is left to config — EN's Enron covers it.
OOD_GENERATOR = {"ja": "inclusionai/ling-2.6-flash", "zh-tw": "google/gemma-4-31b-it"}
OOD_DOMAIN: dict[str, str] = {}

_COLUMNS = ("text_id", "text", "language", "text_type", "source", "source_id",
            "model", "prompt_id", "markdown_mode", "cosine_score", "bucket", "meta")


def drop_unshippable_edits(rows: list[dict]) -> list[dict]:
    """Remove ai_edited rows from license-restricted (mirror-only) sources — wiki40b CC BY-SA,
    PTT unlicensed, Amazon — which generate.edit_row tags `shippable_edit=False`. They exist only
    for scorer validation; shipping an edit is shipping a derivative the license forbids (design §4).
    Human + mirror rows always stay (a mirror is new work, not a derivative)."""
    return [r for r in rows
            if not (r.get("text_type") == "ai_edited"
                    and r.get("meta", {}).get("shippable_edit") is False)]


def assign_buckets(rows: list[dict], cuts: dict = BUCKET_CUTS) -> list[dict]:
    """human=0 / generated=1 land in bucket 0 / n-1 by class; edited maps via its cosine."""
    for row in rows:
        if row.get("cosine_score") is not None:
            lo, hi = cuts[row["language"]]
            row["bucket"] = score_to_bucket(row["cosine_score"], N_BUCKETS, lo, hi)
    return rows


def _doc_key(row: dict) -> tuple:
    return (row["language"], row["source"], row["source_id"])


def _fallback_split(key: tuple) -> str:
    digest = hashlib.sha256("\x00".join(map(str, key)).encode("utf-8")).hexdigest()
    pick = (int(digest, 16) % 1000) / 1000
    cumulative = 0.0
    for split, frac in SPLIT_RATIO.items():
        cumulative += frac
        if pick < cumulative:
            return split
    return "train"


def assign_splits(rows: list[dict]) -> list[dict]:
    """Split by SOURCE DOC (design §14): every derivative of one human doc co-locates.
    Precedence: an inherited split (EN ingests EditLens's test_llama/test_enron) > held-out OOD
    domain > held-out OOD generator (§11) > the edit's `split_tag` (keeps edit prompts disjoint
    → the in-domain ratio equals the edit-tag ratio) > a seeded fallback."""
    by_doc: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_doc[_doc_key(row)].append(row)
    for key, group in by_doc.items():
        lang, source = key[0], key[1]
        inherited = next((r["split"] for r in group if r.get("split")), None)
        generator = next((r["model"] for r in group if r.get("model")), None)
        if inherited:
            split = inherited
        elif source == OOD_DOMAIN.get(lang):
            split = "ood_domain"
        elif generator is not None and generator == OOD_GENERATOR.get(lang):
            split = "ood_generator"
        else:
            edit = next((r for r in group if r["text_type"] == "ai_edited"
                         and r["meta"].get("split_tag")), None)
            split = edit["meta"]["split_tag"] if edit else _fallback_split(key)
        for row in group:
            row["split"] = split
    return rows


def to_split_row(row: dict) -> dict:
    """Strip the build-only fields (human original + API telemetry) from the canonical row."""
    meta = {k: v for k, v in row.get("meta", {}).items() if k not in _DROP_META}
    return {**{k: v for k, v in row.items() if k != "source_text"}, "meta": meta}


def write_splits(rows: list[dict], out_dir: Path = SPLITS_DIR) -> dict[str, int]:
    """One CSV per split PRESENT (train/val/test + any OOD or inherited slice); `meta` JSON-encoded."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_split[row.get("split", "train")].append(to_split_row(row))
    counts: dict[str, int] = {}
    for split, split_rows in by_split.items():
        counts[split] = len(split_rows)
        with (out_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in split_rows:
                writer.writerow({**row, "meta": json.dumps(row.get("meta", {}), ensure_ascii=False)})
    return counts


def write_prompt_manifest(out_dir: Path = SPLITS_DIR) -> Path:
    """Resolve every prompt id in the data → its text, so the dataset is interpretable
    standalone (release best-practice): system styles + edit prompts + mirror variants."""
    entries: list[dict] = []
    for lang in ("en", "ja", "zh-tw"):
        for style in yaml.safe_load((PROMPTS_DIR / "system" / f"{lang}.yaml").read_text("utf-8")):
            text = f"prompts/register/{lang}.md" if style["family"] == "humanizer" else style.get("text", "")
            entries.append({"type": "system", "language": lang, "id": style["id"],
                            "family": style["family"], "text": text})
        for edit in yaml.safe_load((PROMPTS_DIR / "edit" / f"{lang}.yaml").read_text("utf-8")):
            entries.append({"type": "edit", "language": lang, "id": edit["id"],
                            "category": edit["category"], "split": edit["split"], "text": edit["prompt"]})
        by_register = yaml.safe_load((PROMPTS_DIR / "mirror" / f"{lang}.yaml").read_text("utf-8"))
        for register, variants in by_register.items():
            for i, text in enumerate(variants, 1):
                entries.append({"type": "mirror", "language": lang, "id": f"{register}/v{i}", "text": text})
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "prompts_manifest.json"
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def assemble(rows: list[dict], out_dir: Path = SPLITS_DIR) -> dict[str, int]:
    """gated+scored rows → bucket → source-doc split (held-out OOD generator, §11) → strip →
    write splits + manifest.

    TODO before release: a held-out OOD *domain* for ja/zh-tw (config — costs register coverage),
    and decontamination vs {EN test, RAID, COLING ja/zh, long-form} (§14, needs the external sets).
    Near-dedup ran at gating; EN's EditLens test_llama/test_enron OOD ingest is wired in corpora."""
    rows = drop_unshippable_edits(rows)
    assign_buckets(rows)
    assign_splits(rows)
    counts = write_splits(rows, out_dir)
    write_prompt_manifest(out_dir)
    return counts
