"""Validation and selection for the frozen v2 comparison matrix."""
from __future__ import annotations

import json
from pathlib import Path


def load_release_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    validate_release_manifest(manifest)
    return manifest


def validate_release_manifest(manifest: dict) -> None:
    model_ids = [model["id"] for model in manifest["models"]]
    benchmark_ids = [benchmark["id"] for benchmark in manifest["benchmarks"]]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("release manifest contains duplicate model IDs")
    if len(benchmark_ids) != len(set(benchmark_ids)):
        raise ValueError("release manifest contains duplicate benchmark IDs")

    known_benchmarks = set(benchmark_ids)
    for model in manifest["models"]:
        revision = model.get("revision", "")
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(f"{model['id']} must pin a 40-character commit revision")
        if model["kind"] not in {"binary", "graded"}:
            raise ValueError(f"unknown model kind for {model['id']}: {model['kind']}")
        if model.get("adapter") not in {
            "greyscope",
            "editlens-llama",
            "transformers",
            "desklib",
            "meld",
            "fakespot",
            "binoculars",
            "fast-detect-gpt",
        }:
            raise ValueError(f"unknown adapter for {model['id']}: {model.get('adapter')}")
        if model.get("adapter") == "greyscope":
            if model.get("head_type") not in {"corn", "seqcls"}:
                raise ValueError(f"{model['id']} must declare its Greyscope head type")
            if not isinstance(model.get("normalize_unicode"), bool):
                raise ValueError(f"{model['id']} must declare normalize_unicode")
        if not isinstance(model.get("max_length"), int) or model["max_length"] <= 0:
            raise ValueError(f"{model['id']} must declare a positive max_length")
        if not isinstance(model.get("batch_size", 1), int) or model.get("batch_size", 1) <= 0:
            raise ValueError(f"{model['id']} must declare a positive batch_size")
        if model["adapter"] == "editlens-llama":
            base_revision = model.get("base_revision", "")
            if len(base_revision) != 40 or any(
                c not in "0123456789abcdef" for c in base_revision
            ):
                raise ValueError(f"{model['id']} must pin its base model revision")
            if not model.get("base_source"):
                raise ValueError(f"{model['id']} must declare its base model source")
        if model["adapter"] == "meld":
            base_revision = model.get("base_revision", "")
            if len(base_revision) != 40 or any(
                c not in "0123456789abcdef" for c in base_revision
            ):
                raise ValueError(f"{model['id']} must pin its base model revision")
            if not model.get("base_source"):
                raise ValueError(f"{model['id']} must declare its base model source")
        if model["adapter"] == "binoculars":
            for field in ("observer_source", "performer_source"):
                if not model.get(field):
                    raise ValueError(f"{model['id']} must declare {field}")
            for field in ("observer_revision", "performer_revision"):
                value = model.get(field, "")
                if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError(f"{model['id']} must pin {field}")
        referenced = set(model.get("exclude_benchmarks", [])) | set(
            model.get("include_benchmarks", [])
        ) | set(model.get("trained_on_benchmarks", []))
        unknown = referenced - known_benchmarks
        if unknown:
            raise ValueError(f"{model['id']} excludes unknown benchmarks: {sorted(unknown)}")
        if model["source"].startswith(("http://", "https://")):
            if "github.com" not in model["source"]:
                raise ValueError(f"comparison source is not a public model or code repository: {model['id']}")


def release_models_for(manifest: dict, benchmark_id: str) -> list[dict]:
    benchmark = next(
        (row for row in manifest["benchmarks"] if row["id"] == benchmark_id), None
    )
    if benchmark is None:
        raise KeyError(f"unknown benchmark: {benchmark_id}")

    languages = set(benchmark["languages"])
    selected = []
    for model in manifest["models"]:
        if benchmark_id in model.get("exclude_benchmarks", []):
            continue
        model_languages = set(model["languages"])
        explicitly_included = benchmark_id in model.get("include_benchmarks", [])
        if (
            not explicitly_included
            and "multilingual" not in languages
            and not languages.intersection(model_languages)
        ):
            continue
        selected.append(model)
    return selected
