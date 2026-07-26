import json
from pathlib import Path

import pytest

from greyscope.release_manifest import (
    load_release_manifest,
    release_models_for,
    validate_release_manifest,
)


MANIFEST = Path(__file__).parents[1] / "configs" / "release_eval.json"


def test_release_manifest_is_valid_and_frozen():
    manifest = load_release_manifest(MANIFEST)
    assert manifest["cutoff"] == "2026-07-24"
    assert {model["id"] for model in manifest["models"]} == {
        "greyscope-v2-bf16",
        "greyscope-v1-bf16",
        "editlens-llama-3.2-3b",
        "desklib-v1.01",
        "meld",
        "binoculars",
    }
    assert all(len(model["revision"]) == 40 for model in manifest["models"])


def test_models_record_raid_training():
    manifest = load_release_manifest(MANIFEST)
    desklib = next(model for model in manifest["models"] if model["id"] == "desklib-v1.01")
    meld = next(model for model in manifest["models"] if model["id"] == "meld")
    assert desklib["trained_on_benchmarks"] == ["raid"]
    assert meld["trained_on_benchmarks"] == ["raid"]


def test_v1_is_compared_off_its_training_benchmark():
    manifest = load_release_manifest(MANIFEST)
    v1 = next(model for model in manifest["models"] if model["id"] == "greyscope-v1-bf16")
    assert v1["trained_on_benchmarks"] == ["editlens"]
    for benchmark in ("apt-eval", "beemo"):
        selected = {model["id"] for model in release_models_for(manifest, benchmark)}
        assert "greyscope-v1-bf16" in selected


def test_binoculars_pins_both_language_models():
    manifest = load_release_manifest(MANIFEST)
    model = next(model for model in manifest["models"] if model["id"] == "binoculars")
    assert model["observer_source"] == "tiiuae/falcon-7b"
    assert len(model["observer_revision"]) == 40
    assert model["performer_source"] == "tiiuae/falcon-7b-instruct"
    assert len(model["performer_revision"]) == 40


def test_meld_pins_checkpoint_and_backbone():
    manifest = load_release_manifest(MANIFEST)
    model = next(model for model in manifest["models"] if model["id"] == "meld")
    assert model["source"] == "anon-review-meld-2026/meld"
    assert len(model["revision"]) == 40
    assert model["base_source"] == "jhu-clsp/ettin-encoder-400m"
    assert len(model["base_revision"]) == 40
    assert model["max_length"] == 1024


def test_direct_editlens_competitor_is_included():
    manifest = load_release_manifest(MANIFEST)
    selected = {model["id"] for model in release_models_for(manifest, "editlens")}
    assert "editlens-llama-3.2-3b" in selected
    assert not any("roberta" in model_id for model_id in selected)


def test_manifest_rejects_unknown_exclusion():
    manifest = json.loads(MANIFEST.read_text())
    manifest["models"][0]["exclude_benchmarks"] = ["missing"]
    with pytest.raises(ValueError, match="unknown benchmarks"):
        validate_release_manifest(manifest)


def test_manifest_rejects_invalid_batch_size():
    manifest = json.loads(MANIFEST.read_text())
    manifest["models"][0]["batch_size"] = 0
    with pytest.raises(ValueError, match="positive batch_size"):
        validate_release_manifest(manifest)
