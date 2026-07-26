"""Same-row release comparisons for pinned open Transformers detectors.

Smoke first, then run the complete snapshot:

    MODAL_PROFILE=yaoandy107 modal run modal/release_compare.py::transformers_eval \
      --benchmark c-red --snapshot-name c-red-sample \
      --model-id greyscope-v2-bf16 --limit 64
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
app = modal.App("greyscope-release-compare")
outputs_vol = modal.Volume.from_name("editlens-outputs", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["compare", "modal"], frozen=True)
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "PYTHONPATH": "/root/app",
    })
    .add_local_dir(str(ROOT / "greyscope"), remote_path="/root/app/greyscope")
    .add_local_file(str(ROOT / "configs" / "release_eval.json"), remote_path="/root/app/configs/release_eval.json")
)

for snapshot_file in (
    "apt-eval-sample",
    "beemo-sample",
    "c-red-sample",
    "nlpcc-2026-task6",
    "raid-10k",
):
    image = image.add_local_file(
        str(ROOT / "data" / "release" / f"{snapshot_file}.csv"),
        remote_path=f"/root/app/data/release/{snapshot_file}.csv",
    ).add_local_file(
        str(ROOT / "benchmarks" / "manifests" / f"{snapshot_file}.json"),
        remote_path=f"/root/app/benchmarks/manifests/{snapshot_file}.json",
    )


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24 * 1024,
    timeout=6 * 3600,
    secrets=[hf_secret],
    volumes={
        "/root/app/outputs": outputs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def transformers_eval(
    benchmark: str,
    model_id: str,
    snapshot_name: str = "",
    batch_size: int = 0,
    chunk_size: int = 256,
    limit: int = 0,
) -> None:
    import time

    import pandas as pd

    from greyscope.detectors import (
        make_editlens_llama_scorer,
        make_greyscope_scorer,
        make_meld_scorer,
        make_transformers_scorer,
    )
    from greyscope.release_data import snapshot_metadata
    from greyscope.release_manifest import load_release_manifest, release_models_for
    from greyscope.release_runner import evaluate_release_scores, score_snapshot

    manifest = load_release_manifest("/root/app/configs/release_eval.json")
    models = {model["id"]: model for model in release_models_for(manifest, benchmark)}
    if model_id not in models:
        raise ValueError(f"{model_id} is not selected for {benchmark}")
    model = models[model_id]
    batch_size = batch_size or model.get("batch_size", 32)
    if model["adapter"] not in {
        "transformers", "desklib", "fakespot", "greyscope", "editlens-llama", "meld"
    }:
        raise ValueError(f"{model_id} needs its dedicated {model['adapter']} runner")

    snapshot_name = snapshot_name or benchmark
    data_path = Path(f"/root/app/data/release/{snapshot_name}.csv")
    snapshot_path = Path(f"/root/app/benchmarks/manifests/{snapshot_name}.json")
    rows = pd.read_csv(data_path)
    snapshot = json.loads(snapshot_path.read_text())
    suffix = ""
    if limit:
        rows = rows.sort_values("row_id").head(limit).reset_index(drop=True)
        snapshot = snapshot_metadata(
            rows, source=snapshot["source"], revision=snapshot["revision"]
        )
        suffix = f"-smoke-{limit}"

    print(f"loading {model_id} at {model['revision']}", flush=True)
    started = time.perf_counter()
    scorer_factory = {
        "greyscope": make_greyscope_scorer,
        "editlens-llama": make_editlens_llama_scorer,
        "meld": make_meld_scorer,
    }.get(model["adapter"], make_transformers_scorer)
    score_fn, _tokenizer, loaded_model = scorer_factory(
        model, device="cuda", batch_size=batch_size
    )
    load_seconds = time.perf_counter() - started
    output_dir = Path(f"/root/app/outputs/release_eval/{snapshot_name}")
    prediction_path = output_dir / f"{model_id}{suffix}.jsonl"
    predictions = score_snapshot(
        rows,
        score_fn,
        prediction_path,
        model=model,
        snapshot=snapshot,
        chunk_size=chunk_size,
        on_chunk=outputs_vol.commit,
    )
    elapsed = time.perf_counter() - started
    calibration = getattr(loaded_model, "greyscope_calibration", None)
    metrics = evaluate_release_scores(
        rows,
        predictions,
        ternary_thresholds=(calibration["h_thresh"], calibration["ai_thresh"])
        if calibration else None,
        binary_threshold=calibration["binary_threshold"] if calibration else None,
    )
    report = {
        "model": model,
        "snapshot": snapshot,
        "runtime_batch_size": batch_size,
        "load_seconds": load_seconds,
        "elapsed_seconds": elapsed,
        "rows_per_second": len(rows) / max(elapsed - load_seconds, 1e-9),
        "metrics": metrics,
    }
    report_path = output_dir / f"{model_id}{suffix}-metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    outputs_vol.commit()
    del loaded_model
    print(json.dumps(report, indent=2), flush=True)
