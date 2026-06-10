"""Training entrypoints: smoke (validate the pipeline), ablation (one knob), production.

`modal run modal/train.py::production` — full workflow and costs in modal/common.py.
"""

from __future__ import annotations

import subprocess

import modal

from common import (
    _TRAINING_SECRETS, _VOLUMES, app, hf_cache_vol, hf_secret, outputs_vol, triton_cache_vol,
)

SMOKE_OVERRIDES = [
    "model.name=unsloth/Qwen3.5-0.8B-Base",
    "data.train_subset=64", "data.val_subset=16", "data.test_subset=64",
    "training.num_train_epochs=1",
    "training.per_device_train_batch_size=8", "training.per_device_eval_batch_size=2",
    "training.gradient_accumulation_steps=2",
    "training.eval_steps=2", "training.save_steps=4", "training.save_total_limit=2",
    "training.early_stopping_patience=10", "training.logging_steps=1",
    "training.report_to=none", "training.output_dir=outputs/smoke",
    "training.resume_from_checkpoint=false",
]
ABLATION_OVERRIDES = [
    "data.train_subset=8000", "data.val_subset=1000",
    "training.num_train_epochs=2",
    "training.eval_steps=100", "training.save_steps=100", "training.save_total_limit=2",
    "training.early_stopping_patience=3", "training.logging_steps=20",
    "training.output_dir=outputs/ablation",
]


def _run_train(overrides: list[str] | None = None) -> None:
    """Run scripts/train.py inside the container with Hydra overrides."""
    cmd = ["python", "scripts/train.py"]
    if overrides:
        cmd.extend(overrides)
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd="/root/app")


def _parse_overrides(overrides: str) -> list[str] | None:
    """Split the comma-separated `--overrides` flag; Modal CLI parameters are plain strings."""
    parsed = [x for x in overrides.split(",") if x] if overrides else []
    return parsed if parsed else None


@app.function(
    gpu="L4",
    timeout=900,
    secrets=[hf_secret],
    volumes=_VOLUMES,
)
def smoke(overrides: str = "") -> None:
    """Validate the pipeline end to end on Qwen3.5-0.8B and 64 examples (~5 min, L4).
    Extra knobs via e.g. `--overrides "data.seed=43"`."""
    _run_train(SMOKE_OVERRIDES + (_parse_overrides(overrides) or []))
    hf_cache_vol.commit()
    triton_cache_vol.commit()


@app.function(
    gpu="A100-80GB",
    timeout=2 * 3600,
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    single_use_containers=True,
    max_containers=1,
    secrets=_TRAINING_SECRETS,
    volumes=_VOLUMES,
)
def ablation(overrides: str = "") -> None:
    """One-knob ablation on 8k examples x 2 epochs (~45 min, A100-80GB).
    E.g. `--overrides "lora.r=64"`."""
    _run_train(ABLATION_OVERRIDES + (_parse_overrides(overrides) or []))
    outputs_vol.commit()
    hf_cache_vol.commit()
    triton_cache_vol.commit()


@app.function(
    gpu="A100-80GB",
    timeout=8 * 3600,
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    single_use_containers=True,
    max_containers=1,
    secrets=_TRAINING_SECRETS,
    volumes=_VOLUMES,
)
def production(overrides: str = "") -> None:
    """Train the shipped model: Qwen3.5-4B on the full 60k set (~4 h, A100-80GB —
    peaks ~55 GB; Hopper's GDN kernel is broken, fla #640)."""
    _run_train(_parse_overrides(overrides))
    outputs_vol.commit()
    hf_cache_vol.commit()
    triton_cache_vol.commit()
