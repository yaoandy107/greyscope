"""Training entrypoints: smoke (validate the pipeline), ablation (one knob), production.

`modal run modal/train.py::production` — full workflow and costs in modal/common.py.
"""

from __future__ import annotations

import subprocess

import modal

from common import (
    _TRAINING_SECRETS, _VOLUMES, app, hf_cache_vol, hf_secret, outputs_vol, triton_cache_vol,
)

# smoke: validate the trilingual pipeline end to end — splits + language/bucket sampler +
# CORN — on Qwen3.5-0.8B and 64 examples (~5 min, L4) before the full 4B run. The default
# Hydra config (configs/train.yaml) is the shipped recipe, so no --config-name is needed.
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
# ablation: the full recipe on a 12k stratified subset (preserves the language mix) to rank
# ONE knob before the single full run. Vary the knob + a distinct output_dir via --overrides
# (e.g. lora.r=16,training.output_dir=outputs/abl_r16). report_to=none — the curve lands in
# trainer_state.json. 3 epochs / patience 5 so CORN's slower conditional loss reaches its own
# convergence before the comparison. Starts FRESH (resume off) so a stale checkpoint can't
# freeze a baseline mid-run and invalidate the comparison.
ABLATION_OVERRIDES = [
    "data.train_subset=12000", "data.val_subset=600",
    "training.num_train_epochs=3",
    "training.eval_steps=250", "training.save_steps=250", "training.save_total_limit=1",
    "training.early_stopping_patience=5", "training.logging_steps=20",
    "training.resume_from_checkpoint=false",
    "training.report_to=none", "training.output_dir=outputs/ablation",
]


def _run_train(overrides: list[str] | None = None, env: dict[str, str] | None = None) -> None:
    """Run scripts/train.py inside the container with Hydra overrides."""
    import os

    cmd = ["python", "scripts/train.py"]
    if overrides:
        cmd.extend(overrides)
    print(f"+ {' '.join(cmd)}" + (f"  [env {env}]" if env else ""), flush=True)
    subprocess.run(cmd, check=True, cwd="/root/app", env={**os.environ, **(env or {})})


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
    gpu="A100-40GB",  # 4B at batch-8 peaks ~30 GB; 40 GB is enough and ~16% cheaper than 80 GB
    timeout=4 * 3600,
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    single_use_containers=True,
    max_containers=1,
    secrets=_TRAINING_SECRETS,
    volumes=_VOLUMES,
)
def ablation(overrides: str = "", env: str = "") -> None:
    """One-knob ablation on a 12k stratified subset x 3 epochs (~1 h, A100-40GB). Vary ONE
    knob + a distinct output_dir, e.g.:
      --overrides "lora.r=16,training.output_dir=outputs/abl_r16"
    """
    env_map = dict(kv.split("=", 1) for kv in env.split(",") if kv)
    _run_train(ABLATION_OVERRIDES + (_parse_overrides(overrides) or []), env=env_map or None)
    outputs_vol.commit()
    hf_cache_vol.commit()
    triton_cache_vol.commit()


@app.function(
    gpu="A100-40GB",  # 4B at batch-8 fits 40 GB; ~16% cheaper than 80 GB, same A100 compute
    timeout=10 * 3600,  # ~52k rows x 2 epochs / effective-batch-16 ≈ 6.5k steps; leave headroom
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    single_use_containers=True,
    max_containers=1,
    secrets=_TRAINING_SECRETS,
    volumes=_VOLUMES,
)
def production(overrides: str = "") -> None:
    """Train the shipped model: Qwen3.5-4B on the full trilingual splits, balanced jointly
    over language and bucket (configs/train.yaml). The single committed run."""
    _run_train(_parse_overrides(overrides))
    outputs_vol.commit()
    hf_cache_vol.commit()
    triton_cache_vol.commit()
