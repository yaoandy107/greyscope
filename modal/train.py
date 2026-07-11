"""Training entrypoints: smoke (validate the pipeline), ablation (one knob), production.

`modal run modal/train.py::production` — full workflow and costs in modal/common.py.
"""

from __future__ import annotations

import subprocess

import modal

from common import (
    _TRAINING_SECRETS, _VOLUMES, _base_image, app, hf_cache_vol, hf_secret, outputs_vol,
    triton_cache_vol, with_local_sources,
)

# The encoder bake-off arm (mmBERT — ModernBERT arch, Gemma-2 tokenizer) may build its fast
# tokenizer from a SentencePiece model → keep sentencepiece + protobuf as insurance; layer them
# on rather than bloating every training image.
encoder_image = with_local_sources(_base_image.uv_pip_install("sentencepiece", "protobuf"))

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
# v2 base/recipe bake-off: the train_v2 recipe on a stratified subset (preserves the
# 63/19/18 EN/ja/zh-tw mix) so per-language in-domain F1 ranks bases/recipes cheaply
# before the single full run. report_to=none — curve lands in trainer_state.json. Pass
# the varied knob + a distinct output_dir via --overrides (e.g. model.name=..., lora.r=...).
ABLATION_V2_OVERRIDES = [
    "--config-name", "train_v2",
    "data.train_subset=12000", "data.val_subset=600",
    # 3 epochs + patience 5 so each head reaches its OWN convergence before we compare —
    # CORN's conditional loss converges slower (sparser per-task signal) and its curve is
    # bouncy, so a short/impatient budget under-serves it.
    "training.num_train_epochs=3",
    "training.eval_steps=250", "training.save_steps=250", "training.save_total_limit=1",
    "training.early_stopping_patience=5", "training.logging_steps=20",
    # ablations must start FRESH — resuming from a prior (or stalled) run's checkpoint
    # froze a baseline at 300 steps once and invalidated the comparison.
    "training.resume_from_checkpoint=false",
    "training.report_to=none", "training.output_dir=outputs/ablation_v2",
]
# v2 encoder bake-off: the train_v2_encoder recipe (mDeBERTa / XLM-R full-FT + CORN) on the SAME
# 12k stratified subset as ablation_v2, so per-language in-domain + OOD F1 compare directly to the
# 4B-decoder numbers. Vary base + output_dir via --overrides.
ABLATION_V2_ENCODER_OVERRIDES = [
    "--config-name", "train_v2_encoder",
    "data.train_subset=12000", "data.val_subset=600",
    "training.num_train_epochs=3",
    "training.eval_steps=250", "training.save_steps=250", "training.save_total_limit=1",
    "training.early_stopping_patience=5", "training.logging_steps=20",
    "training.resume_from_checkpoint=false",
    "training.report_to=none", "training.output_dir=outputs/ablation_v2_encoder",
]
# v2 recipe smoke: does a base/recipe actually LEARN on real data? The rung BETWEEN
# smoke_v2 (plumbing, 0.8B/64 rows — can't catch a learning collapse) and ablation_v2
# (12k bake-off). The real model on tiny real data, 1 epoch, frequent eval: a collapsed
# recipe reads at/below the random-floor eval_macro_f1 on the FIRST checkpoint, so it fails
# for ~$1 instead of a full ablation. Vary the base + output_dir via --overrides; hold
# everything else constant to isolate one knob. High patience so the whole 1-epoch curve runs.
RECIPE_SMOKE_V2_OVERRIDES = [
    "--config-name", "train_v2",
    "data.train_subset=4000", "data.val_subset=600", "data.test_subset=600",
    "training.num_train_epochs=1",
    "training.eval_steps=100", "training.save_steps=100", "training.save_total_limit=1",
    "training.early_stopping_patience=10", "training.logging_steps=20",
    "training.resume_from_checkpoint=false",
    "training.report_to=none", "training.output_dir=outputs/recipe_smoke_v2",
]
# v2 smoke: the train_v2 recipe (trilingual data + sampler + QAT) shrunk to 0.8B / 64 rows.
SMOKE_V2_OVERRIDES = [
    "--config-name", "train_v2",
    "model.name=unsloth/Qwen3.5-0.8B-Base",
    "data.train_subset=64", "data.val_subset=16", "data.test_subset=64",
    "training.num_train_epochs=1",
    "training.per_device_train_batch_size=8", "training.per_device_eval_batch_size=2",
    "training.gradient_accumulation_steps=2",
    "training.eval_steps=2", "training.save_steps=4", "training.save_total_limit=2",
    "training.early_stopping_patience=10", "training.logging_steps=1",
    "training.report_to=none", "training.output_dir=outputs/smoke_v2",
    "training.resume_from_checkpoint=false",
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
    gpu="A100-40GB",  # 4B at batch-8 peaks ~30 GB; 40 GB is enough and ~16% cheaper than 80 GB
    timeout=4 * 3600,
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    single_use_containers=True,
    max_containers=1,
    secrets=_TRAINING_SECRETS,
    volumes=_VOLUMES,
)
def ablation_v2(overrides: str = "", env: str = "") -> None:
    """v2 base/recipe bake-off on a 12k stratified subset x 2 epochs. Vary ONE knob +
    a distinct output_dir, e.g.:
      --overrides "training.output_dir=outputs/abl_qwen4b"  (default base: Qwen3.5-4B)
      --overrides "lora.r=32,training.output_dir=outputs/abl_r32"
    """
    env_map = dict(kv.split("=", 1) for kv in env.split(",") if kv)
    _run_train(ABLATION_V2_OVERRIDES + (_parse_overrides(overrides) or []), env=env_map or None)
    outputs_vol.commit()
    hf_cache_vol.commit()
    triton_cache_vol.commit()


@app.function(
    image=encoder_image,
    gpu="L4",  # a <1B encoder full-FT @512 fits an L4 with room; ~a quarter the A100 cost
    timeout=3 * 3600,
    secrets=_TRAINING_SECRETS,
    volumes=_VOLUMES,
)
def ablation_v2_encoder(overrides: str = "", env: str = "") -> None:
    """Encoder lite-tier bake-off: a multilingual ENCODER (mDeBERTa-v3 / XLM-R), FULL fine-tune
    + CORN, on the SAME 12k stratified subset as ablation_v2 — tests whether a ~300M encoder
    matches the 4B decoder (and would give an ONNX/MLX-clean browser tier). Vary base + output_dir:
      --overrides "training.output_dir=outputs/abl_mmbert"     (default: mmBERT-base, 307M)
      --overrides "model.name=jhu-clsp/mmBERT-small,training.output_dir=outputs/abl_mmbert_small"
    """
    env_map = dict(kv.split("=", 1) for kv in env.split(",") if kv)
    _run_train(ABLATION_V2_ENCODER_OVERRIDES + (_parse_overrides(overrides) or []), env=env_map or None)
    outputs_vol.commit()
    hf_cache_vol.commit()
    triton_cache_vol.commit()


@app.function(
    gpu="A100-40GB",  # 4B at batch-8 peaks ~30 GB; 40 GB is enough and ~16% cheaper than 80 GB
    timeout=90 * 60,
    secrets=_TRAINING_SECRETS,  # no retries: a crash should surface, not silently restart
    volumes=_VOLUMES,
)
def recipe_smoke_v2(overrides: str = "", env: str = "") -> None:
    """Does a base/recipe LEARN on real data? Tiny (4k x 1 epoch, ~15-25 min) — the rung
    between plumbing smoke_v2 and the 12k ablation_v2, so a collapsed recipe fails cheap.
    A healthy run clears the 0.25 random floor on the first eval; a collapsed one sits
    at/below it. Vary base + output_dir via --overrides; hold everything else constant to
    isolate one knob (e.g. lora.r).
    """
    env_map = dict(kv.split("=", 1) for kv in env.split(",") if kv)
    _run_train(RECIPE_SMOKE_V2_OVERRIDES + (_parse_overrides(overrides) or []), env=env_map or None)
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


@app.function(
    gpu="L4",
    timeout=900,
    secrets=[hf_secret],
    volumes=_VOLUMES,
)
def smoke_v2(overrides: str = "") -> None:
    """Validate the v2 pipeline end to end — trilingual splits + language/bucket sampler —
    on Qwen3.5-0.8B and 64 examples (~5 min, L4) before the full 4B run."""
    _run_train(SMOKE_V2_OVERRIDES + (_parse_overrides(overrides) or []))
    hf_cache_vol.commit()
    triton_cache_vol.commit()


@app.function(
    gpu="A100-40GB",  # 4B at batch-8 fits 40 GB; ~16% cheaper than 80 GB, same A100 compute
    timeout=10 * 3600,  # batch-8/accum-2 → 2x optimizer steps; full ~73k, leave headroom
    retries=modal.Retries(initial_delay=0.0, max_retries=3),
    single_use_containers=True,
    max_containers=1,
    secrets=_TRAINING_SECRETS,
    volumes=_VOLUMES,
)
def production_v2(overrides: str = "") -> None:
    """Train the v2 trilingual model: Qwen3.5-4B (fp8 at export) on the full v2 splits,
    balanced jointly over language and bucket (configs/train_v2.yaml). The single committed run."""
    _run_train(["--config-name", "train_v2"] + (_parse_overrides(overrides) or []))
    outputs_vol.commit()
    hf_cache_vol.commit()
    triton_cache_vol.commit()
