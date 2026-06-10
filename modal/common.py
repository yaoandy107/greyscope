"""Shared Modal runtime for the train / evaluate / release entrypoints: image, app,
volumes, secrets, and the merged-model loader.

Reproducing the model, from the repo root:

    modal token new                                # one-time auth
    modal secret create huggingface-token HF_TOKEN=...  # plus wandb-token for W&B
    modal run modal/train.py::smoke                # pipeline smoke test (~$0.05, L4)
    modal run --detach modal/train.py::production  # full training (~4 h, ~$10, A100-80GB)
    modal run modal/release.py::export_and_validate  # merge LoRA into a deployable model

Run entrypoints by file path as above, not with `-m`: `modal` is the SDK package, not
this directory. `--detach` keeps long runs alive if the CLI disconnects.
"""

from __future__ import annotations

from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "cmake", "libedit-dev", "zlib1g-dev", "python3-dev")
    .uv_sync(extras=["train", "modal"], frozen=True)
    .env({
        "TRITON_CACHE_DIR": "/root/.triton/cache",
        "HF_HOME": "/root/.cache/huggingface",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    })
    # causal-conv1d omitted on purpose; the "fast path is not available" warning is benign.
    .add_local_dir(str(_ROOT / "greyscope"), remote_path="/root/app/greyscope")
    .add_local_dir(str(_ROOT / "scripts"), remote_path="/root/app/scripts")
    .add_local_dir(str(_ROOT / "configs"), remote_path="/root/app/configs")
    # Modal auto-mounts only the entrypoint file; this shared module ships explicitly.
    .add_local_file(str(Path(__file__).resolve()), remote_path="/root/common.py")
)

app = modal.App("greyscope", image=image)

outputs_vol = modal.Volume.from_name("editlens-outputs", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
triton_cache_vol = modal.Volume.from_name("triton-cache", create_if_missing=True)

_VOLUMES = {
    "/root/app/outputs": outputs_vol,
    "/root/.cache/huggingface": hf_cache_vol,
    "/root/.triton/cache": triton_cache_vol,
}
hf_secret = modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])
wandb_secret = modal.Secret.from_name("wandb-token", required_keys=["WANDB_API_KEY"])
_TRAINING_SECRETS = [hf_secret, wandb_secret]


# Paths inside the container; MERGED_DEFAULT is the shipped run's merged artifact.
OUT_ROOT = "/root/app/outputs"
MERGED_DEFAULT = "export_production_v2/merged"


def use_app_packages(forbid_unsloth: bool = True) -> None:
    """Make /root/app (the greyscope package) importable inside the container.
    Plain-transformers entrypoints also assert unsloth hasn't leaked into the process."""
    import sys

    if "/root/app" not in sys.path:
        sys.path.insert(0, "/root/app")
    if forbid_unsloth:
        assert "unsloth" not in sys.modules, "unsloth leaked into the plain-transformers path"


def _load_merged(merged_dir: str, *, dtype, device: str | None = None):
    """Load a merged seq-cls model + tokenizer with plain transformers."""
    use_app_packages()
    from greyscope.inference import load_seqcls_model

    return load_seqcls_model(merged_dir, dtype=dtype, device=device)
