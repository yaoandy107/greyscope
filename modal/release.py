"""Release path, run once per shipped model: merge and validate the LoRA adapter,
confirm the merged model runs on CPU without fla, then push the artifact to the HF Hub.

`modal run modal/release.py::export_and_validate`, then ::export_quantized (the shipped
int4/fp8 artifact), then ::export_cpu_check, then ::push_to_hf.
"""

from __future__ import annotations

from pathlib import Path

import modal

from common import (
    _VOLUMES, MERGED_DEFAULT, OUT_ROOT, _load_merged, app, hf_secret, outputs_vol,
    use_app_packages,
)

# Mirrors a Mac deploy: no fla or unsloth, so transformers takes the portable torch GDN path.
cpu_infer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=5.5.0",
        "accelerate>=1.3.0",
        "peft>=0.14.0",
        "safetensors",
        "sentencepiece",
        "protobuf",
        "numpy",
        "emoji",  # greyscope.preprocess, pulled in via greyscope.inference
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    # Modal auto-mounts only the entrypoint file; the shared module and the greyscope
    # package (used by _load_merged) ship explicitly.
    .add_local_file(str(Path(__file__).resolve().parent / "common.py"), remote_path="/root/common.py")
    .add_local_dir(str(Path(__file__).resolve().parent.parent / "greyscope"), remote_path="/root/app/greyscope")
)


@app.function(
    gpu="L4",
    timeout=2 * 3600,  # full-test scoring at seq 2048 overruns 40 min on L4
    secrets=[hf_secret],
    volumes=_VOLUMES,
)
def export_and_validate(
    ckpt: str = "production_v2",
    val_subset: int = 1000,
    test_subset: int = 0,  # 0 = full test split, reproducing the trained ternary F1
    head: str = "corn",  # must match the trained head (train_v2.yaml model.head)
) -> None:
    """Merge the LoRA adapter into export_<run>/merged and assert the merge is faithful
    (guards against Unsloth #3206 corrupting seq-cls heads). Logic in greyscope/export.py."""
    import os

    use_app_packages(forbid_unsloth=False)  # export.py asserts the leak itself, post-import
    os.chdir("/root/app")  # prepare_v2_data resolves data/v2/splits relatively
    from greyscope import export

    export.export_and_validate(ckpt, OUT_ROOT, val_subset=val_subset,
                               test_subset=test_subset, data_source="v2", head=head,
                               on_saved=outputs_vol.commit)
    outputs_vol.commit()


@app.function(
    gpu="L4",  # fp8 quantization needs sm89+ — L4 yes, A100 (sm80) NO
    timeout=40 * 60,
    secrets=[hf_secret],
    volumes=_VOLUMES,
)
def export_quantized(
    merged: str = MERGED_DEFAULT,
    precision: str = "int4",  # "int4" (HQQ, CPU/MPS-friendly) | "fp8" (sm89+)
    head: str = "corn",
) -> None:
    """Quantize export_<run>/merged into the shipped-precision artifact next to it
    (export_<run>/int4 or /fp8) and probe faithfulness vs bf16. This is what B0 judges
    and what ships to the Hub."""
    import os

    use_app_packages(forbid_unsloth=False)
    os.chdir("/root/app")  # export_quantized's probe loads the v2 splits relatively
    from greyscope import export

    export.export_quantized(f"{OUT_ROOT}/{merged}", precision, head=head)
    outputs_vol.commit()


@app.function(
    image=cpu_infer_image,
    timeout=10 * 60,
    volumes=_VOLUMES,
)
def export_cpu_check(merged: str = MERGED_DEFAULT) -> None:
    """Confirm the merged model runs on CPU without fla, the premise of the Mac/MPS deploy."""
    import importlib.util

    import torch

    fla_present = importlib.util.find_spec("fla") is not None
    assert not fla_present, "fla is installed in the CPU image; this test would prove nothing"

    merged_dir = f"{OUT_ROOT}/{merged}"
    print(f"[cpu] loading {merged_dir} on CPU (float32, no fla)...")
    tok, model = _load_merged(merged_dir, dtype=torch.float32)
    print(f"[cpu] loaded. num_labels={model.config.num_labels}  device={next(model.parameters()).device}")

    texts = [
        "The mitochondria is the powerhouse of the cell, and i think thats pretty neat honestly.",
        "Furthermore, it is imperative to acknowledge that the multifaceted ramifications of this "
        "paradigm necessitate a comprehensive and holistic evaluation of the underlying frameworks.",
    ]
    enc = tok(texts, padding=True, truncation=True, max_length=2048,
              return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits.float(), dim=1)
    scalar = (probs @ torch.arange(model.config.num_labels).float()) / (model.config.num_labels - 1)
    print(f"[cpu] forward OK, logits shape {tuple(logits.shape)}")
    for i, t in enumerate(texts):
        print(f"[cpu]   text{i}: scalar_ai_score={scalar[i].item():.3f}  logits={logits[i].tolist()}")
    print("\n[cpu] PASS: merged seq-cls model runs on CPU without fla.")


@app.function(
    image=cpu_infer_image,
    timeout=30 * 60,
    secrets=[hf_secret],
    volumes=_VOLUMES,
)
def push_to_hf(
    repo: str = "yaoandy107/greyscope-qwen3.5-4b",
    merged: str = MERGED_DEFAULT,
    private: bool = True,
) -> None:
    """Push the merged artifact (weights, tokenizer, calibration.json) to the HF Hub.
    The repo lands private; flip it public on the Hub once the model card is in place."""
    import os

    from huggingface_hub import HfApi

    merged_dir = f"{OUT_ROOT}/{merged}"
    for required in ("model.safetensors", "config.json", "calibration.json"):
        assert os.path.isfile(f"{merged_dir}/{required}"), f"missing {required} in {merged_dir}"

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True)
    print(f"[push] uploading {merged_dir} → {repo} (private={private})...")
    api.upload_folder(
        folder_path=merged_dir,
        repo_id=repo,
        repo_type="model",
        commit_message="Greyscope — merged weights + calibration",
    )
    print(f"[push] done: https://huggingface.co/{repo}")
    print("[push] repo is private; add the model card, then set it public on the Hub.")
