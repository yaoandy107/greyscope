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
        "scipy",  # greyscope.corn (CORN decode)
        "emoji",  # greyscope.preprocess, pulled in via greyscope.inference
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    # Modal auto-mounts only the entrypoint file; the shared module and the greyscope
    # package (used by _load_merged) ship explicitly.
    .add_local_file(str(Path(__file__).resolve().parent / "common.py"), remote_path="/root/common.py")
    .add_local_dir(str(Path(__file__).resolve().parent.parent / "greyscope"), remote_path="/root/app/greyscope")
    # Model cards, pushed as each repo's README.md.
    .add_local_file(str(Path(__file__).resolve().parent.parent / "MODEL_CARD.md"), remote_path="/root/app/MODEL_CARD.md")
    .add_local_file(str(Path(__file__).resolve().parent.parent / "MODEL_CARD_INT4.md"), remote_path="/root/app/MODEL_CARD_INT4.md")
    .add_local_file(str(Path(__file__).resolve().parent.parent / "MODEL_CARD_MLX_INT4.md"), remote_path="/root/app/MODEL_CARD_MLX_INT4.md")
)

# MLX conversion does not need a Mac or a GPU. Running it on a roomy Linux CPU
# worker avoids downloading the 8.4 GB bf16 model to a user's laptop; only the
# finished 4-bit artifact needs to come back to the Mac for latency validation.
mlx_convert_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["mac", "modal", "eval"], frozen=True)
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    })
    .add_local_dir(
        str(Path(__file__).resolve().parent.parent / "greyscope"),
        remote_path="/root/app/greyscope",
    )
    .add_local_dir(
        str(Path(__file__).resolve().parent.parent / "scripts"),
        remote_path="/root/app/scripts",
    )
    .add_local_dir(
        str(Path(__file__).resolve().parent.parent / "data" / "v2" / "splits"),
        remote_path="/root/app/data/v2/splits",
    )
    .add_local_file(
        str(Path(__file__).resolve().parent / "common.py"),
        remote_path="/root/common.py",
    )
)


@app.function(
    gpu="L4",
    timeout=2 * 3600,  # full-test scoring at seq 2048 overruns 40 min on L4
    secrets=[hf_secret],
    volumes=_VOLUMES,
)
def export_and_validate(
    ckpt: str = "production",
    val_subset: int = 1000,
    test_subset: int = 0,  # 0 = full test split, reproducing the trained ternary F1
    head: str = "corn",  # must match the trained head (train.yaml model.head)
) -> None:
    """Merge the LoRA adapter into export_<run>/merged and assert the merge is faithful
    (guards against Unsloth #3206 corrupting seq-cls heads). Logic in greyscope/export.py."""
    import os

    use_app_packages(forbid_unsloth=False)  # export.py asserts the leak itself, post-import
    os.chdir("/root/app")  # prepare_data resolves data/v2/splits relatively
    from greyscope import export

    export.export_and_validate(ckpt, OUT_ROOT, val_subset=val_subset,
                               test_subset=test_subset, head=head,
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
    os.chdir("/root/app")  # export_quantized's probe loads the splits relatively
    from greyscope import export

    export.export_quantized(f"{OUT_ROOT}/{merged}", precision, head=head)
    outputs_vol.commit()


@app.function(
    image=mlx_convert_image,
    cpu=4,
    memory=32 * 1024,
    timeout=2 * 3600,
    volumes={"/root/app/outputs": outputs_vol},
)
def export_mlx(
    merged: str = MERGED_DEFAULT,
    bits: int = 4,
    group_size: int = 64,
) -> None:
    """Convert the merged classifier to MLX on a remote CPU and persist it.

    The artifact is built under ephemeral storage first. It reaches the output
    volume only after the custom classifier reloads and returns three finite CORN
    logits, so an interrupted or incompatible conversion cannot look complete.
    """
    import json
    import shutil
    import tempfile

    import mlx.core as mx
    from mlx_lm.convert import convert
    from mlx_lm.utils import load_model

    use_app_packages()
    from greyscope.mlx_export import stage_model

    source = Path(OUT_ROOT) / merged
    if bits != 4:
        raise ValueError("only the validated 4-bit MLX build is released")
    target = source.parent / "mlx-4bit"
    for required in ("model.safetensors", "config.json", "calibration.json"):
        assert (source / required).is_file(), f"missing {required} in {source}"
    if target.exists():
        raise FileExistsError(f"refusing to replace existing MLX artifact: {target}")

    with tempfile.TemporaryDirectory(prefix="greyscope-mlx-") as tmp:
        work = Path(tmp)
        stage = work / "stage"
        artifact = work / "artifact"
        stage.mkdir()
        stage_model(source, stage)
        convert(
            str(stage),
            str(artifact),
            quantize=True,
            q_bits=bits,
            q_group_size=group_size,
            dtype="bfloat16",
        )
        shutil.copy2(source / "calibration.json", artifact / "calibration.json")

        model, config = load_model(artifact, lazy=False, strict=True)
        logits = model(mx.array([[1, 2, 3, 4]]))
        mx.eval(logits)
        assert tuple(logits.shape) == (1, 3), f"unexpected classifier output {logits.shape}"
        assert bool(mx.all(mx.isfinite(logits)).item()), "MLX classifier returned non-finite logits"

        metadata = {
            "source": merged,
            "bits": bits,
            "group_size": group_size,
            "model_type": config.get("model_type"),
            "validation_logits": [float(x) for x in logits[0].tolist()],
        }
        (artifact / "greyscope_mlx_export.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        shutil.copytree(artifact, target)

    outputs_vol.commit()
    size_gb = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e9
    print(f"[mlx] PASS: wrote {target} ({size_gb:.2f} GB)", flush=True)


@app.function(
    gpu="A100-40GB",
    timeout=30 * 60,
    volumes=_VOLUMES,
)
def mlx_bf16_reference(merged: str = MERGED_DEFAULT) -> None:
    """Score the fixed MLX equivalence probe with the reference bf16 model."""
    import hashlib
    import json

    import torch

    use_app_packages()
    from greyscope.corn import corn_predict_buckets, corn_scalar_score
    from greyscope.equivalence import select_equivalence_probe
    from greyscope.scoring import batch_logits

    source = Path(OUT_ROOT) / merged
    probe = select_equivalence_probe("/root/app/data/v2/splits/val.csv")
    tok, model = _load_merged(str(source), dtype=torch.bfloat16, device="cuda")
    logits = batch_logits(model, tok, probe["prompt"].tolist(), max_length=2048)
    scores = corn_scalar_score(logits)
    buckets = corn_predict_buckets(logits)

    rows = []
    for index, row in probe.iterrows():
        rows.append({
            "text_id": row["text_id"],
            "language": row["language"],
            "bucket": int(row["bucket"]),
            "prompt_sha256": row["prompt_sha256"],
            "logits": [float(value) for value in logits[index]],
            "score": float(scores[index]),
            "predicted_bucket": int(buckets[index]),
        })

    payload = {
        "source": merged,
        "probe": "val language x bucket: 4 deterministic random + longest",
        "max_length": 2048,
        "n": len(rows),
        "rows": rows,
    }
    encoded = json.dumps(payload, indent=2) + "\n"
    payload["rows_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    out_path = source.parent / "mlx-bf16-reference.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    outputs_vol.commit()
    print(f"[mlx] wrote bf16 reference for {len(rows)} rows to {out_path}", flush=True)


@app.function(
    gpu="A100-40GB",
    timeout=30 * 60,
    volumes=_VOLUMES,
)
def mlx_task_reference(merged: str = MERGED_DEFAULT, per_group: int = 20) -> None:
    """Score the deterministic trilingual task probe with reference bf16."""
    import json

    import numpy as np
    import torch

    use_app_packages()
    from greyscope.corn import corn_scalar_score
    from greyscope.equivalence import select_task_probe
    from greyscope.eval import LABEL_TO_ID, detection_from_scalar, evaluate, predict_ternary
    from greyscope.preprocess import clean_text
    from greyscope.scoring import batch_logits

    source = Path(OUT_ROOT) / merged
    calibration = json.loads((source / "calibration.json").read_text())
    probe = select_task_probe("/root/app/data/v2/splits/test.csv", per_group=per_group)
    prompts = [
        calibration["prompt_template"].format(
            text=clean_text(str(text)) if calibration["lowercase"] else str(text)
        )
        for text in probe["text"]
    ]
    tokenizer, model = _load_merged(str(source), dtype=torch.bfloat16, device="cuda")
    logits = batch_logits(model, tokenizer, prompts, max_length=calibration["max_length"])
    scores = corn_scalar_score(logits)
    oriented = -scores if calibration["flip"] else scores
    scaled = np.clip(
        (oriented - calibration["score_min"])
        / (calibration["score_max"] - calibration["score_min"]),
        0.0,
        1.0,
    )
    labels = probe["text_type"].map(LABEL_TO_ID).to_numpy()
    predictions = predict_ternary(scaled, calibration["h_thresh"], calibration["ai_thresh"])
    metrics = evaluate(labels, predictions)
    metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    payload = {
        "source": merged,
        "selection": {"per_group": per_group, "n": len(probe)},
        "metrics": metrics,
        "detection": detection_from_scalar(oriented, labels),
        "rows": [
            {
                "text_id": row["text_id"],
                "language": row["language"],
                "text_type": row["text_type"],
                "score": float(scores[index]),
                "scaled_score": float(scaled[index]),
                "prediction": int(predictions[index]),
                "logits": [float(value) for value in logits[index]],
            }
            for index, row in probe.iterrows()
        ],
    }
    out_path = source.parent / "mlx-task-bf16-reference.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    outputs_vol.commit()
    print(json.dumps({"metrics": metrics, "detection": payload["detection"]}, indent=2))


@app.function(
    image=mlx_convert_image,
    cpu=8,
    memory=32 * 1024,
    timeout=60 * 60,
    volumes={"/root/app/outputs": outputs_vol},
)
def mlx_native_bf16_probe(merged: str = MERGED_DEFAULT) -> None:
    """Isolate native MLX runtime drift before any weight quantization."""
    import json
    import tempfile

    import mlx.core as mx
    import numpy as np
    from mlx_lm import load

    use_app_packages()
    from greyscope.corn import corn_predict_buckets, corn_scalar_score
    from greyscope.equivalence import select_equivalence_probe
    from greyscope.export import quantization_equivalence
    from greyscope.mlx_export import stage_model

    source = Path(OUT_ROOT) / merged
    reference_path = source.parent / "mlx-bf16-reference.json"
    reference = json.loads(reference_path.read_text())
    ref_rows = {row["text_id"]: row for row in reference["rows"]}
    probe = select_equivalence_probe("/root/app/data/v2/splits/val.csv")

    with tempfile.TemporaryDirectory(prefix="greyscope-mlx-bf16-") as tmp:
        stage = Path(tmp)
        stage_model(source, stage)
        model, tokenizer = load(str(stage), lazy=False)
        logits = []
        for index, row in probe.iterrows():
            ref = ref_rows[row["text_id"]]
            assert ref["prompt_sha256"] == row["prompt_sha256"]
            token_ids = tokenizer.encode(
                row["prompt"], add_special_tokens=False
            )[: reference["max_length"]]
            output = model(mx.array([token_ids]))
            mx.eval(output)
            logits.append([float(value) for value in output[0].tolist()])
            print(f"[mlx-bf16] {index + 1:02d}/{len(probe)}", flush=True)

    logits_array = np.asarray(logits)
    scores = corn_scalar_score(logits_array)
    buckets = corn_predict_buckets(logits_array)
    ref_scores = np.asarray([ref_rows[text_id]["score"] for text_id in probe["text_id"]])
    ref_buckets = np.asarray([
        ref_rows[text_id]["predicted_bucket"] for text_id in probe["text_id"]
    ])
    metrics = quantization_equivalence(ref_scores, scores, ref_buckets, buckets)
    payload = {
        "source": merged,
        "runtime": "mlx-bf16",
        "reference": str(reference_path),
        "metrics": metrics,
        "rows": [
            {
                "text_id": row["text_id"],
                "prompt_sha256": row["prompt_sha256"],
                "transformers_score": float(ref_scores[index]),
                "mlx_score": float(scores[index]),
                "transformers_bucket": int(ref_buckets[index]),
                "mlx_bucket": int(buckets[index]),
                "mlx_logits": logits[index],
            }
            for index, row in probe.iterrows()
        ],
    }
    out_path = source.parent / "mlx-native-bf16-probe.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    outputs_vol.commit()
    print(f"[mlx-bf16] metrics={json.dumps(metrics)}", flush=True)


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
    if getattr(model.config, "head_type", "seqcls") == "corn":
        from greyscope.corn import corn_scalar_score
        scalar = torch.from_numpy(corn_scalar_score(logits.float().numpy()))
    else:
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
    repo: str = "yaoandy107/greyscope-v2-qwen3.5-4b",
    merged: str = MERGED_DEFAULT,
    private: bool = True,
) -> None:
    """Push bf16 and int4 artifacts and set their requested Hub visibility."""
    import os
    import shutil

    from huggingface_hub import HfApi

    merged_dir = f"{OUT_ROOT}/{merged}"
    int4_dir = f"{os.path.dirname(merged_dir)}/int4"
    for required in ("model.safetensors", "config.json", "calibration.json"):
        assert os.path.isfile(f"{merged_dir}/{required}"), f"missing {required} in {merged_dir}"
    assert os.path.isdir(int4_dir), f"missing int4 artifact at {int4_dir}"

    shutil.copyfile("/root/app/MODEL_CARD.md", f"{merged_dir}/README.md")
    shutil.copyfile("/root/app/MODEL_CARD_INT4.md", f"{int4_dir}/README.md")
    if not os.path.isfile(f"{int4_dir}/calibration.json"):  # int4 users need the thresholds too
        shutil.copyfile(f"{merged_dir}/calibration.json", f"{int4_dir}/calibration.json")

    api = HfApi(token=os.environ["HF_TOKEN"])
    for repo_id, folder in ((repo, merged_dir), (f"{repo}-int4", int4_dir)):
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"[push] uploading {folder} → {repo_id} (private={private})...")
        api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type="model",
                          commit_message="Greyscope v2 — weights + calibration")
        api.update_repo_settings(repo_id=repo_id, repo_type="model", private=private)
        print(f"[push] done: https://huggingface.co/{repo_id} (private={private})")


@app.function(
    image=cpu_infer_image,
    timeout=30 * 60,
    secrets=[hf_secret],
    volumes=_VOLUMES,
)
def push_mlx_to_hf(
    repo: str = "",
    merged: str = MERGED_DEFAULT,
    bits: int = 4,
    private: bool = True,
) -> None:
    """Upload the validated native MLX Q4 artifact and set its Hub visibility."""
    import os
    import shutil

    from huggingface_hub import HfApi

    if bits != 4:
        raise ValueError("only the validated 4-bit MLX build is released")
    if not repo:
        repo = "yaoandy107/greyscope-v2-qwen3.5-4b-mlx-4bit"
    artifact = Path(OUT_ROOT) / Path(merged).parent / "mlx-4bit"
    for required in ("model.safetensors", "config.json", "calibration.json", "mlx_model.py"):
        assert (artifact / required).is_file(), f"missing {required} in {artifact}"
    shutil.copyfile("/root/app/MODEL_CARD_MLX_INT4.md", artifact / "README.md")

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=artifact,
        repo_id=repo,
        repo_type="model",
        commit_message="Greyscope v2 MLX 4-bit weights and calibration",
    )
    api.update_repo_settings(repo_id=repo, repo_type="model", private=private)
    print(f"[push] done: https://huggingface.co/{repo} (private={private})")
