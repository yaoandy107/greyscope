"""Cost smokes for a same-row adversarial RAID comparison matrix."""
from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
app = modal.App("greyscope-raid-compare")
outputs_vol = modal.Volume.from_name("editlens-outputs", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
raid_cache_vol = modal.Volume.from_name("raid-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["compare", "modal"], frozen=True)
    .uv_pip_install("raid-bench")
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "PYTHONPATH": "/root/app",
    })
    .add_local_dir(str(ROOT / "greyscope"), remote_path="/root/app/greyscope")
    .add_local_file(
        str(ROOT / "configs" / "release_eval.json"),
        remote_path="/root/app/configs/release_eval.json",
    )
)

SAMPLE_PATH = Path("/root/app/outputs/release_eval/raid-adversarial-smoke.csv")
REPORT_DIR = Path("/root/app/outputs/release_eval/raid-adversarial-smoke")
EVAL_PATH = Path("/root/app/outputs/release_eval/raid-adversarial-4992.csv")
EVAL_DIR = Path("/root/app/outputs/release_eval/raid-adversarial-4992")
ENGLISH_EVAL_PATH = Path(
    "/root/app/outputs/release_eval/raid-adversarial-english-4992.csv"
)
ENGLISH_EVAL_DIR = Path(
    "/root/app/outputs/release_eval/raid-adversarial-english-4992"
)
BINOCULARS_EVAL_PATH = Path(
    "/root/app/outputs/release_eval/raid-adversarial-binoculars-compatible.csv"
)


@app.function(
    image=image,
    cpu=4,
    memory=8 * 1024,
    timeout=2 * 3600,
    volumes={
        "/root/app/outputs": outputs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def prepare_sample(rows_per_cell: int = 2) -> None:
    import pandas as pd
    from raid.utils import load_data

    rows = load_data(split="extra", include_adversarial=True)
    required = {"id", "generation", "domain", "attack"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"RAID extra is missing columns: {sorted(missing)}")
    pieces = [
        group.sort_values("id").head(rows_per_cell)
        for _, group in rows.groupby(["domain", "attack"], dropna=False, sort=True)
    ]
    sample = pd.concat(pieces, ignore_index=True).sort_values("id").reset_index(drop=True)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_PATH, index=False)
    outputs_vol.commit()
    print(json.dumps({
        "source_rows": len(rows),
        "sample_rows": len(sample),
        "domains": sorted(sample["domain"].astype(str).unique()),
        "attacks": sorted(sample["attack"].astype(str).unique()),
    }, indent=2))


@app.function(
    image=image,
    cpu=8,
    memory=64 * 1024,
    timeout=2 * 3600,
    volumes={
        "/root/app/outputs": outputs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/raid": raid_cache_vol,
    },
)
def prepare_evaluation(source_documents: int = 208, split: str = "extra") -> None:
    import hashlib

    import pandas as pd
    from raid.utils import load_data

    if split == "extra":
        output_path = EVAL_PATH
        output_dir = EVAL_DIR
    elif split == "train":
        output_path = ENGLISH_EVAL_PATH
        output_dir = ENGLISH_EVAL_DIR
    else:
        raise ValueError("split must be 'extra' or 'train'")
    rows = load_data(split=split, include_adversarial=True)
    required = {"id", "adv_source_id", "source_id", "generation", "model", "domain", "attack"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"RAID {split} is missing columns: {sorted(missing)}")

    attacks = sorted(rows["attack"].astype(str).unique())
    group_columns = ["adv_source_id", "source_id", "domain", "model"]
    groups = (
        rows.groupby(group_columns, dropna=False)
        .agg(n_rows=("id", "size"), n_attacks=("attack", "nunique"))
        .reset_index()
    )
    groups = groups[
        groups["n_rows"].eq(len(attacks)) & groups["n_attacks"].eq(len(attacks))
    ].copy()
    groups["label"] = groups["model"].ne("human").astype(int)

    def stable_key(value: str) -> str:
        return hashlib.sha256(str(value).encode()).hexdigest()

    eligible_sources = []
    for (domain, source_id), candidates in groups.groupby(["domain", "source_id"], sort=True):
        human = candidates[candidates["label"].eq(0)]
        machine = candidates[candidates["label"].eq(1)]
        if not human.empty and not machine.empty:
            eligible_sources.append({
                "domain": domain,
                "source_id": source_id,
                "human_adv_source_id": sorted(
                    human["adv_source_id"].astype(str), key=stable_key
                )[0],
            })
    eligible = pd.DataFrame(eligible_sources)
    domains = sorted(eligible["domain"].astype(str).unique())
    per_domain = {
        domain: source_documents // len(domains) + (index < source_documents % len(domains))
        for index, domain in enumerate(domains)
    }

    selected_groups = []
    model_counts: dict[str, int] = {}
    for domain in domains:
        domain_sources = eligible[eligible["domain"].astype(str).eq(domain)].copy()
        domain_sources["sort_key"] = domain_sources["source_id"].map(stable_key)
        domain_sources = domain_sources.sort_values("sort_key").head(per_domain[domain])
        for source in domain_sources.itertuples(index=False):
            source_groups = groups[
                groups["domain"].eq(source.domain)
                & groups["source_id"].eq(source.source_id)
            ]
            human = source_groups[
                source_groups["adv_source_id"].astype(str).eq(source.human_adv_source_id)
            ].iloc[0]
            machine = source_groups[source_groups["label"].eq(1)].copy()
            machine["model_count"] = machine["model"].map(model_counts).fillna(0)
            machine["sort_key"] = machine["adv_source_id"].map(stable_key)
            chosen = machine.sort_values(["model_count", "sort_key"]).iloc[0]
            model_counts[str(chosen["model"])] = model_counts.get(str(chosen["model"]), 0) + 1
            selected_groups.extend([
                str(human["adv_source_id"]),
                str(chosen["adv_source_id"]),
            ])

    selected = rows[rows["adv_source_id"].astype(str).isin(selected_groups)].copy()
    selected = selected.sort_values(["source_id", "model", "attack", "id"]).reset_index(drop=True)
    expected_rows = source_documents * 2 * len(attacks)
    if len(selected) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(selected)}")
    if selected["adv_source_id"].nunique() != source_documents * 2:
        raise ValueError("evaluation does not contain one human/AI pair per source")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)
    metadata = {
        "split": split,
        "source_rows": len(rows),
        "sample_rows": len(selected),
        "source_documents": source_documents,
        "base_generations": selected["adv_source_id"].nunique(),
        "domains": selected["domain"].value_counts().sort_index().to_dict(),
        "attacks": selected["attack"].value_counts().sort_index().to_dict(),
        "labels": {
            "human": int(selected["model"].eq("human").sum()),
            "ai": int(selected["model"].ne("human").sum()),
        },
        "ai_models": (
            selected[selected["model"].ne("human")]
            .groupby("model")["adv_source_id"]
            .nunique()
            .sort_index()
            .to_dict()
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sample-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    outputs_vol.commit()
    print(json.dumps(metadata, indent=2), flush=True)


@app.function(
    image=image,
    cpu=4,
    memory=8 * 1024,
    timeout=30 * 60,
    secrets=[hf_secret],
    volumes={
        "/root/app/outputs": outputs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def prepare_binoculars_evaluation() -> None:
    import pandas as pd
    from transformers import AutoTokenizer

    from greyscope.release_manifest import load_release_manifest

    if not EVAL_PATH.is_file():
        raise FileNotFoundError("run prepare_evaluation first")
    manifest = load_release_manifest("/root/app/configs/release_eval.json")
    model = next(item for item in manifest["models"] if item["id"] == "binoculars")
    tokenizer = AutoTokenizer.from_pretrained(
        model["observer_source"],
        revision=model["observer_revision"],
    )

    rows = pd.read_csv(EVAL_PATH)
    invalid_ids = []
    for start in range(0, len(rows), 256):
        batch = rows.iloc[start : start + 256]
        encoded = tokenizer(
            batch["generation"].astype(str).tolist(),
            truncation=True,
            max_length=model["max_length"],
            add_special_tokens=True,
        )
        invalid_ids.extend(
            row_id
            for row_id, token_ids in zip(batch["id"], encoded["input_ids"], strict=True)
            if len(token_ids) < 2
        )
    invalid_sources = set(rows.loc[rows["id"].isin(invalid_ids), "source_id"])
    compatible = rows[~rows["source_id"].isin(invalid_sources)].copy()
    compatible.to_csv(BINOCULARS_EVAL_PATH, index=False)
    metadata = {
        "original_rows": len(rows),
        "invalid_rows": len(invalid_ids),
        "removed_source_documents": len(invalid_sources),
        "compatible_rows": len(compatible),
        "policy": "remove every 24-row matched source pair containing a text below two Falcon tokens",
    }
    (EVAL_DIR / "binoculars-compatibility.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    outputs_vol.commit()
    print(json.dumps(metadata, indent=2), flush=True)


def _run_smoke(model_id: str, gpu_name: str) -> None:
    import time

    import numpy as np
    import pandas as pd

    from greyscope.detectors import (
        make_binoculars_scorer,
        make_editlens_llama_scorer,
        make_greyscope_scorer,
        make_meld_scorer,
        make_transformers_scorer,
    )
    from greyscope.release_manifest import load_release_manifest

    if not SAMPLE_PATH.is_file():
        raise FileNotFoundError("run prepare_sample first")
    manifest = load_release_manifest("/root/app/configs/release_eval.json")
    models = {model["id"]: model for model in manifest["models"]}
    if model_id not in models:
        raise ValueError(f"unknown model: {model_id}")
    model = models[model_id]
    scorer_factory = {
        "binoculars": make_binoculars_scorer,
        "editlens-llama": make_editlens_llama_scorer,
        "greyscope": make_greyscope_scorer,
        "meld": make_meld_scorer,
    }.get(model["adapter"], make_transformers_scorer)

    rows = pd.read_csv(SAMPLE_PATH)
    started = time.perf_counter()
    score_fn, _tokenizer, loaded_model = scorer_factory(
        model,
        device="cuda",
        batch_size=model.get("batch_size", 1),
    )
    load_seconds = time.perf_counter() - started
    score_started = time.perf_counter()
    scores = np.asarray(score_fn(rows["generation"].astype(str).tolist()), dtype=float)
    score_seconds = time.perf_counter() - score_started
    if scores.shape != (len(rows),) or not np.isfinite(scores).all():
        raise ValueError("smoke scorer returned invalid scores")
    report = {
        "model_id": model_id,
        "model_revision": model["revision"],
        "gpu": gpu_name,
        "rows": len(rows),
        "load_seconds": load_seconds,
        "score_seconds": score_seconds,
        "rows_per_second": len(rows) / score_seconds,
        "projected_score_seconds_for_10000_rows": score_seconds * 10000 / len(rows),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{model_id}.json").write_text(json.dumps(report, indent=2) + "\n")
    outputs_vol.commit()
    del loaded_model
    print(json.dumps(report, indent=2), flush=True)


def _run_evaluation(
    model_id: str,
    gpu_name: str,
    *,
    data_path: Path = EVAL_PATH,
    output_dir: Path = EVAL_DIR,
) -> None:
    import time

    import numpy as np
    import pandas as pd

    from greyscope.detectors import (
        make_binoculars_scorer,
        make_editlens_llama_scorer,
        make_greyscope_scorer,
        make_meld_scorer,
        make_transformers_scorer,
    )
    from greyscope.release_manifest import load_release_manifest

    if not data_path.is_file():
        raise FileNotFoundError("run prepare_evaluation first")
    manifest = load_release_manifest("/root/app/configs/release_eval.json")
    models = {model["id"]: model for model in manifest["models"]}
    if model_id not in models:
        raise ValueError(f"unknown model: {model_id}")
    model = models[model_id]
    scorer_factory = {
        "binoculars": make_binoculars_scorer,
        "editlens-llama": make_editlens_llama_scorer,
        "greyscope": make_greyscope_scorer,
        "meld": make_meld_scorer,
    }.get(model["adapter"], make_transformers_scorer)

    rows = pd.read_csv(data_path)
    started = time.perf_counter()
    score_fn, _tokenizer, loaded_model = scorer_factory(
        model,
        device="cuda",
        batch_size=model.get("batch_size", 1),
    )
    load_seconds = time.perf_counter() - started
    score_started = time.perf_counter()
    scores = np.asarray(score_fn(rows["generation"].astype(str).tolist()), dtype=float)
    score_seconds = time.perf_counter() - score_started
    if scores.shape != (len(rows),) or not np.isfinite(scores).all():
        raise ValueError("evaluation scorer returned invalid scores")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({"id": rows["id"], "score": scores})
    predictions.to_json(
        output_dir / f"{model_id}.jsonl",
        orient="records",
        lines=True,
    )
    report = {
        "model_id": model_id,
        "model_revision": model["revision"],
        "gpu": gpu_name,
        "rows": len(rows),
        "load_seconds": load_seconds,
        "score_seconds": score_seconds,
        "rows_per_second": len(rows) / score_seconds,
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }
    (output_dir / f"{model_id}-run.json").write_text(json.dumps(report, indent=2) + "\n")
    outputs_vol.commit()
    del loaded_model
    print(json.dumps(report, indent=2), flush=True)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24 * 1024,
    timeout=2 * 3600,
    secrets=[hf_secret],
    volumes={
        "/root/app/outputs": outputs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def standard_smoke(model_id: str) -> None:
    _run_smoke(model_id, "L4")


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=4,
    memory=48 * 1024,
    timeout=2 * 3600,
    secrets=[hf_secret],
    volumes={
        "/root/app/outputs": outputs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def editlens_smoke() -> None:
    _run_smoke("editlens-llama-3.2-3b", "A100-40GB")


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24 * 1024,
    timeout=2 * 3600,
    secrets=[hf_secret],
    volumes={
        "/root/app/outputs": outputs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def standard_evaluate(model_id: str) -> None:
    _run_evaluation(model_id, "L4")


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24 * 1024,
    timeout=2 * 3600,
    secrets=[hf_secret],
    volumes={
        "/root/app/outputs": outputs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def standard_evaluate_english(model_id: str) -> None:
    _run_evaluation(
        model_id,
        "L4",
        data_path=ENGLISH_EVAL_PATH,
        output_dir=ENGLISH_EVAL_DIR,
    )


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=4,
    memory=48 * 1024,
    timeout=2 * 3600,
    secrets=[hf_secret],
    volumes={
        "/root/app/outputs": outputs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def editlens_evaluate() -> None:
    _run_evaluation("editlens-llama-3.2-3b", "A100-40GB")


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=4,
    memory=48 * 1024,
    timeout=2 * 3600,
    secrets=[hf_secret],
    volumes={
        "/root/app/outputs": outputs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def editlens_evaluate_english() -> None:
    _run_evaluation(
        "editlens-llama-3.2-3b",
        "A100-40GB",
        data_path=ENGLISH_EVAL_PATH,
        output_dir=ENGLISH_EVAL_DIR,
    )


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=64 * 1024,
    timeout=2 * 3600,
    secrets=[hf_secret],
    volumes={
        "/root/app/outputs": outputs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def binoculars_smoke() -> None:
    _run_smoke("binoculars", "A100-80GB")


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=64 * 1024,
    timeout=2 * 3600,
    secrets=[hf_secret],
    volumes={
        "/root/app/outputs": outputs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def binoculars_evaluate() -> None:
    _run_evaluation(
        "binoculars",
        "A100-80GB",
        data_path=BINOCULARS_EVAL_PATH,
    )


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=64 * 1024,
    timeout=2 * 3600,
    secrets=[hf_secret],
    volumes={
        "/root/app/outputs": outputs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def binoculars_evaluate_english() -> None:
    _run_evaluation(
        "binoculars",
        "A100-80GB",
        data_path=ENGLISH_EVAL_PATH,
        output_dir=ENGLISH_EVAL_DIR,
    )
