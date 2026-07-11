"""Evaluation entrypoints, reproducing the README tables from the merged model on the
volume: OOD F1, the OpenPangram benchmark suite, and the deploy-ready calibration bundle.
Thin GPU wrappers; the logic lives in greyscope/benchmark.py and greyscope/calibration.py.

`modal run modal/evaluate.py::ood_eval` (likewise ::benchmark_suite, ::bundle_calibration).
"""

from __future__ import annotations

from common import (
    _VOLUMES, MERGED_DEFAULT, OUT_ROOT, _base_image, _load_merged, app, hf_secret, outputs_vol,
    use_app_packages, with_local_sources,
)

EVAL_DATA_DIR = "/tmp/evaldata"

# raid-bench is only needed for the official RAID entrypoint; layer it onto the base image
# (before the local-source mounts, which must stay last) rather than adding it to every
# function's dependencies. uv_pip_install, not pip_install: the uv_sync venv has no pip.
raid_image = with_local_sources(_base_image.uv_pip_install("raid-bench"))


def _json_persister(out_path: str, **extra):
    """Write results (+ extra fields) to `out_path` and commit the volume; passed as
    `on_split` so partial results survive a preempted run."""
    import json

    def persist(results: dict) -> None:
        with open(out_path, "w") as fh:
            json.dump({**extra, **results}, fh, indent=2)
        outputs_vol.commit()

    return persist


@app.function(
    gpu="A100-40GB",
    timeout=2 * 3600,
    secrets=[hf_secret],
    volumes=_VOLUMES,
)
def ood_eval(
    merged: str = MERGED_DEFAULT,
    ref_metrics: str = "production_v2/ternary_metrics.json",
) -> None:
    """OOD ternary macro-F1 on EditLens test_enron (domain shift) and test_llama (generator
    shift), with thresholds calibrated on in-domain val. Writes ood_metrics.json."""
    import json

    import torch

    use_app_packages()
    from greyscope.benchmark import run_ood_eval

    tok, model = _load_merged(f"{OUT_ROOT}/{merged}", dtype=torch.bfloat16, device="cuda")
    print(f"[ood] loaded {merged} (num_labels={model.config.num_labels})")

    # Quote the training run's in-domain F1 instead of re-scoring the full test split.
    with open(f"{OUT_ROOT}/{ref_metrics}") as fh:
        in_domain = float(json.load(fh)["metrics"]["macro_f1"])
    print(f"[ood] in-domain reference macro-F1 {in_domain:.4f} (from {ref_metrics})")

    out_path = f"{OUT_ROOT}/{merged.split('/')[0]}/ood_metrics.json"
    run_ood_eval(model, tok, in_domain=in_domain, on_split=_json_persister(out_path))
    print(f"[ood] wrote {out_path}")


@app.function(
    gpu="L4",  # fp8 inference kernels need sm89+ (L4 yes, A100 no); int4 tinygemm sm80+
    timeout=3 * 3600,
    secrets=[hf_secret],
    volumes=_VOLUMES,
)
def ood_eval_v2(
    model_dir: str,  # relative to outputs root, e.g. "export_qwen4b/int4"
    head: str = "corn",
    val_subset: int = 2000,  # calibration sample; 0 = full val (slow on L4)
    test_subset: int = 3000,  # in-domain judge sample; 0 = full test
    limit: int = 1200,  # per OOD split
    batch_size: int = 4,  # fp8 artifacts need 1: the triton fp8 act-quant NaNs on padded rows
) -> None:
    """Judge an exported artifact AT SHIPPED PRECISION on the v2 protocol — the B0
    scoreboard. Calibrates thresholds on v2 val at this precision, scores in-domain test
    (per-language) + the held-out OOD splits with the frozen thresholds, and writes
    at_precision_metrics.json next to the artifact."""
    import json
    import os

    import torch

    use_app_packages()
    os.chdir("/root/app")  # prepare_v2_data / eval_ood_splits resolve data/v2/splits relatively
    from greyscope.config import DataConfig
    from greyscope.data import prepare_v2_data
    from greyscope.eval import StandaloneScorer, eval_ood_splits, run_ternary_eval

    tok, model = _load_merged(f"{OUT_ROOT}/{model_dir}", dtype=torch.bfloat16, device="cuda")
    print(f"[ood-v2] loaded {model_dir} (num_labels={model.config.num_labels}, head={head})")
    scorer = StandaloneScorer(model, tok, batch_size=batch_size)

    dcfg = DataConfig(n_buckets=4, train_subset=100, seed=42,
                      val_subset=(val_subset or None), test_subset=(test_subset or None))
    data = prepare_v2_data(dcfg)
    ternary = run_ternary_eval(scorer, data.val, data.test, 4, head=head)
    m = ternary["metrics"]
    print(f"[ood-v2] in-domain macro-F1={m['macro_f1']:.4f} "
          f"(h {m['f1_human']:.3f} / ai {m['f1_ai_generated']:.3f} / ed {m['f1_ai_edited']:.3f})")
    for lang, pl in ternary.get("per_language", {}).items():
        print(f"[ood-v2]   [{lang}] macro-F1={pl['macro_f1']:.4f} h={pl['f1_human']:.3f} (n={pl['n']})")

    ood = eval_ood_splits(
        scorer, "data/v2/splits", ["test_llama", "test_enron", "ood_generator"], 4,
        head=head, flip=ternary["score_flipped"],
        h_thresh=ternary["h_thresh"], ai_thresh=ternary["ai_thresh"], limit=limit)
    for name, r in ood.items():
        pl = " ".join(f"{k}={v['macro_f1']:.3f}(n={v['n']})" for k, v in r["per_language"].items())
        print(f"[ood-v2] [{name:13s}] macro-F1={r['macro_f1']:.4f} (n={r['n']})  {pl}")

    m["confusion_matrix"] = m["confusion_matrix"].tolist()
    out_path = f"{OUT_ROOT}/{model_dir}/at_precision_metrics.json"
    with open(out_path, "w") as fh:
        json.dump({"model_dir": model_dir, "head": head, "ternary": ternary, "ood": ood,
                   "val_subset": val_subset, "test_subset": test_subset, "ood_limit": limit}, fh, indent=2)
    outputs_vol.commit()
    print(f"[ood-v2] wrote {out_path}")


@app.function(
    gpu="A100-40GB",
    timeout=2 * 3600,
    secrets=[hf_secret],
    volumes=_VOLUMES,
)
def benchmark_suite(merged: str = MERGED_DEFAULT) -> None:
    """Score Greyscope and every baseline `*_score` column through one harness, calibrated
    on in-domain val, across the OpenPangram splits. Writes benchmark_metrics.json."""
    import torch

    use_app_packages()
    from greyscope.benchmark import greyscope_score_fn, run_benchmark_suite

    tok, model = _load_merged(f"{OUT_ROOT}/{merged}", dtype=torch.bfloat16, device="cuda")
    print(f"[bench] loaded {merged} (num_labels={model.config.num_labels})", flush=True)

    run_root = f"{OUT_ROOT}/{merged.split('/')[0]}"
    out_path = f"{run_root}/benchmark_metrics.json"
    run_benchmark_suite(
        greyscope_score_fn(model, tok),
        data_dir=EVAL_DATA_DIR,
        scores_dir=run_root,
        on_split=_json_persister(out_path, merged=merged),
    )
    print(f"[bench] wrote {out_path}", flush=True)


@app.function(
    gpu="A100-40GB",
    timeout=30 * 60,
    secrets=[hf_secret],
    volumes=_VOLUMES,
)
def bundle_calibration(merged: str = MERGED_DEFAULT) -> None:
    """Write calibration.json into the merged dir so single-text inference needs no
    calibration split at deploy time. Derivation in greyscope/calibration.py."""
    import json

    import pandas as pd
    import torch

    use_app_packages()
    from greyscope.benchmark import fetch_editlens_csv
    from greyscope.calibration import build_calibration
    from greyscope.config import DataConfig
    from greyscope.data import prepare_data

    merged_dir = f"{OUT_ROOT}/{merged}"
    tok, model = _load_merged(merged_dir, dtype=torch.bfloat16, device="cuda")

    data = prepare_data(DataConfig(n_buckets=4, train_subset=100))  # full v1 val + test
    nn_texts = pd.read_csv(fetch_editlens_csv("nonnative_english", EVAL_DATA_DIR))["text"].tolist()
    calib = build_calibration(model, tok, data, {"non_native_english": nn_texts})
    with open(f"{merged_dir}/calibration.json", "w") as fh:
        json.dump(calib, fh, indent=2)
    outputs_vol.commit()
    print(f"[bundle] wrote {merged_dir}/calibration.json", flush=True)


@app.function(
    image=raid_image,
    gpu="A100-40GB",
    timeout=6 * 3600,
    secrets=[hf_secret],
    volumes=_VOLUMES,
)
def raid(merged: str = MERGED_DEFAULT, split: str = "extra", detector_name: str = "greyscope",
         limit: int | None = None, from_hf: bool = False, normalize: bool = True) -> None:
    """Official RAID (raid-bench) eval: TPR@FPR=5% over domain x generator x decoding x
    attack. Labeled splits (train/extra) score locally; `test` only dumps predictions for
    leaderboard submission. `from_hf=True` loads `merged` as an HF repo id (e.g. the v1
    model, for the baseline) instead of a volume dir. `normalize=False` disables the v2
    Unicode hardening — the baseline arm for measuring homoglyph/attack-axis recovery.
    Writes raid_<detector_name>_<split>/{predictions,metadata[,results]}.json."""
    import torch

    use_app_packages()
    from greyscope.benchmark import greyscope_score_fn
    from greyscope.raid_eval import resolve_flip, run_raid

    source = merged if from_hf else f"{OUT_ROOT}/{merged}"
    tok, model = _load_merged(source, dtype=torch.bfloat16, device="cuda")
    print(f"[raid] loaded {source} (num_labels={model.config.num_labels}) normalize={normalize}", flush=True)

    flip = resolve_flip(model, tok)
    out_dir = f"{OUT_ROOT}/raid_{detector_name}_{split}"
    run_raid(greyscope_score_fn(model, tok, normalize=normalize), flip=flip, out_dir=out_dir,
             split=split, detector_name=detector_name, limit=limit)
    outputs_vol.commit()
    print(f"[raid] wrote {out_dir}", flush=True)
