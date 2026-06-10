"""Evaluation entrypoints, reproducing the README tables from the merged model on the
volume: OOD F1, the OpenPangram benchmark suite, and the deploy-ready calibration bundle.
Thin GPU wrappers; the logic lives in greyscope/benchmark.py and greyscope/calibration.py.

`modal run modal/evaluate.py::ood_eval` (likewise ::benchmark_suite, ::bundle_calibration).
"""

from __future__ import annotations

from common import (
    _VOLUMES, MERGED_DEFAULT, OUT_ROOT, _load_merged, app, hf_secret, outputs_vol,
    use_app_packages,
)

EVAL_DATA_DIR = "/tmp/evaldata"


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

    merged_dir = f"{OUT_ROOT}/{merged}"
    tok, model = _load_merged(merged_dir, dtype=torch.bfloat16, device="cuda")

    nn_texts = pd.read_csv(fetch_editlens_csv("nonnative_english", EVAL_DATA_DIR))["text"].tolist()
    calib = build_calibration(model, tok, nn_texts)
    with open(f"{merged_dir}/calibration.json", "w") as fh:
        json.dump(calib, fh, indent=2)
    outputs_vol.commit()
    print(f"[bundle] wrote {merged_dir}/calibration.json", flush=True)
