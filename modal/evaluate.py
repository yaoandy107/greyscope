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
    gpu="L4",  # fp8 inference kernels need sm89+ (L4 yes, A100 no); int4 tinygemm sm80+
    timeout=3 * 3600,
    secrets=[hf_secret],
    volumes=_VOLUMES,
)
def ood_eval(
    model_dir: str,  # relative to outputs root, e.g. "export_qwen4b/int4"
    head: str = "corn",
    val_subset: int = 2000,  # calibration sample; 0 = full val (slow on L4)
    test_subset: int = 3000,  # in-domain judge sample; 0 = full test
    limit: int = 1200,  # per OOD split
    batch_size: int = 4,  # fp8 artifacts need 1: the triton fp8 act-quant NaNs on padded rows
) -> None:
    """Judge an exported artifact AT SHIPPED PRECISION on the shipped protocol — the B0
    scoreboard. Calibrates thresholds on val at this precision, scores in-domain test
    (per-language) + the held-out OOD splits with the frozen thresholds, and writes
    at_precision_metrics.json next to the artifact."""
    import json
    import os

    import torch

    use_app_packages()
    os.chdir("/root/app")  # prepare_data / eval_ood_splits resolve data/v2/splits relatively
    os.environ["HF_DATASETS_CACHE"] = "/tmp/hfds_cache"  # re-parse changed split CSVs (the persistent
    #                                    hf-cache volume otherwise serves a stale arrow for the same path)
    from greyscope.config import DataConfig
    from greyscope.data import prepare_data
    from greyscope.eval import StandaloneScorer, eval_ood_splits, run_ternary_eval

    tok, model = _load_merged(f"{OUT_ROOT}/{model_dir}", dtype=torch.bfloat16, device="cuda")
    print(f"[ood] loaded {model_dir} (num_labels={model.config.num_labels}, head={head})")
    scorer = StandaloneScorer(model, tok, batch_size=batch_size)

    dcfg = DataConfig(n_buckets=4, train_subset=100, seed=42,
                      val_subset=(val_subset or None), test_subset=(test_subset or None))
    data = prepare_data(dcfg)
    ternary = run_ternary_eval(scorer, data.val, data.test, 4, head=head)
    m = ternary["metrics"]
    print(f"[ood] in-domain macro-F1={m['macro_f1']:.4f} "
          f"(h {m['f1_human']:.3f} / ai {m['f1_ai_generated']:.3f} / ed {m['f1_ai_edited']:.3f})")
    for lang, pl in ternary.get("per_language", {}).items():
        print(f"[ood]   [{lang}] macro-F1={pl['macro_f1']:.4f} h={pl['f1_human']:.3f} (n={pl['n']})")

    ood = eval_ood_splits(
        scorer, "data/v2/splits",
        ["test_llama", "test_enron", "ood_generator", "attack_paraphrase_test"], 4,
        head=head, flip=ternary["score_flipped"],
        h_thresh=ternary["h_thresh"], ai_thresh=ternary["ai_thresh"], limit=limit)
    for name, r in ood.items():
        pl = " ".join(f"{k}={v['macro_f1']:.3f}(n={v['n']})" for k, v in r["per_language"].items())
        print(f"[ood] [{name:13s}] macro-F1={r['macro_f1']:.4f} (n={r['n']})  {pl}")

    m["confusion_matrix"] = m["confusion_matrix"].tolist()
    out_path = f"{OUT_ROOT}/{model_dir}/at_precision_metrics.json"
    with open(out_path, "w") as fh:
        json.dump({"model_dir": model_dir, "head": head, "ternary": ternary, "ood": ood,
                   "val_subset": val_subset, "test_subset": test_subset, "ood_limit": limit}, fh, indent=2)
    outputs_vol.commit()
    print(f"[ood] wrote {out_path}")


@app.function(
    gpu="A100-40GB",
    timeout=60 * 60,
    secrets=[hf_secret],
    volumes=_VOLUMES,
)
def paraphrase_attack_eval(ckpt: str, head: str = "corn", val_subset: int = 2000,
                           test_subset: int = 3000, limit: int = 0,
                           split: str = "attack_paraphrase") -> None:
    """Eval-only: re-score an ablation LoRA checkpoint (un-merged) on ONE split — the
    paraphrase-attack robustness slice — with thresholds calibrated on val. Reuses the
    export loader (base + PeftModel, no merge) so a finished ablation arm is judged on a
    fixed slice without retraining. Writes attack_metrics.json next to the checkpoint."""
    import json
    import os

    import torch
    import torch.nn as nn
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # The parsed-CSV arrow is cached in the persistent hf-cache volume keyed by path; a fixed
    # slice at the same path would otherwise serve the stale arrow. Point datasets at a fresh
    # per-run cache so the CSV is re-parsed.
    os.environ["HF_DATASETS_CACHE"] = "/tmp/hfds_cache"

    use_app_packages()
    os.chdir("/root/app")
    from greyscope.config import DataConfig
    from greyscope.data import prepare_data
    from greyscope.eval import StandaloneScorer, eval_ood_splits, run_ternary_eval
    from greyscope.export import _resolve_best_ckpt

    # Guard BEFORE the model load: the attack slice MUST carry human negatives or detection
    # AUROC/TPR@FPR are undefined (single-class) — fail for ~$0, not after a full GPU load.
    import csv as _csv
    with open(f"data/v2/splits/{split}.csv", encoding="utf-8") as fh:
        _types = [row["text_type"] for row in _csv.DictReader(fh)]
    _n_human = sum(t == "human_written" for t in _types)
    print(f"[attack] slice {split}: {len(_types)} rows, {_n_human} human / {len(_types) - _n_human} AI", flush=True)
    assert _n_human > 0, f"{split} has no human rows — stale mount? detection metrics need both classes"

    n_out = (4 - 1) if head == "corn" else 4
    ckpt_dir = _resolve_best_ckpt(f"{OUT_ROOT}/{ckpt}")
    with open(f"{ckpt_dir}/adapter_config.json") as fh:
        base_name = json.load(fh)["base_model_name_or_path"]
    print(f"[attack] ckpt={ckpt_dir} base={base_name} head={head}", flush=True)

    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=n_out, dtype=torch.bfloat16)
    base.config.pad_token_id = tok.pad_token_id
    if base.score.out_features != n_out:  # VLM-composite config ignores num_labels=
        base.score = nn.Linear(base.score.in_features, n_out, bias=base.score.bias is not None).to(torch.bfloat16)
    base.config.head_type, base.config.n_buckets = head, 4
    model = PeftModel.from_pretrained(base, ckpt_dir).to("cuda").eval()
    scorer = StandaloneScorer(model, tok, batch_size=4)

    dcfg = DataConfig(n_buckets=4, train_subset=100, seed=42,
                      val_subset=val_subset, test_subset=test_subset)
    data = prepare_data(dcfg)
    ternary = run_ternary_eval(scorer, data.val, data.test, 4, head=head)
    ood = eval_ood_splits(scorer, "data/v2/splits", [split], 4, head=head,
                          flip=ternary["score_flipped"], h_thresh=ternary["h_thresh"],
                          ai_thresh=ternary["ai_thresh"], limit=(limit or None))
    r = ood[split]
    rd = r["detection"]

    def _f(x):
        return f"{x:.4f}" if isinstance(x, float) else "n/a"
    print(f"[attack] {split}: macro-F1={r['macro_f1']:.4f} AUROC={_f(rd['auroc'])} "
          f"TPR@1%={_f(rd['tpr@fpr1'])} TPR@5%={_f(rd['tpr@fpr5'])} (n={r['n']})", flush=True)
    for lang, pl in r["per_language"].items():
        pd_ = pl["detection"]
        print(f"[attack]   [{lang}] AUROC={_f(pd_['auroc'])} TPR@5%={_f(pd_['tpr@fpr5'])} (n={pl['n']})", flush=True)
    out_path = f"{OUT_ROOT}/{ckpt.rstrip('/')}/attack_metrics.json"
    with open(out_path, "w") as fh:
        json.dump({"ckpt": ckpt, "split": split, "result": ood}, fh, indent=2)
    outputs_vol.commit()
    print(f"[attack] wrote {out_path}", flush=True)


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
    from greyscope.data import prepare_editlens_data

    merged_dir = f"{OUT_ROOT}/{merged}"
    tok, model = _load_merged(merged_dir, dtype=torch.bfloat16, device="cuda")

    # TODO(calibration): this calibrates on EditLens EN only; the trilingual model needs
    # per-language calibration on the val split (hardest human subgroup at ≤1% FPR).
    data = prepare_editlens_data(DataConfig(n_buckets=4, train_subset=100))  # EditLens val + test
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
    model, for the baseline) instead of a volume dir. `normalize=False` disables the
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
