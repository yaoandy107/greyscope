"""Merge the LoRA seq-cls adapter, validate the standalone model, quantize for shipping.

Unsloth #3206 corrupts a merged seq-cls head, so the merge goes through plain PEFT,
and this asserts the standalone reload (no peft/unsloth) reproduces the pre-merge
logits and the trained ternary F1. `head="corn"` covers the K-1 ordinal head.
`export_quantized` then produces the shipped-precision artifact (int4-HQQ or fp8)
from the merged bf16 — unsloth's own torchao save is CausalLM-only, so this path is
plain transformers throughout.
"""
from __future__ import annotations

import glob
import json
import os


def _resolve_best_ckpt(path: str) -> str:
    """Accept an exact checkpoint dir or a run dir. For a run dir, read
    trainer_state.json's `best_model_checkpoint` (load_best_model_at_end records the
    best-by-eval_macro_f1 step, not the last step, which overfits); else the latest."""
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    ckpts = sorted(glob.glob(os.path.join(path, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[1]))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint-* under {path}")
    ts = os.path.join(ckpts[-1], "trainer_state.json")
    best = json.load(open(ts)).get("best_model_checkpoint") if os.path.exists(ts) else None
    if best:
        cand = os.path.join(path, os.path.basename(best))
        if os.path.exists(os.path.join(cand, "adapter_config.json")):
            print(f"[export] resolved best_model_checkpoint → {os.path.basename(cand)}")
            return cand
    print(f"[export] no best_model_checkpoint recorded; using latest {os.path.basename(ckpts[-1])}")
    return ckpts[-1]


def _scalar_decode(head: str, n_buckets: int):
    """Logits → scalar AI-ness in [0, 1] for the given head type (mirrors eval.py)."""
    if head == "corn":
        from greyscope.corn import corn_scalar_score

        return corn_scalar_score
    from greyscope.eval import compute_scalar_score

    return lambda logits: compute_scalar_score(logits, n_buckets)


def export_and_validate(
    ckpt: str,
    out_root: str,
    *,
    val_subset: int = 1000,
    test_subset: int = 0,  # 0 = full test split, reproducing the trained ternary F1
    device: str = "cuda",
    data_source: str = "v1",  # "v2" validates against the trilingual splits, not EditLens
    head: str = "seqcls",  # "corn" decodes K-1 ordinal logits (must match training)
    on_saved=None,
) -> dict:
    """Merge `ckpt`'s adapter → out_root/export_<run>/merged, then assert faithfulness.

    `on_saved` (optional) is called right after the merged weights are written and
    before the sanity checks, so the artifact can be persisted (e.g. a Modal volume
    commit) even if a later check fails. Returns the validation metrics.
    """
    import sys

    import numpy as np
    import torch
    import torch.nn as nn
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from greyscope.config import DataConfig
    from greyscope.data import prepare_data, prepare_v2_data
    from greyscope.eval import (
        LABEL_TO_ID, calibrate_thresholds, evaluate,
        minmax_scale, orient_scores, predict_ternary,
    )
    from greyscope.scoring import batch_logits

    assert "unsloth" not in sys.modules, "unsloth leaked into the plain-transformers export path"

    n_buckets, max_length = 4, 2048
    n_out = (n_buckets - 1) if head == "corn" else n_buckets
    scalar = _scalar_decode(head, n_buckets)
    ckpt_dir = _resolve_best_ckpt(f"{out_root}/{ckpt}")
    run_tag = ckpt.strip("/").split("/")[0]
    merged_dir = f"{out_root}/export_{run_tag}/merged"

    with open(f"{ckpt_dir}/adapter_config.json") as fh:
        adapter_cfg = json.load(fh)
    base_name = adapter_cfg["base_model_name_or_path"]
    print(f"[export] ckpt={ckpt_dir}")
    print(f"[export] base={base_name}  modules_to_save={adapter_cfg.get('modules_to_save')}  "
          f"r={adapter_cfg.get('r')}")

    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    print("[export] loading base AutoModelForSequenceClassification (plain transformers)...")
    base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=n_out, dtype=torch.bfloat16)
    base.config.pad_token_id = tok.pad_token_id
    # This base ignores `num_labels=`; the head must match the trained [n_out, h] adapter slot.
    if base.score.out_features != n_out:
        print(f"[export] resizing base score head {base.score.out_features} → {n_out} "
              f"(num_labels= ignored by VLM-composite config)")
        base.score = nn.Linear(base.score.in_features, n_out, bias=base.score.bias is not None).to(dtype=torch.bfloat16)
        base.config.num_labels = n_out
        base.num_labels = n_out
    # Ride the decode choice in the shipped config, matching what model.py sets at training.
    base.config.head_type = head
    base.config.n_buckets = n_buckets
    fresh_score = base.score.weight.detach().float().clone()

    dcfg = DataConfig(n_buckets=n_buckets, train_subset=100, val_subset=val_subset,
                      test_subset=(test_subset or None), seed=42)
    print(f"[export] preparing {data_source} data (val_subset={val_subset}, "
          f"test={'full' if not test_subset else test_subset})...")
    data = prepare_v2_data(dcfg) if data_source == "v2" else prepare_data(dcfg)
    probe = data.val.select(range(min(64, len(data.val))))

    # `pre` is the pre-save reference computed on the un-merged adapter (proving the merge itself
    # is faithful); it's compared to the reloaded standalone to prove the merge survives
    # save+reload (the #3206 head-reset guard).
    print("[export] attaching LoRA adapter via PeftModel.from_pretrained...")
    peft_model = PeftModel.from_pretrained(base, ckpt_dir).to(device).eval()
    pre = batch_logits(peft_model, tok, probe["prompt"], max_length=max_length)
    print("[export] merge_and_unload()...")
    merged = peft_model.merge_and_unload()
    del peft_model

    print("[export] save_pretrained()...")
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    if on_saved:
        on_saved()
    del merged, base
    torch.cuda.empty_cache()

    print("[export] fresh reload AutoModelForSequenceClassification.from_pretrained(merged)...")
    standalone = AutoModelForSequenceClassification.from_pretrained(merged_dir, dtype=torch.bfloat16).to(device).eval()
    standalone.config.pad_token_id = tok.pad_token_id
    assert "lm_head" not in dict(standalone.named_modules()), "stray lm_head in merged seq-cls model (#3206 symptom)"

    # The decisive #3206 check: the head must carry the trained weights, not a fresh init.
    standalone_score = standalone.score.weight.detach().float().cpu()
    score_delta = (standalone_score - fresh_score.cpu()).abs().max().item()
    print(f"[export] standalone score head |trained - fresh-init| max = {score_delta:.4f} "
          f"(must be >0 → trained head survived merge/save/reload, not reset)")
    assert score_delta > 1e-3, "score head reset to random on merge/reload — #3206 bit us"

    post = batch_logits(standalone, tok, probe["prompt"], max_length=max_length)

    pre_arg, post_arg = pre.argmax(1), post.argmax(1)
    argmax_agree = float((pre_arg == post_arg).mean())
    logit_maxdiff = float(np.abs(pre - post).max())
    score_maxdiff = float(np.abs(scalar(pre) - scalar(post)).max())
    print(f"[export] pre-vs-post  argmax_agree={argmax_agree:.4f}  "
          f"logit_maxdiff={logit_maxdiff:.4f}  scalar_maxdiff={score_maxdiff:.5f}")
    assert argmax_agree == 1.0, "merge changed predictions — export NOT faithful (#3206)"
    assert score_maxdiff < 1e-2, f"scalar score drifted {score_maxdiff} after merge"

    print("[export] scoring val + test with standalone model for ternary F1...")
    val_scores = scalar(batch_logits(standalone, tok, data.val["prompt"], max_length=max_length))
    test_scores = scalar(batch_logits(standalone, tok, data.test["prompt"], max_length=max_length))
    val_labels = np.asarray([LABEL_TO_ID[t] for t in data.val["text_type"]])
    test_labels = np.asarray([LABEL_TO_ID[t] for t in data.test["text_type"]])
    val_oriented, flipped = orient_scores(val_scores, val_labels)
    h_thresh, ai_thresh, _, _ = calibrate_thresholds(val_labels, minmax_scale(val_oriented))
    preds = predict_ternary(minmax_scale(-test_scores if flipped else test_scores), h_thresh, ai_thresh)
    metrics = evaluate(test_labels, preds)
    standalone_f1 = metrics["macro_f1"]

    with open(f"{out_root}/{run_tag}/ternary_metrics.json") as fh:
        ref_f1 = json.load(fh)["metrics"]["macro_f1"]

    print(f"[export] standalone ternary macro-F1 = {standalone_f1:.4f} "
          f"(per-class h={metrics['f1_human']:.3f} / ai={metrics['f1_ai_generated']:.3f} / edit={metrics['f1_ai_edited']:.3f})")
    print(f"[export] reference (in-training) macro-F1 = {ref_f1:.4f}  |Δ| = {abs(standalone_f1 - ref_f1):.4f}")
    if test_subset:
        # A subsampled test carries its own sampling noise vs the full-test reference;
        # the argmax/logit checks above already prove the merge is faithful.
        print(f"[export] test subsampled to {test_subset} — reference-F1 reproduction reported, not asserted")
    else:
        assert abs(standalone_f1 - ref_f1) < 0.01, "standalone F1 diverged from trained — export not faithful"

    print("\n[export] PASS: merged export is faithful.")
    return {"merged_dir": merged_dir, "standalone_f1": standalone_f1, "ref_f1": ref_f1,
            "argmax_agree": argmax_agree, "score_maxdiff": score_maxdiff, "score_delta": score_delta}


def _quantization_config(precision: str):
    """The shipped quantization recipe per precision. Both keep the score head bf16 —
    it is tiny and randomly initialized at train start, so quantizing it now would add
    error it never trained against."""
    if precision == "int4":
        # tile_packed_to_4d, NOT plain packing: plain quantizes only with the mslk lib
        # (torch-ABI-pinned wheels) and its inference GEMM needs sm90 TMA — dead on
        # L4/A100 and user laptops. Tinygemm tiles run on sm80+ and torch ships
        # _weight_int4pack_mm for CPU/MPS.
        # hqq qparams (NOT the tinygemm default): half-quadratic scale selection cuts int4
        # outlier error hard — the default min/max grid drifts the ai_involvement scalar by
        # up to ~0.86 on tail rows. Matches pytorch/Qwen3-8B-INT4's own recipe (~3.7% drop).
        # Only changes how qparams are chosen, not the packed format, so the CPU/MPS int4
        # kernel is unaffected.
        from torchao.quantization import Int4WeightOnlyConfig
        from transformers import TorchAoConfig

        return TorchAoConfig(
            quant_type=Int4WeightOnlyConfig(
                group_size=128, int4_packing_format="tile_packed_to_4d",
                int4_choose_qparams_algorithm="hqq",
            ),
            # score stays bf16 (tiny, random-init). Qwen3.5's GDN low-rank projections (32xH)
            # don't divide into int4 groups and hard-fail conversion — same modules fp8 excludes.
            modules_to_not_convert=[
                "score",
                "linear_attn.conv1d", "linear_attn.in_proj_a", "linear_attn.in_proj_b",
            ],
        )
    if precision == "fp8":
        # Qwen3.5's own shipped format: 128x128 block weights, dynamic activations.
        # The GDN exclusions mirror Qwen's official FP8 configs (e.g. Qwen3.5-27B-FP8):
        # the linear-attention low-rank projections (32xH) and conv1d don't divide into
        # 128x128 blocks — conversion hard-fails without them. Embeddings stay bf16 too.
        from transformers import FineGrainedFP8Config

        return FineGrainedFP8Config(modules_to_not_convert=[
            "score", "embed_tokens",
            "linear_attn.conv1d", "linear_attn.in_proj_a", "linear_attn.in_proj_b",
        ])
    raise ValueError(f"unknown precision {precision!r}; expected 'int4' or 'fp8'")


def export_quantized(
    merged_dir: str,
    precision: str,  # "int4" (HQQ, CPU/MPS-friendly) | "fp8" (needs sm89+: L4 yes, A100 no)
    *,
    head: str = "seqcls",
    device: str = "cuda",
    data_source: str = "v2",
    probe_size: int = 64,
) -> dict:
    """Quantize the merged bf16 export → sibling dir `<merged_dir>/../<precision>`,
    reload from disk, and probe faithfulness against the bf16 model.

    This is the artifact B0 judges and users download (`from_pretrained` just works).
    Probe drift is REPORTED, with only a coarse sanity floor asserted — the real
    at-precision quality measurement is the eval that follows, not this gate.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from greyscope.config import DataConfig
    from greyscope.data import prepare_data, prepare_v2_data
    from greyscope.scoring import batch_logits

    n_buckets, max_length = 4, 2048
    scalar = _scalar_decode(head, n_buckets)
    out_dir = os.path.join(os.path.dirname(merged_dir.rstrip("/")), precision)

    tok = AutoTokenizer.from_pretrained(merged_dir)
    dcfg = DataConfig(n_buckets=n_buckets, train_subset=100, val_subset=probe_size, seed=42)
    data = prepare_v2_data(dcfg) if data_source == "v2" else prepare_data(dcfg)
    probe = data.val.select(range(min(probe_size, len(data.val))))["prompt"]

    print(f"[quant] bf16 reference forward ({merged_dir})...")
    bf16 = AutoModelForSequenceClassification.from_pretrained(merged_dir, dtype=torch.bfloat16).to(device).eval()
    ref = batch_logits(bf16, tok, probe, max_length=max_length)
    del bf16
    torch.cuda.empty_cache()

    print(f"[quant] quantizing to {precision} → {out_dir} ...")
    quantized = AutoModelForSequenceClassification.from_pretrained(
        merged_dir, quantization_config=_quantization_config(precision),
        dtype=torch.bfloat16, device_map=device,
    )
    quantized.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    del quantized
    torch.cuda.empty_cache()

    print(f"[quant] fresh reload from {out_dir} ...")
    reloaded = AutoModelForSequenceClassification.from_pretrained(out_dir, dtype=torch.bfloat16, device_map=device).eval()
    # fp8 probes must run unbatched: transformers' triton fp8 act-quant NaNs on padded
    # rows (zero pad positions → scale 0), and the GDN recurrence smears it row-wide.
    # Weight quality is unaffected — batch-1 and dequantized loads both track bf16.
    probe_batch = 1 if precision == "fp8" else 16
    score_w = reloaded.score.weight
    assert score_w.dtype == torch.bfloat16 and type(score_w.data).__name__ == "Tensor", (
        f"score head must stay plain bf16, got {type(score_w.data).__name__} {score_w.dtype}"
    )
    quant = batch_logits(reloaded, tok, probe, max_length=max_length, batch_size=probe_batch)

    # Gate on the decoded scalar drift, NOT raw-logit argmax: CORN decodes via cumulative
    # sigmoids, so argmax of its K-1 conditional logits isn't the bucket (it read an identical
    # 0.72 for the broken and hqq int4 alike, while the real output diverged). scalar() is the
    # ai_involvement score both heads actually emit. bucket_agree is reported for context.
    if head == "corn":
        from greyscope.corn import corn_predict_buckets

        ref_b, quant_b = corn_predict_buckets(ref), corn_predict_buckets(quant)
    else:
        ref_b, quant_b = ref.argmax(1), quant.argmax(1)
    bucket_agree = float((ref_b == quant_b).mean())
    scalar_maxdiff = float(np.abs(scalar(ref) - scalar(quant)).max())
    print(f"[quant] bf16-vs-{precision}  scalar_maxdiff={scalar_maxdiff:.4f}  "
          f"bucket_agree={bucket_agree:.4f}")
    # Coarse catastrophe gate on a 64-row probe — the real quality bar is the at-precision
    # eval that follows (ood_eval_v2), not this. The broken default-int4-qparams recipe drifted
    # the score 0.68-0.86 on tail rows; int4+hqq sits ~0.26-0.43. 0.55 fails a broken recipe
    # (missing hqq / unexcluded GDN) without tripping on a merely-lossy-but-usable artifact.
    assert scalar_maxdiff <= 0.55, (
        f"{precision} ai-score drifts up to {scalar_maxdiff:.2f} vs bf16 — quantization looks "
        "broken, not merely lossy (default int4 qparams? GDN not excluded?)"
    )

    print(f"\n[quant] PASS: {precision} artifact saved, reloads, and tracks bf16.")
    return {"out_dir": out_dir, "precision": precision,
            "scalar_maxdiff": scalar_maxdiff, "bucket_agree": bucket_agree}
