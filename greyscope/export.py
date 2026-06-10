"""Merge the LoRA seq-cls adapter and validate the standalone model.

Unsloth #3206 corrupts a merged seq-cls head, so the merge goes through plain PEFT,
and this asserts the standalone reload (no peft/unsloth) reproduces the pre-merge
logits and the trained ternary F1.
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


def export_and_validate(
    ckpt: str,
    out_root: str,
    *,
    val_subset: int = 1000,
    test_subset: int = 0,  # 0 = full test split, reproducing the trained ternary F1
    device: str = "cuda",
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
    from greyscope.data import prepare_data
    from greyscope.eval import (
        LABEL_TO_ID, calibrate_thresholds, compute_scalar_score, evaluate,
        minmax_scale, orient_scores, predict_ternary,
    )
    from greyscope.scoring import batch_logits

    assert "unsloth" not in sys.modules, "unsloth leaked into the plain-transformers export path"

    n_buckets, max_length = 4, 2048
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
    base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=n_buckets, dtype=torch.bfloat16)
    base.config.pad_token_id = tok.pad_token_id
    # This base ignores `num_labels=`; the head must match the trained [4, h] adapter slot.
    if base.score.out_features != n_buckets:
        print(f"[export] resizing base score head {base.score.out_features} → {n_buckets} "
              f"(num_labels= ignored by VLM-composite config)")
        base.score = nn.Linear(base.score.in_features, n_buckets, bias=base.score.bias is not None).to(dtype=torch.bfloat16)
        base.config.num_labels = n_buckets
        base.num_labels = n_buckets
    fresh_score = base.score.weight.detach().float().clone()

    print("[export] attaching LoRA adapter via PeftModel.from_pretrained...")
    peft_model = PeftModel.from_pretrained(base, ckpt_dir).to(device).eval()

    dcfg = DataConfig(n_buckets=n_buckets, train_subset=100, val_subset=val_subset,
                      test_subset=(test_subset or None), seed=42)
    print(f"[export] preparing data (val_subset={val_subset}, test={'full' if not test_subset else test_subset})...")
    data = prepare_data(dcfg)
    probe = data.val.select(range(min(64, len(data.val))))

    pre = batch_logits(peft_model, tok, probe["prompt"], max_length=max_length)

    print("[export] merge_and_unload() + save_pretrained()...")
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)
    if on_saved:
        on_saved()
    del peft_model, merged, base
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
    score_maxdiff = float(np.abs(compute_scalar_score(pre, n_buckets) - compute_scalar_score(post, n_buckets)).max())
    print(f"[export] pre-vs-post  argmax_agree={argmax_agree:.4f}  "
          f"logit_maxdiff={logit_maxdiff:.4f}  scalar_maxdiff={score_maxdiff:.5f}")
    assert argmax_agree == 1.0, "merge changed predictions — export NOT faithful (#3206)"
    assert score_maxdiff < 1e-2, f"scalar score drifted {score_maxdiff} after merge"

    print("[export] scoring val + test with standalone model for ternary F1...")
    val_scores = compute_scalar_score(batch_logits(standalone, tok, data.val["prompt"], max_length=max_length), n_buckets)
    test_scores = compute_scalar_score(batch_logits(standalone, tok, data.test["prompt"], max_length=max_length), n_buckets)
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
    assert abs(standalone_f1 - ref_f1) < 0.01, "standalone F1 diverged from trained — export not faithful"

    print("\n[export] PASS: merged export is faithful.")
    return {"merged_dir": merged_dir, "standalone_f1": standalone_f1, "ref_f1": ref_f1,
            "argmax_agree": argmax_agree, "score_maxdiff": score_maxdiff, "score_delta": score_delta}
