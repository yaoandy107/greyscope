"""Build the deploy-ready calibration (calibration.json) from a trained model.

Derives, from the in-domain val split plus a non-native-English human set, everything
single-text inference needs without a calibration split at run time: the orientation
flip, the val score range, the two ternary thresholds, and an accusation-safe binary
human-vs-AI threshold.
"""
from __future__ import annotations


def build_calibration(model, tok, data, extra_human_subgroups=None, *, head: str = "seqcls",
                      n_buckets: int = 4, max_length: int = 2048,
                      binary_fpr_target: float = 0.01) -> dict:
    """Return the calibration dict for `model`, given prepared `data` (a PreparedData whose
    val/test carry `text_type`, and — for v2 — a `language` column).

    `head="corn"` decodes the K−1 ordinal logits via the cumulative-sigmoid product; `head`
    also rides in the output so inference picks the matching decode. `extra_human_subgroups`
    is an optional {name: [texts]} of hard human negatives beyond the val humans (e.g. the
    non-native-English set); the binary threshold is the MAX over every human subgroup at
    `binary_fpr_target`, so the false-accusation rate holds for the hardest one. When val has
    a `language` column its humans are grouped per language — the ja/zh-TW FPR is controlled
    the same way non-native English is, without needing a separate hard-negative set.

    Self-checks (and asserts) that the deployed val-scaled path reproduces the trained
    in-domain ternary F1 before returning, so a broken calibration can't ship silently.
    """
    import numpy as np

    from greyscope.data import PROMPT_TEMPLATE
    from greyscope.eval import (
        LABEL_TO_ID, calibrate_thresholds, evaluate, orient_scores, predict_ternary, threshold_for_fpr,
    )
    from greyscope.preprocess import clean_text
    from greyscope.scoring import score_prompts

    vs = score_prompts(model, tok, data.val["prompt"], n_buckets, head=head, max_length=max_length)
    vlab = np.asarray([LABEL_TO_ID[t] for t in data.val["text_type"]])
    v_or, flip = orient_scores(vs, vlab)
    lo, hi = float(v_or.min()), float(v_or.max())
    span = (hi - lo) or 1.0
    vscaled = np.clip((v_or - lo) / span, 0.0, 1.0)
    h_thresh, ai_thresh, _, _ = calibrate_thresholds(vlab, vscaled)

    # Per-subgroup binary threshold at the target FPR; the MAX binds (hardest human group),
    # so pooling can't let the easy in-domain majority dominate the quantile. v2 groups val
    # humans by language (ja/zh-TW join the worst-subgroup calibration); v1 has one val group.
    human_mask = vlab == 0
    val_humans = vscaled[human_mask]
    subgroup_thr: dict[str, float] = {}
    if "language" in data.val.column_names:
        hlangs = np.asarray(data.val["language"])[human_mask]
        for g in sorted(set(hlangs.tolist())):
            subgroup_thr[f"val_{g}"] = threshold_for_fpr(val_humans[hlangs == g], binary_fpr_target)
    else:
        subgroup_thr["val"] = threshold_for_fpr(val_humans, binary_fpr_target)
    for name, texts in (extra_human_subgroups or {}).items():
        prompts = [PROMPT_TEMPLATE.format(text=clean_text(str(t))) for t in texts]
        s = score_prompts(model, tok, prompts, n_buckets, head=head, max_length=max_length)
        scaled = np.clip(((-s if flip else s) - lo) / span, 0.0, 1.0)
        subgroup_thr[name] = threshold_for_fpr(scaled, binary_fpr_target)
    binding = max(subgroup_thr, key=subgroup_thr.get)
    binary_threshold = subgroup_thr[binding]

    calib = {
        "task": "editlens-ternary",
        "head_type": head,
        "n_buckets": n_buckets,
        "label_names": ["human", "AI-generated", "AI-edited"],  # index = predict_ternary id (0/1/2)
        "bucket_descriptions": ["none", "light", "moderate", "heavy"],  # edit magnitude 0->3; "heavy" = heavy edit or fully generated
        "flip": bool(flip),
        "score_min": lo,
        "score_max": hi,
        "h_thresh": float(h_thresh),
        "ai_thresh": float(ai_thresh),
        "binary_threshold": float(binary_threshold),
        "binary_fpr_target": binary_fpr_target,
        "prompt_template": PROMPT_TEMPLATE,
        "lowercase": True,
        "max_length": max_length,
        "decode": (("scalar=mean_k cumprod(sigmoid(logits))[k] over K-1 tasks; " if head == "corn"
                    else "scalar=(softmax(logits)·arange(n_buckets))/(n_buckets-1); ")
                   + "oriented=-scalar if flip; scaled=clip((oriented-score_min)/(score_max-score_min),0,1); "
                   "human if scaled<h_thresh, AI-generated if scaled>ai_thresh, else AI-edited"),
        "binary_decode": ("optional single human-vs-AI call: AI if scaled>binary_threshold else human; "
                          "binary_threshold is calibrated to binary_fpr_target false positives on the "
                          "HARDEST human subgroup (per-language val humans + any extra hard negatives) so "
                          "the false-accusation rate holds for every group — tune it for your tolerance"),
    }
    print(f"[bundle] head={head} flip={flip} range=[{lo:.4f},{hi:.4f}] h={h_thresh:.4f} ai={ai_thresh:.4f}",
          flush=True)
    print(f"[bundle] binary={binary_threshold:.4f}@fpr{binary_fpr_target} binds on '{binding}'; "
          f"per-subgroup thr: {', '.join(f'{k}={v:.3f}' for k, v in subgroup_thr.items())}", flush=True)

    # Self-check: the deployed (val-scaled) path must reproduce the trained test F1.
    ts = score_prompts(model, tok, data.test["prompt"], n_buckets, head=head, max_length=max_length)
    preds = predict_ternary(np.clip(((-ts if flip else ts) - lo) / span, 0.0, 1.0), h_thresh, ai_thresh)
    m = evaluate(np.asarray([LABEL_TO_ID[t] for t in data.test["text_type"]]), preds)
    print(f"[bundle] deployed-path test ternary macro-F1={m['macro_f1']:.4f} "
          f"h={m['f1_human']:.3f}/ai={m['f1_ai_generated']:.3f}/ed={m['f1_ai_edited']:.3f}", flush=True)
    assert m["macro_f1"] > 0.85, f"deployed-path F1 {m['macro_f1']:.4f} too low — calibration broken"
    return calib
