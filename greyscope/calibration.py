"""Build the deploy-ready calibration (calibration.json) from a trained model.

Derives, from the in-domain val split plus a non-native-English human set, everything
single-text inference needs without a calibration split at run time: the orientation
flip, the val score range, the two ternary thresholds, and an accusation-safe binary
human-vs-AI threshold.
"""
from __future__ import annotations


def build_calibration(model, tok, nn_texts, *, n_buckets: int = 4, max_length: int = 2048,
                      binary_fpr_target: float = 0.01) -> dict:
    """Return the calibration dict for `model`. `nn_texts` = non-native-English human texts.

    Self-checks (and asserts) that the deployed val-scaled path reproduces the trained
    in-domain ternary F1 before returning, so a broken calibration can't ship silently.
    """
    import numpy as np

    from greyscope.config import DataConfig
    from greyscope.data import PROMPT_TEMPLATE, prepare_data
    from greyscope.eval import (
        LABEL_TO_ID, calibrate_thresholds, evaluate, orient_scores, predict_ternary, threshold_for_fpr,
    )
    from greyscope.preprocess import clean_text
    from greyscope.scoring import score_prompts

    data = prepare_data(DataConfig(n_buckets=n_buckets, train_subset=100))  # full val + test

    vs = score_prompts(model, tok, data.val["prompt"], n_buckets, max_length=max_length)
    vlab = np.asarray([LABEL_TO_ID[t] for t in data.val["text_type"]])
    v_or, flip = orient_scores(vs, vlab)
    lo, hi = float(v_or.min()), float(v_or.max())
    span = (hi - lo) or 1.0
    vscaled = np.clip((v_or - lo) / span, 0.0, 1.0)
    h_thresh, ai_thresh, _, _ = calibrate_thresholds(vlab, vscaled)

    # Take the max threshold across human subgroups so the FPR target holds for the
    # hardest one; pooling would let the easy in-domain majority dominate the quantile.
    nn_prompts = [PROMPT_TEMPLATE.format(text=clean_text(str(t))) for t in nn_texts]
    nn_s = score_prompts(model, tok, nn_prompts, n_buckets, max_length=max_length)
    nn_scaled = np.clip(((-nn_s if flip else nn_s) - lo) / span, 0.0, 1.0)
    binary_threshold = max(
        threshold_for_fpr(vscaled[vlab == 0], binary_fpr_target),
        threshold_for_fpr(nn_scaled, binary_fpr_target),
    )
    nn_fpr = float((nn_scaled > binary_threshold).mean())

    calib = {
        "task": "editlens-ternary",
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
        "decode": ("scalar=(softmax(logits)·arange(n_buckets))/(n_buckets-1); "
                   "oriented=-scalar if flip; scaled=clip((oriented-score_min)/(score_max-score_min),0,1); "
                   "human if scaled<h_thresh, AI-generated if scaled>ai_thresh, else AI-edited"),
        "binary_decode": ("optional single human-vs-AI call: AI if scaled>binary_threshold else human; "
                          "binary_threshold is calibrated to binary_fpr_target false positives on a DIVERSE "
                          "human set (in-domain val + non-native English) so the false-accusation rate holds "
                          "for the hardest human group — tune it for your tolerated false-accusation rate"),
    }
    print(f"[bundle] flip={flip} range=[{lo:.4f},{hi:.4f}] h={h_thresh:.4f} ai={ai_thresh:.4f} "
          f"binary={binary_threshold:.4f}@fpr{binary_fpr_target} (non-native FPR={nn_fpr:.4f})", flush=True)

    # Self-check: the deployed (val-scaled) path must reproduce the trained test F1.
    ts = score_prompts(model, tok, data.test["prompt"], n_buckets, max_length=max_length)
    preds = predict_ternary(np.clip(((-ts if flip else ts) - lo) / span, 0.0, 1.0), h_thresh, ai_thresh)
    m = evaluate(np.asarray([LABEL_TO_ID[t] for t in data.test["text_type"]]), preds)
    print(f"[bundle] deployed-path test ternary macro-F1={m['macro_f1']:.4f} "
          f"h={m['f1_human']:.3f}/ai={m['f1_ai_generated']:.3f}/ed={m['f1_ai_edited']:.3f}", flush=True)
    assert m["macro_f1"] > 0.85, f"deployed-path F1 {m['macro_f1']:.4f} too low — calibration broken"
    return calib
