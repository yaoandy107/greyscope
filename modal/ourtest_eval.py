"""Ternary macro-F1 of external detectors on OUR trilingual test set — the reverse
direction of the README's English table (rivals on our splits carry our home-field
advantage; shown alongside the EditLens-splits table, not instead of it).

Each detector is scored as a scalar, oriented + two-threshold-calibrated on our val
(same protocol as the EditLens-splits harness), then macro-F1 on test, overall and
per language. editlens-Llama = meta-llama/Llama-3.2-3B + the PEFT adapter with its
NormedLinear 4-class head reconstructed; its scalar is the expected ordinal value.

    MODAL_PROFILE=yaoandy107 modal run modal/ourtest_eval.py::ourtest_eval --detector v2
    (likewise --detector v1 / --detector editlens; runs are independent, launch in parallel)
"""

from __future__ import annotations

from common import _VOLUMES, MERGED_DEFAULT, OUT_ROOT, _load_merged, app, hf_secret, image

_image = (image
          .add_local_file("data/v2/splits/val.csv", remote_path="/root/ourtest/val.csv")
          .add_local_file("data/v2/splits/test.csv", remote_path="/root/ourtest/test.csv"))


def _load_editlens(device: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    class NormedLinear(torch.nn.Module):
        def __init__(self, hidden: int, n: int):
            super().__init__()
            self.norm = torch.nn.LayerNorm(hidden)
            self.linear = torch.nn.Linear(hidden, n, bias=False)

        def forward(self, x):
            return self.linear(self.norm(x))

    adapter = "pangram/editlens_Llama-3.2-3B"
    tok = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForSequenceClassification.from_pretrained(
        "meta-llama/Llama-3.2-3B", num_labels=4, dtype=torch.bfloat16)
    model.score = NormedLinear(model.config.hidden_size, 4).to(torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter).eval().to(device)
    model.config.pad_token_id = tok.pad_token_id or tok.eos_token_id
    return tok, model


@app.function(
    gpu="L40S",
    timeout=4 * 3600,
    secrets=[hf_secret],
    volumes=_VOLUMES,
    image=_image,
)
def ourtest_eval(detector: str = "v2", batch_size: int = 16, max_length: int = 2048) -> None:
    import json

    import numpy as np
    import pandas as pd
    import torch

    from common import use_app_packages
    use_app_packages()
    from greyscope.eval import (
        LABEL_TO_ID, calibrate_thresholds, evaluate, minmax_scale, orient_scores, predict_ternary,
    )

    val = pd.read_csv("/root/ourtest/val.csv")
    test = pd.read_csv("/root/ourtest/test.csv")

    if detector == "editlens":
        tok, model = _load_editlens("cuda")

        def score(texts: list[str]) -> np.ndarray:
            # EditLens protocol: raw text, no prompt template; expected ordinal value
            # over its 4 edit classes (class order verified by orient_scores on val).
            outs = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                              truncation=True, max_length=max_length).to("cuda")
                    probs = model(**enc).logits.float().softmax(-1).cpu().numpy()
                    outs.append(probs @ np.array([0.0, 1 / 3, 2 / 3, 1.0]))
            return np.concatenate(outs)
    else:
        from greyscope.benchmark import greyscope_score_fn
        source = "yaoandy107/greyscope-qwen3.5-4b" if detector == "v1" else f"{OUT_ROOT}/{MERGED_DEFAULT}"
        tok, model = _load_merged(source, dtype=torch.bfloat16, device="cuda")
        score = greyscope_score_fn(model, tok, max_length=max_length)

    vs = score(val["text"].tolist())
    vlab = val["text_type"].map(LABEL_TO_ID).to_numpy()
    v_or, flipped = orient_scores(vs, vlab)
    h_thresh, ai_thresh, _, _ = calibrate_thresholds(vlab, minmax_scale(v_or))
    print(f"[ourtest] {detector}: calibrated h={h_thresh:.3f} ai={ai_thresh:.3f} flipped={flipped}")

    ts = score(test["text"].tolist())
    tlab = test["text_type"].map(LABEL_TO_ID).to_numpy()
    preds = predict_ternary(minmax_scale(-ts if flipped else ts), h_thresh, ai_thresh)

    results = {"detector": detector, "flipped": bool(flipped),
               "thresholds": {"h": float(h_thresh), "ai": float(ai_thresh)}, "splits": {}}
    for name, mask in [("all", np.ones(len(test), bool)),
                       *((lg, (test["language"] == lg).to_numpy()) for lg in sorted(test["language"].unique()))]:
        m = evaluate(tlab[mask], preds[mask])
        results["splits"][name] = {k: float(m[k]) for k in
                                   ("macro_f1", "f1_human", "f1_ai_generated", "f1_ai_edited")} | {"n": int(mask.sum())}
        print(f"[ourtest] {detector} {name:6s} macro-F1={m['macro_f1']:.4f} "
              f"(h {m['f1_human']:.3f} / ai {m['f1_ai_generated']:.3f} / edit {m['f1_ai_edited']:.3f}; n={mask.sum()})")

    with open(f"{OUT_ROOT}/ourtest_{detector}.json", "w") as fh:
        json.dump(results, fh, indent=2)
