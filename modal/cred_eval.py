"""Score the C-ReD (Simplified-Chinese, arXiv:2604.11796) binary benchmark — the README's
external zh number. Doubly OOD: the model trains on Traditional Chinese only, and C-ReD's
generators (qwen/deepseek/doubao/...) are excluded from the training registry.

Build the sample first (clone github.com/HeraldofLight/C-ReD), then run:

    python - <<'PY'   # /tmp/cred_sample.csv: per domain, 400 human + 45 rows/generator, seed 42
    import csv, sys, random, os
    csv.field_size_limit(sys.maxsize); random.seed(42)
    root = "path/to/C-ReD/benchmark data"; rows = []
    for dom in os.listdir(root):
        d = os.path.join(root, dom)
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            gen = f.replace(".csv", "").split("_")[-1]
            rs = [r for r in csv.DictReader(open(os.path.join(d, f))) if r.get("text", "").strip()]
            rows += [{"text": r["text"], "label": int(gen != "human"), "domain": dom, "generator": gen}
                     for r in random.sample(rs, min(400 if gen == "human" else 45, len(rs)))]
    random.shuffle(rows)
    w = csv.DictWriter(open("/tmp/cred_sample.csv", "w"), fieldnames=["text", "label", "domain", "generator"])
    w.writeheader(); w.writerows(rows)
    PY
    MODAL_PROFILE=yaoandy107 modal run modal/cred_eval.py::cred_eval
"""

from __future__ import annotations

from common import _VOLUMES, MERGED_DEFAULT, OUT_ROOT, _load_merged, app, hf_secret, image


@app.function(
    gpu="L4",
    timeout=2 * 3600,
    secrets=[hf_secret],
    volumes=_VOLUMES,
    image=image.add_local_file("/tmp/cred_sample.csv", remote_path="/root/cred_sample.csv"),
)
def cred_eval(merged: str = MERGED_DEFAULT, batch_size: int = 16) -> None:
    import csv
    import json
    import sys
    from collections import defaultdict

    import numpy as np
    import torch

    from common import use_app_packages
    use_app_packages()
    from greyscope.corn import corn_cumulative_probs
    from greyscope.preprocess import clean_text
    from greyscope.data import PROMPT_TEMPLATE
    from greyscope.scoring import batch_logits

    csv.field_size_limit(sys.maxsize)
    with open("/root/cred_sample.csv") as fh:
        rows = list(csv.DictReader(fh))
    print(f"[cred] {len(rows)} rows")

    tok, model = _load_merged(f"{OUT_ROOT}/{merged}", dtype=torch.bfloat16, device="cuda")
    prompts = [PROMPT_TEMPLATE.format(text=clean_text(r["text"])) for r in rows]
    logits = batch_logits(model, tok, prompts, max_length=2048, batch_size=batch_size)
    ai_prob = corn_cumulative_probs(logits)[:, 0]  # P(y > 0)
    y = np.array([int(r["label"]) for r in rows])

    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(y, ai_prob)

    def tpr_at_fpr(fpr):
        thr = np.quantile(ai_prob[y == 0], 1 - fpr)
        return float((ai_prob[y == 1] >= thr).mean())

    print(f"[cred] AUROC={auroc:.4f}  TPR@1%FPR={tpr_at_fpr(0.01):.4f}  TPR@5%FPR={tpr_at_fpr(0.05):.4f}")

    by = defaultdict(lambda: ([], []))
    for r, p, t in zip(rows, ai_prob, y):
        for key in (f"dom:{r['domain']}", f"gen:{r['generator']}"):
            by[key][0].append(p), by[key][1].append(t)
    for key in sorted(by):
        p, t = np.array(by[key][0]), np.array(by[key][1])
        if len(set(t)) == 2:
            print(f"[cred]   {key:24s} AUROC={roc_auc_score(t, p):.4f} (n={len(t)})")
        else:
            # single-class stratum (per-generator): report mean score instead
            print(f"[cred]   {key:24s} mean_ai_prob={p.mean():.3f} (n={len(t)})")

    with open(f"{OUT_ROOT}/cred_eval.json", "w") as fh:
        json.dump({"auroc": float(auroc), "tpr_fpr1": tpr_at_fpr(0.01),
                   "tpr_fpr5": tpr_at_fpr(0.05), "n": len(rows)}, fh, indent=2)
