"""Score editlens-Llama on our held-out-paraphraser attack slice, so the paraphrase
robustness number is a comparison and not a self-report.

    MODAL_PROFILE=yaoandy107 modal run modal/paraphrase_baseline.py::paraphrase_baseline
"""

from __future__ import annotations

from common import _VOLUMES, OUT_ROOT, app, hf_secret, image
from ourtest_eval import _load_editlens

_image = (image
          .add_local_file("modal/ourtest_eval.py", remote_path="/root/ourtest_eval.py")
          .add_local_file("data/v2/splits/attack_paraphrase_test.csv",
                          remote_path="/root/attack_paraphrase_test.csv"))


@app.function(gpu="L40S", timeout=2 * 3600, secrets=[hf_secret], volumes=_VOLUMES, image=_image)
def paraphrase_baseline(batch_size: int = 16, max_length: int = 2048) -> None:
    import json

    import numpy as np
    import pandas as pd
    import torch

    from common import use_app_packages
    use_app_packages()
    from greyscope.eval import roc_auc, tpr_at_fpr

    df = pd.read_csv("/root/attack_paraphrase_test.csv")
    tok, model = _load_editlens("cuda")

    scores = []
    with torch.no_grad():
        texts = df["text"].astype(str).tolist()
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to("cuda")
            probs = model(**enc).logits.float().softmax(-1).cpu().numpy()
            scores.append(probs @ np.array([0.0, 1 / 3, 2 / 3, 1.0]))
    s = np.concatenate(scores)
    y = (df["text_type"] != "human_written").astype(int).to_numpy()

    out = {"n": len(df), "auroc": roc_auc(y, s),
           "tpr_fpr1": tpr_at_fpr(y, s, 0.01), "tpr_fpr5": tpr_at_fpr(y, s, 0.05)}
    print(f"[paraphrase] editlens-Llama AUROC={out['auroc']:.4f} "
          f"TPR@1%={out['tpr_fpr1']:.4f} TPR@5%={out['tpr_fpr5']:.4f} (n={out['n']})")
    with open(f"{OUT_ROOT}/paraphrase_baseline_editlens.json", "w") as fh:
        json.dump(out, fh, indent=2)
