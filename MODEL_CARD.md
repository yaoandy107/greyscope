---
license: cc-by-nc-sa-4.0
extra_gated_fields:
  First Name: text
  Last Name: text
  Institution: text
  Country: country
  Intended use: text
  I agree to use this model for non-commercial purposes only: checkbox
language:
  - en
library_name: transformers
pipeline_tag: text-classification
base_model: unsloth/Qwen3.5-4B-Base
base_model_relation: finetune
datasets:
  - pangram/editlens_iclr
tags:
  - ai-generated-text-detection
  - ai_detection
  - text-classification
  - lora
model-index:
  - name: greyscope-qwen3.5-4b
    results:
      - task:
          type: text-classification
          name: AI-text detection (ternary)
        dataset:
          name: EditLens
          type: pangram/editlens_iclr
          split: test
        metrics:
          - type: f1
            name: Ternary macro-F1
            value: 0.924
---

# Greyscope (Qwen3.5-4B)

This model is a [`unsloth/Qwen3.5-4B-Base`](https://huggingface.co/unsloth/Qwen3.5-4B-Base) model finetuned for AI-text detection on the [EditLens dataset](https://huggingface.co/datasets/pangram/editlens_iclr). It classifies text as human-written, AI-edited, or AI-generated, and loads with plain `transformers`.

English-only for now; Traditional Chinese and Japanese are planned for v2.

Repository: [`yaoandy107/greyscope`](https://github.com/yaoandy107/greyscope)

## Model details

- The task is ternary AI-text detection (human / AI-edited / AI-generated), plus a continuous 0–1 score for the degree of AI involvement.
- The head is `AutoModelForSequenceClassification` with 4 buckets over edit magnitude; the bf16 LoRA (r=32) is merged into the base.
- The license is CC BY-NC-SA 4.0, research and non-commercial only, inherited from the dataset.
- The task and data follow the EditLens paper ([arXiv:2510.03154](https://arxiv.org/abs/2510.03154)).

The 4-bucket distribution is decoded to a 0–1 score by a weighted average; two validation-calibrated thresholds (shipped in `calibration.json`) split it into human / AI-edited / AI-generated.

## Uses

- Intended use: flagging likely AI-written or AI-edited English text, with a 0–1 score so you can set your own threshold.
- Out of scope: it is not a substitute for human judgment. Don't use it as sole evidence in high-stakes decisions like academic integrity or employment.

## How to use

Requires `transformers>=5.5.0` (Qwen3.5 architecture support).

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "yaoandy107/greyscope-qwen3.5-4b"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo, dtype=torch.bfloat16).eval()
model.config.pad_token_id = tok.pad_token_id or tok.eos_token_id
```

This loads the raw model, which outputs 4 bucket logits. The calibrated decode to a human / AI-edited / AI-generated label and 0–1 score (using `calibration.json`) is in `greyscope/inference.py`.

Weights are ~9 GB in bf16. The calibrated thresholds are tuned for bf16; re-validate them if you load another dtype or quantization.

## Evaluation

Greyscope is evaluated against the open detectors from the [OpenPangram blog](https://www.pangram.com/blog/introducing-open-pangram), on the same splits and protocol. It leads in-domain and on the unseen generator, ties editlens-Llama on Enron, and has the lowest false-positive rate on non-native English; editlens-Llama leads on RAID.

**In-domain (ternary, n=6,115)**

| Detector | Accuracy | Macro-F1 | Human F1 | AI F1 | AI-edited F1 |
|---|---|---|---|---|---|
| **Greyscope** | 0.924 | 0.924 | 0.912 | 0.977 | 0.882 |
| editlens-Llama-3.2-3B | 0.895 | 0.895 | 0.895 | 0.948 | 0.842 |
| editlens-roberta-large | 0.881 | 0.881 | 0.900 | 0.923 | 0.819 |
| Fast-DetectGPT | 0.585 | 0.545 | 0.246 | 0.831 | 0.558 |
| Binoculars | 0.569 | 0.523 | 0.213 | 0.811 | 0.545 |

**Held-out domain: Enron (ternary, n=6,147)**

| Detector | Accuracy | Macro-F1 | Human F1 | AI F1 | AI-edited F1 |
|---|---|---|---|---|---|
| **Greyscope** | 0.864 | 0.867 | 0.882 | 0.905 | 0.816 |
| editlens-Llama-3.2-3B | 0.863 | 0.868 | 0.855 | 0.936 | 0.812 |
| editlens-roberta-large | 0.695 | 0.673 | 0.847 | 0.515 | 0.657 |
| Fast-DetectGPT | 0.625 | 0.589 | 0.261 | 0.886 | 0.619 |
| Binoculars | 0.618 | 0.575 | 0.266 | 0.857 | 0.601 |

**Held-out generator: Llama-70B (ternary, n=5,957)**

| Detector | Accuracy | Macro-F1 | Human F1 | AI F1 | AI-edited F1 |
|---|---|---|---|---|---|
| **Greyscope** | 0.939 | 0.938 | 0.930 | 0.981 | 0.903 |
| editlens-Llama-3.2-3B | 0.921 | 0.920 | 0.918 | 0.965 | 0.877 |
| editlens-roberta-large | 0.860 | 0.859 | 0.908 | 0.879 | 0.791 |
| Fast-DetectGPT | 0.562 | 0.506 | 0.262 | 0.817 | 0.440 |
| Binoculars | 0.540 | 0.478 | 0.227 | 0.796 | 0.411 |

**RAID (TPR at 5% FPR, n=10,000)**

| Detector | TPR@5%FPR ↑ | AUROC ↑ |
|---|---|---|
| **Greyscope** | 0.969 | 0.991 |
| editlens-Llama-3.2-3B | 0.986 | 0.996 |
| editlens-roberta-large | 0.852 | 0.960 |
| Fast-DetectGPT | 0.961 | 0.989 |
| Binoculars | 0.964 | 0.989 |

Scored with RAID's fixed-FPR protocol (per-domain, 5% FPR) on its non-adversarial 10k sample. The OpenPangram blog reports macro-F1, but that leaves detectors at different false-positive rates, so the scores aren't comparable; a detector can rank higher just by flagging more humans.

**Human-Detectors (binary, n=300)**

| Detector | Macro-F1 | FPR ↓ | FNR ↓ |
|---|---|---|---|
| **Greyscope** | 0.983 | 0.033 | 0.000 |
| editlens-Llama-3.2-3B | 0.987 | 0.027 | 0.000 |
| editlens-roberta-large | 0.960 | 0.020 | 0.060 |
| Fast-DetectGPT | 0.735 | 0.487 | 0.013 |
| Binoculars | 0.846 | 0.087 | 0.220 |

**Non-native English (humans only, n=91)**, FPR (lower is better):

| Detector | FPR ↓ |
|---|---|
| **Greyscope** | 0.011 |
| editlens-Llama-3.2-3B | 0.055 |
| editlens-roberta-large | 0.099 |
| Fast-DetectGPT | 0.670 |
| Binoculars | 0.560 |

## Footprint

Warm forward pass on an M1 Pro (MPS, bf16, batch=1; median of 20, one-time model load excluded):

| Detector | Params | Memory | 512-token passage |
|---|---|---|---|
| **Greyscope** | 4.2B | 9.4 GB | 2.6 s |
| editlens-Llama-3.2-3B | 3.2B | 6.9 GB | 1.6 s |
| editlens-roberta-large | 0.4B | 1.0 GB | 0.2 s |

## Limitations and biases

- Research and non-commercial use only (CC BY-NC-SA 4.0).
- English only; Traditional Chinese and Japanese are planned for v2.
- Least reliable on lightly-edited and out-of-domain text.
- The default threshold favors few false accusations (~1% even on non-native English); raise it if you need more recall.

## Training

A single bf16 LoRA run on Qwen3.5-4B-Base with a 4-bucket sequence-classification head, about 4 hours on one A100-80GB. The task and data follow EditLens.

## Citation

```bibtex
@article{Thai2025EditLens,
  title   = {EditLens: Quantifying the Extent of AI Editing in Text},
  author  = {Thai, Katherine and Emi, Bradley and Masrour, Elyas and Iyyer, Mohit},
  journal = {arXiv preprint arXiv:2510.03154},
  year    = {2025}
}
```

## Acknowledgements

- [Open Pangram](https://www.pangram.com/blog/introducing-open-pangram) — the EditLens paper, open dataset, and open-source code this model learned from and builds on.
- [Modal](https://modal.com) — training ran on their free monthly compute credits.
- [Unsloth](https://unsloth.ai) — efficient LoRA fine-tuning.
