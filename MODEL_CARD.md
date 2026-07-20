---
license: apache-2.0
language:
  - en
  - ja
  - zh
library_name: transformers
pipeline_tag: text-classification
base_model: unsloth/Qwen3.5-4B-Base
base_model_relation: finetune
tags:
  - ai-generated-text-detection
  - ai_detection
  - text-classification
  - lora
model-index:
  - name: greyscope-v2-qwen3.5-4b
    results:
      - task:
          type: text-classification
          name: AI-text detection (ternary)
        dataset:
          name: Greyscope v2 trilingual test
          type: greyscope-v2
          split: test
        metrics:
          - type: f1
            name: Ternary macro-F1
            value: 0.877
---

# Greyscope v2 (Qwen3.5-4B)

Greyscope estimates how much of a text is AI-written (human / AI-edited / AI-generated, plus a continuous 0–1 `ai_involvement` score) in English, Japanese, and Traditional Chinese. It is a [`unsloth/Qwen3.5-4B-Base`](https://huggingface.co/unsloth/Qwen3.5-4B-Base) LoRA finetune that loads with plain `transformers`.

What's new over [v1](https://huggingface.co/yaoandy107/greyscope-qwen3.5-4b):

- The weights are Apache-2.0 (v1 was CC BY-NC-SA): v2 trains on our own dataset built from permissively licensed sources, keeping the EditLens dataset for evaluation only, so you can use it commercially.
- It covers Japanese and Traditional Chinese alongside English, at parity (ternary macro-F1 0.875 / 0.872 / 0.880), with a calibrated threshold per language.
- There is also an int4 build, [`yaoandy107/greyscope-v2-qwen3.5-4b-int4`](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b-int4) (HQQ, ~3 GB, 98.4% bucket agreement with bf16).

Repository: [`yaoandy107/greyscope`](https://github.com/yaoandy107/greyscope)

## Model details

- Ternary AI-text detection (human / AI-edited / AI-generated) plus a continuous 0–1 AI-involvement score.
- 4-bucket CORN ordinal head over edit magnitude on `AutoModelForSequenceClassification`; bf16 LoRA (r=32) merged into the base.
- The bucket distribution decodes to the 0–1 score; validation-calibrated thresholds (shipped in `calibration.json`) split it into labels. The binary threshold is set at ≤1% FPR on non-native-English humans, the group detectors most often misfire on; per-language thresholds are included.

## Uses

- Intended use: flagging likely AI-written or AI-edited text in en/ja/zh-TW, with a 0–1 score so you can set your own threshold.
- Out of scope: not a substitute for human judgment. Don't use it as sole evidence in high-stakes decisions like academic integrity or employment.

## How to use

Requires `transformers>=5.5.0` (Qwen3.5 architecture support).

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "yaoandy107/greyscope-v2-qwen3.5-4b"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo, dtype=torch.bfloat16).eval()
model.config.pad_token_id = tok.pad_token_id or tok.eos_token_id
```

This loads the raw model, which outputs the CORN bucket logits. The calibrated decode to a label and 0–1 score (using `calibration.json`) is in [`greyscope/inference.py`](https://github.com/yaoandy107/greyscope/blob/main/greyscope/inference.py):

```bash
python -m greyscope.inference "Paste a paragraph here." --mode ternary  # or binary
```

Weights are ~9 GB in bf16. Use bf16 (not fp16 — it overflows the GDN recurrence); the calibrated thresholds were validated for bf16 and the int4 artifact.

## Evaluation

Our trilingual test set (ternary macro-F1; every detector's thresholds calibrated on our validation split):

| Detector | All | en | ja | zh-TW |
|---|---|---|---|---|
| **Greyscope v2** | 0.877 | 0.875 | 0.872 | 0.880 |
| Greyscope v1 | 0.820 | 0.818 | 0.786 | 0.855 |
| editlens-Llama-3.2-3B | 0.717 | 0.690 | 0.679 | 0.763 |

[C-ReD](https://arxiv.org/abs/2604.11796) (Simplified-Chinese binary, 2026; per-domain AUROC, baselines from the paper, Greyscope on a 4,025-row balanced sample):

| Detector | Film | Composition | Q&A | News | Paper |
|---|---|---|---|---|---|
| **Greyscope v2** | 1.000 | 0.999 | 1.000 | 1.000 | 0.999 |
| LAPD | 0.886 | 0.953 | 0.973 | 0.941 | 0.915 |
| ReMoDetect | 0.973 | 0.873 | 0.976 | 0.865 | 0.913 |
| ImBD | 0.876 | 0.914 | 0.901 | 0.795 | 0.806 |
| Fast-DetectGPT | 0.700 | 0.895 | 0.839 | 0.763 | 0.713 |

English graded ternary (EditLens/[OpenPangram](https://www.pangram.com/blog/introducing-open-pangram) splits and protocol, macro-F1). v2 trails editlens-Llama and v1 here, which train on the much larger EditLens data v2 gives up for its license:

| Detector               | In-domain | Enron (OOD) | Llama-70B (OOD) |
|---|---|---|---|
| **Greyscope v2**       | 0.895 | 0.846 | 0.908 |
| Greyscope v1           | 0.924 | 0.867 | 0.938 |
| editlens-Llama-3.2-3B  | 0.895 | 0.868 | 0.920 |
| editlens-roberta-large | 0.881 | 0.673 | 0.859 |
| Fast-DetectGPT         | 0.545 | 0.589 | 0.506 |
| Binoculars             | 0.523 | 0.575 | 0.478 |

Binary detection ties editlens-Llama: AUROC ≥ 0.999, TPR@1%FPR ≥ 0.993 for both on all the splits above. On independent [RAID](https://raid-bench.xyz/) (10k sample): AUROC 0.995 / TPR@1%FPR 0.944 (editlens-Llama 0.996 / 0.959; closed-source Pangram v3.2 reports 0.999).

Running AI text through a paraphraser the model never saw in training doesn't hide it: on that slice, AUROC 0.998 and TPR@1%FPR 0.984.

## Limitations and biases

- Least reliable on lightly-edited and out-of-domain text.
- The default binary threshold favors few false accusations (~1% even on non-native English); threshold `ai_involvement` yourself for more recall.
- Trained on machine-labeled edit magnitudes (embedding cosine), not human annotations.

## Training

A single bf16 LoRA run on Qwen3.5-4B-Base with a 4-bucket CORN ordinal head, MELD ranking loss, and a joint language×bucket sampler; ~4.5 hours on one A100-40GB. The trilingual dataset (~62k train rows) pairs human text from permissive sources with model-generated mirrors and graded edits from 11 generator families under plain, humanizer, and persona prompts, with held-out generator and paraphraser slices. Recipe and pipeline are in the [repository](https://github.com/yaoandy107/greyscope).

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

- [Open Pangram](https://www.pangram.com/blog/introducing-open-pangram) — the EditLens paper, open dataset (used here for evaluation), and open-source code this model learned from.
- [Modal](https://modal.com) — training ran on their free monthly compute credits.
- [Unsloth](https://unsloth.ai) — efficient LoRA fine-tuning.
