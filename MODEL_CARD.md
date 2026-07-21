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
          name: AI-text detection (binary)
        dataset:
          name: RAID
          type: liamdugan/raid
          split: extra
        metrics:
          - type: auroc
            name: AUROC
            value: 0.995
          - type: recall
            name: TPR@1%FPR
            value: 0.944
      - task:
          type: text-classification
          name: AI-text detection (binary)
        dataset:
          name: C-ReD
          type: c-red
        metrics:
          - type: auroc
            name: AUROC
            value: 0.999
      - task:
          type: text-classification
          name: AI-text detection (ternary)
        dataset:
          name: Greyscope v2 trilingual test (internal)
          type: greyscope-v2
          split: test
        metrics:
          - type: f1
            name: Ternary macro-F1
            value: 0.877
---

# Greyscope v2 (Qwen3.5-4B)

Greyscope estimates *how much* of a text is AI-written, from human through AI-edited to fully
AI-generated, as a continuous 0–1 `ai_involvement` score. It works in English, Japanese, and
Traditional Chinese, and it is a LoRA finetune of
[`unsloth/Qwen3.5-4B-Base`](https://huggingface.co/unsloth/Qwen3.5-4B-Base) that loads with plain
`transformers`.

An [int4 build](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b-int4) is also available (~3 GB,
and it agrees with the bf16 model on 98.4% of buckets). The code is at
[`yaoandy107/greyscope`](https://github.com/yaoandy107/greyscope). The older English-only
[v1](https://huggingface.co/yaoandy107/greyscope-qwen3.5-4b) is still published.

These weights are Apache-2.0; v1 was CC BY-NC-SA. That change had a cost: v2 trained on 24k English rows
against v1's 60k, so v1 still grades English editing better (see Evaluation).

## Usage

You need `transformers>=5.5.0` for the Qwen3.5 architecture.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "yaoandy107/greyscope-v2-qwen3.5-4b"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo, dtype=torch.bfloat16).eval()
model.config.pad_token_id = tok.pad_token_id or tok.eos_token_id
```

This returns raw CORN bucket logits. To turn them into a label and a score, use the calibrated decode in
[`greyscope/inference.py`](https://github.com/yaoandy107/greyscope/blob/main/greyscope/inference.py):

```bash
python -m greyscope.inference "Paste a paragraph here." --mode ternary  # or binary
```

Do not load this model in fp16. fp16 overflows the GDN recurrence and produces wrong scores. bf16 and
the int4 build are both fine.

## Calibration

`calibration.json` contains the thresholds that turn the 0–1 score into labels, including one threshold
per language. The binary threshold is set so that at most 1% of human text is wrongly flagged as AI.
We measure that 1% on text written by non-native English speakers, because detectors produce the most
false positives on this group. Lower the threshold if you need more recall and can accept more false
positives.

## Evaluation

The verdict from the benchmarks below: v2 is the best open detector for Japanese, Traditional
Chinese, text from current (2026) AI models, and AI text paraphrased to dodge detection. At telling
human from AI in English, it ties the best open detector. Its one weak spot is English text that AI
only *edited* rather than wrote: v1, trained on 2.5× more English, is still better there (0.924 vs
0.895 on [EditLens](https://arxiv.org/abs/2510.03154)'s test sets).

### Trilingual test set (ours)

This is our own test set. We report ternary macro-F1, with each detector's thresholds calibrated on our
validation split.

| Detector | All | en | ja | zh-TW |
|---|---|---|---|---|
| **Greyscope v2** | 0.877 | 0.875 | 0.872 | 0.880 |
| Greyscope v1 | 0.820 | 0.818 | 0.786 | 0.855 |
| editlens-Llama-3.2-3B | 0.717 | 0.690 | 0.679 | 0.763 |

### RAID

[RAID](https://raid-bench.xyz/) is an independent benchmark, and we scored its labelled 10,000-row
`extra` split. The baselines are our measurements too: EditLens releases per-row scores for these
detectors, and we computed AUROC and TPR from those columns with the same harness. Neither EditLens nor
Pangram published AUROC on RAID themselves.

| Detector | AUROC | TPR@1%FPR |
|---|---|---|
| **Greyscope v2** | 0.995 | 0.944 |
| Greyscope v1 | 0.991 | 0.935 |
| editlens-Llama-3.2-3B | 0.996 | 0.959 |
| editlens-roberta-large | 0.960 | not reported |
| Pangram v3.2 (closed-source) | 0.999 | not reported |

### C-ReD (Simplified Chinese)

C-ReD is a Simplified-Chinese benchmark that reports AUROC per domain. The baselines are the published
numbers from [its paper](https://arxiv.org/abs/2604.11796), measured on the full benchmark, while we
scored Greyscope on a balanced 4,025-row sample of it.

| Detector | Film | Composition | Q&A | News | Paper |
|---|---|---|---|---|---|
| **Greyscope v2** | 1.000 | 0.999 | 1.000 | 1.000 | 0.999 |
| LAPD | 0.886 | 0.953 | 0.973 | 0.941 | 0.915 |
| ReMoDetect | 0.973 | 0.873 | 0.976 | 0.865 | 0.913 |
| ImBD | 0.876 | 0.914 | 0.901 | 0.795 | 0.806 |
| Fast-DetectGPT | 0.700 | 0.895 | 0.839 | 0.763 | 0.713 |

### Paraphrased AI text

Paraphrasing is a common way to evade detection. We rewrote AI text with a paraphraser held out of
training and scored it against human text (1,793 rows across all three languages). We ran editlens-Llama
on the same rows; note it only trained on English.

| Detector | AUROC | TPR@1%FPR |
|---|---|---|
| **Greyscope v2** | 0.998 | 0.984 |
| editlens-Llama-3.2-3B | 0.939 | 0.423 |

## Limitations

- The model is meant to flag text that is likely AI-written or AI-edited. You can choose your own
  threshold on the score.
- It is least accurate on text with only light AI editing, and on text from domains it never saw.
- Its training labels were measured by a machine (embedding cosine distance), not written by human
  annotators.
- It is not a replacement for human judgment. Do not use its output as the only evidence in important
  decisions, such as academic integrity cases or hiring.

## Training

We trained it in a single bf16 LoRA run (r=32) on Qwen3.5-4B-Base, using a 4-bucket CORN ordinal head, a
MELD ranking loss, and a joint language×bucket sampler. The run took about 4.5 hours on one A100-40GB.

The trilingual dataset has about 62k training rows. Every human text comes from a permissively licensed
source, and 11 generator families rewrote each one in full and edited it at varying strengths. We held
some generators and paraphrasers out for evaluation. The recipe and pipeline are in the
[repository](https://github.com/yaoandy107/greyscope).

## Citation

English graded evaluation during development used the EditLens benchmark:

```bibtex
@article{Thai2025EditLens,
  title   = {EditLens: Quantifying the Extent of AI Editing in Text},
  author  = {Thai, Katherine and Emi, Bradley and Masrour, Elyas and Iyyer, Mohit},
  journal = {arXiv preprint arXiv:2510.03154},
  year    = {2025}
}
```

## Acknowledgements

- [Open Pangram](https://www.pangram.com/blog/introducing-open-pangram) — the EditLens paper, dataset (used here for evaluation), and open-source code.
- [Modal](https://modal.com) — training ran on their free monthly compute credits.
- [Unsloth](https://unsloth.ai) — efficient LoRA fine-tuning.
