---
license: apache-2.0
language:
  - en
  - ja
  - zh
library_name: transformers
inference: false
base_model: unsloth/Qwen3.5-4B-Base
base_model_relation: finetune
tags:
  - ai-generated-text-detection
  - text-classification
  - lora
---

# Greyscope v2

Greyscope estimates how much AI was involved in a passage. It returns a score from 0 to 1 and one
of three labels: `human`, `AI-edited`, or `AI-generated`. It supports English, Japanese, and
Traditional Chinese.

This is the reference bf16 model. Other builds use the same labels and calibration:

| Artifact | Size | Best for |
|---|---:|---|
| [bf16](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b) | 8.4 GB | CUDA, or CPU when speed is not important |
| [MLX Q4](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b-mlx-4bit) | 2.4 GB | Apple Silicon |
| [Transformers int4](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b-int4) | 3.5 GB | NVIDIA GPU when bf16 does not fit |

## Quick start

The [Greyscope repository](https://github.com/yaoandy107/greyscope) contains the calibrated decoder:

```bash
git clone https://github.com/yaoandy107/greyscope
cd greyscope
uv sync
uv run greyscope "Paste a paragraph here."
```

From Python:

```python
from greyscope.inference import detect

result = detect("Paste a paragraph here.")
```

Example output:

```json
{
  "label": "AI-edited",
  "ai_involvement": 0.46,
  "bucket_probs": {"none": 0.18, "light": 0.44, "moderate": 0.31, "heavy": 0.07}
}
```

`ai_involvement` describes the estimated degree of AI involvement; it is not a probability that the
author cheated. `bucket_probs` are the model's probabilities for its four training levels. Use
`--mode binary` if you need a `human` / `AI` label.

Use Greyscope's calibrated decoder rather than a stock Transformers classification pipeline. The
three model logits are CORN conditional logits, not class probabilities.

## Mac performance

Measured on an M1 Pro with 32 GB unified memory.

| Build | Model size | Peak memory | 512 tokens |
|---|---:|---:|---:|
| MLX Q4 | 2.4 GB | 3.0 GB | 2.43 s |
| Transformers bf16 | 8.4 GB | 9.4 GB | 2.54 s |

MLX Q4 matched bf16 quality across 3,007 external benchmark rows. See the
[release metrics](https://github.com/yaoandy107/greyscope/blob/main/benchmarks/results/v2-summary.json)
for details.

## Evaluations

### Graded AI involvement

#### APT-Eval

Uses 3,000 of 14,950 rows: all 300 human passages and a stratified sample of 2,700 polished
passages. Spearman measures whether the score tracks editing amount; higher is better.

| Model | Spearman |
|---|---:|
| **Greyscope v2** | 0.636 |
| [Greyscope v1 (previous release)](https://huggingface.co/yaoandy107/greyscope-qwen3.5-4b) | **0.645** |
| [EditLens Llama-3.2-3B](https://huggingface.co/pangram/editlens_Llama-3.2-3B) | 0.601 |

### Binary AI detection

These evaluations compare human and AI text, so binary-only detectors can be included.

#### Beemo

Uses a 2,997-row sample from the larger Beemo dataset: 333 source documents with all nine variants.
AUROC tests generated and edited text; higher is better.

| Model | AUROC |
|---|---:|
| **Greyscope v2** | 0.819 |
| [Greyscope v1 (previous release)](https://huggingface.co/yaoandy107/greyscope-qwen3.5-4b) | **0.840** |
| [MELD](https://huggingface.co/anon-review-meld-2026/meld/tree/3383dc2f02abacb45a7dd28568e82e0836ec740e) | 0.827 |
| [EditLens Llama-3.2-3B](https://huggingface.co/pangram/editlens_Llama-3.2-3B) | 0.817 |
| [Desklib v1.01](https://huggingface.co/desklib/ai-text-detector-v1.01) | 0.801 |

#### RAID extra

Uses 4,968 rows from RAID `extra`: code, Czech, and German with all 11 attacks. This is an
out-of-domain robustness check, not a supported-language benchmark. Higher is better for both
metrics. MELD and Desklib are excluded because they trained on RAID.

| Model | AUROC | TPR @ 1% FPR |
|---|---:|---:|
| **Greyscope v2** | **0.771** | 0.122 |
| [Binoculars](https://github.com/ahans30/Binoculars) | 0.769 | **0.295** |
| [Greyscope v1 (previous release)](https://huggingface.co/yaoandy107/greyscope-qwen3.5-4b) | 0.720 | 0.130 |
| [EditLens Llama-3.2-3B](https://huggingface.co/pangram/editlens_Llama-3.2-3B) | 0.713 | 0.266 |

Benchmark metadata is in
[`benchmarks/`](https://github.com/yaoandy107/greyscope/tree/main/benchmarks).

## Limitations

- Light AI editing is harder to detect than fully generated text.
- Results change with text length, subject, generator, language, and rewriting method.
- The model reads at most 2,048 tokens from each passage.
- Chinese training data is Traditional Chinese; do not assume the same quality on Simplified Chinese.
- Training edit-strength labels came from embedding distance rather than human annotation.
- Do not use this model as the sole evidence in academic, employment, or disciplinary decisions.

## Training and license

Greyscope v2 is a Qwen3.5-4B LoRA with a four-level CORN ordinal head and a ranking loss. It was
trained on English, Japanese, and Traditional-Chinese text. No EditLens data was used for training.

The weights are Apache-2.0 and the code is MIT. Source texts and the generated training dataset are
not redistributed with the weights. The complete recipe is in the
[repository](https://github.com/yaoandy107/greyscope).
