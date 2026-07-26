---
license: apache-2.0
language: [en, ja, zh]
library_name: mlx
inference: false
base_model: yaoandy107/greyscope-v2-qwen3.5-4b
base_model_relation: quantized
tags: [ai-generated-text-detection, text-classification, mlx, int4]
---

# Greyscope v2 MLX 4-bit

This is the recommended Greyscope build for Apple Silicon. It is 2.4 GB and returns the same
continuous `ai_involvement` score and `human` / `AI-edited` / `AI-generated` labels as the
[bf16 model](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b).

## Quick start

```bash
git clone https://github.com/yaoandy107/greyscope
cd greyscope
uv sync --extra mac
uv run greyscope "Paste a paragraph here."
```

## Mac performance

Measured on an M1 Pro with 32 GB unified memory and batch size 1:

| Peak memory | Load time | 128 tokens | 512 tokens | 1,024 tokens |
|---:|---:|---:|---:|---:|
| 3.0 GB | 2.0 s | 0.66 s | 2.43 s | 5.13 s |

## Quantization check

On a balanced 180-row trilingual sample, MLX 4-bit reached 0.842 ternary macro-F1 and 0.961 binary
AUROC.

Public evaluations, calibration, training details, and limitations are documented on the
[bf16 model card](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b).
