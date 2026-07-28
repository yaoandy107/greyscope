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

| Build | Peak memory | Load time | 128 tokens | 512 tokens | 1,024 tokens |
|---|---:|---:|---:|---:|---:|
| MLX 4-bit | 3.0 GB | 2.0 s | 0.66 s | 2.43 s | 5.13 s |
| Transformers bf16 | 9.4 GB | 16.4 s | 0.74 s | 2.54 s | 5.01 s |

## Quantization check

MLX 4-bit matched bf16 quality across 3,007 external benchmark rows. See the
[release metrics](https://github.com/yaoandy107/greyscope/blob/main/benchmarks/results/v2-summary.json)
for details.

Public evaluations, calibration, training details, and limitations are documented on the
[bf16 model card](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b).
