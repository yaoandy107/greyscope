---
license: apache-2.0
language: [en, ja, zh]
library_name: transformers
inference: false
base_model: yaoandy107/greyscope-v2-qwen3.5-4b
base_model_relation: quantized
tags: [ai-generated-text-detection, text-classification, int4, torchao]
---

# Greyscope v2 int4

This is the 3.5 GB Transformers int4-HQQ build of Greyscope v2. It returns the same continuous
`ai_involvement` score and `human` / `AI-edited` / `AI-generated` labels as the
[bf16 model](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b).

Use this build when you need Transformers but cannot fit bf16. Do not use it on Apple Silicon; use
[MLX 4-bit](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b-mlx-4bit).

## Quick start

```bash
git clone https://github.com/yaoandy107/greyscope
cd greyscope
uv sync --extra int4
uv run greyscope --model int4 "Paste a paragraph here."
```

## Quantization check

The release export checked this artifact against bf16 on 64 rows. Treat this as a smoke test, not a
quality benchmark, and validate it on your own data and hardware.

Evaluations, calibration, training details, and limitations are documented on the
[bf16 model card](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b).
