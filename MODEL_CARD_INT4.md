---
license: apache-2.0
language:
  - en
  - ja
  - zh
library_name: transformers
pipeline_tag: text-classification
base_model: yaoandy107/greyscope-v2-qwen3.5-4b
base_model_relation: quantized
tags:
  - ai-generated-text-detection
  - text-classification
  - int4
  - torchao
---

# Greyscope v2 (Qwen3.5-4B) — int4

This is an int4-HQQ (torchao) quantization of
[`yaoandy107/greyscope-v2-qwen3.5-4b`](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b), a
graded AI-text detector for English, Japanese, and Traditional Chinese. It is about 3 GB on disk, and
uses 3.4 GB of memory on Apple silicon. On the validation set it gives the same bucket as the bf16
model 98.4% of the time.

Use it to save memory, not time. torchao has no fast int4 kernel for Apple silicon yet, so on MPS this
model is slower than bf16: about 13 s per 512-token pass on an M1 Pro, against 2.6 s for bf16.

Usage, calibration (`calibration.json`), and limitations are the same as the bf16 repo. See its
[model card](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b) for details. The code is at
[`yaoandy107/greyscope`](https://github.com/yaoandy107/greyscope).
