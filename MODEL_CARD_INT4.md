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

int4-HQQ (torchao) quantization of [`yaoandy107/greyscope-v2-qwen3.5-4b`](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b), a trilingual (en/ja/zh-TW) graded AI-text detector. ~3 GB on disk (3.4 GB resident on Apple silicon); 98.4% bucket agreement with the bf16 original on the validation set. Note: torchao's int4 kernels have no Apple-silicon fast path yet, so on MPS this runs slower than bf16 (~13 s vs ~2.6 s per 512-token pass on an M1 Pro) — pick it for memory, not speed.

Same usage, calibration (`calibration.json`), and limitations as the bf16 repo — see its [model card](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b). Code: [`yaoandy107/greyscope`](https://github.com/yaoandy107/greyscope).
