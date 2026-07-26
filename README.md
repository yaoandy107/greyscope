# Greyscope

Greyscope estimates AI involvement in text: human, AI-edited, or AI-generated. It returns a
continuous `ai_involvement` score and supports English, Japanese, and Traditional Chinese.

## Install

```bash
uv sync
```

Add `--extra int4` to load the compressed Transformers artifact.
On Apple Silicon, use the native runtime:

```bash
uv sync --extra mac
```

## Use

```bash
uv run greyscope "Paste a paragraph here."
uv run greyscope --model int4 "Paste a paragraph here."
```

```python
from greyscope.inference import detect

result = detect("Paste a paragraph here.")
```

```json
{
  "label": "AI-edited",
  "ai_involvement": 0.46,
  "bucket_probs": {"none": 0.18, "light": 0.44, "moderate": 0.31, "heavy": 0.07}
}
```

Use `--mode binary` for a human/AI label. Treat labels as signals, not proof.

## Models

| Model | Size | Use |
|---|---:|---|
| `greyscope-v2-qwen3.5-4b` | 8.4 GB | Reference bf16 model |
| `greyscope-v2-qwen3.5-4b-mlx-4bit` | 2.4 GB | Recommended on Apple Silicon |
| `greyscope-v2-qwen3.5-4b-int4` | 3.5 GB | Lower-memory Transformers build |
| `greyscope-qwen3.5-4b` | — | Previous English-only v1 |

`greyscope` automatically uses MLX Q4 on Apple Silicon when the Mac dependencies are installed.
Elsewhere it uses bf16. Pass `--model bf16` or `--model int4` to override it.

## Evaluations

APT-Eval measures whether the score tracks AI editing amount. Results use 3,000 of 14,950 rows: all
300 human passages and a stratified sample of 2,700 polished passages.

| Model | Spearman |
|---|---:|
| **Greyscope v2** | 0.636 |
| [Greyscope v1](https://huggingface.co/yaoandy107/greyscope-qwen3.5-4b) | **0.645** |
| [EditLens Llama-3.2-3B](https://huggingface.co/pangram/editlens_Llama-3.2-3B) | 0.601 |
| [MELD](https://huggingface.co/anon-review-meld-2026/meld) | — |
| Desklib v1.01 | — |

MELD and Desklib were not evaluated on APT-Eval.

Beemo results use a 2,997-row sample from the larger dataset: 333 source documents with all nine
variants.

| Model | AUROC |
|---|---:|
| **Greyscope v2** | 0.819 |
| [Greyscope v1](https://huggingface.co/yaoandy107/greyscope-qwen3.5-4b) | **0.840** |
| [MELD](https://huggingface.co/anon-review-meld-2026/meld) | 0.827 |
| [EditLens Llama-3.2-3B](https://huggingface.co/pangram/editlens_Llama-3.2-3B) | 0.817 |
| Desklib v1.01 | 0.801 |

RAID `extra` is an out-of-domain robustness check using 4,968 rows of code, Czech, and German with
all 11 attacks.

| Model | AUROC | TPR @ 1% FPR |
|---|---:|---:|
| **Greyscope v2** | **0.771** | 0.122 |
| [MELD](https://huggingface.co/anon-review-meld-2026/meld) | (0.866*) | (0.294*) |
| [Binoculars](https://github.com/ahans30/Binoculars) | 0.769 | **0.295** |
| [Desklib v1.01](https://huggingface.co/desklib/ai-text-detector-v1.01) | (0.752*) | (0.229*) |
| [Greyscope v1](https://huggingface.co/yaoandy107/greyscope-qwen3.5-4b) | 0.720 | 0.130 |
| [EditLens Llama-3.2-3B](https://huggingface.co/pangram/editlens_Llama-3.2-3B) | 0.713 | 0.266 |

*MELD and Desklib trained on RAID.

Choose v2 for graded scores under Apache-2.0, Apple Silicon support, or Japanese and Traditional
Chinese. For English-only binary detection, compare MELD on your data. V1 remains slightly stronger
on APT-Eval and Beemo here, but its license is non-commercial.

Reproduction details and saved predictions are in [`benchmarks/`](benchmarks/). At v2's bundled
binary threshold, 13.2% of Beemo human passages were flagged. AUROC does not use that threshold.
Recalibrate it on your own data.

## Mac performance

Measured on an M1 Pro with 32 GB unified memory. Quality uses a fixed 180-row trilingual sample;
higher is better.

| Build | Model size | Peak memory | 512 tokens | Macro-F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| MLX Q4 | 2.4 GB | 3.0 GB | 2.43 s | 0.842 | 0.961 |

Raw results are under [`benchmarks/results/`](benchmarks/results/).

## Limitations

- Light AI editing is harder to detect than fully generated text.
- Accuracy can change with language, domain, text length, generator, and rewriting attacks.
- The 1% false-positive target is a calibration-set operating point, not a universal guarantee.
- Do not use Greyscope as the sole evidence for academic, employment, or disciplinary decisions.

## Training and license

V2 is a Qwen3.5-4B LoRA with a four-level CORN ordinal head and ranking loss. No EditLens data was
used for training.

Code is MIT and the v2 weights are Apache-2.0. Training inputs come from public sources with
different upstream terms; the source texts and generated training dataset are not redistributed with
the weights.

See [`configs/train.yaml`](configs/train.yaml) for the recipe and `greyscope/pipeline/` for the data
build.
