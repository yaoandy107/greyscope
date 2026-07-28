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

## Which build should I use?

| Build | Size | Use |
|---|---:|---|
| `greyscope-v2-qwen3.5-4b` | 8.4 GB | NVIDIA GPU or CPU |
| `greyscope-v2-qwen3.5-4b-mlx-4bit` | 2.4 GB | Apple Silicon |
| `greyscope-v2-qwen3.5-4b-int4` | 3.5 GB | NVIDIA GPU when bf16 does not fit |

`greyscope` automatically uses MLX Q4 on Apple Silicon when the Mac dependencies are installed.
Elsewhere it uses bf16. On NVIDIA, pass `--model int4` when memory is limited.

## Evaluations

### Graded AI involvement

#### APT-Eval

APT-Eval measures whether the score tracks AI editing amount. Results use 3,000 of 14,950 rows: all
300 human passages and a stratified sample of 2,700 polished passages. Spearman: higher is better.

| Model | Spearman |
|---|---:|
| **Greyscope v2** | 0.636 |
| [Greyscope v1 (previous release)](https://huggingface.co/yaoandy107/greyscope-qwen3.5-4b) | **0.645** |
| [EditLens Llama-3.2-3B](https://huggingface.co/pangram/editlens_Llama-3.2-3B) | 0.601 |

### Binary AI detection

These evaluations compare human and AI text, so binary-only detectors can be included.

#### Beemo

Beemo results use a 2,997-row sample from the larger dataset: 333 source documents with all nine
variants. AUROC: higher is better.

| Model | AUROC |
|---|---:|
| **Greyscope v2** | 0.819 |
| [Greyscope v1 (previous release)](https://huggingface.co/yaoandy107/greyscope-qwen3.5-4b) | **0.840** |
| [MELD](https://huggingface.co/anon-review-meld-2026/meld/tree/3383dc2f02abacb45a7dd28568e82e0836ec740e) | 0.827 |
| [EditLens Llama-3.2-3B](https://huggingface.co/pangram/editlens_Llama-3.2-3B) | 0.817 |
| [Desklib v1.01](https://huggingface.co/desklib/ai-text-detector-v1.01) | 0.801 |

#### RAID extra

RAID `extra` is an out-of-domain robustness check using 4,968 rows of code, Czech, and German with
all 11 attacks. Higher is better for both metrics. MELD and Desklib are excluded because they
trained on RAID.

| Model | AUROC | TPR @ 1% FPR |
|---|---:|---:|
| **Greyscope v2** | **0.771** | 0.122 |
| [Binoculars](https://github.com/ahans30/Binoculars) | 0.769 | **0.295** |
| [Greyscope v1 (previous release)](https://huggingface.co/yaoandy107/greyscope-qwen3.5-4b) | 0.720 | 0.130 |
| [EditLens Llama-3.2-3B](https://huggingface.co/pangram/editlens_Llama-3.2-3B) | 0.713 | 0.266 |

Benchmark metadata is in [`benchmarks/`](benchmarks/).

## Mac performance

Measured on an M1 Pro with 32 GB unified memory.

| Build | Model size | Peak memory | 512 tokens |
|---|---:|---:|---:|
| MLX Q4 | 2.4 GB | 3.0 GB | 2.43 s |
| Transformers bf16 | 8.4 GB | 9.4 GB | 2.54 s |

MLX Q4 matched bf16 quality across 3,007 external benchmark rows. See the
[`release metrics`](benchmarks/results/v2-summary.json) for details.

## Limitations

- Light AI editing is harder to detect than fully generated text.
- Accuracy can change with language, domain, text length, generator, and rewriting attacks.
- The 1% false-positive target is a calibration-set operating point, not a universal guarantee.
- Do not use Greyscope as the sole evidence for academic, employment, or disciplinary decisions.

## Training and license

Greyscope v2 is a Qwen3.5-4B LoRA with a four-level CORN ordinal head and ranking loss. No EditLens
data was used for training.

Code is MIT and the Greyscope v2 weights are Apache-2.0. Training inputs come from public sources
with different upstream terms; the source texts and generated training dataset are not redistributed
with the weights.

See [`configs/train.yaml`](configs/train.yaml) for the recipe and `greyscope/pipeline/` for the data
build.
