# Release benchmarks

This is the frozen comparison used for the v2 release. It covers Greyscope v1 and v2, the graded
[EditLens Llama model](https://huggingface.co/pangram/editlens_Llama-3.2-3B), and the binary
[MELD](https://huggingface.co/anon-review-meld-2026/meld) and
[Desklib](https://huggingface.co/desklib/ai-text-detector-v1.01) detectors. Binoculars remains in
the RAID extra diagnostic as an older zero-shot reference. The cutoff is 2026-07-24.

## APT-Eval

This uses 3,000 of 14,950 rows: all 300 human passages and a stratified sample of 2,700 polished
passages. Spearman measures whether the score follows the declared amount of polishing. Values in
brackets are 95% row-bootstrap intervals.

| Model | Spearman |
|---|---:|
| Greyscope v2 bf16 | 0.636 [0.610, 0.665] |
| Greyscope v1 bf16 | **0.645 [0.619, 0.674]** |
| EditLens Llama-3.2-3B | 0.601 [0.571, 0.628] |
| MELD | — |
| Desklib v1.01 | — |

MELD and Desklib were not evaluated on APT-Eval.

The paired Greyscope-minus-EditLens difference is +0.035 [0.012, 0.059].

## Beemo

This is a 2,997-row sample from the larger Beemo dataset: 333 source documents with all nine
variants. Intervals resample every variant of each source document together. Every variant derived
from model output is binary AI-involved; edited variants map to the AI-edited ternary class.

| Model | AUROC | TPR @ 1% FPR |
|---|---:|---:|
| Greyscope v2 bf16 | 0.819 [0.794, 0.844] | 0.063 [0.020, 0.221] |
| Greyscope v1 bf16 | **0.840 [0.817, 0.863]** | 0.042 [0.018, 0.262] |
| MELD | 0.827 [0.803, 0.847] | 0.058 [0.000, 0.333] |
| EditLens Llama-3.2-3B | 0.817 [0.792, 0.844] | 0.042 [0.024, 0.171] |
| Desklib v1.01 | 0.801 [0.777, 0.826] | **0.113 [0.005, 0.260]** |

AUROC does not use v2's bundled threshold. At that threshold, Beemo FPR is 13.2% [9.3%, 16.7%] and
TPR is 55.3% [51.3%, 59.3%]. Recalibrate the binary threshold for each deployment.

## RAID adversarial extra (diagnostic)

This uses 4,968 of 2,039,100 rows from 207 matched source documents. Each source contributes human
and AI text under clean and all 11 attack conditions. RAID `extra` covers code, Czech, and German.
One source pair was removed because Binoculars cannot score a text shorter than two Falcon tokens.

| Model | AUROC | TPR @ 1% FPR |
|---|---:|---:|
| Greyscope v2 bf16 | **0.771 [0.733, 0.804]** | 0.122 [0.085, 0.234] |
| MELD | (0.866 [0.840, 0.891]*) | (0.294 [0.253, 0.428]*) |
| Binoculars | 0.769 [0.734, 0.803] | **0.295 [0.190, 0.349]** |
| Desklib v1.01 | (0.752 [0.717, 0.785]*) | (0.229 [0.184, 0.298]*) |
| Greyscope v1 bf16 | 0.720 [0.682, 0.755] | 0.130 [0.096, 0.167] |
| EditLens Llama-3.2-3B | 0.713 [0.673, 0.751] | 0.266 [0.219, 0.316] |

Intervals resample source documents with every paired condition kept together. MELD and Desklib
trained on RAID, so their RAID scores should not be compared directly.

V2 is stronger on Czech (0.902 vs 0.742), German (0.832 vs 0.687), homoglyphs (0.790 vs 0.641), and
zero-width spaces (0.790 vs 0.468). V1 is stronger on code (0.731 vs 0.663). This does not establish
that v2 is better on English prose.

## Mac quantization

Measured on an M1 Pro with 32 GB unified memory. Quality uses a deterministic 180-row trilingual
sample.

| Build | Model size | Peak memory | 512 tokens | Macro-F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| MLX Q4 | 2.4 GB | 3.0 GB | 2.43 s | 0.842 | 0.961 |

Predictions and runtime measurements are in `results/mac_m1_pro_mlx4_*`.

## Frozen data

| Snapshot | Rows | Selection |
|---|---:|---|
| APT-Eval | 3,000 | all 300 human rows + 2,700 stratified polished rows |
| Beemo | 2,997 | 333 source documents, retaining all nine variants |
| RAID adversarial extra | 4,968 | 207 matched sources, clean + all 11 attacks |

APT does not expose reliable parent links for its polished rows, so its sample and intervals operate
at row level. Beemo keeps source-document groups intact. Manifests pin source revisions and hash
normalized labels, provenance, and text.

Raw predictions, metric reports, bootstrap intervals, and paired comparisons are under
[`results/release/`](results/release/). The comparison matrix is
[`configs/release_eval.json`](../configs/release_eval.json).

The release comparisons cost about $12 on Modal. No commercial detector API was called.

## Reproduce

Fetch each pinned public dataset with
[`scripts/fetch_release_data.py`](../scripts/fetch_release_data.py), then build the deterministic
samples for the larger datasets:

```bash
uv run python scripts/build_release_samples.py
```

Run one model and snapshot at a time. The manifest supplies the safe batch size.

```bash
MODAL_PROFILE=yaoandy107 uv run modal run modal/release_compare.py::transformers_eval \
  --benchmark beemo --snapshot-name beemo-sample --model-id greyscope-v2-bf16
```

After downloading predictions, reproduce uncertainty with:

```bash
uv run python scripts/bootstrap_release_results.py beemo-sample --unit source
uv run python scripts/compare_release_results.py beemo-sample \
  greyscope-v2-bf16 editlens-llama-3.2-3b --unit source
```

The adversarial RAID commands incur GPU cost. After preparing and scoring a paired sample with
`modal/raid_compare.py`, reproduce the grouped intervals with:

```bash
uv run python scripts/analyze_raid_adversarial.py \
  data/release/raid-adversarial-4968.csv \
  benchmarks/results/release/raid-adversarial-4968 \
  benchmarks/results/release/raid-adversarial-4968/metrics.json
```
