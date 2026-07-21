# Greyscope

Greyscope estimates how much of a text is AI-written, from human through AI-edited to fully AI-generated, instead of a binary human/AI verdict. It works in English, Japanese, and Traditional Chinese, and the weights are Apache-2.0.

Weights:

- [`yaoandy107/greyscope-v2-qwen3.5-4b`](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b) — bf16 (~9 GB)
- [`yaoandy107/greyscope-v2-qwen3.5-4b-int4`](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b-int4) — int4 HQQ (~3 GB, 98.4% bucket agreement with bf16)
- [`yaoandy107/greyscope-qwen3.5-4b`](https://huggingface.co/yaoandy107/greyscope-qwen3.5-4b) — v1, English-only, CC BY-NC-SA

## What's new in v2

- The weights are Apache-2.0, so you can use them commercially. v1 inherited CC BY-NC-SA from the [EditLens dataset](https://huggingface.co/datasets/pangram/editlens_iclr); v2 trains on our own permissively licensed data and only uses EditLens for evaluation.
- It handles Japanese and Traditional Chinese as well as English, with a separate calibrated threshold for each language.
- There is an int4 build that fits on an 8 GB Mac.

This had a cost. v2 trained on 24k English rows against the 60k v1 and editlens-Llama use, so both still score higher on their own graded English splits. v2's data is newer, and on 2026-era text it beats both (see Results). If you only need English and non-commercial use is fine, use v1.

## Installation

```bash
uv sync
```

For training, add `--extra train`.

## Usage

Classify a passage from the command line (or pipe it on stdin):

```bash
python -m greyscope.inference \
  "Paste a paragraph here." \
  --mode ternary  # or "binary"
```

The result is a JSON object with three fields:

- `label`: in ternary mode (default), `human` / `AI-edited` / `AI-generated`. In binary mode, `human` / `AI`, using a threshold that wrongly flags at most 1% of human text.
- `ai_involvement`: a score from 0 to 1, where 0.0 means human and 1.0 means fully AI.
- `bucket_probs`: how likely each level of AI editing is. The four values sum to 1, and `heavy` includes fully generated text.

```json
{
  "label": "human",
  "ai_involvement": 0.02,
  "bucket_probs": {"none": 0.91, "light": 0.06, "moderate": 0.02, "heavy": 0.01}
}
```

The same thing is available from Python: `from greyscope.inference import detect`.

## Results

### Trilingual test set (ours)

This is our own test set. We report ternary macro-F1, with each detector's thresholds calibrated on our
validation split.

| Detector | All | en | ja | zh-TW |
|---|---|---|---|---|
| **Greyscope v2** | 0.877 | 0.875 | 0.872 | 0.880 |
| Greyscope v1 | 0.820 | 0.818 | 0.786 | 0.855 |
| editlens-Llama-3.2-3B | 0.717 | 0.690 | 0.679 | 0.763 |

### EditLens splits (English)

We report the same metric here. Enron and Llama-70B are out-of-domain (OOD) for every detector shown.

| Detector | In-domain | Enron (OOD) | Llama-70B (OOD) |
|---|---|---|---|
| **Greyscope v2** | 0.895 | 0.846 | 0.908 |
| Greyscope v1 | 0.924 | 0.867 | 0.938 |
| editlens-Llama-3.2-3B | 0.895 | 0.868 | 0.920 |
| editlens-roberta-large | 0.881 | 0.673 | 0.859 |
| Fast-DetectGPT | 0.545 | 0.589 | 0.506 |
| Binoculars | 0.523 | 0.575 | 0.478 |

On the simpler binary question of whether any AI is present, both reach AUROC ≥ 0.999 and TPR ≥ 0.993 at
1% false positives.

### RAID

[RAID](https://raid-bench.xyz/) is an independent benchmark, and we scored its labelled 10,000-row
`extra` split. The baselines are our measurements too: EditLens releases per-row scores for these
detectors, and we computed AUROC and TPR from those columns with the same harness. Neither EditLens nor
Pangram published AUROC on RAID themselves.

| Detector | AUROC | TPR@1%FPR |
|---|---|---|
| **Greyscope v2** | 0.995 | 0.944 |
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

## Footprint

We timed a warm forward pass on an M1 Pro (MPS, batch size 1), taking the median of 20 runs and excluding model load.

| Variant  | Params | Memory | 512-token passage |
| -------- | ------ | ------ | ----------------- |
| bf16     | 4.2B   | 9.4 GB | 2.6 s             |
| int4 HQQ | 4.2B   | 3.4 GB | 12.8 s            |

Use int4 to save memory, not time. torchao has no fast int4 kernel for Apple silicon yet, so on MPS it runs slower than bf16.

## Training

We trained it in a single LoRA run on Qwen3.5-4B-Base, using a 4-bucket CORN ordinal head, a MELD ranking loss, and a joint language×bucket sampler. It took about 4.5 hours on one A100-40GB, which cost around $10 on Modal. The full recipe is in `configs/train.yaml`.

We don't ship the trilingual dataset with the repo, but you can rebuild it from public sources with `greyscope/pipeline/` (the driver is `scripts/build.py`). Expect to spend around $150 on API calls. Once `data/v2/splits/` is in place, you can train on [Modal](https://modal.com) from the repo root:

```bash
modal token new                                    # one-time auth
modal secret create huggingface-token HF_TOKEN=…   # + wandb-token for W&B logging
modal run modal/train.py::smoke                     # validate the pipeline (~$0.05)
modal run --detach modal/train.py::production       # train (~4.5 h, ~$10)
modal run modal/release.py::export_and_validate     # merge LoRA → deploy-ready model
modal run modal/release.py::export_quantized        # int4-HQQ artifact
```

## Limitations

- It is least accurate on text with only light AI editing, and on text from domains it never saw.
- The default binary threshold is set to avoid false accusations (about 1%, even on non-native English). If you can accept more false positives, set your own threshold on `ai_involvement` to catch more AI text.
- Its training labels were measured by a machine (embedding cosine distance), not written by human annotators.
- It is not a replacement for human judgment. Do not use its output as the only evidence in important decisions.

## License

Code is MIT. v2 model weights are Apache-2.0. v1 weights remain CC BY-NC-SA 4.0, inherited from the EditLens dataset they trained on.

## Citation

The English graded evaluation uses the EditLens benchmark:

```bibtex
@article{Thai2025EditLens,
  title   = {EditLens: Quantifying the Extent of AI Editing in Text},
  author  = {Thai, Katherine and Emi, Bradley and Masrour, Elyas and Iyyer, Mohit},
  journal = {arXiv preprint arXiv:2510.03154},
  year    = {2025}
}
```

## Acknowledgements

- [Open Pangram](https://www.pangram.com/blog/introducing-open-pangram) — the EditLens paper, open dataset (used here for evaluation), and open-source code this project learned from.
- [Modal](https://modal.com) — training ran on their free monthly compute credits.
- [Unsloth](https://unsloth.ai) — efficient LoRA fine-tuning.
