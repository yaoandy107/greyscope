# Greyscope

Greyscope estimates how much of a text is AI-written, from human through AI-edited to fully AI-generated, instead of a binary human/AI verdict. It works in English, Japanese, and Traditional Chinese, and the weights are Apache-2.0.

Weights:

- [`yaoandy107/greyscope-v2-qwen3.5-4b`](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b) — bf16 (~9 GB)
- [`yaoandy107/greyscope-v2-qwen3.5-4b-int4`](https://huggingface.co/yaoandy107/greyscope-v2-qwen3.5-4b-int4) — int4 HQQ (~3 GB, 98.4% bucket agreement with bf16)
- [`yaoandy107/greyscope-qwen3.5-4b`](https://huggingface.co/yaoandy107/greyscope-qwen3.5-4b) — v1, English-only, CC BY-NC-SA

## What's new in v2

- Apache-2.0 weights. v1 trained on the [EditLens dataset](https://huggingface.co/datasets/pangram/editlens_iclr) and inherited its CC BY-NC-SA license; v2 trains on our own dataset built from permissively licensed sources, keeping EditLens for evaluation only, so you can use it commercially.
- Japanese and Traditional Chinese support alongside English, with a calibrated threshold per language.
- An int4 build that fits an 8 GB Mac.

The cost: dropping EditLens left v2 with 24k English training rows against the 60k v1 and editlens-Llama train on, so both still score higher on their graded English splits. In exchange, v2's dataset is built from 2026-era generators with graded edits, humanizer and persona prompts, and paraphrase augmentation, and it leads both on that text (see Results). If English-only and non-commercial works for you, use v1.

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

- `label`: in ternary mode (default), `human` / `AI-edited` / `AI-generated`; in binary mode, `human` / `AI` at a threshold capped at 1% false positives on human text
- `ai_involvement`: continuous score in [0, 1] — 0.0 = human, 1.0 = fully AI
- `bucket_probs`: per-bucket probability of the edit-intensity distribution (sums to 1; `heavy` includes fully generated)

```json
{
  "label": "human",
  "ai_involvement": 0.02,
  "bucket_probs": {"none": 0.91, "light": 0.06, "moderate": 0.02, "heavy": 0.01}
}
```

The same is available from Python as `from greyscope.inference import detect`.

## Results

Our trilingual test set (ternary macro-F1; every detector's thresholds calibrated on our validation split):

| Detector              | All   | en    | ja    | zh-TW |
| --------------------- | ----- | ----- | ----- | ----- |
| **Greyscope v2**      | 0.877 | 0.875 | 0.872 | 0.880 |
| Greyscope v1          | 0.820 | 0.818 | 0.786 | 0.855 |
| editlens-Llama-3.2-3B | 0.717 | 0.690 | 0.679 | 0.763 |

No public detection benchmark covers Japanese, so the ja column rests on this held-out split.

[C-ReD](https://arxiv.org/abs/2604.11796) (Simplified-Chinese binary, 2026; per-domain AUROC, baselines from the paper, Greyscope on a 4,025-row balanced sample):

| Detector         | Film   | Composition | Q&A    | News   | Paper  |
| ---------------- | ------ | ----------- | ------ | ------ | ------ |
| **Greyscope v2** | 1.000  | 0.999       | 1.000  | 1.000  | 0.999  |
| LAPD             | 0.886  | 0.953       | 0.973  | 0.941  | 0.915  |
| ReMoDetect       | 0.973  | 0.873       | 0.976  | 0.865  | 0.913  |
| ImBD             | 0.876  | 0.914       | 0.901  | 0.795  | 0.806  |
| Fast-DetectGPT   | 0.700  | 0.895       | 0.839  | 0.763  | 0.713  |

English graded ternary (EditLens/[OpenPangram](https://www.pangram.com/blog/introducing-open-pangram) splits and protocol, macro-F1):

| Detector               | In-domain | Enron (OOD) | Llama-70B (OOD) |
| ---------------------- | --------- | ----------- | --------------- |
| **Greyscope v2**       | 0.895     | 0.846       | 0.908           |
| Greyscope v1           | 0.924     | 0.867       | 0.938           |
| editlens-Llama-3.2-3B  | 0.895     | 0.868       | 0.920           |
| editlens-roberta-large | 0.881     | 0.673       | 0.859           |
| Fast-DetectGPT         | 0.545     | 0.589       | 0.506           |
| Binoculars             | 0.523     | 0.575       | 0.478           |

Binary detection ties editlens-Llama: AUROC ≥ 0.999, TPR@1%FPR ≥ 0.993 for both on all the splits above. On independent [RAID](https://raid-bench.xyz/) (10k sample):

| Detector               | AUROC | TPR@1%FPR |
| ---------------------- | ----- | --------- |
| **Greyscope v2**       | 0.995 | 0.944     |
| editlens-Llama-3.2-3B  | 0.996 | 0.959     |
| editlens-roberta-large | 0.960 | —         |

(Closed-source Pangram v3.2 reports 0.999.)

Running AI text through a paraphraser the model never saw in training doesn't hide it: on that slice, AUROC 0.998 and TPR@1%FPR 0.984.

The shipped binary threshold caps FPR at 1% on non-native-English humans, the group detectors most often misfire on. Per-language thresholds are in `calibration.json`.

## Footprint

Warm forward pass on an M1 Pro (MPS, batch=1; median of 20, one-time model load excluded):

| Variant  | Params | Memory | 512-token passage |
| -------- | ------ | ------ | ----------------- |
| bf16     | 4.2B   | 9.4 GB | 2.6 s             |
| int4 HQQ | 4.2B   | 3.4 GB | 12.8 s            |

int4 saves memory, not time: torchao has no Apple-silicon int4 fast path yet, so it runs slower than bf16 on MPS.

## Training

A single LoRA run on Qwen3.5-4B-Base with a 4-bucket CORN ordinal head, MELD ranking loss, and a joint language×bucket sampler; about 4.5 hours on one A100-40GB (~$10 on Modal). The full recipe is in `configs/train.yaml`.

The trilingual dataset is not distributed with the repo; `greyscope/pipeline/` (driver: `scripts/build.py`) rebuilds it from public sources, which costs roughly $150 in API calls. With `data/v2/splits/` in place, train on [Modal](https://modal.com) from the repo root:

```bash
modal token new                                    # one-time auth
modal secret create huggingface-token HF_TOKEN=…   # + wandb-token for W&B logging
modal run modal/train.py::smoke                     # validate the pipeline (~$0.05)
modal run --detach modal/train.py::production       # train (~4.5 h, ~$10)
modal run modal/release.py::export_and_validate     # merge LoRA → deploy-ready model
modal run modal/release.py::export_quantized        # int4-HQQ artifact
```

## Limitations

- Least reliable on lightly-edited and out-of-domain text.
- The default binary threshold favors few false accusations (~1% even on non-native English); raise recall by thresholding `ai_involvement` yourself if you can tolerate more false positives.
- Trained on machine-labeled edit magnitudes (embedding cosine), not human annotations.
- Not a substitute for human judgment; don't use it as sole evidence in high-stakes decisions.

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
