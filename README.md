# Greyscope

Greyscope estimates how much of a text is AI-written, from human through AI-edited to fully AI-generated, instead of a binary human/AI verdict. It's a Qwen3.5-4B LoRA model trained on the open [EditLens dataset](https://huggingface.co/datasets/pangram/editlens_iclr).

English-only for now; Traditional Chinese and Japanese are next ([Roadmap](#roadmap)).

Weights: [`yaoandy107/greyscope-qwen3.5-4b`](https://huggingface.co/yaoandy107/greyscope-qwen3.5-4b)

## Installation

```bash
uv sync
```

Training needs a CUDA GPU: `uv sync --extra train`.

## Usage

Classify a passage from the command line (or pipe it on stdin):

```bash
python -m greyscope.inference \
  "Paste a paragraph here." \
  --mode ternary  # or "binary"
```

The result is a JSON object with three fields:

- `label`: in ternary mode (default), `human` / `AI-edited` / `AI-generated`; in binary mode, `human` / `AI` at a threshold tuned for ≤1% false accusations (even on non-native English)
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

Greyscope is evaluated against the open detectors from the [OpenPangram blog](https://www.pangram.com/blog/introducing-open-pangram), on the same splits and protocol. It leads in-domain and on the unseen generator, ties editlens-Llama on Enron, and has the lowest false-positive rate on non-native English; editlens-Llama leads on RAID.

**In-domain (ternary, n=6,115)**

| Detector               | Accuracy | Macro-F1 | Human F1 | AI F1 | AI-edited F1 |
| ---------------------- | -------- | -------- | -------- | ----- | ------------ |
| **Greyscope**          | 0.924    | 0.924    | 0.912    | 0.977 | 0.882        |
| editlens-Llama-3.2-3B  | 0.895    | 0.895    | 0.895    | 0.948 | 0.842        |
| editlens-roberta-large | 0.881    | 0.881    | 0.900    | 0.923 | 0.819        |
| Fast-DetectGPT         | 0.585    | 0.545    | 0.246    | 0.831 | 0.558        |
| Binoculars             | 0.569    | 0.523    | 0.213    | 0.811 | 0.545        |

**Held-out domain: Enron (ternary, n=6,147)**

| Detector               | Accuracy | Macro-F1 | Human F1 | AI F1 | AI-edited F1 |
| ---------------------- | -------- | -------- | -------- | ----- | ------------ |
| **Greyscope**          | 0.864    | 0.867    | 0.882    | 0.905 | 0.816        |
| editlens-Llama-3.2-3B  | 0.863    | 0.868    | 0.855    | 0.936 | 0.812        |
| editlens-roberta-large | 0.695    | 0.673    | 0.847    | 0.515 | 0.657        |
| Fast-DetectGPT         | 0.625    | 0.589    | 0.261    | 0.886 | 0.619        |
| Binoculars             | 0.618    | 0.575    | 0.266    | 0.857 | 0.601        |

**Held-out generator: Llama-70B (ternary, n=5,957)**

| Detector               | Accuracy | Macro-F1 | Human F1 | AI F1 | AI-edited F1 |
| ---------------------- | -------- | -------- | -------- | ----- | ------------ |
| **Greyscope**          | 0.939    | 0.938    | 0.930    | 0.981 | 0.903        |
| editlens-Llama-3.2-3B  | 0.921    | 0.920    | 0.918    | 0.965 | 0.877        |
| editlens-roberta-large | 0.860    | 0.859    | 0.908    | 0.879 | 0.791        |
| Fast-DetectGPT         | 0.562    | 0.506    | 0.262    | 0.817 | 0.440        |
| Binoculars             | 0.540    | 0.478    | 0.227    | 0.796 | 0.411        |

**RAID (binary, n=10,000)**

| Detector               | Macro-F1 | FPR ↓ | FNR ↓ |
| ---------------------- | -------- | ----- | ----- |
| **Greyscope**          | 0.888    | 0.003 | 0.105 |
| editlens-Llama-3.2-3B  | 0.930    | 0.003 | 0.062 |
| editlens-roberta-large | 0.736    | 0.007 | 0.288 |
| Fast-DetectGPT         | 0.941    | 0.078 | 0.028 |
| Binoculars             | 0.939    | 0.100 | 0.024 |

editlens-Llama generalizes better here at the same 0.3% false-positive rate. Fast-DetectGPT and Binoculars score higher only by flagging far more humans (8–10% false positives).

**Human-Detectors (binary, n=300)**

| Detector               | Macro-F1 | FPR ↓ | FNR ↓ |
| ---------------------- | -------- | ----- | ----- |
| **Greyscope**          | 0.983    | 0.033 | 0.000 |
| editlens-Llama-3.2-3B  | 0.987    | 0.027 | 0.000 |
| editlens-roberta-large | 0.960    | 0.020 | 0.060 |
| Fast-DetectGPT         | 0.735    | 0.487 | 0.013 |
| Binoculars             | 0.846    | 0.087 | 0.220 |

**Non-native English (humans only, n=91)**, FPR (lower is better):

| Detector               | FPR ↓ |
| ---------------------- | ----- |
| **Greyscope**          | 0.011 |
| editlens-Llama-3.2-3B  | 0.055 |
| editlens-roberta-large | 0.099 |
| Fast-DetectGPT         | 0.670 |
| Binoculars             | 0.560 |

## Footprint

Warm forward pass on an M1 Pro (MPS, bf16, batch=1; median of 20, one-time model load excluded):

| Detector               | Params | Memory | 512-token passage |
| ---------------------- | ------ | ------ | ----------------- |
| **Greyscope**          | 4.2B   | 9.4 GB | 2.6 s             |
| editlens-Llama-3.2-3B  | 3.2B   | 6.9 GB | 1.6 s             |
| editlens-roberta-large | 0.4B   | 1.0 GB | 0.2 s             |

## Training

A single LoRA run on Qwen3.5-4B-Base with a 4-bucket sequence-classification head, about 4 hours on one A100-80GB (~$10 on Modal). The task and data follow EditLens; the full recipe is in `configs/train.yaml`.

Reproduce on [Modal](https://modal.com) (free monthly credits cover it), from the repo root:

```bash
modal token new                                    # one-time auth
modal secret create huggingface-token HF_TOKEN=…   # + wandb-token for W&B logging
modal run modal/train.py::smoke                     # validate the pipeline (~$0.05)
modal run --detach modal/train.py::production       # train (~4 h, ~$10)
modal run modal/release.py::export_and_validate     # merge LoRA → deploy-ready model
```

## Limitations

- Research and non-commercial use only (CC BY-NC-SA 4.0, inherited from the EditLens dataset).
- English-only for now; multilingual is v2 (see the [Roadmap](#roadmap)).
- Least reliable on lightly-edited and out-of-domain text.
- The default threshold favors few false accusations (~1% even on non-native English); raise it if you need more recall.
- Not a substitute for human judgment; don't use it as sole evidence in high-stakes decisions.

## Roadmap

- [x] English (v1)
- [ ] Traditional Chinese and Japanese (v2)
- [ ] Training data expanded with newer generator models (v2)

## License

Code is MIT. Model weights and any redistributed data are CC BY-NC-SA 4.0 (non-commercial), inherited from the EditLens dataset. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{Thai2025EditLens,
  title   = {EditLens: Quantifying the Extent of AI Editing in Text},
  author  = {Thai, Katherine and Emi, Bradley and Masrour, Elyas and Iyyer, Mohit},
  journal = {arXiv preprint arXiv:2510.03154},
  year    = {2025}
}
```

## Acknowledgements

- [Open Pangram](https://www.pangram.com/blog/introducing-open-pangram) — the EditLens paper, open dataset, and open-source code this project learned from and builds on.
- [Modal](https://modal.com) — training ran on their free monthly compute credits.
- [Unsloth](https://unsloth.ai) — efficient LoRA fine-tuning.
