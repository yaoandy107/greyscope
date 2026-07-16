# Greyscope

A detector that estimates *how much* of a text is AI-written (human → AI-edited → AI-generated) as a
continuous `ai_involvement` score, not a binary verdict. Qwen3.5-4B LoRA, 4-bucket CORN ordinal head.
The **shipped** model is English only (CC BY-NC-SA, git tag `v1.0`); this branch builds its successor —
a **trilingual (EN/ja/zh-TW), Apache-2.0** model. The code is de-versioned (one recipe, one set of
entrypoints — the previous model is recoverable from `v1.0`); only the dataset artifacts are versioned,
under `data/v2/`.

## Layout

- `greyscope/` — the model code: `inference.py`, `model.py`/`trainer.py`/`collator.py`/`corn.py`,
  `eval.py`/`benchmark.py`/`calibration.py`, `export.py`, `preprocess.py` (EditLens-ported normalization
  + Unicode hardening + `score_to_bucket`). Training/release run on Modal (`modal/`, `configs/train.yaml`).
- `greyscope/pipeline/` — the trilingual dataset build (EN / ja / zh-TW), an EditLens-recipe clone; all
  three languages built the same way (registry mirror+edit generation). `data/v2/splits/` is complete
  (graded middle filled by EN top-ups 2026-07-11/07-15 + non-native-EN reddit-l2 humans). **EditLens is
  EVAL-only** (its held-out OOD `test_llama`/`test_enron` slices — held out for EditLens too, not train
  domains) so the model ships Apache-2.0 — EN trains on non-EditLens sources only.
- `tests/` — pytest. `data/` and `.agents/memory/` are gitignored.

## Dataset pipeline (`greyscope/pipeline/`)

Stages-as-files, wired by the build driver `scripts/build.py` (`--smoke` cheap-validates new
sources; `--full` is the real run):

`corpora` (human loaders + source-artifact normalization; EN humans — fineweb / arxiv / gutenberg /
wikinews / amazon / stackexchange / reddit-l2 non-native-EN, fetched via Arctic Shift) → `generate`
(1 mirror + 2 edits per doc; `GENERATORS` registry-as-data; reasoning payloads per model) → `gates` (quality + within-type near-dedup) → `score`
(Qwen3-Embedding-8B cosine = edit magnitude) → `assemble` (bucket via per-language `BUCKET_CUTS`,
split by source-doc, exact-text dedup, write CSV + prompt manifest). `decontam` scrubs EN humans vs
RAID + EditLens-test + Beemo **before** generation.
`openrouter` is the cached chat+embed client; `pricing` is the list-price cross-check. Scrapers: `ptt`,
`wikinews`, `twgov`. `--assemble-only` re-runs gate→score→assemble from cached generations (no spend);
`--topup-en N` additively generates N NEW EN docs on the current registry and appends (never overwrites);
`--topup-edits K` adds K more edits/doc to already-generated docs (unused prompts, same split). Both lift
the graded middle by edit volume without a full rebuild.

`paraphrase` (driver `--paraphrase`) is a post-assembly robustness stage: paraphrase existing AI rows
(strong paraphrasers), keep the original label for fully-AI and re-score paraphrased edits vs the human
source. Writes `train_aug_paraphrase.csv` (train-only aug, wired into training via `data.train_extra_files`)
and `attack_paraphrase{,_test}.csv` (held-out-paraphraser eval slices with human negatives). Ablation
2026-07-12 adopted it: attack TPR@1% +13pt, graded-middle recall lifted, in-domain floor flat.

## Conventions (non-obvious)

- **Cache everything → resumable.** Every API response is cached by content hash (`data/v2/cache/`);
  re-runs never re-pay. A killed `--full` resumes from cache. Scrapers cache each fetch too.
- **Pure core + thin network shell.** Parsers/transforms are pure and unit-tested; only the `fetch_*`
  /`chat`/`embed` wrappers touch the network (tests don't).
- **Decontam is EN-only by design** — ja/zh-TW sources are disjoint from public benchmarks (see
  `decontam.py` docstring); their integrity rests on internal held-out slices.
- **Train view vs release view** — `assemble` keeps every edit for *training* (training ≠
  redistribution); the release view (`drop_unshippable=True`) drops edits derived from license-restricted
  sources (wiki40b / ptt / amazon), since shipping an edit ships a derivative. Release is not built yet.
- **Cost is recorded** — chat/embed send `usage:{include:true}`; the build report leads with actual
  `usage.cost`, list-price as cross-check. A spend-cap 403 aborts the build with a clear message.

Design spec, implementation plan, and the running experiment log live in `.agents/memory/`
(`V2_DATASET_DESIGN.md`, `V2_IMPLEMENTATION_PLAN.md`, `EXPERIMENTS.md`) — gitignored working notes.

## Commands

```bash
pytest -q          # tests
ruff check .       # lint
python scripts/build.py                    # dry run (no spend)
python scripts/build.py --assemble-only    # rebuild splits from cache (no spend)
python scripts/build.py --topup-en N       # generate N new EN docs + append (SPENDS)
python scripts/build.py --topup-edits K    # add K more edits/doc to existing docs (SPENDS)
python scripts/build.py --paraphrase-estimate  # grounded cost of the paraphrase stage (no spend)
python scripts/build.py --paraphrase       # build paraphrase aug + attack slices (SPENDS)
```

## Training (`configs/train.yaml`, `modal/train.py::production`)

Qwen3.5-4B LoRA + CORN 4-bucket ordinal head + MELD hard-negative ranking loss, joint
language×bucket sampler (τ=0.5), paraphrase aug folded in (`data.train_extra_files`). Checkpoint
selection on `eval_detection_auroc` (not 4-bucket macro-F1 — the thin middle is noisy). Post-train
OOD scoreboard = `test_llama`/`test_enron`/`ood_generator`/`attack_paraphrase`. Ship bf16 LoRA →
int4-HQQ (`greyscope/export.py`). Modal ladder: `smoke` (0.8B/64 rows, plumbing) → `ablation`
(12k, one knob) → `production` (the full run). **Trained 2026-07-15 (`production-r2`, `boundary_margin=0.003`):
ternary macro-F1 0.877 / ai_edited F1 0.823 / detection AUROC 0.986; ties editlens-Llama-3.2-3B on detection
+ in-domain graded ternary (0.895) and on independent RAID-10k (0.995). Shipped artifact complete at
`export_production-r2/`: `merged/` (bf16 + `calibration.json`) + `int4/` (HQQ). Trilingual calibration binds
the ≤1% FPR binary threshold on the non-native-EN subgroup (0.837; per-lang en/ja/zh-tw 0.055/0.321/0.255);
int4 tracks bf16 at 98.4% bucket agreement. HF push still pending.**
