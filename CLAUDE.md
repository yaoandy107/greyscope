# Greyscope

A detector that estimates *how much* of a text is AI-written (human → AI-edited → AI-generated) as a
continuous `ai_involvement` score, not a binary verdict. Qwen3.5-4B LoRA, 4-bucket CORN ordinal head.
v1 ships EN (CC BY-NC-SA); v2 targets a **trilingual (EN/ja/zh-TW), Apache-2.0** model.

## Layout

- `greyscope/` — the **v1 model** (English): `inference.py`, `model.py`/`trainer.py`/`collator.py`,
  `eval.py`/`benchmark.py`/`calibration.py`, `preprocess.py` (the EditLens-ported text normalization +
  `score_to_bucket`). Training/release run on Modal (`modal/`, `configs/train.yaml`).
- `greyscope/v2/` — the **v2 dataset pipeline**: a trilingual (EN / ja / zh-TW) EditLens-recipe clone,
  all three languages built the same way (registry mirror+edit generation). `data/v2/splits/` is complete
  (EN topped up 2026-07-11 to fill the graded middle → 7.6%, ja/zh untouched); the trilingual model is not
  trained yet. **EditLens is EVAL-only** (the `test_llama`/`test_enron` slices)
  so v2 can ship Apache-2.0 — EN trains on permissive sources only.
- `tests/` — pytest. `data/` and `.agents/memory/` are gitignored.

## v2 pipeline (`greyscope/v2/`)

Stages-as-files, wired by the build driver `scripts/v2_build.py` (`--smoke` cheap-validates new
sources; `--full` is the real run):

`corpora` (human loaders + source-artifact normalization; EN humans are permissive — fineweb / arxiv /
gutenberg / wikinews / amazon / stackexchange) → `generate` (1 mirror + 1 edit per doc; `GENERATORS`
registry-as-data; reasoning payloads per model) → `gates` (quality + within-type near-dedup) → `score`
(Qwen3-Embedding-8B cosine = edit magnitude) → `assemble` (bucket via per-language `BUCKET_CUTS`,
split by source-doc, exact-text dedup, write CSV + prompt manifest). `decontam` scrubs EN humans vs
RAID + EditLens-test + Beemo **before** generation.
`openrouter` is the cached chat+embed client; `pricing` is the list-price cross-check. Scrapers: `ptt`,
`wikinews`, `twgov`. `--assemble-only` re-runs gate→score→assemble from cached generations (no spend);
`--topup-en N` additively generates N NEW EN docs on the current registry and appends (never overwrites),
for lifting the graded middle by edit volume without a full rebuild.

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
python scripts/v2_build.py                    # dry run (no spend)
python scripts/v2_build.py --assemble-only    # rebuild splits from cache (no spend)
python scripts/v2_build.py --topup-en N       # generate N new EN docs + append (SPENDS)
python scripts/v2_build.py --paraphrase-estimate  # grounded cost of the paraphrase stage (no spend)
python scripts/v2_build.py --paraphrase       # build paraphrase aug + attack slices (SPENDS)
```

## v2 training (`configs/train_v2.yaml`, `modal/train.py::production_v2`)

Qwen3.5-4B LoRA + CORN 4-bucket ordinal head + MELD hard-negative ranking loss, joint
language×bucket sampler (τ=0.5), paraphrase aug folded in (`data.train_extra_files`). Checkpoint
selection on `eval_detection_auroc` (not 4-bucket macro-F1 — the thin middle is noisy). Post-train
OOD scoreboard = `test_llama`/`test_enron`/`ood_generator`/`attack_paraphrase`. Ship bf16 LoRA →
int4-HQQ (`greyscope/export.py`). Not trained yet — `production_v2` is the single committed run.
