# Greyscope

A detector that estimates *how much* of a text is AI-written (human → AI-edited → AI-generated) as a
continuous `ai_involvement` score, not a binary verdict. Qwen3.5-4B LoRA, 4-bucket seq-cls head.

## Layout

- `greyscope/` — the **v1 model** (English): `inference.py`, `model.py`/`trainer.py`/`collator.py`,
  `eval.py`/`benchmark.py`/`calibration.py`, `preprocess.py` (the EditLens-ported text normalization +
  `score_to_bucket`). Training/release run on Modal (`modal/`, `configs/train.yaml`).
- `greyscope/v2/` — the **v2 dataset pipeline**: a trilingual (EN / ja / zh-TW) EditLens-recipe clone.
  Built 2026-06-25 (61,694 rows, `data/v2/splits/`); the trilingual model is not trained yet.
- `tests/` — pytest. `data/` and `.agents/memory/` are gitignored.

## v2 pipeline (`greyscope/v2/`)

Stages-as-files, wired by the build driver `scripts/v2_build.py` (`--smoke` cheap-validates new
sources; `--full` is the real run):

`corpora` (human loaders + per-source artifact normalization, §8.7) → `generate` (1 mirror + 1 edit
per doc; `GENERATORS` registry-as-data; reasoning payloads per model) → `gates` (quality + within-type
near-dedup) → `score` (Qwen3-Embedding-8B cosine = edit magnitude) → `assemble` (bucket via
`BUCKET_CUTS`, split by source-doc, drop license-restricted edits, write CSV + prompt manifest).
`decontam` scrubs EN humans vs RAID + EditLens-test **before** generation. `openrouter` is the cached
chat+embed client; `pricing` is the list-price cross-check. Scrapers: `ptt`, `wikinews`, `twgov`.

## Conventions (non-obvious)

- **Cache everything → resumable.** Every API response is cached by content hash (`data/v2/cache/`);
  re-runs never re-pay. A killed `--full` resumes from cache. Scrapers cache each fetch too.
- **Pure core + thin network shell.** Parsers/transforms are pure and unit-tested; only the `fetch_*`
  /`chat`/`embed` wrappers touch the network (tests don't).
- **Decontam is EN-only by design** — ja/zh-TW sources are disjoint from public benchmarks (see
  `decontam.py` docstring); their integrity rests on internal held-out slices.
- **Edited-class licensing routing** — only PD/permissive sources ship edits (a derivative);
  mirror-only sources generate build-only edits that `assemble.drop_unshippable_edits` removes.
- **Cost is recorded** — chat/embed send `usage:{include:true}`; the build report leads with actual
  `usage.cost`, list-price as cross-check.

Design spec, implementation plan, and the running experiment log live in `.agents/memory/`
(`V2_DATASET_DESIGN.md`, `V2_IMPLEMENTATION_PLAN.md`, `EXPERIMENTS.md`) — gitignored working notes.

## Commands

```bash
pytest -q          # tests
ruff check .       # lint
python scripts/v2_build.py            # dry run (no spend)
```
