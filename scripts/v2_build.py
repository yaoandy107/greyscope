"""Full v2 build driver: load → generate → gate → score → assemble.

The asymmetric one-shot build — humans are the free FPR base (~12–15k/lang), the AI side is
budget-capped (~1.5k EN / 6.3k ja / 6.3k zh-TW). Runs ALL sources and
wires `assemble`. Every response is cached → re-runs are free and resumable.

SPENDS MONEY with --smoke/--full. Makes NO calls on import. Modes:

    python scripts/v2_build.py                  # dry run: print the plan + human yields, no spend
    python scripts/v2_build.py --smoke          # tiny NEW-source validation through gen→gate→score (~$0.20)
    python scripts/v2_build.py --full           # the full build (~$50, the locked one-shot target)
    python scripts/v2_build.py --topup-en N      # additively generate N NEW EN docs + append (SPENDS)
    python scripts/v2_build.py --assemble-only  # re-gate/score/assemble from cached generations (no spend)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from greyscope.v2 import assemble, corpora, decontam, gates, generate, openrouter, paraphrase, pricing, score

REPORT_PATH = Path("data/v2/reports/build.md")

# Per-language human source caps → the register-balanced ~12–15k/lang FPR base.
# Each entry: (source, loader(limit), cap). Slow scraped sources (wikinews/tw-gov/ptt) are capped
# lower; the totals oversize the AI budget on purpose (humans are free and the safety win).
HUMAN_PLAN: dict[str, list[tuple]] = {
    # EN = permissive sources (EditLens dropped from TRAINING for Apache-2.0 licensing; it stays an
    # EVAL set via load_editlens_split). Register-balanced like ja/zh; formal is the biggest register
    # (fineweb + arxiv), the rest ~2.5k each. FineWeb/arxiv/wikinews/gutenberg are pre-LLM/PD/CC0/CC-BY.
    "en": [
        ("fineweb", lambda n: corpora.load_fineweb_en(limit=n), 5000),
        ("arxiv-abstracts", lambda n: corpora.load_arxiv_abstracts_en(limit=n), 2500),
        ("gutenberg", lambda n: corpora.load_gutenberg_en(limit=n), 2500),
        ("wikinews-en", lambda n: corpora.load_wikinews_en(limit=n), 2500),
        ("amazon-reviews-en", lambda n: corpora.load_amazon_reviews_en(limit=n), 2500),
        ("stackexchange", lambda n: corpora.load_stackexchange_en(limit=n), 2500),
    ],
    "ja": [
        ("wiki40b-ja", lambda n: corpora.load_wiki40b("ja", limit=n), 4500),
        ("aozora", lambda n: corpora.load_aozora(limit=n), 3500),
        ("wikinews-ja", lambda n: corpora.load_wikinews_ja(limit=n), 2500),
        ("amazon-reviews-ja", lambda n: corpora.load_amazon_reviews_ja(limit=n), 2500),
        ("open2ch", lambda n: corpora.load_open2ch(limit=n), 1000),
    ],
    "zh-tw": [
        ("wiki40b-zh-tw", lambda n: corpora.load_wiki40b("zh-tw", limit=n), 5500),
        ("ptt", lambda n: corpora.load_ptt(limit_per_board=max(1, n // len(corpora.PTT_BOARDS))), 5000),
        ("tw-gov", lambda n: corpora.load_twgov(limit=n), 4500),
    ],
}

# AI-doc budget per language → ×~2 (mirror+edit) ≈ 3k EN / 6.3k ja / 6.3k zh-TW. EN is smaller
# because a cheap-weighted generator mix + the free-AI backbone (below) keep it under the ~$20 cap;
# the smoke measures the real EN per-row cost before the full spend (EN edited is the monitored risk).
AI_DOC_TARGET = {"en": 1500, "ja": 3150, "zh-tw": 3150}

# EN ai_generated comes ENTIRELY from the registry mirror pipeline (like ja/zh-TW). The free permissive
# backbone (Cosmopedia/HelpSteer2) was dropped as a false economy — a stale 2023 model, register-confounded
# (no human twin → a topic shortcut, not AI-ness). Kept as a 0-valued switch (>0 re-enables).
EN_FREE_AI = 0

# Sources the smoke validates end-to-end (gen→gate→score) before the full spend (retires the
# "a new loader never went through generation" risk).
SMOKE_PLAN: dict[str, list[tuple]] = {
    "en": [("fineweb", corpora.load_fineweb_en), ("arxiv-abstracts", corpora.load_arxiv_abstracts_en),
           ("gutenberg", corpora.load_gutenberg_en), ("wikinews-en", corpora.load_wikinews_en),
           ("amazon-reviews-en", corpora.load_amazon_reviews_en), ("stackexchange", corpora.load_stackexchange_en)],
    "ja": [("aozora", corpora.load_aozora), ("wikinews-ja", corpora.load_wikinews_ja),
           ("amazon-reviews-ja", corpora.load_amazon_reviews_ja)],
    "zh-tw": [("tw-gov", corpora.load_twgov)],
}


def _seed(row: dict) -> str:
    return hashlib.sha256(row["text_id"].encode("utf-8")).hexdigest()


def load_humans(language: str) -> tuple[list[dict], list[dict]]:
    """Load + register-balance the human pool, then decontaminate EN against the benchmarks we
    report on (RAID + EditLens-test) BEFORE generation — a contaminated human poisons its
    mirror+edit too, so the spend only buys clean docs. zh/ja have no external target (their
    sources are disjoint from public benchmarks; see decontam doc) → pass through unfiltered.
    Returns (clean, contaminated_dropped)."""
    rows: list[dict] = []
    for source, loader, cap in HUMAN_PLAN[language]:
        got = [r.to_row() for r in loader(cap)]
        print(f"  [{language}] {source}: {len(got)} humans (cap {cap})")
        rows += got
    if language != "en":
        return rows, []
    clean, contaminated = decontam.filter_english(rows, decontam.english_reference())
    print(f"  [en] decontam: {len(contaminated)} contaminated dropped / {len(rows)} → {len(clean)} clean")
    return clean, contaminated


def select_ai_docs(humans: list[dict], n: int) -> list[dict]:
    """Round-robin a seeded sample across sources → balanced register coverage on the AI side
    (the budget buys COVERAGE, not frequency from the biggest source)."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in humans:
        by_source[row["source"]].append(row)
    for rows in by_source.values():
        rows.sort(key=_seed)
    sources = sorted(by_source)
    picked: list[dict] = []
    i = 0
    while len(picked) < n and any(by_source.values()):
        bucket = by_source[sources[i % len(sources)]]
        if bucket:
            picked.append(bucket.pop())
        i += 1
    return picked


def _read_jsonl(path: Path) -> list[dict]:
    # split on "\n" only: json.dumps(ensure_ascii=False) emits raw U+2028/U+2029/U+0085
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text("utf-8").split("\n") if ln.strip()]


# --- additive top-up ---------------------------------------------------------
def run_topup(language: str, n_new: int) -> None:
    """Additively generate n_new NEW docs on the CURRENT registry and APPEND them to the cached
    {lang}.jsonl — the still-valid old generations are kept, not clobbered (generate() opens 'w',
    so we write a temp file then append). Docs already generated — keyed by (source, source_id),
    matching assemble's doc identity — are excluded, so the spend only buys new docs. Lifts the
    graded middle by edit VOLUME (EDITS_PER_DOC edits/doc). SPENDS MONEY. Re-run --assemble-only
    afterwards to re-gate/score/split with the enlarged cache."""
    path = generate.GENERATED_DIR / f"{language}.jsonl"
    cached = _read_jsonl(path)
    done = {(r["source"], r["source_id"]) for r in cached}
    print(f"[topup {language}] cache: {len(cached)} rows across {len(done)} docs")

    humans, _ = load_humans(language)  # reuses decontam for EN — a new doc is clean before it spends
    remaining = [h for h in humans if (h["source"], h["source_id"]) not in done]
    print(f"[topup {language}] {len(remaining)}/{len(humans)} human docs still ungenerated → selecting {n_new}")
    new_docs = select_ai_docs(remaining, n_new)
    if len(new_docs) < n_new:
        print(f"[topup {language}] WARNING: pool exhausted — only {len(new_docs)} new docs available")

    tmp = generate.GENERATED_DIR / f"_{language}_topup.jsonl"
    print(f"[topup {language}] generating mirror + {generate.EDITS_PER_DOC} edits for "
          f"{len(new_docs)} docs (SPENDS MONEY) …")
    new_rows = generate.generate(new_docs, tmp)

    with path.open("a", encoding="utf-8") as fh:  # APPEND — never overwrite the cached rows
        for row in new_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[topup {language}] appended {len(new_rows)} rows → {path} "
          f"(now {len(cached) + len(new_rows)} rows). Run --assemble-only to re-split.")


def _print_recommended_cuts(kept: list[dict]) -> None:
    """Print the p22/p80 edit-cosine cut per language — the numbers to hand-set into
    assemble.BUCKET_CUTS. Buckets 1/2 (the graded middle = the product) are that percentile band;
    assemble() reads the module constant, so editing it is the routing mechanism — this only
    recommends. Prints on every run_build (incl. --assemble-only) so a top-up is re-cut cheaply."""
    by_lang: dict[str, list[float]] = defaultdict(list)
    for r in kept:
        if r["text_type"] == "ai_edited" and r.get("cosine_score") is not None:
            by_lang[r["language"]].append(r["cosine_score"])
    print("\nRecommended BUCKET_CUTS (p22, p80 of edit cosine) — hand-set in assemble.py:")
    for lang in sorted(by_lang):
        vals = sorted(by_lang[lang])
        n = len(vals)
        p22 = vals[min(n - 1, int(0.22 * n))]
        p80 = vals[min(n - 1, int(0.80 * n))]
        print(f"  {lang:6s} n={n:5d}  p22={p22:.3f} p80={p80:.3f}   (current {assemble.BUCKET_CUTS.get(lang)})")


# --- full build --------------------------------------------------------------
def run_build(langs: list[str], *, assemble_only: bool = False) -> None:
    humans: list[dict] = []
    generated: list[dict] = []
    ood: list[dict] = []
    backbone_ai: list[dict] = []
    contaminated: list[dict] = []
    for lang in langs:
        print(f"[{lang}] loading humans …")
        lang_humans, lang_contam = load_humans(lang)
        contaminated += lang_contam
        path = generate.GENERATED_DIR / f"{lang}.jsonl"
        if assemble_only:  # resume from already-generated rows — no API calls, no spend
            cached = _read_jsonl(path)
            print(f"[{lang}] {len(lang_humans)} humans → reusing {len(cached)} cached generated rows …")
            generated += cached
        else:
            ai_docs = select_ai_docs(lang_humans, AI_DOC_TARGET[lang])
            print(f"[{lang}] {len(lang_humans)} humans → generating mirror+edit for {len(ai_docs)} docs …")
            generated += generate.generate(ai_docs, path)
        humans += lang_humans
        if lang == "en":  # EN keeps EditLens's held-out OOD slices for EVAL only
            ood += corpora.load_editlens_split("test_llama") + corpora.load_editlens_split("test_enron")
            if EN_FREE_AI:
                backbone_ai += corpora.load_free_ai_en(limit=EN_FREE_AI)
                print(f"[en] + {len(backbone_ai)} permissive free-AI backbone rows (cosmopedia/helpsteer2)")

    print(f"gating {len(generated)} generated rows …")
    kept, dropped = gates.run_gates(generated)
    print(f"scoring {sum(r['text_type'] == 'ai_edited' for r in kept)} edited rows …")
    score.score_edited(kept)
    _print_recommended_cuts(kept)

    # backbone_ai already carries EditLens labels+cosine → bypasses gates/score, straight into assembly
    rows = humans + kept + ood + backbone_ai
    print(f"assembling {len(rows)} rows → {assemble.SPLITS_DIR} …")
    counts = assemble.assemble(rows)
    _write_build_report(humans, kept, dropped, ood, contaminated, counts, backbone_ai)
    print(f"\nsplits: {counts}\nreport: {REPORT_PATH}")


def _actual_cost_by_model(rows: list[dict]) -> dict[str, dict]:
    """Per-model ACTUAL cost from OpenRouter's `usage.cost` (chat sends `usage:{include:true}`).
    No list price, no flex math — `cost` is already the discounted amount actually billed."""
    by_model: dict[str, dict] = {}
    for row in rows:
        usage = row["meta"].get("usage") or {}
        entry = by_model.setdefault(row["model"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0})
        entry["calls"] += 1
        entry["prompt_tokens"] += usage.get("prompt_tokens", 0)
        entry["completion_tokens"] += usage.get("completion_tokens", 0)
        entry["cost"] += openrouter.cost_of(usage)
    return by_model


def _list_price_estimate(rows: list[dict]) -> float | None:
    """Best-effort list-price total — a cross-check that catches a provider not reporting
    `usage.cost`. None if pricing can't be fetched (offline at report time)."""
    try:
        est = pricing.estimate_cost(rows, pricing.fetch_pricing())
    except Exception:
        return None
    return sum(e["cost"] for e in est.values())


def _write_build_report(humans, kept, dropped, ood, contaminated, counts, backbone_ai=()) -> None:
    chat_rows = kept + dropped
    by_model = _actual_cost_by_model(chat_rows)
    gen_cost = sum(e["cost"] for e in by_model.values())
    embed_texts = {t for r in kept if r["text_type"] == "ai_edited" and r.get("source_text") and r.get("text")
                   for t in (r["source_text"], r["text"])}
    embed_cost = openrouter.embedding_cost(embed_texts)
    est = _list_price_estimate(chat_rows)
    crosscheck = f" · list-price cross-check ${est:.2f}" if est is not None else ""

    out = ["# Greyscope v2 — build report", "",
           f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}", "",
           f"- humans: {len(humans)} · kept AI: {len(kept)} · dropped: {len(dropped)} "
           f"· decontam-dropped: {len(contaminated)} · OOD-ingest: {len(ood)}",
           f"- **actual cost: ${gen_cost + embed_cost:.2f}** "
           f"(generation ${gen_cost:.2f} + embeddings ${embed_cost:.2f}){crosscheck}",
           "", "## Splits", "| split | rows |", "|---|---|"]
    out += [f"| {s} | {n} |" for s, n in sorted(counts.items())]
    shipped = humans + kept + ood + list(backbone_ai)
    out += ["", "## Class × language", "| lang | human | mirror | edited |", "|---|---|---|---|"]
    for lang in ("en", "ja", "zh-tw"):
        c = Counter(r["text_type"] for r in shipped if r["language"] == lang)
        out.append(f"| {lang} | {c['human_written']} | {c['ai_generated']} | {c['ai_edited']} |")
    out += ["", "## Register coverage (human rows)", "| lang | registers |", "|---|---|"]
    for lang in ("en", "ja", "zh-tw"):
        regs = Counter(r["meta"].get("text_register") for r in shipped
                       if r["language"] == lang and r["text_type"] == "human_written")
        out.append(f"| {lang} | {dict(regs)} |")
    out += ["", "## Gate drops", "| reason | n |", "|---|---|"]
    out += [f"| {r} | {n} |" for r, n in Counter(d["drop_reason"] for d in dropped).most_common()]

    out += ["", "## Decontamination (EN humans vs RAID + EditLens-test)",
            f"- dropped {len(contaminated)} contaminated EN humans before generation"]
    if contaminated:
        ov = sorted((c["meta"]["contam_overlap"] for c in contaminated), reverse=True)
        out.append(f"- shared 13-gram overlap: max={ov[0]} · median={ov[len(ov) // 2]} · n={len(ov)}")

    out += ["", "## Cost — actual (OpenRouter usage.cost)",
            "| model | calls | prompt tok | completion tok | $ |", "|---|---|---|---|---|"]
    for model, e in sorted(by_model.items(), key=lambda kv: -kv[1]["cost"]):
        out.append(f"| {model} | {e['calls']} | {e['prompt_tokens']} | {e['completion_tokens']} | {e['cost']:.4f} |")
    out.append(f"| **generation** |  |  |  | **{gen_cost:.4f}** |")
    out.append(f"| embeddings (qwen3-embedding-8b) |  |  |  | {embed_cost:.4f} |")
    out.append(f"| **total** |  |  |  | **{gen_cost + embed_cost:.4f}** |")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


# --- paraphrase augmentation ---------------------------------------------------
# Train aug (AUG_MODEL) and the attack-eval slices (ATTACK_MODEL, held-out family):
# dev slice from val (ablation judging), test slice from test (final report only).
PARA_TRAIN_PER_LANG = 1500
PARA_ATTACK_PER_LANG = 300


# (source split, AI rows, paraphraser pool, output, human-negative split | None for train aug)
def _paraphrase_plan() -> list[tuple[list[dict], list[dict], Path, str | None]]:
    return [
        (paraphrase.select_ai_rows(paraphrase.read_split("train"), PARA_TRAIN_PER_LANG),
         paraphrase.AUG_MODELS, assemble.SPLITS_DIR / "train_aug_paraphrase.csv", None),
        (paraphrase.select_ai_rows(paraphrase.read_split("val"), PARA_ATTACK_PER_LANG),
         paraphrase.ATTACK_MODELS, assemble.SPLITS_DIR / "attack_paraphrase.csv", "val"),
        (paraphrase.select_ai_rows(paraphrase.read_split("test"), PARA_ATTACK_PER_LANG),
         paraphrase.ATTACK_MODELS, assemble.SPLITS_DIR / "attack_paraphrase_test.csv", "test"),
    ]


def run_paraphrase(estimate_only: bool) -> None:
    plan = _paraphrase_plan()
    prices = pricing.fetch_pricing()
    total = 0.0
    for rows, models, out, _ in plan:
        est = paraphrase.estimate_cost(rows, models, prices)
        total += est["cost"]
        slugs = "+".join(m["slug"].split("/")[1] for m in models)
        print(f"  {out.name}: {est['rows']} rows via {slugs} "
              f"(~{est['tokens_in'] // 1000}k in / {est['tokens_out'] // 1000}k out) ≈ ${est['cost']:.2f} list")
    print(f"  TOTAL ≈ ${total:.2f} list (actual usually ~0.85× list; flex already halved)")
    if estimate_only:
        return
    sources = paraphrase.human_sources("train", "val", "test")
    actual = 0.0
    for rows, models, out, human_split in plan:
        kept, cost = paraphrase.run(rows, models, out)
        actual += cost
        kept = paraphrase.rescore_edited(kept, sources)
        if human_split:  # attack slices need human negatives for a defined detection metric
            kept = kept + paraphrase.sample_humans(human_split, PARA_ATTACK_PER_LANG)
        paraphrase.write_rows(kept, out)
        buckets = Counter(r["bucket"] for r in kept)
        print(f"  {out.name}: {len(kept)} rows, buckets {dict(sorted(buckets.items()))}")
    print(f"paraphrase stage actual cost: ${actual:.2f} (+ embeddings, see cache)")


# --- new-source smoke (risk #2) ----------------------------------------------
def run_smoke(per_source: int = 3) -> None:
    humans: list[dict] = []
    for lang, sources in SMOKE_PLAN.items():
        for source, loader in sources:
            got = [r.to_row() for r in loader(limit=per_source)]
            print(f"[smoke] {lang}/{source}: loaded {len(got)} humans")
            humans += got

    print(f"[smoke] generating mirror+edit for {len(humans)} docs (SPENDS ~$0.20) …")
    generated = generate.generate(humans, generate.GENERATED_DIR / "_smoke_newsrc.jsonl")
    kept, dropped = gates.run_gates(generated)
    score.score_edited(kept)

    print("\n[smoke] per-source validation (gen → gate → score):")
    print(f"{'source':<20} {'docs':>4} {'mirror':>6} {'edit':>4} {'kept':>4} {'drop':>4} {'edit cosine':>12}")
    src_of = {h["source_id"]: h["source"] for h in humans}
    for source in sorted({h["source"] for h in humans}):
        gen_s = [r for r in generated if src_of.get(r["source_id"]) == source]
        kept_s = [r for r in kept if src_of.get(r["source_id"]) == source]
        cos = [r["cosine_score"] for r in kept_s
               if r["text_type"] == "ai_edited" and r.get("cosine_score") is not None]
        cos_str = f"{statistics.median(cos):.3f}" if cos else "n/a"
        print(f"{source:<20} {sum(1 for h in humans if h['source']==source):>4} "
              f"{sum(r['text_type']=='ai_generated' for r in gen_s):>6} "
              f"{sum(r['text_type']=='ai_edited' for r in gen_s):>4} "
              f"{len(kept_s):>4} {len(gen_s)-len(kept_s):>4} {cos_str:>12}")
    if dropped:
        print("\n[smoke] drop reasons:", dict(Counter(d["drop_reason"] for d in dropped)))


def _dry_run(langs: list[str]) -> None:
    print("DRY RUN (no spend). Human yields + AI budget per language:\n")
    for lang in langs:
        humans, contam = load_humans(lang)
        scrubbed = f" (− {len(contam)} decontam)" if contam else ""
        print(f"  [{lang}] TOTAL humans: {len(humans)}{scrubbed} | AI docs to generate: {AI_DOC_TARGET[lang]} "
              f"(→ ~{AI_DOC_TARGET[lang] * 2} AI rows)\n")
    print("Run with --smoke (validate new sources, ~$0.20) or --full (the build, ~$50).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Greyscope v2 build driver (SPENDS MONEY with --smoke/--full).")
    parser.add_argument("--smoke", action="store_true", help="tiny new-source validation (~$0.20)")
    parser.add_argument("--full", action="store_true", help="the full build (~$50)")
    parser.add_argument("--assemble-only", action="store_true",
                        help="re-gate/score/assemble from cached generations (no API, no spend)")
    parser.add_argument("--topup-en", type=int, default=0, metavar="N",
                        help="additively generate N NEW EN docs on the current registry + append (SPENDS)")
    parser.add_argument("--paraphrase", action="store_true",
                        help="generate the paraphrase train-aug + attack-eval slices (SPENDS ~$5-8)")
    parser.add_argument("--paraphrase-estimate", action="store_true",
                        help="print the grounded list-price estimate for --paraphrase (no spend)")
    parser.add_argument("--langs", nargs="+", default=["en", "ja", "zh-tw"])
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
    elif args.full:
        run_build(args.langs)
    elif args.topup_en:
        run_topup("en", args.topup_en)
    elif args.paraphrase or args.paraphrase_estimate:
        run_paraphrase(estimate_only=not args.paraphrase)
    elif args.assemble_only:
        run_build(args.langs, assemble_only=True)
    else:
        _dry_run(args.langs)


if __name__ == "__main__":
    main()
