"""Full v2 build driver (design §5/§14): load → generate → gate → score → assemble.

The asymmetric one-shot build — humans are the free FPR base (~12–15k/lang), the AI side is
budget-capped (~1.5k EN / 6.3k ja / 6.3k zh-TW, design §5). Scales the pilot to ALL sources and
wires `assemble`. Every response is cached → re-runs are free and resumable.

SPENDS MONEY. Makes NO calls on import. Three modes:

    python scripts/v2_build.py             # dry run: print the plan + human yields, no spend
    python scripts/v2_build.py --smoke     # tiny NEW-source validation through gen→gate→score (~$0.20)
    python scripts/v2_build.py --full       # the full build (~$50, the locked one-shot target)
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from greyscope.v2 import assemble, corpora, decontam, gates, generate, openrouter, pricing, score

REPORT_PATH = Path("data/v2/reports/build.md")

# Per-language human source caps → the register-balanced ~12–15k/lang FPR base (design §4/§5).
# Each entry: (source, loader(limit), cap). Slow scraped sources (wikinews/tw-gov/ptt) are capped
# lower; the totals oversize the AI budget on purpose (humans are free and the safety win).
HUMAN_PLAN: dict[str, list[tuple]] = {
    "en": [("editlens", lambda n: corpora.load_editlens(limit=n), 13000)],
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

# AI-doc budget per language → ×~2 (mirror+edit) ≈ the design's 1.5k EN / 6.3k ja / 6.3k zh-TW (§5).
AI_DOC_TARGET = {"en": 750, "ja": 3150, "zh-tw": 3150}

# Sources added this session — the smoke validates each end-to-end (gen→gate→score) before the
# full spend (retires "new loaders never went through generation" risk).
SMOKE_PLAN: dict[str, list[tuple]] = {
    "ja": [("aozora", corpora.load_aozora), ("wikinews-ja", corpora.load_wikinews_ja),
           ("amazon-reviews-ja", corpora.load_amazon_reviews_ja)],
    "zh-tw": [("tw-gov", corpora.load_twgov)],
}


def _seed(row: dict) -> str:
    return hashlib.sha256(row["text_id"].encode("utf-8")).hexdigest()


def load_humans(language: str) -> tuple[list[dict], list[dict]]:
    """Load + register-balance the human pool, then decontaminate EN against the benchmarks we
    report on (RAID + EditLens-test, §14.2) BEFORE generation — a contaminated human poisons its
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
    (the budget buys COVERAGE, not frequency from the biggest source — design §5)."""
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


# --- full build --------------------------------------------------------------
def run_build(langs: list[str]) -> None:
    humans: list[dict] = []
    generated: list[dict] = []
    ood: list[dict] = []
    contaminated: list[dict] = []
    for lang in langs:
        print(f"[{lang}] loading humans …")
        lang_humans, lang_contam = load_humans(lang)
        contaminated += lang_contam
        ai_docs = select_ai_docs(lang_humans, AI_DOC_TARGET[lang])
        print(f"[{lang}] {len(lang_humans)} humans → generating mirror+edit for {len(ai_docs)} docs …")
        generated += generate.generate(ai_docs, generate.GENERATED_DIR / f"{lang}.jsonl")
        humans += lang_humans
        if lang == "en":  # EN inherits EditLens's held-out OOD slices (design §11)
            ood += corpora.load_editlens_split("test_llama") + corpora.load_editlens_split("test_enron")

    print(f"gating {len(generated)} generated rows …")
    kept, dropped = gates.run_gates(generated)
    print(f"scoring {sum(r['text_type'] == 'ai_edited' for r in kept)} edited rows …")
    score.score_edited(kept)

    rows = humans + kept + ood
    print(f"assembling {len(rows)} rows → {assemble.SPLITS_DIR} …")
    counts = assemble.assemble(rows)
    _write_build_report(humans, kept, dropped, ood, contaminated, counts)
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


def _write_build_report(humans, kept, dropped, ood, contaminated, counts) -> None:
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
    shipped = humans + kept + ood
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

    out += ["", "## Decontamination (EN humans vs RAID + EditLens-test, §14.2)",
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
    parser.add_argument("--langs", nargs="+", default=["en", "ja", "zh-tw"])
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
    elif args.full:
        run_build(args.langs)
    else:
        _dry_run(args.langs)


if __name__ == "__main__":
    main()
