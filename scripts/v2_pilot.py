"""Phase-0 pilot driver (design §11) — the first end-to-end vertical slice.

    load humans → generate (mirror + edit) → gate → score → reports/pilot.md

It thins the full pipeline to a few sources per language across ≥2 contrasting
registers, and each section of the report retires a build risk (§11): zh-TW Simplified
rate, open2ch yield, tell-density gradient, register distinctness, real cost/CJK token
inflation, gate-firing per generator, the Qwen-cosine sanity check, and whether flex is
actually served.

RUNNING THIS SPENDS MONEY — the first paid OpenRouter calls (~$2 target). It makes NO
calls on import. Run explicitly, e.g.:

    python scripts/v2_pilot.py --per-lang 50 --langs en ja zh-tw

Every response is cached, so a re-run is free and resumable.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

from greyscope.preprocess import count_words
from greyscope.v2 import corpora, gates, generate, score
from greyscope.v2.gates import simplified_ratio

_MD_MARKERS = ("# ", "##", "**", "```", "- ", "* ", "> ", "1. ")  # incl. "# " h1 (Claude opens with it)

REPORT_PATH = Path("data/v2/reports/pilot.md")


# --- human loading: ≥2 contrasting registers per language (design §11) -------
def load_pilot_humans(language: str, per_lang: int) -> list[dict]:
    half = max(1, per_lang // 2)
    rows: list[dict] = []
    notes: list[str] = []
    if language == "en":
        rows += [r.to_row() for r in corpora.load_editlens(limit=per_lang)]
    elif language == "ja":
        rows += [r.to_row() for r in corpora.load_wiki40b("ja", limit=half)]  # formal
        try:
            rows += [r.to_row() for r in corpora.load_open2ch(limit=half)]  # casual (go/no-go)
        except Exception as exc:  # noqa: BLE001 — pilot tolerates a dead source
            notes.append(f"open2ch load failed: {exc}")
    elif language == "zh-tw":
        rows += [r.to_row() for r in corpora.load_wiki40b("zh-tw", limit=half)]  # formal
        try:  # PTT: casual + creative (scraped, cached)
            ptt_boards = {"Gossiping": corpora.CASUAL, "marvel": corpora.CREATIVE}
            rows += [r.to_row() for r in corpora.load_ptt(ptt_boards, limit_per_board=max(1, half // 2))]
        except Exception as exc:  # noqa: BLE001
            notes.append(f"PTT load failed: {exc}")
    for note in notes:
        print(f"  [warn] {language}: {note}")
    return rows


# --- cost (list-price estimate; flex halves flex-served rows) ----------------
def fetch_pricing() -> dict[str, tuple[float, float]]:
    resp = httpx.get("https://openrouter.ai/api/v1/models", timeout=30)
    resp.raise_for_status()
    pricing = {}
    for model in resp.json()["data"]:
        p = model.get("pricing", {})
        pricing[model["id"]] = (float(p.get("prompt", 0)), float(p.get("completion", 0)))
    return pricing


def estimate_cost(rows: list[dict], pricing: dict[str, tuple[float, float]]) -> dict:
    by_model: dict[str, dict] = {}
    for row in rows:
        usage = row["meta"].get("usage") or {}
        model = row["model"]
        pin, pout = pricing.get(model, (0.0, 0.0))
        cost = usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout
        if row["meta"].get("served_tier") == "flex":
            cost *= 0.5  # flex −50% on the served rows
        entry = by_model.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0})
        entry["calls"] += 1
        entry["prompt_tokens"] += usage.get("prompt_tokens", 0)
        entry["completion_tokens"] += usage.get("completion_tokens", 0)
        entry["cost"] += cost
    return by_model


# --- report helpers ----------------------------------------------------------
def _dist(values: list[float]) -> str:
    if not values:
        return "n/a"
    q = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return (f"n={len(values)}  min={min(values):.3f}  p25={q[0]:.3f}  median={q[1]:.3f}  "
            f"p75={q[2]:.3f}  max={max(values):.3f}  mean={statistics.mean(values):.3f}")


def _text_length(text: str, language: str) -> int:
    return count_words(text) if language == "en" else corpora.count_cjk_chars(text)


def _markdown_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return 100 * sum(any(m in r["text"] for m in _MD_MARKERS) for r in rows) / len(rows)


def _reasoning_tokens(row: dict) -> int:
    return row["meta"].get("reasoning_tokens") or 0


def build_report(per_lang: int, humans: dict, kept: list[dict], dropped: list[dict],
                 cost: dict) -> str:
    ai = [r for r in kept if r["text_type"] in ("ai_generated", "ai_edited")]
    flex_models = {g["slug"] for g in generate.GENERATORS if g["flex"]}
    flex_rows = [r for r in (kept + dropped) if r["model"] in flex_models]
    served = Counter(r["meta"].get("served_tier") for r in flex_rows)

    out = ["# Greyscope v2 — Phase-0 pilot report", "",
           f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · per-lang target {per_lang}", ""]

    out.append("## Counts")
    out.append("| lang | humans | mirrors | edits | kept | dropped |")
    out.append("|---|---|---|---|---|---|")
    for lang, h in humans.items():
        mir = sum(r["language"] == lang and r["text_type"] == "ai_generated" for r in kept + dropped)
        edt = sum(r["language"] == lang and r["text_type"] == "ai_edited" for r in kept + dropped)
        k = sum(r["language"] == lang for r in kept)
        d = sum(r["language"] == lang for r in dropped)
        out.append(f"| {lang} | {len(h)} | {mir} | {edt} | {k} | {d} |")

    out += ["", "## Gate firing (drop reasons)"]
    reasons = Counter(d["drop_reason"] for d in dropped)
    out.append("| reason | n |\n|---|---|")
    out += [f"| {r} | {n} |" for r, n in reasons.most_common()] or ["| (none) | 0 |"]
    out += ["", "### Drops per generator"]
    per_gen = Counter(d["model"] for d in dropped if d.get("model"))
    out.append("| model | drops |\n|---|---|")
    out += [f"| {m} | {n} |" for m, n in per_gen.most_common()] or ["| (none) | 0 |"]

    out += ["", "## Flex — actually served (real discount, §5)",
            f"flex-eligible rows: {len(flex_rows)} · served {dict(served)}"]

    out += ["", "## Cost (list-price estimate; flex halved on served rows)",
            "| model | calls | prompt tok | completion tok | est $ |", "|---|---|---|---|---|"]
    total = 0.0
    for model, e in sorted(cost.items(), key=lambda kv: -kv[1]["cost"]):
        total += e["cost"]
        out.append(f"| {model} | {e['calls']} | {e['prompt_tokens']} | {e['completion_tokens']} | {e['cost']:.4f} |")
    out.append(f"| **total** |  |  |  | **{total:.4f}** |")

    all_rows = kept + dropped
    out += ["", "## Realized model mix (vs planned weights)", "| model | rows | % |", "|---|---|---|"]
    mix = Counter(r["model"] for r in all_rows)
    mtot = sum(mix.values()) or 1
    out += [f"| {m} | {n} | {100 * n / mtot:.0f}% |" for m, n in mix.most_common()]

    out += ["", "## Thinking — requested vs actual reasoning_tokens (§6 provenance + fallback check)",
            "| requested | n | median actual reasoning_tok |", "|---|---|---|"]
    by_level: dict[str, list[int]] = {}
    for r in all_rows:
        if "reasoning_request" in r["meta"]:  # AI rows only; humans carry no reasoning
            lvl = generate._label_for(r["meta"]["reasoning_request"])
            by_level.setdefault(lvl, []).append(_reasoning_tokens(r))
    for lvl in ("none", "off", "minimal", "on", "low", "medium", "high"):
        toks = by_level.get(lvl)
        if toks:
            out.append(f"| {lvl} | {len(toks)} | {statistics.median(toks):.0f} |")
    # silent-fallback watch: asked for REAL thinking, got ~0 actual → provider coerced/ignored it
    # (minimal legitimately yields ~0, so it is NOT a fallback signal).
    suspect = Counter(r["model"] for r in all_rows
                      if "reasoning_request" in r["meta"]
                      and generate._label_for(r["meta"]["reasoning_request"]) in ("on", "low", "medium", "high")
                      and _reasoning_tokens(r) == 0)
    if suspect:
        out.append("")
        out.append("Fallback/ignored (asked to think → 0 actual reasoning): "
                   + ", ".join(f"{m} ×{n}" for m, n in suspect.most_common()))

    out += ["", "## Edit-magnitude score distribution (Qwen cosine 1−cos, §8) — re-derive thresholds here"]
    for lang in humans:
        scores = [r["cosine_score"] for r in kept if r["text_type"] == "ai_edited" and r["language"] == lang and r.get("cosine_score") is not None]
        out.append(f"- **{lang}**: {_dist(scores)}")

    out += ["", "## zh-TW Simplified-character rate on AI output (§8.1 / TAIDE decision)"]
    simp = [simplified_ratio(r["text"]) for r in ai if r["language"] == "zh-tw"]
    out.append(f"- {_dist(simp)}")

    out += ["", "## Length & formatting parity (human vs AI — §8 confound check)",
            "| lang | human len (median) | mirror len (median) | human md% | AI md% |", "|---|---|---|---|---|"]
    for lang, hrows in humans.items():
        hlen = [_text_length(r["text"], lang) for r in hrows]
        mlen = [_text_length(r["text"], lang) for r in kept
                if r["language"] == lang and r["text_type"] == "ai_generated"]
        ai_lang = [r for r in ai if r["language"] == lang]
        hmed = statistics.median(hlen) if hlen else 0
        mmed = statistics.median(mlen) if mlen else 0
        out.append(f"| {lang} | {hmed:.0f} | {mmed:.0f} | {_markdown_rate(hrows):.0f}% | {_markdown_rate(ai_lang):.0f}% |")

    out += ["", "## Naturalness samples (eyeball for fluency + register)"]
    seen = set()
    for r in ai:
        key = (r["language"], r["prompt_id"], r["model"], r["text_type"])
        if key in seen:
            continue
        seen.add(key)
        snippet = r["text"][:180].replace("\n", " ")
        out.append(f"- `{r['language']}/{r['prompt_id']}/{r['model']}/{r['text_type']}` — {snippet!r}")
        if len(seen) >= 18:
            break
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Greyscope v2 Phase-0 pilot (SPENDS MONEY).")
    parser.add_argument("--per-lang", type=int, default=50)
    parser.add_argument("--langs", nargs="+", default=["en", "ja", "zh-tw"])
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    humans, all_generated = {}, []
    for lang in args.langs:
        print(f"[{lang}] loading humans …")
        rows = load_pilot_humans(lang, args.per_lang)
        humans[lang] = rows
        print(f"[{lang}] {len(rows)} humans → generating (mirror + edit) …")
        all_generated += generate.generate(rows, generate.GENERATED_DIR / f"{lang}.jsonl")

    print(f"gating {len(all_generated)} generated rows …")
    kept, dropped = gates.run_gates(all_generated)
    print(f"scoring {sum(r['text_type'] == 'ai_edited' for r in kept)} edited rows …")
    score.score_edited(kept)

    pricing = fetch_pricing()
    report = build_report(args.per_lang, humans, kept, dropped, estimate_cost(kept + dropped, pricing))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"\nwrote {args.out}  ({len(kept)} kept, {len(dropped)} dropped)")


if __name__ == "__main__":
    main()
