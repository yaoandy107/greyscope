"""Generation: one mirror + N edits per human doc, from DECOUPLED mirror/edit generator pools.

Consumer apps serve the good tier (ChatGPT web = Sol), so the full-gen (mirror) pool carries the
realistic flagship voice while the EDIT pool — where the per-doc edit multiplier lands — stays cheap
(off/low reasoning, no flagship). The two pools are sampled independently per doc.

Request-building is pure and seeded, so it unit-tests with no network; a re-run reconstructs
the same dataset and reuses the cache. `generate()` is the only networked entry point.
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from greyscope.pipeline import openrouter
from greyscope.pipeline.corpora import count_cjk_chars

PROMPTS_DIR = Path(__file__).parent / "prompts"
GENERATED_DIR = Path("data/v2/generated")

# Reasoning is data: each model lists the literal OpenRouter payloads it supports (probe-measured).
# The row records the payload sent + the ACTUAL reasoning_tokens (truth — providers silently
# coarsen). Graded effort is real only on grok; elsewhere stay thin (off↔low). None ⇒ omit. The EDIT
# pool takes only the cheap payloads (off/none/low/minimal) — see `_reasoning_plan(cheap_only=...)`.
_ALWAYS_LOW = [(1, {"effort": "low"})]                               # gpt-5.5: no minimal, floored low
_GEMINI = [(1, {"effort": "minimal"}), (1, {"enabled": True})]       # gemini-3.x: no off → minimal floor
_TOGGLE = [(1, {"enabled": False}), (1, {"enabled": True})]          # off/on
_TOGGLE_OFF_LEAN = [(3, {"enabled": False}), (1, {"enabled": True})]  # claude: premium → mostly off
_ON_IS_HIGH = [(1, {"enabled": False}), (1, {"effort": "high"})]     # mistral: on = high only
_DEEPSEEK = [(4, {"enabled": False}), (1, {"effort": "high"})]       # deepseek: mostly-off (on=high; no graded low)
_GROK = [(30, {"enabled": False}), (30, {"effort": "low"}),
         (25, {"effort": "medium"}), (15, {"effort": "high"})]       # grok-4.3: the one real graded dial
_GROK45 = [(3, {"effort": "low"}), (2, {"effort": "medium"}), (1, {"effort": "high"})]  # grok-4.5: no off, low floor
_GPT_LUNA = [(4, {"effort": "minimal"}), (3, {"effort": "low"})]  # cheap GPT: casual bulk (probe: low==medium coarsen)
_GPT_SOL = [(2, {"effort": "minimal"}), (1, {"effort": "low"})]      # flagship: cheapest settings (mirror-only)
_HY3 = [(3, {"enabled": False}), (2, {"effort": "low"})]            # tencent: defaults high → send off explicitly
_NO_REASONING = [(1, None)]                                          # ling: non-reasoning

# Registry as data: `weight` = per-language weight in each pool the model is in; `pools` = which pools
# it participates in (default both). The EDIT pool is CHEAP-ONLY (premium models are mirror-only), so
# full-gen carries the realistic flagship voice while the many per-doc edits stay cheap. `flex` = -50%
# tier (OpenAI/Google); zh-TW excludes mainland-CN families (bias rule).
_MIRROR_ONLY = ("mirror",)
GENERATORS: list[dict] = [
    # --- cheap bulk (both pools) ---
    {"family": "tencent", "slug": "tencent/hy3", "flex": False, "reasoning": _HY3, "weight": {"en": 5}},  # cheapest current chat family ($0.14/$0.58); mainland → not zh-TW
    {"family": "google", "slug": "google/gemini-3.1-flash-lite", "flex": True, "reasoning": _GEMINI, "weight": {"en": 4, "ja": 5, "zh-tw": 6}},  # cheap flex carrier ($0.25/$1.50)
    {"family": "deepseek", "slug": "deepseek/deepseek-v4-pro", "flex": False, "reasoning": _DEEPSEEK, "weight": {"en": 3, "ja": 4}},  # cheapest family + top real target (promoted from EN wt1)
    {"family": "deepseek", "slug": "deepseek/deepseek-v4-flash", "flex": False, "reasoning": _DEEPSEEK, "weight": {"en": 3}},  # ultra-cheap edit volume ($0.08/$0.15)
    {"family": "openai", "slug": "openai/gpt-5.6-luna", "flex": True, "reasoning": _GPT_LUNA, "weight": {"en": 3}},  # current cheap GPT; carries GPT-family edits (proxy for Sol)
    {"family": "xai", "slug": "x-ai/grok-4.3", "flex": False, "reasoning": _GROK, "weight": {"en": 3, "ja": 3, "zh-tw": 4}},  # cheap Grok volume
    {"family": "google", "slug": "google/gemini-3.5-flash", "flex": True, "reasoning": _GEMINI, "weight": {"en": 2, "ja": 3, "zh-tw": 3}},
    {"family": "alibaba", "slug": "qwen/qwen3.7-plus", "flex": False, "reasoning": _TOGGLE, "weight": {"en": 2, "ja": 4}},  # mainland: EN-allowed (bias rule is zh-TW-only)
    {"family": "xai", "slug": "x-ai/grok-4.5", "flex": False, "reasoning": _GROK45, "pools": _MIRROR_ONLY, "weight": {"en": 1}},  # current Grok (currency); mirror-only — probe: no "off", "low" floor burns ~2k reasoning tok
    {"family": "moonshot", "slug": "moonshotai/kimi-k2.6", "flex": False, "reasoning": _TOGGLE, "weight": {"en": 1, "ja": 3}},
    {"family": "mistral", "slug": "mistralai/mistral-medium-3.1", "flex": False, "reasoning": _ON_IS_HIGH, "weight": {"en": 1}},  # Western family, EN-only ($0.40/$2)
    {"family": "google-open", "slug": "google/gemma-4-31b-it", "flex": False, "reasoning": _TOGGLE, "weight": {"en": 1, "ja": 4, "zh-tw": 3}},
    # --- premium, MIRROR-ONLY (full-gen carries the realistic flagship voice; edits stay cheap) ---
    {"family": "openai", "slug": "openai/gpt-5.6-sol", "flex": True, "reasoning": _GPT_SOL, "pools": _MIRROR_ONLY, "weight": {"en": 5}},  # ~15% of the EN mirror pool (budget-first)
    {"family": "anthropic", "slug": "anthropic/claude-sonnet-5", "flex": False, "reasoning": _TOGGLE_OFF_LEAN, "pools": _MIRROR_ONLY, "weight": {"en": 1, "ja": 3, "zh-tw": 3}},
    {"family": "google", "slug": "google/gemini-3.1-pro-preview", "flex": True, "reasoning": _GEMINI, "pools": _MIRROR_ONLY, "weight": {"en": 1, "ja": 2, "zh-tw": 2}},  # probe: `gemini-pro-latest` alias 400s → use the concrete slug
    # --- ja/zh legacy (EN top-up does not regenerate them; kept for their cached-parity registry) ---
    {"family": "openai", "slug": "openai/gpt-5.5", "flex": True, "reasoning": _ALWAYS_LOW, "weight": {"ja": 3, "zh-tw": 3}},  # superseded by 5.6 for EN
    {"family": "ling", "slug": "inclusionai/ling-2.6-flash", "flex": False, "reasoning": _NO_REASONING, "weight": {"ja": 3}},
]

MARKDOWN_MODES = ("default", "suppressed")  # config-sampling axis
MAX_COMPLETION_TOKENS = 4096  # generous ceiling so finish_reason="length" flags a real runaway
GEN_CONCURRENCY = 24  # I/O-bound (each call waits ~20s on the API) → many in-flight; retries absorb 429s
EDITS_PER_DOC = 2  # >1 edit/doc densely fills the graded middle (edits land ~58% in b1/b2)
_CHEAP_REASONING = {"none", "off", "low", "minimal"}  # the edit pool's payloads (no thinking-token burn)

# Edits that ship are derivatives → only PD/permissive sources; others get a build-only edit
# (zh-TW scorer validation) tagged shippable_edit=False for assembly to drop.
# Edited-OK (PD / CC0 / CC-BY, attribution-only): aozora, wikinews-ja, tw-gov, open2ch, and the EN
# permissive sources gutenberg (PD), wikinews-en (CC BY), arxiv-abstracts (CC0). Mirror-only (a mirror
# is new work, an edit is a derivative): wiki40b + stackexchange (CC BY-SA share-alike), ptt (unlicensed),
# amazon-reviews-{ja,en} (no grant), fineweb (ODC-BY on the compilation; underlying web copyright murky).
SHIPPABLE_EDIT_SOURCES = {"wikinews-ja", "tw-gov", "aozora", "open2ch",
                          "gutenberg", "wikinews-en", "arxiv-abstracts"}
# Sources that get an edit but no mirror. Aozora (ja creative) is now mirror-eligible via a
# creative-writing prompt seeded by the work's theme — an original AI passage, not a retelling of
# the source (the regurgitation gate drops any verbatim echo). None are ineligible at present.
MIRROR_INELIGIBLE_SOURCES: set[str] = set()

# Per-doc steering, sampled so the AI side isn't keyed on one fixed string.
_LANG_STEER = {
    "en": ("",),
    "ja": ("日本語で書いてください。",),
    "zh-tw": ("請用繁體中文（台灣）書寫。", "請使用台灣慣用的詞彙與語氣，以繁體中文書寫。"),
}
_SUPPRESS_MD = {
    "en": "Do not use any Markdown formatting; write plain prose.",
    "ja": "Markdown記法は使わず、プレーンな文章で書いてください。",
    "zh-tw": "請勿使用任何 Markdown 格式，以純文字書寫。",
}
# Anti-preamble PREVENTION — the primary lever (the strip in row shaping is only the net),
# per-language to avoid a cross-lingual prompting confound.
_OUTPUT_ONLY = {
    "en": "Output only the text itself — no preamble, no commentary, no sign-off, and no surrounding quotes or code fences.",
    "ja": "前置きや説明、結びの挨拶は書かず、求められた本文だけを出力してください。引用符やコードブロックで囲まないでください。",
    "zh-tw": "請只輸出正文本身，不要加開場白、說明或結尾的客套話，也不要用引號或程式碼區塊把內容包起來。",
}
_MIRROR_FALLBACK = "formal"  # text_register with no template (e.g. "mixed")


# --- seeded sampling --------------------------------------------------------
def _seeded_index(seq: tuple | list, *parts) -> int:
    """Reproducible index: hash `parts` → position. No RNG, so re-runs reuse the cache."""
    digest = hashlib.sha256("\x00".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest, 16) % len(seq)


def _seeded_choice(seq: tuple | list, *parts) -> object:
    return seq[_seeded_index(seq, *parts)]


def generators_for(language: str, pool: str | None = None) -> list[dict]:
    """Generators available for `language`, optionally restricted to a `pool` ("mirror"/"edit")."""
    return [g for g in GENERATORS if language in g["weight"]
            and (pool is None or pool in g.get("pools", ("mirror", "edit")))]


def pick_generator(record: dict, generators: list[dict], seed_tag: str = "gen") -> dict:
    """Seeded pick over the per-language registry weights. `seed_tag` decorrelates the picks so a
    doc's mirror and its edits draw independently from their (already pool-filtered) `generators`."""
    lang = record["language"]
    weighted = [g for g in generators for _ in range(g["weight"][lang])]
    return _seeded_choice(weighted, record["source_id"], lang, seed_tag)


def pick_style(record: dict) -> dict:
    """Seeded pick over the language's prompt styles; one per doc (mirror + edit share it)."""
    styles = load_system_styles(record["language"])
    pool = [s for s in styles for _ in range(s.get("weight", 1))]
    return _seeded_choice(pool, record["source"], record["source_id"], "style")


# --- prompt loading (cached file reads) -------------------------------------
@lru_cache(maxsize=None)
def load_register_prompt(language: str) -> str:
    return (PROMPTS_DIR / "register" / f"{language}.md").read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def load_system_styles(language: str) -> tuple[dict, ...]:
    """Prompt-style pool as data: plain / humanizer / persona, each `{id, family, weight, text?}`.
    `family` is the eval evasion-slice handle."""
    raw = yaml.safe_load((PROMPTS_DIR / "system" / f"{language}.yaml").read_text(encoding="utf-8"))
    return tuple(raw)


def _style_instruction(style: dict, language: str) -> str:
    """humanizer → vendored /humanize reference; persona → inline text; plain → nothing."""
    if style["family"] == "humanizer":
        return load_register_prompt(language)
    return style.get("text", "")


@lru_cache(maxsize=None)
def _mirror_by_register(language: str) -> dict[str, tuple[str, ...]]:
    raw = yaml.safe_load((PROMPTS_DIR / "mirror" / f"{language}.yaml").read_text(encoding="utf-8"))
    return {register: tuple(variants) for register, variants in raw.items()}


def load_mirror_variants(text_register: str, language: str) -> tuple[str, ...]:
    by_register = _mirror_by_register(language)
    return by_register.get(text_register) or by_register[_MIRROR_FALLBACK]


@lru_cache(maxsize=None)
def load_edit_prompts(language: str) -> tuple[dict, ...]:
    raw = (PROMPTS_DIR / "edit" / f"{language}.yaml").read_text(encoding="utf-8")
    return tuple(yaml.safe_load(raw))


# --- prompt rendering -------------------------------------------------------
def _length_hint(record: dict) -> str:
    text, lang = record["text"], record["language"]
    if lang == "en":
        words = max(50, round(len(text.split()) / 25) * 25)
        return f"{words} words"
    # zh-TW sources carry more non-CJK noise (URLs/markup) and its models hit the number, so
    # target the CJK count the gate measures → no overshoot; ja prose is clean and undershoots,
    # so len(text) already lands it (an earlier run: zh-tw 1.31× → ~1.0×, ja unchanged at 0.96×).
    basis = count_cjk_chars(text) if lang == "zh-tw" else len(text)
    chars = max(150, round(basis / 50) * 50)
    return f"{chars}字"


def _topic_from_text(text: str, language: str) -> str:
    """Topic anchor when `meta.topic` is absent: the opening clause (keeps the mirror
    topic-matched)."""
    text = text.strip()
    if language == "en":
        first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
        return " ".join(first.split()[:12])
    return re.split(r"[。！？\n]", text, maxsplit=1)[0][:30]


def render_mirror(record: dict, variant: str) -> str:
    topic = record["meta"].get("topic") or _topic_from_text(record["text"], record["language"])
    return variant.format(topic=topic, length_hint=_length_hint(record))


def _system_prompt(record: dict, style: dict, markdown_mode: str, instruction: str = "") -> str:
    lang = record["language"]
    parts = [instruction] if instruction else []
    style_text = _style_instruction(style, lang)
    if style_text:
        parts.append(style_text)
    steer = _seeded_choice(_LANG_STEER[lang], record["source"], record["source_id"], "steer")
    if steer:
        parts.append(steer)
    if markdown_mode == "suppressed":
        parts.append(_SUPPRESS_MD[lang])
    parts.append(_OUTPUT_ONLY[lang])
    return "\n\n".join(parts)


def _mirror_variant(record: dict, generator: dict) -> tuple[str, str]:
    """Seeded mirror-template pick → (variant text, mirror_prompt_id). Single source of the
    pick so the recorded id always matches the prompt rendered (provenance)."""
    by_register = _mirror_by_register(record["language"])
    register = record["meta"].get("text_register", _MIRROR_FALLBACK)
    if register not in by_register:
        register = _MIRROR_FALLBACK
    variants = by_register[register]
    idx = _seeded_index(variants, record["source"], record["source_id"], generator["slug"], "mirror")
    return variants[idx], f"{register}/v{idx + 1}"


def build_mirror_messages(record: dict, generator: dict, style: dict, markdown_mode: str) -> list[dict]:
    variant, _ = _mirror_variant(record, generator)
    messages = []
    system = _system_prompt(record, style, markdown_mode)
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": render_mirror(record, variant)})
    return messages


def build_edit_messages(record: dict, style: dict, markdown_mode: str, edit_prompt: dict) -> list[dict]:
    return [
        {"role": "system", "content": _system_prompt(record, style, markdown_mode, edit_prompt["prompt"])},
        {"role": "user", "content": record["text"]},
    ]


# --- request specs ----------------------------------------------------------
@dataclass
class GenRequest:
    kind: str  # "mirror" | "edit"
    record: dict
    generator: dict
    prompt_id: str
    markdown_mode: str
    messages: list[dict]
    reasoning: dict | None = None  # OpenRouter payload; None ⇒ omit
    edit_prompt: dict | None = None
    mirror_prompt_id: str | None = None  # which mirror register/variant rendered it


def _label_for(payload: dict | None) -> str:
    """Reasoning payload → display label (the payload + reasoning_tokens stay the truth)."""
    if payload is None:
        return "none"
    if payload.get("enabled") is False:
        return "off"
    if "effort" in payload:
        return payload["effort"]
    return "on"


def _reasoning_plan(generator: dict, *seed_parts, cheap_only: bool = False) -> dict | None:
    """Seeded pick from the model's reasoning payloads → a fresh copy to send (None ⇒ omit).
    `cheap_only` (the edit pool) keeps just the off/none/low/minimal payloads — no thinking-token
    burn on a simple edit — falling back to omit-reasoning if a model has no cheap option."""
    pairs = generator["reasoning"]
    if cheap_only:
        pairs = [(w, p) for w, p in pairs if _label_for(p) in _CHEAP_REASONING] or [(1, None)]
    pool = [payload for weight, payload in pairs for _ in range(weight)]
    payload = _seeded_choice(pool, *seed_parts, "reason")
    return dict(payload) if payload is not None else None


def _edit_prompts_for_doc(record: dict) -> list[dict]:
    """EDITS_PER_DOC distinct edit prompts drawn from ONE split — keeps train/val/test edit prompts
    disjoint even with >1 edit/doc (a doc lands in the split its edits' `split_tag` names), seeded per doc."""
    src, sid = record["source"], record["source_id"]
    prompts = load_edit_prompts(record["language"])
    first = _seeded_choice(prompts, src, sid, "edit0")
    split_pool = [p for p in prompts if p["split"] == first["split"]]
    chosen = [first]
    for i in range(1, EDITS_PER_DOC):
        remaining = [p for p in split_pool if p not in chosen] or split_pool
        chosen.append(remaining[_seeded_index(remaining, src, sid, f"edit{i}")])
    return chosen


def build_requests(records: list[dict]) -> list[GenRequest]:
    """One mirror (mirror pool) + EDITS_PER_DOC edits (cheap edit pool) per record; the two pools are
    sampled independently. Mirror skipped for mirror-ineligible sources. No network."""
    requests: list[GenRequest] = []
    for record in records:
        src, sid, lang = record["source"], record["source_id"], record["language"]
        style = pick_style(record)

        if src not in MIRROR_INELIGIBLE_SOURCES:
            m_gen = pick_generator(record, generators_for(lang, "mirror"), "gen-mirror")
            m_md = _seeded_choice(MARKDOWN_MODES, src, sid, m_gen["slug"], "md")
            requests.append(GenRequest(
                "mirror", record, m_gen, style["id"], m_md,
                build_mirror_messages(record, m_gen, style, m_md),
                reasoning=_reasoning_plan(m_gen, src, sid, m_gen["slug"], "mirror"),
                mirror_prompt_id=_mirror_variant(record, m_gen)[1],
            ))

        edit_gens = generators_for(lang, "edit")
        for i, edit_prompt in enumerate(_edit_prompts_for_doc(record)):
            e_gen = pick_generator(record, edit_gens, f"gen-edit{i}")
            e_md = _seeded_choice(MARKDOWN_MODES, src, sid, e_gen["slug"], f"md-edit{i}")
            requests.append(GenRequest(
                "edit", record, e_gen, style["id"], e_md,
                build_edit_messages(record, style, e_md, edit_prompt),
                reasoning=_reasoning_plan(e_gen, src, sid, e_gen["slug"], f"edit{i}", cheap_only=True),
                edit_prompt=edit_prompt,
            ))
    return requests


def build_extra_edit_requests(records: list[dict], cached_rows: list[dict],
                              extra_per_doc: int) -> list[GenRequest]:
    """Additional edits for ALREADY-generated docs (the middle top-up): each doc draws
    `extra_per_doc` prompts it hasn't used yet, from the SAME split its cached edits name —
    split discipline holds, and no cached request is re-issued (a pool change re-seeds
    `_edit_prompts_for_doc`, so re-running the normal path would cache-miss every old edit;
    this path never touches the old draws). Docs without a cached edit are skipped. Pure."""
    used: dict[tuple, set] = {}
    split_of: dict[tuple, str] = {}
    n_cached: dict[tuple, int] = {}
    for row in cached_rows:
        if row.get("text_type") != "ai_edited":
            continue
        key = (row["source"], row["source_id"])
        used.setdefault(key, set()).add(row["meta"]["edit_prompt_id"])
        split_of[key] = row["meta"]["split_tag"]
        n_cached[key] = n_cached.get(key, 0) + 1

    requests: list[GenRequest] = []
    for record in records:
        key = (record["source"], record["source_id"])
        if key not in split_of:
            continue
        src, sid, lang = record["source"], record["source_id"], record["language"]
        style = pick_style(record)
        pool = [p for p in load_edit_prompts(lang)
                if p["split"] == split_of[key] and p["id"] not in used[key]]
        edit_gens = generators_for(lang, "edit")
        for j in range(extra_per_doc):
            if not pool:
                break
            i = n_cached[key] + j  # continue the per-doc edit index → fresh seeds, no collision
            prompt = pool.pop(_seeded_index(pool, src, sid, f"edit{i}"))
            e_gen = pick_generator(record, edit_gens, f"gen-edit{i}")
            e_md = _seeded_choice(MARKDOWN_MODES, src, sid, e_gen["slug"], f"md-edit{i}")
            requests.append(GenRequest(
                "edit", record, e_gen, style["id"], e_md,
                build_edit_messages(record, style, e_md, prompt),
                reasoning=_reasoning_plan(e_gen, src, sid, e_gen["slug"], f"edit{i}", cheap_only=True),
                edit_prompt=prompt,
            ))
    return requests


# --- row shaping (canonical schema, matches corpora.HumanRecord.to_row) ------
def _safe(slug: str) -> str:
    return slug.replace("/", "_").replace(":", "_")


def _usage_reasoning_tokens(usage: dict | None) -> int:
    return ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0


# --- output cleanup: leaked </think> + a leading chat-wrapper (prevention is primary) ---
# EditLens strips a leading wrapper too (remove_ai_header: leading-only, paragraph-level,
# single-paragraph guard). We keep those guards but gate on SHAPE — a short first line that
# is a pure ack OR a colon header naming the output — so content ("My top picks:", "これは…")
# is never the thing removed. Runs before scoring so a wrapper can't skew the edit cosine.
_ACK_ONLY = re.compile(
    r"^(sure|of course|certainly|absolutely|got it|okay|ok|はい|もちろん|了解(です|しました)?|"
    r"承知(しました)?|當然|好的|沒問題)[\s!.。！？：:、，,-]*$", re.I)
_WRAPPER_CUE = re.compile(
    r"(here|below|following|rewrit|revis|edit|version|text|本文|文章|以下|こちら|改寫|改写|"
    r"修正|修改|潤飾|重寫|版本|內容)", re.I)
_HEADER_MAX_CHARS = 100


def _strip_think(text: str) -> str:
    return text.split("</think>")[-1].lstrip() if "</think>" in text else text


def _strip_ai_header(text: str, language: str) -> tuple[str, str | None]:
    """Strip a leading chat-wrapper paragraph → (clean, removed | None). Only a SHORT first
    paragraph that is a pure ack or a colon header naming the output, and only when a body
    follows — so a real opening line is never removed."""
    paras = [p for p in text.split("\n") if p.strip()]
    if len(paras) < 2:
        return text, None
    head = re.sub(r"^\W+", "", paras[0]).rstrip("*_# \t")
    is_wrapper = len(head) <= _HEADER_MAX_CHARS and (
        bool(_ACK_ONLY.match(head)) or (head.endswith((":", "：")) and bool(_WRAPPER_CUE.search(head)))
    )
    return ("\n".join(paras[1:]).lstrip(), paras[0]) if is_wrapper else (text, None)


def _base_row(record: dict, req: GenRequest, result: openrouter.ChatResult, text_type: str) -> dict:
    text, stripped_header = _strip_ai_header(_strip_think(result.text), record["language"])
    # edit prompt id disambiguates the >1 edits/doc (same doc, distinct prompt → unique text_id)
    edit_tag = f"/{req.edit_prompt['id']}" if req.kind == "edit" and req.edit_prompt else ""
    return {
        "text_id": f"{record['language']}/{record['source']}/{record['source_id']}"
                   f"/{text_type}/{_safe(req.generator['slug'])}/{req.prompt_id}{edit_tag}",
        "text": text,
        "language": record["language"],
        "text_type": text_type,
        "source": record["source"],
        "source_id": record["source_id"],
        "source_text": record["text"],  # human original; scorer + length-match need it
        "model": req.generator["slug"],
        "prompt_id": req.prompt_id,
        "markdown_mode": req.markdown_mode,
        "meta": {
            **record["meta"],
            "stripped_header": stripped_header,  # build-only: the chat wrapper removed, if any (review surface)
            "reasoning_request": req.reasoning,  # payload sent (provenance)
            "reasoning_tokens": _usage_reasoning_tokens(result.usage),  # actual (truth)
            # build-only — assembly strips these from the shipped schema (like source_text):
            "served_tier": result.served_tier,
            "finish_reason": result.finish_reason,
            "usage": result.usage,
        },
    }


def mirror_row(record: dict, req: GenRequest, result: openrouter.ChatResult) -> dict:
    row = _base_row(record, req, result, "ai_generated")
    row["cosine_score"] = 1.0  # generated = 1 by class
    row["bucket"] = 3
    row["meta"]["mirror_prompt_id"] = req.mirror_prompt_id
    return row


def edit_row(record: dict, req: GenRequest, result: openrouter.ChatResult) -> dict:
    row = _base_row(record, req, result, "ai_edited")
    row["cosine_score"] = None  # filled by score.py
    row["bucket"] = None
    row["meta"].update({
        "edit_prompt_id": req.edit_prompt["id"],
        "edit_category": req.edit_prompt["category"],
        "split_tag": req.edit_prompt["split"],
        "shippable_edit": record["source"] in SHIPPABLE_EDIT_SOURCES,
    })
    return row


# --- networked execution (FIRST SPEND) --------------------------------------
def run_request(req: GenRequest) -> dict:
    result = openrouter.chat(
        req.messages,
        model=req.generator["slug"],
        service_tier="flex" if req.generator["flex"] else None,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        extra={"reasoning": req.reasoning} if req.reasoning is not None else None,
    )
    return mirror_row(req.record, req, result) if req.kind == "mirror" else edit_row(req.record, req, result)


def generate(records: list[dict], out_path: Path) -> list[dict]:
    """Run every request (cached) and write rows. NETWORKED — the first spend."""
    return run_requests(build_requests(records), out_path)


def run_requests(requests: list[GenRequest], out_path: Path) -> list[dict]:
    """Execute prebuilt requests (cached) and write rows. NETWORKED.
    Requests are independent + I/O-bound → run concurrently; a terminal API failure is logged
    and skipped (kept out of the batch), never allowed to abort the whole run."""
    slots: list[dict | None] = [None] * len(requests)
    with ThreadPoolExecutor(max_workers=GEN_CONCURRENCY) as pool:
        futures = {pool.submit(run_request, req): i for i, req in enumerate(requests)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                slots[i] = fut.result()
            except openrouter.OpenRouterAuthError:
                for pending in futures:  # cap/credit/key is terminal — stop, don't fail every call
                    pending.cancel()
                raise
            except openrouter.OpenRouterError as exc:
                req = requests[i]
                print(f"  [skip] {req.kind} {req.generator['slug']} ({req.record['language']}): {str(exc)[:120]}")
    rows = [r for r in slots if r is not None]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows
