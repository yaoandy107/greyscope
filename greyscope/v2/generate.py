"""Generation: one mirror + one edit per human doc (design §5–6, plan §6).

Request-building is pure and seeded, so it unit-tests with no network; a re-run reconstructs
the same dataset and reuses the cache. `generate()` is the only networked entry point — the
pilot gates the first spend.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from greyscope.v2 import openrouter

PROMPTS_DIR = Path(__file__).parent / "prompts"
GENERATED_DIR = Path("data/v2/generated")

# Reasoning is data: each model lists the literal OpenRouter payloads it supports (probe-measured,
# EXPERIMENTS 2026-06-17/20). The row records the payload sent + the ACTUAL reasoning_tokens (truth —
# providers silently coarsen). Graded effort is real only on grok; elsewhere stay thin. None ⇒ omit.
_ALWAYS_LOW = [(1, {"effort": "low"})]                               # gpt-5.5: no minimal, floored low
_GEMINI = [(1, {"effort": "minimal"}), (1, {"enabled": True})]       # gemini-3.x: no off → minimal floor
_TOGGLE = [(1, {"enabled": False}), (1, {"enabled": True})]          # off/on
_TOGGLE_OFF_LEAN = [(3, {"enabled": False}), (1, {"enabled": True})]  # claude: premium → mostly off
_ON_IS_HIGH = [(1, {"enabled": False}), (1, {"effort": "high"})]     # deepseek/mistral: on = high only
_GROK = [(30, {"enabled": False}), (30, {"effort": "low"}),
         (25, {"effort": "medium"}), (15, {"effort": "high"})]       # the one real graded dial
_NO_REASONING = [(1, None)]                                          # ling: non-reasoning

# Registry as data (slugs verified 2026-06-16): `weight` keys define per-language availability
# (zh-TW excludes mainland models, §5); `flex` = -50% tier (OpenAI/Google); `reasoning` = above.
GENERATORS: list[dict] = [
    {"family": "openai", "slug": "openai/gpt-5.5", "flex": True, "reasoning": _ALWAYS_LOW, "weight": {"en": 4, "ja": 3, "zh-tw": 3}},
    {"family": "google", "slug": "google/gemini-3-flash-preview", "flex": True, "reasoning": _GEMINI, "weight": {"en": 4, "ja": 5, "zh-tw": 6}},  # cheap flex carrier ($0.50/$3)
    {"family": "google", "slug": "google/gemini-3.5-flash", "flex": True, "reasoning": _GEMINI, "weight": {"en": 3, "ja": 3, "zh-tw": 3}},
    {"family": "google", "slug": "google/gemini-3.1-pro-preview", "flex": True, "reasoning": _GEMINI, "weight": {"en": 2, "ja": 2, "zh-tw": 2}},
    {"family": "anthropic", "slug": "anthropic/claude-sonnet-4.6", "flex": False, "reasoning": _TOGGLE_OFF_LEAN, "weight": {"en": 3, "ja": 3, "zh-tw": 3}},
    {"family": "xai", "slug": "x-ai/grok-4.3", "flex": False, "reasoning": _GROK, "weight": {"en": 3, "ja": 3, "zh-tw": 4}},
    {"family": "google-open", "slug": "google/gemma-4-31b-it", "flex": False, "reasoning": _TOGGLE, "weight": {"en": 1, "ja": 4, "zh-tw": 3}},
    {"family": "nvidia", "slug": "nvidia/nemotron-3-ultra-550b-a55b:free", "flex": False, "reasoning": _TOGGLE, "weight": {"en": 1, "ja": 3, "zh-tw": 3}},  # free
    {"family": "alibaba", "slug": "qwen/qwen3.7-plus", "flex": False, "reasoning": _TOGGLE, "weight": {"en": 1, "ja": 4}},  # mainland: EN-allowed (bias rule is zh-TW-only), big EN app-channel
    {"family": "deepseek", "slug": "deepseek/deepseek-v4-pro", "flex": False, "reasoning": _ON_IS_HIGH, "weight": {"en": 1, "ja": 4}},  # flagship = consumer-default + the cheap app workhorse (pro, not the flash mini)
    {"family": "ling", "slug": "inclusionai/ling-2.6-flash", "flex": False, "reasoning": _NO_REASONING, "weight": {"ja": 3}},
    {"family": "moonshot", "slug": "moonshotai/kimi-k2.6", "flex": False, "reasoning": _TOGGLE, "weight": {"en": 1, "ja": 3}},
    {"family": "mistral", "slug": "mistralai/mistral-medium-3-5", "flex": False, "reasoning": _ON_IS_HIGH, "weight": {"en": 1}},  # Western family, ~irrelevant for CJK → EN-only (§5)
]

MARKDOWN_MODES = ("default", "suppressed")  # config-sampling axis (§6)
MAX_COMPLETION_TOKENS = 4096  # generous ceiling so finish_reason="length" flags a real runaway

# Edits that ship are derivatives → only PD/permissive sources; others get a build-only edit
# (zh-TW scorer validation) tagged shippable_edit=False for assembly to drop (design §4/§13).
SHIPPABLE_EDIT_SOURCES = {"editlens", "wikinews-ja", "tw-gov-news", "aozora"}
# A novel can't be mirror-generated from its own text → human + edited only (design §4).
MIRROR_INELIGIBLE_SOURCES = {"aozora"}

# Per-doc steering, sampled so the AI side isn't keyed on one fixed string (§6).
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
# per-language to avoid a cross-lingual prompting confound (§6, EXPERIMENTS 2026-06-21).
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


def generators_for(language: str) -> list[dict]:
    return [g for g in GENERATORS if language in g["weight"]]


def pick_generator(record: dict, generators: list[dict]) -> dict:
    """Seeded pick over the per-language registry weights (§5)."""
    lang = record["language"]
    weighted = [g for g in generators for _ in range(g["weight"][lang])]
    return _seeded_choice(weighted, record["source_id"], lang, "gen")


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
    `family` is the eval evasion-slice handle (§11)."""
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
    text = record["text"]
    if record["language"] == "en":
        words = max(50, round(len(text.split()) / 25) * 25)
        return f"{words} words"
    chars = max(150, round(len(text) / 50) * 50)
    return f"{chars}字"


def _topic_from_text(text: str, language: str) -> str:
    """Topic anchor when `meta.topic` is absent: the opening clause (keeps the mirror
    topic-matched, §2)."""
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
    pick so the recorded id always matches the prompt rendered (§6 provenance)."""
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
    mirror_prompt_id: str | None = None  # which mirror register/variant rendered it (§6)


def _label_for(payload: dict | None) -> str:
    """Reasoning payload → display label (the payload + reasoning_tokens stay the truth)."""
    if payload is None:
        return "none"
    if payload.get("enabled") is False:
        return "off"
    if "effort" in payload:
        return payload["effort"]
    return "on"


def _reasoning_plan(generator: dict, *seed_parts) -> dict | None:
    """Seeded pick from the model's reasoning payloads → a fresh copy to send (None ⇒ omit)."""
    pool = [payload for weight, payload in generator["reasoning"] for _ in range(weight)]
    payload = _seeded_choice(pool, *seed_parts, "reason")
    return dict(payload) if payload is not None else None


def build_requests(records: list[dict]) -> list[GenRequest]:
    """Mirror + edit specs per record (mirror skipped for mirror-ineligible sources). No network."""
    requests: list[GenRequest] = []
    for record in records:
        src, sid = record["source"], record["source_id"]
        generator = pick_generator(record, generators_for(record["language"]))
        style = pick_style(record)
        markdown_mode = _seeded_choice(MARKDOWN_MODES, src, sid, generator["slug"], "md")

        if src not in MIRROR_INELIGIBLE_SOURCES:
            requests.append(GenRequest(
                "mirror", record, generator, style["id"], markdown_mode,
                build_mirror_messages(record, generator, style, markdown_mode),
                reasoning=_reasoning_plan(generator, src, sid, generator["slug"], "mirror"),
                mirror_prompt_id=_mirror_variant(record, generator)[1],
            ))
        edit_prompt = _seeded_choice(load_edit_prompts(record["language"]), src, sid, generator["slug"], "edit")
        requests.append(GenRequest(
            "edit", record, generator, style["id"], markdown_mode,
            build_edit_messages(record, style, markdown_mode, edit_prompt),
            reasoning=_reasoning_plan(generator, src, sid, generator["slug"], "edit"),
            edit_prompt=edit_prompt,
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
    return {
        "text_id": f"{record['language']}/{record['source']}/{record['source_id']}"
                   f"/{text_type}/{_safe(req.generator['slug'])}/{req.prompt_id}",
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
            "reasoning_tokens": _usage_reasoning_tokens(result.usage),  # actual (truth, §6)
            # build-only — assembly strips these from the shipped schema (like source_text):
            "served_tier": result.served_tier,
            "finish_reason": result.finish_reason,
            "usage": result.usage,
        },
    }


def mirror_row(record: dict, req: GenRequest, result: openrouter.ChatResult) -> dict:
    row = _base_row(record, req, result, "ai_generated")
    row["cosine_score"] = 1.0  # generated = 1 by class (§7)
    row["bucket"] = 3
    row["meta"]["mirror_prompt_id"] = req.mirror_prompt_id
    return row


def edit_row(record: dict, req: GenRequest, result: openrouter.ChatResult) -> dict:
    row = _base_row(record, req, result, "ai_edited")
    row["cosine_score"] = None  # filled by score.py (§8)
    row["bucket"] = None
    row["meta"].update({
        "edit_prompt_id": req.edit_prompt["id"],
        "edit_category": req.edit_prompt["category"],
        "split_tag": req.edit_prompt["split"],
        "shippable_edit": record["source"] in SHIPPABLE_EDIT_SOURCES,
    })
    return row


# --- networked execution (FIRST SPEND — gated by the pilot) ------------------
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
    """Run every request (cached) and write rows. NETWORKED — the pilot is the first spend."""
    rows = [run_request(req) for req in build_requests(records)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows
