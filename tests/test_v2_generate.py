"""Tests for generation request-building (greyscope/v2/generate.py).

Pure functions only — no API calls. Guards the mainland-exclusion for zh-TW, the
seeded (reproducible) config sampling, prompt assembly (humanizer injection,
markdown suppression, edit-on-source-text), and the emitted row schema incl. the
edit-shippability licensing flag.
"""

import pytest

from greyscope.v2 import generate as gen
from greyscope.v2.openrouter import ChatResult


def _rec(language="zh-tw", source="ptt", register="creative",
         source_id="marvel/M.1.A.1", text="繁體中文的測試內容。" * 30):
    return {
        "text_id": f"{language}/{source}/{source_id}/human_written",
        "text": text,
        "language": language,
        "source": source,
        "source_id": source_id,
        "meta": {"topic": "說謊的人", "length": len(text), "text_register": register},
    }


def test_mainland_excluded_in_zhtw_allowed_in_en():
    zhtw = {g["slug"] for g in gen.generators_for("zh-tw")}
    assert "openai/gpt-5.5" in zhtw
    for mainland in ("qwen/qwen3.7-plus", "deepseek/deepseek-v4-pro",
                     "inclusionai/ling-2.6-flash", "moonshotai/kimi-k2.6"):
        assert mainland not in zhtw  # bias exclusion is zh-TW-only
    ja = {g["slug"] for g in gen.generators_for("ja")}
    assert {"qwen/qwen3.7-plus", "deepseek/deepseek-v4-pro"} <= ja
    en = {g["slug"] for g in gen.generators_for("en")}
    # mainland models ARE allowed in EN (no bias concern; big EN app-channel)
    assert {"qwen/qwen3.7-plus", "deepseek/deepseek-v4-pro", "moonshotai/kimi-k2.6"} <= en
    assert "inclusionai/ling-2.6-flash" not in en  # ling stays ja-only
    assert "mistralai/mistral-medium-3.1" in en and "mistralai/mistral-medium-3.1" not in (ja | zhtw)  # EN-only
    # v2.2 current-model refresh: new EN generators present; gpt-5.5 dropped from EN; tencent stays off zh-TW
    assert {"tencent/hy3", "openai/gpt-5.6-luna", "openai/gpt-5.6-sol", "deepseek/deepseek-v4-flash"} <= en
    assert "openai/gpt-5.5" not in en and "openai/gpt-5.5" in zhtw
    assert "tencent/hy3" not in zhtw  # tencent = mainland → zh-TW excluded


def test_mirror_edit_pool_decoupling():
    # premium models are MIRROR-ONLY (edits stay cheap); cheap bulk is in both pools
    mirror = {g["slug"] for g in gen.generators_for("en", "mirror")}
    edit = {g["slug"] for g in gen.generators_for("en", "edit")}
    # grok-4.5 is mirror-only too (no "off"; its "low" floor burns ~2k reasoning tokens — probe-measured)
    premium = {"openai/gpt-5.6-sol", "anthropic/claude-sonnet-5",
               "google/gemini-3.1-pro-preview", "x-ai/grok-4.5"}
    assert premium <= mirror and not (premium & edit)
    assert {"tencent/hy3", "deepseek/deepseek-v4-flash", "openai/gpt-5.6-luna"} <= (mirror & edit)


def test_edit_pool_reasoning_is_cheap_only():
    by_slug = {g["slug"]: g for g in gen.GENERATORS}

    def labels(slug, cheap):
        g = by_slug[slug]
        return {gen._label_for(gen._reasoning_plan(g, "s", f"id{i}", cheap_only=cheap)) for i in range(120)}

    # full pool spans graded effort; cheap_only (the edit pool) drops medium/high everywhere
    assert labels("x-ai/grok-4.5", False) == {"low", "medium", "high"}
    assert labels("x-ai/grok-4.5", True) == {"low"}
    assert labels("deepseek/deepseek-v4-pro", True) == {"off"}          # on-is-high → off only
    assert labels("openai/gpt-5.6-luna", True) <= {"minimal", "low"}    # medium dropped
    assert labels("openai/gpt-5.6-sol", False) == {"minimal", "low"}


def test_seeded_choice_deterministic_and_varies():
    seq = ("a", "b", "c", "d")
    assert gen._seeded_choice(seq, "doc-1", "gen") == gen._seeded_choice(seq, "doc-1", "gen")  # stable
    spread = {gen._seeded_choice(seq, f"doc-{i}", "gen") for i in range(50)}
    assert len(spread) > 1  # distinct seeds actually diverge (not a constant)


def test_reasoning_plan_samples_supported_payloads_per_model():
    by_slug = {g["slug"]: g for g in gen.GENERATORS}

    def labels(slug, n=120):
        g = by_slug[slug]
        return {gen._label_for(gen._reasoning_plan(g, "s", f"id{i}", "m")) for i in range(n)}

    # gpt-5.5: always low (no minimal, no off) — fixed payload
    assert gen._reasoning_plan(by_slug["openai/gpt-5.5"], "s", "id", "m") == {"effort": "low"}
    assert labels("openai/gpt-5.5") == {"low"}
    # geminis: minimal floor / on, NEVER off (probe HTTP 400 / silently coarsened to minimal)
    for slug in ("google/gemini-3.1-flash-lite", "google/gemini-3.5-flash", "google/gemini-3.1-pro-preview"):
        assert labels(slug) == {"minimal", "on"}
        assert all(gen._reasoning_plan(by_slug[slug], "s", f"id{i}", "m") != {"enabled": False} for i in range(120))
    # ling: non-reasoning → always None
    assert gen._reasoning_plan(by_slug["inclusionai/ling-2.6-flash"], "s", "id", "m") is None
    assert labels("inclusionai/ling-2.6-flash") == {"none"}
    # grok: the one real graded dial → off + low/medium/high
    grok = labels("x-ai/grok-4.3")
    assert grok <= {"off", "low", "medium", "high"} and {"off", "low", "medium"} <= grok
    # claude toggle is OFF-leaning (premium → mostly no-think)
    claude = [gen._label_for(gen._reasoning_plan(by_slug["anthropic/claude-sonnet-5"], "s", f"id{i}", "m")) for i in range(200)]
    assert claude.count("off") > 2 * claude.count("on")
    # deepseek: real off + on-is-high (no low/medium), off-leaning
    assert labels("deepseek/deepseek-v4-pro") == {"off", "high"}
    # plain toggle (gemma): off/on both appear
    assert labels("google/gemma-4-31b-it") == {"off", "on"}
    # the returned payload is a COPY — safe to embed per-row without mutating the shared profile
    plan = gen._reasoning_plan(by_slug["openai/gpt-5.5"], "s", "id", "m")
    plan["sentinel"] = 1
    assert gen._ALWAYS_LOW[0][1] == {"effort": "low"}


def test_aozora_is_mirror_eligible():
    # aozora (ja creative) mirrors via a creative-writing prompt — fills the ja creative gap.
    rec = _rec(source="aozora", language="ja", register="creative",
               source_id="work-1", text="日本語の物語の本文。" * 20)
    assert [r.kind for r in gen.build_requests([rec])] == ["mirror", "edit", "edit"]


def test_mirror_skipped_for_ineligible_source(monkeypatch):
    monkeypatch.setattr(gen, "MIRROR_INELIGIBLE_SOURCES", {"aozora"})
    rec = _rec(source="aozora", language="ja", register="creative",
               source_id="work-1", text="日本語の物語の本文。" * 20)
    assert [r.kind for r in gen.build_requests([rec])] == ["edit", "edit"]


def test_generate_aborts_on_terminal_auth_error(monkeypatch, tmp_path):
    # a spend-cap/credit 403 is terminal — generate() must abort, not skip into a silent gap.
    def _boom(req):
        raise gen.openrouter.OpenRouterAuthError("spend cap reached")

    monkeypatch.setattr(gen, "run_request", _boom)
    with pytest.raises(gen.openrouter.OpenRouterAuthError):
        gen.generate([_rec()], tmp_path / "out.jsonl")


def test_topic_derived_from_text_when_no_meta_topic():
    rec = _rec(source="open2ch", language="ja", register="casual",
               text="今日は良い天気ですね。散歩に行きました。")
    rec["meta"].pop("topic")
    out = gen.render_mirror(rec, gen.load_mirror_variants("casual", "ja")[0])
    assert "今日は良い天気ですね" in out and "the subject described" not in out


def test_build_requests_mirror_plus_edits():
    reqs = gen.build_requests([_rec()])
    assert [r.kind for r in reqs] == ["mirror"] + ["edit"] * gen.EDITS_PER_DOC
    mirror, edits = reqs[0], reqs[1:]
    assert mirror.messages[-1]["role"] == "user"
    for edit in edits:
        assert edit.messages[-1]["content"] == _rec()["text"]  # edit acts on the source text
        assert edit.edit_prompt["prompt"] in edit.messages[0]["content"]
        assert edit.edit_prompt["split"] in {"train", "val", "test"}
    # the edits share ONE split (prompt-disjointness holds with >1 edit/doc) but use DISTINCT prompts
    assert len({e.edit_prompt["split"] for e in edits}) == 1
    assert len({e.edit_prompt["id"] for e in edits}) == len(edits)


def test_humanizer_style_injects_vendored_prompt_plain_does_not():
    rec, generator = _rec(), gen.generators_for("zh-tw")[0]
    register_prompt = gen.load_register_prompt("zh-tw")
    styles = {s["id"]: s for s in gen.load_system_styles("zh-tw")}
    humanized = gen.build_mirror_messages(rec, generator, styles["humanizer"], "default")[0]["content"]
    plain = gen.build_mirror_messages(rec, generator, styles["plain"], "default")
    assert register_prompt in humanized
    # plain style carries only steering (no vendored humanizer guidance)
    assert all(register_prompt not in m["content"] for m in plain)


def test_persona_committed_and_recorded_as_prompt_id():
    styles = {s["id"]: s for s in gen.load_system_styles("zh-tw")}
    assert {"plain", "humanizer"} <= set(styles)  # backbone
    assert "persona" in {s["family"] for s in styles.values()}  # persona committed, not gated
    req = gen.build_requests([_rec()])[0]  # the chosen style id is recorded, not a register enum
    assert req.prompt_id in styles


def test_suppressed_markdown_instruction_present():
    plain = {s["id"]: s for s in gen.load_system_styles("zh-tw")}["plain"]
    msgs = gen.build_mirror_messages(_rec(), gen.generators_for("zh-tw")[0], plain, "suppressed")
    assert gen._SUPPRESS_MD["zh-tw"] in msgs[0]["content"]


def test_render_mirror_fills_slots():
    out = gen.render_mirror(_rec(), gen.load_mirror_variants("creative", "zh-tw")[0])
    assert "說謊的人" in out and "字" in out
    assert "{topic}" not in out and "{length_hint}" not in out


def test_mirror_variant_fallback_for_unknown_register():
    assert gen.load_mirror_variants("no-such-register", "zh-tw") == gen.load_mirror_variants(gen._MIRROR_FALLBACK, "zh-tw")


def test_mirror_row_schema():
    rec = _rec()
    req = gen.build_requests([rec])[0]
    result = ChatResult(text="AI 產生的文字", model=req.generator["slug"],
                        served_tier="flex", finish_reason="stop", usage={"total_tokens": 5})
    row = gen.mirror_row(rec, req, result)
    assert row["text_type"] == "ai_generated"
    assert row["cosine_score"] == 1.0 and row["bucket"] == 3
    assert row["source_text"] == rec["text"]  # human original retained for the scorer
    assert row["meta"]["served_tier"] == "flex"
    assert gen._safe(req.generator["slug"]) in row["text_id"]  # slug "/" sanitized


def test_edit_row_shippability_by_source():
    result = ChatResult(text="編輯後的文字", model="m", served_tier=None, finish_reason="stop", usage={})
    ptt_edit = gen.build_requests([_rec(source="ptt")])[1]
    ptt_row = gen.edit_row(_rec(source="ptt"), ptt_edit, result)
    assert ptt_row["text_type"] == "ai_edited"
    assert ptt_row["cosine_score"] is None and ptt_row["bucket"] is None  # scorer fills later
    assert ptt_row["meta"]["shippable_edit"] is False  # unlicensed source

    # permissive-EN backbone: gutenberg (PD) edit ships; stackexchange (CC BY-SA share-alike) is
    # mirror-only. Reconciled against the loaders' actual source names.
    gut = _rec(language="en", source="gutenberg", register="creative", source_id="123", text="word " * 100)
    assert gen.edit_row(gut, gen.build_requests([gut])[1], result)["meta"]["shippable_edit"] is True
    se = _rec(language="en", source="stackexchange", register="casual", source_id="h1", text="word " * 100)
    assert gen.edit_row(se, gen.build_requests([se])[1], result)["meta"]["shippable_edit"] is False

    # tw-gov (OGDL) ships; amazon-reviews-ja (Amazon licensing) is mirror-only — both reconciled
    # against the loaders' actual source names.
    tw = _rec(language="zh-tw", source="tw-gov", register="journalistic", source_id="A1")
    assert gen.edit_row(tw, gen.build_requests([tw])[1], result)["meta"]["shippable_edit"] is True
    az = _rec(language="ja", source="amazon-reviews-ja", register="reviews", source_id="r1")
    assert gen.edit_row(az, gen.build_requests([az])[1], result)["meta"]["shippable_edit"] is False


def test_strip_ai_header_removes_wrappers_keeps_content():
    strip = gen._strip_ai_header
    # real leading chat-wrappers removed, body kept (EN, markdown-wrapped, zh-TW colon, JA ack)
    assert strip("Sure, here is the rewritten text:\n\nThe museum opened.", "en")[0] == "The museum opened."
    assert strip("**Here is the edited text:**\n\nBody.", "en")[0] == "Body."
    assert strip("當然，以下是修改後的文字：\n\n內文。", "zh-tw")[0] == "內文。"
    assert strip("もちろん。\n\n本文。", "ja")[0] == "本文。"
    # content is never the thing removed: colon w/o a wrapper cue, content opener, single paragraph
    assert strip("My top three picks:\n\n1. Ramen\n2. Sushi", "en")[1] is None
    assert strip("Here's the thing about this diner: it never disappoints.", "en")[1] is None
    assert strip("これは間違いなく今年一番の作品だ。\n\n続きの本文。", "ja")[1] is None
    assert strip("A single paragraph starting with Sure, with no body to fall back on.", "en")[1] is None


def test_strip_think_and_output_only_prevention():
    assert gen._strip_think("<think>reasoning</think>The content.") == "The content."
    assert gen._strip_think("No trace.") == "No trace."
    sp = gen._system_prompt({"language": "en", "source": "s", "source_id": "x", "meta": {}, "text": "t"},
                            {"id": "plain", "family": "plain"}, "default")
    assert gen._OUTPUT_ONLY["en"] in sp  # prevention line always appended


def test_base_row_strips_header_into_meta():
    rec = _rec(language="en", source="editlens", text="word " * 100)
    req = gen.build_requests([rec])[0]  # mirror
    result = ChatResult(text="Sure, here is the text:\n\nThe content body.", model="m",
                        served_tier=None, finish_reason="stop", usage={})
    row = gen.mirror_row(rec, req, result)
    assert row["text"] == "The content body."
    assert row["meta"]["stripped_header"] == "Sure, here is the text:"
