"""Tests for the OpenRouter chat client (greyscope/v2/openrouter.py).

No network: the cache-hit path is exercised by pre-writing a cached response, the
miss path by stubbing the HTTP post. Guards the flex `served_tier` read-back, the
default-decoding body, and the content-hash cache that makes generation resumable.
"""

import json

import pytest

from greyscope.v2 import openrouter
from greyscope.v2.openrouter import ChatResult, _chat_cache_path, _parse_chat, chat

_RAW = {
    "model": "openai/gpt-5.5",
    "service_tier": "flex",
    "choices": [
        {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(openrouter, "CHAT_CACHE_DIR", tmp_path / "chat")


def test_parse_chat_extracts_text_tier_usage():
    result = _parse_chat(_RAW)
    assert result.text == "hello"
    assert result.served_tier == "flex"  # the actually-billed tier
    assert result.finish_reason == "stop"
    assert result.usage["total_tokens"] == 12


def test_parse_chat_tolerates_empty_response():
    assert _parse_chat({}) == ChatResult(
        text="", model="", served_tier=None, finish_reason=None, usage={}
    )


def test_cache_key_depends_on_request_fields():
    msgs = [{"role": "user", "content": "hi"}]
    base = {"model": "m", "messages": msgs}
    flex = {"model": "m", "messages": msgs, "service_tier": "flex"}
    assert _chat_cache_path(base) == _chat_cache_path(dict(base))  # stable
    assert _chat_cache_path(base) != _chat_cache_path(flex)  # tier ⇒ distinct slot


def test_chat_returns_cached_without_network(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("network must not be hit on a cache hit")

    monkeypatch.setattr(openrouter, "_post_with_retry", _boom)
    body = {
        "model": "openai/gpt-5.5",
        "messages": [{"role": "user", "content": "hi"}],
        "service_tier": "flex",
        "usage": {"include": True},  # chat() adds this → part of the cache key
    }
    path = _chat_cache_path(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_RAW))

    result = chat(body["messages"], model="openai/gpt-5.5", service_tier="flex")
    assert result.text == "hello"
    assert result.served_tier == "flex"


def test_chat_calls_api_on_miss_then_serves_from_cache(monkeypatch):
    calls = []

    def _fake_post(client, path, body, max_retries):
        calls.append(path)
        assert "temperature" not in body and "top_p" not in body  # default decoding
        assert body["service_tier"] == "flex"
        return _RAW

    monkeypatch.setattr(openrouter, "_post_with_retry", _fake_post)
    msgs = [{"role": "user", "content": "hi"}]

    first = chat(msgs, model="openai/gpt-5.5", service_tier="flex")
    second = chat(msgs, model="openai/gpt-5.5", service_tier="flex")

    assert first.text == second.text == "hello"
    assert calls == ["/chat/completions"]  # the second call never paid (cache)


def test_chat_requests_actual_cost(monkeypatch):
    sent = {}

    def _fake_post(client, path, body, max_retries):
        sent.update(body)
        return _RAW

    monkeypatch.setattr(openrouter, "_post_with_retry", _fake_post)
    chat([{"role": "user", "content": "hi"}], model="m")
    assert sent["usage"] == {"include": True}  # asks OpenRouter to return the real billed cost


def test_cost_of_reads_usage_cost():
    assert openrouter.cost_of({"cost": 0.0123}) == 0.0123
    assert openrouter.cost_of({"prompt_tokens": 5}) == 0.0  # no cost field → 0
    assert openrouter.cost_of(None) == 0.0


def test_embed_records_and_sums_actual_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(openrouter, "EMBED_CACHE_DIR", tmp_path / "emb")

    def _fake_embed_batch(client, model, task_type, inputs, max_retries):
        return [[0.1, 0.2] for _ in inputs], 0.04  # $0.04 for the whole batch

    monkeypatch.setattr(openrouter, "_embed_batch", _fake_embed_batch)
    texts = ["alpha", "beta"]
    assert len(openrouter.embed(texts)) == 2
    assert abs(openrouter.embedding_cost(texts) - 0.04) < 1e-9  # per-text split sums back to the batch cost

    openrouter.embed(texts)  # re-run: served from cache, no re-charge
    assert abs(openrouter.embedding_cost(texts) - 0.04) < 1e-9


def test_chat_does_not_cache_truncated_response(monkeypatch):
    calls = []
    truncated = {"model": "m", "choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]}

    def _fake_post(client, path, body, max_retries):
        calls.append(path)
        return truncated

    monkeypatch.setattr(openrouter, "_post_with_retry", _fake_post)
    msgs = [{"role": "user", "content": "hi"}]
    chat(msgs, model="m")
    chat(msgs, model="m")
    assert len(calls) == 2  # truncated reply not cached → re-fetched on the next run


def test_post_with_retry_retries_non_json_body(monkeypatch):
    monkeypatch.setattr(openrouter.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(openrouter, "_api_key", lambda: "k")

    class _Resp:
        status_code = 200
        content = b"\n" * 10

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            if self._payload is None:  # empty/garbled 200 body
                raise json.JSONDecodeError("Expecting value", "", 0)
            return self._payload

    seq = [_Resp(None), _Resp(_RAW)]  # first non-JSON, then good

    class _Client:
        def post(self, url, headers, json):
            return seq.pop(0)

    out = openrouter._post_with_retry(_Client(), "/chat/completions", {"model": "m"}, max_retries=3)
    assert out == _RAW and not seq  # retried past the bad body


def test_post_with_retry_raises_clearly_on_persistent_non_json(monkeypatch):
    monkeypatch.setattr(openrouter.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(openrouter, "_api_key", lambda: "k")

    class _Resp:
        status_code = 200
        content = b"   "

        def raise_for_status(self):
            pass

        def json(self):
            raise json.JSONDecodeError("Expecting value", "", 0)

    class _Client:
        def post(self, url, headers, json):
            return _Resp()

    with pytest.raises(openrouter.OpenRouterError, match="non-JSON"):
        openrouter._post_with_retry(_Client(), "/chat/completions", {"model": "m"}, max_retries=2)


def test_post_with_retry_aborts_on_auth_error_without_retrying(monkeypatch):
    monkeypatch.setattr(openrouter.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(openrouter, "_api_key", lambda: "k")
    calls = []

    class _Resp:
        status_code = 403
        is_success = False
        text = "insufficient credits"
        content = b"insufficient credits"

    class _Client:
        def post(self, url, headers, json):
            calls.append(1)
            return _Resp()

    with pytest.raises(openrouter.OpenRouterAuthError, match="re-run"):
        openrouter._post_with_retry(_Client(), "/chat/completions", {"model": "m"}, max_retries=5)
    assert len(calls) == 1  # cap/credit/key is terminal — surfaced at once, not retried
    assert issubclass(openrouter.OpenRouterAuthError, openrouter.OpenRouterError)
