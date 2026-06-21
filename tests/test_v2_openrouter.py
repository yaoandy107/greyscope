"""Tests for the OpenRouter chat client (greyscope/v2/openrouter.py).

No network: the cache-hit path is exercised by pre-writing a cached response, the
miss path by stubbing the HTTP post. Guards the flex `served_tier` read-back, the
default-decoding body, and the content-hash cache that makes generation resumable
(design §6).
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
    assert result.served_tier == "flex"  # the actually-billed tier (design §5)
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
