"""OpenRouter client — cached chat completions + embeddings over one key.

`OPENROUTER_API_KEY` covers both generation and embeddings. Every call is cached
on disk by content hash, so re-runs never pay twice and a crashed run resumes for
free (the whole "resumable" mechanism).

Flex tier (`service_tier:"flex"`, -50%) is a *chat* lever — OpenAI + Google only,
best-effort, so we read the served `service_tier` back to bill/measure the real
discount; the embeddings endpoint has no flex tier. Decoding stays at the provider
default everywhere (no temperature/top_p) — the realistic distribution.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx

BASE_URL = "https://openrouter.ai/api/v1"
EMBED_MODEL = "qwen/qwen3-embedding-8b"  # locked edit-magnitude scorer
EMBED_CACHE_DIR = Path("data/v2/cache/embeddings")
CHAT_CACHE_DIR = Path("data/v2/cache/chat")

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenRouterError(RuntimeError):
    """An OpenRouter request failed, or failed after exhausting retries."""


class OpenRouterAuthError(OpenRouterError):
    """The key was rejected (401/402/403): spend cap reached, out of credit, or invalid.
    Terminal for a build — every further call fails the same way, so callers abort rather
    than skip. Cached responses are kept, so re-running after adding credit pays only for
    the remainder."""


def _load_dotenv() -> None:
    """Populate os.environ from a repo-root .env (only keys not already set)."""
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        _load_dotenv()
        key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set — export it or add it to a gitignored .env."
        )
    return key


def _cache_path(model: str, task_type: str | None, text: str) -> Path:
    digest = hashlib.sha256(
        "\x00".join((model, task_type or "", text)).encode("utf-8")
    ).hexdigest()
    return EMBED_CACHE_DIR / f"{digest}.json"


def embed(
    texts: list[str],
    *,
    model: str = EMBED_MODEL,
    task_type: str | None = None,
    batch_size: int = 48,
    max_retries: int = 6,
    timeout: float = 60.0,
) -> list[list[float]]:
    """Embed `texts`, returning one vector per input in the same order.

    Cached per `(model, task_type, text)`; only cache-misses hit the API. Set
    `task_type` (e.g. "SEMANTIC_SIMILARITY") to pass a provider task hint through
    to the model; left unset, the endpoint's default is used.
    """
    vectors: list[list[float] | None] = [None] * len(texts)
    pending: list[tuple[int, str]] = []
    for i, text in enumerate(texts):
        path = _cache_path(model, task_type, text)
        if path.exists():
            vectors[i] = json.loads(path.read_text())["embedding"]
        else:
            pending.append((i, text))

    if pending:
        EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=timeout) as client:
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                returned, batch_cost = _embed_batch(
                    client, model, task_type, [t for _, t in batch], max_retries
                )
                per_text = batch_cost / len(batch) if batch_cost else 0.0  # split evenly → exact in aggregate
                for (i, text), vector in zip(batch, returned):
                    vectors[i] = vector
                    _cache_path(model, task_type, text).write_text(
                        json.dumps({"embedding": vector, "cost": per_text})
                    )

    if any(v is None for v in vectors):  # unreachable; guards a silent misalign
        raise OpenRouterError("internal error: some embeddings were not filled")
    return vectors  # type: ignore[return-value]


def cost_of(usage: dict | None) -> float:
    """The actual USD billed for a chat call, read from OpenRouter's `usage.cost`
    (present because `chat()` sends `usage:{include:true}`). 0.0 if absent — e.g. a
    provider that didn't report it, caught by the build report's list-price cross-check."""
    return float((usage or {}).get("cost") or 0.0)


def embedding_cost(texts: Iterable[str], *, model: str = EMBED_MODEL, task_type: str | None = None) -> float:
    """Sum the actual embedding cost recorded in cache for the (deduplicated) `texts` — each was
    embedded once and stored its share of its batch's `usage.cost`, so the sum is exact. 0.0 for
    any text embedded before cost tracking (cached by an earlier run)."""
    total = 0.0
    for text in set(texts):
        path = _cache_path(model, task_type, text)
        if path.exists():
            total += float(json.loads(path.read_text()).get("cost") or 0.0)
    return total


def _auth_headers() -> dict:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }


def _post_with_retry(
    client: httpx.Client, path: str, body: dict, max_retries: int
) -> dict:
    """POST a JSON body to an OpenRouter endpoint, retrying transient failures
    (network errors + retryable HTTP statuses) with exponential backoff + jitter.
    Returns the parsed JSON response."""
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.post(
                f"{BASE_URL}{path}", headers=_auth_headers(), json=body
            )
        except httpx.TransportError as exc:  # transient network failure
            last_error = exc
        else:
            if response.status_code in _RETRYABLE_STATUS:
                last_error = OpenRouterError(f"HTTP {response.status_code}: {response.text[:200]}")
            elif response.status_code in (401, 402, 403):  # cap / credit / invalid key — terminal
                raise OpenRouterAuthError(
                    f"HTTP {response.status_code} — OpenRouter rejected the key (spend cap reached, "
                    f"out of credit, or invalid). Add credit / raise the cap, then re-run; cached "
                    f"responses are preserved so only the remainder is billed. {response.text[:200]}"
                )
            elif response.status_code >= 400:  # other 4xx/5xx (bad request, etc.) — terminal, surface it
                raise OpenRouterError(f"HTTP {response.status_code}: {response.text[:200]}")
            else:
                try:
                    return response.json()
                except json.JSONDecodeError as exc:  # empty/garbled 200 body → retry like a transient failure
                    last_error = OpenRouterError(f"non-JSON body ({len(response.content)}B): {exc}")
        if attempt < max_retries - 1:
            time.sleep(delay + random.uniform(0, delay * 0.25))  # backoff + jitter
            delay *= 2

    raise OpenRouterError(
        f"{path} request failed after {max_retries} attempts: {last_error}"
    )


def _embed_batch(
    client: httpx.Client,
    model: str,
    task_type: str | None,
    inputs: list[str],
    max_retries: int,
) -> tuple[list[list[float]], float]:
    """Returns (vectors, actual_batch_cost). Cost is OpenRouter's `usage.cost` (0.0 if unreported).

    Retries a 200 response that lacks `data`: the embeddings endpoint occasionally returns a transient
    error body with a 200 status (observed mid-run), which `_post_with_retry` passes through as success —
    without this a one-off blip would crash a whole build's scoring stage."""
    body: dict = {"model": model, "input": inputs, "encoding_format": "float", "usage": {"include": True}}
    if task_type:
        body["task_type"] = task_type  # best-effort passthrough to the provider
    delay = 2.0
    last: object = None
    for attempt in range(max_retries):
        raw = _post_with_retry(client, "/embeddings", body, max_retries)
        if "data" in raw:
            rows = sorted(raw["data"], key=lambda d: d["index"])
            if len(rows) != len(inputs):  # provider returned a short batch — fail loudly
                raise OpenRouterError(f"embeddings: requested {len(inputs)}, got {len(rows)}")
            cost = float((raw.get("usage") or {}).get("cost") or 0.0)
            return [row["embedding"] for row in rows], cost
        last = raw.get("error") or raw  # data-less 200 (transient provider error) — back off and retry
        if attempt < max_retries - 1:
            time.sleep(delay + random.uniform(0, delay * 0.25))
            delay *= 2
    raise OpenRouterError(f"embeddings: no 'data' after {max_retries} attempts: {str(last)[:200]}")


@dataclass
class ChatResult:
    """A parsed chat completion. `served_tier` is the tier OpenRouter ACTUALLY
    billed (`flex`/`priority`/`default`/None) — compare against the requested tier
    to catch silent fallback to full price."""

    text: str
    model: str
    served_tier: str | None
    finish_reason: str | None
    usage: dict


def _chat_cache_path(body: dict) -> Path:
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return CHAT_CACHE_DIR / f"{digest}.json"


def _parse_chat(raw: dict) -> ChatResult:
    choice = (raw.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return ChatResult(
        text=message.get("content") or "",
        model=raw.get("model", ""),
        served_tier=raw.get("service_tier"),
        finish_reason=choice.get("finish_reason"),
        usage=raw.get("usage") or {},
    )


def chat(
    messages: list[dict],
    *,
    model: str,
    service_tier: str | None = None,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
    extra: dict | None = None,
    max_retries: int = 6,
    timeout: float = 120.0,
) -> ChatResult:
    """Send one chat completion, cached on disk by request content.

    Decoding stays at the provider default (no temperature/top_p — the realistic
    distribution). Pass `service_tier="flex"` for the -50% tier; the returned
    `ChatResult.served_tier` reports what was actually billed. `reasoning_effort`
    takes the OpenRouter shorthand (minimal/low/medium/high/xhigh/none). `extra`
    passes any other body field straight through and is part of the cache key, so
    distinct configs (e.g. suppressed-markdown variants) never collide.
    """
    body: dict = {"model": model, "messages": messages}
    if service_tier:
        body["service_tier"] = service_tier
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if max_completion_tokens is not None:
        body["max_completion_tokens"] = max_completion_tokens
    if extra:
        body.update(extra)
    body["usage"] = {"include": True}  # OpenRouter returns the actual billed cost in usage.cost

    path = _chat_cache_path(body)
    if path.exists():
        return _parse_chat(json.loads(path.read_text()))

    with httpx.Client(timeout=timeout) as client:
        raw = _post_with_retry(client, "/chat/completions", body, max_retries)

    # Don't cache a truncated reply: a re-run with a higher max_completion_tokens must
    # refetch it, not serve the stale cut-off body (which gates would only drop again).
    if (raw.get("choices") or [{}])[0].get("finish_reason") != "length":
        CHAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw, ensure_ascii=False))
    return _parse_chat(raw)
