"""OpenRouter list-price cost model — a cross-check on the actual `usage.cost` (openrouter.py).

`fetch_pricing` pulls live per-token rates from the `/models` endpoint; `estimate_cost` applies them
to the recorded token usage (halving flex-served rows). The build report leads with the *actual* billed
cost and keeps this estimate alongside to catch a provider that doesn't report cost.
"""

from __future__ import annotations

import httpx


def fetch_pricing() -> dict[str, tuple[float, float]]:
    """`{model_id: (prompt_$_per_token, completion_$_per_token)}` from OpenRouter's live catalog."""
    resp = httpx.get("https://openrouter.ai/api/v1/models", timeout=30)
    resp.raise_for_status()
    pricing = {}
    for model in resp.json()["data"]:
        p = model.get("pricing", {})
        pricing[model["id"]] = (float(p.get("prompt", 0)), float(p.get("completion", 0)))
    return pricing


def estimate_cost(rows: list[dict], pricing: dict[str, tuple[float, float]]) -> dict:
    """Per-model list-price estimate from recorded token usage (flex-served rows halved, §5)."""
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
