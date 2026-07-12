"""Tests for the list-price cost model (greyscope/pipeline/pricing.py): per-model aggregation from
recorded usage and the flex −50% halving. fetch_pricing (network) is not unit-tested."""

from greyscope.pipeline.pricing import estimate_cost


def _row(model, ptok, ctok, served_tier=None):
    return {"model": model, "meta": {"usage": {"prompt_tokens": ptok, "completion_tokens": ctok},
                                      "served_tier": served_tier}}


def test_estimate_cost_aggregates_and_halves_flex():
    pricing = {"m": (1.0, 2.0)}  # $/token (prompt, completion)
    rows = [_row("m", 10, 5), _row("m", 10, 5, served_tier="flex")]
    out = estimate_cost(rows, pricing)["m"]
    # row1 full: 10*1 + 5*2 = 20; row2 flex: 20 * 0.5 = 10 → total 30 over 2 calls
    assert out["calls"] == 2
    assert out["prompt_tokens"] == 20 and out["completion_tokens"] == 10
    assert out["cost"] == 30.0


def test_estimate_cost_unknown_model_is_free():
    assert estimate_cost([_row("unlisted", 100, 100)], {})["unlisted"]["cost"] == 0.0
