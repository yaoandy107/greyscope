"""One-off probe: measure each model's REAL reasoning behavior on OpenRouter by observing
actual reasoning_tokens across off/minimal/low/medium/high. Docs + supported_parameters are
unreliable (effort is silently remapped, never echoed back), so this measures ground truth:
  - reasoning_tokens stays 0 everywhere  -> non-reasoning (or accept-and-ignore)
  - only `off` differs from the rest     -> on/off toggle, no effort dial
  - rises minimal<low<medium<high        -> a real effort dial / budget
Throwaway — read the table, then delete. Cached, so re-runs are free.
"""

from greyscope.v2 import openrouter
from greyscope.v2.generate import GENERATORS

PROMPT = [{"role": "user", "content": (
    "Three friends split a bill. Ana pays twice what Ben pays, and Ben pays $4 more than "
    "Cara. If the total is $44, how much does Cara pay? Reply with only the dollar amount."
)}]
SETTINGS = {
    "off": {"enabled": False},
    "minimal": {"effort": "minimal"},
    "low": {"effort": "low"},
    "medium": {"effort": "medium"},
    "high": {"effort": "high"},
}
MAX_TOKENS = 2048  # fixed: lets effort-as-budget-fraction (Claude/Qwen) reveal itself


def _reasoning_tokens(usage: dict):
    return (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")


def main() -> None:
    errors: dict[tuple[str, str], str] = {}
    header = f"{'model':<40}" + "".join(f"{k:>9}" for k in SETTINGS)
    print(header)
    print("-" * len(header))
    for g in GENERATORS:
        slug = g["slug"]
        cells = []
        for name, reasoning in SETTINGS.items():
            try:
                r = openrouter.chat(PROMPT, model=slug, extra={"reasoning": reasoning},
                                    max_completion_tokens=MAX_TOKENS)
                rt = _reasoning_tokens(r.usage)
                cell = "n/a" if rt is None else str(rt)
                if r.finish_reason == "length":
                    cell += "T"  # hit the cap (thinking ate the budget)
            except Exception as exc:  # noqa: BLE001 — record, don't abort the sweep
                cell = "ERR"
                errors[(slug, name)] = str(exc)[:140]
            cells.append(cell)
        print(f"{slug:<40}" + "".join(f"{c:>9}" for c in cells))
    print("\nlegend: number = reasoning_tokens · 'T' = truncated · 'n/a' = field absent · 'ERR' = call rejected")
    if errors:
        print("\nerrors:")
        for (slug, name), msg in errors.items():
            print(f"  {slug} [{name}]: {msg}")


if __name__ == "__main__":
    main()
