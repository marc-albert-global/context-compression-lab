"""Offline production cost model.

Translates a measured token-reduction percentage into the dollars it would save
at a stated request volume and token price. This is the number a buyer cares
about: not "we cut tokens 30%" but "at your volume that is roughly $X / year".

All figures are projections at an explicit, configurable volume and price, not
measured production spend. The reduction percentages themselves come from the
benchmark on the committed corpus, so the only assumptions a reader has to
accept are volume and price, both shown.
"""

from __future__ import annotations

from dataclasses import dataclass

# Example anchor: Claude Opus input pricing. Swap via the CLI for any model.
DEFAULT_PRICE_PER_MILLION = 5.0
DEFAULT_DAILY_INPUT_TOKENS = 1_000_000  # 1M input tokens/day (a modest workload)


@dataclass
class Saving:
    method: str
    token_reduction_pct: float
    tokens_saved_per_day: float
    usd_saved_per_day: float
    usd_saved_per_year: float


def project_saving(
    method: str,
    token_reduction_pct: float,
    *,
    daily_input_tokens: int = DEFAULT_DAILY_INPUT_TOKENS,
    price_per_million: float = DEFAULT_PRICE_PER_MILLION,
) -> Saving:
    """Project savings for one method at a given volume and price."""
    tokens_saved = daily_input_tokens * (token_reduction_pct / 100.0)
    usd_day = tokens_saved / 1_000_000 * price_per_million
    return Saving(
        method=method,
        token_reduction_pct=round(token_reduction_pct, 1),
        tokens_saved_per_day=round(tokens_saved),
        usd_saved_per_day=round(usd_day, 2),
        usd_saved_per_year=round(usd_day * 365, 0),
    )


def project_all(
    results,
    *,
    daily_input_tokens: int = DEFAULT_DAILY_INPUT_TOKENS,
    price_per_million: float = DEFAULT_PRICE_PER_MILLION,
) -> list[Saving]:
    """Project savings for every available, token-reducing method."""
    out = []
    for r in results:
        if not getattr(r, "available", True) or r.method == "identity":
            continue
        if r.token_reduction <= 0:  # a method that does not save tokens saves no money
            continue
        out.append(project_saving(
            r.method, r.token_reduction,
            daily_input_tokens=daily_input_tokens, price_per_million=price_per_million,
        ))
    out.sort(key=lambda s: s.usd_saved_per_year, reverse=True)
    return out


def markdown_table(savings: list[Saving], *, daily_input_tokens: int, price_per_million: float) -> str:
    head = (
        f"Projection at **{daily_input_tokens:,} input tokens/day** and "
        f"**${price_per_million:.2f}/1M tokens**:\n\n"
        "| Method | Token ↓ % | Tokens saved/day | $/day | $/year |\n"
        "|---|--:|--:|--:|--:|"
    )
    rows = [
        f"| `{s.method}` | {s.token_reduction_pct:.1f} | {s.tokens_saved_per_day:,.0f} | "
        f"${s.usd_saved_per_day:,.2f} | ${s.usd_saved_per_year:,.0f} |"
        for s in savings
    ]
    return head + "\n" + "\n".join(rows)
