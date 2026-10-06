from __future__ import annotations

from .domain import Edge, MarketQuote, ProbabilityEstimate
from .features import implied_probability


def expected_value(probability: float, decimal_odds: float) -> float:
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1")
    return probability * (decimal_odds - 1.0) - (1.0 - probability)


def calculate_edge(quote: MarketQuote, estimate: ProbabilityEstimate) -> Edge:
    if quote.match_id != estimate.match_id:
        raise ValueError("Quote and estimate must belong to the same match")
    if quote.side != estimate.side or quote.line != estimate.line:
        raise ValueError("Quote and estimate must describe the same selection")

    implied = implied_probability(quote.odds)
    return Edge(
        quote=quote,
        estimate=estimate,
        implied_probability=implied,
        expected_value=expected_value(estimate.fair_probability, quote.odds),
    )
