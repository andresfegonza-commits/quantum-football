from __future__ import annotations

from .domain import MarketQuote, MatchSnapshot, ProbabilityEstimate
from .features import build_pre_match_features, estimate_cover_probability


class BaselineAsianHandicapModel:
    name = "baseline_elo_market_context"

    def predict(
        self,
        match: MatchSnapshot,
        quote: MarketQuote,
        *,
        opening_quote: MarketQuote | None = None,
    ) -> ProbabilityEstimate:
        features = build_pre_match_features(match, quote, opening_quote=opening_quote)
        probability = estimate_cover_probability(features)
        return ProbabilityEstimate(
            match_id=match.match_id,
            side=quote.side,
            line=quote.line,
            fair_probability=probability,
            model_name=self.name,
            inputs=features.values,
        )
