from __future__ import annotations

from .ai_layer import QuantExplanationLayer
from .dataset import HistoricalDataset
from .domain import Recommendation
from .edge import calculate_edge
from .models import BaselineAsianHandicapModel
from .risk import RiskEngine


class OpportunityScanner:
    def __init__(
        self,
        *,
        model: BaselineAsianHandicapModel | None = None,
        risk_engine: RiskEngine | None = None,
        explanation_layer: QuantExplanationLayer | None = None,
    ) -> None:
        self.model = model or BaselineAsianHandicapModel()
        self.risk_engine = risk_engine or RiskEngine()
        self.explanation_layer = explanation_layer or QuantExplanationLayer()

    def scan(self, dataset: HistoricalDataset) -> tuple[Recommendation, ...]:
        matches = dataset.matches_by_id
        opening_by_key = _first_quotes_by_selection(dataset.pre_match_quotes())
        recommendations: list[Recommendation] = []
        daily_exposure = 0.0

        for quote in sorted(dataset.pre_match_quotes(), key=lambda item: item.captured_at):
            match = matches[quote.match_id]
            estimate = self.model.predict(
                match,
                quote,
                opening_quote=opening_by_key.get((quote.match_id, quote.side)),
            )
            edge = calculate_edge(quote, estimate)
            risk = self.risk_engine.decide(edge, current_daily_exposure=daily_exposure)
            daily_exposure += risk.stake
            explanation = self.explanation_layer.explain(match, edge, risk)
            recommendations.append(Recommendation(match, edge, risk, explanation))

        return tuple(recommendations)


def _first_quotes_by_selection(quotes: tuple) -> dict[tuple[str, str], object]:
    first_quotes: dict[tuple[str, str], object] = {}
    for quote in sorted(quotes, key=lambda item: item.captured_at):
        first_quotes.setdefault((quote.match_id, quote.side), quote)
    return first_quotes
