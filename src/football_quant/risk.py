from __future__ import annotations

from dataclasses import dataclass

from .domain import Edge, RecommendationStatus, RiskDecision


@dataclass(frozen=True)
class RiskConfig:
    bankroll: float = 1_000.0
    min_expected_value: float = 0.02
    min_probability: float = 0.52
    max_stake_fraction: float = 0.015
    max_daily_exposure_fraction: float = 0.06
    kelly_fraction: float = 0.25


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    net_odds = decimal_odds - 1.0
    if net_odds <= 0:
        return 0.0
    raw = (probability * net_odds - (1.0 - probability)) / net_odds
    return max(0.0, raw)


class RiskEngine:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def decide(self, edge: Edge, *, current_daily_exposure: float = 0.0) -> RiskDecision:
        reasons: list[str] = []
        probability = edge.estimate.fair_probability

        if edge.expected_value < self.config.min_expected_value:
            reasons.append("Expected value below minimum threshold.")
        if probability < self.config.min_probability:
            reasons.append("Model probability below minimum confidence threshold.")
        if current_daily_exposure >= self.config.bankroll * self.config.max_daily_exposure_fraction:
            reasons.append("Daily exposure limit already reached.")

        if reasons:
            return RiskDecision(
                status=RecommendationStatus.REJECT,
                stake=0.0,
                max_loss=0.0,
                reasons=tuple(reasons),
            )

        raw_fraction = kelly_fraction(probability, edge.quote.odds) * self.config.kelly_fraction
        capped_fraction = min(raw_fraction, self.config.max_stake_fraction)
        remaining_daily_capacity = max(
            0.0,
            self.config.bankroll * self.config.max_daily_exposure_fraction - current_daily_exposure,
        )
        stake = min(self.config.bankroll * capped_fraction, remaining_daily_capacity)

        if stake <= 0:
            return RiskDecision(
                status=RecommendationStatus.REJECT,
                stake=0.0,
                max_loss=0.0,
                reasons=("No remaining exposure capacity.",),
            )

        status = RecommendationStatus.ACCEPT if edge.expected_value >= self.config.min_expected_value * 2 else RecommendationStatus.WATCH
        return RiskDecision(
            status=status,
            stake=round(stake, 2),
            max_loss=round(stake, 2),
            reasons=(
                f"EV {edge.expected_value:.2%} clears threshold.",
                f"Fractional Kelly stake capped at {self.config.max_stake_fraction:.2%} of bankroll.",
            ),
        )
