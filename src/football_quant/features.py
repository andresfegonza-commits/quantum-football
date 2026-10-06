from __future__ import annotations

from dataclasses import dataclass
from math import exp

from .domain import BetSide, MarketQuote, MatchSnapshot


@dataclass(frozen=True)
class FeatureVector:
    match_id: str
    side: BetSide
    line: float
    values: dict[str, float]

    def value(self, name: str, default: float = 0.0) -> float:
        return self.values.get(name, default)


def implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    return 1.0 / decimal_odds


def logistic(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def build_pre_match_features(
    match: MatchSnapshot,
    quote: MarketQuote,
    *,
    opening_quote: MarketQuote | None = None,
) -> FeatureVector:
    home_advantage = 0.0 if match.neutral_venue else 65.0
    elo_diff_home = match.home.elo - match.away.elo + home_advantage
    form_diff_home = match.home.recent_goal_diff - match.away.recent_goal_diff
    rest_diff_home = match.home.rest_days - match.away.rest_days
    injury_diff_home = match.away.injuries_index - match.home.injuries_index

    side_multiplier = 1.0 if quote.side == "home" else -1.0
    line_movement = 0.0
    odds_movement = 0.0
    if opening_quote is not None:
        line_movement = quote.line - opening_quote.line
        odds_movement = quote.odds - opening_quote.odds

    values = {
        "elo_diff": side_multiplier * elo_diff_home,
        "form_diff": side_multiplier * form_diff_home,
        "rest_diff": side_multiplier * rest_diff_home,
        "injury_advantage": side_multiplier * injury_diff_home,
        "home_field": 0.0 if quote.side == "away" or match.neutral_venue else 1.0,
        "line": quote.line,
        "line_movement": side_multiplier * line_movement,
        "odds_movement": odds_movement,
        "implied_probability": implied_probability(quote.odds),
    }
    return FeatureVector(match_id=match.match_id, side=quote.side, line=quote.line, values=values)


def estimate_cover_probability(features: FeatureVector) -> float:
    """Transparent baseline that maps features to cover probability for the quoted side."""
    score = (
        0.0045 * features.value("elo_diff")
        + 0.18 * features.value("form_diff")
        + 0.025 * features.value("rest_diff")
        + 0.12 * features.value("injury_advantage")
        - 0.16 * abs(features.value("line"))
        + 0.05 * features.value("line_movement")
    )
    return min(0.78, max(0.22, logistic(score)))


LIVE_FEATURE_PRIORITIES: tuple[str, ...] = (
    "minute",
    "score_state",
    "red_cards",
    "shots",
    "shots_on_target",
    "expected_goals_live",
    "dangerous_attacks",
    "recent_pressure_index",
    "in_play_line",
    "in_play_odds",
)
