from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Literal


BetSide = Literal["home", "away"]


class RecommendationStatus(StrEnum):
    ACCEPT = "accept"
    WATCH = "watch"
    REJECT = "reject"


@dataclass(frozen=True)
class TeamSnapshot:
    team_id: str
    name: str
    elo: float
    recent_goal_diff: float
    rest_days: int
    injuries_index: float = 0.0
    goals_for_avg: float = 0.0
    goals_against_avg: float = 0.0
    shots_for_avg: float = 0.0
    shots_against_avg: float = 0.0
    shots_on_target_for_avg: float = 0.0
    shots_on_target_against_avg: float = 0.0
    corners_for_avg: float = 0.0
    corners_against_avg: float = 0.0


@dataclass(frozen=True)
class MatchSnapshot:
    match_id: str
    kickoff: datetime
    competition: str
    home: TeamSnapshot
    away: TeamSnapshot
    neutral_venue: bool = False


@dataclass(frozen=True)
class Result:
    match_id: str
    home_goals: int
    away_goals: int


@dataclass(frozen=True)
class MarketQuote:
    match_id: str
    bookmaker: str
    side: BetSide
    line: float
    odds: float
    captured_at: datetime
    is_closing: bool = False


@dataclass(frozen=True)
class ProbabilityEstimate:
    match_id: str
    side: BetSide
    line: float
    fair_probability: float
    model_name: str
    inputs: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    quote: MarketQuote
    estimate: ProbabilityEstimate
    implied_probability: float
    expected_value: float


@dataclass(frozen=True)
class RiskDecision:
    status: RecommendationStatus
    stake: float
    max_loss: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Recommendation:
    match: MatchSnapshot
    edge: Edge
    risk: RiskDecision
    explanation: str
