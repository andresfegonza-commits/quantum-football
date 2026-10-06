from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .domain import MarketQuote, MatchSnapshot, Result, TeamSnapshot


@dataclass(frozen=True)
class DataSourceCandidate:
    name: str
    coverage: str
    cost_profile: str
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    recommended_use: str


LOW_COST_DATA_SOURCES: tuple[DataSourceCandidate, ...] = (
    DataSourceCandidate(
        name="football-data.co.uk",
        coverage="Historical results and closing odds for major European leagues.",
        cost_profile="Free",
        strengths=("Long history", "Easy CSV ingestion", "Good for MVP backtests"),
        weaknesses=("Limited event-level data", "Bookmaker coverage varies"),
        recommended_use="Initial historical match and odds baseline.",
    ),
    DataSourceCandidate(
        name="Kaggle football datasets",
        coverage="Mixed historical datasets depending on author and season.",
        cost_profile="Free",
        strengths=("Useful for prototyping", "Often includes team statistics"),
        weaknesses=("Quality varies", "Licensing and freshness must be checked"),
        recommended_use="Feature experiments and offline research.",
    ),
    DataSourceCandidate(
        name="The Odds API",
        coverage="Upcoming fixtures and odds across bookmakers.",
        cost_profile="Low-cost API tier",
        strengths=("Practical current odds feed", "Simple integration"),
        weaknesses=("Historical depth depends on plan", "Rate limits"),
        recommended_use="Daily opportunity scanner after MVP validation.",
    ),
    DataSourceCandidate(
        name="API-Football",
        coverage="Fixtures, lineups, events, standings, injuries, and live match state.",
        cost_profile="Freemium to paid",
        strengths=("Wide football coverage", "Live endpoints available"),
        weaknesses=("Premium tiers needed for serious scale", "Provider mapping required"),
        recommended_use="Live layer once pre-match edge is validated.",
    ),
)


class HistoricalMatchProvider(Protocol):
    def load_matches(self) -> list[MatchSnapshot]:
        ...

    def load_results(self) -> list[Result]:
        ...


class OddsProvider(Protocol):
    def load_quotes(self) -> list[MarketQuote]:
        ...


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class CsvHistoricalProvider:
    """Simple MVP provider for curated CSVs with normalized columns."""

    def __init__(self, matches_path: Path, results_path: Path, quotes_path: Path) -> None:
        self.matches_path = matches_path
        self.results_path = results_path
        self.quotes_path = quotes_path

    def load_matches(self) -> list[MatchSnapshot]:
        rows = _read_csv(self.matches_path)
        matches: list[MatchSnapshot] = []
        for row in rows:
            matches.append(
                MatchSnapshot(
                    match_id=row["match_id"],
                    kickoff=parse_datetime(row["kickoff"]),
                    competition=row["competition"],
                    home=TeamSnapshot(
                        team_id=row["home_team_id"],
                        name=row["home_name"],
                        elo=float(row["home_elo"]),
                        recent_goal_diff=float(row["home_recent_goal_diff"]),
                        rest_days=int(row["home_rest_days"]),
                        injuries_index=float(row.get("home_injuries_index", 0.0) or 0.0),
                    ),
                    away=TeamSnapshot(
                        team_id=row["away_team_id"],
                        name=row["away_name"],
                        elo=float(row["away_elo"]),
                        recent_goal_diff=float(row["away_recent_goal_diff"]),
                        rest_days=int(row["away_rest_days"]),
                        injuries_index=float(row.get("away_injuries_index", 0.0) or 0.0),
                    ),
                    neutral_venue=_parse_bool(row.get("neutral_venue", "false")),
                )
            )
        return matches

    def load_results(self) -> list[Result]:
        return [
            Result(
                match_id=row["match_id"],
                home_goals=int(row["home_goals"]),
                away_goals=int(row["away_goals"]),
            )
            for row in _read_csv(self.results_path)
        ]

    def load_quotes(self) -> list[MarketQuote]:
        return [
            MarketQuote(
                match_id=row["match_id"],
                bookmaker=row["bookmaker"],
                side=row["side"],  # type: ignore[arg-type]
                line=float(row["line"]),
                odds=float(row["odds"]),
                captured_at=parse_datetime(row["captured_at"]),
                is_closing=_parse_bool(row.get("is_closing", "false")),
            )
            for row in _read_csv(self.quotes_path)
        ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}
