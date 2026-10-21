from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .football_data_uk import load_football_data_uk_csv, season_label_from_code


GOAL_MARKETS: tuple[str, ...] = (
    "over_25",
    "btts",
    "first_half_over_05",
    "first_half_over_15",
    "second_half_over_05",
    "second_half_over_15",
)

GOAL_FEATURES: tuple[str, ...] = (
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_recent_goal_diff",
    "away_recent_goal_diff",
    "recent_goal_diff",
    "rest_diff",
    "home_goals_for_avg",
    "home_goals_against_avg",
    "away_goals_for_avg",
    "away_goals_against_avg",
    "goal_profile_diff",
    "shot_profile_diff",
    "target_profile_diff",
    "corner_profile_diff",
    "opening_over25_implied",
    "opening_under25_implied",
)


@dataclass(frozen=True)
class GoalMarketRow:
    season: str
    match_id: str
    features: dict[str, float]
    labels: dict[str, int]
    odds: dict[str, float]


def load_goal_market_rows_for_seasons(
    data_root: Path,
    *,
    season_codes: tuple[str, ...],
) -> tuple[GoalMarketRow, ...]:
    rows: list[GoalMarketRow] = []
    for code in season_codes:
        rows.extend(load_goal_market_rows_from_csv(data_root / code / "E0.csv", season_code=code))
    return tuple(rows)


def load_goal_market_rows_from_csv(path: Path, *, season_code: str) -> tuple[GoalMarketRow, ...]:
    season = season_label_from_code(season_code)
    raw_rows = _read_rows(path)
    dataset = load_football_data_uk_csv(path, season=season, competition="Premier League")
    goal_rows: list[GoalMarketRow] = []

    for raw, match, result in zip(raw_rows, dataset.matches, dataset.results, strict=True):
        total_goals = result.home_goals + result.away_goals
        first_half_goals = int(raw["HTHG"]) + int(raw["HTAG"])
        second_half_goals = total_goals - first_half_goals
        features = _features(raw, match)
        labels = {
            "over_25": int(total_goals > 2.5),
            "btts": int(result.home_goals > 0 and result.away_goals > 0),
            "first_half_over_05": int(first_half_goals > 0.5),
            "first_half_over_15": int(first_half_goals > 1.5),
            "second_half_over_05": int(second_half_goals > 0.5),
            "second_half_over_15": int(second_half_goals > 1.5),
        }
        odds = {
            "opening_over_25": _float_or_zero(raw.get("Avg>2.5")),
            "opening_under_25": _float_or_zero(raw.get("Avg<2.5")),
            "closing_over_25": _float_or_zero(raw.get("AvgC>2.5")),
            "closing_under_25": _float_or_zero(raw.get("AvgC<2.5")),
        }
        goal_rows.append(GoalMarketRow(season=season, match_id=match.match_id, features=features, labels=labels, odds=odds))

    return tuple(goal_rows)


def feature_matrix(rows: tuple[GoalMarketRow, ...], feature_names: tuple[str, ...] = GOAL_FEATURES) -> list[list[float]]:
    return [[row.features[name] for name in feature_names] for row in rows]


def labels(rows: tuple[GoalMarketRow, ...], market: str) -> list[int]:
    return [row.labels[market] for row in rows]


def _features(raw: dict[str, str], match) -> dict[str, float]:
    home = match.home
    away = match.away
    over_odds = _float_or_zero(raw.get("Avg>2.5"))
    under_odds = _float_or_zero(raw.get("Avg<2.5"))
    return {
        "home_elo": home.elo,
        "away_elo": away.elo,
        "elo_diff": home.elo - away.elo,
        "home_recent_goal_diff": home.recent_goal_diff,
        "away_recent_goal_diff": away.recent_goal_diff,
        "recent_goal_diff": home.recent_goal_diff - away.recent_goal_diff,
        "rest_diff": home.rest_days - away.rest_days,
        "home_goals_for_avg": home.goals_for_avg,
        "home_goals_against_avg": home.goals_against_avg,
        "away_goals_for_avg": away.goals_for_avg,
        "away_goals_against_avg": away.goals_against_avg,
        "goal_profile_diff": (
            home.goals_for_avg - home.goals_against_avg - away.goals_for_avg + away.goals_against_avg
        ),
        "shot_profile_diff": (
            home.shots_for_avg - home.shots_against_avg - away.shots_for_avg + away.shots_against_avg
        ),
        "target_profile_diff": (
            home.shots_on_target_for_avg
            - home.shots_on_target_against_avg
            - away.shots_on_target_for_avg
            + away.shots_on_target_against_avg
        ),
        "corner_profile_diff": (
            home.corners_for_avg - home.corners_against_avg - away.corners_for_avg + away.corners_against_avg
        ),
        "opening_over25_implied": 1.0 / over_odds if over_odds > 1.0 else 0.0,
        "opening_under25_implied": 1.0 / under_odds if under_odds > 1.0 else 0.0,
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [row for row in csv.DictReader(handle) if row.get("HomeTeam") and row.get("AwayTeam")]


def _float_or_zero(value: str | None) -> float:
    if value is None or value.strip() == "":
        return 0.0
    return float(value)
