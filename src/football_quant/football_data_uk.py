from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .dataset import HistoricalDataset
from .domain import MarketQuote, MatchSnapshot, Result, TeamSnapshot


@dataclass
class TeamState:
    elo: float = 1500.0
    recent_diffs: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    goals_for: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    goals_against: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    shots_for: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    shots_against: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    shots_on_target_for: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    shots_on_target_against: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    corners_for: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    corners_against: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    last_played: datetime | None = None

    @property
    def recent_goal_diff(self) -> float:
        if not self.recent_diffs:
            return 0.0
        return sum(self.recent_diffs) / len(self.recent_diffs)

    def average(self, metric: str) -> float:
        values = getattr(self, metric)
        if not values:
            return 0.0
        return sum(values) / len(values)


def load_football_data_uk_csv(
    path: Path,
    *,
    season: str,
    competition: str,
    bookmaker: str = "market-average",
) -> HistoricalDataset:
    rows = _read_rows(path)
    states: dict[str, TeamState] = {}
    matches: list[MatchSnapshot] = []
    results: list[Result] = []
    quotes: list[MarketQuote] = []

    for index, row in enumerate(rows, start=1):
        kickoff = _parse_kickoff(row["Date"], row.get("Time", "15:00"))
        home_name = row["HomeTeam"].strip()
        away_name = row["AwayTeam"].strip()
        home_state = states.setdefault(home_name, TeamState())
        away_state = states.setdefault(away_name, TeamState())
        match_id = f"{season}-{row['Div']}-{index:03d}"

        matches.append(
            MatchSnapshot(
                match_id=match_id,
                kickoff=kickoff,
                competition=competition,
                home=TeamSnapshot(
                    team_id=_team_id(home_name),
                    name=home_name,
                    elo=home_state.elo,
                    recent_goal_diff=home_state.recent_goal_diff,
                    rest_days=_rest_days(home_state, kickoff),
                    goals_for_avg=home_state.average("goals_for"),
                    goals_against_avg=home_state.average("goals_against"),
                    shots_for_avg=home_state.average("shots_for"),
                    shots_against_avg=home_state.average("shots_against"),
                    shots_on_target_for_avg=home_state.average("shots_on_target_for"),
                    shots_on_target_against_avg=home_state.average("shots_on_target_against"),
                    corners_for_avg=home_state.average("corners_for"),
                    corners_against_avg=home_state.average("corners_against"),
                ),
                away=TeamSnapshot(
                    team_id=_team_id(away_name),
                    name=away_name,
                    elo=away_state.elo,
                    recent_goal_diff=away_state.recent_goal_diff,
                    rest_days=_rest_days(away_state, kickoff),
                    goals_for_avg=away_state.average("goals_for"),
                    goals_against_avg=away_state.average("goals_against"),
                    shots_for_avg=away_state.average("shots_for"),
                    shots_against_avg=away_state.average("shots_against"),
                    shots_on_target_for_avg=away_state.average("shots_on_target_for"),
                    shots_on_target_against_avg=away_state.average("shots_on_target_against"),
                    corners_for_avg=away_state.average("corners_for"),
                    corners_against_avg=away_state.average("corners_against"),
                ),
            )
        )

        home_goals = int(row["FTHG"])
        away_goals = int(row["FTAG"])
        results.append(Result(match_id, home_goals, away_goals))
        quotes.extend(_asian_handicap_quotes(row, match_id, kickoff, bookmaker))
        _update_team_states(home_state, away_state, row, kickoff, home_goals, away_goals)

    return HistoricalDataset(matches=tuple(matches), results=tuple(results), quotes=tuple(quotes))


def load_football_data_uk_seasons(
    data_root: Path,
    *,
    season_codes: tuple[str, ...],
    division: str = "E0",
    competition: str = "Premier League",
) -> HistoricalDataset:
    datasets = []
    for code in season_codes:
        datasets.append(
            load_football_data_uk_csv(
                data_root / code / f"{division}.csv",
                season=season_label_from_code(code),
                competition=competition,
            )
        )

    return HistoricalDataset(
        matches=tuple(match for dataset in datasets for match in dataset.matches),
        results=tuple(result for dataset in datasets for result in dataset.results),
        quotes=tuple(quote for dataset in datasets for quote in dataset.quotes),
    )


def season_label_from_code(code: str) -> str:
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"Unsupported football-data season code: {code}")
    start_year = 2000 + int(code[:2])
    end_year = 2000 + int(code[2:])
    if end_year < start_year:
        end_year += 100
    return f"{start_year}-{end_year}"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [row for row in csv.DictReader(handle) if row.get("HomeTeam") and row.get("AwayTeam")]


def _parse_kickoff(date_value: str, time_value: str) -> datetime:
    value = f"{date_value.strip()} {time_value.strip() or '15:00'}"
    return datetime.strptime(value, "%d/%m/%Y %H:%M")


def _team_id(name: str) -> str:
    return name.lower().replace("'", "").replace(" ", "-")


def _rest_days(state: TeamState, kickoff: datetime) -> int:
    if state.last_played is None:
        return 7
    return max(0, (kickoff.date() - state.last_played.date()).days)


def _asian_handicap_quotes(
    row: dict[str, str],
    match_id: str,
    kickoff: datetime,
    bookmaker: str,
) -> list[MarketQuote]:
    quotes: list[MarketQuote] = []
    opening_line = _float_or_none(row.get("AHh"))
    closing_line = _float_or_none(row.get("AHCh"))

    if opening_line is not None:
        quotes.extend(
            _selection_quotes(
                match_id=match_id,
                bookmaker=bookmaker,
                home_line=opening_line,
                home_odds=_float_or_none(row.get("AvgAHH")),
                away_odds=_float_or_none(row.get("AvgAHA")),
                captured_at=kickoff - timedelta(hours=8),
                is_closing=False,
            )
        )

    if closing_line is not None:
        quotes.extend(
            _selection_quotes(
                match_id=match_id,
                bookmaker=bookmaker,
                home_line=closing_line,
                home_odds=_float_or_none(row.get("AvgCAHH")),
                away_odds=_float_or_none(row.get("AvgCAHA")),
                captured_at=kickoff - timedelta(minutes=10),
                is_closing=True,
            )
        )

    return quotes


def _selection_quotes(
    *,
    match_id: str,
    bookmaker: str,
    home_line: float,
    home_odds: float | None,
    away_odds: float | None,
    captured_at: datetime,
    is_closing: bool,
) -> list[MarketQuote]:
    quotes: list[MarketQuote] = []
    if home_odds is not None:
        quotes.append(MarketQuote(match_id, bookmaker, "home", home_line, home_odds, captured_at, is_closing))
    if away_odds is not None:
        quotes.append(MarketQuote(match_id, bookmaker, "away", -home_line, away_odds, captured_at, is_closing))
    return quotes


def _float_or_none(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _update_team_states(
    home_state: TeamState,
    away_state: TeamState,
    row: dict[str, str],
    kickoff: datetime,
    home_goals: int,
    away_goals: int,
) -> None:
    goal_diff = home_goals - away_goals
    expected_home = 1.0 / (1.0 + 10 ** ((away_state.elo - home_state.elo) / 400))
    actual_home = 1.0 if goal_diff > 0 else 0.5 if goal_diff == 0 else 0.0
    adjustment = 24 * (actual_home - expected_home)
    home_state.elo += adjustment
    away_state.elo -= adjustment
    home_state.recent_diffs.append(float(goal_diff))
    away_state.recent_diffs.append(float(-goal_diff))
    home_state.goals_for.append(float(home_goals))
    home_state.goals_against.append(float(away_goals))
    away_state.goals_for.append(float(away_goals))
    away_state.goals_against.append(float(home_goals))
    _append_pair(home_state.shots_for, away_state.shots_against, row.get("HS"))
    _append_pair(away_state.shots_for, home_state.shots_against, row.get("AS"))
    _append_pair(home_state.shots_on_target_for, away_state.shots_on_target_against, row.get("HST"))
    _append_pair(away_state.shots_on_target_for, home_state.shots_on_target_against, row.get("AST"))
    _append_pair(home_state.corners_for, away_state.corners_against, row.get("HC"))
    _append_pair(away_state.corners_for, home_state.corners_against, row.get("AC"))
    home_state.last_played = kickoff
    away_state.last_played = kickoff


def _append_pair(primary: deque[float], secondary: deque[float], value: str | None) -> None:
    parsed = _float_or_none(value)
    if parsed is None:
        return
    primary.append(parsed)
    secondary.append(parsed)
