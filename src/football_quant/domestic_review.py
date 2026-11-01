from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .football_data_download import DEFAULT_DIVISIONS


DISCOVERY_SEASONS: tuple[str, ...] = ("1920", "2021", "2122", "2223")
HOLDOUT_SEASONS: tuple[str, ...] = ("2324", "2425", "2526", "2627")


@dataclass(frozen=True)
class DomesticSlice:
    bets: int
    profit: float
    roi: float
    hit_rate: float


@dataclass(frozen=True)
class DomesticReview:
    league: str
    family: str
    discovery: DomesticSlice
    holdout: DomesticSlice


def review_domestic_leagues(data_root: Path) -> tuple[DomesticReview, ...]:
    reviews: list[DomesticReview] = []
    for division, league in DEFAULT_DIVISIONS:
        rows = _load_rows(data_root, division)
        for family in ("fav_120_170", "home_fav_120_170", "over_25", "under_25"):
            reviews.append(
                DomesticReview(
                    league=league,
                    family=family,
                    discovery=_slice(_season_rows(rows, DISCOVERY_SEASONS), family),
                    holdout=_slice(_season_rows(rows, HOLDOUT_SEASONS), family),
                )
            )
    return tuple(reviews)


def format_domestic_review(reviews: tuple[DomesticReview, ...]) -> str:
    lines = [
        "Domestic League Review",
        "----------------------",
        "Discovery 2019/20-2022/23. Holdout 2023/24-2026/27.",
        "These are flat market slices, not locked rules.",
        "",
    ]
    for league in (name for _code, name in DEFAULT_DIVISIONS):
        lines.append(league)
        lines.append(
            f"{'family':<18} {'disc_n':>7} {'disc_roi':>9} {'disc_hit':>8} "
            f"{'hold_n':>7} {'hold_roi':>9} {'hold_hit':>8}"
        )
        for review in reviews:
            if review.league != league:
                continue
            lines.append(
                f"{review.family:<18} {review.discovery.bets:7d} {review.discovery.roi:+9.2%} "
                f"{review.discovery.hit_rate:8.1%} {review.holdout.bets:7d} "
                f"{review.holdout.roi:+9.2%} {review.holdout.hit_rate:8.1%}"
            )
        lines.append("")

    survivors = [
        review
        for review in reviews
        if review.discovery.bets >= 80
        and review.discovery.roi > 0
        and review.holdout.bets >= 40
        and review.holdout.roi > 0
    ]
    lines.append("Survivors (green on discovery and holdout):")
    if not survivors:
        lines.append("  none")
    for review in survivors:
        lines.append(
            f"  {review.league} {review.family}: "
            f"disc {review.discovery.roi:+.2%} ({review.discovery.bets}), "
            f"hold {review.holdout.roi:+.2%} ({review.holdout.bets})"
        )
    return "\n".join(lines)


def _load_rows(data_root: Path, division: str) -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for season in DISCOVERY_SEASONS + HOLDOUT_SEASONS:
        path = data_root / season / f"{division}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if not row.get("HomeTeam") or not row.get("AwayTeam"):
                    continue
                row["_season"] = season
                rows.append(row)
    return tuple(rows)


def _season_rows(rows: tuple[dict[str, str], ...], seasons: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    allowed = set(seasons)
    return tuple(row for row in rows if row.get("_season") in allowed)


def _slice(rows: tuple[dict[str, str], ...], family: str) -> DomesticSlice:
    profits: list[float] = []
    for row in rows:
        profit = _family_profit(row, family)
        if profit is not None:
            profits.append(profit)
    if not profits:
        return DomesticSlice(0, 0.0, 0.0, 0.0)
    total = sum(profits)
    wins = sum(1 for profit in profits if profit > 0)
    return DomesticSlice(len(profits), total, total / len(profits), wins / len(profits))


def _family_profit(row: dict[str, str], family: str) -> float | None:
    home_odds = _odds(row, "AvgH", "B365H")
    away_odds = _odds(row, "AvgA", "B365A")
    over_odds = _odds(row, "Avg>2.5", "B365>2.5")
    under_odds = _odds(row, "Avg<2.5", "B365<2.5")
    result = row.get("FTR", "")
    try:
        total_goals = int(row["FTHG"]) + int(row["FTAG"])
    except (KeyError, ValueError):
        return None

    if family == "fav_120_170":
        if home_odds < 1.20 or away_odds <= 1.0:
            return None
        if home_odds <= away_odds:
            if home_odds > 1.70:
                return None
            return home_odds - 1.0 if result == "H" else -1.0
        if away_odds < 1.20 or away_odds > 1.70:
            return None
        return away_odds - 1.0 if result == "A" else -1.0

    if family == "home_fav_120_170":
        if home_odds < 1.20 or home_odds > 1.70 or away_odds <= 1.0 or home_odds > away_odds:
            return None
        return home_odds - 1.0 if result == "H" else -1.0

    if family == "over_25":
        if over_odds <= 1.0:
            return None
        return over_odds - 1.0 if total_goals > 2.5 else -1.0

    if family == "under_25":
        if under_odds <= 1.0:
            return None
        return under_odds - 1.0 if total_goals < 2.5 else -1.0

    raise ValueError(f"Unknown family: {family}")


def _odds(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        raw = row.get(key, "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 1.0:
            return value
    return 0.0
