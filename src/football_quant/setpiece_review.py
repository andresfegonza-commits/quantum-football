from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .domestic_review import (
    DISCOVERY_SEASONS,
    HOLDOUT_SEASONS,
    DomesticReview,
    DomesticSlice,
    _load_rows,
    _odds,
    _season_rows,
)
from .football_data_download import DEFAULT_DIVISIONS


PROXY_ODDS = 1.85


@dataclass(frozen=True)
class LeagueBaseRate:
    league: str
    matches: int
    mean_corners: float
    mean_yellows: float
    over_c95: float
    over_c105: float
    over_y35: float
    over_y45: float


def review_setpiece_markets(data_root: Path) -> tuple[DomesticReview, ...]:
    reviews: list[DomesticReview] = []
    for division, league in DEFAULT_DIVISIONS:
        rows = _load_rows(data_root, division)
        for family in _FAMILY_NAMES:
            reviews.append(
                DomesticReview(
                    league=league,
                    family=family,
                    discovery=_slice(_season_rows(rows, DISCOVERY_SEASONS), family),
                    holdout=_slice(_season_rows(rows, HOLDOUT_SEASONS), family),
                )
            )
    return tuple(reviews)


def league_base_rates(data_root: Path, seasons: tuple[str, ...]) -> tuple[LeagueBaseRate, ...]:
    rates: list[LeagueBaseRate] = []
    for division, league in DEFAULT_DIVISIONS:
        rows = _season_rows(_load_rows(data_root, division), seasons)
        corners: list[int] = []
        yellows: list[int] = []
        for row in rows:
            totals = _totals(row)
            if totals is None:
                continue
            corners.append(totals[0])
            yellows.append(totals[1])
        if not corners:
            rates.append(LeagueBaseRate(league, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        n = len(corners)
        rates.append(
            LeagueBaseRate(
                league=league,
                matches=n,
                mean_corners=sum(corners) / n,
                mean_yellows=sum(yellows) / n,
                over_c95=sum(1 for value in corners if value > 9.5) / n,
                over_c105=sum(1 for value in corners if value > 10.5) / n,
                over_y35=sum(1 for value in yellows if value > 3.5) / n,
                over_y45=sum(1 for value in yellows if value > 4.5) / n,
            )
        )
    return tuple(rates)


def format_setpiece_review(reviews: tuple[DomesticReview, ...], data_root: Path) -> str:
    lines = [
        "Pre-match Corners & Yellow Cards Review",
        "---------------------------------------",
        "Discovery 2019/20-2022/23. Holdout 2023/24-2026/27.",
        f"ROI uses proxy odds {PROXY_ODDS:.2f} (no historical corner/card books in football-data).",
        "Break-even hit rate at 1.85 is 54.1%. These are not locked rules.",
        "",
        "League base rates (all seasons on disk):",
    ]
    all_seasons = DISCOVERY_SEASONS + HOLDOUT_SEASONS
    lines.append(
        f"{'league':<16} {'n':>5} {'avgC':>6} {'O9.5':>7} {'O10.5':>7} "
        f"{'avgY':>6} {'O3.5':>7} {'O4.5':>7}"
    )
    for rate in league_base_rates(data_root, all_seasons):
        lines.append(
            f"{rate.league:<16} {rate.matches:5d} {rate.mean_corners:6.2f} "
            f"{rate.over_c95:7.1%} {rate.over_c105:7.1%} {rate.mean_yellows:6.2f} "
            f"{rate.over_y35:7.1%} {rate.over_y45:7.1%}"
        )
    lines.append("")

    for league in (name for _code, name in DEFAULT_DIVISIONS):
        lines.append(league)
        lines.append(
            f"{'family':<22} {'disc_n':>7} {'disc_roi':>9} {'disc_hit':>8} "
            f"{'hold_n':>7} {'hold_roi':>9} {'hold_hit':>8}"
        )
        for review in reviews:
            if review.league != league:
                continue
            lines.append(
                f"{review.family:<22} {review.discovery.bets:7d} {review.discovery.roi:+9.2%} "
                f"{review.discovery.hit_rate:8.1%} {review.holdout.bets:7d} "
                f"{review.holdout.roi:+9.2%} {review.holdout.hit_rate:8.1%}"
            )
        lines.append("")

    survivors = [
        review
        for review in reviews
        if review.discovery.bets >= 80
        and review.discovery.roi >= 0.02
        and review.holdout.bets >= 40
        and review.holdout.roi >= 0.02
    ]
    lines.append("Survivors (proxy 1.85, +2% ROI on discovery and holdout):")
    if not survivors:
        lines.append("  none")
    for review in survivors:
        lines.append(
            f"  {review.league} {review.family}: "
            f"disc {review.discovery.roi:+.2%} ({review.discovery.bets}), "
            f"hold {review.holdout.roi:+.2%} ({review.holdout.bets})"
        )
    return "\n".join(lines)


_FAMILY_NAMES: tuple[str, ...] = (
    "over_c95",
    "under_c95",
    "over_c105",
    "under_c105",
    "over_c95_open",
    "under_c95_closed",
    "over_c95_even",
    "under_c95_short_fav",
    "over_y35",
    "under_y35",
    "over_y45",
    "under_y45",
    "over_y35_even",
    "under_y45_short_fav",
)


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
    totals = _totals(row)
    if totals is None:
        return None
    corners, yellows = totals
    if not _filter_allows(row, family):
        return None
    if family.startswith("over_c"):
        won = corners > _corner_line(family)
    elif family.startswith("under_c"):
        won = corners < _corner_line(family)
    elif family.startswith("over_y"):
        won = yellows > _yellow_line(family)
    elif family.startswith("under_y"):
        won = yellows < _yellow_line(family)
    else:
        raise ValueError(f"Unknown family: {family}")
    return PROXY_ODDS - 1.0 if won else -1.0


def _filter_allows(row: dict[str, str], family: str) -> bool:
    home_odds = _odds(row, "AvgH", "B365H")
    away_odds = _odds(row, "AvgA", "B365A")
    over_25 = _odds(row, "Avg>2.5", "B365>2.5")
    if home_odds <= 1.0 or away_odds <= 1.0:
        return False
    fav = min(home_odds, away_odds)
    if family.endswith("_open"):
        return 1.0 < over_25 <= 1.80
    if family.endswith("_closed"):
        return over_25 >= 2.10
    if family.endswith("_even"):
        return fav >= 1.80
    if family.endswith("_short_fav"):
        return 1.20 <= fav <= 1.45
    return True


def _corner_line(family: str) -> float:
    return 10.5 if "c105" in family else 9.5


def _yellow_line(family: str) -> float:
    return 4.5 if "y45" in family else 3.5


def _totals(row: dict[str, str]) -> tuple[int, int] | None:
    try:
        corners = int(row["HC"]) + int(row["AC"])
        yellows = int(row["HY"]) + int(row["AY"])
    except (KeyError, ValueError):
        return None
    return corners, yellows
