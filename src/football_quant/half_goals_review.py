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


PROXY_ODDS: dict[str, float] = {
    "1h_over_05": 1.44,
    "1h_over_15": 2.80,
    "2h_over_05": 1.30,
    "2h_over_15": 2.10,
    "both_halves_over_05": 1.85,
    "ht00_2h_over_05": 1.30,
    "ht00_2h_over_15": 2.10,
    "open_1h_over_05": 1.44,
    "open_2h_over_15": 2.10,
}

_FAMILY_NAMES: tuple[str, ...] = tuple(PROXY_ODDS)


@dataclass(frozen=True)
class HalfGoalBaseRate:
    league: str
    matches: int
    mean_first_half: float
    mean_second_half: float
    second_half_share: float
    over_1h_05: float
    over_1h_15: float
    over_2h_05: float
    over_2h_15: float
    both_halves: float


def review_half_goal_markets(data_root: Path) -> tuple[DomesticReview, ...]:
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


def league_half_goal_rates(data_root: Path, seasons: tuple[str, ...]) -> tuple[HalfGoalBaseRate, ...]:
    rates: list[HalfGoalBaseRate] = []
    for division, league in DEFAULT_DIVISIONS:
        rows = [_halves(row) for row in _season_rows(_load_rows(data_root, division), seasons)]
        halves = [item for item in rows if item is not None]
        if not halves:
            rates.append(HalfGoalBaseRate(league, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        n = len(halves)
        first = sum(item[0] for item in halves)
        second = sum(item[1] for item in halves)
        total = first + second
        rates.append(
            HalfGoalBaseRate(
                league=league,
                matches=n,
                mean_first_half=first / n,
                mean_second_half=second / n,
                second_half_share=(second / total) if total else 0.0,
                over_1h_05=sum(1 for item in halves if item[0] > 0.5) / n,
                over_1h_15=sum(1 for item in halves if item[0] > 1.5) / n,
                over_2h_05=sum(1 for item in halves if item[1] > 0.5) / n,
                over_2h_15=sum(1 for item in halves if item[1] > 1.5) / n,
                both_halves=sum(1 for item in halves if item[0] > 0.5 and item[1] > 0.5) / n,
            )
        )
    return tuple(rates)


def format_half_goals_review(reviews: tuple[DomesticReview, ...], data_root: Path) -> str:
    lines = [
        "First / Second Half Goals Review",
        "--------------------------------",
        "Discovery 2019/20-2022/23. Holdout 2023/24-2026/27.",
        "football-data has HTHG/HTAG but no 1H/2H book odds. ROI uses conservative proxy prices.",
        "These are not locked rules. Only take a family if the live price is at least the proxy.",
        "",
        "Proxy odds and break-even hit rates:",
        "  1h_over_05 / open_1h_over_05 @ 1.44 (69.4%)",
        "  1h_over_15 @ 2.80 (35.7%)",
        "  2h_over_05 / ht00_2h_over_05 @ 1.30 (76.9%)",
        "  2h_over_15 / open_2h_over_15 / ht00_2h_over_15 @ 2.10 (47.6%)",
        "  both_halves_over_05 @ 1.85 (54.1%)",
        "  open_* requires pre-match Over 2.5 odds <= 1.70",
        "  ht00_* requires 0-0 at half-time",
        "",
        "League base rates (all seasons on disk):",
    ]
    lines.append(
        f"{'league':<16} {'n':>5} {'1H':>5} {'2H':>5} {'2H%':>6} "
        f"{'1HO05':>6} {'1HO15':>6} {'2HO05':>6} {'2HO15':>6} {'both':>6}"
    )
    for rate in league_half_goal_rates(data_root, DISCOVERY_SEASONS + HOLDOUT_SEASONS):
        lines.append(
            f"{rate.league:<16} {rate.matches:5d} {rate.mean_first_half:5.2f} "
            f"{rate.mean_second_half:5.2f} {rate.second_half_share:6.1%} "
            f"{rate.over_1h_05:6.1%} {rate.over_1h_15:6.1%} "
            f"{rate.over_2h_05:6.1%} {rate.over_2h_15:6.1%} {rate.both_halves:6.1%}"
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
    lines.append("Survivors (proxy price, +2% ROI on discovery and holdout):")
    if not survivors:
        lines.append("  none")
    for review in survivors:
        lines.append(
            f"  {review.league} {review.family} @{PROXY_ODDS[review.family]:.2f}: "
            f"disc {review.discovery.roi:+.2%} ({review.discovery.bets}), "
            f"hold {review.holdout.roi:+.2%} ({review.holdout.bets})"
        )
    return "\n".join(lines)


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
    halves = _halves(row)
    if halves is None:
        return None
    first, second, _total = halves
    if family.startswith("open_") and not _is_open_match(row):
        return None
    if family.startswith("ht00_") and first != 0:
        return None
    if family in {"1h_over_05", "open_1h_over_05"}:
        won = first > 0.5
    elif family == "1h_over_15":
        won = first > 1.5
    elif family in {"2h_over_05", "ht00_2h_over_05"}:
        won = second > 0.5
    elif family in {"2h_over_15", "open_2h_over_15", "ht00_2h_over_15"}:
        won = second > 1.5
    elif family == "both_halves_over_05":
        won = first > 0.5 and second > 0.5
    else:
        raise ValueError(f"Unknown family: {family}")
    return PROXY_ODDS[family] - 1.0 if won else -1.0


def _is_open_match(row: dict[str, str]) -> bool:
    over_25 = _odds(row, "Avg>2.5", "B365>2.5")
    return 1.0 < over_25 <= 1.70


def _halves(row: dict[str, str]) -> tuple[int, int, int] | None:
    try:
        first = int(row["HTHG"]) + int(row["HTAG"])
        total = int(row["FTHG"]) + int(row["FTAG"])
    except (KeyError, ValueError):
        return None
    if total < first:
        return None
    return first, total - first, total
