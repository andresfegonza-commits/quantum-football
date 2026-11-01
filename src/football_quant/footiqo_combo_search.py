from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

from .footiqo import FootiqoMatch, _market_won, load_footiqo_matches
from .footiqo_accumulators import (
    AccumulatorLeg,
    AccumulatorSlip,
    _card_date,
    _card_season,
    accumulator_report,
)
from .footiqo_btts import LOCKED_BTTS_LEAGUES, parse_footiqo_date
from .footiqo_btts_refine import DISCOVERY_SEASONS, HOLDOUT_SEASONS


@dataclass(frozen=True)
class LegSpec:
    market: str
    min_odds: float
    max_odds: float


@dataclass(frozen=True)
class ComboFamily:
    name: str
    specs: tuple[LegSpec, ...]
    min_legs: int
    max_legs: int
    min_combined_odds: float
    require_all_markets: bool = False


@dataclass(frozen=True)
class ComboSlice:
    slips: int
    profit: float
    roi: float
    hit_rate: float
    avg_odds: float
    avg_legs: float


@dataclass(frozen=True)
class ComboReview:
    universe: str
    family: str
    discovery: ComboSlice
    holdout: ComboSlice


COMBO_FAMILIES: tuple[ComboFamily, ...] = (
    ComboFamily("fav_d2", (LegSpec("FAV", 1.20, 1.50),), 2, 2, 1.44),
    ComboFamily("fav_d2_wide", (LegSpec("FAV", 1.20, 1.70),), 2, 2, 1.44),
    ComboFamily("fav_t3", (LegSpec("FAV", 1.15, 1.40),), 3, 3, 1.52),
    ComboFamily("home_d2", (LegSpec("H", 1.30, 1.70),), 2, 2, 1.69),
    ComboFamily("u25_d2", (LegSpec("U25", 1.55, 2.00),), 2, 2, 2.40),
    ComboFamily("u25_d2_short", (LegSpec("U25", 1.40, 1.70),), 2, 2, 1.96),
    ComboFamily("bttsn_d2_mid", (LegSpec("BTTSN", 1.50, 1.90),), 2, 2, 2.25),
    ComboFamily("bttsn_d2_locked", (LegSpec("BTTSN", 1.90, 2.40),), 2, 2, 3.61),
    ComboFamily("bttsy_d2", (LegSpec("BTTSY", 1.50, 1.90),), 2, 2, 2.25),
    ComboFamily("o25_d2", (LegSpec("O25", 1.55, 2.00),), 2, 2, 2.40),
    ComboFamily(
        "mix_fav_u25",
        (LegSpec("FAV", 1.20, 1.50), LegSpec("U25", 1.50, 1.90)),
        2,
        2,
        1.80,
        True,
    ),
    ComboFamily(
        "mix_fav_bttsn",
        (LegSpec("FAV", 1.20, 1.50), LegSpec("BTTSN", 1.50, 2.00)),
        2,
        2,
        1.80,
        True,
    ),
    ComboFamily(
        "mix_u25_bttsn",
        (LegSpec("U25", 1.50, 2.00), LegSpec("BTTSN", 1.50, 2.00)),
        2,
        2,
        2.25,
        True,
    ),
    ComboFamily(
        "mix_fav_o15",
        (LegSpec("FAV", 1.20, 1.50), LegSpec("O15", 1.15, 1.35)),
        2,
        2,
        1.40,
        True,
    ),
)
COMBO_UNIVERSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ucl_el", LOCKED_BTTS_LEAGUES),
    ("ucl", ("Champions League",)),
    ("el", ("Europa League",)),
    ("conf", ("Conference League",)),
)


def load_combo_matches(paths: tuple[Path, ...], leagues: tuple[str, ...] | None = None) -> tuple[FootiqoMatch, ...]:
    matches = tuple(match for path in paths for match in load_footiqo_matches(path))
    if leagues is None:
        return matches
    allowed = set(leagues)
    return tuple(match for match in matches if match.league in allowed)


def build_family_slip(matches: tuple[FootiqoMatch, ...], family: ComboFamily) -> AccumulatorSlip | None:
    if family.require_all_markets:
        return _mix_slip(matches, family)
    return _greedy_slip(matches, family)


def backtest_combo_family(
    matches: tuple[FootiqoMatch, ...],
    family: ComboFamily,
) -> tuple[AccumulatorSlip, ...]:
    ordered = sorted(matches, key=lambda match: (parse_footiqo_date(match.match_date), match.match_id))
    slips: list[AccumulatorSlip] = []
    for _day, group in groupby(ordered, key=_card_date):
        slip = build_family_slip(tuple(group), family)
        if slip is not None:
            slips.append(slip)
    return tuple(slips)


def review_combo_families(
    matches: tuple[FootiqoMatch, ...],
    *,
    families: tuple[ComboFamily, ...] = COMBO_FAMILIES,
    universes: tuple[tuple[str, tuple[str, ...]], ...] = COMBO_UNIVERSES,
) -> tuple[ComboReview, ...]:
    reviews: list[ComboReview] = []
    for universe, leagues in universes:
        selected = tuple(match for match in matches if match.league in leagues)
        for family in families:
            reviews.append(
                ComboReview(
                    universe=universe,
                    family=family.name,
                    discovery=_slice(backtest_combo_family(_season_matches(selected, DISCOVERY_SEASONS), family)),
                    holdout=_slice(backtest_combo_family(_season_matches(selected, HOLDOUT_SEASONS), family)),
                )
            )
    return tuple(reviews)


def format_combo_search(
    matches: tuple[FootiqoMatch, ...],
    reviews: tuple[ComboReview, ...],
) -> str:
    lines = [
        "UEFA Combo Search",
        "-----------------",
        "Same-night slips, one market per match. Discovery 2015/16-2020/21, holdout 2021/22-2026/27.",
        "O05/O15-only accumulators are excluded; they already failed walk-forward.",
        "",
        _format_card_inventory(matches),
        "",
    ]
    for universe, _leagues in COMBO_UNIVERSES:
        lines.append(universe)
        lines.append(
            f"{'family':<20} {'disc_n':>7} {'disc_roi':>9} {'disc_hit':>8} {'disc_odds':>9} "
            f"{'hold_n':>7} {'hold_roi':>9} {'hold_hit':>8} {'hold_odds':>9}"
        )
        for review in reviews:
            if review.universe != universe:
                continue
            lines.append(
                f"{review.family:<20} {review.discovery.slips:7d} {review.discovery.roi:+9.2%} "
                f"{review.discovery.hit_rate:8.1%} {review.discovery.avg_odds:9.2f} "
                f"{review.holdout.slips:7d} {review.holdout.roi:+9.2%} "
                f"{review.holdout.hit_rate:8.1%} {review.holdout.avg_odds:9.2f}"
            )
        lines.append("")

    survivors = [
        review
        for review in reviews
        if review.universe == "ucl_el"
        and review.discovery.slips >= 40
        and review.discovery.roi > 0
        and review.holdout.slips >= 30
        and review.holdout.roi > 0
    ]
    lines.append("Survivors (green on UCL+EL discovery and holdout):")
    if not survivors:
        lines.append("  none")
    for review in survivors:
        conference = next(
            (item for item in reviews if item.universe == "conf" and item.family == review.family),
            None,
        )
        conf_text = "n/a"
        if conference is not None:
            conf_text = f"{conference.holdout.roi:+.2%} hold / {conference.discovery.roi:+.2%} disc ({conference.holdout.slips + conference.discovery.slips} slips)"
        lines.append(
            f"  {review.family}: disc {review.discovery.roi:+.2%} ({review.discovery.slips}), "
            f"hold {review.holdout.roi:+.2%} ({review.holdout.slips}), Conference {conf_text}"
        )
    return "\n".join(lines)


def _greedy_slip(matches: tuple[FootiqoMatch, ...], family: ComboFamily) -> AccumulatorSlip | None:
    candidates = sorted(
        [leg for match in matches for leg in _match_legs(match, family.specs)],
        key=lambda leg: (leg.odds, leg.match_id),
    )
    selected: list[AccumulatorLeg] = []
    combined_odds = 1.0
    combined_probability = 1.0
    used: set[str] = set()
    for leg in candidates:
        if leg.match_id in used or len(selected) >= family.max_legs:
            continue
        selected.append(leg)
        used.add(leg.match_id)
        combined_odds *= leg.odds
        combined_probability *= leg.probability
        if len(selected) >= family.min_legs and combined_odds >= family.min_combined_odds:
            break
    return _finish_slip(matches, family, selected, combined_odds, combined_probability)


def _mix_slip(matches: tuple[FootiqoMatch, ...], family: ComboFamily) -> AccumulatorSlip | None:
    if len(family.specs) != 2:
        raise ValueError(f"{family.name} mix families must declare exactly two markets")
    first_legs = [leg for match in matches for leg in _match_legs(match, (family.specs[0],))]
    second_legs = [leg for match in matches for leg in _match_legs(match, (family.specs[1],))]
    best: tuple[AccumulatorLeg, AccumulatorLeg] | None = None
    best_score = 0.0
    for first in first_legs:
        for second in second_legs:
            if first.match_id == second.match_id:
                continue
            combined_odds = first.odds * second.odds
            if combined_odds < family.min_combined_odds:
                continue
            score = first.probability * second.probability
            if best is None or score > best_score or (score == best_score and combined_odds < best[0].odds * best[1].odds):
                best = (first, second)
                best_score = score
    if best is None:
        return None
    selected = list(best)
    combined_odds = selected[0].odds * selected[1].odds
    combined_probability = selected[0].probability * selected[1].probability
    return _finish_slip(matches, family, selected, combined_odds, combined_probability)


def _finish_slip(
    matches: tuple[FootiqoMatch, ...],
    family: ComboFamily,
    selected: list[AccumulatorLeg],
    combined_odds: float,
    combined_probability: float,
) -> AccumulatorSlip | None:
    if len(selected) < family.min_legs or combined_odds < family.min_combined_odds:
        return None
    won = all(leg.won for leg in selected)
    return AccumulatorSlip(
        card_date=_card_date(matches[0]),
        season=_card_season(matches),
        legs=tuple(selected),
        combined_odds=combined_odds,
        combined_probability=combined_probability,
        expected_value=combined_probability * (combined_odds - 1.0) - (1.0 - combined_probability),
        won=won,
        profit=combined_odds - 1.0 if won else -1.0,
    )


def _match_legs(match: FootiqoMatch, specs: tuple[LegSpec, ...]) -> list[AccumulatorLeg]:
    legs: list[AccumulatorLeg] = []
    for spec in specs:
        market, odds = _resolved_quote(match, spec.market)
        if market is None or odds < spec.min_odds or odds > spec.max_odds:
            continue
        legs.append(
            AccumulatorLeg(
                match_id=match.match_id,
                match_date=match.match_date,
                season=match.season,
                league=match.league,
                market=market,
                odds=odds,
                probability=1.0 / odds,
                won=_selection_won(match, market),
            )
        )
    return legs


def _resolved_quote(match: FootiqoMatch, market: str) -> tuple[str | None, float]:
    if market == "FAV":
        home = match.odds.get("H", 0.0)
        away = match.odds.get("A", 0.0)
        if home <= 1.0 or away <= 1.0:
            return None, 0.0
        if home <= away:
            return "H", home
        return "A", away
    odds = match.odds.get(market, 0.0)
    if odds <= 1.0:
        return None, 0.0
    return market, odds


def _selection_won(match: FootiqoMatch, market: str) -> bool:
    if market == "H":
        return match.home_goals > match.away_goals
    if market == "D":
        return match.home_goals == match.away_goals
    if market == "A":
        return match.away_goals > match.home_goals
    return _market_won(match, market)


def _season_matches(matches: tuple[FootiqoMatch, ...], seasons: tuple[str, ...]) -> tuple[FootiqoMatch, ...]:
    allowed = set(seasons)
    return tuple(match for match in matches if match.season in allowed)


def _slice(slips: tuple[AccumulatorSlip, ...]) -> ComboSlice:
    if not slips:
        return ComboSlice(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    report = accumulator_report(slips)
    return ComboSlice(
        slips=len(slips),
        profit=report.total_profit,
        roi=report.roi,
        hit_rate=report.hit_rate,
        avg_odds=sum(slip.combined_odds for slip in slips) / len(slips),
        avg_legs=sum(len(slip.legs) for slip in slips) / len(slips),
    )


def _format_card_inventory(matches: tuple[FootiqoMatch, ...]) -> str:
    eligible = tuple(match for match in matches if match.league in LOCKED_BTTS_LEAGUES)
    lines = ["Card inventory UCL+EL"]
    for label, seasons in (("discovery", DISCOVERY_SEASONS), ("holdout", HOLDOUT_SEASONS)):
        selected = _season_matches(eligible, seasons)
        cards: dict[object, int] = defaultdict(int)
        for match in selected:
            cards[_card_date(match)] += 1
        sizes = tuple(cards.values())
        multi = sum(1 for size in sizes if size >= 2)
        lines.append(
            f"{label}: matches={len(selected)}, cards={len(sizes)}, "
            f"cards_2plus={multi}, avg_card={sum(sizes) / len(sizes) if sizes else 0:.1f}"
        )
    return "\n".join(lines)
