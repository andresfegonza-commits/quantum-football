from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .footiqo_btts import (
    BttsNoFilter,
    flat_btts_no_backtest,
    load_btts_rows,
    run_fixed_btts_no_walk_forward,
    seasons_in_order,
)
from .goal_markets import GoalMarketRow
from .goal_value import ValueConfig, _aggregate


BASE_BLEND = ValueConfig(0.5, 0.02, False)
BASE_MODEL = ValueConfig(1.0, 0.02, False)
MODEL_NO_EV = ValueConfig(1.0, 0.0, False)
CONFIGS: tuple[tuple[str, ValueConfig], ...] = (
    ("blend_50_ev02", BASE_BLEND),
    ("model_100_ev02", BASE_MODEL),
    ("model_100_ev00", MODEL_NO_EV),
)
DISCOVERY_SEASONS: tuple[str, ...] = (
    "2015/2016",
    "2016/2017",
    "2017/2018",
    "2018/2019",
    "2019/2020",
    "2020/2021",
)
HOLDOUT_SEASONS: tuple[str, ...] = (
    "2021/2022",
    "2022/2023",
    "2023/2024",
    "2024/2025",
    "2025/2026",
    "2026/2027",
)
CANDIDATE_FILTERS: tuple[BttsNoFilter, ...] = (
    BttsNoFilter("base"),
    BttsNoFilter("history", require_history=True),
    BttsNoFilter("odds_190_plus", min_odds=1.90),
    BttsNoFilter("odds_210_plus", min_odds=2.10),
    BttsNoFilter("odds_190_240", min_odds=1.90, max_odds=2.40),
    BttsNoFilter("favorite_45", min_favorite_implied=0.45),
    BttsNoFilter("favorite_55", min_favorite_implied=0.55),
    BttsNoFilter("model_gt_market", model_above_market=True),
    BttsNoFilter("low_recent_btts", max_combined_btts_rate=0.50),
    BttsNoFilter("history_odds_190", require_history=True, min_odds=1.90),
    BttsNoFilter("history_model_edge", require_history=True, model_above_market=True),
    BttsNoFilter("favorite_45_odds_190", min_favorite_implied=0.45, min_odds=1.90),
    BttsNoFilter("odds_240_plus", min_odds=2.40),
    BttsNoFilter("odds_260_plus", min_odds=2.60),
    BttsNoFilter("odds_280_plus", min_odds=2.80),
    BttsNoFilter("odds_300_plus", min_odds=3.00),
    BttsNoFilter("odds_240_300", min_odds=2.40, max_odds=3.00),
    BttsNoFilter("odds_300_400", min_odds=3.00, max_odds=4.00),
    BttsNoFilter("odds_240_plus_fav55", min_odds=2.40, min_favorite_implied=0.55),
    BttsNoFilter("odds_240_plus_history", min_odds=2.40, require_history=True),
    BttsNoFilter("odds_240_plus_lowbtts", min_odds=2.40, max_combined_btts_rate=0.50),
)
FLAT_ODDS_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("flat_all", 1.01, 20.0),
    ("flat_190_240", 1.90, 2.40),
    ("flat_240_plus", 2.40, 20.0),
    ("flat_260_plus", 2.60, 20.0),
    ("flat_280_plus", 2.80, 20.0),
    ("flat_300_plus", 3.00, 20.0),
    ("flat_240_300", 2.40, 3.00),
    ("flat_300_400", 3.00, 4.00),
    ("flat_400_plus", 4.00, 20.0),
)


@dataclass(frozen=True)
class SliceResult:
    label: str
    bets: int
    profit: float
    roi: float
    hit_rate: float
    max_drawdown: float


@dataclass(frozen=True)
class RuleReview:
    config_name: str
    filter_name: str
    discovery: SliceResult
    holdout: SliceResult
    europa: SliceResult
    conference: SliceResult


def review_working_btts_rules(paths: tuple[Path, ...]) -> tuple[RuleReview, ...]:
    ucl = load_btts_rows(paths, league="Champions League")
    europa = load_btts_rows(paths, league="Europa League")
    conference = load_btts_rows(paths, league="Conference League")
    reviews: list[RuleReview] = []
    for config_name, config in CONFIGS:
        for filters in CANDIDATE_FILTERS:
            reviews.append(
                RuleReview(
                    config_name=config_name,
                    filter_name=filters.name,
                    discovery=_slice(ucl, config, filters, DISCOVERY_SEASONS),
                    holdout=_slice(ucl, config, filters, HOLDOUT_SEASONS, train_rows=_rows(ucl, DISCOVERY_SEASONS)),
                    europa=_external(europa, ucl, config, filters),
                    conference=_external(conference, ucl, config, filters),
                )
            )
    return tuple(reviews)


def format_btts_rule_review(reviews: tuple[RuleReview, ...]) -> str:
    lines = [
        "BTTS No Rule Review",
        "-------------------",
        "Discovery = UCL 2015/16-2020/21 walk-forward. Holdout = UCL 2021/22-2026/27.",
        "Europa/Conference use the UCL-trained expanding model, never their own labels for selection.",
        "",
    ]
    for config_name, _config in CONFIGS:
        lines.append(config_name)
        lines.append(
            f"{'filter':<22} {'disc_n':>7} {'disc_roi':>9} {'hold_n':>7} {'hold_roi':>9} "
            f"{'el_n':>6} {'el_roi':>8} {'conf_n':>7} {'conf_roi':>9}"
        )
        for review in reviews:
            if review.config_name != config_name:
                continue
            lines.append(
                f"{review.filter_name:<22} {review.discovery.bets:7d} {review.discovery.roi:+9.2%} "
                f"{review.holdout.bets:7d} {review.holdout.roi:+9.2%} "
                f"{review.europa.bets:6d} {review.europa.roi:+8.2%} "
                f"{review.conference.bets:7d} {review.conference.roi:+9.2%}"
            )
        lines.append("")
    survivors = [
        review
        for review in reviews
        if review.discovery.bets >= 80
        and review.discovery.roi > 0
        and review.holdout.bets >= 40
        and review.holdout.roi > 0
        and review.europa.bets >= 80
        and review.europa.roi > 0
    ]
    lines.append("Survivors (green on UCL discovery, UCL holdout and Europa):")
    if not survivors:
        lines.append("  none")
    for review in survivors:
        lines.append(
            f"  {review.config_name}/{review.filter_name}: "
            f"holdout {review.holdout.roi:+.2%} ({review.holdout.bets}), "
            f"Europa {review.europa.roi:+.2%} ({review.europa.bets}), "
            f"Conference {review.conference.roi:+.2%} ({review.conference.bets})"
        )
    high_odds = [
        review
        for review in reviews
        if "240" in review.filter_name or "260" in review.filter_name or "280" in review.filter_name or "300" in review.filter_name
    ]
    lines.append("")
    lines.append("High-odds model slices (no extra survivor cut):")
    lines.append(
        f"{'config':<16} {'filter':<24} {'disc_n':>7} {'disc_roi':>9} {'hold_n':>7} {'hold_roi':>9} "
        f"{'el_n':>6} {'el_roi':>8}"
    )
    for review in high_odds:
        lines.append(
            f"{review.config_name:<16} {review.filter_name:<24} "
            f"{review.discovery.bets:7d} {review.discovery.roi:+9.2%} "
            f"{review.holdout.bets:7d} {review.holdout.roi:+9.2%} "
            f"{review.europa.bets:6d} {review.europa.roi:+8.2%}"
        )
    return "\n".join(lines)


def format_flat_odds_buckets(paths: tuple[Path, ...]) -> str:
    ucl = load_btts_rows(paths, league="Champions League")
    europa = load_btts_rows(paths, league="Europa League")
    conference = load_btts_rows(paths, league="Conference League")
    lines = [
        "Flat BTTS No by odds bucket (no model)",
        "--------------------------------------",
        f"{'bucket':<16} {'ucl_disc':>16} {'ucl_hold':>16} {'europa':>16} {'conf':>16}",
    ]
    for name, min_odds, max_odds in FLAT_ODDS_BUCKETS:
        disc = _flat(_rows(ucl, DISCOVERY_SEASONS), min_odds, max_odds)
        hold = _flat(_rows(ucl, HOLDOUT_SEASONS), min_odds, max_odds)
        el = _flat(europa, min_odds, max_odds)
        conf = _flat(conference, min_odds, max_odds)
        lines.append(
            f"{name:<16} {_fmt_slice(disc):>16} {_fmt_slice(hold):>16} {_fmt_slice(el):>16} {_fmt_slice(conf):>16}"
        )
    return "\n".join(lines)


def _flat(rows: tuple[GoalMarketRow, ...], min_odds: float, max_odds: float) -> SliceResult:
    subset = tuple(row for row in rows if min_odds <= row.odds.get("btts_no", 0.0) <= max_odds)
    report = flat_btts_no_backtest(subset)
    return SliceResult(
        label="flat",
        bets=len(report.bets),
        profit=report.total_profit,
        roi=report.roi,
        hit_rate=report.hit_rate,
        max_drawdown=report.max_drawdown,
    )


def _fmt_slice(result: SliceResult) -> str:
    if result.bets == 0:
        return "n=0"
    return f"n={result.bets} {result.roi:+.1%}"


def _slice(
    rows: tuple[GoalMarketRow, ...],
    config: ValueConfig,
    filters: BttsNoFilter,
    seasons: tuple[str, ...],
    *,
    train_rows: tuple[GoalMarketRow, ...] | None = None,
) -> SliceResult:
    subset = _rows(rows, seasons)
    if not subset:
        return _empty(f"{filters.name}:{seasons[0]}")
    if train_rows is None:
        report = run_fixed_btts_no_walk_forward(subset, config=config, filters=filters).aggregate
    else:
        available = seasons_in_order(train_rows + subset)
        report = run_fixed_btts_no_walk_forward(
            train_rows + subset,
            config=config,
            seasons=available,
            filters=filters,
        ).aggregate
        holdout_ids = {row.match_id for row in subset}
        holdout_bets = [bet for bet in report.bets if bet.match_id in holdout_ids]
        report = _aggregate(holdout_bets)
    return SliceResult(
        label=filters.name,
        bets=len(report.bets),
        profit=report.total_profit,
        roi=report.roi,
        hit_rate=report.hit_rate,
        max_drawdown=report.max_drawdown,
    )


def _external(
    target: tuple[GoalMarketRow, ...],
    ucl: tuple[GoalMarketRow, ...],
    config: ValueConfig,
    filters: BttsNoFilter,
) -> SliceResult:
    if not target:
        return _empty(filters.name)
    combined = ucl + target
    seasons = seasons_in_order(combined)
    report = run_fixed_btts_no_walk_forward(combined, config=config, seasons=seasons, filters=filters).aggregate
    target_ids = {row.match_id for row in target}
    bets = [bet for bet in report.bets if bet.match_id in target_ids]
    sliced = _aggregate(bets)
    return SliceResult(
        label=filters.name,
        bets=len(sliced.bets),
        profit=sliced.total_profit,
        roi=sliced.roi,
        hit_rate=sliced.hit_rate,
        max_drawdown=sliced.max_drawdown,
    )


def _rows(rows: tuple[GoalMarketRow, ...], seasons: tuple[str, ...]) -> tuple[GoalMarketRow, ...]:
    wanted = set(seasons)
    return tuple(row for row in rows if row.season in wanted)


def _empty(label: str) -> SliceResult:
    return SliceResult(label=label, bets=0, profit=0.0, roi=0.0, hit_rate=0.0, max_drawdown=0.0)
