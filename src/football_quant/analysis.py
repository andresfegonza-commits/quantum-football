from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from .backtesting import BacktestBet


@dataclass(frozen=True)
class SegmentMetrics:
    segment: str
    bets: int
    staked: float
    profit: float
    roi: float
    hit_rate: float
    average_clv: float | None


SegmentKey = Callable[[BacktestBet], str]


def summarize_segments(bets: Iterable[BacktestBet], key: SegmentKey) -> tuple[SegmentMetrics, ...]:
    grouped: dict[str, list[BacktestBet]] = defaultdict(list)
    for bet in bets:
        grouped[key(bet)].append(bet)

    summaries = [_summarize(segment, segment_bets) for segment, segment_bets in grouped.items()]
    return tuple(sorted(summaries, key=lambda item: (item.roi, item.profit), reverse=True))


def side_key(bet: BacktestBet) -> str:
    return f"side={bet.side}"


def line_key(bet: BacktestBet) -> str:
    return f"line={bet.line:+.2f}"


def odds_bucket_key(bet: BacktestBet) -> str:
    if bet.odds < 1.75:
        bucket = "<1.75"
    elif bet.odds < 2.0:
        bucket = "1.75-1.99"
    elif bet.odds < 2.25:
        bucket = "2.00-2.24"
    else:
        bucket = ">=2.25"
    return f"odds={bucket}"


def clv_bucket_key(bet: BacktestBet) -> str:
    if bet.closing_line_value is None:
        bucket = "missing"
    elif bet.closing_line_value < -0.03:
        bucket = "negative"
    elif bet.closing_line_value <= 0.03:
        bucket = "flat"
    else:
        bucket = "positive"
    return f"clv={bucket}"


def _summarize(segment: str, bets: list[BacktestBet]) -> SegmentMetrics:
    staked = sum(bet.stake for bet in bets)
    profit = sum(bet.profit for bet in bets)
    wins = sum(1 for bet in bets if bet.profit > 0)
    clv_values = [bet.closing_line_value for bet in bets if bet.closing_line_value is not None]
    return SegmentMetrics(
        segment=segment,
        bets=len(bets),
        staked=staked,
        profit=profit,
        roi=profit / staked if staked else 0.0,
        hit_rate=wins / len(bets) if bets else 0.0,
        average_clv=sum(clv_values) / len(clv_values) if clv_values else None,
    )


def format_segment_report(title: str, segments: tuple[SegmentMetrics, ...], *, limit: int = 12) -> str:
    lines = [title, "-" * len(title)]
    for item in segments[:limit]:
        clv = "n/a" if item.average_clv is None else f"{item.average_clv:+.4f}"
        lines.append(
            f"{item.segment}: bets={item.bets}, profit={item.profit:+.2f}, "
            f"roi={item.roi:+.2%}, hit={item.hit_rate:.2%}, clv={clv}"
        )
    return "\n".join(lines)
