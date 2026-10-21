from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .analysis import SegmentMetrics
from .backtesting import BacktestBet


BetPredicate = Callable[[BacktestBet], bool]


@dataclass(frozen=True)
class StrategyScenario:
    name: str
    description: str
    predicate: BetPredicate


@dataclass(frozen=True)
class StrategyResult:
    scenario: StrategyScenario
    metrics: SegmentMetrics


PROMISING_LINES = {-1.5, -1.25, -1.0, -0.5, 0.25}


def default_strategy_scenarios() -> tuple[StrategyScenario, ...]:
    return (
        StrategyScenario(
            "baseline",
            "All bets selected by the model EV threshold.",
            lambda bet: True,
        ),
        StrategyScenario(
            "odds_ge_2",
            "Only selections with decimal odds greater than or equal to 2.00.",
            lambda bet: bet.odds >= 2.0,
        ),
        StrategyScenario(
            "odds_2_to_224",
            "Only selections in the strongest observed odds bucket: 2.00 to 2.24.",
            lambda bet: 2.0 <= bet.odds < 2.25,
        ),
        StrategyScenario(
            "away_only",
            "Only away-side Asian handicap selections.",
            lambda bet: bet.side == "away",
        ),
        StrategyScenario(
            "away_odds_ge_2",
            "Away-side selections with odds greater than or equal to 2.00.",
            lambda bet: bet.side == "away" and bet.odds >= 2.0,
        ),
        StrategyScenario(
            "promising_lines",
            "Only line values that were positive in the first segmented pass.",
            lambda bet: bet.line in PROMISING_LINES,
        ),
        StrategyScenario(
            "promising_lines_odds_ge_2",
            "Promising line values combined with odds greater than or equal to 2.00.",
            lambda bet: bet.line in PROMISING_LINES and bet.odds >= 2.0,
        ),
    )


def evaluate_strategy_scenarios(
    bets: Iterable[BacktestBet],
    scenarios: Iterable[StrategyScenario] | None = None,
) -> tuple[StrategyResult, ...]:
    scenario_list = tuple(scenarios or default_strategy_scenarios())
    bet_list = tuple(bets)
    results = []
    for scenario in scenario_list:
        selected = [bet for bet in bet_list if scenario.predicate(bet)]
        results.append(StrategyResult(scenario=scenario, metrics=_metrics(scenario.name, selected)))
    return tuple(sorted(results, key=lambda item: (item.metrics.roi, item.metrics.profit), reverse=True))


def format_strategy_report(results: tuple[StrategyResult, ...]) -> str:
    lines = ["Strategy Scenarios", "------------------"]
    for result in results:
        metrics = result.metrics
        clv = "n/a" if metrics.average_clv is None else f"{metrics.average_clv:+.4f}"
        lines.append(
            f"{result.scenario.name}: bets={metrics.bets}, profit={metrics.profit:+.2f}, "
            f"roi={metrics.roi:+.2%}, hit={metrics.hit_rate:.2%}, clv={clv}"
        )
    return "\n".join(lines)


def _metrics(segment: str, bets: list[BacktestBet]) -> SegmentMetrics:
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
