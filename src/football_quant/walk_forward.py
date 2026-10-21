from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .analysis import SegmentMetrics
from .backtesting import run_backtest
from .football_data_uk import load_football_data_uk_csv, season_label_from_code
from .strategy import StrategyScenario, default_strategy_scenarios, evaluate_strategy_scenarios


@dataclass(frozen=True)
class WalkForwardFold:
    train_seasons: tuple[str, ...]
    test_season: str
    selected_scenario: str
    selected_min_ev: float
    train_metrics: SegmentMetrics
    test_metrics: SegmentMetrics


@dataclass(frozen=True)
class WalkForwardReport:
    folds: tuple[WalkForwardFold, ...]
    aggregate: SegmentMetrics


def run_walk_forward(
    data_root: Path,
    *,
    season_codes: tuple[str, ...],
    min_ev_grid: tuple[float, ...] = (0.0, 0.02, 0.05, 0.08),
    min_train_bets: int = 50,
    include_closing_quotes: bool = False,
) -> WalkForwardReport:
    if len(season_codes) < 2:
        raise ValueError("At least two seasons are required for walk-forward validation")

    scenarios = default_strategy_scenarios()
    datasets = {
        code: load_football_data_uk_csv(
            data_root / code / "E0.csv",
            season=season_label_from_code(code),
            competition="Premier League",
        )
        for code in season_codes
    }

    folds: list[WalkForwardFold] = []
    aggregate_test_bets = []
    for index in range(1, len(season_codes)):
        train_codes = season_codes[:index]
        test_code = season_codes[index]
        selected = _select_best_training_strategy(
            [datasets[code] for code in train_codes],
            scenarios=scenarios,
            min_ev_grid=min_ev_grid,
            min_train_bets=min_train_bets,
            include_closing_quotes=include_closing_quotes,
        )
        scenario, min_ev, train_metrics = selected
        test_report = run_backtest(
            datasets[test_code],
            min_expected_value=min_ev,
            include_closing_quotes=include_closing_quotes,
        )
        test_bets = [bet for bet in test_report.bets if scenario.predicate(bet)]
        aggregate_test_bets.extend(test_bets)
        test_metrics = _metrics(scenario.name, test_bets)
        folds.append(
            WalkForwardFold(
                train_seasons=tuple(season_label_from_code(code) for code in train_codes),
                test_season=season_label_from_code(test_code),
                selected_scenario=scenario.name,
                selected_min_ev=min_ev,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )
        )

    return WalkForwardReport(
        folds=tuple(folds),
        aggregate=_metrics("walk_forward", aggregate_test_bets),
    )


def format_walk_forward_report(report: WalkForwardReport) -> str:
    lines = ["Walk-Forward Validation", "-----------------------"]
    for fold in report.folds:
        lines.append(
            f"test={fold.test_season}, scenario={fold.selected_scenario}, "
            f"min_ev={fold.selected_min_ev:.2f}, "
            f"train_roi={fold.train_metrics.roi:+.2%}, train_bets={fold.train_metrics.bets}, "
            f"test_roi={fold.test_metrics.roi:+.2%}, test_profit={fold.test_metrics.profit:+.2f}, "
            f"test_bets={fold.test_metrics.bets}"
        )
    aggregate = report.aggregate
    lines.append(
        f"aggregate: bets={aggregate.bets}, profit={aggregate.profit:+.2f}, "
        f"roi={aggregate.roi:+.2%}, hit={aggregate.hit_rate:.2%}, "
        f"clv={'n/a' if aggregate.average_clv is None else f'{aggregate.average_clv:+.4f}'}"
    )
    return "\n".join(lines)


def _select_best_training_strategy(
    train_datasets,
    *,
    scenarios: tuple[StrategyScenario, ...],
    min_ev_grid: tuple[float, ...],
    min_train_bets: int,
    include_closing_quotes: bool,
) -> tuple[StrategyScenario, float, SegmentMetrics]:
    best: tuple[StrategyScenario, float, SegmentMetrics] | None = None
    for min_ev in min_ev_grid:
        train_bets = []
        for dataset in train_datasets:
            train_bets.extend(
                run_backtest(
                    dataset,
                    min_expected_value=min_ev,
                    include_closing_quotes=include_closing_quotes,
                ).bets
            )
        for result in evaluate_strategy_scenarios(train_bets, scenarios):
            if result.metrics.bets < min_train_bets:
                continue
            candidate = (result.scenario, min_ev, result.metrics)
            if best is None or _score(candidate[2]) > _score(best[2]):
                best = candidate

    if best is None:
        baseline = scenarios[0]
        train_bets = []
        for dataset in train_datasets:
            train_bets.extend(
                run_backtest(
                    dataset,
                    min_expected_value=min_ev_grid[0],
                    include_closing_quotes=include_closing_quotes,
                ).bets
            )
        return baseline, min_ev_grid[0], _metrics(baseline.name, train_bets)

    return best


def _score(metrics: SegmentMetrics) -> float:
    sample_penalty = 1.0 - min(metrics.bets, 250) / 250
    return metrics.roi - 0.05 * sample_penalty


def _metrics(segment: str, bets) -> SegmentMetrics:
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
