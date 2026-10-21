from __future__ import annotations

from dataclasses import dataclass

from sklearn.pipeline import Pipeline

from .goal_markets import GoalMarketRow, feature_matrix
from .goal_models import GoalBacktestReport, GoalBet, train_goal_model


@dataclass(frozen=True)
class ValueConfig:
    model_weight: float
    min_expected_value: float
    use_closing_odds: bool


@dataclass(frozen=True)
class ValueFold:
    train_seasons: tuple[str, ...]
    test_season: str
    config: ValueConfig
    train_report: GoalBacktestReport
    test_report: GoalBacktestReport


@dataclass(frozen=True)
class ValueWalkForwardReport:
    folds: tuple[ValueFold, ...]
    aggregate: GoalBacktestReport


def run_market_aware_ou_walk_forward(
    rows: tuple[GoalMarketRow, ...],
    *,
    seasons: tuple[str, ...],
    model_weights: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    min_ev_grid: tuple[float, ...] = (0.02, 0.05, 0.08, 0.1, 0.15),
    use_closing_odds: bool = False,
    min_train_bets: int = 50,
) -> ValueWalkForwardReport:
    folds: list[ValueFold] = []
    aggregate_bets: list[GoalBet] = []

    for index in range(1, len(seasons)):
        train_seasons = seasons[:index]
        test_season = seasons[index]
        train_rows = tuple(row for row in rows if row.season in train_seasons)
        test_rows = tuple(row for row in rows if row.season == test_season)
        model = train_goal_model(train_rows, "over_25")
        config, train_report = _select_config(
            model,
            train_rows,
            model_weights=model_weights,
            min_ev_grid=min_ev_grid,
            use_closing_odds=use_closing_odds,
            min_train_bets=min_train_bets,
        )
        test_report = backtest_market_aware_ou(model, test_rows, config=config)
        aggregate_bets.extend(test_report.bets)
        folds.append(
            ValueFold(
                train_seasons=train_seasons,
                test_season=test_season,
                config=config,
                train_report=train_report,
                test_report=test_report,
            )
        )

    return ValueWalkForwardReport(folds=tuple(folds), aggregate=_aggregate(aggregate_bets))


def backtest_market_aware_ou(
    model: Pipeline,
    rows: tuple[GoalMarketRow, ...],
    *,
    config: ValueConfig,
) -> GoalBacktestReport:
    model_probabilities = [float(probability[1]) for probability in model.predict_proba(feature_matrix(rows))]
    bets: list[GoalBet] = []
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for row, model_probability in zip(rows, model_probabilities, strict=True):
        market_probability = _market_over_probability(row, use_closing_odds=config.use_closing_odds)
        if market_probability is None:
            continue
        over_probability = config.model_weight * model_probability + (1.0 - config.model_weight) * market_probability
        under_probability = 1.0 - over_probability
        over_odds, under_odds = _odds(row, use_closing_odds=config.use_closing_odds)

        candidates = (
            ("over", over_odds, over_probability, row.labels["over_25"] == 1),
            ("under", under_odds, under_probability, row.labels["over_25"] == 0),
        )
        best = None
        for selection, odds, probability, won in candidates:
            if odds <= 1.0:
                continue
            expected_value = probability * (odds - 1.0) - (1.0 - probability)
            if best is None or expected_value > best[3]:
                best = (selection, odds, probability, expected_value, won)
        if best is None or best[3] < config.min_expected_value:
            continue

        selection, odds, probability, expected_value, won = best
        profit = odds - 1.0 if won else -1.0
        equity += profit
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
        bets.append(
            GoalBet(
                match_id=row.match_id,
                market="over_25",
                selection=selection,
                odds=odds,
                probability=probability,
                expected_value=expected_value,
                profit=profit,
            )
        )

    return _report(bets, max_drawdown=abs(max_drawdown))


def format_value_walk_forward_report(report: ValueWalkForwardReport) -> str:
    lines = ["Market-Aware O/U Walk-Forward", "-----------------------------"]
    for fold in report.folds:
        cfg = fold.config
        lines.append(
            f"test={fold.test_season}, weight={cfg.model_weight:.2f}, min_ev={cfg.min_expected_value:.2f}, "
            f"train_roi={fold.train_report.roi:+.2%}, train_bets={len(fold.train_report.bets)}, "
            f"test_roi={fold.test_report.roi:+.2%}, test_profit={fold.test_report.total_profit:+.2f}, "
            f"test_bets={len(fold.test_report.bets)}"
        )
    aggregate = report.aggregate
    lines.append(
        f"aggregate: bets={len(aggregate.bets)}, profit={aggregate.total_profit:+.2f}, "
        f"roi={aggregate.roi:+.2%}, hit={aggregate.hit_rate:.2%}, max_dd={aggregate.max_drawdown:.2f}"
    )
    return "\n".join(lines)


def _select_config(
    model: Pipeline,
    rows: tuple[GoalMarketRow, ...],
    *,
    model_weights: tuple[float, ...],
    min_ev_grid: tuple[float, ...],
    use_closing_odds: bool,
    min_train_bets: int,
) -> tuple[ValueConfig, GoalBacktestReport]:
    best: tuple[ValueConfig, GoalBacktestReport] | None = None
    for weight in model_weights:
        for min_ev in min_ev_grid:
            config = ValueConfig(weight, min_ev, use_closing_odds)
            report = backtest_market_aware_ou(model, rows, config=config)
            if len(report.bets) < min_train_bets:
                continue
            if best is None or _score(report) > _score(best[1]):
                best = (config, report)

    if best is None:
        config = ValueConfig(1.0, min_ev_grid[0], use_closing_odds)
        return config, backtest_market_aware_ou(model, rows, config=config)
    return best


def _market_over_probability(row: GoalMarketRow, *, use_closing_odds: bool) -> float | None:
    over_odds, under_odds = _odds(row, use_closing_odds=use_closing_odds)
    if over_odds <= 1.0 or under_odds <= 1.0:
        return None
    over_implied = 1.0 / over_odds
    under_implied = 1.0 / under_odds
    return over_implied / (over_implied + under_implied)


def _odds(row: GoalMarketRow, *, use_closing_odds: bool) -> tuple[float, float]:
    if use_closing_odds:
        return row.odds["closing_over_25"], row.odds["closing_under_25"]
    return row.odds["opening_over_25"], row.odds["opening_under_25"]


def _score(report: GoalBacktestReport) -> float:
    sample_penalty = 1.0 - min(len(report.bets), 250) / 250
    return report.roi - 0.05 * sample_penalty


def _report(bets: list[GoalBet], *, max_drawdown: float) -> GoalBacktestReport:
    total_profit = sum(bet.profit for bet in bets)
    total_staked = float(len(bets))
    wins = sum(1 for bet in bets if bet.profit > 0)
    return GoalBacktestReport(
        bets=tuple(bets),
        total_profit=total_profit,
        total_staked=total_staked,
        roi=total_profit / total_staked if total_staked else 0.0,
        hit_rate=wins / len(bets) if bets else 0.0,
        max_drawdown=max_drawdown,
    )


def _aggregate(bets: list[GoalBet]) -> GoalBacktestReport:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for bet in bets:
        equity += bet.profit
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    return _report(bets, max_drawdown=abs(max_drawdown))
