from __future__ import annotations

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .goal_markets import GOAL_FEATURES, GOAL_MARKETS, GoalMarketRow, feature_matrix, labels


@dataclass(frozen=True)
class PredictiveMetrics:
    market: str
    brier: float
    log_loss: float
    accuracy: float
    auc: float | None
    samples: int


@dataclass(frozen=True)
class GoalBet:
    match_id: str
    market: str
    selection: str
    odds: float
    probability: float
    expected_value: float
    profit: float


@dataclass(frozen=True)
class GoalBacktestReport:
    bets: tuple[GoalBet, ...]
    total_profit: float
    total_staked: float
    roi: float
    hit_rate: float
    max_drawdown: float


@dataclass(frozen=True)
class GoalWalkForwardFold:
    train_seasons: tuple[str, ...]
    test_season: str
    predictive: tuple[PredictiveMetrics, ...]
    over_under_backtest: GoalBacktestReport


@dataclass(frozen=True)
class GoalWalkForwardReport:
    folds: tuple[GoalWalkForwardFold, ...]
    aggregate_over_under: GoalBacktestReport


def train_goal_model(rows: tuple[GoalMarketRow, ...], market: str) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1_000)),
        ]
    )
    model.fit(feature_matrix(rows), labels(rows, market))
    return model


def evaluate_goal_model(model: Pipeline, rows: tuple[GoalMarketRow, ...], market: str) -> PredictiveMetrics:
    y_true = labels(rows, market)
    probabilities = _positive_probabilities(model, rows)
    predictions = [int(probability >= 0.5) for probability in probabilities]
    auc = None
    if len(set(y_true)) == 2:
        auc = roc_auc_score(y_true, probabilities)
    return PredictiveMetrics(
        market=market,
        brier=brier_score_loss(y_true, probabilities),
        log_loss=log_loss(y_true, probabilities, labels=[0, 1]),
        accuracy=accuracy_score(y_true, predictions),
        auc=auc,
        samples=len(rows),
    )


def backtest_over_under_25(
    model: Pipeline,
    rows: tuple[GoalMarketRow, ...],
    *,
    min_expected_value: float = 0.02,
    use_closing_odds: bool = False,
) -> GoalBacktestReport:
    probabilities = _positive_probabilities(model, rows)
    bets: list[GoalBet] = []
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for row, over_probability in zip(rows, probabilities, strict=True):
        under_probability = 1.0 - over_probability
        over_key = "closing_over_25" if use_closing_odds else "opening_over_25"
        under_key = "closing_under_25" if use_closing_odds else "opening_under_25"
        candidates = (
            ("over_25", "over", row.odds[over_key], over_probability, row.labels["over_25"] == 1),
            ("over_25", "under", row.odds[under_key], under_probability, row.labels["over_25"] == 0),
        )
        best = None
        for market, selection, odds, probability, won in candidates:
            if odds <= 1.0:
                continue
            expected_value = probability * (odds - 1.0) - (1.0 - probability)
            if best is None or expected_value > best[4]:
                best = (market, selection, odds, probability, expected_value, won)
        if best is None or best[4] < min_expected_value:
            continue

        market, selection, odds, probability, expected_value, won = best
        profit = odds - 1.0 if won else -1.0
        equity += profit
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
        bets.append(
            GoalBet(
                match_id=row.match_id,
                market=market,
                selection=selection,
                odds=odds,
                probability=probability,
                expected_value=expected_value,
                profit=profit,
            )
        )

    total_profit = sum(bet.profit for bet in bets)
    total_staked = float(len(bets))
    wins = sum(1 for bet in bets if bet.profit > 0)
    return GoalBacktestReport(
        bets=tuple(bets),
        total_profit=total_profit,
        total_staked=total_staked,
        roi=total_profit / total_staked if total_staked else 0.0,
        hit_rate=wins / len(bets) if bets else 0.0,
        max_drawdown=abs(max_drawdown),
    )


def run_goal_walk_forward(
    rows: tuple[GoalMarketRow, ...],
    *,
    seasons: tuple[str, ...],
    min_expected_value: float = 0.02,
    use_closing_odds: bool = False,
) -> GoalWalkForwardReport:
    folds: list[GoalWalkForwardFold] = []
    aggregate_bets: list[GoalBet] = []

    for index in range(1, len(seasons)):
        train_seasons = seasons[:index]
        test_season = seasons[index]
        train_rows = tuple(row for row in rows if row.season in train_seasons)
        test_rows = tuple(row for row in rows if row.season == test_season)

        predictive = []
        over_model = None
        for market in GOAL_MARKETS:
            model = train_goal_model(train_rows, market)
            if market == "over_25":
                over_model = model
            predictive.append(evaluate_goal_model(model, test_rows, market))

        if over_model is None:
            raise RuntimeError("Over 2.5 model was not trained")

        backtest = backtest_over_under_25(
            over_model,
            test_rows,
            min_expected_value=min_expected_value,
            use_closing_odds=use_closing_odds,
        )
        aggregate_bets.extend(backtest.bets)
        folds.append(
            GoalWalkForwardFold(
                train_seasons=train_seasons,
                test_season=test_season,
                predictive=tuple(predictive),
                over_under_backtest=backtest,
            )
        )

    return GoalWalkForwardReport(
        folds=tuple(folds),
        aggregate_over_under=_aggregate_backtest(aggregate_bets),
    )


def format_goal_walk_forward_report(report: GoalWalkForwardReport) -> str:
    lines = ["Goal Market Walk-Forward", "------------------------"]
    for fold in report.folds:
        over_metrics = _metric(fold, "over_25")
        btts_metrics = _metric(fold, "btts")
        first_half = _metric(fold, "first_half_over_05")
        backtest = fold.over_under_backtest
        lines.append(
            f"test={fold.test_season}, over25_brier={over_metrics.brier:.4f}, "
            f"over25_auc={_fmt_auc(over_metrics.auc)}, btts_brier={btts_metrics.brier:.4f}, "
            f"fh_o05_brier={first_half.brier:.4f}, ou_bets={len(backtest.bets)}, "
            f"ou_roi={backtest.roi:+.2%}, ou_profit={backtest.total_profit:+.2f}"
        )
    aggregate = report.aggregate_over_under
    lines.append(
        f"aggregate_ou: bets={len(aggregate.bets)}, profit={aggregate.total_profit:+.2f}, "
        f"roi={aggregate.roi:+.2%}, hit={aggregate.hit_rate:.2%}, max_dd={aggregate.max_drawdown:.2f}"
    )
    return "\n".join(lines)


def _positive_probabilities(model: Pipeline, rows: tuple[GoalMarketRow, ...]) -> list[float]:
    return [float(probability[1]) for probability in model.predict_proba(feature_matrix(rows))]


def _metric(fold: GoalWalkForwardFold, market: str) -> PredictiveMetrics:
    return next(metric for metric in fold.predictive if metric.market == market)


def _fmt_auc(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _aggregate_backtest(bets: list[GoalBet]) -> GoalBacktestReport:
    total_profit = sum(bet.profit for bet in bets)
    total_staked = float(len(bets))
    wins = sum(1 for bet in bets if bet.profit > 0)
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for bet in bets:
        equity += bet.profit
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    return GoalBacktestReport(
        bets=tuple(bets),
        total_profit=total_profit,
        total_staked=total_staked,
        roi=total_profit / total_staked if total_staked else 0.0,
        hit_rate=wins / len(bets) if bets else 0.0,
        max_drawdown=abs(max_drawdown),
    )
