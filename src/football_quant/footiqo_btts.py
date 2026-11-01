from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .footiqo import FootiqoMatch, load_footiqo_matches
from .goal_markets import GoalMarketRow, feature_matrix, labels
from .goal_models import GoalBacktestReport, GoalBet, PredictiveMetrics
from .goal_value import ValueConfig, ValueFold, ValueWalkForwardReport, _aggregate, _report, _score


BTTS_FEATURES: tuple[str, ...] = (
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_goals_for_avg",
    "home_goals_against_avg",
    "away_goals_for_avg",
    "away_goals_against_avg",
    "goal_profile_diff",
    "home_btts_rate",
    "away_btts_rate",
    "combined_btts_rate",
    "home_clean_sheet_rate",
    "away_clean_sheet_rate",
    "home_fail_rate",
    "away_fail_rate",
    "rest_diff",
    "implied_btts_no",
    "implied_btts_yes",
    "favorite_implied",
)


@dataclass
class _TeamState:
    elo: float = 1500.0
    goals_for: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    goals_against: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    btts: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    clean_sheets: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    failed_to_score: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    last_played: datetime | None = None

    def average(self, metric: str) -> float:
        values = getattr(self, metric)
        if not values:
            return 0.0
        return sum(values) / len(values)


def load_btts_rows(
    paths: tuple[Path, ...],
    *,
    league: str | None = "Champions League",
) -> tuple[GoalMarketRow, ...]:
    matches = tuple(match for path in paths for match in load_footiqo_matches(path))
    if league and league.lower() != "all":
        matches = tuple(match for match in matches if match.league == league)
    return build_btts_rows(matches)


def build_btts_rows(matches: tuple[FootiqoMatch, ...]) -> tuple[GoalMarketRow, ...]:
    ordered = sorted(matches, key=_match_sort_key)
    states: dict[str, _TeamState] = {}
    rows: list[GoalMarketRow] = []

    for match in ordered:
        kickoff = parse_footiqo_date(match.match_date)
        home_state = states.setdefault(match.home_team, _TeamState())
        away_state = states.setdefault(match.away_team, _TeamState())
        btts_no_odds = match.odds.get("BTTSN", 0.0)
        btts_yes_odds = match.odds.get("BTTSY", 0.0)
        if btts_no_odds <= 1.0:
            _update_state(home_state, away_state, match, kickoff)
            continue

        rows.append(
            GoalMarketRow(
                season=match.season,
                match_id=match.match_id,
                features=_features(home_state, away_state, match, kickoff),
                labels={"btts_no": int(match.home_goals == 0 or match.away_goals == 0)},
                odds={"btts_no": btts_no_odds, "btts_yes": btts_yes_odds},
            )
        )
        _update_state(home_state, away_state, match, kickoff)

    return tuple(rows)


def parse_footiqo_date(value: str) -> datetime:
    return datetime.strptime(value, "%d-%m-%y %H:%M")


def seasons_in_order(rows: tuple[GoalMarketRow, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for row in rows:
        if row.season not in seen:
            seen.append(row.season)
    return tuple(seen)


def train_btts_no_model(rows: tuple[GoalMarketRow, ...]) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1_000)),
        ]
    )
    model.fit(feature_matrix(rows, BTTS_FEATURES), labels(rows, "btts_no"))
    return model


def evaluate_btts_no_model(model: Pipeline, rows: tuple[GoalMarketRow, ...]) -> PredictiveMetrics:
    y_true = labels(rows, "btts_no")
    probabilities = _probabilities(model, rows)
    predictions = [int(probability >= 0.5) for probability in probabilities]
    auc = roc_auc_score(y_true, probabilities) if len(set(y_true)) == 2 else None
    return PredictiveMetrics(
        market="btts_no",
        brier=brier_score_loss(y_true, probabilities),
        log_loss=log_loss(y_true, probabilities, labels=[0, 1]),
        accuracy=accuracy_score(y_true, predictions),
        auc=auc,
        samples=len(rows),
    )


@dataclass(frozen=True)
class BttsNoFilter:
    name: str = "base"
    min_odds: float = 1.01
    max_odds: float = 20.0
    min_favorite_implied: float = 0.0
    require_history: bool = False
    model_above_market: bool = False
    max_combined_btts_rate: float = 1.01


LOCKED_BTTS_LEAGUES: tuple[str, ...] = ("Champions League", "Europa League")
LOCKED_BTTS_CONFIG = ValueConfig(0.5, 0.02, False)
LOCKED_BTTS_FILTER = BttsNoFilter("odds_190_240", min_odds=1.90, max_odds=2.40)


def evaluate_locked_btts_signal(
    row: GoalMarketRow,
    model_probability: float,
    *,
    config: ValueConfig = LOCKED_BTTS_CONFIG,
    filters: BttsNoFilter = LOCKED_BTTS_FILTER,
) -> GoalBet | None:
    market_probability = _market_btts_no_probability(row)
    if market_probability is None:
        return None
    probability = config.model_weight * model_probability + (1.0 - config.model_weight) * market_probability
    odds = row.odds["btts_no"]
    if odds <= 1.0:
        return None
    expected_value = probability * (odds - 1.0) - (1.0 - probability)
    if expected_value < config.min_expected_value:
        return None
    if not _passes_filter(row, model_probability, market_probability, filters):
        return None
    won = row.labels.get("btts_no") == 1
    profit = odds - 1.0 if won else -1.0
    return GoalBet(
        match_id=row.match_id,
        market="btts_no",
        selection="no",
        odds=odds,
        probability=probability,
        expected_value=expected_value,
        profit=profit,
    )


def backtest_btts_no(
    model: Pipeline,
    rows: tuple[GoalMarketRow, ...],
    *,
    config: ValueConfig,
    filters: BttsNoFilter | None = None,
) -> GoalBacktestReport:
    model_probabilities = _probabilities(model, rows)
    bets: list[GoalBet] = []
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for row, model_probability in zip(rows, model_probabilities, strict=True):
        market_probability = _market_btts_no_probability(row)
        if market_probability is None:
            continue
        probability = config.model_weight * model_probability + (1.0 - config.model_weight) * market_probability
        odds = row.odds["btts_no"]
        if odds <= 1.0:
            continue
        expected_value = probability * (odds - 1.0) - (1.0 - probability)
        if expected_value < config.min_expected_value:
            continue
        if filters is not None and not _passes_filter(row, model_probability, market_probability, filters):
            continue

        won = row.labels["btts_no"] == 1
        profit = odds - 1.0 if won else -1.0
        equity += profit
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
        bets.append(
            GoalBet(
                match_id=row.match_id,
                market="btts_no",
                selection="no",
                odds=odds,
                probability=probability,
                expected_value=expected_value,
                profit=profit,
            )
        )

    return _report(bets, max_drawdown=abs(max_drawdown))


def flat_btts_no_backtest(rows: tuple[GoalMarketRow, ...]) -> GoalBacktestReport:
    bets: list[GoalBet] = []
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for row in rows:
        odds = row.odds["btts_no"]
        if odds <= 1.0:
            continue
        won = row.labels["btts_no"] == 1
        profit = odds - 1.0 if won else -1.0
        equity += profit
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
        bets.append(
            GoalBet(
                match_id=row.match_id,
                market="btts_no",
                selection="no",
                odds=odds,
                probability=1.0 / odds,
                expected_value=0.0,
                profit=profit,
            )
        )
    return _report(bets, max_drawdown=abs(max_drawdown))


def run_btts_no_walk_forward(
    rows: tuple[GoalMarketRow, ...],
    *,
    seasons: tuple[str, ...] | None = None,
    model_weights: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    min_ev_grid: tuple[float, ...] = (0.0, 0.02, 0.05, 0.08),
    min_train_bets: int = 40,
) -> ValueWalkForwardReport:
    ordered_seasons = seasons or seasons_in_order(rows)
    if len(ordered_seasons) < 2:
        raise ValueError("At least two seasons are required for walk-forward validation")

    folds: list[ValueFold] = []
    aggregate_bets: list[GoalBet] = []

    for index in range(1, len(ordered_seasons)):
        train_seasons = ordered_seasons[:index]
        test_season = ordered_seasons[index]
        train_rows = tuple(row for row in rows if row.season in train_seasons)
        test_rows = tuple(row for row in rows if row.season == test_season)
        if len(set(labels(train_rows, "btts_no"))) < 2 or not test_rows:
            empty = _report([], max_drawdown=0.0)
            folds.append(
                ValueFold(
                    train_seasons=train_seasons,
                    test_season=test_season,
                    config=ValueConfig(1.0, min_ev_grid[0], False),
                    train_report=empty,
                    test_report=empty,
                )
            )
            continue

        model = train_btts_no_model(train_rows)
        config, train_report = _select_config(
            model,
            train_rows,
            model_weights=model_weights,
            min_ev_grid=min_ev_grid,
            min_train_bets=min_train_bets,
        )
        test_report = backtest_btts_no(model, test_rows, config=config)
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


def run_fixed_btts_no_walk_forward(
    rows: tuple[GoalMarketRow, ...],
    *,
    config: ValueConfig,
    seasons: tuple[str, ...] | None = None,
    filters: BttsNoFilter | None = None,
) -> ValueWalkForwardReport:
    ordered_seasons = seasons or seasons_in_order(rows)
    if len(ordered_seasons) < 2:
        raise ValueError("At least two seasons are required for walk-forward validation")

    folds: list[ValueFold] = []
    aggregate_bets: list[GoalBet] = []
    for index in range(1, len(ordered_seasons)):
        train_seasons = ordered_seasons[:index]
        test_season = ordered_seasons[index]
        train_rows = tuple(row for row in rows if row.season in train_seasons)
        test_rows = tuple(row for row in rows if row.season == test_season)
        if len(set(labels(train_rows, "btts_no"))) < 2 or not test_rows:
            empty = _report([], max_drawdown=0.0)
            folds.append(
                ValueFold(
                    train_seasons=train_seasons,
                    test_season=test_season,
                    config=config,
                    train_report=empty,
                    test_report=empty,
                )
            )
            continue
        model = train_btts_no_model(train_rows)
        test_report = backtest_btts_no(model, test_rows, config=config, filters=filters)
        aggregate_bets.extend(test_report.bets)
        folds.append(
            ValueFold(
                train_seasons=train_seasons,
                test_season=test_season,
                config=config,
                train_report=backtest_btts_no(model, train_rows, config=config, filters=filters),
                test_report=test_report,
            )
        )
    return ValueWalkForwardReport(folds=tuple(folds), aggregate=_aggregate(aggregate_bets))


def format_btts_no_walk_forward_report(
    report: ValueWalkForwardReport,
    *,
    rows: tuple[GoalMarketRow, ...],
) -> str:
    lines = ["Champions BTTS No Walk-Forward", "-------------------------------"]
    for fold in report.folds:
        test_rows = tuple(row for row in rows if row.season == fold.test_season)
        baseline = flat_btts_no_backtest(test_rows)
        cfg = fold.config
        lines.append(
            f"test={fold.test_season}, weight={cfg.model_weight:.2f}, min_ev={cfg.min_expected_value:.2f}, "
            f"train_roi={fold.train_report.roi:+.2%}, train_bets={len(fold.train_report.bets)}, "
            f"test_roi={fold.test_report.roi:+.2%}, test_profit={fold.test_report.total_profit:+.2f}, "
            f"test_bets={len(fold.test_report.bets)}, "
            f"flat_roi={baseline.roi:+.2%}, flat_bets={len(baseline.bets)}"
        )
    aggregate = report.aggregate
    held_out = tuple(row for row in rows if any(fold.test_season == row.season for fold in report.folds))
    baseline = flat_btts_no_backtest(held_out)
    lines.append(
        f"aggregate: bets={len(aggregate.bets)}, profit={aggregate.total_profit:+.2f}, "
        f"roi={aggregate.roi:+.2%}, hit={aggregate.hit_rate:.2%}, max_dd={aggregate.max_drawdown:.2f}"
    )
    lines.append(
        f"flat_held_out: bets={len(baseline.bets)}, profit={baseline.total_profit:+.2f}, "
        f"roi={baseline.roi:+.2%}, hit={baseline.hit_rate:.2%}, max_dd={baseline.max_drawdown:.2f}"
    )
    lines.append("fixed_oos:")
    for weight, min_ev in ((0.0, 0.0), (0.0, 0.02), (1.0, 0.02), (0.5, 0.02)):
        frozen = run_fixed_btts_no_walk_forward(rows, config=ValueConfig(weight, min_ev, False))
        lines.append(
            f"  weight={weight:.2f} min_ev={min_ev:.2f}: bets={len(frozen.aggregate.bets)}, "
            f"profit={frozen.aggregate.total_profit:+.2f}, roi={frozen.aggregate.roi:+.2%}, "
            f"hit={frozen.aggregate.hit_rate:.2%}"
        )
    lines.append("refined_oos:")
    for weight, min_ev, filt in (
        (0.5, 0.02, BttsNoFilter("odds_190_240", min_odds=1.90, max_odds=2.40)),
        (1.0, 0.02, BttsNoFilter("favorite_45_odds_190", min_favorite_implied=0.45, min_odds=1.90)),
    ):
        frozen = run_fixed_btts_no_walk_forward(rows, config=ValueConfig(weight, min_ev, False), filters=filt)
        lines.append(
            f"  weight={weight:.2f} min_ev={min_ev:.2f} {filt.name}: bets={len(frozen.aggregate.bets)}, "
            f"profit={frozen.aggregate.total_profit:+.2f}, roi={frozen.aggregate.roi:+.2%}, "
            f"hit={frozen.aggregate.hit_rate:.2%}"
        )
    return "\n".join(lines)


def _features(home_state: _TeamState, away_state: _TeamState, match: FootiqoMatch, kickoff: datetime) -> dict[str, float]:
    btts_no_odds = match.odds.get("BTTSN", 0.0)
    btts_yes_odds = match.odds.get("BTTSY", 0.0)
    home_odds = match.odds.get("H", 0.0)
    away_odds = match.odds.get("A", 0.0)
    implied_no = 1.0 / btts_no_odds if btts_no_odds > 1.0 else 0.0
    implied_yes = 1.0 / btts_yes_odds if btts_yes_odds > 1.0 else 0.0
    total_implied = implied_no + implied_yes
    if total_implied > 0:
        implied_no /= total_implied
        implied_yes /= total_implied
    favorite_implied = 0.0
    if home_odds > 1.0 and away_odds > 1.0:
        favorite_implied = max(1.0 / home_odds, 1.0 / away_odds)
    return {
        "home_elo": home_state.elo,
        "away_elo": away_state.elo,
        "elo_diff": home_state.elo - away_state.elo,
        "home_goals_for_avg": home_state.average("goals_for"),
        "home_goals_against_avg": home_state.average("goals_against"),
        "away_goals_for_avg": away_state.average("goals_for"),
        "away_goals_against_avg": away_state.average("goals_against"),
        "goal_profile_diff": (
            home_state.average("goals_for")
            - home_state.average("goals_against")
            - away_state.average("goals_for")
            + away_state.average("goals_against")
        ),
        "home_btts_rate": home_state.average("btts"),
        "away_btts_rate": away_state.average("btts"),
        "combined_btts_rate": (home_state.average("btts") + away_state.average("btts")) / 2.0,
        "home_clean_sheet_rate": home_state.average("clean_sheets"),
        "away_clean_sheet_rate": away_state.average("clean_sheets"),
        "home_fail_rate": home_state.average("failed_to_score"),
        "away_fail_rate": away_state.average("failed_to_score"),
        "rest_diff": float(_rest_days(home_state, kickoff) - _rest_days(away_state, kickoff)),
        "implied_btts_no": implied_no,
        "implied_btts_yes": implied_yes,
        "favorite_implied": favorite_implied,
    }


def _update_state(home_state: _TeamState, away_state: _TeamState, match: FootiqoMatch, kickoff: datetime) -> None:
    goal_diff = match.home_goals - match.away_goals
    expected_home = 1.0 / (1.0 + 10 ** ((away_state.elo - home_state.elo) / 400))
    actual_home = 1.0 if goal_diff > 0 else 0.5 if goal_diff == 0 else 0.0
    adjustment = 24 * (actual_home - expected_home)
    home_state.elo += adjustment
    away_state.elo -= adjustment

    both_scored = float(match.home_goals > 0 and match.away_goals > 0)
    home_state.goals_for.append(float(match.home_goals))
    home_state.goals_against.append(float(match.away_goals))
    away_state.goals_for.append(float(match.away_goals))
    away_state.goals_against.append(float(match.home_goals))
    home_state.btts.append(both_scored)
    away_state.btts.append(both_scored)
    home_state.clean_sheets.append(float(match.away_goals == 0))
    away_state.clean_sheets.append(float(match.home_goals == 0))
    home_state.failed_to_score.append(float(match.home_goals == 0))
    away_state.failed_to_score.append(float(match.away_goals == 0))
    home_state.last_played = kickoff
    away_state.last_played = kickoff


def _rest_days(state: _TeamState, kickoff: datetime) -> int:
    if state.last_played is None:
        return 7
    return max(0, (kickoff.date() - state.last_played.date()).days)


def _match_sort_key(match: FootiqoMatch) -> tuple[datetime, str]:
    return parse_footiqo_date(match.match_date), match.match_id


def _probabilities(model: Pipeline, rows: tuple[GoalMarketRow, ...]) -> list[float]:
    return [float(probability[1]) for probability in model.predict_proba(feature_matrix(rows, BTTS_FEATURES))]


def _passes_filter(
    row: GoalMarketRow,
    model_probability: float,
    market_probability: float,
    filters: BttsNoFilter,
) -> bool:
    odds = row.odds["btts_no"]
    if odds < filters.min_odds or odds > filters.max_odds:
        return False
    if row.features.get("favorite_implied", 0.0) < filters.min_favorite_implied:
        return False
    if row.features.get("combined_btts_rate", 0.0) > filters.max_combined_btts_rate:
        return False
    if filters.model_above_market and model_probability <= market_probability:
        return False
    if filters.require_history and not _has_history(row):
        return False
    return True


def _has_history(row: GoalMarketRow) -> bool:
    return _team_has_history(row, "home") and _team_has_history(row, "away")


def _team_has_history(row: GoalMarketRow, side: str) -> bool:
    return any(
        row.features.get(name, 0.0) > 0.0
        for name in (
            f"{side}_goals_for_avg",
            f"{side}_goals_against_avg",
            f"{side}_btts_rate",
            f"{side}_clean_sheet_rate",
            f"{side}_fail_rate",
        )
    )


def _market_btts_no_probability(row: GoalMarketRow) -> float | None:
    no_odds = row.odds["btts_no"]
    yes_odds = row.odds["btts_yes"]
    if no_odds <= 1.0 or yes_odds <= 1.0:
        return None
    no_implied = 1.0 / no_odds
    yes_implied = 1.0 / yes_odds
    return no_implied / (no_implied + yes_implied)


def _select_config(
    model: Pipeline,
    rows: tuple[GoalMarketRow, ...],
    *,
    model_weights: tuple[float, ...],
    min_ev_grid: tuple[float, ...],
    min_train_bets: int,
) -> tuple[ValueConfig, GoalBacktestReport]:
    best: tuple[ValueConfig, GoalBacktestReport] | None = None
    for weight in model_weights:
        for min_ev in min_ev_grid:
            config = ValueConfig(weight, min_ev, False)
            report = backtest_btts_no(model, rows, config=config)
            if len(report.bets) < min_train_bets:
                continue
            if best is None or _score(report) > _score(best[1]):
                best = (config, report)
    if best is None:
        config = ValueConfig(1.0, min_ev_grid[0], False)
        return config, backtest_btts_no(model, rows, config=config)
    return best
