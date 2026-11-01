from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date
from itertools import groupby
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .footiqo import FootiqoMatch, _market_won, load_footiqo_matches
from .footiqo_btts import parse_footiqo_date
from .goal_markets import GoalMarketRow, feature_matrix, labels
from .goal_models import GoalBacktestReport, GoalBet
from .goal_value import _report


OVER_ACC_MARKETS: tuple[str, ...] = ("O05", "O15")
OVER_FEATURES: tuple[str, ...] = (
    "home_elo",
    "away_elo",
    "home_goals_for_avg",
    "home_goals_against_avg",
    "away_goals_for_avg",
    "away_goals_against_avg",
    "combined_goals_avg",
    "implied",
)


@dataclass(frozen=True)
class AccumulatorConfig:
    min_combined_odds: float = 1.40
    min_legs: int = 2
    max_legs: int = 4
    min_leg_odds: float = 1.08
    max_leg_odds: float = 1.32
    min_combined_ev: float = -1.0


@dataclass(frozen=True)
class AccumulatorLeg:
    match_id: str
    match_date: str
    season: str
    league: str
    market: str
    odds: float
    probability: float
    won: bool


@dataclass(frozen=True)
class AccumulatorSlip:
    card_date: date
    season: str
    legs: tuple[AccumulatorLeg, ...]
    combined_odds: float
    combined_probability: float
    expected_value: float
    won: bool
    profit: float


@dataclass
class _TeamState:
    elo: float = 1500.0
    goals_for: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    goals_against: deque[float] = field(default_factory=lambda: deque(maxlen=5))

    def average(self, metric: str) -> float:
        values = getattr(self, metric)
        if not values:
            return 0.0
        return sum(values) / len(values)


def load_over_matches(
    paths: tuple[Path, ...],
    *,
    league: str | None = "all",
) -> tuple[FootiqoMatch, ...]:
    matches = tuple(match for path in paths for match in load_footiqo_matches(path))
    if league and league.lower() != "all":
        matches = tuple(match for match in matches if match.league == league)
    return matches


def build_over_rows(matches: tuple[FootiqoMatch, ...], market: str) -> tuple[GoalMarketRow, ...]:
    ordered = sorted(matches, key=lambda match: (parse_footiqo_date(match.match_date), match.match_id))
    states: dict[str, _TeamState] = {}
    rows: list[GoalMarketRow] = []
    for match in ordered:
        home_state = states.setdefault(match.home_team, _TeamState())
        away_state = states.setdefault(match.away_team, _TeamState())
        odds = match.odds.get(market, 0.0)
        if odds > 1.0:
            rows.append(
                GoalMarketRow(
                    season=match.season,
                    match_id=match.match_id,
                    features=_over_features(home_state, away_state, odds),
                    labels={market: int(_market_won(match, market))},
                    odds={market: odds},
                )
            )
        _update_state(home_state, away_state, match)
    return tuple(rows)


def train_over_model(rows: tuple[GoalMarketRow, ...], market: str) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1_000)),
        ]
    )
    model.fit(feature_matrix(rows, OVER_FEATURES), labels(rows, market))
    return model


def build_card_accumulator(
    matches: tuple[FootiqoMatch, ...],
    *,
    config: AccumulatorConfig,
    probabilities: dict[tuple[str, str], float] | None = None,
) -> AccumulatorSlip | None:
    if not matches:
        return None
    candidates = _candidate_legs(matches, config=config, probabilities=probabilities)
    if len(candidates) < config.min_legs:
        return None

    selected: list[AccumulatorLeg] = []
    combined_odds = 1.0
    combined_probability = 1.0
    for leg in candidates:
        if len(selected) >= config.max_legs:
            break
        selected.append(leg)
        combined_odds *= leg.odds
        combined_probability *= leg.probability
        if len(selected) >= config.min_legs and combined_odds >= config.min_combined_odds:
            break

    if len(selected) < config.min_legs or combined_odds < config.min_combined_odds:
        return None

    expected_value = combined_probability * (combined_odds - 1.0) - (1.0 - combined_probability)
    if expected_value < config.min_combined_ev:
        return None

    won = all(leg.won for leg in selected)
    kickoff = parse_footiqo_date(matches[0].match_date)
    return AccumulatorSlip(
        card_date=kickoff.date(),
        season=_card_season(matches),
        legs=tuple(selected),
        combined_odds=combined_odds,
        combined_probability=combined_probability,
        expected_value=expected_value,
        won=won,
        profit=combined_odds - 1.0 if won else -1.0,
    )


def backtest_over_accumulators(
    matches: tuple[FootiqoMatch, ...],
    *,
    config: AccumulatorConfig | None = None,
    probabilities: dict[tuple[str, str], float] | None = None,
) -> tuple[AccumulatorSlip, ...]:
    config = config or AccumulatorConfig()
    slips: list[AccumulatorSlip] = []
    ordered = sorted(matches, key=lambda match: (parse_footiqo_date(match.match_date), match.match_id))
    for _day, group in groupby(ordered, key=_card_date):
        slip = build_card_accumulator(tuple(group), config=config, probabilities=probabilities)
        if slip is not None:
            slips.append(slip)
    return tuple(slips)


def run_over_accumulator_walk_forward(
    matches: tuple[FootiqoMatch, ...],
    *,
    config: AccumulatorConfig | None = None,
    use_model: bool = True,
) -> tuple[AccumulatorSlip, ...]:
    config = config or AccumulatorConfig()
    seasons = _seasons_in_order(matches)
    if len(seasons) < 2:
        raise ValueError("At least two seasons are required for walk-forward validation")

    slips: list[AccumulatorSlip] = []
    for index in range(1, len(seasons)):
        train_matches = tuple(match for match in matches if match.season in seasons[:index])
        test_matches = tuple(match for match in matches if match.season == seasons[index])
        probabilities = None
        if use_model:
            probabilities = _model_probabilities(train_matches, test_matches)
            if not probabilities:
                continue
        slips.extend(backtest_over_accumulators(test_matches, config=config, probabilities=probabilities))
    return tuple(slips)


def accumulator_report(slips: tuple[AccumulatorSlip, ...]) -> GoalBacktestReport:
    bets: list[GoalBet] = []
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for slip in slips:
        equity += slip.profit
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
        bets.append(
            GoalBet(
                match_id=f"{slip.card_date.isoformat()}:{len(slip.legs)}",
                market="over_acc",
                selection="+".join(f"{leg.market}:{leg.match_id}" for leg in slip.legs),
                odds=slip.combined_odds,
                probability=slip.combined_probability,
                expected_value=slip.expected_value,
                profit=slip.profit,
            )
        )
    return _report(bets, max_drawdown=abs(max_drawdown))


def format_over_accumulator_report(
    slips: tuple[AccumulatorSlip, ...],
    *,
    title: str,
) -> str:
    report = accumulator_report(slips)
    if not slips:
        return f"{title}\n{'-' * len(title)}\nslips=0"
    avg_legs = sum(len(slip.legs) for slip in slips) / len(slips)
    avg_odds = sum(slip.combined_odds for slip in slips) / len(slips)
    o05 = sum(1 for slip in slips for leg in slip.legs if leg.market == "O05")
    o15 = sum(1 for slip in slips for leg in slip.legs if leg.market == "O15")
    lines = [
        title,
        "-" * len(title),
        (
            f"slips={len(slips)}, profit={report.total_profit:+.2f}, roi={report.roi:+.2%}, "
            f"hit={report.hit_rate:.2%}, max_dd={report.max_drawdown:.2f}"
        ),
        f"avg_legs={avg_legs:.2f}, avg_odds={avg_odds:.2f}, o05_legs={o05}, o15_legs={o15}",
    ]
    by_season: dict[str, list[AccumulatorSlip]] = defaultdict(list)
    for slip in slips:
        by_season[slip.season].append(slip)
    for season, season_slips in by_season.items():
        season_report = accumulator_report(tuple(season_slips))
        lines.append(
            f"{season}: slips={len(season_slips)}, roi={season_report.roi:+.2%}, "
            f"profit={season_report.total_profit:+.2f}, hit={season_report.hit_rate:.2%}"
        )
    return "\n".join(lines)


def format_over_accumulator_walk_forward(
    matches: tuple[FootiqoMatch, ...],
    *,
    config: AccumulatorConfig,
) -> str:
    rule_slips = backtest_over_accumulators(matches, config=config)
    model_config = AccumulatorConfig(
        min_combined_odds=config.min_combined_odds,
        min_legs=config.min_legs,
        max_legs=config.max_legs,
        min_leg_odds=config.min_leg_odds,
        max_leg_odds=config.max_leg_odds,
        min_combined_ev=0.0,
    )
    model_slips = run_over_accumulator_walk_forward(matches, config=model_config, use_model=True)
    double_config = AccumulatorConfig(
        min_combined_odds=config.min_combined_odds,
        min_legs=2,
        max_legs=2,
        min_leg_odds=max(config.min_leg_odds, 1.12),
        max_leg_odds=min(config.max_leg_odds, 1.28),
        min_combined_ev=-1.0,
    )
    double_slips = backtest_over_accumulators(matches, config=double_config)
    return "\n\n".join(
        [
            format_over_accumulator_report(rule_slips, title="UEFA Over 0.5/1.5 Accumulators (rule)"),
            format_over_accumulator_report(double_slips, title="UEFA Over 1.5 Doubles (2 legs, 1.12-1.28)"),
            format_over_accumulator_report(model_slips, title="UEFA Over 0.5/1.5 Accumulators (model walk-forward)"),
        ]
    )


def _candidate_legs(
    matches: tuple[FootiqoMatch, ...],
    *,
    config: AccumulatorConfig,
    probabilities: dict[tuple[str, str], float] | None,
) -> list[AccumulatorLeg]:
    legs: list[AccumulatorLeg] = []
    for match in matches:
        best: AccumulatorLeg | None = None
        for market in OVER_ACC_MARKETS:
            odds = match.odds.get(market, 0.0)
            if odds < config.min_leg_odds or odds > config.max_leg_odds:
                continue
            probability = (probabilities or {}).get((match.match_id, market), _implied_probability(match, market))
            if probability <= 0:
                continue
            candidate = AccumulatorLeg(
                match_id=match.match_id,
                match_date=match.match_date,
                season=match.season,
                league=match.league,
                market=market,
                odds=odds,
                probability=probability,
                won=_market_won(match, market),
            )
            if best is None or _leg_rank(candidate) > _leg_rank(best):
                best = candidate
        if best is not None:
            legs.append(best)
    return sorted(legs, key=lambda leg: (-leg.probability, leg.odds))


def _leg_rank(leg: AccumulatorLeg) -> tuple[int, float]:
    # Prefer Over 1.5 when both markets qualify: it reaches 1.40 with fewer correlated 1.02 shots.
    return (1 if leg.market == "O15" else 0, leg.probability)


def _implied_probability(match: FootiqoMatch, market: str) -> float:
    over_odds = match.odds.get(market, 0.0)
    under_market = {"O05": "U05", "O15": "U15"}[market]
    under_odds = match.odds.get(under_market, 0.0)
    if over_odds <= 1.0:
        return 0.0
    if under_odds <= 1.0:
        return 1.0 / over_odds
    over_implied = 1.0 / over_odds
    under_implied = 1.0 / under_odds
    return over_implied / (over_implied + under_implied)


def _model_probabilities(
    train_matches: tuple[FootiqoMatch, ...],
    test_matches: tuple[FootiqoMatch, ...],
) -> dict[tuple[str, str], float]:
    probabilities: dict[tuple[str, str], float] = {}
    for market in OVER_ACC_MARKETS:
        train_rows = build_over_rows(train_matches, market)
        test_rows = build_over_rows(train_matches + test_matches, market)
        test_ids = {match.match_id for match in test_matches}
        test_rows = tuple(row for row in test_rows if row.match_id in test_ids)
        if len(set(labels(train_rows, market))) < 2 or not test_rows:
            continue
        model = train_over_model(train_rows, market)
        preds = model.predict_proba(feature_matrix(test_rows, OVER_FEATURES))
        for row, probability in zip(test_rows, preds, strict=True):
            probabilities[(row.match_id, market)] = float(probability[1])
    return probabilities


def _over_features(home_state: _TeamState, away_state: _TeamState, odds: float) -> dict[str, float]:
    home_for = home_state.average("goals_for")
    home_against = home_state.average("goals_against")
    away_for = away_state.average("goals_for")
    away_against = away_state.average("goals_against")
    return {
        "home_elo": home_state.elo,
        "away_elo": away_state.elo,
        "home_goals_for_avg": home_for,
        "home_goals_against_avg": home_against,
        "away_goals_for_avg": away_for,
        "away_goals_against_avg": away_against,
        "combined_goals_avg": home_for + home_against + away_for + away_against,
        "implied": 1.0 / odds if odds > 1.0 else 0.0,
    }


def _update_state(home_state: _TeamState, away_state: _TeamState, match: FootiqoMatch) -> None:
    goal_diff = match.home_goals - match.away_goals
    expected_home = 1.0 / (1.0 + 10 ** ((away_state.elo - home_state.elo) / 400))
    actual_home = 1.0 if goal_diff > 0 else 0.5 if goal_diff == 0 else 0.0
    adjustment = 24 * (actual_home - expected_home)
    home_state.elo += adjustment
    away_state.elo -= adjustment
    home_state.goals_for.append(float(match.home_goals))
    home_state.goals_against.append(float(match.away_goals))
    away_state.goals_for.append(float(match.away_goals))
    away_state.goals_against.append(float(match.home_goals))


def _card_date(match: FootiqoMatch) -> date:
    return parse_footiqo_date(match.match_date).date()


def _card_season(matches: tuple[FootiqoMatch, ...]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for match in matches:
        counts[match.season] += 1
    return max(counts, key=counts.__getitem__)


def _seasons_in_order(matches: tuple[FootiqoMatch, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for match in sorted(matches, key=lambda item: parse_footiqo_date(item.match_date)):
        if match.season not in seen:
            seen.append(match.season)
    return tuple(seen)
