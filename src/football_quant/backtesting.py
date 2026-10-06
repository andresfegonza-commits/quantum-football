from __future__ import annotations

from dataclasses import dataclass

from .asian_handicap import settle_asian_handicap
from .dataset import HistoricalDataset
from .edge import calculate_edge
from .models import BaselineAsianHandicapModel


@dataclass(frozen=True)
class BacktestBet:
    match_id: str
    side: str
    line: float
    odds: float
    probability: float
    expected_value: float
    stake: float
    profit: float
    closing_line_value: float | None


@dataclass(frozen=True)
class BacktestReport:
    bets: tuple[BacktestBet, ...]
    total_staked: float
    total_profit: float
    roi: float
    hit_rate: float
    max_drawdown: float
    average_clv: float | None


def run_backtest(
    dataset: HistoricalDataset,
    *,
    model: BaselineAsianHandicapModel | None = None,
    min_expected_value: float = 0.02,
    flat_stake: float = 1.0,
) -> BacktestReport:
    if flat_stake <= 0:
        raise ValueError("flat_stake must be positive")

    model = model or BaselineAsianHandicapModel()
    matches = dataset.matches_by_id
    results = dataset.results_by_id
    opening_by_key = _first_quotes_by_selection(dataset.pre_match_quotes())
    closing_by_key = {
        (quote.match_id, quote.side, quote.line): quote for quote in dataset.closing_quotes()
    }

    bets: list[BacktestBet] = []
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for quote in sorted(dataset.pre_match_quotes(), key=lambda item: item.captured_at):
        match = matches[quote.match_id]
        result = results[quote.match_id]
        opening_quote = opening_by_key.get((quote.match_id, quote.side))
        estimate = model.predict(match, quote, opening_quote=opening_quote)
        edge = calculate_edge(quote, estimate)
        if edge.expected_value < min_expected_value:
            continue

        settlement = settle_asian_handicap(
            side=quote.side,
            line=quote.line,
            odds=quote.odds,
            home_goals=result.home_goals,
            away_goals=result.away_goals,
            stake=flat_stake,
        )
        equity += settlement.profit
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)

        closing_quote = closing_by_key.get((quote.match_id, quote.side, quote.line))
        clv = None
        if closing_quote is not None:
            clv = quote.odds - closing_quote.odds

        bets.append(
            BacktestBet(
                match_id=quote.match_id,
                side=quote.side,
                line=quote.line,
                odds=quote.odds,
                probability=estimate.fair_probability,
                expected_value=edge.expected_value,
                stake=flat_stake,
                profit=settlement.profit,
                closing_line_value=clv,
            )
        )

    total_staked = sum(bet.stake for bet in bets)
    total_profit = sum(bet.profit for bet in bets)
    wins = sum(1 for bet in bets if bet.profit > 0)
    clv_values = [bet.closing_line_value for bet in bets if bet.closing_line_value is not None]
    return BacktestReport(
        bets=tuple(bets),
        total_staked=total_staked,
        total_profit=total_profit,
        roi=total_profit / total_staked if total_staked else 0.0,
        hit_rate=wins / len(bets) if bets else 0.0,
        max_drawdown=abs(max_drawdown),
        average_clv=sum(clv_values) / len(clv_values) if clv_values else None,
    )


def _first_quotes_by_selection(quotes: tuple) -> dict[tuple[str, str], object]:
    first_quotes: dict[tuple[str, str], object] = {}
    for quote in sorted(quotes, key=lambda item: item.captured_at):
        first_quotes.setdefault((quote.match_id, quote.side), quote)
    return first_quotes
