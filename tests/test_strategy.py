from pathlib import Path

from football_quant.backtesting import run_backtest
from football_quant.football_data_uk import load_football_data_uk_csv
from football_quant.strategy import evaluate_strategy_scenarios


def test_strategy_scenarios_include_baseline() -> None:
    dataset = load_football_data_uk_csv(
        Path("data/raw/football-data/2324/E0.csv"),
        season="2023-2024",
        competition="Premier League",
    )
    report = run_backtest(dataset, min_expected_value=0.02)
    results = evaluate_strategy_scenarios(report.bets)
    by_name = {result.scenario.name: result.metrics for result in results}
    assert by_name["baseline"].bets == len(report.bets)
    assert by_name["odds_ge_2"].bets <= by_name["baseline"].bets
