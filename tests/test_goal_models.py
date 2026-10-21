from pathlib import Path

from football_quant.goal_markets import load_goal_market_rows_for_seasons
from football_quant.goal_models import backtest_over_under_25, evaluate_goal_model, run_goal_walk_forward, train_goal_model


def test_goal_market_rows_load_labels_and_odds() -> None:
    rows = load_goal_market_rows_for_seasons(Path("data/raw/football-data"), season_codes=("2324",))
    assert len(rows) == 380
    assert {"over_25", "btts", "first_half_over_05"}.issubset(rows[0].labels)
    assert rows[0].odds["opening_over_25"] > 1


def test_goal_model_training_and_ou_backtest() -> None:
    rows = load_goal_market_rows_for_seasons(Path("data/raw/football-data"), season_codes=("1920", "2021"))
    train_rows = tuple(row for row in rows if row.season == "2019-2020")
    test_rows = tuple(row for row in rows if row.season == "2020-2021")
    model = train_goal_model(train_rows, "over_25")
    metrics = evaluate_goal_model(model, test_rows, "over_25")
    report = backtest_over_under_25(model, test_rows, min_expected_value=0.02)
    assert metrics.samples == 380
    assert report.total_staked >= 0


def test_goal_walk_forward_runs() -> None:
    rows = load_goal_market_rows_for_seasons(
        Path("data/raw/football-data"),
        season_codes=("1920", "2021", "2122"),
    )
    report = run_goal_walk_forward(
        rows,
        seasons=("2019-2020", "2020-2021", "2021-2022"),
        min_expected_value=0.02,
    )
    assert len(report.folds) == 2
    assert report.aggregate_over_under.total_staked >= 0
