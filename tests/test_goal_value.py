from pathlib import Path

from football_quant.football_data_uk import season_label_from_code
from football_quant.goal_markets import load_goal_market_rows_for_seasons
from football_quant.goal_value import run_market_aware_ou_walk_forward


def test_market_aware_ou_walk_forward_runs() -> None:
    codes = ("1920", "2021", "2122")
    rows = load_goal_market_rows_for_seasons(Path("data/raw/football-data"), season_codes=codes)
    report = run_market_aware_ou_walk_forward(
        rows,
        seasons=tuple(season_label_from_code(code) for code in codes),
        model_weights=(0.5, 1.0),
        min_ev_grid=(0.02, 0.05),
        min_train_bets=25,
    )
    assert len(report.folds) == 2
    assert report.aggregate.total_staked >= 0
