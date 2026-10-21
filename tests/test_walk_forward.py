from pathlib import Path

from football_quant.walk_forward import run_walk_forward


def test_walk_forward_runs_across_seasons() -> None:
    report = run_walk_forward(
        Path("data/raw/football-data"),
        season_codes=("1920", "2021", "2122"),
        min_ev_grid=(0.0, 0.02),
        min_train_bets=25,
    )
    assert len(report.folds) == 2
    assert report.aggregate.bets >= 0
