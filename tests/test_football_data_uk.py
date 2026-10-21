from pathlib import Path

from football_quant.backtesting import run_backtest
from football_quant.football_data_uk import load_football_data_uk_csv, load_football_data_uk_seasons


def test_load_football_data_uk_csv() -> None:
    csv_path = Path("data/raw/football-data/2324/E0.csv")
    dataset = load_football_data_uk_csv(
        csv_path,
        season="2023-2024",
        competition="Premier League",
    )
    assert len(dataset.matches) == 380
    assert len(dataset.results) == 380
    assert len(dataset.pre_match_quotes()) > 0
    assert len(dataset.closing_quotes()) > 0


def test_backtest_football_data_uk_csv() -> None:
    dataset = load_football_data_uk_csv(
        Path("data/raw/football-data/2324/E0.csv"),
        season="2023-2024",
        competition="Premier League",
    )
    report = run_backtest(dataset, min_expected_value=0.05)
    assert report.total_staked >= 0


def test_load_multiple_football_data_uk_seasons() -> None:
    dataset = load_football_data_uk_seasons(
        Path("data/raw/football-data"),
        season_codes=("1920", "2021", "2122", "2223", "2324", "2425"),
    )
    assert len(dataset.matches) == 2280
    assert len(dataset.results) == 2280
    assert len(dataset.pre_match_quotes()) > 0
