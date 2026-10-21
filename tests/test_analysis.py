from pathlib import Path

from football_quant.analysis import line_key, summarize_segments
from football_quant.backtesting import run_backtest
from football_quant.football_data_uk import load_football_data_uk_csv


def test_segment_summary_from_real_csv() -> None:
    dataset = load_football_data_uk_csv(
        Path("data/raw/football-data/2324/E0.csv"),
        season="2023-2024",
        competition="Premier League",
    )
    report = run_backtest(dataset, min_expected_value=0.02)
    segments = summarize_segments(report.bets, line_key)
    assert segments
    assert sum(segment.bets for segment in segments) == len(report.bets)
