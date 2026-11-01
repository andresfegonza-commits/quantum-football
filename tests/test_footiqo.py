from pathlib import Path

from football_quant.footiqo import backtest_footiqo_markets, load_footiqo_matches


def test_load_footiqo_champions_matches() -> None:
    matches = load_footiqo_matches(Path("data/raw/footiqo/champions-league.md"))
    assert matches
    assert matches[0].league == "Champions League"
    assert "O25" in matches[0].odds


def test_backtest_footiqo_markets() -> None:
    matches = load_footiqo_matches(Path("data/raw/footiqo/champions-league.md"))
    reports = backtest_footiqo_markets(matches)
    assert reports
    assert all(report.bets > 0 for report in reports)
