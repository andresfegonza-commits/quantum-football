from pathlib import Path

from football_quant.footiqo import FootiqoMatch
from football_quant.footiqo_btts import evaluate_locked_btts_signal
from football_quant.footiqo_btts_scan import append_paper_signals, scan_locked_btts_no
from football_quant.goal_markets import GoalMarketRow


def test_locked_rule_accepts_only_190_240_band() -> None:
    accepted = evaluate_locked_btts_signal(_row(2.10), 0.62)
    assert accepted is not None
    assert accepted.odds == 2.10
    assert accepted.expected_value >= 0.02
    assert evaluate_locked_btts_signal(_row(1.70), 0.62) is None
    assert evaluate_locked_btts_signal(_row(2.80), 0.62) is None


def test_scan_uses_prior_seasons_and_skips_conference(tmp_path: Path) -> None:
    matches = _two_seasons() + (
        _match("99", "10-09-18 20:45", "2017/2018", "Alpha", "Gamma", 1, 0, bttsn=2.10),
        FootiqoMatch(
            match_id="conf",
            match_date="10-09-18 20:45",
            league="Conference League",
            season="2017/2018",
            home_team="Zeta",
            away_team="Omega",
            home_goals=0,
            away_goals=0,
            odds={"BTTSN": 2.10, "BTTSY": 1.75, "H": 1.70, "D": 3.50, "A": 5.00},
        ),
    )
    report = scan_locked_btts_no(matches, current_season="2017/2018")
    assert report.current_season == "2017/2018"
    assert "2017/2018" not in report.train_seasons
    assert all(signal.match.league != "Conference League" for signal in report.signals)
    assert all(1.90 <= signal.bet.odds <= 2.40 for signal in report.signals)

    path = tmp_path / "paper.csv"
    first = append_paper_signals(path, report.signals)
    second = append_paper_signals(path, report.signals)
    assert first == len(report.signals)
    assert second == 0


def _two_seasons() -> tuple[FootiqoMatch, ...]:
    matches: list[FootiqoMatch] = []
    match_id = 1
    for season_index, season in enumerate(("2015/2016", "2016/2017", "2017/2018")):
        year = 15 + season_index
        for week in range(8):
            matches.append(
                _match(
                    str(match_id),
                    f"{10 + week:02d}-09-{year:02d} 20:45",
                    season,
                    "Alpha" if week % 2 == 0 else "Beta",
                    "Gamma" if week % 2 == 0 else "Delta",
                    2 if week % 2 == 0 else 0,
                    0 if week % 3 == 0 else 1,
                    bttsn=1.95 + 0.04 * week,
                )
            )
            match_id += 1
    return tuple(matches)


def _match(
    match_id: str,
    match_date: str,
    season: str,
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    *,
    bttsn: float,
) -> FootiqoMatch:
    return FootiqoMatch(
        match_id=match_id,
        match_date=match_date,
        league="Champions League",
        season=season,
        home_team=home_team,
        away_team=away_team,
        home_goals=home_goals,
        away_goals=away_goals,
        odds={"BTTSN": bttsn, "BTTSY": 1.80, "H": 1.80, "D": 3.60, "A": 4.20},
    )


def _row(odds_no: float) -> GoalMarketRow:
    return GoalMarketRow(
        season="2026/2027",
        match_id="x",
        features={},
        labels={"btts_no": 1},
        odds={"btts_no": odds_no, "btts_yes": 1.75},
    )
