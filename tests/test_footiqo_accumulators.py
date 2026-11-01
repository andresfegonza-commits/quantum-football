from football_quant.footiqo import FootiqoMatch
from football_quant.footiqo_accumulators import (
    AccumulatorConfig,
    backtest_over_accumulators,
    build_card_accumulator,
    run_over_accumulator_walk_forward,
)


def test_accumulator_combines_safe_overs_until_min_odds() -> None:
    matches = (
        _match("1", "01-10-25 18:45", "2025/2026", "Alpha", "Beta", 2, 1, o05=1.04, o15=1.20),
        _match("2", "01-10-25 21:00", "2025/2026", "Gamma", "Delta", 1, 1, o05=1.03, o15=1.22),
        _match("3", "01-10-25 21:00", "2025/2026", "Echo", "Foxtrot", 0, 0, o05=1.10, o15=1.12),
    )
    slip = build_card_accumulator(matches, config=AccumulatorConfig())
    assert slip is not None
    assert slip.combined_odds >= 1.40
    assert len(slip.legs) >= 2
    assert all(leg.market == "O15" for leg in slip.legs)
    assert not slip.won
    assert slip.profit == -1.0


def test_accumulator_uses_one_market_per_match() -> None:
    matches = (_match("1", "01-10-25 18:45", "2025/2026", "Alpha", "Beta", 3, 0, o05=1.10, o15=1.20),)
    slip = build_card_accumulator(
        matches,
        config=AccumulatorConfig(min_legs=1, min_combined_odds=1.15),
    )
    assert slip is not None
    assert len(slip.legs) == 1
    assert slip.legs[0].market == "O15"


def test_backtest_groups_by_card_and_walk_forward_runs() -> None:
    matches = (
        _match("1", "01-09-24 18:45", "2024/2025", "Alpha", "Beta", 2, 0, o15=1.20),
        _match("2", "01-09-24 21:00", "2024/2025", "Gamma", "Delta", 1, 1, o15=1.21),
        _match("3", "02-09-25 18:45", "2025/2026", "Alpha", "Gamma", 2, 1, o15=1.18),
        _match("4", "02-09-25 21:00", "2025/2026", "Beta", "Delta", 3, 1, o15=1.19),
        _match("5", "03-09-25 18:45", "2025/2026", "Echo", "Foxtrot", 1, 1, o15=1.22),
        _match("6", "03-09-25 21:00", "2025/2026", "Alpha", "Delta", 2, 2, o15=1.20),
    )
    slips = backtest_over_accumulators(matches, config=AccumulatorConfig())
    assert len(slips) == 3
    assert all(slip.combined_odds >= 1.40 for slip in slips)
    assert all(slip.won for slip in slips)

    wf = run_over_accumulator_walk_forward(matches, config=AccumulatorConfig(), use_model=False)
    assert len(wf) == 2
    assert wf[0].season == "2025/2026"


def _match(
    match_id: str,
    match_date: str,
    season: str,
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    *,
    o05: float = 1.04,
    o15: float = 1.20,
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
        odds={
            "O05": o05,
            "U05": 10.0,
            "O15": o15,
            "U15": 4.50,
            "O25": 1.80,
            "U25": 2.00,
            "O35": 3.00,
            "U35": 1.40,
            "BTTSY": 1.80,
            "BTTSN": 2.00,
            "H": 1.90,
            "D": 3.50,
            "A": 4.00,
        },
    )
