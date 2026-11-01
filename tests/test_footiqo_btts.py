from football_quant.footiqo import FootiqoMatch
from football_quant.footiqo_btts_refine import _flat
from football_quant.footiqo_btts import (
    BTTS_FEATURES,
    BttsNoFilter,
    build_btts_rows,
    flat_btts_no_backtest,
    run_btts_no_walk_forward,
    run_fixed_btts_no_walk_forward,
    seasons_in_order,
)
from football_quant.goal_value import ValueConfig


def test_build_btts_rows_sorts_and_avoids_lookahead() -> None:
    matches = (
        _match("2", "02-09-16 20:45", "2016/2017", "Barcelona", "Celtic", 7, 0, bttsn=1.80),
        _match("1", "13-09-16 20:45", "2016/2017", "Barcelona", "Celtic", 1, 1, bttsn=2.10),
    )
    rows = build_btts_rows(matches)
    assert [row.match_id for row in rows] == ["2", "1"]
    first, second = rows
    assert first.features["home_goals_for_avg"] == 0.0
    assert second.features["home_goals_for_avg"] == 7.0
    assert first.labels["btts_no"] == 1
    assert second.labels["btts_no"] == 0
    assert set(BTTS_FEATURES) <= set(first.features)


def test_btts_no_walk_forward_runs_on_synthetic_seasons() -> None:
    matches = _synthetic_seasons()
    rows = build_btts_rows(matches)
    report = run_btts_no_walk_forward(
        rows,
        seasons=seasons_in_order(rows),
        model_weights=(0.5, 1.0),
        min_ev_grid=(0.0, 0.05),
        min_train_bets=4,
    )
    assert len(report.folds) == 2
    assert report.aggregate.total_staked >= 0
    baseline = flat_btts_no_backtest(rows)
    assert baseline.bets
    frozen = run_fixed_btts_no_walk_forward(rows, config=ValueConfig(1.0, 0.0, False))
    assert len(frozen.folds) == 2
    filtered = run_fixed_btts_no_walk_forward(
        rows,
        config=ValueConfig(1.0, 0.0, False),
        filters=BttsNoFilter("history", require_history=True, min_odds=2.00),
    )
    assert len(filtered.aggregate.bets) <= len(frozen.aggregate.bets)
    high = _flat(rows, 2.10, 20.0)
    assert high.bets <= len(rows)


def _synthetic_seasons() -> tuple[FootiqoMatch, ...]:
    matches: list[FootiqoMatch] = []
    match_id = 1
    for season_index, season in enumerate(("2015/2016", "2016/2017", "2017/2018")):
        year = 15 + season_index
        for week in range(8):
            home_goals = 2 if week % 2 == 0 else 0
            away_goals = 0 if week % 3 == 0 else 1
            matches.append(
                _match(
                    str(match_id),
                    f"{10 + week:02d}-09-{year:02d} 20:45",
                    season,
                    "Alpha" if week % 2 == 0 else "Beta",
                    "Gamma" if week % 2 == 0 else "Delta",
                    home_goals,
                    away_goals,
                    bttsn=1.90 + 0.05 * week,
                    bttsy=1.85,
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
    bttsy: float = 1.90,
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
            "BTTSN": bttsn,
            "BTTSY": bttsy,
            "H": 1.80,
            "D": 3.60,
            "A": 4.20,
            "O15": 1.20,
            "U15": 4.50,
            "O25": 1.70,
            "U25": 2.10,
            "O35": 2.60,
            "U35": 1.50,
        },
    )
