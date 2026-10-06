from football_quant.asian_handicap import settle_asian_handicap, split_quarter_line


def test_split_quarter_line() -> None:
    assert split_quarter_line(-0.25) == (-0.5, 0.0)
    assert split_quarter_line(0.75) == (0.5, 1.0)


def test_home_minus_quarter_win_half_on_draw() -> None:
    settlement = settle_asian_handicap(
        side="home",
        line=-0.25,
        odds=2.0,
        home_goals=1,
        away_goals=1,
        stake=10,
    )
    assert settlement.profit == -5
    assert [leg.outcome for leg in settlement.legs] == ["loss", "push"]


def test_away_plus_quarter_half_win_on_draw() -> None:
    settlement = settle_asian_handicap(
        side="away",
        line=0.25,
        odds=2.0,
        home_goals=1,
        away_goals=1,
        stake=10,
    )
    assert settlement.profit == 5
    assert [leg.outcome for leg in settlement.legs] == ["push", "win"]


def test_full_loss() -> None:
    settlement = settle_asian_handicap(
        side="home",
        line=-1.0,
        odds=1.9,
        home_goals=0,
        away_goals=1,
        stake=10,
    )
    assert settlement.profit == -10
