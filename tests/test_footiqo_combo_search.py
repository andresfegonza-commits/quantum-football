from football_quant.footiqo import FootiqoMatch
from football_quant.footiqo_combo_search import (
    COMBO_FAMILIES,
    ComboFamily,
    LegSpec,
    backtest_combo_family,
    build_family_slip,
    format_combo_search,
    review_combo_families,
)


def test_favorite_double_picks_two_shortest_favorites() -> None:
    matches = (
        _match("1", "01-10-25 18:45", "2025/2026", "Champions League", home=1.25, away=4.00, hg=2, ag=0),
        _match("2", "01-10-25 21:00", "2025/2026", "Champions League", home=3.80, away=1.35, hg=0, ag=1),
        _match("3", "01-10-25 21:00", "2025/2026", "Champions League", home=1.80, away=2.10, hg=1, ag=1),
    )
    family = ComboFamily("fav_d2", (LegSpec("FAV", 1.20, 1.50),), 2, 2, 1.44)
    slip = build_family_slip(matches, family)
    assert slip is not None
    assert {leg.match_id for leg in slip.legs} == {"1", "2"}
    assert {leg.market for leg in slip.legs} == {"H", "A"}
    assert round(slip.combined_odds, 4) == round(1.25 * 1.35, 4)
    assert slip.won


def test_mix_requires_different_markets_and_matches() -> None:
    matches = (
        _match("1", "01-10-25 21:00", "2025/2026", "Europa League", home=1.30, u25=1.80, hg=2, ag=0),
        _match("2", "01-10-25 21:00", "2025/2026", "Europa League", home=2.20, u25=1.60, hg=0, ag=0),
        _match("3", "01-10-25 21:00", "2025/2026", "Europa League", home=1.40, u25=2.40, hg=3, ag=1),
    )
    family = ComboFamily(
        "mix_fav_u25",
        (LegSpec("FAV", 1.20, 1.50), LegSpec("U25", 1.50, 1.90)),
        2,
        2,
        1.80,
        True,
    )
    slip = build_family_slip(matches, family)
    assert slip is not None
    markets = {leg.market for leg in slip.legs}
    assert "U25" in markets
    assert "H" in markets
    assert len({leg.match_id for leg in slip.legs}) == 2


def test_family_skips_cards_without_enough_legs() -> None:
    matches = (_match("1", "01-10-25 21:00", "2025/2026", "Champions League", home=1.25, hg=1, ag=0),)
    family = ComboFamily("fav_d2", (LegSpec("FAV", 1.20, 1.50),), 2, 2, 1.44)
    assert build_family_slip(matches, family) is None


def test_review_splits_discovery_and_holdout() -> None:
    matches = (
        _match("1", "01-10-18 21:00", "2018/2019", "Champions League", home=1.25, away=4.00, hg=2, ag=0),
        _match("2", "01-10-18 21:00", "2018/2019", "Champions League", home=3.90, away=1.30, hg=0, ag=1),
        _match("3", "01-10-23 21:00", "2023/2024", "Champions League", home=1.22, away=4.20, hg=0, ag=1),
        _match("4", "01-10-23 21:00", "2023/2024", "Champions League", home=3.70, away=1.28, hg=1, ag=0),
    )
    family = ComboFamily("fav_d2", (LegSpec("FAV", 1.20, 1.50),), 2, 2, 1.44)
    reviews = review_combo_families(matches, families=(family,), universes=(("ucl", ("Champions League",)),))
    assert len(reviews) == 1
    assert reviews[0].discovery.slips == 1
    assert reviews[0].discovery.roi > 0
    assert reviews[0].holdout.slips == 1
    assert reviews[0].holdout.roi < 0
    text = format_combo_search(matches, reviews)
    assert "fav_d2" in text
    assert "Survivors" in text


def test_backtest_groups_by_card_date() -> None:
    matches = (
        _match("1", "01-10-25 18:45", "2025/2026", "Champions League", home=1.25, away=4.00, hg=1, ag=0),
        _match("2", "01-10-25 21:00", "2025/2026", "Champions League", home=1.30, away=3.80, hg=2, ag=0),
        _match("3", "02-10-25 21:00", "2025/2026", "Champions League", home=1.28, away=4.10, hg=0, ag=2),
        _match("4", "02-10-25 21:00", "2025/2026", "Champions League", home=1.32, away=3.90, hg=1, ag=0),
    )
    family = next(item for item in COMBO_FAMILIES if item.name == "fav_d2")
    slips = backtest_combo_family(matches, family)
    assert len(slips) == 2
    assert slips[0].won
    assert not slips[1].won


def _match(
    match_id: str,
    match_date: str,
    season: str,
    league: str,
    *,
    home: float = 1.90,
    away: float = 4.00,
    u25: float = 2.00,
    hg: int = 1,
    ag: int = 0,
) -> FootiqoMatch:
    return FootiqoMatch(
        match_id=match_id,
        match_date=match_date,
        league=league,
        season=season,
        home_team="Alpha",
        away_team="Beta",
        home_goals=hg,
        away_goals=ag,
        odds={
            "O05": 1.05,
            "U05": 10.0,
            "O15": 1.22,
            "U15": 4.20,
            "O25": 1.80,
            "U25": u25,
            "O35": 3.10,
            "U35": 1.35,
            "BTTSY": 1.75,
            "BTTSN": 2.05,
            "H": home,
            "D": 3.60,
            "A": away,
        },
    )
