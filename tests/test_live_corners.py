from football_quant.live_corners import LiveCornerEntry, evaluate_live_corner_forward_test


def test_live_corner_forward_test_report() -> None:
    entries = (
        LiveCornerEntry(
            date="2026-10-06",
            competition="Premier League",
            match="Favorite vs Rival",
            favorite_team="Favorite",
            favorite_side="home",
            prematch_favorite_odds=1.35,
            minute=84,
            favorite_goals=0,
            opponent_goals=0,
            corners_at_entry=8,
            odds_corner_more=1.45,
            red_card_favorite=False,
            red_card_opponent=False,
            final_total_corners=9,
        ),
        LiveCornerEntry(
            date="2026-10-06",
            competition="Premier League",
            match="Favorite 2 vs Rival 2",
            favorite_team="Favorite 2",
            favorite_side="away",
            prematch_favorite_odds=1.48,
            minute=86,
            favorite_goals=1,
            opponent_goals=1,
            corners_at_entry=10,
            odds_corner_more=1.40,
            red_card_favorite=False,
            red_card_opponent=False,
            final_total_corners=10,
        ),
        LiveCornerEntry(
            date="2026-10-06",
            competition="Premier League",
            match="Favorite 3 vs Rival 3",
            favorite_team="Favorite 3",
            favorite_side="home",
            prematch_favorite_odds=1.30,
            minute=81,
            favorite_goals=2,
            opponent_goals=1,
            corners_at_entry=7,
            odds_corner_more=1.50,
            red_card_favorite=False,
            red_card_opponent=False,
            final_total_corners=8,
        ),
    )

    report = evaluate_live_corner_forward_test(entries)
    assert report.qualified == 2
    assert report.settled == 2
    assert report.wins == 1
    assert report.total_profit == -0.55
