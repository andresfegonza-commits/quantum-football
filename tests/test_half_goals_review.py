from pathlib import Path

from football_quant.half_goals_review import format_half_goals_review, review_half_goal_markets


def test_half_goals_review_splits_halves_and_open_filter(tmp_path: Path) -> None:
    _write_season(
        tmp_path / "1920" / "D1.csv",
        (
            ("Bayern", "Augsburg", "2", "1", "1", "0", "1.55"),
            ("Stuttgart", "Gladbach", "0", "1", "0", "0", "1.60"),
        ),
    )
    _write_season(
        tmp_path / "2425" / "D1.csv",
        (("Dortmund", "Frankfurt", "3", "1", "1", "1", "1.50"),),
    )
    reviews = review_half_goal_markets(tmp_path)
    first_half = next(
        review for review in reviews if review.league == "Bundesliga" and review.family == "1h_over_05"
    )
    assert first_half.discovery.bets == 2
    assert first_half.discovery.hit_rate == 0.5
    assert first_half.holdout.hit_rate == 1.0

    second_half = next(
        review for review in reviews if review.league == "Bundesliga" and review.family == "2h_over_15"
    )
    assert second_half.discovery.hit_rate == 0.5
    assert second_half.holdout.hit_rate == 1.0

    ht00 = next(
        review for review in reviews if review.league == "Bundesliga" and review.family == "ht00_2h_over_05"
    )
    assert ht00.discovery.bets == 1
    assert ht00.discovery.hit_rate == 1.0

    open_2h = next(
        review for review in reviews if review.league == "Bundesliga" and review.family == "open_2h_over_15"
    )
    assert open_2h.discovery.bets == 2
    assert "proxy prices" in format_half_goals_review(reviews, tmp_path)


def _write_season(path: Path, matches: tuple[tuple[str, ...], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Div,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG,Avg>2.5"]
    for match in matches:
        lines.append("D1," + ",".join(match))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
