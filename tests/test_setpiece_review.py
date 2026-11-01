from pathlib import Path

from football_quant.setpiece_review import format_setpiece_review, review_setpiece_markets


def test_setpiece_review_counts_corners_and_yellows(tmp_path: Path) -> None:
    _write_season(
        tmp_path / "1920" / "D1.csv",
        (
            ("Bayern", "Augsburg", "12", "4", "2", "1", "1.40", "7.00", "1.60"),
            ("Stuttgart", "Gladbach", "4", "3", "1", "1", "1.90", "3.80", "2.20"),
        ),
    )
    _write_season(
        tmp_path / "2425" / "D1.csv",
        (("Dortmund", "Frankfurt", "8", "5", "3", "2", "1.40", "6.50", "1.55"),),
    )
    reviews = review_setpiece_markets(tmp_path)
    over_c95 = next(
        review for review in reviews if review.league == "Bundesliga" and review.family == "over_c95"
    )
    assert over_c95.discovery.bets == 2
    assert over_c95.discovery.hit_rate == 0.5
    assert over_c95.holdout.bets == 1
    assert over_c95.holdout.hit_rate == 1.0
    over_y35 = next(
        review for review in reviews if review.league == "Bundesliga" and review.family == "over_y35"
    )
    assert over_y35.holdout.hit_rate == 1.0
    assert "proxy odds 1.85" in format_setpiece_review(reviews, tmp_path)


def _write_season(path: Path, matches: tuple[tuple[str, ...], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Div,HomeTeam,AwayTeam,HC,AC,HY,AY,AvgH,AvgA,Avg>2.5"]
    for match in matches:
        lines.append("D1," + ",".join(match))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
