from pathlib import Path

from football_quant.domestic_review import format_domestic_review, review_domestic_leagues


def test_domestic_review_splits_discovery_and_holdout(tmp_path: Path) -> None:
    _write_season(
        tmp_path / "1920" / "D1.csv",
        (
            ("Bayern", "Augsburg", "1", "0", "H", "1.40", "7.00", "1.80", "2.00"),
            ("Stuttgart", "Gladbach", "2", "1", "H", "1.45", "6.50", "1.70", "2.10"),
        ),
    )
    _write_season(
        tmp_path / "2425" / "D1.csv",
        (("Stuttgart", "Koln", "0", "1", "A", "1.42", "7.20", "1.75", "2.05"),),
    )
    reviews = review_domestic_leagues(tmp_path)
    bundesliga = [review for review in reviews if review.league == "Bundesliga" and review.family == "home_fav_120_170"]
    assert len(bundesliga) == 1
    assert bundesliga[0].discovery.bets == 2
    assert bundesliga[0].discovery.roi > 0
    assert bundesliga[0].holdout.bets == 1
    assert bundesliga[0].holdout.roi < 0
    assert "Bundesliga" in format_domestic_review(reviews)


def _write_season(path: Path, matches: tuple[tuple[str, ...], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Div,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgA,Avg>2.5,Avg<2.5",
    ]
    for match in matches:
        lines.append("D1," + ",".join(match))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
