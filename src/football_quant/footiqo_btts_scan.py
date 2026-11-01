from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .footiqo import FootiqoMatch, load_footiqo_matches
from .footiqo_btts import (
    BTTS_FEATURES,
    LOCKED_BTTS_LEAGUES,
    build_btts_rows,
    evaluate_locked_btts_signal,
    parse_footiqo_date,
    train_btts_no_model,
)
from .goal_markets import feature_matrix
from .goal_models import GoalBet


PAPER_FIELDNAMES: tuple[str, ...] = (
    "match_id",
    "match_date",
    "league",
    "season",
    "home_team",
    "away_team",
    "odds_btts_no",
    "probability",
    "expected_value",
    "home_goals",
    "away_goals",
    "settled",
    "profit",
)


@dataclass(frozen=True)
class LockedBttsSignal:
    match: FootiqoMatch
    bet: GoalBet
    settled: bool


@dataclass(frozen=True)
class LockedBttsScanReport:
    current_season: str
    train_seasons: tuple[str, ...]
    train_matches: int
    scanned: int
    signals: tuple[LockedBttsSignal, ...]


def load_locked_league_matches(paths: tuple[Path, ...]) -> tuple[FootiqoMatch, ...]:
    matches = tuple(match for path in paths for match in load_footiqo_matches(path))
    return tuple(match for match in matches if match.league in LOCKED_BTTS_LEAGUES)


def scan_locked_btts_no(
    matches: tuple[FootiqoMatch, ...],
    *,
    current_season: str | None = None,
) -> LockedBttsScanReport:
    eligible = tuple(match for match in matches if match.league in LOCKED_BTTS_LEAGUES)
    if not eligible:
        raise ValueError("No Champions League or Europa League matches available")

    seasons = _seasons_from_matches(eligible)
    season = current_season or seasons[-1]
    train_seasons = tuple(item for item in seasons if item != season)
    if not train_seasons:
        raise ValueError("At least one prior season is required to train the locked BTTS No model")

    train_matches = tuple(match for match in eligible if match.season in train_seasons)
    scan_matches = tuple(match for match in eligible if match.season == season)
    train_rows = build_btts_rows(train_matches)
    model = train_btts_no_model(train_rows)

    scan_ids = {match.match_id for match in scan_matches}
    scan_rows = tuple(row for row in build_btts_rows(train_matches + scan_matches) if row.match_id in scan_ids)
    match_by_id = {match.match_id: match for match in scan_matches}
    probabilities = {
        row.match_id: float(probability[1])
        for row, probability in zip(scan_rows, model.predict_proba(feature_matrix(scan_rows, BTTS_FEATURES)), strict=True)
    }

    signals: list[LockedBttsSignal] = []
    for row in scan_rows:
        bet = evaluate_locked_btts_signal(row, probabilities[row.match_id])
        if bet is None:
            continue
        match = match_by_id[row.match_id]
        signals.append(LockedBttsSignal(match=match, bet=bet, settled=True))

    return LockedBttsScanReport(
        current_season=season,
        train_seasons=train_seasons,
        train_matches=len(train_matches),
        scanned=len(scan_rows),
        signals=tuple(signals),
    )


def format_locked_btts_scan(report: LockedBttsScanReport) -> str:
    lines = [
        "Locked BTTS No Scan",
        "-------------------",
        (
            f"rule=blend_50/50 ev>=2% odds=1.90-2.40 leagues={', '.join(LOCKED_BTTS_LEAGUES)} "
            f"stake=1u"
        ),
        (
            f"train_seasons={report.train_seasons[0]}..{report.train_seasons[-1]} "
            f"({report.train_matches} matches), scan={report.current_season} ({report.scanned} matches)"
        ),
        f"signals={len(report.signals)}",
    ]
    settled = [signal for signal in report.signals if signal.settled]
    if settled:
        profit = sum(signal.bet.profit for signal in settled)
        wins = sum(1 for signal in settled if signal.bet.profit > 0)
        lines.append(
            f"paper_settled={len(settled)}, profit={profit:+.2f}, "
            f"roi={profit / len(settled):+.2%}, hit={wins / len(settled):.2%}"
        )
    if not report.signals:
        lines.append("No current-season Champions/Europa matches pass the locked rule.")
        return "\n".join(lines)

    lines.append("")
    for signal in report.signals:
        match = signal.match
        status = "SETTLED" if signal.settled else "OPEN"
        lines.append(
            f"{status} {match.match_date} {match.league}: {match.home_team} vs {match.away_team} "
            f"BTTSN={signal.bet.odds:.2f} p={signal.bet.probability:.3f} ev={signal.bet.expected_value:+.3f} "
            f"score={match.home_goals}-{match.away_goals} profit={signal.bet.profit:+.2f}"
        )
    return "\n".join(lines)


def append_paper_signals(path: Path, signals: tuple[LockedBttsSignal, ...]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_paper_ids(path)
    recorded = 0
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAPER_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for signal in signals:
            if signal.match.match_id in existing:
                continue
            writer.writerow(_paper_row(signal))
            existing.add(signal.match.match_id)
            recorded += 1
    return recorded


def _paper_row(signal: LockedBttsSignal) -> dict[str, object]:
    match = signal.match
    return {
        "match_id": match.match_id,
        "match_date": match.match_date,
        "league": match.league,
        "season": match.season,
        "home_team": match.home_team,
        "away_team": match.away_team,
        "odds_btts_no": f"{signal.bet.odds:.2f}",
        "probability": f"{signal.bet.probability:.4f}",
        "expected_value": f"{signal.bet.expected_value:.4f}",
        "home_goals": match.home_goals,
        "away_goals": match.away_goals,
        "settled": int(signal.settled),
        "profit": f"{signal.bet.profit:.2f}",
    }


def _existing_paper_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["match_id"] for row in csv.DictReader(handle) if row.get("match_id")}


def _seasons_from_matches(matches: tuple[FootiqoMatch, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for match in sorted(matches, key=lambda item: parse_footiqo_date(item.match_date)):
        if match.season not in seen:
            seen.append(match.season)
    return tuple(seen)
