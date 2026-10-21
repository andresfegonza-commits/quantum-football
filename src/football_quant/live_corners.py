from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


FIELDNAMES: tuple[str, ...] = (
    "date",
    "competition",
    "match",
    "favorite_team",
    "favorite_side",
    "prematch_favorite_odds",
    "minute",
    "favorite_goals",
    "opponent_goals",
    "corners_at_entry",
    "odds_corner_more",
    "red_card_favorite",
    "red_card_opponent",
    "final_total_corners",
    "notes",
)


@dataclass(frozen=True)
class LiveCornerEntry:
    date: str
    competition: str
    match: str
    favorite_team: str
    favorite_side: str
    prematch_favorite_odds: float
    minute: int
    favorite_goals: int
    opponent_goals: int
    corners_at_entry: int
    odds_corner_more: float
    red_card_favorite: bool
    red_card_opponent: bool
    final_total_corners: int | None
    notes: str = ""

    @property
    def favorite_is_winning(self) -> bool:
        return self.favorite_goals > self.opponent_goals

    @property
    def is_settled(self) -> bool:
        return self.final_total_corners is not None


@dataclass(frozen=True)
class LiveCornerDecision:
    entry: LiveCornerEntry
    qualifies: bool
    reasons: tuple[str, ...]
    won: bool | None
    profit: float | None


@dataclass(frozen=True)
class LiveCornerReport:
    decisions: tuple[LiveCornerDecision, ...]
    qualified: int
    settled: int
    wins: int
    total_profit: float
    roi: float
    hit_rate: float
    break_even_rate: float


def write_live_corner_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()


def load_live_corner_entries(path: Path) -> tuple[LiveCornerEntry, ...]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return tuple(_entry_from_row(row) for row in csv.DictReader(handle) if any(row.values()))


def evaluate_live_corner_forward_test(
    entries: tuple[LiveCornerEntry, ...],
    *,
    favorite_odds_max: float = 1.50,
    super_favorite_odds_max: float | None = None,
    min_minute: int = 84,
    min_corner_odds: float = 1.40,
    exclude_favorite_red_card: bool = True,
) -> LiveCornerReport:
    decisions = tuple(
        evaluate_live_corner_entry(
            entry,
            favorite_odds_max=favorite_odds_max,
            super_favorite_odds_max=super_favorite_odds_max,
            min_minute=min_minute,
            min_corner_odds=min_corner_odds,
            exclude_favorite_red_card=exclude_favorite_red_card,
        )
        for entry in entries
    )
    settled_decisions = [decision for decision in decisions if decision.qualifies and decision.profit is not None]
    total_profit = sum(decision.profit or 0.0 for decision in settled_decisions)
    wins = sum(1 for decision in settled_decisions if decision.won)
    settled = len(settled_decisions)
    qualified = sum(1 for decision in decisions if decision.qualifies)
    average_odds = (
        sum(decision.entry.odds_corner_more for decision in settled_decisions) / settled if settled else min_corner_odds
    )
    return LiveCornerReport(
        decisions=decisions,
        qualified=qualified,
        settled=settled,
        wins=wins,
        total_profit=total_profit,
        roi=total_profit / settled if settled else 0.0,
        hit_rate=wins / settled if settled else 0.0,
        break_even_rate=1.0 / average_odds if average_odds > 1.0 else 0.0,
    )


def evaluate_live_corner_entry(
    entry: LiveCornerEntry,
    *,
    favorite_odds_max: float,
    super_favorite_odds_max: float | None,
    min_minute: int,
    min_corner_odds: float,
    exclude_favorite_red_card: bool,
) -> LiveCornerDecision:
    reasons: list[str] = []
    odds_limit = super_favorite_odds_max or favorite_odds_max
    if entry.prematch_favorite_odds > odds_limit:
        reasons.append("favorite odds above threshold")
    if entry.minute < min_minute:
        reasons.append("entry before minimum minute")
    if entry.favorite_is_winning:
        reasons.append("favorite is already winning")
    if entry.odds_corner_more < min_corner_odds:
        reasons.append("corner odds below threshold")
    if exclude_favorite_red_card and entry.red_card_favorite:
        reasons.append("favorite has red card")

    qualifies = not reasons
    won = None
    profit = None
    if qualifies and entry.final_total_corners is not None:
        won = entry.final_total_corners > entry.corners_at_entry
        profit = entry.odds_corner_more - 1.0 if won else -1.0

    return LiveCornerDecision(entry=entry, qualifies=qualifies, reasons=tuple(reasons), won=won, profit=profit)


def format_live_corner_report(report: LiveCornerReport) -> str:
    lines = ["Live Corner Forward Test", "------------------------"]
    lines.append(f"qualified={report.qualified}, settled={report.settled}, wins={report.wins}")
    lines.append(
        f"profit={report.total_profit:+.2f}, roi={report.roi:+.2%}, "
        f"hit_rate={report.hit_rate:.2%}, break_even={report.break_even_rate:.2%}"
    )
    pending = report.qualified - report.settled
    if pending:
        lines.append(f"pending={pending}")
    rejected = len(report.decisions) - report.qualified
    if rejected:
        lines.append(f"rejected={rejected}")
    return "\n".join(lines)


def _entry_from_row(row: dict[str, str]) -> LiveCornerEntry:
    return LiveCornerEntry(
        date=row["date"],
        competition=row["competition"],
        match=row["match"],
        favorite_team=row["favorite_team"],
        favorite_side=row["favorite_side"],
        prematch_favorite_odds=float(row["prematch_favorite_odds"]),
        minute=int(row["minute"]),
        favorite_goals=int(row["favorite_goals"]),
        opponent_goals=int(row["opponent_goals"]),
        corners_at_entry=int(row["corners_at_entry"]),
        odds_corner_more=float(row["odds_corner_more"]),
        red_card_favorite=_parse_bool(row.get("red_card_favorite", "")),
        red_card_opponent=_parse_bool(row.get("red_card_opponent", "")),
        final_total_corners=_parse_optional_int(row.get("final_total_corners", "")),
        notes=row.get("notes", ""),
    )


def _parse_bool(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"true", "1", "yes", "y", "si", "sí"})


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(value)
