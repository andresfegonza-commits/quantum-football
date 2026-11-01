from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


PAPER_FIELDNAMES: tuple[str, ...] = (
    "bet_id",
    "card_date",
    "league",
    "strategy",
    "book",
    "stake",
    "combined_odds",
    "rule_min_combined",
    "rule_compliant",
    "legs",
    "status",
    "profit",
    "notes",
)
DEFAULT_PAPER_PATH = Path("data/forward/paper_bets.csv")


@dataclass(frozen=True)
class PaperBet:
    bet_id: str
    card_date: str
    league: str
    strategy: str
    book: str
    stake: float
    combined_odds: float
    rule_min_combined: float
    rule_compliant: bool
    legs: str
    status: str = "open"
    profit: float | None = None
    notes: str = ""


def append_paper_bets(path: Path, bets: tuple[PaperBet, ...]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _existing_ids(path)
    recorded = 0
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAPER_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for bet in bets:
            if bet.bet_id in existing:
                continue
            writer.writerow(_row(bet))
            existing.add(bet.bet_id)
            recorded += 1
    return recorded


def load_paper_bets(path: Path) -> tuple[PaperBet, ...]:
    if not path.exists():
        return ()
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(_from_row(row) for row in csv.DictReader(handle) if row.get("bet_id"))


def settle_paper_bet(path: Path, bet_id: str, *, won: bool, notes: str | None = None) -> PaperBet:
    bets = list(load_paper_bets(path))
    updated: list[PaperBet] = []
    settled: PaperBet | None = None
    for bet in bets:
        if bet.bet_id != bet_id:
            updated.append(bet)
            continue
        profit = bet.stake * (bet.combined_odds - 1.0) if won else -bet.stake
        settled = PaperBet(
            bet_id=bet.bet_id,
            card_date=bet.card_date,
            league=bet.league,
            strategy=bet.strategy,
            book=bet.book,
            stake=bet.stake,
            combined_odds=bet.combined_odds,
            rule_min_combined=bet.rule_min_combined,
            rule_compliant=bet.rule_compliant,
            legs=bet.legs,
            status="settled",
            profit=profit,
            notes=bet.notes if notes is None else notes,
        )
        updated.append(settled)
    if settled is None:
        raise KeyError(f"Unknown paper bet: {bet_id}")
    _rewrite(path, tuple(updated))
    return settled


def format_paper_ledger(bets: tuple[PaperBet, ...]) -> str:
    lines = ["Paper Ledger", "------------"]
    if not bets:
        lines.append("bets=0")
        return "\n".join(lines)
    settled = tuple(bet for bet in bets if bet.status == "settled" and bet.profit is not None)
    open_bets = tuple(bet for bet in bets if bet.status == "open")
    staked = sum(bet.stake for bet in settled)
    profit = sum(bet.profit or 0.0 for bet in settled)
    wins = sum(1 for bet in settled if (bet.profit or 0.0) > 0)
    lines.append(f"open={len(open_bets)}, settled={len(settled)}")
    if settled:
        lines.append(
            f"settled_stake={staked:.2f}, profit={profit:+.2f}, "
            f"roi={profit / staked:+.2%}, hit={wins / len(settled):.2%}"
        )
    lines.append("")
    for bet in bets:
        profit_text = "pending" if bet.profit is None else f"{bet.profit:+.2f}"
        flag = "" if bet.rule_compliant else " below-rule-min"
        lines.append(
            f"{bet.status.upper()} {bet.card_date} {bet.league} {bet.strategy}: "
            f"{bet.legs} @{bet.combined_odds:.3f} {bet.book} stake={bet.stake:.1f} "
            f"profit={profit_text}{flag}"
        )
    return "\n".join(lines)


def stuttgart_win_bet() -> PaperBet:
    return PaperBet(
        bet_id="2026-10-23-d1-stuttgart-win",
        card_date="2026-10-23",
        league="Bundesliga",
        strategy="paper_fav_single",
        book="1xBet",
        stake=1.0,
        combined_odds=1.42,
        rule_min_combined=1.20,
        rule_compliant=False,
        legs="Stuttgart win (vs Gladbach)",
        notes="Paper single. Favorite 1.20-1.70, no same-league double. Not a locked rule. Final 2-3.",
    )


def madrid_inter_combo_bet() -> PaperBet:
    return PaperBet(
        bet_id="2026-10-21-ucl-fav_d2-madrid-inter",
        card_date="2026-10-21",
        league="Champions League",
        strategy="fav_d2_wide",
        book="1xBet",
        stake=1.0,
        combined_odds=1.611,
        rule_min_combined=1.44,
        rule_compliant=True,
        legs="Real Madrid win (vs Leipzig) + Inter win (vs Shakhtar)",
        notes="Madrid 4-1, Inter 1-0.",
    )


def villa_btts_no_bet() -> PaperBet:
    return PaperBet(
        bet_id="2026-10-21-ucl-bttsn-villa-viking",
        card_date="2026-10-21",
        league="Champions League",
        strategy="locked_btts_no",
        book="1xBet",
        stake=1.0,
        combined_odds=2.20,
        rule_min_combined=1.90,
        rule_compliant=True,
        legs="BTTS No (Aston Villa vs Viking)",
        notes="Scan quote 2.20 in 1.90-2.40 band. Final 3-1; both teams scored.",
    )


def saturday_singles_bets() -> tuple[PaperBet, ...]:
    return (
        PaperBet(
            bet_id="2026-10-24-i1-lazio-win",
            card_date="2026-10-24",
            league="Serie A",
            strategy="paper_fav_single",
            book="1xBet",
            stake=1.0,
            combined_odds=1.524,
            rule_min_combined=1.20,
            rule_compliant=False,
            legs="Lazio win (vs Parma)",
            notes="Paper single. Favorite 1.20-1.70. Serie A home favs lose historically. Not a locked rule. Final 2-1.",
        ),
        PaperBet(
            bet_id="2026-10-24-d1-dortmund-win",
            card_date="2026-10-24",
            league="Bundesliga",
            strategy="paper_fav_single",
            book="1xBet",
            stake=1.0,
            combined_odds=1.443,
            rule_min_combined=1.20,
            rule_compliant=False,
            legs="Dortmund win (vs Frankfurt)",
            notes="Paper single. Favorite 1.20-1.70. Bundesliga favs lose historically. Not a locked rule. Final 1-0.",
        ),
        PaperBet(
            bet_id="2026-10-24-sp1-atletico-win",
            card_date="2026-10-24",
            league="La Liga",
            strategy="paper_fav_single",
            book="1xBet",
            stake=1.0,
            combined_odds=1.341,
            rule_min_combined=1.20,
            rule_compliant=False,
            legs="Atletico Madrid win (vs Deportivo)",
            notes="Paper single. Favorite 1.20-1.70. Matches the surviving La Liga home-fav slice, still not locked. Final 1-0.",
        ),
    )


def sunday_nov1_bets() -> tuple[PaperBet, ...]:
    return (
        PaperBet(
            bet_id="2026-11-01-f1-lyon-win",
            card_date="2026-11-01",
            league="Ligue 1",
            strategy="paper_fav_single",
            book="1xBet",
            stake=1.0,
            combined_odds=1.44,
            rule_min_combined=1.20,
            rule_compliant=False,
            legs="Lyon win (vs Angers)",
            notes="Paper single. Favorite 1.20-1.70. Ligue 1 home favs are not locked.",
        ),
        PaperBet(
            bet_id="2026-11-01-sp1-madrid-1h-over25",
            card_date="2026-11-01",
            league="La Liga",
            strategy="paper_1h_over_25",
            book="1xBet",
            stake=1.0,
            combined_odds=1.64,
            rule_min_combined=2.00,
            rule_compliant=False,
            legs="1H Over 2.5 (Racing Santander vs Real Madrid)",
            notes="Paper 1H Over 2.5. Not a locked rule. HT 0-3.",
        ),
    )


def pending_paper_bets() -> tuple[PaperBet, ...]:
    return todays_combo_bets() + (stuttgart_win_bet(),) + saturday_singles_bets() + sunday_nov1_bets()


def todays_combo_bets() -> tuple[PaperBet, ...]:
    return (
        PaperBet(
            bet_id="2026-10-22-el-fav_d2-leverkusen-benfica",
            card_date="2026-10-22",
            league="Europa League",
            strategy="fav_d2_wide",
            book="1xBet",
            stake=1.0,
            combined_odds=1.632,
            rule_min_combined=1.44,
            rule_compliant=True,
            legs="Leverkusen win (OFI vs Leverkusen) + Benfica win (Omonia vs Benfica)",
        ),
        PaperBet(
            bet_id="2026-10-22-conf-fav_d2-brighton-twente",
            card_date="2026-10-22",
            league="Conference League",
            strategy="fav_d2_wide",
            book="1xBet",
            stake=1.0,
            combined_odds=1.394,
            rule_min_combined=1.44,
            rule_compliant=False,
            legs="Brighton win (Jablonec vs Brighton) + Twente win (Lincoln vs Twente)",
            notes="Actual combined 1.394 is below the 1.44 historical minimum.",
        ),
    )


def _row(bet: PaperBet) -> dict[str, object]:
    return {
        "bet_id": bet.bet_id,
        "card_date": bet.card_date,
        "league": bet.league,
        "strategy": bet.strategy,
        "book": bet.book,
        "stake": f"{bet.stake:.2f}",
        "combined_odds": f"{bet.combined_odds:.3f}",
        "rule_min_combined": f"{bet.rule_min_combined:.2f}",
        "rule_compliant": int(bet.rule_compliant),
        "legs": bet.legs,
        "status": bet.status,
        "profit": "" if bet.profit is None else f"{bet.profit:.2f}",
        "notes": bet.notes,
    }


def _from_row(row: dict[str, str]) -> PaperBet:
    profit_raw = row.get("profit", "").strip()
    return PaperBet(
        bet_id=row["bet_id"],
        card_date=row["card_date"],
        league=row["league"],
        strategy=row["strategy"],
        book=row["book"],
        stake=float(row["stake"]),
        combined_odds=float(row["combined_odds"]),
        rule_min_combined=float(row["rule_min_combined"]),
        rule_compliant=row["rule_compliant"] in {"1", "true", "True"},
        legs=row["legs"],
        status=row["status"],
        profit=None if profit_raw == "" else float(profit_raw),
        notes=row.get("notes", ""),
    )


def _existing_ids(path: Path) -> set[str]:
    return {bet.bet_id for bet in load_paper_bets(path)}


def _rewrite(path: Path, bets: tuple[PaperBet, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAPER_FIELDNAMES)
        writer.writeheader()
        for bet in bets:
            writer.writerow(_row(bet))
