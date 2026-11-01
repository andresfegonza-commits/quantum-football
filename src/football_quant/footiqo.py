from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FootiqoMatch:
    match_id: str
    match_date: str
    league: str
    season: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    odds: dict[str, float]


@dataclass(frozen=True)
class MarketBacktest:
    market: str
    bets: int
    profit: float
    roi: float
    hit_rate: float


ODDS_MARKETS: tuple[str, ...] = ("O05", "U05", "O15", "U15", "O25", "U25", "O35", "U35", "BTTSY", "BTTSN")
STORED_ODDS: tuple[str, ...] = ODDS_MARKETS + ("H", "D", "A")


def load_footiqo_matches(path: Path) -> tuple[FootiqoMatch, ...]:
    lines = [_normalize_line(line) for line in path.read_text(encoding="utf-8").splitlines()]
    result_rows = _parse_table(lines, ("id", "matchDate", "Country", "League", "Season", "homeTeam", "awayTeam", "referee", "FTHG", "FTAG", "FTR"))
    odds_rows = _parse_table(lines, ("id", "matchDate", "Country", "League", "Season", "homeTeam", "awayTeam", "H", "D", "A"))
    results = {
        row["id"]: row
        for row in result_rows
        if row.get("id", "").isdigit() and _is_int(row.get("FTHG", "")) and _is_int(row.get("FTAG", ""))
    }
    matches: list[FootiqoMatch] = []
    for row in odds_rows:
        if not row.get("id", "").isdigit():
            continue
        result = results.get(row["id"])
        if result is None:
            continue
        odds = {market: _float_or_zero(row.get(market, "")) for market in STORED_ODDS}
        matches.append(
            FootiqoMatch(
                match_id=row["id"],
                match_date=row["matchDate"],
                league=row["League"],
                season=row["Season"],
                home_team=row["homeTeam"],
                away_team=row["awayTeam"],
                home_goals=int(result["FTHG"]),
                away_goals=int(result["FTAG"]),
                odds=odds,
            )
        )
    return tuple(matches)


def backtest_footiqo_markets(matches: tuple[FootiqoMatch, ...]) -> tuple[MarketBacktest, ...]:
    reports = []
    for market in ODDS_MARKETS:
        profits: list[float] = []
        for match in matches:
            odds = match.odds.get(market, 0.0)
            if odds <= 1.0:
                continue
            won = _market_won(match, market)
            profits.append(odds - 1.0 if won else -1.0)
        total = sum(profits)
        wins = sum(1 for profit in profits if profit > 0)
        reports.append(
            MarketBacktest(
                market=market,
                bets=len(profits),
                profit=total,
                roi=total / len(profits) if profits else 0.0,
                hit_rate=wins / len(profits) if profits else 0.0,
            )
        )
    return tuple(sorted(reports, key=lambda report: report.roi, reverse=True))


def format_footiqo_backtest(matches: tuple[FootiqoMatch, ...], reports: tuple[MarketBacktest, ...]) -> str:
    lines = ["Footiqo UEFA Backtest", "---------------------", f"matches={len(matches)}"]
    for report in reports:
        lines.append(
            f"{report.market}: bets={report.bets}, profit={report.profit:+.2f}, "
            f"roi={report.roi:+.2%}, hit={report.hit_rate:.2%}"
        )
    return "\n".join(lines)


def _market_won(match: FootiqoMatch, market: str) -> bool:
    total_goals = match.home_goals + match.away_goals
    if market == "O05":
        return total_goals > 0.5
    if market == "U05":
        return total_goals < 0.5
    if market == "O15":
        return total_goals > 1.5
    if market == "U15":
        return total_goals < 1.5
    if market == "O25":
        return total_goals > 2.5
    if market == "U25":
        return total_goals < 2.5
    if market == "O35":
        return total_goals > 3.5
    if market == "U35":
        return total_goals < 3.5
    if market == "BTTSY":
        return match.home_goals > 0 and match.away_goals > 0
    if market == "BTTSN":
        return match.home_goals == 0 or match.away_goals == 0
    raise ValueError(f"Unsupported market: {market}")


def _parse_table(lines: list[str], expected_header: tuple[str, ...]) -> list[dict[str, str]]:
    header = None
    rows: list[dict[str, str]] = []
    for line in lines:
        if line.startswith("## "):
            header = None
            continue
        cells = _cells(line)
        if not cells:
            continue
        if tuple(cells[: len(expected_header)]) == expected_header:
            header = cells
            continue
        if cells and cells[0] == "id":
            header = None
            continue
        if header is None:
            continue
        if cells and cells[0] == "---":
            continue
        if len(cells) < len(header) or cells[:2] == ["matchDate", ""]:
            continue
        if cells[0] == "id":
            continue
        rows.append({name: cells[index] for index, name in enumerate(header) if index < len(cells)})
    return rows


def _normalize_line(line: str) -> str:
    return re.sub(r"^\s*\d+\|\|", "|", line.strip())


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _float_or_zero(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _is_int(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False
