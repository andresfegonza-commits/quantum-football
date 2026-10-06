from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .dataset import HistoricalDataset
from .domain import MarketQuote, MatchSnapshot, Result, TeamSnapshot


def build_sample_dataset() -> HistoricalDataset:
    base_time = datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc)
    matches = (
        MatchSnapshot(
            match_id="EPL-001",
            kickoff=base_time,
            competition="Premier League",
            home=TeamSnapshot("ars", "Arsenal", 1850, 1.1, 6, 0.1),
            away=TeamSnapshot("tot", "Tottenham", 1765, 0.2, 4, 0.2),
        ),
        MatchSnapshot(
            match_id="LAL-001",
            kickoff=base_time + timedelta(days=1),
            competition="La Liga",
            home=TeamSnapshot("rma", "Real Madrid", 1925, 1.4, 5, 0.0),
            away=TeamSnapshot("sev", "Sevilla", 1690, -0.4, 3, 0.3),
        ),
        MatchSnapshot(
            match_id="SER-001",
            kickoff=base_time + timedelta(days=2),
            competition="Serie A",
            home=TeamSnapshot("int", "Inter", 1875, 0.9, 4, 0.1),
            away=TeamSnapshot("nap", "Napoli", 1810, 0.7, 5, 0.1),
        ),
    )
    results = (
        Result("EPL-001", 2, 1),
        Result("LAL-001", 3, 0),
        Result("SER-001", 1, 1),
    )
    quotes = (
        MarketQuote("EPL-001", "sample-book", "home", -0.25, 1.92, base_time - timedelta(hours=8)),
        MarketQuote("EPL-001", "sample-book", "home", -0.5, 1.88, base_time - timedelta(minutes=10), True),
        MarketQuote("LAL-001", "sample-book", "home", -1.0, 1.95, base_time + timedelta(hours=12)),
        MarketQuote("LAL-001", "sample-book", "home", -1.25, 1.86, base_time + timedelta(days=1, minutes=-5), True),
        MarketQuote("SER-001", "sample-book", "away", +0.25, 1.91, base_time + timedelta(days=1, hours=5)),
        MarketQuote("SER-001", "sample-book", "away", +0.25, 1.84, base_time + timedelta(days=2, minutes=-7), True),
    )
    return HistoricalDataset(matches=matches, results=results, quotes=quotes)
