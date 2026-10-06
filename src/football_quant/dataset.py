from __future__ import annotations

from dataclasses import dataclass

from .domain import MarketQuote, MatchSnapshot, Result


@dataclass(frozen=True)
class HistoricalDataset:
    matches: tuple[MatchSnapshot, ...]
    results: tuple[Result, ...]
    quotes: tuple[MarketQuote, ...]

    def __post_init__(self) -> None:
        match_ids = {match.match_id for match in self.matches}
        result_ids = {result.match_id for result in self.results}
        quote_ids = {quote.match_id for quote in self.quotes}
        missing_results = match_ids - result_ids
        missing_matches = (result_ids | quote_ids) - match_ids
        if missing_results:
            raise ValueError(f"Missing results for matches: {sorted(missing_results)}")
        if missing_matches:
            raise ValueError(f"Missing match rows for ids: {sorted(missing_matches)}")

    @property
    def matches_by_id(self) -> dict[str, MatchSnapshot]:
        return {match.match_id: match for match in self.matches}

    @property
    def results_by_id(self) -> dict[str, Result]:
        return {result.match_id: result for result in self.results}

    def quotes_for_match(self, match_id: str) -> tuple[MarketQuote, ...]:
        return tuple(quote for quote in self.quotes if quote.match_id == match_id)

    def pre_match_quotes(self) -> tuple[MarketQuote, ...]:
        matches = self.matches_by_id
        return tuple(
            quote
            for quote in self.quotes
            if quote.captured_at <= matches[quote.match_id].kickoff and not quote.is_closing
        )

    def closing_quotes(self) -> tuple[MarketQuote, ...]:
        return tuple(quote for quote in self.quotes if quote.is_closing)
