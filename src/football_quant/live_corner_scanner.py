from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .live_corners import FIELDNAMES, LiveCornerEntry, evaluate_live_corner_entry, load_live_corner_entries


class LiveOddsClient(Protocol):
    def fixtures(self, **params: Any) -> dict[str, Any]: ...

    def fixture(self, fixture_id: str | int, **params: Any) -> dict[str, Any]: ...

    def fixture_odds(self, fixture_id: str | int, market: str | None = None) -> dict[str, Any]: ...

    def fixture_events(self, fixture_id: str | int) -> dict[str, Any]: ...


_DETAIL_CACHE: dict[str, dict[str, Any]] = {}
_SEEN_LIVE: set[str] = set()
_ARCHIVED_EVENTS: set[str] = set()


def reset_live_watch_state() -> None:
    _DETAIL_CACHE.clear()
    _SEEN_LIVE.clear()
    _ARCHIVED_EVENTS.clear()


@dataclass(frozen=True)
class LiveCornerScanResult:
    fixture_id: str
    match: str
    qualifies: bool
    reasons: tuple[str, ...]
    entry: LiveCornerEntry | None


def scan_live_corner_window(
    client: LiveOddsClient,
    *,
    watch_minute: int = 80,
    min_minute: int = 84,
    favorite_odds_max: float = 1.50,
    min_corner_odds: float = 1.40,
    record_path: Path | None = None,
    snapshot_path: Path | None = None,
    events_path: Path | None = None,
) -> tuple[tuple[LiveCornerScanResult, ...], int]:
    payload = client.fixtures(status="live")
    fixtures = payload.get("data", []) if isinstance(payload, dict) else []
    fixtures = enrich_live_fixtures(client, fixtures)
    live_ids = {str(fixture.get("id", "")) for fixture in fixtures if fixture.get("id") is not None}
    _SEEN_LIVE.update(live_ids)
    if snapshot_path is not None:
        append_live_snapshots(snapshot_path, fixtures)
    if events_path is not None:
        archive_finished_events(client, live_ids, events_path)
    results: list[LiveCornerScanResult] = []
    recorded = 0
    for fixture in fixtures:
        fixture_id = str(fixture.get("id", ""))
        match_name = _match_name(fixture)
        minute = _fixture_minute(fixture)
        if minute is None:
            results.append(LiveCornerScanResult(fixture_id, match_name, False, ("fixture is not in a minute state",), None))
            continue
        if minute < watch_minute:
            results.append(
                LiveCornerScanResult(
                    fixture_id,
                    match_name,
                    False,
                    (f"before watch window ({minute}<{watch_minute})",),
                    None,
                )
            )
            continue
        odds_payload = client.fixture_odds(fixture["id"])
        result = scan_fixture_for_live_corner(
            fixture,
            odds_payload,
            min_minute=min_minute,
            favorite_odds_max=favorite_odds_max,
            min_corner_odds=min_corner_odds,
        )
        if record_path is not None and result.qualifies and result.entry is not None:
            if append_signal(record_path, result.entry):
                recorded += 1
        results.append(result)
    return tuple(results), recorded


def watch_sleep_seconds(results: tuple[LiveCornerScanResult, ...], *, poll_seconds: int, idle_seconds: int) -> int:
    return poll_seconds if results else idle_seconds


def format_live_corner_scan(results: tuple[LiveCornerScanResult, ...], recorded: int | None = None) -> str:
    lines = ["Live Corner Watch", "-----------------"]
    if not results:
        lines.append("No live top-5 fixtures.")
        return "\n".join(lines)
    odds_calls = sum(1 for result in results if not result.reasons or not result.reasons[0].startswith("before watch window"))
    lines.append(
        f"live={len(results)}, fetched_odds={odds_calls}, qualifies={sum(1 for result in results if result.qualifies)}"
    )
    if recorded is not None:
        lines.append(f"recorded={recorded}")
    lines.append("")
    for result in results:
        status = "QUALIFIES" if result.qualifies else "SKIP"
        detail = "; ".join(result.reasons) or "ok"
        extra = ""
        if result.entry is not None:
            extra = (
                f" min={result.entry.minute} fav={result.entry.favorite_team}"
                f" @{result.entry.prematch_favorite_odds:.2f}"
                f" score={result.entry.favorite_goals}-{result.entry.opponent_goals}"
                f" corners={result.entry.corners_at_entry}"
            )
        lines.append(f"{status} {result.match}{extra} ({detail})")
    return "\n".join(lines)


def settle_live_corner_signals(
    client: LiveOddsClient,
    path: Path,
    *,
    events_path: Path | None = None,
) -> int:
    if not path.exists():
        return 0
    entries = list(load_live_corner_entries(path))
    updated = 0
    rewritten: list[LiveCornerEntry] = []
    for entry in entries:
        if entry.final_total_corners is not None:
            rewritten.append(entry)
            continue
        fixture_id = _fixture_id_from_notes(entry.notes)
        if fixture_id is None:
            rewritten.append(entry)
            continue
        payload = client.fixture(fixture_id)
        fixture = payload.get("data", payload) if isinstance(payload, dict) else {}
        if not _fixture_is_finished(fixture):
            rewritten.append(entry)
            continue
        corners = fixture.get("corners", {})
        try:
            total = int(corners.get("home", 0)) + int(corners.get("away", 0))
        except (TypeError, ValueError):
            rewritten.append(entry)
            continue
        if events_path is not None:
            append_fixture_events(client, fixture_id, events_path)
        rewritten.append(
            LiveCornerEntry(
                date=entry.date,
                competition=entry.competition,
                match=entry.match,
                favorite_team=entry.favorite_team,
                favorite_side=entry.favorite_side,
                prematch_favorite_odds=entry.prematch_favorite_odds,
                minute=entry.minute,
                favorite_goals=entry.favorite_goals,
                opponent_goals=entry.opponent_goals,
                corners_at_entry=entry.corners_at_entry,
                odds_corner_more=entry.odds_corner_more,
                red_card_favorite=entry.red_card_favorite,
                red_card_opponent=entry.red_card_opponent,
                final_total_corners=total,
                notes=entry.notes,
            )
        )
        updated += 1
    if updated:
        _rewrite_signals(path, tuple(rewritten))
    return updated


def scan_fixture_for_live_corner(
    fixture: dict,
    odds_payload: dict,
    *,
    min_minute: int = 84,
    favorite_odds_max: float = 1.50,
    min_corner_odds: float = 1.40,
) -> LiveCornerScanResult:
    fixture_id = str(fixture["id"])
    match_name = _match_name(fixture)
    minute = _fixture_minute(fixture)
    if minute is None:
        return LiveCornerScanResult(fixture_id, match_name, False, ("fixture is not in a minute state",), None)

    favorite = _favorite_from_odds(odds_payload)
    if favorite is None:
        return LiveCornerScanResult(fixture_id, match_name, False, ("missing 1x2 favorite odds",), None)

    favorite_side, favorite_odds = favorite
    home = fixture["teams"]["home"]["name"]
    away = fixture["teams"]["away"]["name"]
    goals = fixture.get("goals", {})
    corners = fixture.get("corners", {})
    cards = fixture.get("cards", {})
    if favorite_side == "home":
        favorite_team = home
        favorite_goals = int(goals.get("home", 0))
        opponent_goals = int(goals.get("away", 0))
        red_card_favorite = int(cards.get("home", {}).get("red", 0)) > 0
        red_card_opponent = int(cards.get("away", {}).get("red", 0)) > 0
    else:
        favorite_team = away
        favorite_goals = int(goals.get("away", 0))
        opponent_goals = int(goals.get("home", 0))
        red_card_favorite = int(cards.get("away", {}).get("red", 0)) > 0
        red_card_opponent = int(cards.get("home", {}).get("red", 0)) > 0

    total_corners = int(corners.get("home", 0)) + int(corners.get("away", 0))
    corner_odds = _one_more_corner_odds(odds_payload, total_corners)
    if corner_odds is None:
        return LiveCornerScanResult(fixture_id, match_name, False, ("missing one-more-corner inplay odds",), None)

    entry = LiveCornerEntry(
        date=str(fixture.get("kickoff_utc", ""))[:10],
        competition=fixture.get("league", {}).get("name", ""),
        match=match_name,
        favorite_team=favorite_team,
        favorite_side=favorite_side,
        prematch_favorite_odds=favorite_odds,
        minute=minute,
        favorite_goals=favorite_goals,
        opponent_goals=opponent_goals,
        corners_at_entry=total_corners,
        odds_corner_more=corner_odds,
        red_card_favorite=red_card_favorite,
        red_card_opponent=red_card_opponent,
        final_total_corners=None,
        notes=f"fixture_id={fixture_id}",
    )
    decision = evaluate_live_corner_entry(
        entry,
        favorite_odds_max=favorite_odds_max,
        super_favorite_odds_max=None,
        min_minute=min_minute,
        min_corner_odds=min_corner_odds,
        exclude_favorite_red_card=True,
    )
    return LiveCornerScanResult(fixture_id, match_name, decision.qualifies, decision.reasons, entry)


def enrich_live_fixtures(client: LiveOddsClient, fixtures: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    for fixture in fixtures:
        fixture_id = str(fixture.get("id", ""))
        detail = _DETAIL_CACHE.get(fixture_id)
        if detail is None and fixture_id:
            payload = client.fixture(fixture_id)
            raw = payload.get("data", payload) if isinstance(payload, dict) else {}
            detail = raw if isinstance(raw, dict) else {}
            _DETAIL_CACHE[fixture_id] = detail
        merged = dict(fixture)
        if detail:
            if not merged.get("odds") and detail.get("odds"):
                merged["odds"] = detail["odds"]
            if not merged.get("goals") and detail.get("goals"):
                merged["goals"] = detail["goals"]
            if not merged.get("corners") and detail.get("corners"):
                merged["corners"] = detail["corners"]
            if not merged.get("cards") and detail.get("cards"):
                merged["cards"] = detail["cards"]
        enriched.append(merged)
    return enriched


def archive_finished_events(client: LiveOddsClient, live_ids: set[str], events_path: Path) -> int:
    written = 0
    for fixture_id in sorted(_SEEN_LIVE - live_ids - _ARCHIVED_EVENTS):
        if not fixture_id:
            continue
        if append_fixture_events(client, fixture_id, events_path):
            _ARCHIVED_EVENTS.add(fixture_id)
            written += 1
    return written


def append_fixture_events(client: LiveOddsClient, fixture_id: str, path: Path) -> bool:
    if fixture_id in _ARCHIVED_EVENTS:
        return False
    payload = client.fixture_events(fixture_id)
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    events = data.get("events", data if isinstance(data, list) else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "fixture_id": fixture_id,
                    "events": events,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    _ARCHIVED_EVENTS.add(fixture_id)
    return True


def append_live_snapshots(path: Path, fixtures: list[dict], *, captured_at: str | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = captured_at or datetime.now(timezone.utc).isoformat()
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for fixture in fixtures:
            handle.write(json.dumps(_snapshot_row(fixture, stamp), ensure_ascii=False) + "\n")
            written += 1
    return written


def _second_half_goals(current: Any, half: Any, minute: int | None) -> int | None:
    if minute is not None and minute < 45:
        return 0
    if current is None or half is None:
        return None
    try:
        return max(0, int(current) - int(half))
    except (TypeError, ValueError):
        return None


def _snapshot_row(fixture: dict, captured_at: str) -> dict[str, Any]:
    odds = fixture.get("odds") or {}
    one_x_two = odds.get("1x2") or {}
    prematch = one_x_two.get("closing") or one_x_two.get("opening") or {}
    goals = fixture.get("goals") or {}
    corners = fixture.get("corners") or {}
    cards = fixture.get("cards") or {}
    home_cards = cards.get("home") or {}
    away_cards = cards.get("away") or {}
    minute = _fixture_minute(fixture)
    return {
        "captured_at": captured_at,
        "fixture_id": fixture.get("id"),
        "league": (fixture.get("league") or {}).get("name"),
        "match": _match_name(fixture),
        "status": fixture.get("status"),
        "minute": minute,
        "home_goals": goals.get("home"),
        "away_goals": goals.get("away"),
        "ht_home_goals": goals.get("half_home"),
        "ht_away_goals": goals.get("half_away"),
        "sh_home_goals": _second_half_goals(goals.get("home"), goals.get("half_home"), minute),
        "sh_away_goals": _second_half_goals(goals.get("away"), goals.get("half_away"), minute),
        "home_corners": corners.get("home"),
        "away_corners": corners.get("away"),
        "home_yellows": home_cards.get("yellow"),
        "away_yellows": away_cards.get("yellow"),
        "home_reds": home_cards.get("red"),
        "away_reds": away_cards.get("red"),
        "prematch_home": prematch.get("home"),
        "prematch_draw": prematch.get("draw"),
        "prematch_away": prematch.get("away"),
        "inplay_1x2": one_x_two.get("inplay"),
        "goal_line": (odds.get("goal_line") or {}).get("inplay") or (odds.get("goal_line") or {}).get("closing"),
        "card_line": (odds.get("card_line") or {}).get("inplay") or (odds.get("card_line") or {}).get("closing"),
        "corner_line": (odds.get("corner_line") or {}).get("inplay") or (odds.get("corner_line") or {}).get("closing"),
    }


def append_signal(path: Path, entry: LiveCornerEntry) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = _existing_signal_keys(path)
    key = _signal_key(entry)
    if key in existing_keys:
        return False
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "date": entry.date,
                "competition": entry.competition,
                "match": entry.match,
                "favorite_team": entry.favorite_team,
                "favorite_side": entry.favorite_side,
                "prematch_favorite_odds": entry.prematch_favorite_odds,
                "minute": entry.minute,
                "favorite_goals": entry.favorite_goals,
                "opponent_goals": entry.opponent_goals,
                "corners_at_entry": entry.corners_at_entry,
                "odds_corner_more": entry.odds_corner_more,
                "red_card_favorite": entry.red_card_favorite,
                "red_card_opponent": entry.red_card_opponent,
                "final_total_corners": "",
                "notes": entry.notes,
            }
        )
    return True


def _favorite_from_odds(payload: dict) -> tuple[str, float] | None:
    market = _bet365_odds(payload).get("1x2", {})
    prices = market.get("closing") or market.get("opening") or {}
    home = prices.get("home")
    away = prices.get("away")
    if home is None or away is None:
        return None
    return ("home", float(home)) if float(home) <= float(away) else ("away", float(away))


def _one_more_corner_odds(payload: dict, total_corners: int) -> float | None:
    market = _bet365_odds(payload).get("corner_line", {})
    inplay = market.get("inplay")
    if not inplay:
        return None
    line = float(inplay.get("line", -999))
    expected_line = total_corners + 0.5
    if abs(line - expected_line) > 0.01:
        return None
    over = inplay.get("over")
    return float(over) if over is not None else None


def _bet365_odds(payload: dict) -> dict:
    bookmakers = payload.get("data", {}).get("bookmakers", [])
    for bookmaker in bookmakers:
        if bookmaker.get("slug") == "bet365":
            return bookmaker.get("odds", {})
    return {}


def _parse_minute(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fixture_minute(fixture: dict) -> int | None:
    for key in ("status_code", "minute", "elapsed"):
        minute = _parse_minute(fixture.get(key))
        if minute is not None:
            return minute
    clock = fixture.get("clock") or fixture.get("status") or {}
    if isinstance(clock, dict):
        return _parse_minute(clock.get("minute") or clock.get("elapsed"))
    return None


def _match_name(fixture: dict) -> str:
    teams = fixture.get("teams", {})
    home = teams.get("home", {}).get("name", "?")
    away = teams.get("away", {}).get("name", "?")
    return f"{home} vs {away}"


def _fixture_id_from_notes(notes: str) -> str | None:
    match = re.search(r"fixture_id=([A-Za-z0-9_-]+)", notes or "")
    return match.group(1) if match else None


def _fixture_is_finished(fixture: dict) -> bool:
    status = str(fixture.get("status") or fixture.get("status_name") or "").lower()
    if status in {"finished", "ft", "full time", "match finished"}:
        return True
    return bool(fixture.get("finished") or fixture.get("is_finished"))


def _rewrite_signals(path: Path, entries: tuple[LiveCornerEntry, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "date": entry.date,
                    "competition": entry.competition,
                    "match": entry.match,
                    "favorite_team": entry.favorite_team,
                    "favorite_side": entry.favorite_side,
                    "prematch_favorite_odds": entry.prematch_favorite_odds,
                    "minute": entry.minute,
                    "favorite_goals": entry.favorite_goals,
                    "opponent_goals": entry.opponent_goals,
                    "corners_at_entry": entry.corners_at_entry,
                    "odds_corner_more": entry.odds_corner_more,
                    "red_card_favorite": entry.red_card_favorite,
                    "red_card_opponent": entry.red_card_opponent,
                    "final_total_corners": "" if entry.final_total_corners is None else entry.final_total_corners,
                    "notes": entry.notes,
                }
            )


def _existing_signal_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {_signal_key_from_row(row) for row in csv.DictReader(handle) if any(row.values())}


def _signal_key(entry: LiveCornerEntry) -> str:
    fixture_id = _fixture_id_from_notes(entry.notes)
    if fixture_id:
        return fixture_id
    return "|".join([entry.date, entry.match])


def _signal_key_from_row(row: dict[str, str]) -> str:
    fixture_id = _fixture_id_from_notes(row.get("notes", ""))
    if fixture_id:
        return fixture_id
    return "|".join([row.get("date", ""), row.get("match", "")])
