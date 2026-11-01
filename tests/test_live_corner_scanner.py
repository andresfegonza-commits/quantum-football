from pathlib import Path

from football_quant.live_corner_scanner import (
    append_live_snapshots,
    scan_fixture_for_live_corner,
    scan_live_corner_window,
    settle_live_corner_signals,
    watch_sleep_seconds,
)


def test_scan_fixture_qualifies_for_one_more_corner() -> None:
    fixture = _fixture(status_code=84, home_goals=0, away_goals=0, home_corners=4, away_corners=4)
    odds = _odds(favorite_away=1.40, corner_line=8.5, corner_over=1.45)

    result = scan_fixture_for_live_corner(fixture, odds)

    assert result.qualifies
    assert result.entry is not None
    assert result.entry.favorite_side == "away"
    assert result.entry.corners_at_entry == 8
    assert result.entry.odds_corner_more == 1.45


def test_watch_sleeps_longer_when_no_live_fixtures() -> None:
    live = scan_fixture_for_live_corner(
        _fixture(status_code=84, home_goals=0, away_goals=0, home_corners=4, away_corners=4),
        _odds(favorite_away=1.40, corner_line=8.5, corner_over=1.45),
    )
    assert watch_sleep_seconds((), poll_seconds=180, idle_seconds=600) == 600
    assert watch_sleep_seconds((live,), poll_seconds=180, idle_seconds=600) == 180


def test_live_snapshots_store_minute_score_and_corners(tmp_path: Path) -> None:
    path = tmp_path / "snaps.jsonl"
    written = append_live_snapshots(
        path,
        [_fixture(status_code=84, home_goals=1, away_goals=1, home_corners=4, away_corners=3)],
        captured_at="2026-10-31T19:00:00+00:00",
    )
    assert written == 1
    row = path.read_text(encoding="utf-8").strip()
    assert '"minute": 84' in row
    assert '"Home vs Away"' in row
    assert '"home_goals": 1' in row
    assert '"home_corners": 4' in row
    assert '"sh_home_goals": 1' in row


def test_enrich_copies_prematch_and_card_lines_into_snapshots(tmp_path: Path) -> None:
    from football_quant.live_corner_scanner import reset_live_watch_state

    reset_live_watch_state()
    fixture = _fixture(status_code=33, home_goals=0, away_goals=1, home_corners=2, away_corners=1)
    fixture["cards"] = {"home": {"yellow": 1, "red": 0}, "away": {"yellow": 2, "red": 0}}
    client = _FakeClient(
        fixtures=[fixture],
        odds={},
        finished={
            "id": 123,
            "odds": {
                "1x2": {"closing": {"home": 1.40, "draw": 4.5, "away": 7.0}},
                "card_line": {"closing": 3.5},
                "goal_line": {"closing": 2.5},
                "corner_line": {"closing": 10},
            },
        },
    )
    path = tmp_path / "snaps.jsonl"
    scan_live_corner_window(client, record_path=None, snapshot_path=path)
    row = path.read_text(encoding="utf-8")
    assert '"prematch_home": 1.4' in row or '"prematch_home": 1.40' in row
    assert '"card_line": 3.5' in row
    assert '"away_yellows": 2' in row
    assert client.odds_calls == 0


def test_events_are_archived_once_when_match_leaves_live_list(tmp_path: Path) -> None:
    from football_quant.live_corner_scanner import reset_live_watch_state

    reset_live_watch_state()
    fixture = _fixture(status_code=90, home_goals=1, away_goals=0, home_corners=4, away_corners=3)
    client = _FakeClient(fixtures=[fixture], odds={})
    events = tmp_path / "events.jsonl"
    scan_live_corner_window(client, record_path=None, snapshot_path=None, events_path=events)
    assert not events.exists()

    client._fixtures = []
    scan_live_corner_window(client, record_path=None, snapshot_path=None, events_path=events)
    scan_live_corner_window(client, record_path=None, snapshot_path=None, events_path=events)
    lines = events.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"type": "goal"' in lines[0]


def test_watch_window_skips_odds_before_minute_80() -> None:
    from football_quant.live_corner_scanner import reset_live_watch_state

    reset_live_watch_state()
    client = _FakeClient(
        fixtures=[_fixture(status_code=62, home_goals=0, away_goals=1, home_corners=3, away_corners=2)],
        odds=_odds(favorite_away=1.40, corner_line=5.5, corner_over=1.45),
    )
    results, recorded = scan_live_corner_window(client, record_path=None)
    assert recorded == 0
    assert not results[0].qualifies
    assert "watch window" in results[0].reasons[0]
    assert client.odds_calls == 0


def test_append_is_idempotent_per_fixture(tmp_path: Path) -> None:
    from football_quant.live_corners import LiveCornerEntry
    from football_quant.live_corner_scanner import append_signal

    path = tmp_path / "live.csv"
    first = LiveCornerEntry(
        "2026-10-31", "Bundesliga", "Bayern vs Dortmund", "Bayern", "home",
        1.33, 90, 2, 2, 7, 2.35, False, False, None, "fixture_id=7093046",
    )
    second = LiveCornerEntry(
        "2026-10-31", "Bundesliga", "Bayern vs Dortmund", "Bayern", "home",
        1.33, 92, 2, 2, 7, 2.35, False, False, None, "fixture_id=7093046",
    )
    assert append_signal(path, first)
    assert not append_signal(path, second)


def test_settle_writes_final_corners_when_fixture_is_finished(tmp_path: Path) -> None:
    path = tmp_path / "live.csv"
    client = _FakeClient(
        fixtures=[],
        odds={},
        finished={"id": 123, "status": "finished", "corners": {"home": 6, "away": 5}},
    )
    from football_quant.live_corners import LiveCornerEntry
    from football_quant.live_corner_scanner import append_signal

    append_signal(
        path,
        LiveCornerEntry(
            "2026-10-31",
            "Premier League",
            "Home vs Away",
            "Away",
            "away",
            1.40,
            84,
            0,
            1,
            8,
            1.45,
            False,
            False,
            None,
            "fixture_id=123",
        ),
    )
    assert settle_live_corner_signals(client, path) == 1
    assert path.read_text(encoding="utf-8").count("11") == 1


def test_scan_fixture_rejects_when_corner_line_is_not_one_more() -> None:
    fixture = _fixture(status_code=84, home_goals=0, away_goals=0, home_corners=4, away_corners=4)
    odds = _odds(favorite_away=1.40, corner_line=10.5, corner_over=1.45)

    result = scan_fixture_for_live_corner(fixture, odds)

    assert not result.qualifies
    assert result.reasons == ("missing one-more-corner inplay odds",)


def _fixture(*, status_code: int, home_goals: int, away_goals: int, home_corners: int, away_corners: int) -> dict:
    return {
        "id": 123,
        "league": {"name": "Test League"},
        "teams": {"home": {"name": "Home"}, "away": {"name": "Away"}},
        "kickoff_utc": "2026-10-20T20:00:00+00:00",
        "status_code": status_code,
        "goals": {"home": home_goals, "away": away_goals, "half_home": 0, "half_away": 0},
        "corners": {"home": home_corners, "away": away_corners},
        "cards": {"home": {"red": 0}, "away": {"red": 0}},
    }


class _FakeClient:
    def __init__(self, fixtures: list[dict], odds: dict, finished: dict | None = None) -> None:
        self._fixtures = fixtures
        self._odds = odds
        self._finished = finished or {}
        self.odds_calls = 0

    def fixtures(self, **params):
        return {"data": self._fixtures}

    def fixture(self, fixture_id, **params):
        return {"data": self._finished}

    def fixture_odds(self, fixture_id, market=None):
        self.odds_calls += 1
        return self._odds

    def fixture_events(self, fixture_id):
        return {"data": {"fixture_id": fixture_id, "events": [{"type": "goal", "minute": 48, "team": "home"}]}}


def _odds(*, favorite_away: float, corner_line: float, corner_over: float) -> dict:
    return {
        "data": {
            "bookmakers": [
                {
                    "slug": "bet365",
                    "odds": {
                        "1x2": {"closing": {"home": 3.0, "draw": 3.4, "away": favorite_away}},
                        "corner_line": {"inplay": {"line": corner_line, "over": corner_over, "under": 2.6}},
                    },
                }
            ]
        }
    }
