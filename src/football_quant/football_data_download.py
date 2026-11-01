from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"
DEFAULT_SEASONS: tuple[str, ...] = (
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
    "2526",
    "2627",
)
DEFAULT_DIVISIONS: tuple[tuple[str, str], ...] = (
    ("E0", "Premier League"),
    ("D1", "Bundesliga"),
    ("SP1", "La Liga"),
    ("I1", "Serie A"),
    ("F1", "Ligue 1"),
)


@dataclass(frozen=True)
class DownloadResult:
    season: str
    division: str
    path: Path
    bytes_written: int
    status: str


def football_data_url(season: str, division: str) -> str:
    return f"{FOOTBALL_DATA_BASE}/{season}/{division}.csv"


def download_football_data_files(
    data_root: Path,
    *,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    divisions: tuple[str, ...] = tuple(code for code, _name in DEFAULT_DIVISIONS),
) -> tuple[DownloadResult, ...]:
    results: list[DownloadResult] = []
    for season in seasons:
        for division in divisions:
            path = data_root / season / f"{division}.csv"
            url = football_data_url(season, division)
            try:
                with urlopen(url, timeout=30) as response:
                    payload = response.read()
            except HTTPError as exc:
                results.append(DownloadResult(season, division, path, 0, f"http_{exc.code}"))
                continue
            except URLError as exc:
                results.append(DownloadResult(season, division, path, 0, f"error:{exc.reason}"))
                continue
            if len(payload) < 80 or b"HomeTeam" not in payload.split(b"\n", 1)[0]:
                results.append(DownloadResult(season, division, path, 0, "invalid"))
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            results.append(DownloadResult(season, division, path, len(payload), "ok"))
    return tuple(results)


def format_download_report(results: tuple[DownloadResult, ...]) -> str:
    lines = ["Football-data download", "----------------------"]
    ok = [item for item in results if item.status == "ok"]
    lines.append(f"ok={len(ok)}/{len(results)}")
    for item in results:
        extra = f" {item.bytes_written}b" if item.status == "ok" else ""
        lines.append(f"{item.season}/{item.division}: {item.status}{extra} -> {item.path}")
    return "\n".join(lines)
