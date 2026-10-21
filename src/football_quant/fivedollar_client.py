from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://api.5dollarfootballapi.com/v1"
API_KEY_ENV = "FIVEDOLLAR_FOOTBALL_API_KEY"


class FiveDollarApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class FiveDollarFootballClient:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: int = 20

    @classmethod
    def from_env(cls) -> "FiveDollarFootballClient":
        api_key = os.getenv(API_KEY_ENV) or _read_windows_user_env(API_KEY_ENV)
        if not api_key:
            raise FiveDollarApiError(f"Missing {API_KEY_ENV} environment variable")
        return cls(api_key=api_key)

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self._url(path, params or {})
        request = Request(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": "football-quant-intelligence/0.1",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read().decode("utf-8")
        except Exception as exc:  # pragma: no cover - concrete urllib errors vary by platform
            raise FiveDollarApiError(f"5DollarFootballAPI request failed: {exc}") from exc
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise FiveDollarApiError("5DollarFootballAPI returned non-JSON response") from exc
        if isinstance(parsed, dict) and parsed.get("success") in {0, False}:
            raise FiveDollarApiError(str(parsed.get("message") or parsed.get("error") or "API returned success=0"))
        return parsed

    def fixtures(self, **params: Any) -> dict[str, Any]:
        return self.get("/fixtures", params)

    def fixture(self, fixture_id: str | int, **params: Any) -> dict[str, Any]:
        return self.get(f"/fixtures/{fixture_id}", params)

    def fixture_events(self, fixture_id: str | int) -> dict[str, Any]:
        return self.get(f"/fixtures/{fixture_id}/events")

    def fixture_corner_odds(self, fixture_id: str | int) -> dict[str, Any]:
        return self.get(f"/fixtures/{fixture_id}/odds", {"market": "corner"})

    def _url(self, path: str, params: dict[str, Any]) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        query = {key: value for key, value in params.items() if value is not None}
        suffix = f"?{urlencode(query)}" if query else ""
        return f"{self.base_url.rstrip('/')}{normalized_path}{suffix}"


def _read_windows_user_env(name: str) -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            return str(value) if value else None
    except OSError:
        return None
