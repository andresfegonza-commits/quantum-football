import pytest

from football_quant.fivedollar_client import FiveDollarApiError, FiveDollarFootballClient


def test_client_builds_urls_with_query_params() -> None:
    client = FiveDollarFootballClient(api_key="secret", base_url="https://example.test/v1")
    assert client._url("/fixtures", {"date": "2026-10-20", "live": None}) == (
        "https://example.test/v1/fixtures?date=2026-10-20"
    )
    assert client._url("fixtures/123/odds", {"market": "corner"}) == (
        "https://example.test/v1/fixtures/123/odds?market=corner"
    )


def test_client_requires_env_key(monkeypatch) -> None:
    monkeypatch.delenv("FIVEDOLLAR_FOOTBALL_API_KEY", raising=False)
    monkeypatch.setattr("football_quant.fivedollar_client._read_windows_user_env", lambda name: None)
    with pytest.raises(FiveDollarApiError):
        FiveDollarFootballClient.from_env()


def test_client_reads_env_key(monkeypatch) -> None:
    monkeypatch.setenv("FIVEDOLLAR_FOOTBALL_API_KEY", "abc123")
    client = FiveDollarFootballClient.from_env()
    assert client.api_key == "abc123"
