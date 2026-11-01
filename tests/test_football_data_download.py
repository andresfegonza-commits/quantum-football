from football_quant.football_data_download import football_data_url
from football_quant.paper_ledger import stuttgart_win_bet


def test_football_data_url_uses_season_and_division() -> None:
    assert football_data_url("2425", "D1") == "https://www.football-data.co.uk/mmz4281/2425/D1.csv"


def test_stuttgart_paper_bet_is_not_a_locked_rule() -> None:
    bet = stuttgart_win_bet()
    assert bet.league == "Bundesliga"
    assert bet.strategy == "paper_fav_single"
    assert not bet.rule_compliant
    assert 1.20 <= bet.combined_odds <= 1.70
