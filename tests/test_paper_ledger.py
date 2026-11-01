from pathlib import Path

from football_quant.paper_ledger import (
    PaperBet,
    append_paper_bets,
    format_paper_ledger,
    load_paper_bets,
    saturday_singles_bets,
    sunday_nov1_bets,
    settle_paper_bet,
    todays_combo_bets,
    madrid_inter_combo_bet,
    villa_btts_no_bet,
)


def test_append_is_idempotent_and_flags_short_conference_price(tmp_path: Path) -> None:
    path = tmp_path / "paper.csv"
    bets = todays_combo_bets()
    assert append_paper_bets(path, bets) == 2
    assert append_paper_bets(path, bets) == 0
    loaded = load_paper_bets(path)
    assert loaded[0].combined_odds == 1.632
    assert loaded[0].rule_compliant
    assert loaded[1].combined_odds == 1.394
    assert not loaded[1].rule_compliant
    villa = villa_btts_no_bet()
    assert villa.strategy == "locked_btts_no"
    assert 1.90 <= villa.combined_odds <= 2.40
    assert villa.rule_compliant
    ucl_combo = madrid_inter_combo_bet()
    assert ucl_combo.combined_odds == 1.611
    assert ucl_combo.rule_compliant
    singles = saturday_singles_bets()
    assert [bet.combined_odds for bet in singles] == [1.524, 1.443, 1.341]
    assert all(not bet.rule_compliant for bet in singles)
    assert all(1.20 <= bet.combined_odds <= 1.70 for bet in singles)
    sunday = sunday_nov1_bets()
    assert [bet.combined_odds for bet in sunday] == [1.44, 1.64]
    assert sunday[0].legs.startswith("Lyon win")
    assert "1H Over 2.5" in sunday[1].legs
    assert all(not bet.rule_compliant for bet in sunday)


def test_settle_win_and_loss_update_profit(tmp_path: Path) -> None:
    path = tmp_path / "paper.csv"
    append_paper_bets(
        path,
        (
            PaperBet(
                "a",
                "2026-10-22",
                "Europa League",
                "fav_d2_wide",
                "1xBet",
                1.0,
                1.632,
                1.44,
                True,
                "Leverkusen + Benfica",
            ),
            PaperBet(
                "b",
                "2026-10-22",
                "Conference League",
                "fav_d2_wide",
                "1xBet",
                1.0,
                1.394,
                1.44,
                False,
                "Brighton + Twente",
            ),
        ),
    )
    win = settle_paper_bet(path, "a", won=True)
    loss = settle_paper_bet(path, "b", won=False)
    assert win.profit is not None and round(win.profit, 3) == 0.632
    assert loss.profit == -1.0
    text = format_paper_ledger(load_paper_bets(path))
    assert "roi=" in text
    assert "below-rule-min" in text
