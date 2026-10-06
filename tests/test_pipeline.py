from football_quant.backtesting import run_backtest
from football_quant.edge import calculate_edge
from football_quant.models import BaselineAsianHandicapModel
from football_quant.risk import RiskConfig, RiskEngine
from football_quant.sample_data import build_sample_dataset
from football_quant.scanner import OpportunityScanner


def test_sample_backtest_runs() -> None:
    report = run_backtest(build_sample_dataset(), min_expected_value=-1)
    assert len(report.bets) > 0
    assert report.total_staked > 0


def test_risk_rejects_negative_edge() -> None:
    dataset = build_sample_dataset()
    quote = dataset.pre_match_quotes()[0]
    match = dataset.matches_by_id[quote.match_id]
    estimate = BaselineAsianHandicapModel().predict(match, quote)
    edge = calculate_edge(quote, estimate)
    decision = RiskEngine(RiskConfig(min_expected_value=1.0)).decide(edge)
    assert decision.status == "reject"
    assert decision.stake == 0


def test_scanner_returns_explanations() -> None:
    recommendations = OpportunityScanner().scan(build_sample_dataset())
    assert recommendations
    assert all(item.explanation for item in recommendations)
