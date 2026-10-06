"""Football quantitative intelligence MVP."""

from .asian_handicap import HandicapSettlement, settle_asian_handicap
from .backtesting import BacktestReport, run_backtest
from .risk import RiskConfig, RiskEngine
from .scanner import OpportunityScanner

__all__ = [
    "BacktestReport",
    "HandicapSettlement",
    "OpportunityScanner",
    "RiskConfig",
    "RiskEngine",
    "run_backtest",
    "settle_asian_handicap",
]
