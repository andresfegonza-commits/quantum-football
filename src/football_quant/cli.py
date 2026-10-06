from __future__ import annotations

import argparse

from .backtesting import run_backtest
from .sample_data import build_sample_dataset
from .scanner import OpportunityScanner


def main() -> None:
    parser = argparse.ArgumentParser(description="Football quantitative intelligence MVP")
    parser.add_argument("command", choices=("scan-sample", "backtest-sample"))
    args = parser.parse_args()

    dataset = build_sample_dataset()
    if args.command == "scan-sample":
        for recommendation in OpportunityScanner().scan(dataset):
            print(recommendation.explanation)
    elif args.command == "backtest-sample":
        report = run_backtest(dataset)
        print(f"Bets: {len(report.bets)}")
        print(f"Total staked: {report.total_staked:.2f}")
        print(f"Total profit: {report.total_profit:.2f}")
        print(f"ROI: {report.roi:.2%}")
        print(f"Hit rate: {report.hit_rate:.2%}")
        print(f"Max drawdown: {report.max_drawdown:.2f}")
        print(f"Average CLV: {report.average_clv if report.average_clv is not None else 'n/a'}")


if __name__ == "__main__":
    main()
