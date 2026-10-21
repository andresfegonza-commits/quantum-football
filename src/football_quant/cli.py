from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import (
    clv_bucket_key,
    format_segment_report,
    line_key,
    odds_bucket_key,
    side_key,
    summarize_segments,
)
from .backtesting import run_backtest
from .football_data_uk import load_football_data_uk_csv, load_football_data_uk_seasons, season_label_from_code
from .fivedollar_client import FiveDollarApiError, FiveDollarFootballClient
from .goal_markets import load_goal_market_rows_for_seasons
from .goal_models import format_goal_walk_forward_report, run_goal_walk_forward
from .goal_value import format_value_walk_forward_report, run_market_aware_ou_walk_forward
from .live_corners import (
    evaluate_live_corner_forward_test,
    format_live_corner_report,
    load_live_corner_entries,
    write_live_corner_template,
)
from .sample_data import build_sample_dataset
from .scanner import OpportunityScanner
from .strategy import evaluate_strategy_scenarios, format_strategy_report
from .walk_forward import format_walk_forward_report, run_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(description="Football quantitative intelligence MVP")
    parser.add_argument(
        "command",
        choices=(
            "scan-sample",
            "backtest-sample",
            "backtest-football-data",
            "analyze-football-data",
            "strategy-football-data",
            "strategy-football-data-seasons",
            "walk-forward-football-data",
            "goal-walk-forward",
            "goal-value-walk-forward",
            "live-corner-template",
            "live-corner-report",
            "fivedollar-check",
            "fivedollar-fixtures",
            "fivedollar-events",
            "fivedollar-corner-odds",
        ),
    )
    parser.add_argument(
        "--csv",
        default="data/raw/football-data/2324/E0.csv",
        help="Path to a football-data.co.uk CSV file.",
    )
    parser.add_argument("--min-ev", type=float, default=0.02, help="Minimum expected value filter.")
    parser.add_argument("--min-train-bets", type=int, default=50, help="Minimum training bets per scenario.")
    parser.add_argument(
        "--include-closing-quotes",
        action="store_true",
        help="Treat closing odds as tradable near-kickoff quotes.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["1920", "2021", "2122", "2223", "2324", "2425"],
        help="football-data.co.uk season codes to include.",
    )
    parser.add_argument(
        "--forward-csv",
        default="data/forward/live_corner_forward_test.csv",
        help="Path to live corner forward-test CSV.",
    )
    parser.add_argument("--min-minute", type=int, default=84, help="Minimum minute for live corner entries.")
    parser.add_argument("--fixture-id", help="Fixture id for 5DollarFootballAPI fixture-specific commands.")
    parser.add_argument("--date", help="Fixture date for 5DollarFootballAPI queries, provider format dependent.")
    parser.add_argument("--league", help="League id or slug for 5DollarFootballAPI queries, provider format dependent.")
    parser.add_argument("--status", choices=("all", "scheduled", "live", "finished", "unknown"), help="Fixture status.")
    parser.add_argument("--live", action="store_true", help="Request live fixtures where supported.")
    args = parser.parse_args()

    if args.command == "scan-sample":
        dataset = build_sample_dataset()
        for recommendation in OpportunityScanner().scan(dataset):
            print(recommendation.explanation)
    elif args.command == "backtest-sample":
        dataset = build_sample_dataset()
        report = run_backtest(dataset)
        _print_report(report)
    elif args.command == "backtest-football-data":
        dataset = load_football_data_uk_csv(
            Path(args.csv),
            season="2023-2024",
            competition="Premier League",
        )
        report = run_backtest(dataset, min_expected_value=args.min_ev, include_closing_quotes=args.include_closing_quotes)
        _print_report(report)
    elif args.command == "analyze-football-data":
        dataset = load_football_data_uk_csv(
            Path(args.csv),
            season="2023-2024",
            competition="Premier League",
        )
        report = run_backtest(dataset, min_expected_value=args.min_ev, include_closing_quotes=args.include_closing_quotes)
        _print_report(report)
        print()
        print(format_segment_report("By Side", summarize_segments(report.bets, side_key)))
        print()
        print(format_segment_report("By Asian Handicap Line", summarize_segments(report.bets, line_key)))
        print()
        print(format_segment_report("By Odds Bucket", summarize_segments(report.bets, odds_bucket_key)))
        print()
        print(format_segment_report("By Closing Line Value", summarize_segments(report.bets, clv_bucket_key)))
    elif args.command == "strategy-football-data":
        dataset = load_football_data_uk_csv(
            Path(args.csv),
            season="2023-2024",
            competition="Premier League",
        )
        report = run_backtest(dataset, min_expected_value=args.min_ev, include_closing_quotes=args.include_closing_quotes)
        _print_report(report)
        print()
        print(format_strategy_report(evaluate_strategy_scenarios(report.bets)))
    elif args.command == "strategy-football-data-seasons":
        dataset = load_football_data_uk_seasons(
            Path("data/raw/football-data"),
            season_codes=tuple(args.seasons),
        )
        report = run_backtest(dataset, min_expected_value=args.min_ev)
        _print_report(report)
        print()
        print(format_strategy_report(evaluate_strategy_scenarios(report.bets)))
    elif args.command == "walk-forward-football-data":
        report = run_walk_forward(
            Path("data/raw/football-data"),
            season_codes=tuple(args.seasons),
            min_train_bets=args.min_train_bets,
            include_closing_quotes=args.include_closing_quotes,
        )
        print(format_walk_forward_report(report))
    elif args.command == "goal-walk-forward":
        rows = load_goal_market_rows_for_seasons(
            Path("data/raw/football-data"),
            season_codes=tuple(args.seasons),
        )
        seasons = tuple(season_label_from_code(code) for code in args.seasons)
        report = run_goal_walk_forward(
            rows,
            seasons=seasons,
            min_expected_value=args.min_ev,
            use_closing_odds=args.include_closing_quotes,
        )
        print(format_goal_walk_forward_report(report))
    elif args.command == "goal-value-walk-forward":
        rows = load_goal_market_rows_for_seasons(
            Path("data/raw/football-data"),
            season_codes=tuple(args.seasons),
        )
        seasons = tuple(season_label_from_code(code) for code in args.seasons)
        report = run_market_aware_ou_walk_forward(
            rows,
            seasons=seasons,
            use_closing_odds=args.include_closing_quotes,
            min_train_bets=args.min_train_bets,
        )
        print(format_value_walk_forward_report(report))
    elif args.command == "live-corner-template":
        path = Path(args.forward_csv)
        write_live_corner_template(path)
        print(f"Template ready: {path}")
    elif args.command == "live-corner-report":
        entries = load_live_corner_entries(Path(args.forward_csv))
        report = evaluate_live_corner_forward_test(entries, min_minute=args.min_minute)
        print(format_live_corner_report(report))
    elif args.command == "fivedollar-check":
        client = _fivedollar_client()
        status = args.status or ("live" if args.live else None)
        payload = client.fixtures(date=args.date, league=args.league, status=status)
        _print_json(payload)
    elif args.command == "fivedollar-fixtures":
        client = _fivedollar_client()
        status = args.status or ("live" if args.live else None)
        payload = client.fixtures(date=args.date, league=args.league, status=status)
        _print_json(payload)
    elif args.command == "fivedollar-events":
        if not args.fixture_id:
            raise SystemExit("--fixture-id is required")
        client = _fivedollar_client()
        _print_json(client.fixture_events(args.fixture_id))
    elif args.command == "fivedollar-corner-odds":
        if not args.fixture_id:
            raise SystemExit("--fixture-id is required")
        client = _fivedollar_client()
        _print_json(client.fixture_corner_odds(args.fixture_id))


def _print_report(report) -> None:
    print(f"Bets: {len(report.bets)}")
    print(f"Total staked: {report.total_staked:.2f}")
    print(f"Total profit: {report.total_profit:.2f}")
    print(f"ROI: {report.roi:.2%}")
    print(f"Hit rate: {report.hit_rate:.2%}")
    print(f"Max drawdown: {report.max_drawdown:.2f}")
    print(f"Average CLV: {report.average_clv if report.average_clv is not None else 'n/a'}")


def _fivedollar_client() -> FiveDollarFootballClient:
    try:
        return FiveDollarFootballClient.from_env()
    except FiveDollarApiError as exc:
        raise SystemExit(str(exc)) from exc


def _print_json(payload) -> None:
    output = json.dumps(payload, indent=2, ensure_ascii=False)
    print(output[:10_000])
    if len(output) > 10_000:
        print("\n... truncated ...")


if __name__ == "__main__":
    main()
