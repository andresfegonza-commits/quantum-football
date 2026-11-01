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
from .footiqo import backtest_footiqo_markets, format_footiqo_backtest, load_footiqo_matches
from .footiqo_accumulators import AccumulatorConfig, format_over_accumulator_walk_forward, load_over_matches
from .footiqo_btts import format_btts_no_walk_forward_report, load_btts_rows, run_btts_no_walk_forward
from .footiqo_btts_refine import format_btts_rule_review, format_flat_odds_buckets, review_working_btts_rules
from .footiqo_btts_scan import append_paper_signals, format_locked_btts_scan, load_locked_league_matches, scan_locked_btts_no
from .footiqo_combo_search import format_combo_search, load_combo_matches, review_combo_families
from .domestic_review import format_domestic_review, review_domestic_leagues
from .setpiece_review import format_setpiece_review, review_setpiece_markets
from .half_goals_review import format_half_goals_review, review_half_goal_markets
from .football_data_download import download_football_data_files, format_download_report
from .paper_ledger import DEFAULT_PAPER_PATH, format_paper_ledger, load_paper_bets
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
from .live_corner_scanner import (
    format_live_corner_scan,
    scan_live_corner_window,
    settle_live_corner_signals,
    watch_sleep_seconds,
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
            "live-corner-scan",
            "live-corner-settle",
            "footiqo-backtest",
            "footiqo-btts-walk-forward",
            "footiqo-over-acc-walk-forward",
            "footiqo-btts-refine",
            "footiqo-btts-scan",
            "footiqo-combo-search",
            "paper-ledger",
            "football-data-download",
            "football-data-domestic-review",
            "football-data-setpiece-review",
            "football-data-half-goals-review",
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
    parser.add_argument("--watch-minute", type=int, default=80, help="Start fetching live corner odds from this minute.")
    parser.add_argument("--poll-seconds", type=int, default=0, help="Repeat the live corner scan on this interval.")
    parser.add_argument("--idle-seconds", type=int, default=600, help="Sleep when no live fixtures are found.")
    parser.add_argument("--max-cycles", type=int, default=1, help="Maximum live corner scan cycles when polling. 0 means run until stopped.")
    parser.add_argument(
        "--snapshot-jsonl",
        default="data/forward/live_snapshots.jsonl",
        help="Append-only live minute snapshots for later backtests.",
    )
    parser.add_argument(
        "--events-jsonl",
        default="data/forward/live_events.jsonl",
        help="Append-only goal/card/corner event timelines after matches finish.",
    )
    parser.add_argument("--fixture-id", help="Fixture id for 5DollarFootballAPI fixture-specific commands.")
    parser.add_argument("--date", help="Fixture date for 5DollarFootballAPI queries, provider format dependent.")
    parser.add_argument("--league", help="League id or slug for 5DollarFootballAPI queries, provider format dependent.")
    parser.add_argument("--status", choices=("all", "scheduled", "live", "finished", "unknown"), help="Fixture status.")
    parser.add_argument("--live", action="store_true", help="Request live fixtures where supported.")
    parser.add_argument("--record-signals", action="store_true", help="Append qualifying live corner signals to CSV.")
    parser.add_argument(
        "--footiqo-files",
        nargs="+",
        default=[
            "data/raw/footiqo/champions-league.md",
            "data/raw/footiqo/europa-league.md",
            "data/raw/footiqo/conference-league.md",
        ],
        help="Footiqo markdown/CSV export files to backtest.",
    )
    parser.add_argument(
        "--footiqo-league",
        default=None,
        help="League filter for Footiqo commands. Defaults to Champions League for BTTS and all UEFA for accumulators.",
    )
    parser.add_argument("--min-combined-odds", type=float, default=1.40, help="Minimum accumulator odds.")
    parser.add_argument("--min-legs", type=int, default=2, help="Minimum accumulator legs.")
    parser.add_argument("--max-legs", type=int, default=4, help="Maximum accumulator legs.")
    parser.add_argument(
        "--paper-csv",
        default="data/forward/btts_no_paper.csv",
        help="CSV path for locked BTTS No paper-trade signals.",
    )
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
    elif args.command == "live-corner-scan":
        import time

        client = _fivedollar_client()
        path = Path(args.forward_csv)
        events_path = Path(args.events_jsonl)
        cycle = 0
        unlimited = args.poll_seconds > 0 and args.max_cycles <= 0
        max_cycles = 1 if args.poll_seconds <= 0 else (args.max_cycles if args.max_cycles > 0 else 1)
        while True:
            cycle += 1
            if args.poll_seconds:
                label = "unlimited" if unlimited else str(max_cycles)
                print(f"cycle={cycle}/{label}")
            results: tuple = ()
            try:
                results, recorded = scan_live_corner_window(
                    client,
                    watch_minute=args.watch_minute,
                    min_minute=args.min_minute,
                    record_path=path,
                    snapshot_path=Path(args.snapshot_jsonl),
                    events_path=events_path,
                )
                settled = settle_live_corner_signals(client, path, events_path=events_path)
                print(format_live_corner_scan(results, recorded=recorded))
                if settled:
                    print(f"settled={settled}")
            except FiveDollarApiError as exc:
                print(f"api_error={exc}")
                if args.poll_seconds <= 0:
                    raise
                time.sleep(args.poll_seconds)
                continue
            if args.poll_seconds <= 0:
                break
            if not unlimited and cycle >= max_cycles:
                break
            time.sleep(watch_sleep_seconds(results, poll_seconds=args.poll_seconds, idle_seconds=args.idle_seconds))
    elif args.command == "live-corner-settle":
        client = _fivedollar_client()
        path = Path(args.forward_csv)
        settled = settle_live_corner_signals(client, path, events_path=Path(args.events_jsonl))
        print(f"settled={settled} path={path}")
        entries = load_live_corner_entries(path) if path.exists() else ()
        if entries:
            print(format_live_corner_report(evaluate_live_corner_forward_test(entries, min_minute=args.min_minute)))
    elif args.command == "footiqo-backtest":
        matches = tuple(
            match
            for file_name in args.footiqo_files
            for match in load_footiqo_matches(Path(file_name))
        )
        reports = backtest_footiqo_markets(matches)
        print(format_footiqo_backtest(matches, reports))
    elif args.command == "footiqo-btts-walk-forward":
        rows = load_btts_rows(
            tuple(Path(file_name) for file_name in args.footiqo_files),
            league=args.footiqo_league or "Champions League",
        )
        report = run_btts_no_walk_forward(rows, min_train_bets=args.min_train_bets)
        print(format_btts_no_walk_forward_report(report, rows=rows))
    elif args.command == "footiqo-over-acc-walk-forward":
        matches = load_over_matches(
            tuple(Path(file_name) for file_name in args.footiqo_files),
            league=args.footiqo_league or "all",
        )
        config = AccumulatorConfig(
            min_combined_odds=args.min_combined_odds,
            min_legs=args.min_legs,
            max_legs=args.max_legs,
        )
        print(format_over_accumulator_walk_forward(matches, config=config))
    elif args.command == "footiqo-btts-refine":
        files = tuple(Path(file_name) for file_name in args.footiqo_files)
        reviews = review_working_btts_rules(files)
        print(format_flat_odds_buckets(files))
        print()
        print(format_btts_rule_review(reviews))
    elif args.command == "footiqo-btts-scan":
        matches = load_locked_league_matches(tuple(Path(file_name) for file_name in args.footiqo_files))
        report = scan_locked_btts_no(matches)
        print(format_locked_btts_scan(report))
        if args.record_signals:
            recorded = append_paper_signals(Path(args.paper_csv), report.signals)
            print(f"recorded={recorded} path={args.paper_csv}")
    elif args.command == "footiqo-combo-search":
        matches = load_combo_matches(tuple(Path(file_name) for file_name in args.footiqo_files))
        reviews = review_combo_families(matches)
        print(format_combo_search(matches, reviews))
    elif args.command == "paper-ledger":
        print(format_paper_ledger(load_paper_bets(DEFAULT_PAPER_PATH)))
    elif args.command == "football-data-download":
        results = download_football_data_files(Path("data/raw/football-data"))
        print(format_download_report(results))
    elif args.command == "football-data-domestic-review":
        print(format_domestic_review(review_domestic_leagues(Path("data/raw/football-data"))))
    elif args.command == "football-data-setpiece-review":
        root = Path("data/raw/football-data")
        print(format_setpiece_review(review_setpiece_markets(root), root))
    elif args.command == "football-data-half-goals-review":
        root = Path("data/raw/football-data")
        print(format_half_goals_review(review_half_goal_markets(root), root))


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
