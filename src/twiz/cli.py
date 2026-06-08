import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backtest import run_backtests
from .config import load_env_file, load_feeds_config, load_watchlist_config
from .ingest.macro import fetch_fred_observations
from .ingest.market import fetch_market_data
from .ingest.reddit import fetch_reddit_items
from .ingest.rss import fetch_rss_items
from .models import RawItem
from .options import build_option_snapshots
from .reporting import build_report_bundle, write_report_files
from .sentiment import aggregate_sentiment, analyze_sentiment_items
from .storage import ProjectPaths, read_json, read_jsonl, write_json, write_jsonl
from .technical import analyze_technical_snapshots

try:
    import typer
except Exception:  # pragma: no cover - fallback CLI covers this case
    typer = None


def resolve_date(value: str) -> str:
    return date.today().isoformat() if value == "today" else value


def _raw_item_from_dict(row: Dict[str, Any]) -> RawItem:
    engagement = {
        str(key): float(value)
        for key, value in (row.get("engagement", {}) or {}).items()
    }
    return RawItem(
        source=str(row.get("source", "")),
        source_type=str(row.get("source_type", "")),
        url=str(row.get("url", "")),
        published_at=str(row.get("published_at", "")),
        ticker_candidates=[str(ticker) for ticker in row.get("ticker_candidates", [])],
        title=str(row.get("title", "")),
        text=str(row.get("text", "")),
        engagement=engagement,
    )


def _reset_run_artifacts(paths: ProjectPaths, report_date: str) -> None:
    for path in [
        paths.processed / f"warnings-{report_date}.json",
        paths.raw / f"items-{report_date}.jsonl",
    ]:
        if path.exists():
            path.unlink()


def _append_warnings(paths: ProjectPaths, report_date: str, warnings: List[str]) -> None:
    if not warnings:
        return
    path = paths.processed / f"warnings-{report_date}.json"
    existing = read_json(path, default=[]) or []
    merged: List[str] = []
    for warning in existing + warnings:
        if warning not in merged:
            merged.append(warning)
    write_json(path, merged)


def _load_or_create_market(
    paths: ProjectPaths,
    report_date: str,
    live: bool = False,
) -> Dict[str, Any]:
    market_path = paths.raw / f"market-{report_date}.json"
    market = read_json(market_path, default=None)
    if market:
        return market
    config = load_watchlist_config()
    market, warnings = fetch_market_data(config["watchlist"], live=live)
    write_json(market_path, market)
    _append_warnings(paths, report_date, warnings)
    return market


def cmd_ingest(
    source: str = "all",
    report_date: str = "today",
    live: bool = False,
    limit: int = 50,
) -> Path:
    paths = ProjectPaths()
    paths.ensure()
    load_env_file()
    report_date = resolve_date(report_date)
    watch_config = load_watchlist_config()
    feeds_config = load_feeds_config()
    warnings: List[str] = []

    if source in {"all", "market"}:
        market_data, provider_warnings = fetch_market_data(watch_config["watchlist"], live=live)
        write_json(paths.raw / f"market-{report_date}.json", market_data)
        warnings.extend(provider_warnings)

    raw_items: List[RawItem] = []
    if source in {"all", "reddit"}:
        reddit_items, reddit_warnings = fetch_reddit_items(
            watch_config["watchlist"],
            watch_config["subreddits"],
            watch_config["ignore_terms"],
            limit=limit,
            live=live,
        )
        raw_items.extend(reddit_items)
        warnings.extend(reddit_warnings)

    if source in {"all", "rss", "macro"}:
        rss_items, rss_warnings = fetch_rss_items(
            feeds_config.get("rss_feeds", []),
            watch_config["watchlist"],
            watch_config["ignore_terms"],
            live=live,
            limit=limit,
        )
        raw_items.extend(rss_items)
        warnings.extend(rss_warnings)

    if source in {"all", "reddit", "rss", "macro"}:
        write_jsonl(paths.raw / f"items-{report_date}.jsonl", raw_items)

    if source in {"all", "macro"}:
        fred_rows, fred_warnings = fetch_fred_observations(
            feeds_config.get("fred_series", []),
            live=live,
        )
        write_json(paths.processed / f"macro-{report_date}.json", fred_rows)
        warnings.extend(fred_warnings)

    _append_warnings(paths, report_date, warnings)
    return paths.raw


def cmd_analyze(report_date: str = "today", use_finbert: bool = False) -> Path:
    paths = ProjectPaths()
    paths.ensure()
    load_env_file()
    report_date = resolve_date(report_date)
    watch_config = load_watchlist_config()
    market_data = _load_or_create_market(paths, report_date, live=False)

    item_rows = read_jsonl(paths.raw / f"items-{report_date}.jsonl")
    raw_items = [_raw_item_from_dict(row) for row in item_rows]
    signals = analyze_sentiment_items(
        raw_items,
        watch_config["watchlist"],
        watch_config["ignore_terms"],
        prefer_finbert=use_finbert,
    )
    sentiment_summary = aggregate_sentiment(signals)
    write_json(
        paths.processed / f"sentiment-{report_date}.json",
        [signal.to_dict() for signal in signals],
    )
    write_json(paths.processed / f"sentiment-summary-{report_date}.json", sentiment_summary)

    chart_dir = paths.processed / "charts" / report_date
    technical = analyze_technical_snapshots(market_data, output_dir=chart_dir)
    write_json(paths.processed / f"technical-{report_date}.json", technical)
    return paths.processed


def cmd_backtest(
    report_date: str = "today",
    fee_bps: float = 1.0,
    slippage_bps: float = 2.0,
) -> Path:
    paths = ProjectPaths()
    paths.ensure()
    report_date = resolve_date(report_date)
    market_data = _load_or_create_market(paths, report_date, live=False)
    equity_dir = paths.processed / "equity" / report_date
    backtests = run_backtests(
        market_data,
        output_dir=equity_dir,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    write_json(paths.processed / f"backtests-{report_date}.json", backtests)
    warnings = [
        f"{ticker}: {warning}"
        for ticker, result in backtests.items()
        for warning in result.get("warnings", [])
    ]
    _append_warnings(paths, report_date, warnings)
    return paths.processed


def cmd_report(report_date: str = "today", live: bool = False, use_finbert: bool = False) -> Path:
    paths = ProjectPaths()
    paths.ensure()
    load_env_file()
    report_date = resolve_date(report_date)
    _reset_run_artifacts(paths, report_date)
    cmd_ingest(source="all", report_date=report_date, live=live)
    cmd_analyze(report_date=report_date, use_finbert=use_finbert)
    cmd_backtest(report_date=report_date)
    market_data = _load_or_create_market(paths, report_date, live=False)
    options = build_option_snapshots(market_data, live=live)
    write_json(paths.processed / f"options-{report_date}.json", options)
    bundle = build_report_bundle(paths, report_date, load_watchlist_config()["watchlist"])
    markdown_path, _ = write_report_files(paths, bundle)
    return markdown_path


def cmd_dashboard() -> int:
    dashboard_path = Path(__file__).with_name("dashboard.py")
    try:
        return subprocess.run(["streamlit", "run", str(dashboard_path)], check=False).returncode
    except FileNotFoundError:
        print(
            "streamlit is not installed. Run `uv sync` or "
            "`uv run streamlit run src/twiz/dashboard.py`."
        )
        return 1


if typer is not None:
    app = typer.Typer(help="Trading Wizzard personal market report CLI.")

    @app.command("ingest")
    def ingest_command(
        source: str = typer.Option("all", help="all, market, reddit, rss, or macro"),
        report_date: str = typer.Option("today", "--date"),
        live: bool = typer.Option(False, "--live/--offline"),
        limit: int = typer.Option(50, help="Maximum social/news items to ingest."),
    ) -> None:
        path = cmd_ingest(source=source, report_date=report_date, live=live, limit=limit)
        typer.echo(f"Ingest artifacts written under {path}")

    @app.command("analyze")
    def analyze_command(
        report_date: str = typer.Option("today", "--date"),
        use_finbert: bool = typer.Option(False, "--finbert/--no-finbert"),
    ) -> None:
        path = cmd_analyze(report_date=report_date, use_finbert=use_finbert)
        typer.echo(f"Analysis artifacts written under {path}")

    @app.command("backtest")
    def backtest_command(
        report_date: str = typer.Option("today", "--date"),
        fee_bps: float = typer.Option(1.0),
        slippage_bps: float = typer.Option(2.0),
    ) -> None:
        path = cmd_backtest(report_date=report_date, fee_bps=fee_bps, slippage_bps=slippage_bps)
        typer.echo(f"Backtest artifacts written under {path}")

    @app.command("report")
    def report_command(
        report_date: str = typer.Option("today", "--date"),
        live: bool = typer.Option(False, "--live/--offline"),
        use_finbert: bool = typer.Option(False, "--finbert/--no-finbert"),
    ) -> None:
        path = cmd_report(report_date=report_date, live=live, use_finbert=use_finbert)
        typer.echo(f"Report written to {path}")

    @app.command("dashboard")
    def dashboard_command() -> None:
        raise typer.Exit(cmd_dashboard())


def _argparse_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="twiz")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["ingest", "analyze", "backtest", "report"]:
        command = subparsers.add_parser(name)
        command.add_argument("--date", default="today", dest="report_date")
    subparsers.add_parser("dashboard")
    args = parser.parse_args(argv)
    if args.command == "ingest":
        print(f"Ingest artifacts written under {cmd_ingest(report_date=args.report_date)}")
    elif args.command == "analyze":
        print(f"Analysis artifacts written under {cmd_analyze(report_date=args.report_date)}")
    elif args.command == "backtest":
        print(f"Backtest artifacts written under {cmd_backtest(report_date=args.report_date)}")
    elif args.command == "report":
        print(f"Report written to {cmd_report(report_date=args.report_date)}")
    elif args.command == "dashboard":
        return cmd_dashboard()
    return 0


def main() -> None:
    if typer is not None:
        app()
    else:
        raise SystemExit(_argparse_main(sys.argv[1:]))
