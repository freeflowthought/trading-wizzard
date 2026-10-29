import argparse
import subprocess
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backtest import run_backtests
from .config import (
    WatchedAsset,
    load_env_file,
    load_events_config,
    load_feeds_config,
    load_ipos_config,
    load_prediction_markets_config,
    load_watchlist_config,
    validate_event_threshold,
)
from .daily_context import refresh_daily_context
from .events import build_event_calendar, fetch_yfinance_earnings_dates
from .fundamentals import build_fundamental_snapshots
from .ingest.cross_market import fetch_cross_market
from .ingest.macro import fetch_fred_observations
from .ingest.market import fetch_market_data, market_source, valid_market_rows
from .ingest.prediction_markets import fetch_prediction_markets
from .ingest.public_macro import fetch_public_macro
from .ingest.reddit import fetch_reddit_items
from .ingest.rss import fetch_rss_items
from .ingest.sec_capex import refresh_cloud_capex
from .ipo import build_ipo_snapshots, ipo_events
from .learning import build_learning_replays
from .models import RawItem
from .options import build_option_snapshots
from .playbook import build_playbooks, leadership_snapshot, macro_gate
from .portfolio_context import refresh_portfolio_context
from .provenance import begin_analysis_snapshot, finalize_analysis_snapshot, verify_snapshot
from .reporting import build_report_bundle, write_report_files
from .sentiment import aggregate_sentiment, analyze_sentiment_items
from .storage import ProjectPaths, read_json, read_jsonl, write_json, write_jsonl
from .technical import analyze_technical_snapshots

try:
    import typer
except Exception:  # pragma: no cover - fallback CLI covers this case
    typer = None


def resolve_date(value: str) -> str:
    resolved = date.today() if value == "today" else date.fromisoformat(value)
    if resolved > date.today():
        raise ValueError("Report date cannot be in the future.")
    return resolved.isoformat()


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


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _dated_market(market: Dict[str, Any], report_date: str) -> Dict[str, Any]:
    return {
        ticker: valid_market_rows(rows, date.fromisoformat(report_date))
        for ticker, rows in market.items()
    }


def _dated_macro(rows: List[Dict[str, Any]], report_date: str) -> List[Dict[str, Any]]:
    dated = []
    for row in rows:
        history = [
            obs for obs in row.get("history", [])
            if str(obs.get("date", ""))[:10] <= report_date
        ]
        if history:
            latest = max(history, key=lambda obs: str(obs.get("date", "")))
            dated.append({
                **row, "history": history, "date": latest["date"], "value": latest["value"]
            })
        elif row.get("date") and str(row["date"])[:10] <= report_date:
            dated.append(row)
    return dated


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
    if market is not None:
        # Old sample artifacts predate explicit row provenance.
        warnings = read_json(paths.processed / f"warnings-{report_date}.json", default=[]) or []
        for ticker, rows in market.items():
            if any(f"sample OHLCV for {ticker};" in warning for warning in warnings):
                for row in rows:
                    row.setdefault("source", "sample")
        return _dated_market(market, report_date)
    config = load_watchlist_config()
    market, warnings = fetch_market_data(
        config["watchlist"], live=live, as_of=date.fromisoformat(report_date)
    )
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

    if source not in {"all", "market", "reddit", "rss", "macro", "predictions", "cross-market"}:
        raise ValueError(f"Unknown ingest source: {source}")
    if limit < 0:
        raise ValueError("Ingest limit cannot be negative.")
    if source in {"all", "market"}:
        market_path = paths.raw / f"market-{report_date}.json"
        if live or not market_path.exists():
            _progress(
                "Collecting market prices..." if live else "Creating offline sample prices..."
            )
            market_data, provider_warnings = fetch_market_data(
                watch_config["watchlist"], live=live, as_of=date.fromisoformat(report_date)
            )
            write_json(market_path, market_data)
            warnings.extend(provider_warnings)
            if live:
                context_assets = [WatchedAsset(str(ticker), str(ticker), [], [])
                                  for ticker in feeds_config.get("market_context_tickers", [])
                                  if str(ticker) not in market_data]
                if context_assets:
                    _progress("Collecting sector and historical-study references...")
                    context, context_warnings = fetch_market_data(
                        context_assets, live=True, as_of=date.fromisoformat(report_date)
                    )
                    write_json(paths.raw / f"market-context-{report_date}.json", context)
                    warnings.extend(context_warnings)

    items_path = paths.raw / f"items-{report_date}.jsonl"
    collect_items = live or not items_path.exists()
    raw_items: List[RawItem] = []
    if collect_items and source in {"all", "reddit"}:
        _progress("Collecting Reddit discussions (bounded RSS fallback)...")
        reddit_items, reddit_warnings = fetch_reddit_items(
            watch_config["watchlist"], watch_config["subreddits"],
            watch_config["ignore_terms"], limit=limit, live=live,
        )
        raw_items.extend(reddit_items)
        warnings.extend(reddit_warnings)

    if collect_items and source in {"all", "rss", "macro"}:
        _progress("Collecting news feeds...")
        rss_items, rss_warnings = fetch_rss_items(
            feeds_config.get("rss_feeds", []), watch_config["watchlist"],
            watch_config["ignore_terms"], live=live, limit=limit,
        )
        raw_items.extend(rss_items)
        warnings.extend(rss_warnings)

    if collect_items and source in {"all", "reddit", "rss", "macro"}:
        # Refreshing one source must not erase other cached source families.
        retained = read_jsonl(items_path) if source != "all" else []
        selected_types = {"reddit", "reddit_rss", "jina_reddit_search"}
        retained = [
            row for row in retained
            if (row.get("source_type") not in selected_types) == (source == "reddit")
        ]
        merged = {str(row.get("url")): row for row in retained}
        merged.update({row.url: row.to_dict() for row in raw_items})
        write_jsonl(items_path, merged.values())

    macro_path = paths.processed / f"macro-{report_date}.json"
    if source in {"all", "macro"} and (live or not macro_path.exists()):
        _progress("Collecting macro series...")
        fred_rows, fred_warnings = fetch_fred_observations(
            feeds_config.get("fred_series", []), live=live, limit=40,
        )
        public_rows, public_warnings = fetch_public_macro(report_date, live=live)
        # A fallback keeps its own full history and source; never splice series vintages.
        existing_ids = {row.get("series_id") for row in fred_rows}
        public_rows = [row for row in public_rows if row.get("series_id") not in existing_ids]
        write_json(macro_path, _dated_macro(fred_rows+public_rows, report_date))
        warnings.extend(fred_warnings)
        warnings.extend(public_warnings)

    predictions_path = paths.processed / f"prediction-markets-{report_date}.json"
    if source in {"all", "macro", "predictions"} and (live or not predictions_path.exists()):
        if report_date != date.today().isoformat() and predictions_path.exists():
            warnings.append("Existing historical Polymarket cache preserved; "
                            "current beliefs cannot backfill historical probabilities.")
        else:
            _progress("Collecting prediction-market event probabilities...")
            discovery_audit = {}
            prediction_config = load_prediction_markets_config(
                paths.config / "prediction-markets.yml")
            discovery_options = {"discovery_audit": discovery_audit} if (
                prediction_config.get("discovery", {}).get("enabled") is True) else {}
            prediction_rows, prediction_warnings = fetch_prediction_markets(
                prediction_config, report_date, live=live, **discovery_options,
            )
            write_json(predictions_path, prediction_rows)
            if discovery_audit:
                write_json(paths.processed / f"prediction-discovery-{report_date}.json",
                           discovery_audit)
            warnings.extend(prediction_warnings)

    cross_market_path = paths.processed / f"cross-market-{report_date}.json"
    if source in {"all", "cross-market"} and (live or not cross_market_path.exists()):
        if report_date != date.today().isoformat() and cross_market_path.exists():
            warnings.append("Existing historical continuous-market cache preserved; "
                            "current quotes cannot backfill historical references.")
        else:
            _progress("Collecting continuous-market risk references...")
            cross_rows, cross_warnings = fetch_cross_market(report_date, live=live)
            write_json(cross_market_path, cross_rows)
            warnings.extend(cross_warnings)

    _append_warnings(paths, report_date, warnings)
    return paths.raw


def cmd_analyze(
    report_date: str = "today", use_finbert: bool = False, build_playbooks: bool = True,
    archive: bool = True,
) -> Path:
    paths = ProjectPaths()
    paths.ensure()
    load_env_file()
    report_date = resolve_date(report_date)
    watch_config = load_watchlist_config(paths.config / "watchlist.yml")
    market_data = _load_or_create_market(paths, report_date, live=False)
    draft = begin_analysis_snapshot(paths, report_date, metadata={
        "kind": "technical_sentiment_analysis", "prefer_finbert": use_finbert,
        "macro_allocation_included": False,
    }) if archive else None
    input_paths = ProjectPaths(Path(draft["input_root"])) if draft else paths
    if draft:
        watch_config = load_watchlist_config(input_paths.config / "watchlist.yml")
        market_data = _dated_market(
            read_json(input_paths.raw / f"market-{report_date}.json", default={}), report_date
        )
    item_rows = read_jsonl(input_paths.raw / f"items-{report_date}.jsonl")
    raw_items = [_raw_item_from_dict(row) for row in item_rows]
    analysis_clock = datetime.fromisoformat(draft["analysis_clock_utc"]) if draft else (
        datetime.now(timezone.utc))
    clock_day = analysis_clock.astimezone().date().isoformat()
    sentiment_cutoff = analysis_clock if report_date == clock_day else datetime.combine(
        date.fromisoformat(report_date), time.max, tzinfo=timezone.utc
    )
    signals = analyze_sentiment_items(
        raw_items,
        watch_config["watchlist"],
        watch_config["ignore_terms"],
        prefer_finbert=use_finbert,
        now=sentiment_cutoff,
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
    if build_playbooks:
        _build_events_and_playbooks(
            paths, report_date, market_data,
            options=read_json(paths.processed / f"options-{report_date}.json", default={}),
            fundamentals=read_json(
                paths.processed / f"fundamentals-{report_date}.json", default={}
            ),
        )
    if draft:
        receipt = finalize_analysis_snapshot(draft, {
            "kind": "technical_sentiment_analysis", "date": report_date,
            "technical": technical, "sentiment_summary": sentiment_summary,
            "sentiment_signals": [signal.to_dict() for signal in signals],
            "source_observation_dates": {ticker: rows[-1]["date"] for ticker, rows in
                                         market_data.items() if rows},
            "macro_allocation_included": False,
            "sentiment_cutoff_utc": sentiment_cutoff.isoformat(),
        })
        write_json(paths.processed / f"analysis-run-{report_date}.json", receipt)
        _progress(f"Technical analysis archived: {receipt['run_id']}")
    return paths.processed


def _build_events_and_playbooks(
    paths: ProjectPaths,
    report_date: str,
    market_data: Dict[str, Any],
    live: bool = False,
    options: Optional[Dict[str, Any]] = None,
    fundamentals: Optional[Dict[str, Any]] = None,
) -> None:
    """Event scan first (his calls are event-anchored), then the playbook
    that combines levels + macro gate + events + options context."""
    from datetime import date as _date

    as_of = _date.fromisoformat(report_date)
    market_data = {
        ticker: rows for ticker, rows in _dated_market(market_data, report_date).items()
        if market_source(rows) == "yfinance"
    }
    cached_events = read_json(paths.processed / f"events-{report_date}.json", default=[]) or []
    earnings = {
        ticker: event["date"] for event in cached_events if event.get("kind") == "earnings"
        for ticker in event.get("tickers", [])
    }
    if live and as_of == date.today():
        earnings = fetch_yfinance_earnings_dates(list(market_data))
    ipos = load_ipos_config()
    events = build_event_calendar(
        as_of,
        earnings_dates=earnings,
        manual=load_events_config(),
    )
    events.extend(ipo_events(ipos, as_of))
    events.sort(key=lambda row: (row["date"], -int(row.get("importance", 0))))
    write_json(paths.processed / f"events-{report_date}.json", events)
    write_json(
        paths.processed / f"ipos-{report_date}.json",
        build_ipo_snapshots(ipos, market_data, as_of),
    )
    leadership = leadership_snapshot(market_data)
    write_json(paths.processed / f"leadership-{report_date}.json", leadership)
    macro_rows = read_json(paths.processed / f"macro-{report_date}.json", default=[]) or []
    macro_rows = _dated_macro(macro_rows, report_date)
    gate = macro_gate(macro_rows, as_of=as_of)
    write_json(paths.processed / f"macro-gate-{report_date}.json", gate)
    technical = read_json(paths.processed / f"technical-{report_date}.json", default={}) or {}
    playbooks = build_playbooks(
        market_data,
        technical,
        options_snapshots=options,
        macro_rows=macro_rows,
        events=events,
        as_of=as_of,
        fundamentals_map=fundamentals,
        leadership=leadership,
    )
    write_json(paths.processed / f"playbook-{report_date}.json", playbooks)


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


def cmd_report(
    report_date: str = "today", live: bool = False, use_finbert: bool = False,
    event_threshold: Optional[float] = None,
) -> Path:
    paths = ProjectPaths()
    paths.ensure()
    load_env_file()
    report_date = resolve_date(report_date)
    prediction_config = load_prediction_markets_config(paths.config / "prediction-markets.yml")
    threshold = validate_event_threshold(prediction_config["main_case_threshold"] if (
        event_threshold is None) else event_threshold)
    if live and report_date == date.today().isoformat():
        write_json(paths.processed / f"warnings-{report_date}.json", [])
    cmd_ingest(source="all", report_date=report_date, live=live)
    _progress("Analyzing sentiment and technical levels...")
    cmd_analyze(
        report_date=report_date, use_finbert=use_finbert, build_playbooks=False, archive=False
    )
    _progress("Running historical comparison...")
    cmd_backtest(report_date=report_date)
    market_data = _load_or_create_market(paths, report_date, live=False)
    current_snapshot = live and report_date == date.today().isoformat()
    provider_market = {
        ticker: rows for ticker, rows in market_data.items() if market_source(rows) == "yfinance"
    }
    _progress("Building options and fundamental context...")
    options_path = paths.processed / f"options-{report_date}.json"
    options = read_json(options_path, default=None) if not current_snapshot else None
    if options is None:
        options = build_option_snapshots(
            provider_market, live=current_snapshot, valuation_date=date.fromisoformat(report_date)
        )
        write_json(options_path, options)
    fundamentals_path = paths.processed / f"fundamentals-{report_date}.json"
    fundamentals = read_json(fundamentals_path, default=None) if not current_snapshot else None
    if fundamentals is None:
        fundamentals = build_fundamental_snapshots(list(provider_market), live=current_snapshot)
        write_json(fundamentals_path, fundamentals)
    if live and not current_snapshot:
        _append_warnings(paths, report_date, [
            "Historical report: current options, fundamentals, and earnings snapshots were skipped."
        ])
    _progress("Building event calendar and report...")
    _build_events_and_playbooks(
        paths, report_date, market_data, live=live, options=options, fundamentals=fundamentals
    )
    study_market = dict(read_json(paths.raw / f"market-context-{report_date}.json", {}) or {})
    study_market.update(market_data)
    write_json(paths.processed / f"analyst-replays-{report_date}.json",
               build_learning_replays(study_market, report_date=report_date))
    if current_snapshot:
        _progress("Checking SEC for newer cloud-customer 10-Q/10-K filings...")
        try:
            receipt = refresh_cloud_capex(paths, report_date)
            write_json(paths.processed / f"cloud-capex-refresh-{report_date}.json", receipt)
            pending = [t for t, row in receipt["issuers"].items()
                       if row.get("status") in ("filing_pending_in_companyfacts",
                                                "submissions_unavailable",
                                                "companyfacts_unavailable")]
            if pending:
                _append_warnings(paths, report_date, [
                    "SEC capex refresh incomplete for " + ", ".join(pending)])
        except Exception as error:
            _append_warnings(paths, report_date, [
                "SEC capex refresh unavailable; cached filings used: " + type(error).__name__])
    _progress("Refreshing dated direction and spending context from local history...")
    try:
        context = refresh_daily_context(paths, report_date, market_data)
        _append_warnings(paths, report_date, context.get("warnings", []))
    except Exception as error:
        _append_warnings(paths, report_date, ["Daily context refresh unavailable: " + str(error)])
    _progress("Refreshing the news-cycle portfolio reference...")
    try:
        portfolio = refresh_portfolio_context(paths, report_date, market_data, live=live)
        _append_warnings(paths, report_date, portfolio.get("warnings", []))
    except Exception as error:
        reason = "Portfolio reference unavailable: " + type(error).__name__
        for prefix in ("portfolio-plan", "portfolio-training"):
            write_json(paths.processed / f"{prefix}-{report_date}.json", {
                "date": report_date, "status": "unavailable", "reason": reason})
        _append_warnings(paths, report_date, [reason])
    draft = begin_analysis_snapshot(paths, report_date, metadata={
        "kind": "report", "event_threshold": threshold,
        "threshold_source": "config" if event_threshold is None else "cli",
        "live_requested": live,
    })
    archived_paths = ProjectPaths(Path(draft["input_root"]))
    bundle = build_report_bundle(
        archived_paths, report_date,
        load_watchlist_config(archived_paths.config / "watchlist.yml")["watchlist"],
        event_threshold=threshold,
        analysis_clock=datetime.fromisoformat(draft["analysis_clock_utc"]),
    )
    bundle.analysis_run = {key: value for key, value in draft.items() if key != "input_root"}
    markdown_path, html_path = write_report_files(paths, bundle)
    receipt = finalize_analysis_snapshot(draft, bundle.to_dict(), outputs={
        "report.md": markdown_path.read_bytes(), "report.html": html_path.read_bytes(),
        "details.md": (paths.reports / f"{report_date}-details.md").read_bytes(),
        "details.html": (paths.reports / f"{report_date}-details.html").read_bytes(),
    })
    write_json(paths.reports / f"{report_date}-run.json", receipt)
    _progress(f"Analysis archived: {receipt['run_id']} (SHA256 {receipt['manifest_sha256']})")
    return markdown_path


def cmd_audit(run: Path, expected_hash: Optional[str] = None, list_inputs: bool = False) -> int:
    result = verify_snapshot(run, expected_hash)
    if not result["valid"]:
        print("Analysis archive integrity failed: " + "; ".join(result["errors"]))
        return 1
    manifest = result["manifest"]
    print(f"Verified analysis {manifest['run_id']} at {manifest['analysis_clock_utc']}")
    print("Effective settings: " + str(manifest.get("metadata", {})))
    print("Historical options/probability coverage: " + str(manifest.get("coverage", {})))
    print("Saved outputs are read as captured; no current providers or freshness rerun.")
    if list_inputs:
        for entry in manifest["files"]:
            if entry["role"] in {"raw_input", "processed_input", "config"}:
                print(f"{entry['original_path']} SHA256 {entry['sha256']}")
    return 0


def cmd_evaluate(protocol: Optional[Path] = None, dataset_dir: Optional[Path] = None) -> Path:
    from .evaluation_runs import run_evaluation_experiment

    receipt = run_evaluation_experiment(protocol, dataset_dir)
    print(f"Research experiment saved: {receipt['path']}")
    print(f"Completion manifest SHA256: {receipt['manifest_sha256']}")
    return Path(receipt["path"])


def cmd_audit_evaluation(run: Path, expected_hash: Optional[str] = None) -> int:
    from .evaluation_runs import verify_evaluation_experiment

    result = verify_evaluation_experiment(run, expected_hash)
    if not result["valid"]:
        print(f"Evaluation {result['status']}: " + "; ".join(result["errors"]))
        return 1
    print("Verified frozen evaluation inputs, helpers, sealed selection and outputs.")
    print("Research execution completed; this does not establish the user's accuracy goal.")
    return 0


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
        source: str = typer.Option(
            "all", help="all, market, reddit, rss, macro, predictions, or cross-market"
        ),
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
        event_threshold: Optional[float] = typer.Option(
            None, "--event-threshold", help="Event main-case ratio: greater than 0.5 and at most 1."
        ),
    ) -> None:
        options = {"event_threshold": event_threshold} if event_threshold is not None else {}
        path = cmd_report(report_date=report_date, live=live, use_finbert=use_finbert, **options)
        typer.echo(f"Report written to {path}")

    @app.command("audit")
    def audit_command(
        run: str = typer.Option(..., "--run", help="Saved analysis directory."),
        expected_hash: Optional[str] = typer.Option(None, "--hash"),
        list_inputs: bool = typer.Option(False, "--list-inputs"),
    ) -> None:
        raise typer.Exit(cmd_audit(Path(run), expected_hash, list_inputs))

    @app.command("events")
    def events_command(
        report_date: str = typer.Option("today", "--date"),
    ) -> None:
        paths = ProjectPaths()
        paths.ensure()
        resolved = resolve_date(report_date)
        rows = read_json(paths.processed / f"events-{resolved}.json", default=None)
        if rows is None:
            cmd_analyze(report_date=resolved)
            rows = read_json(paths.processed / f"events-{resolved}.json", default=[]) or []
        for row in rows:
            typer.echo(
                f"{row['date']} (+{row.get('days_away')}d) "
                f"[{row['kind']}] {row['name']}"
            )

    @app.command("evaluate")
    def evaluate_command(
        protocol: Optional[str] = typer.Option(None, "--protocol"),
        dataset_dir: Optional[str] = typer.Option(None, "--dataset-dir"),
    ) -> None:
        cmd_evaluate(
            Path(protocol) if protocol else None, Path(dataset_dir) if dataset_dir else None
        )

    @app.command("audit-evaluation")
    def audit_evaluation_command(
        run: str = typer.Option(..., "--run"),
        expected_hash: Optional[str] = typer.Option(None, "--hash"),
    ) -> None:
        raise typer.Exit(cmd_audit_evaluation(Path(run), expected_hash))

    @app.command("dashboard")
    def dashboard_command() -> None:
        raise typer.Exit(cmd_dashboard())


def _argparse_main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="twiz")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["ingest", "analyze", "backtest", "report", "events"]:
        command = subparsers.add_parser(name)
        command.add_argument("--date", default="today", dest="report_date")
        if name in {"ingest", "report"}:
            mode = command.add_mutually_exclusive_group()
            mode.add_argument("--live", action="store_true")
            mode.add_argument("--offline", action="store_false", dest="live")
            command.set_defaults(live=False)
        if name in {"analyze", "report"}:
            model = command.add_mutually_exclusive_group()
            model.add_argument("--finbert", action="store_true", dest="use_finbert")
            model.add_argument("--no-finbert", action="store_false", dest="use_finbert")
            command.set_defaults(use_finbert=False)
        if name == "ingest":
            command.add_argument("--source", default="all")
            command.add_argument("--limit", type=int, default=50)
        if name == "backtest":
            command.add_argument("--fee-bps", type=float, default=1.0)
            command.add_argument("--slippage-bps", type=float, default=2.0)
        if name == "report":
            command.add_argument("--event-threshold", type=float, default=None)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--run", type=Path, required=True)
    audit.add_argument("--hash", dest="expected_hash")
    audit.add_argument("--list-inputs", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--protocol", type=Path)
    evaluate.add_argument("--dataset-dir", type=Path)
    evaluation_audit = subparsers.add_parser("audit-evaluation")
    evaluation_audit.add_argument("--run", type=Path, required=True)
    evaluation_audit.add_argument("--hash", dest="expected_hash")
    subparsers.add_parser("dashboard")
    args = parser.parse_args(argv)
    if args.command == "ingest":
        path = cmd_ingest(
            source=args.source, report_date=args.report_date, live=args.live, limit=args.limit
        )
        print(f"Ingest artifacts written under {path}")
    elif args.command == "analyze":
        path = cmd_analyze(report_date=args.report_date, use_finbert=args.use_finbert)
        print(f"Analysis artifacts written under {path}")
    elif args.command == "backtest":
        path = cmd_backtest(
            report_date=args.report_date, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps
        )
        print(f"Backtest artifacts written under {path}")
    elif args.command == "events":
        paths = ProjectPaths()
        paths.ensure()
        resolved = resolve_date(args.report_date)
        rows = read_json(paths.processed / f"events-{resolved}.json", default=None)
        if rows is None:
            cmd_analyze(report_date=resolved)
            rows = read_json(paths.processed / f"events-{resolved}.json", default=[]) or []
        for row in rows:
            print(f"{row['date']} (+{row.get('days_away')}d) [{row['kind']}] {row['name']}")
    elif args.command == "report":
        options = {"event_threshold": args.event_threshold} if (
            args.event_threshold is not None) else {}
        path = cmd_report(
            report_date=args.report_date, live=args.live, use_finbert=args.use_finbert,
            **options,
        )
        print(f"Report written to {path}")
    elif args.command == "audit":
        return cmd_audit(args.run, args.expected_hash, args.list_inputs)
    elif args.command == "evaluate":
        cmd_evaluate(args.protocol, args.dataset_dir)
    elif args.command == "audit-evaluation":
        return cmd_audit_evaluation(args.run, args.expected_hash)
    elif args.command == "dashboard":
        return cmd_dashboard()
    return 0


def main() -> None:
    if typer is not None:
        app()
    else:
        raise SystemExit(_argparse_main(sys.argv[1:]))
