"""Adversarial checks for report dates and provider edge cases."""

import math
from datetime import date
from pathlib import Path
from unittest.mock import patch

from twiz import cli
from twiz.config import DEFAULT_WATCHLIST
from twiz.models import OptionContract
from twiz.options import summarize_options
from twiz.reporting import build_report_bundle
from twiz.storage import ProjectPaths, write_json, write_jsonl


def test_historical_radar_does_not_display_future_news_or_observations(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    report_date = "2026-06-07"
    write_jsonl(paths.raw / f"items-{report_date}.jsonl", [
        {"source_type": "fed_rss", "title": "Future news", "source": "Fed",
         "published_at": "Tue, 27 Oct 2026 12:00:00 GMT", "url": "https://example.test/future"},
        {"source_type": "fed_rss", "title": "Prior news", "source": "Fed",
         "published_at": "Sat, 06 Jun 2026 12:00:00 GMT", "url": "https://example.test/prior"},
    ])
    write_json(paths.processed / f"macro-{report_date}.json", [
        {"source": "FRED", "series_id": "DGS10", "date": "2026-10-27", "value": "4.0"},
        {"source": "FRED", "series_id": "UNRATE", "date": "2026-06-01", "value": "4.1"},
    ])
    radar = build_report_bundle(paths, report_date, DEFAULT_WATCHLIST[:1]).macro_radar
    assert any(row.get("title") == "Prior news" for row in radar)
    assert not any(row.get("title") == "Future news" for row in radar)
    assert not any(row.get("series_id") == "DGS10" for row in radar)


def test_nonfinite_option_volatility_cannot_poison_snapshot_metrics():
    contracts = [
        OptionContract("C", "call", 100, "2026-11-20", bid=2, ask=3,
                       implied_volatility=math.nan),
        OptionContract("P", "put", 100, "2026-11-20", bid=2, ask=3,
                       implied_volatility=0.3),
    ]
    snapshot = summarize_options("TEST", 100, contracts, "2026-11-20")
    for value in [*snapshot.greeks.values(), *snapshot.skew_summary.values()]:
        assert value is None or math.isfinite(value)


def test_fallback_cli_passes_live_and_finbert_report_flags():
    with patch("twiz.cli.cmd_report", return_value=Path("/tmp/report.md")) as report:
        assert cli._argparse_main([
            "report", "--date", "2026-06-07", "--live", "--finbert",
        ]) == 0
    report.assert_called_once_with(
        report_date="2026-06-07", live=True, use_finbert=True,
    )


def test_legacy_sample_warning_preserves_demo_provenance(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    report_date = "2026-06-07"
    rows = [{"date": "2026-06-05", "open": 99, "high": 101,
             "low": 98, "close": 100, "volume": 1000}]
    write_json(paths.raw / f"market-{report_date}.json", {"SPY": rows})
    write_json(paths.processed / f"warnings-{report_date}.json", [
        "Using deterministic sample OHLCV for SPY; pass --live for market data.",
    ])
    cached = cli._load_or_create_market(paths, report_date)
    assert cached["SPY"][0]["source"] == "sample"
    bundle = build_report_bundle(paths, report_date, DEFAULT_WATCHLIST[:1])
    assert bundle.watchlist_table[0]["data_source"] == "sample"
    assert bundle.playbook == {}


def test_cache_discards_nonfinite_bar_before_technical_analysis():
    from twiz.ingest.market import valid_market_rows

    good = {"date": "2026-06-05", "open": 99, "high": 101,
            "low": 98, "close": 100, "volume": 1000, "source": "yfinance"}
    bad = {**good, "date": "2026-06-06", "close": math.nan}
    assert valid_market_rows([good, bad], as_of=date(2026, 6, 7)) == [good]
