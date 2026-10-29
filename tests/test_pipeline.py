from datetime import date, timedelta
from unittest.mock import patch

import pytest

from twiz import cli
from twiz.config import DEFAULT_WATCHLIST
from twiz.ingest.market import fetch_market_data, synthetic_ohlcv
from twiz.ingest.reddit import fetch_reddit_rss_items
from twiz.reporting import build_report_bundle, render_markdown
from twiz.storage import ProjectPaths, write_json, write_jsonl


def test_sample_bars_end_at_report_date():
    rows = synthetic_ohlcv("SPY", days=260, as_of=date(2026, 10, 25))
    assert len(rows) == 260
    assert rows[-1]["date"] == "2026-10-23"
    assert all(row["source"] == "sample" for row in rows)
    assert [row["date"] for row in rows] == sorted(row["date"] for row in rows)


def test_live_provider_failure_never_invents_prices():
    with patch("twiz.ingest.market.fetch_yfinance_ohlcv", return_value=([], ["failed"])):
        market, warnings = fetch_market_data(DEFAULT_WATCHLIST[:1], live=True)
    assert market == {"SPY": []}
    assert any("unavailable" in warning for warning in warnings)


def test_market_cache_is_cut_off_at_report_date(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    write_json(paths.raw / "market-2026-10-27.json", {"SPY": [
        {"date": "2026-10-28", "close": 101},
        {"date": "2026-10-26", "close": 99},
    ]})
    market = cli._load_or_create_market(paths, "2026-10-27")
    assert market["SPY"] == [{"date": "2026-10-26", "close": 99}]


def test_offline_ingest_preserves_live_artifacts(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    cached = {"SPY": [{"date": "2026-10-27", "close": 100, "source": "yfinance"}]}
    write_json(paths.raw / "market-2026-10-27.json", cached)
    write_jsonl(paths.raw / "items-2026-10-27.jsonl", [{"url": "cached-url"}])
    write_json(paths.processed / "macro-2026-10-27.json", [{"value": "4.1"}])
    before = {path: path.read_bytes() for path in [
        paths.raw / "market-2026-10-27.json",
        paths.raw / "items-2026-10-27.jsonl",
        paths.processed / "macro-2026-10-27.json",
    ]}
    with patch("twiz.cli.ProjectPaths", return_value=paths), patch(
        "twiz.cli.fetch_market_data", side_effect=AssertionError("must use cache")
    ):
        cli.cmd_ingest(report_date="2026-10-27")
    assert all(path.read_bytes() == content for path, content in before.items())


def test_sample_report_withholds_trading_calls(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    report_date = "2026-10-27"
    write_json(paths.raw / f"market-{report_date}.json", {
        "SPY": synthetic_ohlcv("SPY", as_of=date(2026, 10, 27)),
    })
    write_json(paths.processed / f"playbook-{report_date}.json", {
        "SPY": {"staged_entries": [{"batch": 1, "price": 90, "fraction": 0.3}]},
    })
    write_json(paths.processed / f"leadership-{report_date}.json", {
        "state": "intact", "benchmark": "SPY", "leaders": {}, "rule": "income mode on",
    })
    write_json(paths.processed / f"backtests-{report_date}.json", {
        "SPY": {"metrics": {"total_return": 2.0, "benchmark_return": 1.0}},
    })
    bundle = build_report_bundle(paths, report_date, DEFAULT_WATCHLIST[:1])
    markdown = render_markdown(bundle, detailed=True)
    assert "DEMO DATA PRESENT" in markdown
    assert "sample / 2026-10-27" in markdown
    assert "Staged entries" not in markdown
    assert "income mode on" not in markdown
    assert bundle.watchlist_table[0]["backtest_return"] is None
    assert bundle.playbook == {}


def test_rss_failure_respects_request_budget(monkeypatch):
    monkeypatch.setenv("REDDIT_RSS_MAX_REQUESTS", "2")
    with patch("twiz.ingest.reddit._fetch_reddit_rss", side_effect=TimeoutError) as fetch:
        rows, warnings = fetch_reddit_rss_items(
            DEFAULT_WATCHLIST, ["stocks", "options", "investing"], [], limit=50,
        )
    assert rows == []
    assert fetch.call_count == 2
    assert any("budget" in warning for warning in warnings)


def test_zero_limit_does_not_fetch_reddit():
    with patch("twiz.ingest.reddit._fetch_reddit_rss") as fetch:
        assert fetch_reddit_rss_items(DEFAULT_WATCHLIST, ["stocks"], [], limit=0) == ([], [])
    fetch.assert_not_called()


def test_rejects_invalid_and_future_report_dates():
    with pytest.raises(ValueError):
        cli.resolve_date("not-a-date")
    with pytest.raises(ValueError):
        cli.resolve_date((date.today() + timedelta(days=1)).isoformat())


def test_yahoo_filters_invalid_bar_and_passes_historical_cutoff(monkeypatch):
    import sys
    from datetime import datetime
    from types import SimpleNamespace
    from unittest.mock import Mock

    from twiz.ingest.market import fetch_yfinance_ohlcv

    entries = [
        (datetime(2026, 10, 26), {
            "Open": 99, "High": 102, "Low": 98, "Close": 100, "Volume": 1000,
        }),
        (datetime(2026, 10, 27), {
            "Open": 100, "High": 102, "Low": 99, "Close": float("nan"), "Volume": 1000,
        }),
        (datetime(2026, 10, 28), {
            "Open": 100, "High": 102, "Low": 99, "Close": 101, "Volume": 1000,
        }),
    ]
    history = Mock(return_value=SimpleNamespace(empty=False, iterrows=lambda: iter(entries)))
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(
        Ticker=lambda ticker: SimpleNamespace(history=history),
    ))
    rows, warnings = fetch_yfinance_ohlcv("SPY", as_of=date(2026, 10, 27))
    assert [row["date"] for row in rows] == ["2026-10-26"]
    assert history.call_args.kwargs["end"] == "2026-10-28"
    assert history.call_args.kwargs["timeout"] == 10
    assert history.call_args.kwargs["auto_adjust"] is False
    assert any("discarded 1" in warning and "2026-10-26" in warning for warning in warnings)


def test_historical_macro_excludes_future_observations():
    rows = [{"series_id": "DGS10", "date": "2026-10-28", "value": "5.1", "history": [
        {"date": "2026-10-26", "value": "4.2"},
        {"date": "2026-10-28", "value": "5.1"},
    ]}]
    result = cli._dated_macro(rows, "2026-10-27")
    assert result[0]["value"] == "4.2"
    assert len(result[0]["history"]) == 1
