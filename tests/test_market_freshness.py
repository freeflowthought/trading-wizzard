import sys
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from twiz.ingest.market import fetch_yfinance_ohlcv
from twiz.sessions import expected_complete_session, is_us_equity_session


def _now(hour=21):
    return datetime(2026, 10, 27, hour, tzinfo=timezone.utc)


@pytest.mark.parametrize("report_date,now,expected", [
    ("2026-10-27", _now(21), "2026-10-27"),
    ("2026-10-27", _now(19), "2026-10-26"),
    ("2026-10-25", _now(), "2026-10-23"),
    ("2026-06-19", _now(), "2026-06-18"),
    ("2026-04-03", _now(), "2026-04-02"),
])
def test_latest_completed_equity_session(report_date, now, expected):
    assert expected_complete_session(report_date, now) == expected


def test_bank_only_holidays_remain_equity_sessions():
    assert is_us_equity_session(date(2026, 10, 12))  # Columbus Day.
    assert is_us_equity_session(date(2026, 11, 11))  # Veterans Day.


def _instrument(metadata):
    row = {"Open": 715.12, "High": 722.87, "Low": 713.59,
           "Close": float("nan"), "Adj Close": float("nan"), "Volume": 30_000_000}
    frame = SimpleNamespace(empty=False, iterrows=lambda: iter([(datetime(2026, 10, 27), row)]))
    return SimpleNamespace(history=Mock(return_value=frame), history_metadata=metadata)


def _metadata():
    close_time = int(datetime(2026, 10, 27, 20, tzinfo=timezone.utc).timestamp())
    return {"regularMarketPrice": 721.25, "regularMarketTime": close_time,
            "currentTradingPeriod": {"regular": {"end": close_time}},
            "exchangeTimezoneName": "America/New_York"}


def test_after_close_metadata_recovers_missing_qqq_daily_close(monkeypatch):
    instrument = _instrument(_metadata())
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=lambda _: instrument))
    rows, warnings = fetch_yfinance_ohlcv("QQQ", as_of=date(2026, 10, 27), now=_now())
    assert rows[-1]["close"] == 721.25
    assert rows[-1]["date"] == "2026-10-27"
    assert rows[-1]["close_source"] == "regular_market_metadata"
    assert any("metadata" in warning for warning in warnings)


@pytest.mark.parametrize("change", [
    {"regularMarketTime": 1793127600},  # 15:00 NY is not a closing quote.
    {"regularMarketPrice": 900},  # Outside the reported session range.
    {"regularMarketPrice": float("nan")},
    {"exchangeTimezoneName": "invalid/zone"},
])
def test_unconfirmed_metadata_does_not_invent_a_close(monkeypatch, change):
    instrument = _instrument({**_metadata(), **change})
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=lambda _: instrument))
    rows, _ = fetch_yfinance_ohlcv("QQQ", as_of=date(2026, 10, 27), now=_now())
    assert rows == []


def test_intraday_bar_is_not_treated_as_a_completed_close(monkeypatch):
    instrument = _instrument(_metadata())
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=lambda _: instrument))
    rows, _ = fetch_yfinance_ohlcv("QQQ", as_of=date(2026, 10, 27), now=_now(19))
    assert rows == []
