"""Nearest-expiry chain capture and reproducible valuation assumptions, without network."""
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from twiz.models import OptionContract
from twiz.options import (
    black_scholes_greeks,
    build_option_snapshot,
    build_option_snapshots,
    enrich_contract_greeks,
    fetch_yfinance_option_contracts,
    summarize_options,
    years_to_expiration,
)


def contract(symbol="CALL", kind="call", strike=100, expiration="2026-11-20", **kwargs):
    fields = {"bid": 2, "ask": 3, "last": 2.5, "implied_volatility": .3,
              "volume": 10, "open_interest": 20, **kwargs}
    return OptionContract(symbol, kind, strike, expiration, **fields)


def test_full_chain_is_copied_before_greeks_and_display_selection():
    supplied = [contract(f"{kind}{i}", kind, 70+i) for kind in ("call", "put") for i in range(40)]
    snapshot = summarize_options("QQQ", 100, supplied, "2026-11-20",
                                 valuation_date=date(2026, 10, 20))
    assert len(snapshot.full_chain_contracts) == 80 and len(snapshot.contracts) == 20
    assert all(row["delta"] is None for row in snapshot.full_chain_contracts)
    assert all(row.delta is None for row in supplied)
    assert all(row["delta"] is not None for row in snapshot.contracts)
    assert snapshot.full_chain_contracts[0]["symbol"] == "call0"
    snapshot.contracts[0]["bid"] = 999
    assert all(row["bid"] == 2 for row in snapshot.full_chain_contracts)
    assert snapshot.chain_metadata["returned_contract_count"] == 80
    assert snapshot.chain_metadata["display_contract_count"] == 20
    assert snapshot.chain_metadata["all_expirations_captured"] is False


def test_supplied_other_expiry_and_invalid_iv_are_captured_honestly_before_filtering():
    supplied = [
        contract("RAW", expiration="2026-11-20"), contract("OTHER", expiration="2026-12-18")
    ]
    supplied[0].implied_volatility = float("nan")
    snapshot = summarize_options("QQQ", 100, supplied, "2026-11-20",
                                 valuation_date=date(2026, 10, 20))
    assert len(snapshot.full_chain_contracts) == 2 and len(snapshot.contracts) == 1
    assert snapshot.full_chain_contracts[0]["implied_volatility"] is None
    assert math.isnan(supplied[0].implied_volatility)
    assert snapshot.contracts[0]["delta"] is None
    assert snapshot.chain_metadata["eligible_contract_count"] == 1
    assert snapshot.chain_metadata["scope"] == "supplied_contract_records_including_other_expiries"
    json.dumps(snapshot.to_dict(), allow_nan=False)


def test_explicit_historical_valuation_changes_greeks_with_date_not_wall_clock():
    day, expiration = date(2026, 6, 1), "2026-07-01"
    result = enrich_contract_greeks([contract(expiration=expiration)], 100, .02,
                                    valuation_date=day, dividend_yield=.01)[0]
    expected = black_scholes_greeks("call", 100, 100, 30/365, .02, .3, .01)
    assert result.delta == expected["delta"] and result.theta == expected["theta"]
    later = enrich_contract_greeks([contract(expiration=expiration)], 100, .02,
                                   valuation_date=date(2026, 6, 30), dividend_yield=.01)[0]
    assert later.theta != result.theta
    snapshot = summarize_options("QQQ", 100, [contract(expiration=expiration)], expiration,
                                 valuation_date=day, risk_free_rate=.02, dividend_yield=.01)
    assert snapshot.greeks["delta"] == expected["delta"]
    assert snapshot.chain_metadata["valuation_date"] == "2026-06-01"
    assert snapshot.chain_metadata["risk_free_rate"] == .02
    assert snapshot.chain_metadata["dividend_yield"] == .01
    assert "European" in snapshot.chain_metadata["greek_model"]
    assert snapshot.chain_metadata["quote_timestamp"] is None
    assert snapshot.chain_metadata["quote_age_verified"] is False


def test_act365_floor_and_legacy_today_keyword_remain_compatible():
    assert years_to_expiration("2026-06-30", today=date(2026, 6, 1)) == 29/365
    assert years_to_expiration("2026-06-01", valuation_date=date(2026, 6, 1)) == 1/365
    assert years_to_expiration("2026-05-01", valuation_date=date(2026, 6, 1)) == 1/365
    with pytest.raises(ValueError, match="conflict"):
        years_to_expiration("2026-06-30", date(2026, 6, 1), valuation_date=date(2026, 6, 2))


class Frame:
    def __init__(self, rows):
        self.rows, self.empty = rows, not rows

    def iterrows(self):
        return iter(enumerate(self.rows))

    def head(self, limit):
        return Frame(self.rows[:limit])


def test_provider_last_trade_time_is_preserved_and_is_not_a_quote_timestamp(monkeypatch):
    trade = datetime(2026, 10, 27, 12, tzinfo=timezone(timedelta(hours=-4)))
    frame = Frame([{"contractSymbol": "C", "strike": 100, "lastTradeDate": trade,
                    "bid": 1, "ask": 2, "impliedVolatility": .2}])
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=lambda _: SimpleNamespace(
        options=["2026-11-20"], option_chain=lambda _: SimpleNamespace(calls=frame, puts=None)
    )))
    contracts, expiry, _ = fetch_yfinance_option_contracts("QQQ", 100)
    assert contracts[0].last_trade_at == "2026-10-27T16:00:00+00:00"
    snapshot = summarize_options("QQQ", 100, contracts, expiry, valuation_date=date(2026, 10, 27))
    assert snapshot.full_chain_contracts[0]["last_trade_at"] == contracts[0].last_trade_at
    assert snapshot.chain_metadata["quote_timestamp"] is None
    assert snapshot.chain_metadata["option_chain_fetched_at"] is None
    assert "not a bid/ask quote" in snapshot.chain_metadata["last_trade_time_meaning"]
    frame.rows[0]["lastTradeDate"] = datetime(2026, 10, 27, 12)  # Unknown zone is not invented.
    assert fetch_yfinance_option_contracts("QQQ", 100)[0][0].last_trade_at is None


def test_daily_snapshot_skips_expired_and_same_day_chain_after_completed_session(monkeypatch):
    frame = Frame([{"contractSymbol": "C", "strike": 100, "bid": 1, "ask": 2,
                    "impliedVolatility": .2}])
    fetch = Mock(return_value=SimpleNamespace(calls=frame, puts=None))
    instrument = SimpleNamespace(options=["2026-11-06", "bad", "2026-10-28",
                                          "2026-10-29", "2026-10-27"], option_chain=fetch)
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=lambda _: instrument))
    contracts, expiry, _ = fetch_yfinance_option_contracts(
        "QQQ", 100, valuation_date=date(2026, 10, 28)
    )
    assert expiry == "2026-10-29" and contracts[0].expiration == expiry
    fetch.assert_called_once_with("2026-10-29")


def test_no_future_expiration_is_unknown_instead_of_reusing_same_day_chain(monkeypatch):
    fetch = Mock()
    instrument = SimpleNamespace(options=["2026-10-28"], option_chain=fetch)
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=lambda _: instrument))
    contracts, expiry, warnings = fetch_yfinance_option_contracts(
        "QQQ", 100, valuation_date=date(2026, 10, 28)
    )
    assert contracts == [] and expiry is None and warnings
    fetch.assert_not_called()


@pytest.mark.parametrize("valuation", [date(2026, 6, 1), date.today()+timedelta(days=3)])
def test_live_builder_never_backfills_other_valuation_dates(monkeypatch, valuation):
    fetch = Mock(side_effect=AssertionError("current chain is not historical data"))
    monkeypatch.setattr("twiz.options.fetch_yfinance_option_contracts", fetch)
    snapshot = build_option_snapshot("QQQ", 100, live=True, valuation_date=valuation)
    assert not snapshot.contracts and not snapshot.full_chain_contracts
    assert snapshot.chain_metadata["option_chain_fetched_at"] is None
    assert snapshot.chain_metadata["collection_attempted"] is False
    assert snapshot.chain_metadata["valuation_date"] == valuation.isoformat()
    assert any("not backfilled" in warning for warning in snapshot.warnings)
    fetch.assert_not_called()


def test_live_collection_threads_underlying_and_valuation_assumptions(monkeypatch):
    today = date.today()
    expiry = (today+timedelta(days=30)).isoformat()
    fetch = Mock(return_value=([contract(expiration=expiry)], expiry, []))
    monkeypatch.setattr("twiz.options.fetch_yfinance_option_contracts", fetch)
    result = build_option_snapshots({"QQQ": [{
        "date": (today-timedelta(days=1)).isoformat(), "close": 100, "source": "yfinance",
        "fetched_at": "2026-10-28T01:00:00+00:00",
    }]}, live=True, valuation_date=today, risk_free_rate=.03, dividend_yield=.01)["QQQ"]
    metadata = result["chain_metadata"]
    assert metadata["source"] == "yfinance" and metadata["collection_attempted"] is True
    assert metadata["underlying_spot"] == 100 and metadata["underlying_source"] == "yfinance"
    assert metadata["underlying_date"] == result["underlying_date"]
    assert metadata["underlying_fetched_at"] == "2026-10-28T01:00:00+00:00"
    assert metadata["option_chain_fetched_at"] == result["fetched_at"]
    assert metadata["valuation_date"] == today.isoformat()
    assert metadata["risk_free_rate"] == .03 and metadata["dividend_yield"] == .01
    assert metadata["quote_age_verified"] is False
    assert "nearest_expiry" in metadata["scope"]
    assert metadata["expiry_selection"] == "strictly_after_daily_valuation_date"
    assert metadata["same_day_expiry_excluded"] is True
    assert len(result["full_chain_contracts"]) == 1


def test_missing_provider_positioning_stays_distinct_from_real_zero(monkeypatch):
    frame = Frame([
        {"contractSymbol": "UNKNOWN", "strike": 99, "impliedVolatility": .2,
         "volume": math.nan, "openInterest": None},
        {"contractSymbol": "ZERO", "strike": 100, "impliedVolatility": .2,
         "volume": 0, "openInterest": 0},
        {"contractSymbol": "KNOWN", "strike": 101, "impliedVolatility": .2,
         "volume": 10, "openInterest": 20},
    ])
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=lambda _: SimpleNamespace(
        options=["2026-11-20"], option_chain=lambda _: SimpleNamespace(calls=frame, puts=None)
    )))
    contracts, expiry, _ = fetch_yfinance_option_contracts("QQQ", 100)
    assert contracts[0].volume is None and contracts[0].open_interest is None
    assert contracts[1].volume == contracts[1].open_interest == 0
    snapshot = summarize_options("QQQ", 100, contracts, expiry, valuation_date=date(2026, 10, 20))
    raw = {row["symbol"]: row for row in snapshot.full_chain_contracts}
    selected = {row["symbol"]: row for row in snapshot.contracts}
    for captured in (raw, selected):
        assert captured["UNKNOWN"]["volume"] is None
        assert captured["UNKNOWN"]["open_interest"] is None
        assert captured["ZERO"]["volume"] == captured["ZERO"]["open_interest"] == 0
    assert snapshot.skew_summary["call_wall"] == 101
    expected = {"total": 3, "known": 2, "missing": 1, "known_zero": 1}
    coverage = snapshot.chain_metadata["eligible_field_coverage"]
    assert coverage["volume"] == coverage["open_interest"] == expected
    assert "not as observed zero" in snapshot.chain_metadata["positioning_aggregation"]
    json.dumps(snapshot.to_dict(), allow_nan=False)


def test_cli_records_requested_report_valuation_date_without_fetching_current_chain(tmp_path,
                                                                                 monkeypatch):
    from twiz import cli
    from twiz.storage import ProjectPaths, write_json

    day = "2026-06-08"
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    market = {"QQQ": [{"date": day, "close": 100, "source": "yfinance"}]}
    write_json(paths.raw / f"market-{day}.json", market)
    monkeypatch.setattr(cli, "ProjectPaths", lambda root=None: paths if root is None else
                        ProjectPaths(root))
    for name in ("cmd_ingest", "cmd_analyze", "cmd_backtest", "_build_events_and_playbooks"):
        monkeypatch.setattr(cli, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_load_or_create_market", lambda *args, **kwargs: market)
    options = Mock(return_value={})
    monkeypatch.setattr(cli, "build_option_snapshots", options)
    monkeypatch.setattr(cli, "build_fundamental_snapshots", Mock(return_value={}))
    cli.cmd_report(day, live=True)
    options.assert_called_once_with(market, live=False, valuation_date=date(2026, 6, 8))
