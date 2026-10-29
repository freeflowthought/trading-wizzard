"""Crypto references survive independently of the SPY/QQQ-only equity reference scope."""
import json
from datetime import datetime, timezone

import pytest

from twiz.ingest import cross_market as cm

NOW = datetime(2026, 10, 28, 1, tzinfo=timezone.utc)
DATE = "2026-10-27"


def market(symbol, market_id, mark=770.24, index=770.10, last=770.21):
    return {"symbol": symbol, "market_id": market_id, "market_type": "perp", "status": "active",
            "is_frozen": False, "market_deployer": "lighter", "mark_price": str(mark),
            "index_price": str(index), "last_trade_price": last, "daily_price_change": 0.540385,
            "daily_quote_token_volume": 16_556_246.605, "open_interest_usd": "9299615.570304",
            "open_interest": 6036.8298, "base_interest_rate": 0.0001}


def payload():
    qqq = market("QQQ", 129, 722.83, 722.66, 722.85)
    qqq.update(daily_price_change=1.54812, daily_quote_token_volume=22_429_174.819,
               open_interest_usd="14639286.898")
    return {"code": 200, "order_book_details": [market("SPY", 128), qqq],
            "spot_order_book_details": []}


def crypto_payload():
    return [{"universe": [{"name": "ETH"}, {"name": "BTC"}]}, [
        {"markPx": "2342.8", "oraclePx": "2343.7", "prevDayPx": "2300", "funding": "-0.00001",
         "openInterest": "123", "dayNtlVlm": "456000"},
        {"markPx": "80656", "oraclePx": "80690", "prevDayPx": "80000", "funding": "0.0000125",
         "openInterest": "43473.16026", "dayNtlVlm": "8000000"},
    ]]


@pytest.fixture(autouse=True)
def mock_crypto_transport(monkeypatch):
    monkeypatch.setattr(cm, "_fetch_crypto_contexts", crypto_payload)


def fetch(monkeypatch, data=None):
    monkeypatch.setattr(cm, "_fetch_details", lambda: payload() if data is None else data)
    return cm._fetch_equity_rows(NOW)


def test_exact_equity_perpetuals_use_provider_percentage_and_usd_units(monkeypatch):
    rows, warnings = fetch(monkeypatch)
    assert not warnings and [row["reference_for"] for row in rows] == ["SPY", "QQQ"]
    spy, qqq = rows
    assert spy["mark_price"] == 770.24 and qqq["mark_price"] == 722.83
    assert qqq["day_change_pct"] == 1.54812  # Already a percent; do not multiply by 100.
    assert spy["basis_pct"] == pytest.approx((770.24/770.10-1)*100)
    assert spy["open_interest_usd"] == 9299615.570304
    assert spy["open_interest_notional_usd"] == 9299615.570304
    assert spy["open_interest_base"] is None  # No inference from differently scaled OI units.
    for row in rows:
        assert row["source"] == "Lighter" and row["asset_type"] == "equity_perpetual"
        assert row["mapping_verified"] is True and row["status"] == "reference_only"
        assert row["funding_rate_raw"] is None and row["funding_unit"] == "unavailable"
        assert row["exchange_updated_at"] is None and row["quote_age_verified"] is False
        assert "not official ETF close" in row["reason"] and "age unknown" in row["reason"]
        assert row["source_url"] == cm.SPEC_URL and row["price_type"] == "mark_price"
    json.dumps(rows, allow_nan=False)


def test_one_get_and_configuration_cannot_remap_to_another_instrument(monkeypatch):
    calls = []
    monkeypatch.setattr(cm, "_fetch_details", lambda: calls.append(True) or payload())
    config = {"references": [{"reference_for": "QQQ", "instrument_name": "SPX"}]}
    rows, warnings = cm.fetch_cross_market(DATE, live=True, now=NOW, config=config)
    assert not warnings and len(calls) == 1
    assert [row["instrument_name"] for row in rows
            if row["asset_class"] == "equity"] == ["SPY", "QQQ"]


def test_spx_meme_and_rhqqq_spot_never_substitute_for_etfs(monkeypatch):
    data = payload()
    data["order_book_details"].append(market("SPX", 999, 0.388, 0.388, 0.388))
    data["spot_order_book_details"] = [market("rhQQQ", 129, 1, 1, 1)]
    rows, warnings = fetch(monkeypatch, data)
    assert not warnings and rows[1]["mark_price"] == 722.83
    assert len(rows) == 2 and all(row["instrument_name"] in {"SPY", "QQQ"} for row in rows)


@pytest.mark.parametrize("change", [{"market_id": 999}, {"symbol": "SPX"},
    {"market_type": "spot"}, {"status": "inactive"}, {"is_frozen": True},
    {"is_frozen": None}, {"symbol": "rhQQQ"}])
def test_exact_identity_and_active_unfrozen_perpetual_guards(monkeypatch, change):
    data = payload()
    data["order_book_details"][0].update(change)
    rows, warnings = fetch(monkeypatch, data)
    assert warnings and rows[0]["status"] == "unavailable" and rows[0]["mark_price"] is None
    assert rows[0]["mapping_verified"] is False and rows[1]["status"] == "reference_only"


@pytest.mark.parametrize("field,value", [("mark_price", "nan"), ("index_price", 0),
    ("last_trade_price", -1), ("daily_price_change", "inf"),
    ("daily_quote_token_volume", 9999), ("open_interest_usd", -1),
    ("daily_quote_token_volume", True)])
def test_invalid_market_numbers_are_not_emitted_or_replaced(monkeypatch, field, value):
    data = payload()
    data["order_book_details"][0][field] = value
    rows, warnings = fetch(monkeypatch, data)
    assert warnings and rows[0]["status"] == "unavailable" and rows[0]["mark_price"] is None
    json.dumps(rows, allow_nan=False)


def test_duplicate_exact_market_is_not_arbitrarily_selected(monkeypatch):
    data = payload()
    data["order_book_details"].append(dict(data["order_book_details"][0]))
    rows, warnings = fetch(monkeypatch, data)
    assert warnings and rows[0]["status"] == "unavailable"


def test_missing_perpetual_is_not_backfilled_from_spot_table(monkeypatch):
    data = payload()
    data["spot_order_book_details"] = data["order_book_details"]
    data["order_book_details"] = []
    rows, warnings = fetch(monkeypatch, data)
    assert warnings and all(row["status"] == "unavailable" for row in rows)


def test_provider_error_code_cannot_supply_valid_quotes(monkeypatch):
    data = payload()
    data["code"] = 500
    rows, warnings = fetch(monkeypatch, data)
    assert warnings and all(row["mark_price"] is None for row in rows)


def test_network_failure_keeps_two_unavailable_rows(monkeypatch):
    def fail():
        raise TimeoutError("provider unavailable")
    monkeypatch.setattr(cm, "_fetch_details", fail)
    rows, warnings = cm.fetch_cross_market(DATE, live=True, now=NOW)
    assert len(rows) == 4 and warnings
    assert all(row["status"] == "unavailable" for row in rows if row["asset_class"] == "equity")
    assert all(row["status"] == "reference_only" for row in rows if row["asset_class"] == "crypto")


def test_offline_and_historical_reports_never_fetch_current_prices(monkeypatch):
    calls = []
    monkeypatch.setattr(cm, "_fetch_details", lambda: calls.append(True))
    assert cm.fetch_cross_market(DATE, live=False, now=NOW)[0] == []
    assert cm.fetch_cross_market("2026-10-26", live=True, now=NOW)[0] == []
    assert not calls


def test_zero_usd_oi_and_negative_daily_change_are_valid_at_volume_threshold(monkeypatch):
    data = payload()
    data["order_book_details"][0].update(open_interest_usd=0, daily_price_change=-2.5,
                                         daily_quote_token_volume=10_000)
    rows, warnings = fetch(monkeypatch, data)
    assert not warnings and rows[0]["open_interest_usd"] == 0 and rows[0]["day_change_pct"] == -2.5



def test_four_reference_rows_keep_equity_scope_and_crypto_identity_separate(monkeypatch):
    calls = {"crypto": 0, "equity": 0}
    def crypto_provider():
        calls["crypto"] += 1
        return crypto_payload()
    def equity_provider():
        calls["equity"] += 1
        return payload()
    monkeypatch.setattr(cm, "_fetch_crypto_contexts", crypto_provider)
    monkeypatch.setattr(cm, "_fetch_details", equity_provider)
    rows, warnings = cm.fetch_cross_market(DATE, live=True, now=NOW)
    assert not warnings and calls == {"crypto": 1, "equity": 1}
    assert [row["symbol"] for row in rows] == ["BTC", "ETH", "SPY", "QQQ"]
    crypto, equity = rows[:2], rows[2:]
    assert all(row["asset_class"] == "crypto" and not row["equity_mapping_verified"]
               for row in crypto)
    assert all(row["asset_class"] == "equity" and row["equity_mapping_verified"] for row in equity)
    assert crypto[0]["day_change_pct"] == pytest.approx((80656/80000-1)*100)
    assert crypto[0]["open_interest_notional_usd"] == pytest.approx(43473.16026*80656)
    assert crypto[1]["funding_rate_raw"] == -0.00001
    assert all(row["exchange_updated_at"] is None and not row["quote_age_verified"] for row in rows)


def test_crypto_provider_failure_does_not_remove_equity_references(monkeypatch):
    def fail():
        raise TimeoutError("crypto provider unavailable")
    monkeypatch.setattr(cm, "_fetch_crypto_contexts", fail)
    monkeypatch.setattr(cm, "_fetch_details", payload)
    rows, warnings = cm.fetch_cross_market(DATE, live=True, now=NOW)
    assert len(rows) == 4 and warnings
    assert all(row["status"] == "unavailable" for row in rows[:2])
    assert all(row["status"] == "reference_only" for row in rows[2:])


@pytest.mark.parametrize("field,value", [("markPx", "NaN"), ("prevDayPx", 0),
    ("openInterest", -1), ("funding", "inf"), ("dayNtlVlm", -1)])
def test_invalid_crypto_context_is_isolated_to_one_asset(monkeypatch, field, value):
    data = crypto_payload()
    data[1][1][field] = value
    monkeypatch.setattr(cm, "_fetch_crypto_contexts", lambda: data)
    monkeypatch.setattr(cm, "_fetch_details", payload)
    rows, warnings = cm.fetch_cross_market(DATE, live=True, now=NOW)
    assert warnings and rows[0]["status"] == "unavailable"
    assert all(row["status"] == "reference_only" for row in rows[1:])
    json.dumps(rows, allow_nan=False)


def test_ambiguous_crypto_indices_are_not_used_as_price_fallbacks(monkeypatch):
    data = crypto_payload()
    data[0]["universe"].append({"name": "BTC"})  # Context count mismatch is a source-wide failure.
    monkeypatch.setattr(cm, "_fetch_crypto_contexts", lambda: data)
    monkeypatch.setattr(cm, "_fetch_details", payload)
    rows, warnings = cm.fetch_cross_market(DATE, live=True, now=NOW)
    assert warnings and all(row["status"] == "unavailable" for row in rows[:2])
    assert all(row["status"] == "reference_only" for row in rows[2:])
