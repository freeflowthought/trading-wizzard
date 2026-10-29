"""Regional probabilities are subordinate to observed US and sector transmission."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from twiz.config import load_prediction_markets_config
from twiz.scenarios import _semiconductor_transmission, build_scenario_context

DAY = "2026-10-27"
NOW = datetime(2026, 10, 28, 2, tzinfo=timezone.utc)
DAYS = ["2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-26", DAY]


def event(driver, yes):
    contract = next(row for row in load_prediction_markets_config()["markets"]
                    if row.get("driver") == driver)
    return {"id": contract["id"], "driver": driver, "question": contract["expected_question"],
            "source": "Polymarket Gamma", "status": "uncertain", "yes_probability": yes,
            "no_probability": 1-yes, "fetched_at": NOW.isoformat(),
            "market_updated_at": (NOW-timedelta(minutes=10)).isoformat(),
            "end_date": (NOW+timedelta(days=1)).isoformat(),
            "liquidity_usd": 10000, "spread": .02}


def observation(series, first, last):
    history = [{"date": day, "value": first+(last-first)*i/5} for i, day in enumerate(DAYS)]
    return {"series_id": series, "source": "ECB reference FX" if series in {
        "USDJPY", "USDKRW"} else "FRED", "history": history, **history[-1]}


def market(semis_end=98):
    return {ticker: [{"date": day, "close": 100+(last-100)*i/5,
                      "source": "yfinance", "session_complete": True,
                      "price_basis": "consistent_test_basis"} for i, day in enumerate(DAYS)]
            for ticker, last in (("SPY", 101), ("SMH", semis_end))}


def context(regional=(), extra_macro=(), prices=None, core=None, allocation_policy=None):
    macro = [observation("DGS10", 5.2, 4.9), observation("BAMLH0A0HYM2", 3.4, 3.3)]
    core = [event("fed_hold", .96), event("near_term_ceasefire", .9)] if core is None else core
    return build_scenario_context(DAY, [*macro, *extra_macro], [*core, *regional],
                                  prices or {}, now=NOW, allocation_policy=allocation_policy)


def test_regional_limits_and_explanation_scale_with_configured_capital():
    result = context(extra_macro=[observation("USDJPY", 150, 145)], prices=market(99),
                     allocation_policy={"max_new_capital_pct": 100})
    assert result["regional_transmission"][0]["transmission_confirmed"]
    assert result["policy"]["initial_cap_pct"] == 10
    assert result["policy"]["post_event_cap_pct"] == 20
    assert "首批10%" in result["regional_transmission"][0]["action"]
    assert "合计20%" in result["regional_transmission"][0]["action"]
    assert "首批再压至10%以内" in result["headline"]


def test_registry_binds_exact_questions_and_preserves_no_complement_semantics():
    registered = {row["driver"]: row for row in load_prediction_markets_config()["markets"]}
    assert registered["boj_hold"]["id"] == "3251602"
    assert registered["boj_hold"]["expected_question"] == (
        "No change in Bank of Japan’s interest rates after the October 2026 meeting?")
    assert registered["bok_hike_25"]["id"] == "3940931"
    result = context([event("boj_hold", .01), event("bok_hike_25", .01)])
    japan, korea = result["regional_transmission"]
    assert japan["event_main_case"] == korea["event_main_case"] == "No"
    assert "不自动等于加息" in japan["event_semantics"]
    assert "不自动等于维持或降息" in korea["event_semantics"]
    assert japan["event_probability"] == korea["event_probability"] == .99
    assert japan["event_yes_probability"] == .01 and japan["event_no_probability"] == .99
    assert [row["priority"] for row in result["event_priority"]] == ["high", "secondary", "lower"]


def test_isolated_high_regional_probability_never_changes_us_core_or_creates_buy_authority():
    baseline = context()
    regional = [event("boj_hold", .01), event("bok_hike_25", .99)]
    result = context(regional)
    assert result["state"] == baseline["state"] == "repair_bias"
    assert result["policy"] == baseline["policy"]
    missing_us = context(regional, core=[])
    macro_only = context(core=[])
    # Verified US rates/credit can now support a1% macro-only repair plan;
    # regional contracts cannot grant additional authority or change that policy.
    assert missing_us["state"] == macro_only["state"] == "repair_bias"
    assert missing_us["policy"] == macro_only["policy"]
    assert missing_us["policy"]["initial_cap_pct"] == 1
    stressed = build_scenario_context(DAY, [observation("DGS10", 4.9, 5.2),
        observation("BAMLH0A0HYM2", 3.1, 3.5)], [event("fed_hold", .96),
        event("near_term_ceasefire", .9), *regional], {}, now=NOW)
    assert stressed["state"] == "stress_bias" and stressed["policy"]["initial_cap_pct"] == 0


def test_japan_fx_needs_same_period_semiconductor_underperformance_before_broad_cap():
    fx = observation("USDJPY", 150, 150/1.02)
    no_semis = context([event("boj_hold", .01)], [fx])
    assert no_semis["policy"]["initial_cap_pct"] == 2
    strong_semis = context([event("boj_hold", .01)], [fx], market(102))
    assert strong_semis["policy"]["initial_cap_pct"] == 2
    result = context([event("boj_hold", .01)], [fx], market(99))
    japan = result["regional_transmission"][0]
    assert result["state"] == "repair_bias"
    assert result["policy"]["initial_cap_pct"] == 1
    assert result["policy"]["post_event_cap_pct"] == 2
    assert japan["transmission_confirmed"] is japan["qqq_priority_risk"] is True
    assert japan["fx"]["local_currency_strength_5_pct"] == pytest.approx(2)
    assert japan["semiconductor"]["vs_spy_5d_pp"] == pytest.approx(-2)
    assert japan["semiconductor"]["start_date"] == DAYS[0]
    assert japan["semiconductor"]["end_date"] == DAY


def test_korea_fx_plus_semis_is_sector_caution_with_no_us_budget_or_direction_override():
    fx = observation("USDKRW", 1300, 1340)
    result = context([event("bok_hike_25", .99)], [fx], market())
    korea = result["regional_transmission"][1]
    assert korea["priority"] == "lower" and korea["transmission_confirmed"] is True
    assert korea["qqq_priority_risk"] is True and "半导体" in korea["action"]
    assert result["state"] == "repair_bias"
    assert result["policy"]["initial_cap_pct"] == 2
    assert result["policy"]["post_event_cap_pct"] == 5
    alone = context([event("bok_hike_25", .99)], [fx], market(102))
    assert alone["regional_transmission"][1]["transmission_confirmed"] is False
    assert alone["policy"]["initial_cap_pct"] == 2


@pytest.mark.parametrize("change", [
    {"liquidity_usd": 999}, {"spread": .11}, {"id": "wrong"},
    {"market_updated_at": (NOW-timedelta(days=2)).isoformat()},
    {"fetched_at": (NOW+timedelta(days=1)).isoformat()},
    {"question": "Will the Bank of Korea leave rates unchanged?"},
])
def test_thin_stale_or_wrong_regional_contract_is_unknown_not_an_event_main_case(change):
    result = context([{**event("bok_hike_25", .99), **change}])
    korea = result["regional_transmission"][1]
    assert korea["event_quality"] == "unavailable" and korea["event_main_case"] == "unknown"
    assert korea["event_probability"] is None
    assert result["policy"]["initial_cap_pct"] == 2
    assert not any(row["driver"] == "bok_hike_25" for row in result["events"])


def test_saved_bok_quote_is_rejected_as_thin_and_cannot_be_backfilled_into_october27():
    # Exact fields from the saved Gamma3940931 capture; test needs no ignored local artifact.
    raw = {"question": "Will the Bank of Korea hike by 25 bps at the November 2026 meeting?",
           "updatedAt": "2026-10-28T08:10:03.446401Z", "endDate": "2026-11-26T00:00:00Z",
           "liquidity": "921.9903", "spread": .05}
    row = {**event("bok_hike_25", .845), "fetched_at": "2026-10-28T08:14:23.642137+00:00",
           "market_updated_at": raw["updatedAt"], "end_date": raw["endDate"],
           "liquidity_usd": float(raw["liquidity"]), "spread": raw["spread"]}
    assert raw["question"] == row["question"] and row["liquidity_usd"] < 1000
    now = datetime(2026, 10, 28, 9, tzinfo=timezone.utc)
    current = build_scenario_context("2026-10-28", [], [row], {}, now=now)
    assert current["regional_transmission"][1]["event_quality"] == "unavailable"
    row["liquidity_usd"] = 10000  # Date still prevents using Oct28 information as Oct27.
    historical = build_scenario_context(DAY, [], [row], {}, now=now)
    assert historical["events"] == []


@pytest.mark.parametrize("change", ["incomplete", "missing", "duplicate", "sample", "future"])
def test_invalid_or_misaligned_semiconductor_prices_do_not_establish_transmission(change):
    prices = market()
    if change == "incomplete":
        prices["SMH"][-1]["session_complete"] = False
    elif change == "missing":
        prices["SMH"].pop(2)
    elif change == "duplicate":
        prices["SMH"].append(deepcopy(prices["SMH"][-1]))
    elif change == "sample":
        prices["SMH"][-1]["source"] = "sample"
    else:
        prices["SMH"][-1]["date"] = "2026-10-28"
    result = context([event("boj_hold", .01)], [observation("USDJPY", 150, 145)], prices)
    assert result["regional_transmission"][0]["semiconductor"]["status"] == "unavailable"
    assert result["regional_transmission"][0]["transmission_confirmed"] is False
    assert result["policy"]["initial_cap_pct"] == 2


@pytest.mark.parametrize("stamps", [
    ["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07", DAY],
    ["2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"],
])
def test_sparse_or_stale_fx_stays_visible_but_cannot_join_current_semis_into_a_broad_cap(stamps):
    fx = observation("USDJPY", 150, 145)
    for item, stamp in zip(fx["history"], stamps):
        item["date"] = stamp
    fx.update(fx["history"][-1])
    result = context([event("boj_hold", .01)], [fx], market())
    japan = result["regional_transmission"][0]
    assert japan["fx"]["start_date_5"] == stamps[0]
    assert japan["fx"]["change_5_pct"] < -3  # Display the actual dated quote movement.
    assert japan["fx"]["alignment"] == "misaligned"
    assert japan["fx"]["pressure"] is None and japan["transmission_confirmed"] is False
    assert result["policy"]["initial_cap_pct"] == 2
    assert "不授予区域风控权限" in japan["reason"]


def test_missing_fx_dates_never_supply_regional_authority():
    fx = observation("USDJPY", 150, 145)
    fx["history"] = fx["history"][-4:]
    result = context([event("boj_hold", .01)], [fx], market())
    japan = result["regional_transmission"][0]
    assert japan["fx"]["start_date_5"] is None and japan["fx"]["pressure"] is None
    assert japan["fx"]["alignment"] == "unavailable"
    assert result["policy"]["initial_cap_pct"] == 2


def test_fx_later_than_semiconductor_session_is_visible_without_authority(monkeypatch):
    monkeypatch.setattr("twiz.scenarios.expected_complete_session", lambda *args, **kwargs: DAY)
    fx = observation("USDJPY", 150, 145)
    fx["history"][-1]["date"] = "2026-10-28"
    fx.update(fx["history"][-1])
    result = build_scenario_context("2026-10-28", [fx], [], market(), now=NOW)
    japan = result["regional_transmission"][0]
    assert japan["fx"]["date"] == "2026-10-28"
    assert japan["fx"]["alignment"] == "misaligned" and japan["fx"]["pressure"] is None
    assert japan["transmission_confirmed"] is False


def test_cross_ticker_price_basis_mismatch_cannot_generate_regional_authority():
    prices = market()
    for row in prices["SMH"]:
        row["price_basis"] = "raw_as_traded"
    result = context([event("boj_hold", .01)], [observation("USDJPY", 150, 145)], prices)
    assert result["regional_transmission"][0]["semiconductor"]["status"] == "unavailable"
    assert result["policy"]["initial_cap_pct"] == 2


@pytest.mark.parametrize("fetched,flag,expected", [
    ("2026-10-28T01:14:33+00:00", "missing", True),
    ("2026-10-27T19:00:00+00:00", "missing", False),  # Before the Oct27 regular close.
    (None, "missing", False),
    ("2026-10-28T01:14:33", "missing", False),  # An unzoned retrieval is not evidence.
    ("2026-10-28T01:14:33+00:00", False, False),
    ("2026-10-28T01:14:33+00:00", None, False),
    ("2027-01-01T23:00:00+00:00", "missing", False),  # Future capture cannot prove completion.
])
def test_legacy_missing_completion_flag_requires_zoned_after_close_capture(fetched, flag, expected):
    prices = market(99)
    for row in prices["SMH"]:
        if flag == "missing":
            del row["session_complete"]
        else:
            row["session_complete"] = flag
        row["fetched_at"] = fetched
    result = context([event("boj_hold", .01)], [observation("USDJPY", 150, 145)], prices)
    japan = result["regional_transmission"][0]
    assert japan["transmission_confirmed"] is expected
    assert result["policy"]["initial_cap_pct"] == (1 if expected else 2)


def test_semiconductor_completion_proof_rejects_an_unzoned_analysis_clock():
    with pytest.raises(ValueError, match="timezone"):
        _semiconductor_transmission(market(), DAY, now=NOW.replace(tzinfo=None))
