"""Independent guards on scenario identity, timing, and shared capital policy."""

import math
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone

import pytest

from twiz.config import load_prediction_markets_config
from twiz.scenarios import build_scenario_context
from twiz.sessions import is_us_equity_session
from twiz.structure import build_dynamic_action_plan, build_market_structure

DAY = "2026-10-27"
NOW = datetime(2026, 10, 28, 2, tzinfo=timezone.utc)


def event(driver, yes):
    registered = next(row for row in load_prediction_markets_config()["markets"]
                      if row.get("driver") == driver)
    return {"id": registered["id"], "driver": driver, "label": registered["label"],
            "question": registered["expected_question"], "source": "Polymarket Gamma",
            "yes_probability": yes, "no_probability": 1-yes, "status": "uncertain",
            "market_updated_at": (NOW-timedelta(hours=1)).isoformat(),
            "fetched_at": NOW.isoformat(), "end_date": (NOW+timedelta(days=1)).isoformat(),
            "liquidity_usd": 10000, "spread": 0.02}


@pytest.mark.parametrize("change", [
    {"source": "unverified"}, {"id": "wrong-id"}, {"driver": "fed_hold"},
    {"question": "Will Bitcoin exceed $200,000 by December?"},
    {"market_updated_at": (NOW-timedelta(days=2)).isoformat()},
    {"fetched_at": (NOW+timedelta(seconds=1)).isoformat()},
    {"fetched_at": (NOW-timedelta(days=1)).isoformat()},
    {"end_date": NOW.isoformat()}, {"status": "unavailable"},
    {"liquidity_usd": 999}, {"spread": .11}, {"yes_probability": True},
])
def test_wrong_or_unusable_contract_cannot_become_a_geopolitical_no(change):
    bad = {**event("near_term_ceasefire", .05), **change}
    result = build_scenario_context(DAY, [], [bad], {}, now=NOW)
    assert result["events"] == []
    assert result["state"] == "unknown"
    assert result["policy"]["initial_cap_pct"] == 0
    assert not any("No 95" in value for value in result["causal_chain"])


def test_unzoned_scenario_clock_is_rejected_explicitly():
    with pytest.raises(ValueError, match="timezone"):
        build_scenario_context(DAY, [], [event("fed_hold", .96)], {},
                               now=NOW.replace(tzinfo=None))


def structure_fixture():
    zones = {tier: {"lower": lower, "upper": lower+4, "center": lower+2,
                    "credible": True, "strength": "corroborated",
                    "origins": [{"family": "price_structure", "price": lower+2,
                                 "date": "2026-10-01"}]}
             for tier, lower in (("support", 90.0), ("deep", 80.0), ("extreme", 70.0))}
    signals = {tier: {"state": "wait", "zone": tier, "buy_signal": False}
               for tier in zones}
    signals["support"] = {"state": "confirmed_retest", "zone": "support",
                          "reclaim_date": "2026-10-26", "retest_date": DAY,
                          "swing_low": 91.0, "buy_signal": True}
    return {"date": DAY, "by_ticker": {"QQQ": {
        "status": "ready", "price_date": DAY, "close": 97.0,
        "data_source": "yfinance", "session_complete": True,
        "features": {"atr14": 2.0, "price_basis": "yahoo_adjusted_history",
                     "confirmation_bars": [{"date": DAY, "low": 95, "high": 98, "close": 97}]},
        "zones": zones, "signals": signals, "signal": signals["support"],
        "recovery_levels": [{"price": price, "source": "confirmed_price_pivot",
                             "date": "2026-10-01", "confirmed_at": "2026-10-06"}
                            for price in (105, 115, 125)],
    }, "SPY": {"status": "unavailable"}}}


def test_same_price_event_flip_changes_region_budget_and_risk_off_stays_blocked():
    macro = [{"series_id": series, "source": "FRED", "date": DAY, "value": end,
              "history": [{"date": f"2026-10-{22+i}", "value": start+(end-start)*i/5}
                          for i in range(6)]}
             for series, start, end in (("DGS10", 5.2, 4.9),
                                        ("BAMLH0A0HYM2", 3.4, 3.3))]
    policies = []
    for probability in (.9, .1):
        scenario = build_scenario_context(DAY, macro, [event("fed_hold", .96),
            event("near_term_ceasefire", probability)], {}, now=NOW)
        policies.append({"primary_ticker": "QQQ", "entry_fractions": [.4, .35, .25],
                         "max_campaign_loss_pct": .5, **scenario["policy"]})
    imminent = [{"date": "2026-10-28", "importance": 5,
                 "name": "FOMC", "precision": "scheduled"}]
    plans = [build_dynamic_action_plan(structure_fixture(), {"state": "risk_on"}, policy,
                                       imminent)
             for policy in policies]
    assert plans[0]["selected_tier"] == "support"
    assert plans[0]["budget"]["planned_capital_pct"] == pytest.approx(2)
    assert plans[0]["buy_now_pct"] == plans[0]["budget"]["currently_enabled_pct"] == .8
    assert plans[0]["event_context"]["pre_event_probe_enabled"] is True
    assert plans[1]["selected_tier"] == "extreme"
    assert plans[1]["buy_now_pct"] == plans[1]["budget"]["currently_enabled_pct"] == 0
    assert plans[0]["entries"][0]["price"] == 96
    assert plans[0]["entries"][0]["modeled_entry_cost"] == 97
    assert plans[1]["entries"][0]["price"] == 76
    blocked = build_dynamic_action_plan(structure_fixture(), {"state": "risk_off"},
                                        policies[0], imminent)
    assert blocked["buy_now_pct"] == blocked["budget"]["currently_enabled_pct"] == 0
    for plan in plans:
        assert sum(entry["account_pct"] for entry in plan["entries"]) <= 10
        assert plan["stop"]["modeled_loss_pct"] <= .5
        assert plan["stop"]["price"] < min(entry["price"] for entry in plan["entries"])
        assert sum(exit["trim_original_dip_pct"] for exit in plan["exits"]) == 100
        assert all(row["new_allocation_pct"] == 0 for row in plan["other_assets"])
        prices = [entry["price"] for entry in plan["entries"]]
        assert prices[0] > prices[1] > prices[2]
        assert all(entry["currently_enabled_pct"] == 0 for entry in plan["entries"][1:])
        assert all("新报告" in entry["trigger"] and "独立确认" in entry["trigger"]
                   for entry in plan["entries"][1:])
        assert [entry["price_type"] for entry in plan["entries"]] == [
            "atr_zone_approach_watch", "atr_zone_approach_watch", "zone_upper_watch"]
        assert all("非独立历史枢轴" in entry["price_origin"]
                   for entry in plan["entries"][:2])
        modeled_costs = [entry["modeled_entry_cost"] for entry in plan["entries"]]
        assert plan["target_constraints"]["highest_entry_cost"] == max(modeled_costs)
        assert plan["stop"]["modeled_loss_pct"] == pytest.approx(sum(
            entry["account_pct"]*(1-plan["stop"]["price"]/entry["modeled_entry_cost"])
            for entry in plan["entries"]))


def history():
    days, cursor = [], date.fromisoformat(DAY)
    while len(days) < 320:
        if is_us_equity_session(cursor):
            days.append(cursor.isoformat())
        cursor -= timedelta(days=1)
    return [{"date": day, "source": "yfinance", "session_complete": True,
             "price_basis": "yahoo_adjusted_history", "volume": 1000,
             "open": 100+i*.1+4*math.sin(i/7), "close": 100+i*.1+4*math.sin(i/7),
             "high": 102+i*.1+4*math.sin(i/7), "low": 98+i*.1+4*math.sin(i/7)}
            for i, day in enumerate(reversed(days))]


def test_future_and_explicitly_incomplete_prices_do_not_rewrite_current_structure():
    rows = history()
    base = build_market_structure(DAY, {"QQQ": rows})
    adversarial = rows + [{**rows[-1], "date": "2026-10-28", "low": 1, "high": 999},
                          {**rows[-1], "session_complete": False, "low": 1, "high": 999}]
    assert build_market_structure(DAY, {"QQQ": adversarial}) == base
    incomplete = deepcopy(rows)
    incomplete[-1]["session_complete"] = False
    assert build_market_structure(DAY, {"QQQ": incomplete})["by_ticker"]["QQQ"][
        "status"] == "unavailable"


def test_dynamic_regions_scale_with_current_data_instead_of_author_historical_levels():
    rows = history()
    doubled = [{**row, **{key: row[key]*2 for key in ("open", "high", "low", "close")}}
               for row in rows]
    first = build_market_structure(DAY, {"QQQ": rows})["by_ticker"]["QQQ"]
    second = build_market_structure(DAY, {"QQQ": doubled})["by_ticker"]["QQQ"]
    assert first["status"] == second["status"] == "ready"
    assert second["structural_floor"] == pytest.approx(first["structural_floor"]*2)
    for tier, zone in first["zones"].items():
        if zone is not None:
            assert second["zones"][tier]["center"] == pytest.approx(zone["center"]*2)


def test_rally_through_old_target_requires_future_targets_above_quote_and_costs():
    data = structure_fixture()
    data["by_ticker"]["QQQ"]["close"] = 110
    plan = build_dynamic_action_plan(data, {"state": "unknown"},
                                     {"preferred_entry_tier": "support"})
    targets = [row["price"] for row in plan["exits"] if row["price"] is not None]
    assert targets
    cost_ceiling = max(110, *(row["price"] for row in plan["entries"]))
    assert targets[0] >= cost_ceiling + .5 * data["by_ticker"]["QQQ"]["features"]["atr14"]
    assert all(right-left >= .5 * data["by_ticker"]["QQQ"]["features"]["atr14"]
               for left, right in zip(targets, targets[1:]))
    observed = {row["price"] for row in data["by_ticker"]["QQQ"]["recovery_levels"]}
    assert set(targets) <= observed


def test_no_observed_target_above_current_quote_withholds_plan_instead_of_faking_higher_levels():
    data = structure_fixture()
    data["by_ticker"]["QQQ"]["close"] = 126
    plan = build_dynamic_action_plan(data, {"state": "unknown"},
                                     {"preferred_entry_tier": "support"})
    assert plan["entries"] == plan["exits"] == []
    assert plan["buy_now_pct"] == plan["budget"]["currently_enabled_pct"] == 0
    assert plan["budget"]["planned_capital_pct"] == 0


def test_nearly_identical_observed_targets_do_not_become_three_profit_stages():
    data = structure_fixture()
    item = data["by_ticker"]["QQQ"]
    origin = item["recovery_levels"][0]
    item["recovery_levels"] = [{**origin, "price": price}
                               for price in (150, 150.01, 150.02, 160, 170)]
    plan = build_dynamic_action_plan(data, {"state": "unknown"},
                                     {"preferred_entry_tier": "support"})
    targets = [row["price"] for row in plan["exits"] if row["price"] is not None]
    assert targets
    assert all(right-left >= .5*item["features"]["atr14"]
               for left, right in zip(targets, targets[1:]))


def test_dip_plan_cannot_average_down_after_its_stop_even_with_event_authority():
    data = structure_fixture()
    data["by_ticker"]["QQQ"]["close"] = 65
    plan = build_dynamic_action_plan(data, {"state": "risk_on"}, {
        "preferred_entry_tier": "extreme", "initial_cap_pct": 2,
        "event_main_case_confirmed": True, "before_event_cap_pct": 2,
    })
    assert plan["buy_now_pct"] == plan["budget"]["currently_enabled_pct"] == 0
    assert all(row["currently_enabled_pct"] == 0 for row in plan["entries"])
