"""Dynamic price regions require dated evidence and a later confirmation sequence."""

import json
import math
from copy import deepcopy
from datetime import date, timedelta
from statistics import mean

import pytest

from twiz.sessions import is_us_equity_session
from twiz.structure import (
    _clusters,
    _pivots,
    _weekly,
    _zone_signal,
    build_dynamic_action_plan,
    build_market_structure,
    validate_structure_parameters,
)

DATE = "2026-10-27"


def bars(n=510):
    days, day = [], date.fromisoformat(DATE)
    while len(days) < n:
        if is_us_equity_session(day):
            days.append(day.isoformat())
        day -= timedelta(days=1)
    rows = []
    for i, stamp in enumerate(reversed(days)):
        level = 65 if i < 140 else 80 if i < 280 else 100 if i < n - 75 else 120
        close = level + 2 * math.sin(i / 6)
        rows.append(
            {
                "date": stamp,
                "open": close - 0.2,
                "close": close,
                "low": close - 1,
                "high": close + 1,
                "volume": 1000,
                "source": "yfinance",
                "session_complete": True,
                "price_basis": "test_adjusted",
            }
        )
    return rows


def structure(rows=None):
    return build_market_structure(DATE, {"QQQ": rows or bars(), "SPY": rows or bars()})


def action_structure(confirmed=False):
    zone = {
        "lower": 90,
        "upper": 100,
        "center": 95,
        "credible": True,
        "strength": "corroborated",
        "origins": [
            {"family": "price_structure", "price": 92, "date": "2026-04-01"},
            {"family": "moving_average", "price": 94, "date": "2026-10-20"},
        ],
        "families": ["price_structure", "moving_average"],
    }
    item = {
        "status": "ready",
        "price_date": DATE,
        "close": 104.5 if confirmed else 120,
        "data_source": "yfinance",
        "session_complete": True,
        "features": {"atr14": 4, "price_basis": "test_adjusted",
                     "confirmation_bars": [{"date": DATE, "low": 99, "high": 105, "close": 104.5}]
                     if confirmed else []},
        "zones": {"support": None, "deep": None, "extreme": zone},
        "recovery_levels": [
            {
                "price": p,
                "kind": "pivot_high",
                "source": "confirmed_price_pivot",
                "date": "2026-10-01",
            }
            for p in (112, 124, 136, 150)
        ],
        "signal": {
            "state": "confirmed_retest" if confirmed else "wait",
            "zone": "extreme",
            "reclaim_date": "2026-10-26" if confirmed else None,
            "retest_date": DATE if confirmed else None,
            "swing_low": 89 if confirmed else None,
        },
    }
    return {"date": DATE, "by_ticker": {"QQQ": item, "SPY": deepcopy(item)}}


def test_daily_weekly_and_atr_features_are_derived_from_completed_data():
    rows = bars()
    result = structure(rows)["by_ticker"]["QQQ"]
    assert result["status"] == "ready" and result["price_date"] == DATE
    assert result["features"]["sma60"] == pytest.approx(mean(r["close"] for r in rows[-60:]))
    assert all(result["features"][f"sma{n}"] is not None for n in (20, 50, 60, 200))
    assert all(result["features"][f"weekly_sma{n}"] is not None for n in (20, 30, 50))
    assert result["features"]["zone_anchor_date"] == rows[-6]["date"]
    assert result["features"]["atr14"] > 0 and result["signal"]["probability"] is None
    json.dumps(result, allow_nan=False)


def test_future_bars_cannot_change_current_regions_projection_or_signals():
    rows = bars()
    future = {
        **rows[-1],
        "date": "2026-10-28",
        "close": 1000,
        "open": 1000,
        "low": 999,
        "high": 1001,
    }
    assert structure(rows) == structure(rows + [future])


def test_parameter_snapshot_drives_actual_zone_boundaries_and_confirmation_cutoff():
    rows = bars()
    default = build_market_structure(DATE, {"QQQ": rows})["by_ticker"]["QQQ"]
    narrow = build_market_structure(DATE, {"QQQ": rows}, parameters={
        "zone_buffer_atr": 0.1, "confirmation_sessions": 10, "atr_period": 20,
        "range_sessions": 42})["by_ticker"]["QQQ"]
    assert narrow["features"]["parameters"]["zone_buffer_atr"] == 0.1
    assert narrow["features"]["zone_anchor_date"] == rows[-11]["date"]
    assert len(narrow["features"]["confirmation_bars"]) == 10
    assert narrow["features"]["atr_method"].startswith("20日")
    assert narrow["features"]["range_sessions"] == 42
    assert "recent_range63" not in narrow["features"]
    assert narrow["zones"] != default["zones"]
    assert "10日" in narrow["evidence"][0]
    assert validate_structure_parameters()["zone_buffer_atr"] == 0.25


def test_direct_yahoo_chart_source_retains_its_trace_and_is_not_relabeled_yfinance():
    rows = [{**row, "source": "Yahoo chart", "raw_body_file": "raw/QQQ-0.body",
             "raw_json_pointer": "/chart/result/0/indicators/quote/0",
             "raw_timestamp_pointer": f"/chart/result/0/timestamp/{index}",
             "raw_array_index": index, "fetched_at": "2026-10-28T14:00:00+00:00"}
            for index, row in enumerate(bars())]
    item = build_market_structure(DATE, {"QQQ": rows})["by_ticker"]["QQQ"]
    assert item["status"] == "ready" and item["data_source"] == "Yahoo chart"
    del rows[-1]["raw_body_file"]
    result = build_market_structure(DATE, {"QQQ": rows})["by_ticker"]["QQQ"]
    assert result["status"] == "unavailable"


@pytest.mark.parametrize("params", [{"unknown": 1}, {"atr_period": True},
                                     {"range_sessions": 63.5}, {"cluster_atr": float("nan")},
                                     {"zone_buffer_atr": -1}, {"pivot_right": 0}])
def test_invalid_parameters_are_rejected_not_silently_ignored(params):
    with pytest.raises(ValueError, match="parameter"):
        build_market_structure(DATE, {}, parameters=params)


def test_custom_pivot_right_side_changes_when_pivot_is_knowable():
    rows = [{"date": f"2026-10-{i+1:02}", "low": 99, "high": 101} for i in range(10)]
    rows[4]["low"] = 95
    assert not _pivots(rows[:7], left=2, right=3)
    point = _pivots(rows[:7], left=2, right=2)[0]
    assert point["confirmed_at"] == rows[6]["date"]


def test_custom_trim_and_price_parameters_change_plan_without_moving_cap():
    item = action_structure()
    policy = {"entry_offsets_atr": [1.5, 0.75, 0], "stop_buffer_atr": 0.5,
              "trim_fractions": [0.5, 0.3, 0.2], "target_r_multiples": [1, 1.5, 2]}
    plan = build_dynamic_action_plan(item, {"state": "risk_on"}, policy)
    assert [row["price"] for row in plan["entries"]] == [106, 103, 100]
    assert [row["trim_original_dip_pct"] for row in plan["exits"]] == [50, 30, 20]
    assert plan["stop"]["price"] == 88
    assert plan["budget"]["planned_capital_pct"] <= 10
    assert plan["stop"]["modeled_loss_pct"] <= 0.5
    assert "50/30/20" in plan["trim_basis"]
    assert "1.5/0.75/0ATR" in plan["entry_reference"]
    assert "1/1.5/2R" in plan["spacing_assessment"]["target_note"]


def test_invalid_trim_cannot_silently_use_old_40_30_30():
    plan = build_dynamic_action_plan(action_structure(), {}, {"trim_fractions": [0.5, 0.5, 0.5]})
    assert plan["entries"] == [] and plan["buy_now_pct"] == 0
    assert "100%" in plan["limitations"][-1]


def test_price_scale_changes_all_regions_instead_of_reusing_historical_author_numbers():
    original = bars()
    scaled = [
        {**row, **{key: row[key] * 3 for key in ("open", "high", "low", "close")}}
        for row in original
    ]
    a, b = structure(original)["by_ticker"]["QQQ"], structure(scaled)["by_ticker"]["QQQ"]
    assert b["structural_floor"] == pytest.approx(a["structural_floor"] * 3)
    for tier in ("support", "deep", "extreme"):
        if a["zones"][tier]:
            assert b["zones"][tier]["lower"] == pytest.approx(a["zones"][tier]["lower"] * 3)
    assert b["projection"]["price"] == pytest.approx(a["projection"]["price"] * 3)


def test_nonoverlapping_tiers_and_explicit_origin_families():
    result = structure()["by_ticker"]["QQQ"]
    zones = [
        result["zones"][tier] for tier in ("support", "deep", "extreme") if result["zones"][tier]
    ]
    assert all(upper["lower"] > lower["upper"] for upper, lower in zip(zones, zones[1:]))
    assert all(
        origin["date"] <= result["features"]["zone_anchor_date"]
        for z in zones
        for origin in z["origins"]
    )
    assert all("projection" not in z["independent_families"] for z in zones)


def test_multiple_ma_windows_are_one_family_and_projection_is_not_independent_proof():
    origins = [
        {"family": "moving_average", "label": name, "price": price, "date": DATE}
        for name, price in [("sma200", 100), ("weekly20", 100.1), ("weekly50", 100.2)]
    ]
    origins.append({"family": "projection", "label": "projected", "price": 100.3, "date": DATE})
    cluster = _clusters(origins, 2, 500)[0]
    assert cluster["independent_families"] == ["moving_average"]
    assert not cluster["credible"]
    origins.append(
        {
            "family": "price_structure",
            "label": "prior_traded_weekly_close",
            "price": 100.4,
            "date": "2026-04-01",
        }
    )
    assert _clusters(origins, 2, 500)[0]["credible"]


def test_pivots_only_become_available_after_their_right_hand_bars():
    rows = bars(12)
    for row in rows:
        row.update(open=100, close=100, high=101, low=99)
    rows[5]["low"] = 95
    assert not any(p["date"] == rows[5]["date"] for p in _pivots(rows[:8]))
    pivot = next(p for p in _pivots(rows[:9]) if p["date"] == rows[5]["date"])
    assert pivot["confirmed_at"] == rows[8]["date"]


def test_only_completed_week_closes_include_the_good_friday_holiday_week():
    rows = [{"date": "2026-04-02", "close": 100}, {"date": "2026-04-09", "close": 101}]
    assert _weekly(rows, "2026-04-09") == [{"date": "2026-04-02", "close": 100}]


@pytest.mark.parametrize(
    "change",
    [
        {"source": "sample"},
        {"source": "unverified"},
        {"session_complete": False},
        {"close": float("nan")},
        {"volume": -1},
        {"low": -1},
    ],
)
def test_invalid_latest_bar_cannot_create_current_structure(change):
    rows = bars()
    rows[-1].update(change)
    assert structure(rows)["by_ticker"]["QQQ"]["status"] == "unavailable"


def test_duplicate_current_dates_staleness_short_history_and_mixed_basis_are_explicit():
    rows = bars()
    assert structure(rows + [dict(rows[-1])])["by_ticker"]["QQQ"]["status"] == "unavailable"
    assert structure(rows[:-1])["by_ticker"]["QQQ"]["status"] == "unavailable"
    assert structure(bars(150))["by_ticker"]["QQQ"]["status"] == "insufficient_history"
    rows[-1]["price_basis"] = "different_basis"
    assert structure(rows)["by_ticker"]["QQQ"]["status"] == "unavailable"


def test_negative_projection_is_not_emitted_as_a_price_target():
    rows = bars()
    for row in rows:
        row.update(open=100, close=100, low=99, high=101)
    rows[-30]["high"] = 500
    result = structure(rows)["by_ticker"]["QQQ"]
    assert result["projection"]["status"] == "unavailable" and result["projection"]["price"] is None
    assert result["zones"]["extreme"] is None


def test_touch_and_sweep_are_not_buys_until_a_later_retest_holds():
    zone = {"lower": 95, "upper": 100}
    touch = [{"date": "2026-10-26", "low": 97, "close": 99}]
    assert not _zone_signal(zone, touch, 4, "extreme")["buy_signal"]
    sweep = [{"date": "2026-10-26", "low": 94, "close": 101}]
    assert _zone_signal(zone, sweep, 4, "extreme")["state"] == "sweep_reclaim"
    assert not _zone_signal(zone, sweep, 4, "extreme")["buy_signal"]
    confirmed = _zone_signal(zone, sweep + [{"date": DATE, "low": 98, "close": 102}], 4, "extreme")
    assert confirmed["state"] == "confirmed_retest" and confirmed["retest_date"] == DATE


def test_wait_action_keeps_conditional_shared_budget_but_no_executed_or_immediate_buy():
    plan = build_dynamic_action_plan(action_structure(), {"state": "unknown"}, {})
    assert plan["status"] == "conditional_plan" and plan["buy_now_pct"] == 0
    assert 0 < plan["budget"]["currently_enabled_pct"] <= 2
    assert sum(row["account_pct"] for row in plan["entries"]) <= 10
    assert plan["stop"]["modeled_loss_pct"] <= 0.5
    assert all(row["new_allocation_pct"] == 0 for row in plan["other_assets"])
    assert [row["trim_original_dip_pct"] for row in plan["exits"]] == [40, 30, 30]


def test_current_confirmed_retest_can_create_signal_budget_without_claiming_a_fill():
    plan = build_dynamic_action_plan(
        action_structure(True), {"state": "unknown"}, {"initial_cap_pct": 1}
    )
    assert 0 < plan["buy_now_pct"] <= 1
    assert "不代表已成交" in plan["current_action"]
    assert plan["entries"][0]["price"] == 104  # Engineering watch reference.
    assert plan["entries"][0]["modeled_entry_cost"] == 104.5  # Completed signal price.
    assert plan["stop"]["modeled_loss_pct"] <= 0.5


def test_configurable_full_capital_ceiling_accepts_confirmed_structure_without_legacy_risk_veto():
    data = action_structure(True)
    data["by_ticker"]["QQQ"]["recovery_levels"].append({
        "price": 200, "source": "confirmed_price_pivot", "date": "2026-10-01"})
    policy = {"max_new_capital_pct": 100, "max_campaign_loss_pct": 100,
              "unknown_macro_initial_cap_pct": 15}
    plan = build_dynamic_action_plan(data, {"state": "risk_on"}, policy)
    assert plan["status"] == "conditional_plan"
    assert [row["account_pct"] for row in plan["entries"]] == [40, 35, 25]
    assert plan["budget"]["planned_capital_pct"] == 100
    assert plan["buy_now_pct"] == plan["budget"]["currently_enabled_pct"] == 40
    assert .5 < plan["stop"]["modeled_loss_pct"] < 100
    assert all(row["new_allocation_pct"] == 0 for row in plan["other_assets"])
    unknown = build_dynamic_action_plan(data, {"state": "unknown"}, policy)
    assert unknown["buy_now_pct"] == 15
    legacy = build_dynamic_action_plan(data, {"state": "risk_on"}, {
        "max_new_capital_pct": 10, "max_campaign_loss_pct": .5})
    assert legacy["budget"]["planned_capital_pct"] <= 10
    assert legacy["stop"]["modeled_loss_pct"] <= .5
    decimals = build_dynamic_action_plan(data, {"state": "risk_on"}, {
        **policy, "entry_fractions": [.004, .322, .674]})
    assert sum(row["account_pct"] for row in decimals["entries"]) <= 100
    assert decimals["budget"]["planned_capital_pct"] <= 100


@pytest.mark.parametrize("change", [
    {"max_new_capital_pct": 101}, {"max_campaign_loss_pct": 101},
    {"max_new_capital_pct": 0}, {"max_campaign_loss_pct": 0},
    {"max_new_capital_pct": float("nan")}, {"max_campaign_loss_pct": True},
    {"unknown_macro_initial_cap_pct": 101},
])
def test_invalid_configurable_limits_never_allocate_more_than_one_account(change):
    plan = build_dynamic_action_plan(action_structure(True), {"state": "risk_on"}, {
        "max_new_capital_pct": 100, "max_campaign_loss_pct": 100, **change})
    assert plan["status"] == "unavailable" and plan["entries"] == []
    assert plan["buy_now_pct"] == 0


def test_pre_event_and_provisional_limits_are_configured_within_account_ceiling():
    data = action_structure(True)
    data["by_ticker"]["QQQ"]["recovery_levels"].append({
        "price": 200, "source": "confirmed_price_pivot", "date": "2026-10-01"})
    policy = {"max_new_capital_pct": 100, "max_campaign_loss_pct": 100,
              "before_event_cap_pct": 20, "initial_cap_pct": 30,
              "event_main_case_confirmed": True, "provisional_cap_pct": 15}
    events = [{"date": "2026-10-28", "importance": 5, "precision": "scheduled", "name": "FOMC"}]
    plan = build_dynamic_action_plan(data, {"state": "risk_on"}, policy, events)
    assert plan["buy_now_pct"] == 20
    assert plan["event_context"]["pre_event_probe_enabled"]
    assert plan["event_context"]["before_event_cap_pct"] == 20
    data["by_ticker"]["QQQ"]["zones"]["extreme"]["strength"] = "working_hypothesis"
    provisional = build_dynamic_action_plan(data, {"state": "risk_on"}, policy)
    assert provisional["budget"]["planned_capital_pct"] == 15
    assert "配置的15%" in "".join(provisional["limitations"])


def test_macro_event_scenario_and_chasing_guards_cannot_expand_position_authority():
    confirmed = action_structure(True)
    blocked = build_dynamic_action_plan(confirmed, {"state": "risk_off"}, {})
    assert blocked["buy_now_pct"] == blocked["budget"]["currently_enabled_pct"] == 0
    event = {"date": "2026-10-28", "importance": 5, "name": "FOMC", "precision": "scheduled"}
    assert (
        build_dynamic_action_plan(confirmed, {"state": "risk_on"}, {}, [event])["buy_now_pct"] == 0
    )
    assert (
        build_dynamic_action_plan(confirmed, {"state": "risk_on"}, {"initial_cap_pct": 0})[
            "buy_now_pct"
        ]
        == 0
    )
    confirmed["by_ticker"]["QQQ"]["close"] = 110
    assert build_dynamic_action_plan(confirmed, {"state": "risk_on"}, {})["buy_now_pct"] == 0


def test_working_hypothesis_or_missing_higher_targets_does_not_allocate_a_full_ten_percent():
    data = action_structure()
    data["by_ticker"]["QQQ"]["zones"]["extreme"]["strength"] = "working_hypothesis"
    assert (
        build_dynamic_action_plan(data, {"state": "risk_on"}, {})["budget"]["planned_capital_pct"]
        <= 2
    )
    data = action_structure()
    data["by_ticker"]["QQQ"]["recovery_levels"] = [data["by_ticker"]["QQQ"]["recovery_levels"][1]]
    plan = build_dynamic_action_plan(data, {"state": "risk_on"}, {})
    assert plan["budget"]["planned_capital_pct"] <= 2
    assert [r["price"] for r in plan["exits"]] == [124, None, None]


def test_missing_extreme_keeps_clear_wait_direction_and_visible_references():
    data = action_structure()
    data["by_ticker"]["QQQ"]["zones"]["extreme"] = None
    plan = build_dynamic_action_plan(data, {"state": "risk_on"}, {})
    assert plan["status"] == "waiting_structure" and "不追价" in plan["current_action"]
    assert plan["buy_now_pct"] == 0


def test_stale_action_input_cannot_reenable_a_budget_and_probabilities_never_become_stock_odds():
    data = action_structure(True)
    data["by_ticker"]["QQQ"]["price_date"] = "2026-10-26"
    assert build_dynamic_action_plan(data, {"state": "risk_on"}, {})["buy_now_pct"] == 0
    result = structure()
    assert result["risk_context"]["probability"] is None
    assert all(row["signal"]["probability"] is None for row in result["by_ticker"].values())


def test_verified_event_main_case_allows_only_limited_confirmed_probe_before_event():
    event = {"date": "2026-10-28", "importance": 5, "name": "FOMC", "precision": "scheduled"}
    policy = {"event_main_case_confirmed": True, "before_event_cap_pct": 2, "initial_cap_pct": 1}
    confirmed = action_structure(True)
    plan = build_dynamic_action_plan(confirmed, {"state": "unknown"}, policy, [event])
    assert 0 < plan["buy_now_pct"] <= 1
    assert plan["event_context"]["pre_event_probe_enabled"]
    assert all(row["currently_enabled_pct"] == 0 for row in plan["entries"][1:])
    assert (
        build_dynamic_action_plan(action_structure(), {"state": "unknown"}, policy, [event])[
            "buy_now_pct"
        ]
        == 0
    )
    assert (
        build_dynamic_action_plan(confirmed, {"state": "risk_off"}, policy, [event])["buy_now_pct"]
        == 0
    )
    policy["event_main_case_confirmed"] = False
    assert (
        build_dynamic_action_plan(confirmed, {"state": "unknown"}, policy, [event])["buy_now_pct"]
        == 0
    )


def test_dip_prices_descend_and_later_batches_require_independent_confirmation():
    plan = build_dynamic_action_plan(action_structure(True), {"state": "risk_on"}, {})
    prices = [row["price"] for row in plan["entries"]]
    assert prices[0] > prices[1] > prices[2]
    for row in plan["entries"][1:]:
        assert row["price_type"] in {"atr_zone_approach_watch", "zone_upper_watch"}
        assert row["currently_enabled_pct"] == 0 and "新报告" in row["trigger"]
        assert "不是预先下单" in row["trigger"]
        assert "收盘守住或收复本档" in row["trigger"]
    assert plan["stop"]["modeled_loss_pct"] <= 0.5


def test_future_recovery_price_and_same_bar_reclaim_retest_cannot_create_a_buy():
    data = action_structure(True)
    data["by_ticker"]["QQQ"]["signal"]["reclaim_date"] = DATE
    assert build_dynamic_action_plan(data, {"state": "risk_on"}, {})["buy_now_pct"] == 0
    data = action_structure()
    for target in data["by_ticker"]["QQQ"]["recovery_levels"]:
        target["date"] = "2026-10-28"
    assert (
        build_dynamic_action_plan(data, {"state": "risk_on"}, {})["budget"]["planned_capital_pct"]
        == 0
    )


def support_policy_structure(confirmed=False):
    data = action_structure()
    item = data["by_ticker"]["QQQ"]
    item["zones"]["support"] = {
        "lower": 110,
        "upper": 116,
        "center": 113,
        "credible": True,
        "strength": "corroborated",
        "origins": [
            {"family": "price_structure", "price": 112, "date": "2026-09-01"},
            {"family": "moving_average", "price": 114, "date": "2026-10-20"},
        ],
        "families": ["price_structure", "moving_average"],
    }
    item["signals"] = {
        "support": {
            "state": "confirmed_retest" if confirmed else "wait",
            "zone": "support",
            "probability": None,
            "reclaim_date": "2026-10-26" if confirmed else None,
            "retest_date": DATE if confirmed else None,
            "swing_low": 110.5 if confirmed else None,
        },
        "extreme": item["signal"],
    }
    if confirmed:
        item["close"] = 120.5
        item["features"]["confirmation_bars"] = [
            {"date": DATE, "low": 115.5, "high": 121, "close": 120.5}]
    return data


def test_policy_changes_selected_price_region_and_budget_without_changing_zones():
    data = support_policy_structure()
    original_zones = deepcopy(data["by_ticker"]["QQQ"]["zones"])
    repair = build_dynamic_action_plan(
        data,
        {"state": "risk_on"},
        {
            "preferred_entry_tier": "support",
            "initial_cap_pct": 1,
        },
    )
    adverse = build_dynamic_action_plan(
        data,
        {"state": "caution"},
        {
            "preferred_entry_tier": "extreme",
            "initial_cap_pct": 0,
        },
    )
    assert repair["selected_tier"] == "support" and repair["entries"][0]["price"] == 120
    assert repair["budget"]["planned_capital_pct"] <= 2
    assert adverse["selected_tier"] == "extreme" and adverse["entries"][0]["price"] == 104
    assert adverse["budget"]["currently_enabled_pct"] == 0
    assert repair["buy_now_pct"] == adverse["buy_now_pct"] == 0
    assert data["by_ticker"]["QQQ"]["zones"] == original_zones
    assert repair["signal"]["probability"] is None


def test_selected_support_uses_its_own_confirmation_instead_of_extreme_signal():
    data = support_policy_structure(confirmed=True)
    assert data["by_ticker"]["QQQ"]["signal"]["state"] == "wait"
    result = build_dynamic_action_plan(
        data,
        {"state": "risk_on"},
        {
            "preferred_entry_tier": "support",
            "initial_cap_pct": 1,
        },
    )
    assert result["signal"]["zone"] == "support" and result["buy_now_pct"] > 0
    assert result["budget"]["planned_capital_pct"] <= 2
    data["by_ticker"]["QQQ"]["signals"].pop("support")
    assert (
        build_dynamic_action_plan(
            data,
            {"state": "risk_on"},
            {
                "preferred_entry_tier": "support",
                "initial_cap_pct": 1,
            },
        )["buy_now_pct"]
        == 0
    )



def test_targets_clear_spot_highest_cost_and_volatility_spacing_then_stay_unassigned():
    data = action_structure()
    plan = build_dynamic_action_plan(data, {"state": "risk_on"}, {})
    close = data["by_ticker"]["QQQ"]["close"]
    target_floor = max(close, max(row["price"] for row in plan["entries"]))+2
    targets = [row["price"] for row in plan["exits"] if row["price"] is not None]
    assert all(price > target_floor for price in targets)
    assert all(b-a >= 2 for a, b in zip(targets, targets[1:]))
    assert all(row["reward_risk_multiple"] >= i
               for i, row in enumerate(plan["exits"], 1) if row["price"])
    data["by_ticker"]["QQQ"]["close"] = 160
    stale_targets = build_dynamic_action_plan(data, {"state": "risk_on"}, {})
    assert stale_targets["entries"] == stale_targets["exits"] == []
    assert stale_targets["budget"]["planned_capital_pct"] == 0


def test_current_confirmed_cost_is_used_for_loss_not_lower_watch_reference():
    data = action_structure(True)
    plan = build_dynamic_action_plan(data, {"state": "unknown"}, {})
    first = plan["entries"][0]
    assert first["modeled_entry_cost"] == 104.5 and first["price"] == 104
    loss = sum(row["account_pct"]*(1-plan["stop"]["price"]/row["modeled_entry_cost"])
               for row in plan["entries"])
    assert plan["stop"]["modeled_loss_pct"] == pytest.approx(loss)
    assert loss <= .5
