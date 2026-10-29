"""A single campaign cannot accidentally multiply account exposure across a watchlist."""
import json
import math
from copy import deepcopy

import pytest

from twiz.allocation import build_action_plan
from twiz.config import load_allocation_config

POLICY = {"primary_ticker": "QQQ", "max_new_capital_pct": 10,
          "entry_fractions": [0.4, 0.35, 0.25], "trim_fractions": [0.4, 0.3, 0.3],
          "unknown_macro_initial_cap_pct": 2, "max_campaign_loss_pct": 0.5,
          "min_entry_spacing_pct": 0.75}
DATE = "2026-10-27"


def inputs():
    watchlist = [{"ticker": ticker, "close": 100, "data_source": "yfinance", "data_date": DATE,
                  "stale": False, "close_source": "regular_market_metadata"}
                 for ticker in ("QQQ", "SPY", "NVDA", "SMH")]
    tech = {"QQQ": {"indicators": {"close": 100, "sma20": 105, "sma50": 110}, "levels": [
        {"kind": "support", "price": p, "sources": ["pivot"]}
        for p in [99.9, 99.5, 99.0, 98.5, 98.0, 97.0, 96.0]
    ] + [{"kind": "resistance", "price": p} for p in [101, 103, 106.5, 110, 120]]}}
    return watchlist, tech


def plan(state="risk_on", policy=None, watchlist=None, technical=None):
    rows, tech = inputs()
    return build_action_plan(DATE, watchlist or rows, technical or tech,
                             {"state": state}, policy or POLICY)


def test_one_campaign_total_account_budget_and_no_immediate_fill():
    result = plan()
    assert result["status"] == "conditional_plan" and result["buy_now_pct"] == 0
    assert [entry["price"] for entry in result["entries"]] == [99.5, 98.5, 97]
    assert [entry["account_pct"] for entry in result["entries"]] == [4, 3.5, 2.5]
    assert sum(entry["account_pct"] for entry in result["entries"]) <= 10
    assert all(row["new_allocation_pct"] == 0 for row in result["other_assets"])
    assert {row["ticker"] for row in result["other_assets"]} == {"SPY", "NVDA", "SMH"}
    assert not result["leverage"] and "日收盘" in result["entries"][0]["trigger"]
    assert result["price_date"] == DATE
    assert result["model_inputs"]["close_source"] == "regular_market_metadata"


def test_unknown_macro_enables_only_conditional_two_percent_and_reserves_eight():
    result = plan("unknown")
    assert [entry["currently_enabled_pct"] for entry in result["entries"]] == [2, 0, 0]
    assert [entry["account_pct"] for entry in result["entries"]] == [4, 3.5, 2.5]
    assert result["budget"]["currently_enabled_pct"] == 2
    assert result["budget"]["uncommitted_cash_pct"] == 8
    assert "不是账户实际现金" in result["budget_scope"]
    assert result["buy_now_pct"] == 0 and "暂不启用" in result["current_action"]


def test_macro_caution_halves_first_batch_and_riskoff_blocks_all_new_buys():
    caution = plan("caution")
    assert [entry["currently_enabled_pct"] for entry in caution["entries"]] == [2, 3.5, 2.5]
    blocked = plan("risk_off")
    assert blocked["status"] == "blocked" and blocked["buy_now_pct"] == 0
    assert not any(entry["currently_enabled_pct"] for entry in blocked["entries"])
    assert blocked["budget"]["currently_enabled_pct"] == 0


def test_risk_model_scales_all_batches_and_uses_cash_weighted_share_cost():
    policy = {**POLICY, "max_campaign_loss_pct": 0.05}
    result = plan(policy=policy)
    cash = [entry["account_pct"] for entry in result["entries"]]
    prices = [entry["price"] for entry in result["entries"]]
    stop = result["stop"]["price"]
    expected_loss = sum(pct*(1-stop/entry) for pct, entry in zip(cash, prices))
    assert result["stop"]["modeled_loss_pct"] == pytest.approx(expected_loss)
    assert 0 < expected_loss <= 0.05
    assert all(pct < base for pct, base in zip(cash, [4, 3.5, 2.5]))
    weighted_cost = sum(cash)/sum(pct/price for pct, price in zip(cash, prices))
    assert result["budget"]["estimated_weighted_entry"] == pytest.approx(weighted_cost)
    assert "跳空" in "".join(result["limitations"])
    assert "实际损失超过模型" in "".join(result["limitations"])


def test_profit_ladder_is_actual_distinct_resistance_with_original_share_trims():
    result = plan()
    entry, stop = result["entries"][0]["price"], result["stop"]["price"]
    assert [stage["price"] for stage in result["exits"]] == [103, 106.5, 110]
    assert [stage["trim_original_dip_pct"] for stage in result["exits"]] == [40, 30, 30]
    assert sum(stage["trim_original_dip_pct"] for stage in result["exits"]) == 100
    for multiple, stage in enumerate(result["exits"], 1):
        assert stage["price"] >= entry+multiple*(entry-stop)
        assert stage["reference_return_pct"] == pytest.approx((stage["price"]/entry-1)*100)
    assert "非剩余股数或核心仓" in result["trim_basis"]
    assert "取消" in "".join(result["limitations"])
    assert "实际成交成本尚未知" in result["entry_reference"]
    assert "日收盘跌破96.00" in result["stop"]["trigger"]
    assert "取消所有未成交加仓" in result["stop"]["trigger"]


def test_missing_higher_targets_remain_unavailable_instead_of_extrapolated():
    rows, technical = inputs()
    technical["QQQ"]["levels"] = [level for level in technical["QQQ"]["levels"]
                                   if level["kind"] != "resistance" or level["price"] == 103]
    result = plan(watchlist=rows, technical=technical)
    assert result["status"] == "conditional_plan"
    assert [stage["price"] for stage in result["exits"]] == [103, None, None]
    assert result["exits"][1]["reference_return_pct"] is None


@pytest.mark.parametrize("case", ["no_stop", "no_resistance", "crowded_supports"])
def test_incomplete_structure_cannot_create_a_numeric_action_plan(case):
    rows, technical = inputs()
    if case == "no_stop":
        technical["QQQ"]["levels"] = [{"kind": "support", "price": p} for p in [99, 97, 95]]
    elif case == "no_resistance":
        technical["QQQ"]["levels"] = [level for level in technical["QQQ"]["levels"]
                                       if level["kind"] != "resistance"]
    else:
        technical["QQQ"]["levels"] = [{"kind": "support", "price": p}
                                      for p in [99.5, 99.4, 99.3, 99.2]]
    result = plan(watchlist=rows, technical=technical)
    assert result["status"] == "unavailable" and result["entries"] == [] and result["exits"] == []
    assert result["stop"]["price"] is None and result["limitations"]


@pytest.mark.parametrize("override", [
    {"data_source": "sample"}, {"data_source": "unverified"}, {"stale": True},
    {"data_date": "2026-10-20"}, {"data_date": "2026-10-28"},
    {"close": math.nan}, {"close": -1}, {"close": 101},
])
def test_untrusted_stale_or_mismatched_data_disables_actions(override):
    rows, technical = inputs()
    rows[0].update(override)
    result = plan(watchlist=rows, technical=technical)
    assert result["status"] == "unavailable" and not result["entries"]
    assert result["stop"]["price"] is None
    assert result["buy_now_pct"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field,value", [("sma50", None), ("sma20", math.inf), ("close", 99)])
def test_incomplete_indicators_cannot_supply_action_prices(field, value):
    rows, technical = inputs()
    technical["QQQ"]["indicators"][field] = value
    assert plan(watchlist=rows, technical=technical)["status"] == "unavailable"


@pytest.mark.parametrize("policy_change", [
    {"max_new_capital_pct": 101}, {"entry_fractions": [0.5, 0.5, 0.5]},
    {"entry_fractions": [0.5, None, 0.5]}, {"trim_fractions": [0.3, 0.3, 0.3]},
    {"max_campaign_loss_pct": 0}, {"unknown_macro_initial_cap_pct": -1},
    {"min_entry_spacing_pct": 0},
])
def test_invalid_policy_does_not_allocate_account_funds(policy_change):
    result = plan(policy={**deepcopy(POLICY), **policy_change})
    assert result["status"] == "unavailable" and result["buy_now_pct"] == 0


def test_old_session_is_rejected_without_trusting_callers_stale_flag(monkeypatch):
    monkeypatch.setattr("twiz.allocation.expected_complete_session", lambda _: DATE)
    rows, technical = inputs()
    rows[0].update(data_date="2026-10-26", stale=False)
    result = plan(watchlist=rows, technical=technical)
    assert result["status"] == "unavailable" and not result["entries"]
    assert result["stop"]["price"] is None


def test_current_config_allows_full_unlevered_cap_without_recommending_immediate_buy(tmp_path):
    configured = load_allocation_config()
    assert configured["max_new_capital_pct"] == configured["max_campaign_loss_pct"] == 100
    assert configured["unknown_macro_initial_cap_pct"] == configured["provisional_cap_pct"] == 2
    missing = load_allocation_config(tmp_path / "not-created.yml")
    assert missing["max_new_capital_pct"] == missing["max_campaign_loss_pct"] == 100
    result = plan(policy=configured)
    assert result["status"] == "conditional_plan" and result["buy_now_pct"] == 0
    assert [r["account_pct"] for r in result["entries"]] == [40, 35, 25]
    assert sum(r["currently_enabled_pct"] for r in result["entries"]) == 100
    assert result["budget"]["planned_capital_pct"] == 100
    assert .5 < result["stop"]["modeled_loss_pct"] < 100
    assert all(r["new_allocation_pct"] == 0 for r in result["other_assets"])


def test_explicit_unknown_macro_limit_is_configurable_within_shared_cap():
    result = plan("unknown", policy={**POLICY, "max_new_capital_pct": 100,
                                     "max_campaign_loss_pct": 100,
                                     "unknown_macro_initial_cap_pct": 15})
    assert result["status"] == "conditional_plan"
    assert [row["currently_enabled_pct"] for row in result["entries"]] == [15, 0, 0]
    assert result["budget"]["uncommitted_cash_pct"] == 85
    assert plan("unknown", policy={**POLICY, "unknown_macro_initial_cap_pct": 11})[
        "status"] == "unavailable"


def test_decimal_fractions_never_exceed_full_account_from_float_roundoff():
    result = plan(policy={**POLICY, "max_new_capital_pct": 100, "max_campaign_loss_pct": 100,
                          "entry_fractions": [.004, .322, .674]})
    assert sum(r["account_pct"] for r in result["entries"]) <= 100
    assert result["budget"]["planned_capital_pct"] <= 100


@pytest.mark.parametrize("change", [
    {"max_new_capital_pct": 101}, {"max_campaign_loss_pct": 101},
    {"max_new_capital_pct": 0}, {"max_campaign_loss_pct": 0},
    {"max_new_capital_pct": float("inf")}, {"max_campaign_loss_pct": True},
    {"max_new_capital_pct": 10, "unknown_macro_initial_cap_pct": 11},
])
def test_loader_rejects_invalid_current_capital_policy(tmp_path, change):
    import yaml

    path = tmp_path / "allocation.yml"
    path.write_text(yaml.safe_dump(change))
    with pytest.raises(ValueError, match="Allocation parameter"):
        load_allocation_config(path)
