"""Shared exposure follows explicit account policy rather than an obsolete fixed profile."""

import pytest

from twiz.allocation import build_action_plan


def campaign(policy_change=None, state="unknown", deep=True):
    row = {"ticker": "QQQ", "close": 100, "data_source": "yfinance",
           "data_date": "2026-10-27", "stale": False}
    watchlist = [row] + [{**row, "ticker": f"T{i}"} for i in range(100)]
    technical = {"QQQ": {
        "indicators": {"close": 100, "sma20": 110, "sma50": 105},
        "levels": ([{"kind": "support", "price": price}
                    for price in ([95, 85, 75, 65] if deep else [99.5, 98.5, 97, 96])]
                   + [{"kind": "resistance", "price": price}
                      for price in ([130, 160, 190] if deep else [103, 106.5, 110])]),
    }}
    policy = {"primary_ticker": "QQQ", "max_new_capital_pct": 10,
              "entry_fractions": [0.4, 0.35, 0.25], "trim_fractions": [0.4, 0.3, 0.3],
              "unknown_macro_initial_cap_pct": 2, "max_campaign_loss_pct": 0.5,
              "min_entry_spacing_pct": 0.75, **(policy_change or {})}
    return build_action_plan("2026-10-27", watchlist, technical, {"state": state}, policy)


def test_large_watchlist_does_not_multiply_funded_campaigns_or_stop_risk():
    plan = campaign()
    assert plan["status"] == "conditional_plan"
    assert len(plan["other_assets"]) == 100
    assert all(asset["new_allocation_pct"] == 0 for asset in plan["other_assets"])
    assert sum(entry["account_pct"] for entry in plan["entries"]) <= 10
    assert plan["budget"]["currently_enabled_pct"] <= 2
    stop = plan["stop"]["price"]
    assert stop < min(entry["price"] for entry in plan["entries"])
    modeled_risk = sum(entry["account_pct"] * (1 - stop / entry["price"])
                       for entry in plan["entries"])
    assert modeled_risk == pytest.approx(plan["stop"]["modeled_loss_pct"])
    assert modeled_risk <= 0.5
    assert sum(exit["trim_original_dip_pct"] for exit in plan["exits"]) == 100
    assert [exit["price"] for exit in plan["exits"]] == [130, 160, 190]


@pytest.mark.parametrize("change,deep", [
    ({"max_campaign_loss_pct": 5}, True), ({"unknown_macro_initial_cap_pct": 5}, False),
])
def test_explicit_limits_can_exceed_legacy_defaults_without_exceeding_shared_cap(change, deep):
    plan = campaign(change, deep=deep)
    assert plan["status"] == "conditional_plan"
    assert sum(row["account_pct"] for row in plan["entries"]) <= 10
    if "max_campaign_loss_pct" in change:
        assert .5 < plan["stop"]["modeled_loss_pct"] <= change["max_campaign_loss_pct"]
    else:
        assert 2 < plan["budget"]["currently_enabled_pct"] <= change[
            "unknown_macro_initial_cap_pct"]
