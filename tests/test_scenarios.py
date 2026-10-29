from datetime import datetime, timedelta, timezone

import pytest

from twiz.config import load_prediction_markets_config
from twiz.scenarios import build_scenario_context

NOW = datetime(2026, 10, 28, 2, tzinfo=timezone.utc)
DAY = "2026-10-27"


def event(driver, yes):
    registered = next(row for row in load_prediction_markets_config()["markets"]
                      if row.get("driver") == driver)
    return {"driver": driver, "label": driver, "question": registered["expected_question"],
            "id": registered["id"], "source": "Polymarket Gamma",
            "yes_probability": yes, "no_probability": 1-yes, "status": "uncertain",
            "market_updated_at": (NOW-timedelta(hours=1)).isoformat(),
            "fetched_at": NOW.isoformat(), "end_date": (NOW+timedelta(days=1)).isoformat(),
            "liquidity_usd": 10000, "spread": 0.02}


def macro(series, start, end):
    return {"series_id": series, "source": "FRED", "date": DAY, "value": end,
            "history": [{"date": f"2026-10-{22+i}", "value": start+(end-start)*i/5}
                        for i in range(6)]}


def test_probability_threshold_changes_real_action_with_identical_prices():
    rows = [macro("DGS10", 5.2, 4.99), macro("BAMLH0A0HYM2", 3.4, 3.3)]
    events = [event("fed_hold", .96), event("near_term_ceasefire", .90)]
    bullish = build_scenario_context(DAY, rows, events, {}, now=NOW)
    events[1] = event("near_term_ceasefire", .10)
    bearish = build_scenario_context(DAY, rows, events, {}, now=NOW)
    assert bullish["state"] == "repair_bias"
    assert bullish["policy"]["initial_cap_pct"] == 2
    assert bearish["state"] == "stress_bias"
    assert bearish["policy"]["initial_cap_pct"] == 0
    assert all(branch["probability"] is None for branch in bullish["branches"])


def test_exact_65_is_main_case_but_unknown_quote_is_not_no():
    rows = [event("fed_hold", .65), event("near_term_ceasefire", .35)]
    result = build_scenario_context(DAY, [], rows, {}, now=NOW)
    assert [e["main_case"] for e in result["events"]] == ["Yes", "No"]
    rows[0]["market_updated_at"] = (NOW-timedelta(days=2)).isoformat()
    rows[1]["liquidity_usd"] = 1
    result = build_scenario_context(DAY, [], rows, {}, now=NOW)
    assert result["events"] == [] and result["state"] == "unknown"
    assert result["policy"]["initial_cap_pct"] == 0


def test_high_probability_does_not_override_observed_transmission_stress():
    rows = [macro("DGS10", 4.9, 5.2), macro("BAMLH0A0HYM2", 3.1, 3.5)]
    result = build_scenario_context(DAY, rows, [event("fed_hold", .96),
        event("near_term_ceasefire", .90)], {}, now=NOW)
    assert result["state"] == "stress_bias"
    assert result["policy"]["initial_cap_pct"] == 0


def test_missing_drift_or_unverified_author_numbers_do_not_become_observed_stress():
    rows = [{"series_id": "DGS10", "source": "FRED", "date": DAY, "value": 4.99},
            {"series_id": "BAMLH0A0HYM2", "source": "author", "date": DAY, "value": 15}]
    result = build_scenario_context(DAY, rows, [event("fed_hold", .96),
        event("near_term_ceasefire", .90)], {}, now=NOW)
    assert result["state"] != "stress_bias"
    assert result["macro_features"]["credit"] is None


def test_future_macro_and_history_mismatch_are_not_backfilled():
    rows = [macro("DGS10", 5, 4.9)]
    rows[0]["history"].append({"date": "2026-10-28", "value": 10})
    result = build_scenario_context(DAY, rows, [], {}, now=NOW)
    assert result["macro_features"]["yield10"]["value"] == 4.9
    result = build_scenario_context("2026-10-20", rows,
                                    [event("fed_hold", .96)], {}, now=NOW)
    assert not result["events"]


@pytest.mark.parametrize("macro_kind,event_kind,expected", [
    ("repair", "support", ("repair_bias", 20, 50)),
    ("stress", "support", ("stress_bias", 0, 20)),
    ("none", "support", ("defensive", 10, 35)),
    ("yield_only", "support", ("defensive", 20, 35)),
    ("repair", "missing", ("repair_bias", 10, 35)),
    ("none", "missing", ("unknown", 0, 20)),
])
def test_scenario_budgets_scale_from_configured_ceiling_and_legacy_defaults_match(
        macro_kind, event_kind, expected):
    rows = ([macro("DGS10", 5.2, 4.9), macro("BAMLH0A0HYM2", 3.4, 3.3)]
            if macro_kind == "repair" else [macro("DGS10", 4.9, 5.2),
                macro("BAMLH0A0HYM2", 3.1, 3.5)] if macro_kind == "stress"
            else [macro("DGS10", 5.2, 4.9)] if macro_kind == "yield_only" else [])
    events = [event("fed_hold", .96), event("near_term_ceasefire", .9)] if (
        event_kind == "support") else []
    current = build_scenario_context(DAY, rows, events, {}, now=NOW,
                                     allocation_policy={"max_new_capital_pct": 100})
    assert (current["state"], current["policy"]["initial_cap_pct"],
            current["policy"]["post_event_cap_pct"]) == expected
    assert current["policy"]["max_new_capital_pct"] == 100
    assert "总额不超过100%" in current["branches"][1]["action"]
    legacy = build_scenario_context(DAY, rows, events, {}, now=NOW)
    assert legacy["policy"]["max_new_capital_pct"] == 10
    for key in ("initial_cap_pct", "post_event_cap_pct", "before_event_cap_pct"):
        assert current["policy"][key] == pytest.approx(legacy["policy"][key] * 10)


def test_declared_fraction_overrides_change_policy_without_changing_market_state():
    rows = [macro("DGS10", 5.2, 4.9), macro("BAMLH0A0HYM2", 3.4, 3.3)]
    events = [event("fed_hold", .96)]
    policy = {"max_new_capital_pct": 40, "scenario_budget_fractions": {
        "repair_strong": {"initial": .3, "post_event": .8}, "before_event": .1}}
    result = build_scenario_context(DAY, rows, events, {}, now=NOW, allocation_policy=policy)
    assert result["state"] == "repair_bias"
    assert result["policy"]["initial_cap_pct"] == 12
    assert result["policy"]["post_event_cap_pct"] == 32
    assert result["policy"]["before_event_cap_pct"] == 4
    defensive = build_scenario_context(DAY, [], events, {}, now=NOW, allocation_policy={
        "max_new_capital_pct": 40,
        "scenario_budget_fractions": {"defensive": {"initial": .3, "post_event": .6}}})
    assert defensive["policy"]["initial_cap_pct"] == 12
    assert defensive["policy"]["post_event_cap_pct"] == 24


@pytest.mark.parametrize("override", [
    {"max_new_capital_pct": 101}, {"max_new_capital_pct": 0},
    {"scenario_budget_fractions": {"unknown_field": .2}},
    {"scenario_budget_fractions": {"before_event": float("nan")}},
    {"scenario_budget_fractions": {"before_event": 1.01}},
    {"scenario_budget_fractions": {"repair_strong": {"initial": .8, "post_event": .2}}},
    {"scenario_budget_fractions": {"repair_strong": {"initial": True, "post_event": 1}}},
    {"scenario_budget_fractions": {"regional": {"initial": .1}}},
])
def test_invalid_declared_fractions_rejected_before_event_observations(override):
    class UnreadableEvent(dict):
        def get(self, key, default=None):
            raise AssertionError("Invalid funding policy must fail before reading events")

    with pytest.raises(ValueError):
        build_scenario_context(DAY, [], [UnreadableEvent()], {}, now=NOW,
                                allocation_policy=override)
