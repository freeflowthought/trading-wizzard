from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from twiz.ingest import prediction_discovery as discovery
from twiz.ingest.prediction_markets import fetch_prediction_markets
from twiz.scenarios import build_scenario_context

NOW = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
DAY = "2026-10-28"
CONFIG = {"main_case_threshold": .65, "discovery": {"enabled": True, "active_pages": 1,
          "search_queries": [], "max_events": 6}, "markets": [{"id": "999"}]}


def event(identity, title, question, probability=.8, **overrides):
    payload = {"id": str(identity), "question": question, "active": True, "closed": False,
               "archived": False, "outcomes": ["Yes", "No"],
               "outcomePrices": [probability, 1-probability],
               "updatedAt": (NOW-timedelta(minutes=5)).isoformat(),
               "endDate": (NOW+timedelta(days=5)).isoformat(),
               "description": "Exact settlement conditions based on official announcements.",
               "liquidityNum": 20000, "spread": .02}
    payload.update(overrides)
    return {"id": str(identity), "title": title, "slug": "event-"+str(identity),
            "markets": [payload]}


def collect(monkeypatch, events, config=None):
    calls = []
    def endpoint(path, params, timeout):
        calls.append((path, params, timeout))
        return events
    monkeypatch.setattr(discovery, "_fetch_json", endpoint)
    audit = {}
    rows, warnings = fetch_prediction_markets(config or CONFIG, DAY, live=True, now=NOW,
                                              discovery_audit=audit)
    return rows, warnings, audit, calls


def test_live_discovery_chooses_new_ids_by_relevance_not_fixed_watchlist(monkeypatch):
    events = [event(10, "Sports", "Will a basketball team win?", .99),
              event(20, "US-Iran ceasefire", "Will the US-Iran ceasefire continue?", .9),
              event(30, "Fed decision", "Will there be no change in Fed interest rates?", .7)]
    rows, _, audit, calls = collect(monkeypatch, events)
    assert [row["id"] for row in rows] == ["30", "20"]
    assert rows[0]["yes_probability"] == .7  # A higher probability does not set importance.
    assert "999" not in audit["selected_market_ids"]
    assert any(r["market_id"] == "10" and r["decision"] == "excluded"
               for r in audit["candidates"])
    assert audit["requests"][0]["response"] == events and len(calls) == 1


@pytest.mark.parametrize("overrides", [{"liquidityNum": 10}, {"spread": .3},
    {"endDate": (NOW-timedelta(seconds=1)).isoformat()}, {"closed": True},
    {"updatedAt": (NOW-timedelta(days=2)).isoformat()}, {"description": ""}])
def test_important_but_unusable_events_remain_in_rejection_audit(monkeypatch, overrides):
    rows, warnings, audit, _ = collect(monkeypatch, [event(
        30, "Fed decision", "Will there be no change in Fed interest rates?", **overrides)])
    assert rows == [] and warnings
    assert audit["candidates"][0]["decision"] == "excluded"
    assert audit["candidates"][0]["reason"]


def test_main_case_and_action_do_not_require_any_particular_iran_contract(monkeypatch):
    rows, _, _, _ = collect(monkeypatch, [event(
        444, "Fed decision", "Will there be no change in Fed interest rates?", .8)])
    history = [{"series_id": series, "source": "FRED", "history": [
        {"date": f"2026-10-{23+i}", "value": first+(last-first)*i/5}
        for i in range(6)]} for series, first, last in (
            ("DGS10", 5.1, 4.9), ("BAMLH0A0HYM2", 3.4, 3.3))]
    result = build_scenario_context(DAY, history, rows, {}, now=NOW, contracts=[])
    assert result["state"] == "repair_bias"
    assert result["policy"]["initial_cap_pct"] == 2
    assert result["event_selection"]["legacy_required_iran_contract"] is False
    changed = deepcopy(rows)
    changed[0]["driver"] = "geopolitical_risk"
    result = build_scenario_context(DAY, history, changed, {}, now=NOW, contracts=[])
    assert result["events"] == []  # Reclassification prevents changing a cached driver.


def test_same_event_keeps_dominant_yes_outcome_not_high_no_on_tiny_alternative(monkeypatch):
    group = event(1, "Fed decision", "Will the Fed cut interest rates?", .01)
    alternative = event(2, "Fed decision", "Will the Fed increase interest rates?", .75)
    group["markets"] += alternative["markets"]
    rows, _, _, _ = collect(monkeypatch, [group])
    assert rows[0]["id"] == "2" and rows[0]["driver"] == "fed_hike"
    assert rows[0]["discovery"]["eligible_sibling_market_ids"] == ["1", "2"]


def test_new_dynamic_risk_event_can_change_budget_without_static_registry(monkeypatch):
    rows, _, _, _ = collect(monkeypatch, [event(
        555, "US-Iran conflict", "Will the US attack Iran before November?", .8)])
    result = build_scenario_context(DAY, [], rows, {}, now=NOW, contracts=[])
    assert result["state"] == "stress_bias"
    assert result["policy"]["initial_cap_pct"] == 0
    assert result["events"][0]["id"] == "555"


def test_current_discovery_never_runs_to_fill_historical_probability(monkeypatch):
    monkeypatch.setattr(discovery, "_fetch_json", lambda *args: pytest.fail("Unexpected fetch"))
    rows, warnings = fetch_prediction_markets(CONFIG, "2026-10-27", live=True, now=NOW)
    assert not rows and any("historical" in warning for warning in warnings)


@pytest.mark.parametrize("question,expected", [
    ("Will the Fed NOT raise interest rates?", "observe_only"),
    ("Will the Fed increase its balance sheet?", "observe_only"),
    ("Will Israel lift the Gaza blockade?", "observe_only"),
    ("Will the Bank of Korea NOT hike by 25 bps?", "observe_only"),
    ("Will the Israel-Iran ceasefire be cancelled?", "observe_only"),
    ("Will the Fed cancel its planned interest rate hike?", "observe_only"),
    ("Will the Fed pause interest rate cuts?", "observe_only"),
    ("Will the Fed rule out an interest rate increase?", "observe_only"),
    ("Will Israel halt its blockade by November?", "observe_only"),
    ("Will Israel cancel its attack on Iran?", "observe_only"),
    ("Will a Fed member dissent in favor of cutting interest rates?", "observe_only"),
    ("Will the Fed hold or hike interest rates?", "observe_only"),
    ("Will the Fed cut interest rates if inflation rises?", "observe_only"),
])
def test_ambiguous_or_negated_questions_never_get_automatic_direction(question, expected):
    classified = discovery.classify_market(question)
    assert classified is not None and classified["driver"] == expected


def test_partial_search_results_cannot_delete_a_sibling_risk_outcome(monkeypatch):
    group = event(1, "Fed decision", "Will the Fed cut interest rates?", .1)
    group["markets"] += event(2, "Fed decision", "Will the Fed hike interest rates?", .9)["markets"]
    partial = deepcopy(group)
    partial["markets"] = partial["markets"][:1]
    responses = iter([[group], {"events": [partial]}])
    monkeypatch.setattr(discovery, "_fetch_json", lambda *args: next(responses))
    config = deepcopy(CONFIG)
    config["discovery"]["search_queries"] = ["Fed"]
    rows, _ = fetch_prediction_markets(config, DAY, live=True, now=NOW)
    assert rows[0]["id"] == "2" and rows[0]["driver"] == "fed_hike"


@pytest.mark.parametrize("mutation", [{"id": "not-a-market"},
    {"question": "Will the Fed hike interest rates by100bps in November2027?"},
    {"yes_probability": .999, "no_probability": .001}])
def test_saved_identity_and_quote_cannot_be_changed_without_original_binding(monkeypatch, mutation):
    rows, _, _, _ = collect(monkeypatch, [event(
        5, "Fed decision", "Will the Fed hike interest rates?", .8)])
    assert discovery.discovery_contract(rows[0]) is not None
    rows[0].update(mutation)
    assert discovery.discovery_contract(rows[0]) is None
    result = build_scenario_context(DAY, [], rows, {}, now=NOW, contracts=[])
    assert not result["events"] and result["policy"]["initial_cap_pct"] == 0


def test_representative_compares_outcomes_only_at_same_near_analysis_deadline(monkeypatch):
    group = event(1, "Fed decision", "Will the Fed cut interest rates?", .1)
    near = event(2, "Fed decision", "Will the Fed hike interest rates?", .8)["markets"][0]
    distant = event(3, "Fed decision", "Will there be no change in Fed interest rates?", .95,
                    endDate=(NOW+timedelta(days=60)).isoformat())["markets"][0]
    group["markets"] += [near, distant]
    rows, _, _, _ = collect(monkeypatch, [group])
    assert rows[0]["id"] == "2" and rows[0]["discovery"]["days_to_end"] == 5


def test_core_event_channels_do_not_crowd_out_oil_geopolitics(monkeypatch):
    groups = [event(i, "Fed decision", "Will there be no change in Fed interest rates?", .7)
              for i in range(1, 5)]
    groups += [event(9, "US-Iran conflict", "Will the US attack Iran?", .8)]
    rows, _, _, _ = collect(monkeypatch, groups)
    assert [r["id"] for r in rows] == ["1", "9"]


@pytest.mark.parametrize("value", ["invalid", [], ["invalid"], 123])
def test_malformed_saved_discovery_becomes_unknown_without_crashing(value):
    assert discovery.discovery_contract({"selection_mode": discovery.DISCOVERY_VERSION,
                                         "discovery": value}) is None
