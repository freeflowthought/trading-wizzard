"""Public prediction-market prices remain conditional beliefs with explicit data gates."""
import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from twiz.ingest import prediction_markets as pm

NOW = datetime(2026, 10, 28, 1, 0, tzinfo=timezone.utc)  # October 27 in Toronto.
REPORT_DATE = "2026-10-27"
CONFIG = {"markets": [{"id": "123", "label": "停火", "topic": "geopolitics",
                       "yes_scenario": "停火延续条件", "no_scenario": "冲突升级条件"}]}


def market(**overrides):
    payload = {"id": "123", "question": "Ceasefire continues through October 31?",
               "slug": "ceasefire-continues-october-31", "active": True, "closed": False,
               "archived": False, "outcomes": '["Yes", "No"]', "outcomePrices": '["0.7", "0.3"]',
               "updatedAt": "2026-10-28T00:30:00Z", "endDate": "2026-10-31T23:59:00Z",
               "liquidityNum": 5000, "spread": 0.02, "bestBid": 0.69, "bestAsk": 0.71}
    payload.update(overrides)
    return payload


def fetch(monkeypatch, payload=None, **kwargs):
    monkeypatch.setattr(pm, "_fetch_market", lambda market_id, timeout: payload or market())
    return pm.fetch_prediction_markets(CONFIG, REPORT_DATE, live=True, now=NOW, **kwargs)


def test_yes_no_labels_are_mapped_independently_of_order_and_input_encoding(monkeypatch):
    rows, warnings = fetch(monkeypatch, market(outcomes=["No", "Yes"], outcomePrices=[0.3, 0.7]))
    row = rows[0]
    assert not warnings and row["yes_probability"] == 0.7 and row["no_probability"] == 0.3
    assert row["status"] == "yes_leaning"
    assert row["fetched_at"] != row["market_updated_at"]
    assert row["yes_scenario"] == "停火延续条件"
    assert row["price_type"] == "outcomePrices_market_implied_belief"
    assert "not a verified event" in row["reason"]
    assert row["url"] == "https://polymarket.com/event/ceasefire-continues-october-31"


@pytest.mark.parametrize("yes,status", [(0.65, "yes_leaning"), (0.35, "no_leaning"),
                                       (0.50, "uncertain"), (0, "no_leaning"), (1, "yes_leaning")])
def test_belief_labels_are_not_yes_no_event_facts(monkeypatch, yes, status):
    rows, _ = fetch(monkeypatch, market(outcomePrices=[yes, 1-yes]))
    assert rows[0]["status"] == status and rows[0]["yes_probability"] == yes


@pytest.mark.parametrize("override", [
    {"closed": True}, {"active": False}, {"archived": True}, {"active": "true"},
    {"question": None}, {"spread": "nan"}, {"spread": -0.1},
    {"updatedAt": "2026-10-26T00:30:00Z"}, {"updatedAt": "2026-10-28T02:00:00Z"},
    {"updatedAt": "2026-10-28T00:30:00"}, {"updatedAt": None},
    {"endDate": "2026-10-27T23:59:00Z"}, {"endDate": "2026-10-31T23:59:00"},
    {"liquidityNum": 999}, {"liquidityNum": "nan", "liquidity": None},
    {"spread": 0.11}, {"spread": -0.1, "bestBid": None, "bestAsk": None},
    {"spread": None, "bestBid": None, "bestAsk": None},
    {"bestBid": 0.8, "bestAsk": 0.7}, {"bestBid": "nan"}, {"bestAsk": -0.1},
    {"outcomes": ["Up", "Down"]}, {"outcomes": ["Yes", "Yes"]},
    {"outcomePrices": [0.2, 0.2]}, {"outcomePrices": ["NaN", "0.3"]},
    {"outcomePrices": [-0.1, 1.1]}, {"outcomePrices": "not json"}, {"id": "456"},
])
def test_invalid_market_has_no_fabricated_probability(monkeypatch, override):
    rows, warnings = fetch(monkeypatch, market(**override))
    row = rows[0]
    assert row["status"] == "unavailable" and row["reason"] and warnings
    assert row["yes_probability"] is None and row["no_probability"] is None
    json.dumps(row, allow_nan=False)


def test_zero_one_sided_bid_is_valid_when_spread_is_available(monkeypatch):
    rows, warnings = fetch(monkeypatch, market(bestBid=0, bestAsk=None, spread=0.05))
    assert not warnings and rows[0]["yes_probability"] == 0.7


def test_quotes_can_supply_spread_but_wide_quotes_cannot_be_hidden_by_reported_spread(monkeypatch):
    rows, _ = fetch(monkeypatch, market(spread=None))
    assert rows[0]["status"] == "yes_leaning"
    rows, warnings = fetch(monkeypatch, market(bestBid=0.1, bestAsk=0.9, spread=0.01))
    assert warnings and rows[0]["status"] == "unavailable"


def test_local_date_guard_prevents_current_probabilities_in_historical_reports(monkeypatch):
    called = []
    monkeypatch.setattr(pm, "_fetch_market", lambda *args, **kwargs: called.append(args))
    rows, warnings = pm.fetch_prediction_markets(CONFIG, "2026-10-26", live=True, now=NOW)
    assert not rows and warnings and not called
    rows, warnings = pm.fetch_prediction_markets(CONFIG, REPORT_DATE, live=False, now=NOW)
    assert not rows and warnings and not called


def test_ten_request_cap_and_duplicate_ids_reuse_results(monkeypatch):
    calls = []
    def provider(market_id, timeout):
        calls.append((market_id, timeout))
        return market(id=market_id)
    monkeypatch.setattr(pm, "_fetch_market", provider)
    config = {"markets": [{"id": str(i)} for i in range(12)] + [{"id": "0"}]}
    rows, warnings = pm.fetch_prediction_markets(config, REPORT_DATE, live=True, now=NOW)
    assert len(calls) == 10 and all(0 < timeout <= 5 for _, timeout in calls)
    assert len(rows) == 13 and rows[10]["status"] == "unavailable"
    assert rows[-1]["yes_probability"] == 0.7 and warnings


def test_total_collection_budget_stops_further_requests(monkeypatch):
    elapsed, calls = [0.0], []
    monkeypatch.setattr(pm.time, "monotonic", lambda: elapsed[0])
    def provider(market_id, timeout):
        calls.append(timeout)
        elapsed[0] += timeout
        return market(id=market_id)
    monkeypatch.setattr(pm, "_fetch_market", provider)
    config = {"markets": [{"id": str(i)} for i in range(8)]}
    rows, warnings = pm.fetch_prediction_markets(config, REPORT_DATE, live=True, now=NOW)
    assert calls == [5, 5, 5, 5] and elapsed[0] == 20
    assert rows[4]["status"] == "unavailable" and warnings


def test_fetch_failure_is_row_level_and_ids_cannot_be_urls(monkeypatch):
    def failing(*args, **kwargs):
        raise TimeoutError("provider unavailable")
    monkeypatch.setattr(pm, "_fetch_market", failing)
    rows, warnings = pm.fetch_prediction_markets(CONFIG, REPORT_DATE, live=True, now=NOW)
    assert rows[0]["status"] == "unavailable" and "provider unavailable" in warnings[0]
    bad = deepcopy(CONFIG)
    bad["markets"][0]["id"] = "https://elsewhere.example/"
    rows, warnings = pm.fetch_prediction_markets(bad, REPORT_DATE, live=True, now=NOW)
    assert "numeric Gamma" in warnings[0]


def test_event_slug_is_from_provider_payload_and_unsafe_slug_is_not_a_link(monkeypatch):
    rows, _ = fetch(monkeypatch, market(events=[{"slug": "ceasefire-group"}]))
    assert rows[0]["url"] == "https://polymarket.com/event/ceasefire-group"
    rows, _ = fetch(monkeypatch, market(slug="../external?query=1"))
    assert rows[0]["url"] is None


def test_contract_rules_dates_tokens_and_raw_response_hash_are_preserved(monkeypatch):
    import hashlib

    payload = market(description="Settlement uses only qualifying US air/missile strikes.",
                     createdAt="2026-10-01T12:00:00Z", startDate="2026-10-01T13:00:00Z",
                     resolutionSource="https://example.test/resolution",
                     clobTokenIds='["111", "222"]')
    rows, _ = fetch(monkeypatch, payload)
    metadata = rows[0]["contract_metadata"]
    assert metadata["description"] == payload["description"]
    assert metadata["created_at"] == payload["createdAt"]
    assert metadata["start_date"] == payload["startDate"]
    assert metadata["end_date"] == payload["endDate"]
    assert metadata["outcomes"] == ["Yes", "No"]
    assert metadata["clob_token_ids"] == ["111", "222"]
    assert metadata["resolution_source"] == payload["resolutionSource"]
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False).encode()).hexdigest()
    assert metadata["raw_response_sha256"] == expected


def test_configurable_collection_label_and_semantic_scope_are_not_broad_ceasefire(monkeypatch):
    monkeypatch.setattr(pm, "_fetch_market", lambda *args, **kwargs: market())
    config = deepcopy(CONFIG)
    config["main_case_threshold"] = .8
    config["markets"][0]["settlement_scope_note"] = "只按美方符合条件的打击结算，不代表全面和平。"
    rows, _ = pm.fetch_prediction_markets(config, REPORT_DATE, live=True, now=NOW)
    assert rows[0]["status"] == "uncertain"
    assert rows[0]["yes_probability"] == .7
    assert "不代表全面和平" in rows[0]["settlement_scope_note"]
