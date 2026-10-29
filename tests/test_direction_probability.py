"""Causal probability clocks and honest baseline fallback, on synthetic prices."""

import copy
import math
from datetime import date, timedelta

import pytest

from twiz.direction_probability import (
    _auxiliary,
    _fit_analog,
    _predict,
    _select_analog,
    estimate_direction,
)
from twiz.sessions import is_us_equity_session


@pytest.fixture(scope="module")
def prices():
    rows, day = [], date(2018, 1, 2)
    while len(rows) < 1000:
        if is_us_equity_session(day):
            i = len(rows)
            close = 100+i*.025+5*math.sin(i/13)+2*math.sin(i/31)
            rows.append({"date": day.isoformat(), "open": close-.2*math.sin(i/3),
                         "close": close, "high": close+1, "low": close-1,
                         "volume": 10000, "session_complete": True,
                         "source": "yfinance", "price_basis": "synthetic_consistent"})
        day += timedelta(days=1)
    return rows


@pytest.fixture(scope="module")
def estimate(prices):
    return estimate_direction(prices, prices[-1]["date"])


def test_complementary_bounded_estimate_has_explicit_horizon_and_validation(estimate, prices):
    assert estimate["status"] == "available"
    assert .02 <= estimate["probability_up"] <= .98
    assert estimate["probability_up"]+estimate["probability_down"] == pytest.approx(1)
    assert estimate["horizon_sessions"] == 20
    assert estimate["training_label_last"] <= prices[-6]["date"]
    assert estimate["validation"]["samples"] >= 20
    assert estimate["used_price_prefix_sha256"]
    assert estimate["excluded_inputs"] == ["unavailable_historical_option_chain"]


def test_failed_conditional_fit_is_named_baseline_instead_of_a_new_direction_call(estimate):
    if estimate["model_kind"] == "historical_base_rate":
        assert estimate["calibration_shrinkage"] == 0
        assert not estimate["conditional_model_beats_prior"]
        assert not any(r["active"] for r in estimate["contributions"])
        assert estimate["validation"]["brier"] == estimate["validation"]["prior_brier"]
    else:
        assert estimate["calibration_shrinkage"] > 0
        assert estimate["validation"]["brier"] <= estimate["validation"]["prior_brier"]+1e-10


class Future(dict):
    def get(self, key, default=None):
        if key not in {"date", "available_at"}:
            raise AssertionError("Future numeric payload inspected: "+key)
        return super().get(key, default)


def test_future_prices_and_event_facts_do_not_change_forecast(prices, estimate):
    future = Future(date="2099-12-31", available_at="2099-12-31T21:15:00+00:00")
    result = estimate_direction(prices+[future], prices[-1]["date"],
                                capex_rows=[future], event_rows=[future], macro_rows=[future])
    assert result == estimate


def test_auxiliary_missing_future_and_stale_values_do_not_become_observations():
    day = date(2026, 10, 28)
    future = Future(date="2026-10-29", available_at="2026-10-29T20:15:00+00:00")
    stale = {"available_at": "2025-01-02T21:15:00+00:00",
             "aggregate": {"status": "available", "median_cash_capex_yoy_pct": 900},
             "values": {"yield10_pct": 100}, "direction": 1, "source_refs": ["old"]}
    x = _auxiliary(day, [stale, future], [stale, future], [stale, future])
    assert x[2:] == [None]*7


def test_event_impact_is_separate_feature_not_the_stock_probability():
    row = {"available_at": "2026-10-27T20:15:00+00:00", "direction": .8,
           "id": "one-event",
           "source_refs": [{"url": "https://example.test/dated-event"}]}
    assert _auxiliary(date(2026, 10, 28), [], [row], [])[-1] == .8


def test_correlated_daily_market_quotes_are_not_added_as_independent_events():
    common = {"correlation_family": "monetary_policy", "source_refs": ["dated-source"]}
    rows = [{**common, "available_at": "2026-10-26T20:15:00+00:00", "direction": .4},
            {**common, "available_at": "2026-10-27T20:15:00+00:00", "direction": .2}]
    assert _auxiliary(date(2026, 10, 28), [], rows, [])[-1] == .2


def test_conflicting_same_clock_family_quotes_are_neutral_regardless_of_order():
    common = {"correlation_family": "monetary_policy", "source_refs": ["dated-source"],
              "available_at": "2026-10-27T20:15:00+00:00"}
    rows = [{**common, "direction": .5}, {**common, "direction": -.5}]
    assert _auxiliary(date(2026, 10, 28), [], rows, [])[-1] is None
    assert _auxiliary(date(2026, 10, 28), [], list(reversed(rows)), [])[-1] is None


def test_daily_snapshot_does_not_refresh_old_cash_capex_filing():
    issuer = {"status": "available", "cash_capex_yoy_pct": 100.,
              "current": {kind: {"available_at_proxy": "2020-01-02T21:15:00+00:00"}
                          for kind in ("capex", "revenue")}}
    row = {"available_at": "2026-10-28T20:15:00+00:00", "issuers": {"A": issuer, "B": issuer},
           "aggregate": {"status": "available", "included_issuers": ["A", "B"],
                         "median_cash_capex_yoy_pct": 100.}}
    assert _auxiliary(date(2026, 10, 28), [row], [], [])[2] is None


def test_uniform_up_labels_do_not_claim_conditional_advantage_from_unclipped_probabilities(prices):
    rows = copy.deepcopy(prices)
    for i, row in enumerate(rows):
        row.update(open=100+i, close=100.5+i, high=101+i, low=99+i)
    result = estimate_direction(rows, rows[-1]["date"])
    assert result["model_kind"] == "historical_base_rate"
    assert not result["conditional_model_beats_prior"]
    assert result["validation"]["brier"] == pytest.approx(result["validation"]["prior_brier"])


def test_missing_auxiliary_dimensions_have_no_distance_authority():
    records = [{"x": [float(i % 7)]*11+[None]*9, "label": i % 2} for i in range(100)]
    price = _fit_analog(records, 11, 20)
    combined = _fit_analog(records, 20, 20)
    query = [{"x": [2.5]*11+[None]*9}]
    assert _predict(price, query) == pytest.approx(_predict(combined, query))


def test_analog_hyperparameters_do_not_use_validation_labels():
    records = [{"date": str(i), "x": [float(i % 7)]*11, "label": i % 2}
               for i in range(150)]
    first = records[120:]
    opposite = [{**r, "label": 1-r["label"]} for r in first]
    one, _ = _select_analog(records[:90], records[90:120], first, 11)
    two, _ = _select_analog(records[:90], records[90:120], opposite, 11)
    assert one == two


@pytest.mark.parametrize("mutation", ["gap", "incomplete", "basis", "stale"])
def test_bad_price_context_is_unavailable(prices, mutation):
    rows = copy.deepcopy(prices)
    if mutation == "gap":
        rows.pop(400)
    elif mutation == "incomplete":
        rows[-1]["session_complete"] = False
    elif mutation == "basis":
        rows[-1]["price_basis"] = "unknown"
    else:
        rows.pop()
    assert estimate_direction(rows, prices[-1]["date"])["status"] == "unavailable"


def test_model_does_not_mutate_original_source(prices):
    original = copy.deepcopy(prices)
    estimate_direction(prices, prices[-1]["date"])
    assert prices == original


def test_wrong_target_horizon_rejected(prices):
    with pytest.raises(ValueError, match="20-session"):
        estimate_direction(prices, prices[-1]["date"], horizon_sessions=5)
