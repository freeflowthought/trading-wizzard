"""Synthetic chronology checks; these never inspect the frozen real test outcomes."""

import json
import math
from copy import deepcopy
from dataclasses import replace

import pytest

from twiz.evaluation import (
    FEATURES,
    EvaluationProtocol,
    _apply_calibrator,
    _calendar,
    _calibrate,
    _features,
    _macro_at,
    _macro_index,
    _metrics,
    _outcome_label,
    _prices,
    _probability,
    _shunge_proxy,
    run_walk_forward_evaluation,
)


def research_fixture():
    days = _calendar("2023-01-03", "2025-01-31")
    market = {}
    for asset, phase in (("QQQ", 0), ("SPY", 1), ("AAA", 2), ("SMH", 3), ("IGV", 4)):
        rows = []
        for i, stamp in enumerate(days):
            close = 100*math.exp(i*.0004+.025*math.sin(i/7+phase)+.015*math.sin(i/19))
            rows.append({"date": stamp, "open": close, "high": close*1.01,
                         "low": close*.99, "close": close, "volume": 1000,
                         "source": "yfinance", "session_complete": True,
                         "price_basis": "synthetic_consistent_split_basis"})
        market[asset] = rows
    macro = [{"series_id": series, "source": "FRED public CSV",
              "history": [{"date": stamp, "value": level+.1*math.sin(i/17)}
                          for i, stamp in enumerate(days)]}
             for series, level in (("DGS10", 4.2), ("BAMLH0A0HYM2", 3.2),
                                   ("DCOILBRENTEU", 70))]
    protocol = EvaluationProtocol(
        evaluation_start=days[-41], evaluation_end=days[-1],
        primary_horizon_sessions=5, secondary_horizons_sessions=(1, 20),
        individual_tickers=("AAA",), window_sessions=126,
        pretest_validation_sessions=21, penalties=(.1, 1),
        stock_direction_thresholds=(.45, .5), bootstrap_resamples=40)
    return days, market, macro, protocol


@pytest.fixture(scope="module")
def evaluated():
    days, market, macro, protocol = research_fixture()
    return days, market, macro, protocol, run_walk_forward_evaluation(market, macro, protocol)


def test_protocol_config_is_immutable_and_preserves_the_frozen_grid():
    config = {"evaluation_start": "2026-04-27", "evaluation_end": "2026-10-27",
              "primary_tickers": ["QQQ", "SPY"], "training": {
                  "penalties": [.1, 1, 10], "stock_direction_thresholds": [.45, .5, .55, .6]},
              "scoring": {"bootstrap_seed": 42},
              "target": {"minimum_paired_gain_percentage_points": 3}}
    protocol = EvaluationProtocol.from_config(config)
    assert protocol.penalties == (.1, 1, 10) and protocol.bootstrap_seed == 42
    assert protocol.minimum_paired_gain_percentage_points == 3
    with pytest.raises(AttributeError):
        protocol.evaluation_start = "2020-01-01"
    with pytest.raises(ValueError):
        replace(protocol, embargo_sessions=-1).validate()
    for field, value in (("prediction_time", "before_US_open"),
                         ("label", "open_to_close_direction"), ("flat_return_label", "down")):
        with pytest.raises(ValueError, match="Unsupported"):
            EvaluationProtocol.from_config({**config, field: value})


def test_calibration_reuses_exact_standardization_and_clipping_when_extrapolating():
    probabilities = [.49+i*.02/99 for i in range(100)]
    fitted = _calibrate(probabilities, [i % 2 for i in range(100)])
    assert fitted["available"]
    probability = .9
    expected = _probability(fitted["transform_model"], [math.log(probability/(1-probability))])
    assert _apply_calibrator(probability, fitted) == expected
    assert expected < .7  # Omitting fitted clipping falsely produced about .958.


def test_explicit_macro_publication_time_overrides_observation_date_and_keeps_vintages():
    rows = [{"series_id": "DGS10", "source": "FRED public CSV", "history": [
        {"date": "2025-01-01", "value": 4, "available_at": "2025-01-02T15:00:00Z"},
        {"date": "2025-01-01", "value": 9, "available_at": "2025-06-01T15:00:00Z"},
        {"date": "2025-01-01", "value": 99},  # Unknown revision cannot bypass timed vintages.
        {"date": "2025-01-03", "value": 99, "available_at": "2025-01-03T15:00:00"},
    ]}]
    indexed = _macro_index(rows)
    assert _macro_at(indexed, "2025-01-01")["yield"] is None
    known = _macro_at(indexed, "2025-01-03")
    assert known["yield"] == 4
    assert known["yield_availability_basis"] == "explicit_publication_time"
    assert "2025-01-02" in known["yield_known_at"]


def test_outcome_price_basis_must_match_before_a_direction_label_is_defined():
    current = {"close": 100, "price_basis": "raw"}
    assert _outcome_label(current, {"close": 50, "price_basis": "split_adjusted"}) is None
    assert _outcome_label(current, {"close": 101, "price_basis": "raw"}) == 1
    assert _outcome_label(current, {"close": 99, "price_basis": "raw"}) == 0


def test_every_mature_origin_and_predeclared_asset_is_scored(evaluated):
    days, _, _, protocol, result = evaluated
    origins = [day for day in days if protocol.evaluation_start <= day <= protocol.evaluation_end]
    for horizon in (1, 5, 20):
        primary = result["primary"]["horizons"][str(horizon)]
        stocks = result["individual"]["horizons"][str(horizon)]
        assert primary["scheduled_cases"] == (len(origins)-horizon)*2
        assert stocks["scheduled_cases"] == len(origins)-horizon
        assert primary["outcome_coverage"] == primary["prediction_coverage"] == 1
    assert all(row["outcome_date"] <= protocol.evaluation_end for row in result["predictions"])
    assert result["status"] == "target_not_met"  # Too few dates for the preregistered gate.


def test_training_labels_mature_before_fit_with_a_full_session_embargo(evaluated):
    days, _, _, protocol, result = evaluated
    for model in result["models"]:
        fit_i = days.index(model["fit_date"])
        assert days.index(model["training_label_cutoff"]) <= fit_i-protocol.embargo_sessions
        assert (days.index(model["training_origin_cutoff"])+model["horizon_sessions"]
                <= fit_i-protocol.embargo_sessions)
        assert len(model["coefficients"]) == len(FEATURES)+1
    for selected in result["validation"]:
        assert selected["latest_label_date"] < protocol.evaluation_start
        assert selected["selected"]["training_label_cutoff"] < selected["start"]
        assert (days.index(selected["calibration_label_cutoff"])
                < days.index(selected["selection_start"])-protocol.embargo_sessions)


def test_future_price_changes_cannot_change_prior_predictions_or_model_selection(evaluated):
    days, market, macro, protocol, first = evaluated
    changed = deepcopy(market)
    change_day = days[-6]
    for rows in changed.values():
        for row in rows:
            if row["date"] >= change_day:
                for key in ("open", "high", "low", "close"):
                    row[key] *= 3
    second = run_walk_forward_evaluation(changed, macro, protocol)
    assert first["validation"] == second["validation"]
    def before(result):
        return [(r["ticker"], r["origin_date"], r["horizon_sessions"], r["prediction_up"],
                 r["probability_up"], r["model_id"])
                for r in result["predictions"] if r["origin_date"] < change_day]
    assert before(first) == before(second)


def test_future_appended_data_never_enters_the_fixed_evaluation_window(evaluated):
    _, market, macro, protocol, first = evaluated
    changed = deepcopy(market)
    for rows in changed.values():
        rows.append({**rows[-1], "date": "2025-02-03", "close": 999999})
    assert run_walk_forward_evaluation(changed, macro, protocol) == first


def test_missing_asset_still_contributes_scheduled_misses_not_selective_win_rate(evaluated):
    _, market, macro, protocol, _ = evaluated
    reduced = {key: value for key, value in market.items() if key != "AAA"}
    result = run_walk_forward_evaluation(reduced, macro, protocol)
    metrics = result["individual"]["primary_horizon"]
    assert metrics["scheduled_cases"] > 0 and metrics["known_outcomes"] == 0
    assert metrics["outcome_coverage"] == metrics["prediction_coverage"] == 0
    assert metrics["improvement_gate"] is False
    assert all(r["blocked_missing_data"] for r in result["predictions"]
               if r["ticker"] == "AAA")


def test_options_and_current_event_prices_cannot_fill_historical_features(evaluated):
    _, market, macro, protocol, _ = evaluated
    result = run_walk_forward_evaluation(market, macro, protocol,
                                        archived_options={"current_iv": .3},
                                        archived_prediction_markets={"today_yes": .99})
    assert not any(r["historical_options_used"] or r["historical_event_probabilities_used"]
                   for r in result["predictions"])
    assert "not_used" in result["input_availability"]["prediction_markets"]
    assert result["reference_claims"]["status"].startswith("user_reported_unverified")
    json.dumps(result, allow_nan=False)


def test_macro_uses_prior_completed_session_and_rejects_current_or_stale_observations():
    indexed = _macro_index([{"series_id": "DGS10", "source": "FRED public CSV", "history": [
        {"date": "2026-10-26", "value": 4.2}, {"date": "2026-10-27", "value": 9}]}])
    assert _macro_at(indexed, "2026-10-26")["yield"] == 4.2
    assert _macro_at(indexed, "2026-11-10")["yield"] is None
    assert _macro_at(_macro_index([{"series_id": "DGS10", "source": "sample",
                                   "history": [{"date": "2026-10-26", "value": 4}]}]),
                     "2026-10-26")["yield"] is None


def test_feature_date_and_macro_dates_are_auditable(evaluated):
    _, _, _, _, result = evaluated
    assert all(r["input_dates"]["price_end"] == r["origin_date"]
               for r in result["predictions"])
    assert all(r["input_dates"]["macro"]["cutoff"] < r["origin_date"]
               and r["input_dates"]["macro"]["yield_date"] <=
               r["input_dates"]["macro"]["cutoff"] for r in result["predictions"])


def test_unverified_incomplete_nonfinite_and_duplicate_prices_are_rejected():
    days, market, macro, _ = research_fixture()
    price = market["QQQ"][-1]
    for change in ({"source": "sample"}, {"session_complete": False},
                   {"close": float("nan")}, {"low": -1}):
        assert days[-1] not in _prices({"QQQ": [{**price, **change}]}, days[-1])["QQQ"]
    assert not _prices({"QQQ": [price, price]}, days[-1])["QQQ"]
    clean = _prices(market, days[-1])
    values = _features(clean, "QQQ", days, len(days)-1,
                       _macro_at(_macro_index(macro), days[-2]))
    assert len(values["x"]) == len(FEATURES) and all(math.isfinite(v) for v in values["x"])


def test_frozen_baseline_precedence_keeps_macro_stress_and_dip_exception_explicit():
    feature = {"close": 110, "sma20": 105, "sma50": 100,
               "macro": {"yield": 5.1, "hy": 3}, "x": [0.0]*len(FEATURES)}
    assert _shunge_proxy(feature) == 0
    feature["macro"]["yield"] = 4.5
    assert _shunge_proxy(feature) == 1
    feature.update(close=90, sma20=95)
    assert _shunge_proxy(feature) == 0
    feature["x"][11], feature["x"][14] = -.1, 1
    assert _shunge_proxy(feature) == 1
    feature["macro"]["yield"] = None
    assert _shunge_proxy(feature) == 0


def test_missing_predictions_are_misses_and_block_the_complete_coverage_gate():
    days = _calendar("2026-04-27", "2026-10-20")
    rows = [{"origin_date": day, "actual_up": 1, "prediction_up": 1,
             "shunge_proxy_up": 0, "probability_up": .7} for day in days]
    rows[0]["prediction_up"] = rows[0]["probability_up"] = None
    result = _metrics(rows, EvaluationProtocol(bootstrap_resamples=40), 5)
    assert result["accuracy"] == (len(rows)-1)/len(rows)
    assert result["checks"]["complete_cohort_coverage"] is False
    assert result["improvement_gate"] is False
    assert result["always_up_accuracy"] == 1


def test_beating_a_weak_proxy_while_losing_always_up_cannot_pass_research_gate():
    days = _calendar("2026-04-27", "2026-10-20")[:120]
    rows = [{"origin_date": day, "actual_up": 1, "prediction_up": int(i % 10 != 0),
             "shunge_proxy_up": int(i % 5 != 0), "probability_up": .7}
            for i, day in enumerate(days) for _ in range(2)]
    metrics = _metrics(rows, EvaluationProtocol(bootstrap_resamples=200), 5)
    assert metrics["accuracy"] == .9 and metrics["shunge_proxy_accuracy"] == .8
    assert metrics["paired_gain"] > .03
    assert metrics["checks"]["beats_both_baselines"] is False
    assert metrics["improvement_gate"] is False


def test_flat_returns_are_labeled_up_and_session_horizon_skips_holidays(evaluated):
    days, market, macro, protocol, _ = evaluated
    changed = deepcopy(market)
    origin = days[-10]
    changed["QQQ"][-5]["close"] = changed["QQQ"][-10]["close"]
    changed["QQQ"][-5]["open"] = changed["QQQ"][-5]["close"]
    changed["QQQ"][-5]["high"] = changed["QQQ"][-5]["close"]*1.01
    changed["QQQ"][-5]["low"] = changed["QQQ"][-5]["close"]*.99
    result = run_walk_forward_evaluation(changed, macro, protocol)
    row = next(r for r in result["predictions"]
               if r["ticker"] == "QQQ" and r["origin_date"] == origin
               and r["horizon_sessions"] == 5)
    assert row["actual_up"] == 1
    assert "2025-01-20" not in days
