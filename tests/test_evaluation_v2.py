"""Synthetic tests only: no frozen half-year research data is opened here."""

import json
import math
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from twiz import evaluation as v1
from twiz.evaluation_v2 import (
    EvaluationV2Protocol,
    _better,
    _Cohort,
    _prepare,
    _score,
    run_walk_forward_evaluation_v2,
    select_pretest_models_v2,
)


def fixture_data():
    days = v1._calendar("2022-01-03", "2024-12-31")
    market = {}
    for ticker, phase in (("QQQ", 0), ("SPY", 1), ("AAA", 2), ("SMH", 3), ("IGV", 4)):
        market[ticker] = []
        for i, day in enumerate(days):
            close = 100*math.exp(i*.0003+.03*math.sin(i/6+phase)+.01*math.sin(i/23))
            market[ticker].append({"date": day, "close": close, "open": close,
                                   "low": close*.99, "high": close*1.01, "volume": 1000,
                                   "source": "yfinance", "session_complete": True,
                                   "price_basis": "synthetic_consistent_basis"})
    macro = [{"series_id": series, "source": "FRED public CSV",
              "history": [{"date": day, "value": level+.1*math.sin(i/13)}
                          for i, day in enumerate(days)]}
             for series, level in (("DGS10", 4.1), ("BAMLH0A0HYM2", 3.2),
                                   ("DCOILBRENTEU", 75))]
    protocol = EvaluationV2Protocol(
        evaluation_start=days[-26], evaluation_end=days[-1],
        secondary_horizons_sessions=(1,), individual_tickers=("AAA",),
        pretest_validation_sessions=21, candidate_windows_sessions=(63, 126),
        mean_loss_penalties=(.001,), calibration_window_sessions=21,
        prior_window_sessions=126, bootstrap_resamples=20)
    return days, market, macro, protocol


@pytest.fixture(scope="module")
def studied():
    days, market, macro, protocol = fixture_data()
    prior = {ticker: [row for row in rows if row["date"] < protocol.evaluation_start]
             for ticker, rows in market.items()}
    earlier_macro = [{**row, "history": [item for item in row["history"]
                                        if item["date"] < protocol.evaluation_start]}
                     for row in macro]
    selection = select_pretest_models_v2(prior, earlier_macro, protocol)
    result = run_walk_forward_evaluation_v2(market, macro, protocol, selection)
    return days, market, macro, protocol, selection, result


def test_v2_config_freezes_default_grid_and_one_direction_threshold():
    config = {"training": {"candidate_windows_sessions": [252, 504, 1260],
                            "mean_loss_penalties": [.001, .01],
                            "pretest_validation_sessions": 126,
                            "stock_direction_thresholds": [.5]}}
    protocol = EvaluationV2Protocol.from_config(config)
    assert protocol.candidate_windows_sessions == (252, 504, 1260)
    assert protocol.mean_loss_penalties == (.001, .01)
    with pytest.raises(ValueError, match="threshold"):
        replace(protocol, stock_direction_thresholds=(.45, .5)).validate()
    with pytest.raises(ValueError, match="three"):
        replace(protocol, pretest_validation_sessions=125).validate()


@pytest.mark.parametrize("key,value", [
    ("prediction_time", "before_US_open"), ("label", "open_to_close_direction"),
    ("flat_return_label", "down"),
])
def test_v2_cannot_silently_accept_incompatible_forecast_or_label_semantics(key, value):
    with pytest.raises(ValueError, match=key):
        EvaluationV2Protocol.from_config({key: value})


def test_selection_labels_and_all_calibration_labels_precede_fit_with_embargo(studied):
    days, _, _, protocol, selection, result = studied
    for chosen in selection["validation"]:
        assert chosen["latest_label_date"] < protocol.evaluation_start
        assert len(chosen["prior_scores"]["blocks"]) == 3
        for candidate in chosen["candidates"]:
            assert len(candidate["scores"]["blocks"]) == 3
        models = chosen["models"]
        for model in models:
            fit = days.index(model["fit_date"])
            assert days.index(model["training_label_cutoff"]) <= fit-protocol.embargo_sessions
            cal = model["calibrator"]
            if cal["label_cutoff"]:
                assert days.index(cal["label_cutoff"]) <= fit-protocol.embargo_sessions
                assert days.index(cal["origin_cutoff"]) <= (
                    fit-model["horizon_sessions"]-protocol.embargo_sessions)
            assert model["fit_date"] < protocol.evaluation_start
    for model in result["models"]:
        assert days.index(model["training_label_cutoff"]) <= (
            days.index(model["fit_date"])-protocol.embargo_sessions)


def test_mean_loss_penalty_is_scaled_by_training_sample_count(studied):
    _, market, macro, protocol, _, _ = studied
    prepared = _prepare(market, macro, protocol)
    data = _Cohort(prepared, protocol.primary_tickers, 5, protocol)
    model = data.fit(prepared["start_i"], {"kind": "ridge_logistic", "window_sessions": 126,
                                          "mean_loss_penalty": .001})
    assert model["summed_loss_penalty"] == .001*model["training_samples"]
    assert model["penalty"] == model["summed_loss_penalty"]
    x = np.array([[math.sin(i), math.cos(i)] for i in range(100)])
    y = np.array([i % 3 != 0 for i in range(100)], dtype=int)
    a = v1._logistic(x, y, .01*len(y))
    b = v1._logistic(np.tile(x, (2, 1)), np.tile(y, 2), .01*len(y)*2)
    assert a["coefficients"] == pytest.approx(b["coefficients"], abs=1e-7)


def summary(brier, loss, block_briers):
    return {"aggregate": {"brier": brier, "logloss": loss},
            "blocks": [{"brier": x} for x in block_briers]}


def test_calibration_needs_probability_quality_and_two_independent_blocks():
    raw = summary(.25, .7, [.25, .25, .25])
    assert _better(summary(.24, .69, [.24, .24, .26]), raw, False)
    assert not _better(summary(.24, .71, [.23, .23, .26]), raw, False)
    assert not _better(summary(.24, .69, [.20, .26, .26]), raw, False)
    assert not _better(summary(.25, .7, [.25, .25, .25]), raw, False)
    # Model selection additionally demands strict aggregate logloss improvement over prior.
    assert not _better(summary(.24, .7, [.24, .24, .26]), raw, True)


def test_missing_forecast_is_penalized_in_selection_instead_of_being_dropped():
    rows = [{"actual_up": 1, "p": .8}, {"actual_up": 0, "p": None}]
    score = _score(rows, "p")
    assert score["brier"] == pytest.approx((.04+1)/2)
    assert score["accuracy"] == .5 and score["prediction_cases"] == 1
    assert score["logloss"] > 10


def test_sealed_selection_remains_identical_when_retest_prices_change(studied):
    _, market, macro, protocol, selection, result = studied
    changed = deepcopy(market)
    for rows in changed.values():
        for row in rows:
            if row["date"] >= protocol.evaluation_start:
                for key in ("open", "high", "low", "close"):
                    row[key] *= 2
    # Even a full input is date-filtered for selection, independently of the caller's clipping.
    assert select_pretest_models_v2(changed, macro, protocol) == selection
    new = run_walk_forward_evaluation_v2(changed, macro, protocol, selection)
    assert result["validation"] == new["validation"]
    first = protocol.evaluation_start
    original_predictions = [row for row in result["predictions"] if row["origin_date"] == first]
    # Fit parameters at first origin never use its still-unmatured retest outcome.
    old_models = {m["id"]: m for m in result["models"]}
    new_models = {m["id"]: m for m in new["models"]}
    for row in original_predictions:
        assert old_models[row["model_id"]] == new_models[row["model_id"]]


def test_every_mature_case_and_forecast_authorship_remains_visible(studied):
    _, _, _, protocol, _, result = studied
    primary = result["primary"]["horizons"]["5"]
    assert primary["scheduled_cases"] == (26-5)*2
    assert result["individual"]["horizons"]["5"]["scheduled_cases"] == 26-5
    assert all(row["probability_authorship"] in {
        "historical_pooled_prior_only", "raw_logistic_score_uncalibrated",
        "rolling_OOF_Platt_pretest_supported_not_guaranteed"} for row in result["predictions"])
    assert result["goal_completion_proven"] is False
    assert result["halfyear_result_type"] == "adaptive_retest_not_untouched_holdout"
    assert not any(row["historical_options_used"] or row["historical_event_probabilities_used"]
                   for row in result["predictions"])
    assert max(row["outcome_date"] for row in result["predictions"]) <= protocol.evaluation_end
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", ["source", "protocol", "late_labels", "missing_cohort"])
def test_mismatched_selection_is_rejected_before_running_retest(studied, change):
    _, market, macro, protocol, selection, _ = studied
    bad = deepcopy(selection)
    if change == "source":
        bad["source_sha256"]["engine"] = "changed"
    elif change == "protocol":
        bad["protocol_sha256"] = "changed"
    elif change == "late_labels":
        bad["validation"][0]["latest_label_date"] = protocol.evaluation_start
    else:
        bad["validation"].pop()
    with pytest.raises(ValueError):
        run_walk_forward_evaluation_v2(market, macro, protocol, bad)


def test_prior_fallback_is_explicit_and_never_a_claim_of_model_improvement(monkeypatch):
    _, market, macro, protocol = fixture_data()
    monkeypatch.setattr("twiz.evaluation_v2._better", lambda *args, **kwargs: False)
    selection = select_pretest_models_v2(market, macro, protocol)
    assert all(row["selected"]["kind"] == "prior_only" for row in selection["validation"])
    result = run_walk_forward_evaluation_v2(market, macro, protocol, selection)
    assert result["primary"]["primary_horizon"]["prior_only_fallback"] is True
    assert result["primary"]["primary_horizon"]["improvement_gate"] is False
    assert all(row["probability_authorship"] == "historical_pooled_prior_only"
               for row in result["predictions"])


@pytest.mark.parametrize("input_kind", ["prices", "macro"])
def test_sealed_selection_cannot_be_applied_to_different_pretest_inputs(studied, input_kind):
    _, market, macro, protocol, selection, _ = studied
    changed, macro_changed = deepcopy(market), deepcopy(macro)
    if input_kind == "prices":
        row = next(row for row in changed["QQQ"]
                   if row["date"] == protocol.evaluation_start)
        previous = changed["QQQ"][changed["QQQ"].index(row)-1]
        for key in ("open", "high", "low", "close"):
            previous[key] *= 1.001
    else:
        macro_changed[0]["history"][-40]["value"] += .1
    with pytest.raises(ValueError, match="inputs differ"):
        run_walk_forward_evaluation_v2(changed, macro_changed, protocol, selection)
