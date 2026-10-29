"""Synthetic residual-offset evidence only; no historical labels are read."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from twiz.event_residual import TARGET, _fit, fit_event_residual, predict_event_residual

CUTOFF = "2026-10-28T13:00:00Z"


def examples(groups=60, per_group=4):
    rows, start = [], datetime(2024, 1, 1, tzinfo=timezone.utc)
    for group in range(groups):
        for i in range(per_group):
            origin = start + timedelta(days=10 * group, hours=i)
            y = group % 2
            rows.append({"id": f"{group}-{i}", "episode_id": f"event-{group:03d}",
                         "origin_at": origin.isoformat(),
                         "features_available_at": origin.isoformat(),
                         "outcome_available_at": (origin + timedelta(days=2)).isoformat(),
                         "baseline_model_fit_at": (origin - timedelta(days=1)).isoformat(),
                         "baseline_label_cutoff": (origin - timedelta(days=2)).isoformat(),
                         "baseline_feature_available_at": origin.isoformat(),
                         "baseline_probability": .65, "label": y, "label_definition": TARGET,
                         "event_features": {"signal": 1.0 if y else -1.0, "optional": None},
                         "observed_mask": {"signal": True, "optional": False}})
    return rows


def fit(rows, **changes):
    return fit_event_residual(rows, as_of=CUTOFF,
                              event_feature_names=("signal", "optional"), **changes)


def predict(artifact, **changes):
    kwargs = {"baseline_probability": .7, "event_features": {"signal": 1., "optional": None},
              "observed_mask": {"signal": True, "optional": False}, "as_of": CUTOFF,
              "baseline_model_fit_at": CUTOFF, "baseline_feature_available_at": CUTOFF,
              "features_available_at": CUTOFF}
    kwargs.update(changes)
    return predict_event_residual(artifact, **kwargs)


def test_known_event_signal_beats_same_price_prior_on_untouched_validation():
    artifact = fit(examples())
    assert artifact["validated"] and artifact["event_authority_enabled"]
    assert artifact["model"]["intercept"] == 0
    assert artifact["model"]["coefficients"][0] > 0
    assert artifact["model"]["coefficients"][1] == 0
    assert artifact["validation"]["brier"] < artifact["price_prior_validation"]["brier"]
    assert artifact["validation"]["log_loss"] < artifact["price_prior_validation"]["log_loss"]
    assert artifact["production_eligible"] is artifact["probability_calibrated"] is False
    result = predict(artifact)
    adjustment = sum(item["log_odds_contribution"] for item in result["contributions"])
    expected = 1 / (1 + np.exp(-(np.log(.7 / .3) + adjustment)))
    assert result["probability"] == pytest.approx(expected)
    assert result["contributions"][1]["log_odds_contribution"] == 0


def test_missing_events_contribute_exactly_zero_even_with_invalid_unused_values():
    rows = examples()
    for row in rows:
        row["event_features"]["optional"] = float("nan")
    artifact = fit(rows)
    result = predict(artifact, event_features={"signal": float("nan"), "optional": object()},
                     observed_mask={"signal": False, "optional": False})
    assert result["probability"] == .7 and result["event_log_odds_adjustment"] == 0
    assert not result["event_authority_enabled"]
    assert all(item["standardized_value"] == item["log_odds_contribution"] == 0
               for item in result["contributions"])


def test_no_skill_falls_back_without_forcing_positive_weights_or_intercept():
    rows = examples()
    for row in rows:
        row["event_features"]["signal"] = 1.
    artifact = fit(rows)
    assert artifact["status"] == "price_prior_fallback"
    assert artifact["model"]["coefficients"] == [0., 0.]
    assert artifact["validation"] == artifact["price_prior_validation"]
    assert predict(artifact)["probability"] == .7
    reversed_rows = examples()
    for row in reversed_rows:
        row["event_features"]["signal"] *= -1
    assert fit(reversed_rows)["model"]["coefficients"][0] < 0


def test_validation_labels_cannot_choose_penalty_refit_or_observed_scales():
    rows = examples()
    expected = fit(rows)
    validation_ids = set(expected["blocks"]["validation"]["ids"])
    for row in rows:
        if row["id"] in validation_ids:
            row["label"] = 1 - row["label"]
    changed = fit(rows)
    assert expected["model"] == changed["model"]
    assert expected["penalty_selection"] == changed["penalty_selection"]
    assert changed["status"] == "price_prior_fallback"
    assert predict(changed)["probability"] == .7


def test_one_class_calibration_is_scored_without_a_calibration_evidence_claim():
    rows = examples()
    for row in rows:
        if 30 <= int(row["episode_id"][-3:]) < 45:
            row["label"] = 1
    artifact = fit(rows)
    assert artifact["penalty_selection"] and artifact["validation"]["samples"] > 0
    assert artifact["calibration_has_both_classes"] is False
    assert artifact["independent_probability_calibration"] is False


class NoOutcomes(dict):
    def get(self, key, default=None):
        if key in {"label", "baseline_probability", "event_features", "observed_mask"}:
            raise AssertionError("Future numeric field inspected")
        return super().get(key, default)


def test_future_rows_are_excluded_before_label_prior_or_event_numeric_reads():
    rows = examples()
    future = NoOutcomes({**rows[-1], "id": "future",
                         "outcome_available_at": "2027-01-01T00:00:00Z"})
    before, after = fit(rows), fit(rows + [future])
    assert before["model"] == after["model"]
    assert before["input_sha256"] == after["input_sha256"]


@pytest.mark.parametrize("field,value", [
    ("baseline_model_fit_at", "2026-10-28T14:00:00Z"),
    ("baseline_feature_available_at", "2026-10-28T14:00:00Z"),
    ("baseline_label_cutoff", "2026-10-28T14:00:00Z"),
])
def test_dated_price_prior_proofs_cannot_use_information_after_origin(field, value):
    row = NoOutcomes({**examples()[0], field: value})
    assert fit([row])["accepted_samples"] == 0


def test_refit_purges_all_labels_before_validation_and_episodes_are_disjoint():
    rows = examples()
    for row in rows:
        if row["episode_id"] in {"event-029", "event-044"}:
            row["outcome_available_at"] = "2026-01-01T00:00:00Z"
    artifact = fit(rows)
    blocks = artifact["blocks"]
    assert "event-029" not in blocks["train"]["episodes"]
    assert "event-044" not in blocks["calibration"]["episodes"]
    assert not set(blocks["train"]["episodes"]) & set(blocks["calibration"]["episodes"])
    validation_start = min(row["origin_at"] for row in rows
                           if row["id"] in blocks["validation"]["ids"])
    assert artifact["model"]["training_outcome_cutoff"] < validation_start


def test_observed_only_standardization_and_equal_episode_duplication_weights():
    rows = examples(12)
    for i, row in enumerate(rows):
        row["observed_mask"]["optional"] = i % 3 == 0
        row["event_features"]["optional"] = float(i) if i % 3 == 0 else None
    model = _fit(rows, ("signal", "optional"), 1.)
    observed = [i for i in range(len(rows)) if i % 3 == 0]
    assert model["mean"][1] == pytest.approx(np.mean(observed))
    copied = [deepcopy(row) for row in rows if row["episode_id"] == "event-001"]
    other = _fit(rows + copied * 20, ("signal", "optional"), 1.)
    assert model["mean"] == pytest.approx(other["mean"])
    assert model["coefficients"] == pytest.approx(other["coefficients"])


def test_thousands_of_quotes_from_eleven_episodes_remain_insufficient():
    artifact = fit(examples(11, 100))
    assert artifact["status"] == "insufficient_history"
    assert artifact["event_episodes"] == 11
    assert predict(artifact)["probability"] == .7


def test_future_event_model_or_feature_clock_returns_only_dated_price_prior():
    artifact = fit(examples())
    assert predict(artifact, features_available_at="2027-01-01T00:00:00Z")["probability"] == .7
    future = {**artifact, "as_of": "2027-01-01T00:00:00Z"}
    future_result = predict(future)
    assert future_result["probability"] == .7 and future_result["validated"] is False
    with pytest.raises(ValueError, match="Price prior"):
        predict(artifact, baseline_model_fit_at="2027-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="Price prior"):
        predict(artifact, baseline_feature_available_at="2027-01-01T00:00:00Z")
    with pytest.raises(ValueError, match="Price prior"):
        predict(artifact, baseline_feature_available_at=None)


def test_future_price_proof_is_rejected_before_reading_the_numeric_prior():
    class FuturePrior:
        def __float__(self):
            raise AssertionError("Future prior numeric value inspected")

    artifact = fit(examples())
    with pytest.raises(ValueError, match="Price prior"):
        predict(artifact, baseline_probability=FuturePrior(),
                baseline_feature_available_at="2027-01-01T00:00:00Z")


def test_no_input_mutation_and_duplicate_or_invalid_rows_cannot_silently_change_targets():
    rows, before = examples(), examples()
    artifact = fit(rows)
    assert rows == before
    duplicate = {**rows[0], "label": 1 - rows[0]["label"]}
    duplicated = fit(rows + [duplicate])
    assert sum(r["reason"] == "duplicate_identity" for r in duplicated["excluded"]) == 2
    invalid = {**rows[0], "id": "bad", "label_definition": "swing_profit"}
    assert fit(rows + [invalid])["model"] == artifact["model"]


def test_trade_profit_or_other_horizon_cannot_be_exposed_as_equity_up_probability():
    with pytest.raises(ValueError):
        fit(examples(), label_definition="net_swing_trade_profit_positive")
    artifact = fit(examples())
    assert predict(artifact)["label_definition"] == TARGET
    with pytest.raises(ValueError, match="equity target"):
        predict(artifact, baseline_label_definition="swing_profit")
    with pytest.raises(ValueError, match="equity target"):
        predict({**artifact, "label_definition": "swing_profit"})


@pytest.mark.parametrize("changes", [
    {"penalties": [0]}, {"penalties": [float("nan")]}, {"min_event_groups": 11},
    {"min_samples": 0},
])
def test_invalid_research_protocol_is_rejected(changes):
    with pytest.raises(ValueError):
        fit(examples(), **changes)
