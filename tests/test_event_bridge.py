from copy import deepcopy
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from twiz.event_bridge import _fit, fit_event_bridge, predict_event_bridge

TARGET = "close_20_sessions_ahead_above_origin_completed_close"
CUTOFF = "2026-10-28T13:00:00Z"


def examples(groups=120, per_group=4):
    start, rows = datetime(2024, 1, 1, tzinfo=timezone.utc), []
    for group in range(groups):
        origin = start+timedelta(days=group*5)
        outcome = int(group % 2 == 1)
        for sample in range(per_group):
            stamp = origin+timedelta(hours=sample)
            rows.append({"id": f"{group}-{sample}", "episode_id": f"event-{group:03}",
                         "origin_at": stamp.isoformat(), "features_available_at": stamp.isoformat(),
                         "outcome_available_at": (stamp+timedelta(days=2)).isoformat(),
                         "features": {"event_probability": 0.9 if outcome else 0.1,
                                      "price": 0.0}, "label": outcome,
                         "label_definition": TARGET})
    return rows


def fit(rows, **changes):
    return fit_event_bridge(rows, as_of=CUTOFF,
                            feature_names=("event_probability", "price"),
                            baseline_feature_names=("price",), label_definition=TARGET, **changes)


def test_known_synthetic_event_signal_is_learned_with_chronological_disjoint_groups():
    artifact = fit(examples())
    assert artifact["probability_estimate_eligible"]
    assert artifact["production_eligible"] is False
    assert artifact["model"]["coefficients"][1] > 0
    blocks = artifact["blocks"]
    assert not set(blocks["fit"]["episodes"]) & set(blocks["calibration"]["episodes"])
    assert not set(blocks["calibration"]["episodes"]) & set(blocks["validation"]["episodes"])
    result = predict_event_bridge(artifact, {"event_probability": 0.9, "price": 0},
                                  as_of=CUTOFF, features_available_at=CUTOFF)
    assert result["equity_up_probability"] > 0.5
    log_odds = result["intercept"]+sum(row["log_odds_contribution"]
                                      for row in result["contributions"])
    assert result["raw_probability"] == pytest.approx(1/(1+np.exp(-log_odds)))
    assert result["automatic_policy_promotion"] is False


def test_thousands_of_quotes_from_eleven_events_do_not_fake_independent_history():
    artifact = fit(examples(groups=11, per_group=100))
    assert artifact["status"] == "insufficient_history"
    assert artifact["independent_event_episodes"] == 11
    result = predict_event_bridge(artifact, {"event_probability": 0.9, "price": 0},
                                  as_of=CUTOFF, features_available_at=CUTOFF)
    assert result["equity_up_probability"] is None


def test_quote_duplication_within_episode_does_not_increase_learned_weight():
    rows = examples(groups=10)
    copied = [{**row, "id": row["id"]+"-copy"} for row in rows if row["episode_id"] == "event-001"]
    a = _fit(rows, ("event_probability", "price"), 1)
    b = _fit(rows+copied*10, ("event_probability", "price"), 1)
    assert a["coefficients"] == pytest.approx(b["coefficients"])
    assert a["mean"] == pytest.approx(b["mean"])


def test_future_labels_and_future_features_cannot_enter_fit():
    rows = examples()
    before = deepcopy(rows)
    future = {**rows[-1], "id": "future", "outcome_available_at": "2026-10-29T00:00:00Z"}
    unseen_feature = {**rows[-1], "id": "future-feature", "features_available_at": CUTOFF}
    a, b = fit(rows), fit(rows+[future, unseen_feature])
    assert a["model"] == b["model"]
    assert a["input_sha256"] == b["input_sha256"]
    assert len(b["excluded"]) == 2
    assert rows == before


def test_labels_crossing_next_block_are_purged_before_model_fit():
    rows = examples()
    for row in rows:
        if row["episode_id"] == "event-071":
            row["outcome_available_at"] = "2026-01-01T00:00:00Z"
    result = fit(rows)
    assert "event-071" not in result["blocks"]["fit"]["episodes"]


def test_single_class_and_different_target_never_return_a_validated_probability():
    rows = examples()
    for row in rows:
        row["label"] = 1
    assert fit(rows)["status"] == "insufficient_purged_blocks"
    for row in rows:
        row["label_definition"] = "a_different_horizon_or_profit_target"
    result = fit(rows)
    assert result["accepted_samples"] == 0


def test_one_sided_calibration_still_records_research_fit_without_claiming_probability():
    rows = examples()
    for row in rows:
        if 72 <= int(row["episode_id"].split("-")[1]) < 96:
            row["label"] = 1
    result = fit(rows)
    assert result["model"]["training_samples"] > 0
    assert result["validation"]["samples"] > 0
    assert result["calibrator"]["available"] is False
    assert result["probability_estimate_eligible"] is False
    assert result["production_eligible"] is False


def test_validation_cannot_choose_penalty_or_refit_coefficients():
    rows = examples()
    first = fit(rows)
    validation_groups = set(first["blocks"]["validation"]["episodes"])
    changed = [{**row, "label": 1-row["label"]}
               if row["episode_id"] in validation_groups else row for row in rows]
    second = fit(changed)
    assert first["model"] == second["model"]
    assert first["calibrator"] == second["calibrator"]
    assert second["probability_estimate_eligible"] is False


def test_predict_cannot_use_model_fitted_in_future():
    artifact = fit(examples(groups=11))
    with pytest.raises(ValueError, match="precede"):
        predict_event_bridge(artifact, {}, as_of="2026-10-27T13:00:00Z",
                             features_available_at="2026-10-27T13:00:00Z")
