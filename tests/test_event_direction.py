from copy import deepcopy

import pytest

from twiz.event_direction import aggregate_event_direction

NOW = "2026-10-28T13:00:00Z"
END = "2026-11-18T13:00:00Z"


def component(**changes):
    return {"event_id": "fed", "risk_group": "US_policy_October", "channel": "us_rates",
            "yes_probability": 0.8, "no_probability": 0.2,
            "yes_equity_effect": 0.5, "no_equity_effect": -1,
            "available_at": "2026-10-28T12:00:00Z", "end_at": "2026-10-29T18:00:00Z",
            "source_ref": "archived/raw.json#/history/1", **changes}


def aggregate(rows, **changes):
    return aggregate_event_direction(rows, as_of=NOW, horizon_end=END,
                                     channel_weights={"us_rates": 3, "geopolitics": 1},
                                     **changes)


def test_conditional_effects_and_weights_are_features_not_equity_probabilities():
    war = component(event_id="war", risk_group="war", channel="geopolitics",
                    yes_probability=0.7, no_probability=0.3,
                    yes_equity_effect=-1, no_equity_effect=0.25)
    result = aggregate([component(), war])
    # rates .8*.5+.2*(-1)=.2; war .7*(-1)+.3*.25=-.625
    assert result["direction_score"] == pytest.approx((3*0.2-0.625)/4)
    assert result["channel_coverage"] == 1
    assert result["equity_up_probability"] is None
    assert [row["source_ref"] for row in result["contributions"]]


def test_binary_rewording_with_swapped_effects_has_same_impact():
    a = component()
    b = component(yes_probability=0.2, no_probability=0.8,
                  yes_equity_effect=-1, no_equity_effect=0.5)
    assert aggregate([a])["direction_score"] == pytest.approx(aggregate([b])["direction_score"])


def test_duplicate_or_opposite_contracts_do_not_multiply_shared_risk_weight():
    a = component()
    duplicate = {**a, "event_id": "fed-alternative"}
    assert aggregate([a, duplicate])["direction_score"] == aggregate([a])["direction_score"]
    assert aggregate([a, duplicate])["excluded"][0]["reason"] == "same_risk_already_represented"
    other = component(event_id="fed2", risk_group="US_policy_November",
                      yes_probability=1, no_probability=0)
    result = aggregate([a, other])
    assert sum(row["weight"] for row in result["contributions"]) == 3
    assert result["channel_coverage"] == 0.75


def test_one_event_cannot_gain_weight_by_changing_its_risk_group():
    a = component()
    changed_group = {**a, "risk_group": "renamed-duplicate"}
    result = aggregate([a, changed_group])
    assert result["status"] == "unavailable"
    assert all(row["reason"] == "conflicting_event_identity_mapping" for row in result["excluded"])


def test_soft_aggregation_and_variable_threshold_can_be_compared():
    a = component(yes_probability=0.6, no_probability=0.4)
    assert aggregate([a])["status"] == "ready"
    assert aggregate([a], main_case_only=True)["status"] == "unavailable"
    assert aggregate([a], main_case_only=True, event_threshold=0.55)["status"] == "ready"


def test_already_expected_level_and_repricing_are_kept_separate():
    result = aggregate([component(reference_yes_probability=0.9,
                                  reference_available_at="2026-10-27T12:00:00Z")])
    event = result["contributions"][0]
    assert event["expected_impact"] > 0
    assert event["repricing_impact"] == pytest.approx(-0.15)


@pytest.mark.parametrize("reference_at", [None, "2026-10-28T12:00:00Z",
                                         "2026-10-28T14:00:00Z", "2026-10-27"])
def test_repricing_cannot_use_unknown_or_future_reference_quote(reference_at):
    result = aggregate([component(reference_yes_probability=0.9,
                                  reference_available_at=reference_at)])
    assert result["contributions"][0]["repricing_impact"] is None


def test_ambiguous_both_main_outcomes_do_not_pick_yes_by_order():
    row = component(yes_probability=0.51, no_probability=0.51)
    result = aggregate([row], main_case_only=True, event_threshold=0.51)
    assert result["status"] == "unavailable"
    assert result["excluded"][0]["reason"] == "ambiguous_binary_main_case"


@pytest.mark.parametrize("changes,reason", [
    ({"available_at": "2026-10-28T14:00:00Z"}, "unavailable_or_invalid_timestamp"),
    ({"available_at": "2026-10-26T12:00:00Z"}, "stale"),
    ({"end_at": "2026-12-01T12:00:00Z"}, "outside_swing_horizon"),
    ({"yes_probability": float("nan")}, "invalid_binary_probability"),
    ({"no_equity_effect": None}, "both_conditional_effects_required"),
    ({"risk_group": None}, "missing_identity_or_shared_risk_group"),
    ({"channel": "unknown"}, "channel_has_no_weight"),
])
def test_unknown_and_unavailable_inputs_do_not_give_directional_authority(changes, reason):
    result = aggregate([component(**changes)])
    assert result["direction_score"] is None and result["channel_coverage"] == 0
    assert result["excluded"][0]["reason"] == reason


def test_input_not_mutated_and_missing_channels_do_not_amplify_remaining_event():
    rows = [component()]
    before = deepcopy(rows)
    result = aggregate(rows)
    assert rows == before
    assert result["channel_coverage"] == 0.75
    assert result["direction_score"] == pytest.approx(0.15)


def test_invalid_weight_or_naive_cutoff_fails_explicitly():
    with pytest.raises(ValueError):
        aggregate_event_direction([], as_of=NOW, horizon_end=END,
                                  channel_weights={"us_rates": -1})
    with pytest.raises(ValueError):
        aggregate_event_direction([], as_of="2026-10-28", horizon_end=END, channel_weights={})
    with pytest.raises(ValueError, match="sum"):
        aggregate_event_direction([], as_of=NOW, horizon_end=END,
                                  channel_weights={"us_rates": 1e308, "war": 1e308})
