"""Synthetic campaign selection only; no price-backed profit or author skill is asserted."""

from copy import deepcopy
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest
import yaml

from twiz.sessions import is_us_equity_session
from twiz.swing_optimizer import (
    CORE_KEYS,
    STRATEGY_KEYS,
    _hash,
    freeze_candidate_search,
    freeze_opportunity_universe,
    select_swing_parameters,
    validate_candidate_parameters,
)


def search():
    return freeze_candidate_search({
        "selection": {"maximum_candidates": 2, "seed": 20261028},
        "event_direction": {"variants": ["learned_event_bridge", "price_only"]}})


def clock(day, hour=16, minute=15):
    stamp = datetime.combine(day, time(hour, minute), tzinfo=ZoneInfo("America/New_York"))
    return stamp.isoformat()


def block(day, name, partition, candidate_id, digest, execution_id, pnls=(.2, -.1)):
    dates, cursor = [], date.fromisoformat(day)
    while len(dates) < 5:
        if is_us_equity_session(cursor):
            dates.append(cursor)
        cursor += timedelta(days=1)
    start, end = clock(dates[0], 9, 30), clock(dates[-1])
    campaigns = []
    for i, pnl in enumerate(pnls):
        campaigns.append({"opportunity_id": f"{name}-{i}", "origin_at": clock(dates[i*2]),
                          "outcome_available_at": end, "status": "closed",
                          "net_account_pnl_pct": pnl, "new_capital_pct": 4,
                          "modeled_loss_pct": .2,
                          "entry_signal_at": clock(dates[i*2]),
                          "entered_at": clock(dates[i*2+1], 9, 30),
                          "exit_signal_at": clock(dates[i*2+1]),
                          "exited_at": clock(dates[i*2+2], 9, 30)})
    curve = [{"at": start, "equity": 100}]
    for i, stamp in enumerate(dates):
        curve.append({"at": clock(stamp), "equity": 100+(pnls[0] if i >= 2 else 0)
                      +(pnls[1] if i == 4 else 0)})
    return {"candidate_id": candidate_id, "parameters_sha256": digest,
            "execution_policy_id": execution_id, "block_id": name, "partition": partition,
            "block_start": start, "block_end": end, "outcome_available_at": end,
            "campaigns": campaigns, "equity_curve": curve}


def setup(val_blocks=3):
    frozen = search()
    default, contender = frozen["candidates"]
    records, metadata = [], []
    specifications = [("2024-01-02", "train", "train"),
                      ("2024-04-01", "v1", "validation"),
                      ("2024-07-01", "v2", "validation"),
                      ("2024-10-01", "v3", "validation"),
                      ("2024-11-04", "v4", "validation")][:val_blocks+1]
    for day, name, partition in specifications:
        baseline = block(day, name, partition, "baseline", None, frozen["execution_policy_id"])
        records.append(baseline)
        metadata.append({key: baseline[key] for key in (
            "block_id", "partition", "block_start", "block_end", "outcome_available_at")}
                        | {"opportunities": [{key: row[key] for key in (
                            "opportunity_id", "origin_at", "outcome_available_at")}
                            for row in baseline["campaigns"]]})
        for candidate in (default, contender):
            pnls = (.3, -.05) if candidate["id"] == contender["id"] else (.2, -.1)
            records.append(block(day, name, partition, candidate["id"],
                                 candidate["parameters_sha256"],
                                 frozen["execution_policy_id"], pnls))
    frozen["opportunity_universe"] = freeze_opportunity_universe(metadata)
    return frozen, records, default, contender


def select(frozen, records, minimum=2):
    return select_swing_parameters(frozen, records, selection_at="2025-01-15T22:00:00Z",
                                   min_closed_campaigns=minimum)


def test_real_declared_grid_freezes_at_most96_with_default_and_explicit_event_variants():
    with open("config/evaluation-swing.yml") as handle:
        config = yaml.safe_load(handle)
    first = freeze_candidate_search(config)
    assert first == freeze_candidate_search(deepcopy(config))
    assert first["status"] == "frozen" and len(first["candidates"]) == 96
    assert first["unsupported_declared_fields"] == []
    assert sum(row["is_default"] for row in first["candidates"]) == 1
    assert first["candidates"][0]["event_variant"] == "learned_event_bridge"
    assert {row["event_variant"] for row in first["candidates"]} == {
        "price_only", "signed_weighted_event_score", "learned_event_bridge"}
    assert all(row["parameters"]["event_threshold"] > .5
               and row["parameters"]["trade_probability_threshold"] > .5
               for row in first["candidates"])
    assert "not_implemented" in first["simulator_status"]
    assert first["candidate_sha256"] == (
        "2b803fa8bc70fc6bcce79a9a9d88da416dc6db43ccbb5b1c433103895b8a8bfd"
    )


def test_v2_search_hashes_every_strategy_knob_without_changing_v1_schema():
    with open("config/evaluation-swing-v2.yml") as handle:
        config = yaml.safe_load(handle)
    frozen = freeze_candidate_search(config)
    assert frozen == freeze_candidate_search(deepcopy(config))
    assert len(frozen["candidates"]) == 96
    first = frozen["candidates"][0]
    assert set(first["parameters"]) == CORE_KEYS | STRATEGY_KEYS
    assert first["parameters"]["entry_rule"] == "close_hold"
    assert first["parameters"]["entry_ladder"] == "atr_approach"
    assert first["parameters"]["watch_sessions"] == 10
    assert first["parameters"]["max_entry_extension_atr"] == .5
    for key, replacement in (("entry_rule", "retest"), ("entry_ladder", "zone_edges"),
                             ("watch_sessions", 20), ("max_entry_extension_atr", 2)):
        content = {key: deepcopy(first[key]) for key in (
            "parameters", "structure_parameters", "event_variant")}
        content["parameters"][key] = replacement
        assert _hash(content) != first["parameters_sha256"]
    for key in STRATEGY_KEYS:
        for value in config["search_space"][key]:
            for variant in config["event_direction"]["variants"]:
                assert any(
                    row["parameters"] == {**first["parameters"], key: value}
                    and row["event_variant"] == variant
                    and row["structure_parameters"] == first["structure_parameters"]
                    for row in frozen["candidates"]
                )
    legacy = freeze_candidate_search({"defaults": {"entry_rule": "close_hold"}})
    assert legacy["unsupported_declared_fields"] == ["defaults.entry_rule"]
    partial = dict(first["parameters"])
    partial.pop("watch_sessions")
    with pytest.raises(ValueError, match="all four"):
        validate_candidate_parameters(partial)


@pytest.mark.parametrize("values", [
    {"entry_rule": "intraday"}, {"entry_ladder": "invented"}, {"watch_sessions": 21},
    {"watch_sessions": True}, {"max_entry_extension_atr": float("nan")},
    {"max_entry_extension_atr": 0},
])
def test_v2_strategy_values_fail_closed(values):
    with pytest.raises(ValueError):
        freeze_candidate_search({"protocol_version": "author-swing-v2", "defaults": values})


def test_unknown_search_knobs_are_reported_and_unsafe_known_values_rejected():
    result = freeze_candidate_search({"search_space": {"secret_future_price": [1]}})
    assert result["status"] == "unsupported_fields" and result["candidates"] == []
    assert result["unsupported_declared_fields"] == ["search_space.secret_future_price"]
    for config in ({"defaults": {"trade_probability_threshold": .5}},
                   {"defaults": {"entry_fractions": [.5, .5, .5]}},
                   {"fixed_constraints": {"max_new_capital_pct": 11}},
                   {"selection": {"maximum_candidates": 97}}):
        with pytest.raises(ValueError):
            freeze_candidate_search(config)


def test_chronological_paired_records_select_only_profit_and_risk_improving_candidate():
    frozen, records, _, contender = setup()
    result = select(frozen, records)
    assert result["status"] == "selected_research_candidate"
    assert result["selected_id"] == contender["id"]
    assert result["parameter_application_verified"] is False
    assert result["goal_completion_proven"] is result["author_comparison_proven"] is False
    assert result["evaluation_basis"] == "supplied_campaign_records_until_simulator_validates"
    assert result["selection_opportunities"] == {"declared": 8, "mature": 8}


def test_lower_profit_cannot_win_even_with_higher_win_rate_and_lower_drawdown():
    frozen, records, default, contender = setup()
    for row in records:
        if row["candidate_id"] == contender["id"]:
            row["campaigns"][0]["net_account_pnl_pct"] = .01
            row["campaigns"][1]["net_account_pnl_pct"] = .01
            row["equity_curve"][-3:]=[{**p, "equity": 100.01 if i < 2 else 100.02}
                                      for i, p in enumerate(row["equity_curve"][-3:])]
    result = select(frozen, records)
    assert result["selected_id"] == default["id"] and result["status"] == "target_not_met"


def test_missing_losing_block_cannot_leave_three_easy_blocks_and_win():
    frozen, records, default, contender = setup(4)
    records = [row for row in records if not (row["candidate_id"] == contender["id"]
                                              and row["block_id"] == "v4")]
    result = select(frozen, records)
    assert result["selected_id"] == default["id"]
    diagnostic = next(row for row in result["diagnostics"]
                      if row["candidate_id"] == contender["id"])
    assert "missing_required_candidate_or_baseline_block" in diagnostic["reasons"]


def test_matching_block_id_with_a_different_date_window_is_rejected():
    frozen, records, default, contender = setup(4)
    changed = next(row for row in records if row["candidate_id"] == contender["id"]
                   and row["block_id"] == "v4")
    changed["block_end"] = "2024-11-09T21:15:00+00:00"
    changed["outcome_available_at"] = changed["block_end"]
    changed["equity_curve"][-1]["at"] = changed["block_end"]
    result = select(frozen, records)
    assert result["selected_id"] == default["id"]
    assert any(row["reason"] == "different_frozen_comparison_window" for row in result["excluded"])


@pytest.mark.parametrize("middle_invalid", [False, True])
def test_third_copy_cannot_resurrect_a_tainted_required_block(middle_invalid):
    frozen, records, default, contender = setup(4)
    original = next(row for row in records if row["candidate_id"] == contender["id"]
                    and row["block_id"] == "v4")
    middle = deepcopy(original)
    if middle_invalid:
        middle["parameters_sha256"] = "conflicting-parameter-hash"
    records.extend([middle, deepcopy(original)])
    result = select(frozen, records)
    assert result["selected_id"] == default["id"]
    assert sum(row["reason"] == "duplicate_or_conflicting_block_permanently_tainted"
               for row in result["excluded"]) == 2


def test_no_baseline_or_frozen_denominator_means_no_selection():
    frozen, records, default, _ = setup()
    del frozen["opportunity_universe"]
    assert select(frozen, records)["status"] == "insufficient"
    frozen, records, default, _ = setup()
    records = [row for row in records if row["candidate_id"] != "baseline"]
    result = select(frozen, records)
    assert result["status"] == "insufficient" and result["selected_id"] == default["id"]


def test_pending_common_maturity_cannot_be_accelerated_by_omitting_long_holding_records():
    frozen, records, default, contender = setup()
    metadata = deepcopy(frozen["opportunity_universe"]["blocks"])
    metadata[-1]["outcome_available_at"] = "2025-02-01T22:00:00+00:00"
    for row in metadata[-1]["opportunities"]:
        row["outcome_available_at"] = "2025-02-01T22:00:00+00:00"
    frozen["opportunity_universe"] = freeze_opportunity_universe(metadata)
    records = [row for row in records if not (row["candidate_id"] == contender["id"]
                                              and row["block_id"] == "v3")]
    result = select(frozen, records)
    assert result["status"] == "insufficient" and result["selected_id"] == default["id"]
    assert result["selection_opportunities"] == {"declared": 8, "mature": 6}


class ForbiddenTest(dict):
    def get(self, key, default=None):
        if key not in {"candidate_id", "partition"}:
            pytest.fail("Test outcomes or metrics were read during parameter selection")
        return super().get(key, default)


def test_test_partition_is_ignored_before_reading_any_its_outcomes():
    frozen, records, _, _ = setup()
    expected = select(frozen, records)
    records.append(ForbiddenTest(candidate_id="baseline", partition="test"))
    actual = select(frozen, records)
    assert actual["selected_id"] == expected["selected_id"]
    assert actual["excluded"][-1]["reason"] == "test_or_unknown_partition_ignored"


class ForbiddenFuture(dict):
    def get(self, key, default=None):
        if key not in {"candidate_id", "partition", "block_id"}:
            pytest.fail("An unmatured comparison block's outcomes were read")
        return super().get(key, default)


def test_unmatured_train_validation_panels_are_gated_before_numeric_outcome_reads():
    frozen, records, _, _ = setup()
    metadata = deepcopy(frozen["opportunity_universe"]["blocks"])
    metadata[-1]["outcome_available_at"] = "2025-02-01T22:00:00Z"
    frozen["opportunity_universe"] = freeze_opportunity_universe(metadata)
    guarded = [ForbiddenFuture(row) if row["block_id"] == "v3" else row for row in records]
    assert select(frozen, guarded)["status"] == "insufficient"


def test_training_common_maturity_must_precede_first_validation_origin():
    frozen, records, default, _ = setup()
    metadata = deepcopy(frozen["opportunity_universe"]["blocks"])
    metadata[0]["outcome_available_at"] = "2024-04-10T22:00:00Z"
    for row in metadata[0]["opportunities"]:
        row["outcome_available_at"] = "2024-04-10T22:00:00Z"
    frozen["opportunity_universe"] = freeze_opportunity_universe(metadata)
    result = select(frozen, records)
    assert result["status"] == "insufficient" and result["selected_id"] == default["id"]
    assert all("missing_or_nonchronological_train_validation" in row["reasons"]
               for row in result["diagnostics"])


@pytest.mark.parametrize("change", ["capital", "risk", "overlap", "parameter_hash", "nav"])
def test_unsafe_or_unbound_records_cannot_promote_candidate(change):
    frozen, records, default, contender = setup()
    row = next(row for row in records if row["candidate_id"] == contender["id"]
               and row["block_id"] == "v1")
    if change == "capital":
        row["campaigns"][0]["new_capital_pct"] = 10.1
    elif change == "risk":
        row["campaigns"][0]["modeled_loss_pct"] = .51
    elif change == "overlap":
        row["campaigns"][1]["entered_at"] = row["campaigns"][0]["entered_at"]
    elif change == "parameter_hash":
        row["parameters_sha256"] = "changed"
    else:
        row["equity_curve"][-1]["equity"] = 999
    result = select(frozen, records)
    assert result["selected_id"] == default["id"] and result["excluded"]


def test_nav_drawdown_is_ordered_peak_to_trough_not_maximum_campaign_loss():
    frozen, records, _, contender = setup()
    row = next(row for row in records if row["candidate_id"] == contender["id"]
               and row["block_id"] == "v1")
    row["equity_curve"][2]["equity"] = 103
    row["equity_curve"][3]["equity"] = 99
    result = select(frozen, records)
    stats = next(row for row in result["diagnostics"] if row["candidate_id"] == contender["id"])
    assert stats["validation"]["max_drawdown_pct"] == pytest.approx(4/103*100)
    assert stats["passed"] is False


def test_same_close_entry_and_sparse_two_endpoint_nav_cannot_promote_candidate():
    for defect in ("same_close", "sparse_nav"):
        frozen, records, default, contender = setup()
        row = next(row for row in records if row["candidate_id"] == contender["id"]
                   and row["block_id"] == "v1")
        if defect == "same_close":
            row["campaigns"][0]["entered_at"] = row["campaigns"][0]["origin_at"]
        else:
            row["equity_curve"] = [row["equity_curve"][0], row["equity_curve"][-1]]
        result = select(frozen, records)
        assert result["selected_id"] == default["id"]
        reason = ("filled_trade_not_close_decision_next_session_open" if defect == "same_close"
                  else "missing_common_daily_nav_close_grid")
        assert any(entry["reason"] == reason for entry in result["excluded"])


def test_sparse_samples_return_default_not_fabricated_optimum_and_watch_uses_no_capital():
    frozen, records, default, contender = setup()
    assert select(frozen, records, 20)["status"] == "insufficient"
    for block_row in records:
        if block_row["candidate_id"] == contender["id"]:
            row = block_row["campaigns"][1]
            row.update(status="unfilled", net_account_pnl_pct=0, new_capital_pct=0,
                       entered_at=None, exited_at=None)
            block_row["equity_curve"][-1]["equity"] = 100.3
    result = select(frozen, records, 1)
    stats = next(row for row in result["diagnostics"] if row["candidate_id"] == contender["id"])
    assert stats["validation"]["opportunities"] == 6
    assert stats["validation"]["closed_campaigns"] == 3
    assert stats["validation"]["entry_coverage"] == .5
    assert stats["validation"]["opportunity_success"] == .5


def test_selection_requires_an_aware_clock():
    frozen, records, _, _ = setup()
    with pytest.raises(ValueError):
        select_swing_parameters(frozen, records, selection_at=datetime(2025, 1, 1))


def test_finite_external_navs_cannot_overflow_into_a_winning_aggregate():
    frozen, records, default, contender = setup()
    for row in records:
        if row["candidate_id"] == contender["id"]:
            row["campaigns"][0]["net_account_pnl_pct"] = 1e150
            row["campaigns"][1]["net_account_pnl_pct"] = 0
            for point in row["equity_curve"][3:]:
                point["equity"] = 1e150
    result = select(frozen, records)
    assert result["selected_id"] == default["id"]
    diagnostic = next(row for row in result["diagnostics"]
                      if row["candidate_id"] == contender["id"])
    assert "nonfinite_or_nonpositive_aggregate_nav" in diagnostic["reasons"]
