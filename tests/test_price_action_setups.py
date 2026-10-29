"""Synthetic daily bottom formation and simulator-contract checks only."""

import copy
from datetime import date, datetime, timedelta

import pytest

from twiz.price_action_setups import METHOD, _advance, _clock, generate_price_action_setups
from twiz.sessions import is_us_equity_session
from twiz.structure import _atr
from twiz.swing_optimizer import _hash, freeze_candidate_search
from twiz.swing_simulator import simulate_swing_book

POLICY = {
    "support_lookback_sessions": 20,
    "confirmation_sessions": 2,
    "min_pullback_atr": 2,
    "max_rebound_atr": 3,
    "watch_sessions": 5,
}


def candidate(**changes):
    return freeze_candidate_search({
        "protocol_version": "author-swing-v2",
        "defaults": {"event_variant": "price_only", "entry_rule": "close_hold",
                     "entry_ladder": "atr_approach", "confirm_sessions": 1,
                     "watch_sessions": 5, "holding_sessions": 3, **changes},
        "event_direction": {"variants": [changes.get("event_variant", "price_only")]},
        "selection": {"maximum_candidates": 1},
    })["candidates"][0]


def bars(n=280):
    days, day = [], date(2023, 1, 3)
    while len(days) < n:
        if is_us_equity_session(day):
            days.append(day)
        day += timedelta(days=1)
    result = [{"date": day.isoformat(), "open": 108, "high": 109, "low": 107,
               "close": 108, "volume": 1000, "source": "yfinance",
               "price_basis": "synthetic_test_consistent_adjusted_basis",
               "session_complete": True} for day in days]
    result[-3].update(open=101, high=103, low=98, close=99)
    result[-2].update(open=100, high=101, low=98.5, close=99.5)
    result[-1].update(open=99.5, high=101, low=99, close=100)
    return result


def generate(rows=None, *, policy=None, value=None, start=None, end=None, **kwargs):
    rows = rows if rows is not None else bars()
    return generate_price_action_setups(
        {"QQQ": rows}, start or rows[-1]["date"], end or rows[-1]["date"],
        value or candidate(), policy=policy or POLICY, **kwargs,
    )


def test_bottom_requires_two_subsequent_completed_days_not_candidate_hold_one():
    rows = bars()
    result = generate(rows, start=rows[-3]["date"])
    assert result["opportunity_count"] == 3 and result["ready_count"] == 1
    assert [row["status"] for row in result["dispositions"]] == [
        "nonqualifying", "nonqualifying", "ready"
    ]
    setup = result["setups"][0]
    proof = setup["causal_bottom"]
    assert proof["anchor_date"] == rows[-3]["date"]
    assert proof["anchor_age_sessions"] == 2
    assert proof["confirmation_clocks"] == [_clock(date.fromisoformat(row["date"]))
                                              for row in rows[-2:]]
    assert candidate()["parameters"]["confirm_sessions"] == 1
    assert setup["generation_parameters"]["confirmation_sessions"] == 2
    assert setup["origin_at"] == setup["available_at"] == setup["features_available_at"]
    assert setup["origin_at"] == proof["confirmation_clocks"][-1]
    assert setup["event_context"] is None
    assert setup["zone_lower"] == 98
    assert setup["zone_upper"] == pytest.approx(98 + .5 * _atr(rows, 14))
    assert setup["zone_center"] == pytest.approx(98 + .25 * _atr(rows, 14))
    assert setup["price_basis"] == rows[-1]["price_basis"]
    assert setup["candidate_id"] == candidate()["id"]
    assert setup["parameters_sha256"] == candidate()["parameters_sha256"]


def test_atr_is_actual_last_fourteen_true_ranges_with_previous_close():
    rows = bars()
    expected = sum(max(cur["high"] - cur["low"], abs(cur["high"] - prev["close"]),
                       abs(cur["low"] - prev["close"]))
                   for prev, cur in zip(rows[-15:-1], rows[-14:])) / 14
    result = generate(rows)
    assert result["setups"][0]["atr"] == expected
    assert result["setups"][0]["causal_bottom"]["pullback_atr"] == (109 - 100) / expected


@pytest.mark.parametrize("new_low", [97, 98])
def test_latest_equal_or_lower_low_restarts_confirmation_clock(new_low):
    rows = bars()
    rows[-1]["low"] = new_low
    result = generate(rows)
    assert result["ready_count"] == 0
    proof = result["dispositions"][0]["diagnostic"]
    assert proof["anchor_date"] == rows[-1]["date"] and proof["anchor_age_sessions"] == 0
    assert result["dispositions"][0]["reason"] == "anchor_old_enough"


def test_does_not_buy_when_final_close_is_still_falling():
    rows = bars()
    rows[-1]["close"] = 99.4
    result = generate(rows)
    assert result["ready_count"] == 0
    assert result["dispositions"][0]["reason"] == "final_close_rising"


def test_three_day_confirmation_is_independent_of_execution_hold():
    result = generate(policy={**POLICY, "confirmation_sessions": 3})
    assert result["ready_count"] == 0
    assert result["dispositions"][0]["reason"] == "anchor_old_enough"


def test_insufficient_pullback_from_prior_high_is_not_a_bottom_entry():
    result = generate(policy={**POLICY, "min_pullback_atr": 6})
    assert result["ready_count"] == 0
    assert result["dispositions"][0]["reason"] == "sufficient_pullback"


def test_recovered_price_is_not_qualified_just_because_old_anchor_was_low():
    rows = bars()
    rows[-1].update(open=108, high=110, close=109)
    result = generate(rows, policy={**POLICY, "max_rebound_atr": 4})
    assert result["ready_count"] == 0
    assert result["dispositions"][0]["reason"] == "sufficient_pullback"


def test_rebound_extension_gate_rejects_chasing_despite_large_prior_pullback():
    rows = bars()
    rows[-40]["high"] = 140
    rows[-1].update(open=104, high=107, close=106)
    result = generate(rows, policy={**POLICY, "max_rebound_atr": 1})
    assert result["ready_count"] == 0
    assert result["dispositions"][0]["reason"] == "rebound_not_extended"


def test_prior_high_excludes_origin_bar_high():
    rows = bars()
    rows[-1]["high"] = 160
    result = generate(rows)
    proof = result["dispositions"][0]["diagnostic"]
    assert proof["prior_high"] == 109
    assert proof["prior_high_date"] != rows[-1]["date"]


def test_anchor_is_selected_only_from_declared_support_lookback():
    rows = bars()
    rows[-25].update(open=90, high=95, low=80, close=92)
    small = generate(rows)
    large = generate(rows, policy={**POLICY, "support_lookback_sessions": 30})
    assert small["setups"][0]["causal_bottom"]["anchor_low"] == 98
    assert large["dispositions"][0]["diagnostic"]["anchor_low"] == 80
    assert large["ready_count"] == 0


def test_once_per_exact_anchor_asset_preserves_later_daily_dispositions():
    rows = bars()
    original_day = rows[-1]["date"]
    for close in (100.5, 101):
        day = _advance(date.fromisoformat(rows[-1]["date"]), 1)
        rows.append({**rows[-1], "date": day.isoformat(), "open": close - .2,
                     "low": close - .5, "high": close + .5, "close": close})
    result = generate(rows, start=original_day)
    assert result["ready_count"] == 1 and result["opportunity_count"] == 3
    assert [d["status"] for d in result["dispositions"]] == ["ready", "already_emitted",
                                                              "already_emitted"]
    assert result["latest_decision"]["QQQ"]["first_emitted_at"] == result["setups"][0]["origin_at"]


def test_same_anchor_date_different_assets_have_independent_setups():
    rows = bars()
    result = generate_price_action_setups(
        {"SPY": copy.deepcopy(rows), "QQQ": rows}, rows[-1]["date"], rows[-1]["date"],
        candidate(), policy=POLICY,
    )
    assert result["ready_count"] == result["opportunity_count"] == 2
    assert {s["asset"] for s in result["setups"]} == {"QQQ", "SPY"}


def test_causal_prefix_does_not_read_or_hash_future_numeric_values():
    class Future(dict):
        def get(self, key, *args):
            assert key == "date", f"Future numerical field read: {key}"
            return super().get(key, *args)

    rows = bars()
    before = generate(rows)
    future = Future(date=_advance(date.fromisoformat(rows[-1]["date"]), 1).isoformat(),
                    open=object(), close=object(), low=object(), high=object(), volume=object())
    after = generate(rows + [future], start=rows[-1]["date"], end=rows[-1]["date"])
    assert before == after


def test_source_policy_method_and_generation_hashes_are_bound_without_mutation():
    rows, value, policy = bars(), candidate(), copy.deepcopy(POLICY)
    original = copy.deepcopy((rows, value, policy))
    result = generate(rows, value=value, policy=policy)
    setup = result["setups"][0]
    assert (rows, value, policy) == original
    assert result["policy_sha256"] == _hash(POLICY)
    assert result["method_sha256"] == _hash(METHOD)
    assert setup["source_history_sha256"] == _hash(rows)
    assert set(setup["source_code_sha256"]) == {"price_action_setups.py", "structure.py",
                                               "sessions.py", "swing_optimizer.py"}
    changed = copy.deepcopy(rows)
    changed[0]["volume"] += 1
    other = generate(changed)["setups"][0]
    assert other["zone_lower"] == setup["zone_lower"]
    assert other["upstream_generation_hash"] != setup["upstream_generation_hash"]
    assert other["source_history_sha256"] != setup["source_history_sha256"]


@pytest.mark.parametrize("change,reason", [
    ({"session_complete": False}, "unverified_or_inconsistent_price_basis_or_sessions"),
    ({"price_basis": "unknown"}, "unverified_or_inconsistent_price_basis_or_sessions"),
    ({"price_basis": "different"}, "unverified_or_inconsistent_price_basis_or_sessions"),
    ({"source": "sample"}, "invalid_or_duplicate_predecision_price_observation"),
    ({"close": float("nan")}, "invalid_or_duplicate_predecision_price_observation"),
])
def test_invalid_completed_history_remains_an_unavailable_disposition(change, reason):
    rows = bars()
    rows[0].update(change)
    result = generate(rows)
    assert result["opportunity_count"] == 1 and result["ready_count"] == 0
    assert result["dispositions"][0]["status"] == "unavailable"
    assert result["dispositions"][0]["reason"] == reason


@pytest.mark.parametrize("mode", ["missing", "duplicate", "missing_origin"])
def test_history_must_have_contiguous_complete_unique_sessions(mode):
    rows = bars()
    end = rows[-1]["date"]
    if mode == "missing":
        del rows[-20]
    elif mode == "duplicate":
        rows.insert(1, copy.deepcopy(rows[0]))
    else:
        rows.pop()
    result = generate(rows, start=end, end=end)
    assert result["ready_count"] == 0
    assert result["dispositions"][0]["status"] == "unavailable"


def test_259_sessions_and_zero_atr_are_unavailable_not_fake_bottoms():
    short = generate(bars(259))
    assert short["dispositions"][0]["reason"] == "insufficient_predecision_history"
    rows = bars()
    for row in rows:
        row.update(open=100, high=100, low=100, close=100)
    result = generate(rows)
    assert result["dispositions"][0]["reason"] == "nonpositive_causal_atr"


def test_watch_and_comparison_maturity_use_sessions_and_keep_ny_clock():
    rows = bars()
    result = generate(rows)
    setup = result["setups"][0]
    origin_day = date.fromisoformat(rows[-1]["date"])
    assert setup["watch_expires_at"] == _clock(_advance(origin_day, 5))
    assert setup["comparison_available_at"] == _clock(_advance(origin_day, 5 + 3 + 1))
    assert setup["author_explicit_deadline"] is None


@pytest.mark.parametrize("key,value", [
    ("support_lookback_sessions", 9), ("support_lookback_sessions", 64),
    ("confirmation_sessions", 1), ("confirmation_sessions", 6),
    ("min_pullback_atr", .9), ("min_pullback_atr", 6.1),
    ("max_rebound_atr", .9), ("max_rebound_atr", 4.1),
    ("watch_sessions", 0), ("watch_sessions", 11),
    ("confirmation_sessions", True), ("min_pullback_atr", float("nan")),
])
def test_exact_policy_bounds_reject_invalid_values(key, value):
    with pytest.raises(ValueError, match="policy"):
        generate(policy={**POLICY, key: value})


@pytest.mark.parametrize("policy", [{}, {**POLICY, "unused": 1}])
def test_missing_or_unused_policy_keys_rejected(policy):
    with pytest.raises(ValueError, match="exactly"):
        generate_price_action_setups({}, "2024-01-02", "2024-01-02", candidate(), policy=policy)


def test_candidate_watch_mismatch_tamper_and_legacy_schema_rejected():
    with pytest.raises(ValueError, match="watch"):
        generate(value=candidate(watch_sessions=4))
    changed = candidate()
    changed["parameters"]["confirm_sessions"] = 2
    with pytest.raises(ValueError, match="bound"):
        generate(value=changed)
    legacy = freeze_candidate_search({})["candidates"][0]
    with pytest.raises(ValueError, match="version-two"):
        generate(value=legacy)


def test_missing_events_do_not_veto_the_generator_or_price_only_execution():
    history, value = bars(), candidate()
    generated = generate(history, value=value)
    setup = generated["setups"][0]
    assert setup["event_context"] is None
    first_reference = setup["zone_upper"] + setup["atr"]
    future = []
    for index in range(12):
        day = _advance(date.fromisoformat(history[-1]["date"]), index + 1)
        close = first_reference + .3 + index * .1
        future.append({**history[-1], "date": day.isoformat(), "open": close - .1,
                       "close": close, "high": close + .3, "low": first_reference - .2})
    result = simulate_swing_book(
        {"QQQ": history + future}, generated["setups"],
        execution_parameters=generated["execution_parameters"], candidate=value,
        candidate_id=value["id"], parameters_sha256=value["parameters_sha256"],
        block_id="synthetic-bottom", partition="validation", block_start=setup["origin_at"],
        block_end=_clock(date.fromisoformat(future[-1]["date"])), event_variant="price_only",
    )
    campaign = result["campaigns"][0]
    assert campaign["event_evidence"] == "price_only_ablation"
    assert campaign["fills"]
    first_fill = campaign["fills"][0]
    assert datetime.fromisoformat(first_fill["at"]) > datetime.fromisoformat(setup["origin_at"])
    assert first_fill["at"].startswith(future[1]["date"])
    signal_day = datetime.fromisoformat(campaign["entry_signal_at"]).date().isoformat()
    assert signal_day == future[0]["date"]


def test_auxiliary_event_variant_cannot_change_price_formation_or_fabricate_context():
    plain = generate()
    event_variant = generate(value=candidate(event_variant="learned_event_bridge"))
    assert plain["ready_count"] == event_variant["ready_count"] == 1
    assert event_variant["setups"][0]["event_context"] is None
    assert plain["setups"][0]["causal_bottom"] == event_variant["setups"][0]["causal_bottom"]
