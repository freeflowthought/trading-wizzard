"""Synthetic OHLC execution tests; no historical swing outcomes are inspected."""

import hashlib
import json
from datetime import date, datetime, timedelta

import pytest

from twiz.sessions import is_us_equity_session
from twiz.swing_optimizer import (
    CORE_KEYS,
    STRATEGY_KEYS,
    _validate_block,
    freeze_candidate_search,
)
from twiz.swing_simulator import _clock, _signal, simulate_swing_book

BASIS = "synthetic_test_split_consistent_basis"


def fixture(n=30):
    dates, cursor = [], date(2024, 1, 2)
    while len(dates) < n:
        if is_us_equity_session(cursor):
            dates.append(cursor)
        cursor += timedelta(days=1)
    bars = [
        {
            "date": day.isoformat(),
            "open": 101.0,
            "high": 102.0,
            "low": 100.6,
            "close": 101.0,
            "volume": 1000,
            "source": "Yahoo chart",
            "session_complete": True,
            "price_basis": BASIS,
        }
        for day in dates
    ]
    bars[1]["low"], bars[2]["low"], bars[3]["open"] = 99, 100.2, 102
    setup = {
        "opportunity_id": "case-a",
        "asset": "QQQ",
        "origin_at": _clock(dates[0]).isoformat(),
        "available_at": _clock(dates[1], True).isoformat(),
        "features_available_at": _clock(dates[0]).isoformat(),
        "watch_expires_at": _clock(dates[15]).isoformat(),
        "zone_lower": 99,
        "zone_upper": 101,
        "atr": 2,
        "price_basis": BASIS,
        "upstream_generation_hash": "frozen-origin-only-features",
        "author_explicit_deadline": None,
    }
    return dates, {"QQQ": bars}, setup


def change(market, i, **values):
    row = market["QQQ"][i]
    row.update(values)
    row["high"] = max(row["high"], row["open"], row["close"], row["low"])
    row["low"] = min(row["low"], row["open"], row["close"])


def run(market=None, setup=None, dates=None, params=None, **kwargs):
    if market is None:
        dates, market, setup = fixture()
    parameters = {"confirm_sessions": 1, "holding_sessions": 3, **(params or {})}
    variant = kwargs.get("event_variant", "price_only")
    candidate = freeze_candidate_search(
        {
            "protocol_version": (
                "author-swing-v2" if set(parameters) & STRATEGY_KEYS else "author-swing-v1"
            ),
            "defaults": {
                **{key: value for key, value in parameters.items()
                   if key in CORE_KEYS | STRATEGY_KEYS},
                "event_variant": variant,
            },
            "event_direction": {"variants": [variant]},
            "selection": {"maximum_candidates": 1},
        }
    )["candidates"][0]
    bound_setups = [
        {
            **raw,
            "candidate_id": candidate["id"],
            "parameters_sha256": candidate["parameters_sha256"],
        }
        for raw in (setup if isinstance(setup, list) else [setup])
    ]
    return simulate_swing_book(
        market,
        bound_setups,
        execution_parameters=parameters,
        candidate=candidate,
        candidate_id=candidate["id"],
        parameters_sha256=candidate["parameters_sha256"],
        block_id="synthetic",
        partition="validation",
        block_start=_clock(dates[0], True),
        block_end=_clock(dates[-1]),
        **kwargs,
    )


def test_touch_reclaim_later_retest_then_next_open_actual_price_and_costs():
    dates, market, setup = fixture()
    result = run(market, setup, dates)
    trade = result["campaigns"][0]
    assert trade["status"] == "closed"
    buy, sell = trade["fills"]
    assert buy["at"] == _clock(dates[3], True).isoformat()
    assert trade["entry_signal_at"] == _clock(dates[2]).isoformat()
    assert buy["price"] == pytest.approx(102 * 1.0002)
    assert buy["fee"] == pytest.approx(buy["quantity"] * buy["price"] * 0.0001)
    assert sell["at"] == _clock(dates[6], True).isoformat()
    assert sell["reason"] == "holding_deadline_close"
    expected = sell["quantity"] * sell["price"] - sell["fee"] - 4
    assert trade["net_account_pnl_pct"] == pytest.approx(expected)
    assert result["equity_curve"][-1]["equity"] == pytest.approx(100 + expected)
    assert result["evaluation_basis"] == "price_based_daily_close_next_open_simulator"
    _validate_block(
        result, {"parameters_sha256": result["parameters_sha256"]}, result["execution_policy_id"]
    )
    json.dumps(result, allow_nan=False)


def test_holding_and_confirmation_parameters_change_actual_fill_dates():
    dates, market, setup = fixture()
    quick = run(market, setup, dates)
    slower = run(market, setup, dates, params={"confirm_sessions": 3, "holding_sessions": 5})
    assert quick["campaigns"][0]["entered_at"] == _clock(dates[3], True).isoformat()
    assert slower["campaigns"][0]["entered_at"] == _clock(dates[5], True).isoformat()
    assert slower["campaigns"][0]["exited_at"] == _clock(dates[10], True).isoformat()


def test_daily_close_stop_ignores_intraday_low_and_executes_next_open_gap():
    dates, market, setup = fixture()
    change(market, 4, low=90, close=101)
    change(market, 5, close=97)
    change(market, 6, open=85, low=84, close=90)
    result = run(market, setup, dates, params={"holding_sessions": 10})
    trade = result["campaigns"][0]
    assert trade["exit_signal_at"] == _clock(dates[5]).isoformat()
    sell = trade["fills"][-1]
    assert sell["at"] == _clock(dates[6], True).isoformat()
    assert sell["price"] == pytest.approx(85 * 0.9998)
    assert sell["reason"] == "close_stop"
    assert trade["net_account_pnl_pct"] < -0.5  # Modeled .5% is not a gap guarantee.
    assert trade["modeled_loss_pct"] <= 0.5 and trade["new_capital_pct"] <= 10


@pytest.mark.parametrize("trims", [[0.4, 0.3, 0.3], [0.5, 0.3, 0.2], [0.3, 0.3, 0.4]])
def test_original_quantity_partial_trims_after_three_independently_confirmed_batches(trims):
    dates, market, setup = fixture()
    setup["target_prices"] = [104, 106, 108]
    for i, values in {
        4: {"low": 99.9, "close": 100.8},
        5: {"low": 99.9, "close": 100.8},
        7: {"low": 99.2, "close": 100},
        8: {"low": 99.4, "close": 100},
        9: {"open": 100.2, "close": 105},
        10: {"open": 107, "close": 107},
        11: {"open": 106, "close": 109},
        12: {"open": 109, "close": 109},
    }.items():
        change(market, i, **values)
    result = run(market, setup, dates, params={"holding_sessions": 20, "trim_fractions": trims})
    trade = result["campaigns"][0]
    buys = [fill for fill in trade["fills"] if fill["side"] == "buy"]
    sells = [fill for fill in trade["fills"] if fill["side"] == "sell"]
    assert [fill["batch"] for fill in buys] == [1, 2, 3]
    assert [fill["at"] for fill in buys] == [_clock(dates[i], True).isoformat() for i in (3, 6, 9)]
    original = sum(fill["quantity"] for fill in buys)
    assert [fill["quantity"] / original for fill in sells] == pytest.approx(trims)
    assert trade["status"] == "closed" and trade["new_capital_pct"] == pytest.approx(10)
    assert trade["modeled_loss_pct"] <= 0.5 + 1e-9


def test_first_trim_cancels_later_adds_even_if_lower_zones_reclaim_again():
    dates, market, setup = fixture()
    setup["target_prices"] = [104, 106, 108]
    change(market, 3, close=105)
    change(market, 4, open=106, close=100, low=99.8)
    change(market, 5, close=100, low=99.8)
    change(market, 6, close=97)
    change(market, 7, open=90, close=90)
    result = run(market, setup, dates, params={"holding_sessions": 20})
    fills = result["campaigns"][0]["fills"]
    assert sum(fill["side"] == "buy" for fill in fills) == 1
    assert [fill["reason"] for fill in fills if fill["side"] == "sell"] == [
        "close_take_profit_1",
        "close_stop",
    ]


def test_fifo_shared_book_blocks_other_ready_campaign_and_watch_does_not_reserve_cash():
    dates, market, setup = fixture()
    other = {**setup, "opportunity_id": "case-b"}
    result = run(market, [other, setup], dates)
    by_id = {row["opportunity_id"]: row for row in result["campaigns"]}
    assert by_id["case-a"]["status"] == "closed"
    assert by_id["case-b"]["status"] == "blocked" and by_id["case-b"]["new_capital_pct"] == 0
    assert by_id["case-b"]["net_account_pnl_pct"] == 0
    market["AAA"] = [
        {**bar, "low": 110, "open": 111, "close": 111, "high": 112} for bar in market["QQQ"]
    ]
    watching = {**setup, "opportunity_id": "earlier-watch", "asset": "AAA"}
    result = run(market, [watching, setup], dates)
    by_id = {row["opportunity_id"]: row for row in result["campaigns"]}
    assert by_id["case-a"]["status"] == "closed"
    assert by_id["earlier-watch"]["status"] == "expired"
    assert by_id["earlier-watch"]["new_capital_pct"] == 0


def test_position_or_ready_signal_at_end_is_pending_not_forced_closed_or_won():
    dates, market, setup = fixture(30)
    result = run(market, setup, dates[:5], params={"holding_sessions": 20})
    trade = result["campaigns"][0]
    assert trade["status"] == "pending" and trade["net_account_pnl_pct"] is None
    assert result["status"] == "pending"
    assert result["equity_curve"][-1]["equity"] != 100
    assert not any(fill["side"] == "sell" for fill in trade["fills"])


def test_missing_required_next_open_does_not_substitute_a_later_quote():
    dates, market, setup = fixture()
    market["QQQ"].pop(6)  # Holding-three close exit needs exactly this next open.
    result = run(market, setup, dates)
    assert result["status"] == "incomplete" and result["equity_curve"] == []
    assert not any(fill["side"] == "sell" for fill in result["campaigns"][0]["fills"])
    assert result["partial_equity_curve"]


def test_future_bars_cannot_change_existing_triggers_fills_or_nav():
    dates, market, setup = fixture()
    expected = run(market, setup, dates)
    future = {
        **market["QQQ"][-1],
        "date": "2025-01-02",
        "open": 10000,
        "close": 10000,
        "high": 10001,
        "low": 9999,
    }
    assert run({"QQQ": market["QQQ"] + [future]}, setup, dates) == expected


def test_learned_variant_never_invents_probability_and_threshold_changes_entry_authority():
    dates, market, setup = fixture()
    missing = run(market, setup, dates, event_variant="learned_event_bridge")
    assert missing["campaigns"][0]["status"] == "unavailable"
    assert missing["campaigns"][0]["fills"] == []
    setup["event_context"] = {
        "available_at": setup["origin_at"],
        "probability_estimate_eligible": True,
        "equity_up_probability": 0.75,
        "event_threshold": 0.65,
        "main_case_only": False,
        "model_id": "past-fit",
    }
    supported = run(market, setup, dates, event_variant="learned_event_bridge")
    blocked = run(
        market,
        setup,
        dates,
        event_variant="learned_event_bridge",
        params={"trade_probability_threshold": 0.85},
    )
    assert supported["campaigns"][0]["status"] == "closed"
    assert blocked["campaigns"][0]["status"] == "unfilled"
    assert blocked["campaigns"][0]["fills"] == []


def test_zone_width_stop_takeprofit_and_fraction_parameters_have_observable_effects():
    dates, market, setup = fixture()
    change(market, 1, close=100.5)
    change(market, 2, close=100.5)
    narrow = run(market, setup, dates, params={"zone_width_atr": 0.25})
    wide = run(market, setup, dates, params={"zone_width_atr": 1})
    assert narrow["campaigns"][0]["entered_at"] != wide["campaigns"][0]["entered_at"]
    dates, market, setup = fixture()
    change(market, 4, close=97.8)
    close_stop = run(market, setup, dates, params={"stop_atr": 0.75, "holding_sessions": 10})
    farther = run(market, setup, dates, params={"stop_atr": 1.5, "holding_sessions": 10})
    assert close_stop["campaigns"][0]["exited_at"] != farther["campaigns"][0]["exited_at"]
    dates, market, setup = fixture()
    change(market, 4, close=110)
    near_tp = run(market, setup, dates, params={"take_profit_r": 1.5, "holding_sessions": 10})
    far_tp = run(market, setup, dates, params={"take_profit_r": 3, "holding_sessions": 10})
    assert len(near_tp["campaigns"][0]["fills"]) > len(far_tp["campaigns"][0]["fills"])
    small = run(market, setup, dates, params={"entry_fractions": [0.25, 0.35, 0.4]})
    large = run(market, setup, dates, params={"entry_fractions": [0.5, 0.3, 0.2]})
    assert (
        small["campaigns"][0]["fills"][0]["quantity"]
        < large["campaigns"][0]["fills"][0]["quantity"]
    )


def test_cross_split_is_incomplete_and_unknown_author_deadline_stays_unknown():
    dates, market, setup = fixture()
    result = run(
        market,
        setup,
        dates,
        corporate_actions=[
            {"ticker": "QQQ", "type": "split", "effective_date": dates[5].isoformat(), "ratio": 2}
        ],
    )
    assert result["status"] == "incomplete" and result["equity_curve"] == []
    assert result["campaigns"][0]["status"] == "unavailable"
    assert result["campaigns"][0]["author_explicit_deadline"] is None
    assert result["campaigns"][0]["literal_forecast_status"] == "not_scored_by_trade_simulator"


def test_upstream_unavailable_opportunities_remain_in_denominator_without_funding():
    dates, market, setup = fixture()
    unavailable = {
        "opportunity_id": "no-history",
        "asset": "IPO",
        "origin_at": setup["origin_at"],
        "comparison_available_at": _clock(dates[-1]).isoformat(),
        "reason": "short_history",
    }
    result = run(market, setup, dates, unavailable_opportunities=[unavailable])
    assert len(result["campaigns"]) == 2
    record = result["campaigns"][1]
    assert record["status"] == "unavailable" and record["new_capital_pct"] == 0
    assert record["reasons"] == ["short_history"]


def test_full_frozen_candidate_binds_execution_values_and_allows_named_baseline_alias():
    dates, market, setup = fixture()
    candidate = freeze_candidate_search(
        {
            "defaults": {
                "confirm_sessions": 3,
                "holding_sessions": 3,
                "event_variant": "price_only",
            },
            "event_direction": {"variants": ["price_only"]},
            "selection": {"maximum_candidates": 1},
        }
    )["candidates"][0]
    arguments = {
        "candidate_id": "baseline",
        "parameters_sha256": candidate["parameters_sha256"],
        "block_id": "bound",
        "partition": "validation",
        "block_start": _clock(dates[0], True),
        "block_end": _clock(dates[-1]),
        "event_variant": "price_only",
    }
    setup = {
        **setup,
        "candidate_id": candidate["id"],
        "parameters_sha256": candidate["parameters_sha256"],
    }
    with pytest.raises(ValueError, match="full frozen candidate"):
        simulate_swing_book(market, [setup], **arguments)
    with pytest.raises(ValueError, match="differ from the frozen"):
        simulate_swing_book(
            market,
            [setup],
            candidate=candidate,
            execution_parameters={"confirm_sessions": 1},
            **arguments,
        )
    result = simulate_swing_book(market, [setup], candidate=candidate, **arguments)
    assert result["candidate_id"] == "baseline" and result["frozen_candidate_id"] == candidate["id"]
    assert result["candidate_binding_verified"] is True
    assert result["campaigns"][0]["entered_at"] == _clock(dates[5], True).isoformat()
    with pytest.raises(ValueError, match="setup belongs"):
        simulate_swing_book(
            market, [{**setup, "candidate_id": "other-candidate"}], candidate=candidate, **arguments
        )


def test_signed_event_missing_is_unavailable_and_known_bearish_evidence_is_unfilled():
    dates, market, setup = fixture()
    missing = run(market, setup, dates, event_variant="signed_weighted_event_score")
    assert missing["campaigns"][0]["event_evidence"]["status"] == "unavailable"
    assert missing["campaigns"][0]["status"] == "unavailable"
    setup["event_context"] = {
        "channel_weights": {"rates": 1},
        "components": [
            {
                "event_id": "synthetic-event",
                "risk_group": "policy",
                "channel": "rates",
                "yes_probability": 0.1,
                "no_probability": 0.9,
                "yes_equity_effect": 1,
                "no_equity_effect": -1,
                "available_at": setup["origin_at"],
                "end_at": _clock(dates[10]).isoformat(),
                "source_ref": "synthetic-archive",
            }
        ],
    }
    bearish = run(market, setup, dates, event_variant="signed_weighted_event_score")
    assert bearish["campaigns"][0]["event_evidence"]["status"] == "ready"
    assert bearish["campaigns"][0]["status"] == "unfilled"


def test_unfunded_watch_expiry_does_not_require_quotes_after_expiry():
    dates, market, setup = fixture()
    for row in market["QQQ"]:
        row.update(open=111, high=112, low=110, close=111)
    market["QQQ"] = market["QQQ"][:16]
    result = run(market, setup, dates)
    assert result["campaigns"][0]["status"] == "expired"
    assert result["status"] == "complete" and result["equity_curve"][-1]["equity"] == 100


def test_watch_deadline_cancels_unfilled_adds_but_keeps_position_risk_management():
    dates, market, setup = fixture()
    setup["watch_expires_at"] = _clock(dates[4]).isoformat()
    # First batch enters on day3. A new center reclaim/retest after day4 cannot fund an add.
    change(market, 5, low=99.9, close=100.8)
    change(market, 6, low=99.9, close=100.8)
    result = run(market, setup, dates, params={"holding_sessions": 10})
    trade = result["campaigns"][0]
    assert trade["status"] == "closed"
    assert sum(fill["side"] == "buy" for fill in trade["fills"]) == 1
    assert trade["fills"][-1]["reason"] == "holding_deadline_close"


def test_future_model_fit_time_cannot_hide_behind_an_earlier_probability_available_time():
    dates, market, setup = fixture()
    setup["event_context"] = {
        "available_at": setup["origin_at"],
        "probability_estimate_eligible": True,
        "equity_up_probability": 0.9,
        "event_threshold": 0.65,
        "main_case_only": False,
        "model_fit_at": _clock(dates[5]).isoformat(),
    }
    result = run(market, setup, dates, event_variant="learned_event_bridge")
    assert result["campaigns"][0]["status"] == "unavailable"


@pytest.mark.parametrize("first_open", [106.903, 110.5, 120.123])
def test_scaled_risk_and_capital_remain_below_exact_sealed_selector_ceilings(first_open):
    dates, market, setup = fixture()
    change(market, 3, open=first_open)
    for i, values in {
        4: {"low": 99.9, "close": 100.8},
        5: {"low": 99.9, "close": 100.8},
        7: {"low": 99.2, "close": 100},
        8: {"low": 99.4, "close": 100},
    }.items():
        change(market, i, **values)
    result = run(market, setup, dates, params={"holding_sessions": 15})
    trade = result["campaigns"][0]
    assert trade["status"] == "closed"
    assert trade["modeled_loss_pct"] <= 0.5
    assert trade["new_capital_pct"] <= 10
    _validate_block(
        result, {"parameters_sha256": result["parameters_sha256"]}, result["execution_policy_id"]
    )


@pytest.mark.parametrize(
    "params", [{"lookback_sessions": 63}, {"unknown_parameter": 1}, {"event_threshold": 0.5}]
)
def test_unsupported_generation_knobs_or_invalid_execution_values_raise(params):
    with pytest.raises(ValueError):
        run(params=params)


def test_v2_close_hold_fills_before_legacy_retest_without_changing_v1():
    dates, market, setup = fixture()
    legacy = run(market, setup, dates)["campaigns"][0]
    held = run(market, setup, dates, params={"entry_rule": "close_hold"})["campaigns"][0]
    assert legacy["entered_at"] == _clock(dates[3], True).isoformat()
    assert held["entered_at"] == _clock(dates[2], True).isoformat()
    assert held["entry_signal_at"] == _clock(dates[1]).isoformat()


def test_v2_confirmation_resets_on_failed_close_and_later_adds_need_their_own_signal():
    dates, market, setup = fixture()
    change(market, 2, close=100)
    change(market, 4, low=100, close=101)
    held = run(market, setup, dates, params={
        "entry_rule": "close_hold", "confirm_sessions": 2})["campaigns"][0]
    assert held["entered_at"] == _clock(dates[6], True).isoformat()
    dates, market, setup = fixture()
    for i, values in {
        1: {"open": 103, "low": 102, "close": 103},
        2: {"open": 103, "close": 101},
        3: {"open": 102, "low": 101, "close": 102},
        4: {"open": 102, "low": 100, "close": 101},
    }.items():
        change(market, i, **values)
    result = run(market, setup, dates, params={
        "entry_rule": "close_hold", "entry_ladder": "atr_approach", "holding_sessions": 10})
    trade = result["campaigns"][0]
    buys = [fill for fill in trade["fills"] if fill["side"] == "buy"]
    assert trade["applied_entry_references"] == [102.5, 101.5, 100.5]
    assert [fill["at"] for fill in buys] == [
        _clock(dates[i], True).isoformat() for i in (2, 4, 5)]
    assert trade["modeled_loss_pct"] <= .5 and trade["new_capital_pct"] <= 10
    _validate_block(result, {"parameters_sha256": result["parameters_sha256"]},
                    result["execution_policy_id"])


def test_v2_sweep_requires_strict_low_and_same_bar_reclaim_then_completed_holds():
    state = {"phase": "watch"}
    bar = {"low": 100, "high": 102, "close": 101}
    assert not _signal(state, bar, 100, 99, 1, 1, "sweep_reclaim")
    assert not _signal(state, {**bar, "low": 99, "close": 99.5},
                       100, 99, 2, 1, "sweep_reclaim")
    assert not _signal(state, bar, 100, 99, 3, 1, "sweep_reclaim")
    assert not _signal(state, {**bar, "low": 99}, 100, 99, 4, 2, "sweep_reclaim")
    assert _signal(state, bar, 100, 99, 5, 2, "sweep_reclaim")


def test_v2_may_use_known_origin_close_but_never_a_bar_before_availability():
    dates, market, setup = fixture()
    setup["available_at"] = setup["origin_at"]
    change(market, 0, low=99)
    v2 = run(market, setup, dates, params={"entry_rule": "close_hold"})["campaigns"][0]
    v1 = run(market, setup, dates)["campaigns"][0]
    assert v2["entry_signal_at"] == setup["origin_at"]
    assert v2["entered_at"] == _clock(dates[1], True).isoformat()
    assert v1["entered_at"] == _clock(dates[3], True).isoformat()
    setup["available_at"] = _clock(dates[2]).isoformat()
    delayed = run(market, setup, dates, params={"entry_rule": "close_hold"})["campaigns"][0]
    assert delayed["entry_signal_at"] == _clock(dates[2]).isoformat()
    assert delayed["entered_at"] == _clock(dates[3], True).isoformat()


def test_v2_next_open_chase_cap_cancels_first_or_add_without_erasing_funded_position():
    dates, market, setup = fixture()
    change(market, 3, open=110)
    legacy = run(market, setup, dates)["campaigns"][0]
    capped = run(market, setup, dates, params={
        "entry_rule": "retest", "max_entry_extension_atr": .5})["campaigns"][0]
    assert legacy["entered_at"] is not None
    assert capped["status"] == "cancelled" and capped["fills"] == []
    assert any("extension" in reason for reason in capped["reasons"])
    dates, market, setup = fixture()
    change(market, 3, low=99.9, close=100.8)
    change(market, 4, open=104)
    trade = run(market, setup, dates, params={
        "entry_rule": "close_hold", "max_entry_extension_atr": .5,
        "holding_sessions": 8})["campaigns"][0]
    assert trade["status"] == "closed"
    assert sum(fill["side"] == "buy" for fill in trade["fills"]) == 1
    assert trade["fills"][-1]["reason"] == "holding_deadline_close"
    assert any("extension" in reason for reason in trade["reasons"])


def test_v2_watch_shortens_from_origin_not_delayed_availability_and_maturity_stays_frozen():
    dates, market, setup = fixture()
    for row in market["QQQ"][:6]:
        row.update(open=111, high=112, low=110, close=111)
    change(market, 6, low=99, close=101)
    setup["available_at"] = _clock(dates[4], True).isoformat()
    short = run(market, setup, dates, params={
        "entry_rule": "close_hold", "watch_sessions": 5})["campaigns"][0]
    longer = run(market, setup, dates, params={
        "entry_rule": "close_hold", "watch_sessions": 10})["campaigns"][0]
    assert short["status"] == "expired" and short["fills"] == []
    assert longer["entered_at"] == _clock(dates[7], True).isoformat()
    assert short["effective_watch_expires_at"] == _clock(dates[5]).isoformat()
    assert short["outcome_available_at"] == longer["outcome_available_at"]
    setup["watch_expires_at"] = _clock(dates[3]).isoformat()
    setup["available_at"] = setup["origin_at"]
    original = run(market, setup, dates, params={
        "entry_rule": "close_hold", "watch_sessions": 20})["campaigns"][0]
    assert datetime.fromisoformat(original["effective_watch_expires_at"]) == (
        datetime.fromisoformat(setup["watch_expires_at"]))


def test_v2_deadline_close_signal_fills_next_open_and_short_watch_blocks_later_adds():
    dates, market, setup = fixture()
    for row in market["QQQ"][:5]:
        row.update(open=111, high=112, low=110, close=111)
    change(market, 5, low=99, close=101)
    change(market, 6, low=99, close=101)
    change(market, 7, low=99, close=101)
    trade = run(market, setup, dates, params={
        "entry_rule": "close_hold", "watch_sessions": 5,
        "holding_sessions": 8})["campaigns"][0]
    assert trade["entered_at"] == _clock(dates[6], True).isoformat()
    assert sum(fill["side"] == "buy" for fill in trade["fills"]) == 1
    assert trade["status"] == "closed"


EXIT_POLICY = {"first_trim_r": .5, "trailing_atr": 1., "resistance_lookback_sessions": 10}


def test_disabled_exit_policy_preserves_captured_v1_v2_outputs_byte_for_byte():
    outputs = []
    for params in ({}, {"entry_rule": "close_hold"}, {"entry_rule": "sweep_reclaim"},
                   {"entry_rule": "close_hold", "entry_ladder": "atr_approach"}):
        result = run(params=params)
        assert result == run(params=params, exit_policy=None)
        assert "exit_policy_sha256" not in result
        assert "stop_revisions" not in result["campaigns"][0]
        outputs.append(result)
    body = json.dumps(outputs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(body).hexdigest() == (
        "a736aca1a27fc8cbb540ba98bf6140945a8d144da9104a2d2cd18e8fce3c1928")


@pytest.mark.parametrize("policy", [
    {}, {**EXIT_POLICY, "undeclared": 1}, {**EXIT_POLICY, "first_trim_r": True},
    {**EXIT_POLICY, "first_trim_r": .49}, {**EXIT_POLICY, "first_trim_r": 2.01},
    {**EXIT_POLICY, "trailing_atr": .99}, {**EXIT_POLICY, "trailing_atr": float("nan")},
    {**EXIT_POLICY, "resistance_lookback_sessions": 9},
    {**EXIT_POLICY, "resistance_lookback_sessions": 64},
    {**EXIT_POLICY, "resistance_lookback_sessions": 10.0},
])
def test_exit_policy_requires_complete_finite_typed_bounded_parameters(policy):
    with pytest.raises(ValueError):
        run(exit_policy=policy)


def test_early_trim_needs_one_close_then_fill_before_protection_and_gap_exit():
    dates, market, setup = fixture()
    change(market, 5, close=104)
    change(market, 6, open=104, close=103)
    change(market, 7, close=101)
    change(market, 8, open=70, close=75)
    params = {"confirm_sessions": 3, "holding_sessions": 20}
    waiting = run(market, setup, dates[:6], params=params, exit_policy=EXIT_POLICY)["campaigns"][0]
    assert waiting["entered_at"] == _clock(dates[5], True).isoformat()
    assert waiting["first_trim_filled_at"] is None and waiting["stop_revisions"] == []
    result = run(market, setup, dates, params=params, exit_policy=EXIT_POLICY)
    trade = result["campaigns"][0]
    buy, trim, stop = trade["fills"]
    assert trim["signal_at"] == _clock(dates[5]).isoformat()  # No three-day top confirmation.
    assert trim["at"] == trade["first_trim_filled_at"] == _clock(dates[6], True).isoformat()
    assert trim["quantity"] / buy["quantity"] == pytest.approx(.4)
    assert stop["quantity"] / buy["quantity"] == pytest.approx(.6)
    assert trade["stop_revisions"][0]["at"] == _clock(dates[6]).isoformat()
    assert stop["reason"] == "close_stop" and stop["at"] == _clock(dates[8], True).isoformat()
    assert stop["price"] == pytest.approx(70 * .9998)
    assert trade["net_account_pnl_pct"] < -.5  # Gaps can exceed modeled account risk.
    assert trade["modeled_loss_pct"] <= .5 and trade["new_capital_pct"] <= 10
    _validate_block(result, {"parameters_sha256": result["parameters_sha256"]},
                    result["execution_policy_id"])


def test_target_uses_only_pre_entry_highs_in_last_n_actual_sessions():
    dates, market, setup = fixture(40)
    setup["watch_expires_at"] = _clock(dates[30]).isoformat()
    change(market, 1, low=100.6)
    change(market, 2, low=100.6)
    change(market, 0, high=103.2)  # Closer, but outside ten preceding US sessions.
    change(market, 15, high=104.1)
    change(market, 18, low=99)
    change(market, 19, low=100.2)
    change(market, 20, open=102, high=108)  # Entry-day high is not known at its open.
    change(market, 21, high=103.3)
    trade = run(market, setup, dates, params={"holding_sessions": 10},
                exit_policy=EXIT_POLICY)["campaigns"][0]
    revision = trade["target_revisions"][0]
    assert trade["entered_at"] == _clock(dates[20], True).isoformat()
    assert revision["basis"] == "known_prior_completed_high"
    assert revision["source_date"] == dates[15].isoformat()
    assert revision["new_targets"][0] == 104.1
    assert revision["new_targets"][1:] == revision["old_targets"][1:]
    assert revision["source_available_at"] == _clock(dates[15]).isoformat()
    assert revision["inputs_available_at"] == trade["entered_at"]
    change(market, 21, high=200)
    changed = run(market, setup, dates, params={"holding_sessions": 10},
                  exit_policy=EXIT_POLICY)["campaigns"][0]
    assert changed["target_revisions"] == trade["target_revisions"]


def test_campaign_fee_adjusted_breakeven_and_trailing_stop_never_loosen():
    dates, market, setup = fixture()
    change(market, 3, close=105)
    change(market, 4, open=103, close=104)
    change(market, 5, close=106)
    change(market, 6, close=105)
    policy = {**EXIT_POLICY, "trailing_atr": 4.}
    trade = run(market, setup, dates, params={"holding_sessions": 12},
                exit_policy=policy)["campaigns"][0]
    buy, trim = trade["fills"][:2]
    remaining = buy["quantity"] - trim["quantity"]
    proceeds = trim["quantity"] * trim["price"] - trim["fee"]
    expected = (trade["cash_committed"] - proceeds) / (remaining * .9998 * .9999)
    first = trade["stop_revisions"][0]
    assert first["campaign_net_breakeven"] == pytest.approx(expected)
    assert first["new_stop"] == pytest.approx(max(
        trade["original_stop_price"], expected, first["trailing_price"]))
    stops = [item["new_stop"] for item in trade["stop_revisions"]]
    assert stops == sorted(stops)
    assert all(datetime.fromisoformat(item["at"]) >= datetime.fromisoformat(
        trade["first_trim_filled_at"]) for item in trade["stop_revisions"])
    assert sum(fill["side"] == "buy" for fill in trade["fills"]) == 1


def test_exit_treatment_has_separate_immutable_identity_and_preserves_explicit_targets():
    dates, market, setup = fixture()
    setup["target_prices"] = [104, 106, 108]
    policy = dict(EXIT_POLICY)
    baseline = run(market, setup, dates, params={"holding_sessions": 10})
    treated = run(market, setup, dates, params={"holding_sessions": 10}, exit_policy=policy)
    assert policy == EXIT_POLICY
    assert treated["parameters_sha256"] == baseline["parameters_sha256"]
    assert treated["candidate_parameters_include_exit_policy"] is False
    trade = treated["campaigns"][0]
    assert trade["exit_policy_sha256"] == treated["exit_policy_sha256"]
    assert trade["applied_target_prices"] == setup["target_prices"]
    assert trade["target_revisions"][0]["basis"] == "explicit_targets_preserved"
    other = run(market, setup, dates, params={"holding_sessions": 10},
                exit_policy={**policy, "trailing_atr": 3})
    assert other["exit_policy_sha256"] != treated["exit_policy_sha256"]
    json.dumps(treated, allow_nan=False)


def test_none_and_three_key_exit_policy_keep_pre_retention_outputs_byte_exact():
    outputs = []
    for params in ({}, {"entry_rule": "close_hold"}, {"entry_rule": "sweep_reclaim"},
                   {"entry_rule": "close_hold", "entry_ladder": "atr_approach"}):
        for policy in (None, EXIT_POLICY):
            outputs.append(run(params=params, exit_policy=policy))
    body = json.dumps(outputs, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(body).hexdigest() == (
        "32ae7e6f39b75b90216b1024e9f1e11d272dfab914a8fad14aef77f400caf582")


@pytest.mark.parametrize("retention", [None, True, "0.5", -.001, 1.001, float("nan"), float("inf")])
def test_retention_requires_finite_fraction_in_closed_unit_interval(retention):
    with pytest.raises(ValueError, match="profit retention"):
        run(exit_policy={**EXIT_POLICY, "minimum_profit_retention": retention})


def retention_scenario():
    dates, market, setup = fixture()
    setup["target_prices"] = [104, 106, 108]
    for i, values in {
        4: {"low": 99.9, "close": 100.8}, 5: {"low": 99.9, "close": 100.8},
        6: {"open": 100.4}, 7: {"low": 99.2, "close": 100},
        8: {"low": 99.4, "close": 100}, 9: {"open": 100.2, "close": 103},
        10: {"close": 105}, 11: {"open": 105, "close": 104},
        12: {"close": 103}, 13: {"close": 101}, 14: {"open": 99},
    }.items():
        change(market, i, **values)
    return dates, market, setup


def test_retention_uses_net_peak_after_adds_and_partial_sale_and_gap_can_breach_goal():
    dates, market, setup = retention_scenario()
    policy = {**EXIT_POLICY, "trailing_atr": 4, "minimum_profit_retention": .5}
    result = run(market, setup, dates, params={"holding_sessions": 20}, exit_policy=policy)
    trade = result["campaigns"][0]
    buys = [fill for fill in trade["fills"] if fill["side"] == "buy"]
    trim, stop = [fill for fill in trade["fills"] if fill["side"] == "sell"]
    assert len(buys) == 3
    original = sum(fill["quantity"] for fill in buys)
    assert trim["quantity"] == pytest.approx(.4 * original)
    assert stop["quantity"] == pytest.approx(.6 * original)
    revision = trade["stop_revisions"][0]
    cost = sum(fill["quantity"] * fill["price"] + fill["fee"] for fill in buys)
    proceeds = trim["quantity"] * trim["price"] - trim["fee"]
    remaining = original - trim["quantity"]
    peak = original * market["QQQ"][10]["close"] * .9998 * .9999 - cost
    expected_floor = (cost - proceeds + .5 * peak) / (remaining * .9998 * .9999)
    assert revision["peak_net_liquidation_profit"] == pytest.approx(peak)
    assert revision["peak_net_profit_at"] == _clock(dates[10]).isoformat()
    assert revision["profit_retention_floor"] == pytest.approx(expected_floor)
    assert revision["new_stop"] == pytest.approx(expected_floor)
    assert revision["profit_peak_basis"].startswith("completed_close")
    # Hypothetical liquidation AT the floor retains exactly the declared net profit.
    assert proceeds + remaining * expected_floor * .9998 * .9999 - cost == pytest.approx(.5 * peak)
    assert trade["net_account_pnl_pct"] < .5 * peak  # Actual next-open gap is not a guarantee.
    assert stop["reason"] == "close_stop" and stop["price"] == pytest.approx(99 * .9998)
    _validate_block(result, {"parameters_sha256": result["parameters_sha256"]},
                    result["execution_policy_id"])


def test_peak_is_observed_before_trim_but_profit_floor_waits_for_actual_fill():
    dates, market, setup = retention_scenario()
    policy = {**EXIT_POLICY, "minimum_profit_retention": .75}
    pending = run(market, setup, dates[:11], params={"holding_sessions": 20},
                  exit_policy=policy)["campaigns"][0]
    assert pending["peak_net_liquidation_profit"] > 0
    assert pending["first_trim_filled_at"] is None
    assert pending["stop_revisions"] == []
    assert pending["applied_stop_price"] == pending["original_stop_price"]
    result = run(market, setup, dates, params={"holding_sessions": 20}, exit_policy=policy)
    trade = result["campaigns"][0]
    assert all(datetime.fromisoformat(revision["at"]) >= datetime.fromisoformat(
        trade["first_trim_filled_at"]) for revision in trade["stop_revisions"])


def test_nonprofitable_campaign_has_zero_peak_and_no_invented_retained_gain():
    dates, market, setup = fixture()
    # A tiny explicit gross target can be hit while still losing after exit costs.
    setup["target_prices"] = [102.031, 104, 106]
    change(market, 3, close=102.031)
    change(market, 4, open=102.031, close=102.031)
    policy = {**EXIT_POLICY, "trailing_atr": 4, "minimum_profit_retention": 1.}
    trade = run(market, setup, dates, params={"holding_sessions": 20},
                exit_policy=policy)["campaigns"][0]
    assert trade["first_trim_filled_at"] is not None
    assert trade["peak_net_liquidation_profit"] == 0
    assert all(revision["profit_retention_floor"] == pytest.approx(
        revision["campaign_net_breakeven"]) for revision in trade["stop_revisions"])


def test_new_policy_version_hash_zero_retention_and_future_prices_are_separate():
    dates, market, setup = retention_scenario()
    params = {"holding_sessions": 20}
    policy = {**EXIT_POLICY, "minimum_profit_retention": 0.}
    old = run(market, setup, dates, params=params, exit_policy=EXIT_POLICY)
    new = run(market, setup, dates, params=params, exit_policy=policy)
    assert old["exit_policy"]["version"] == "asymmetric-close-exit-v1"
    assert new["exit_policy"]["version"] == "asymmetric-close-exit-v2"
    assert old["exit_policy_sha256"] != new["exit_policy_sha256"]
    assert old["parameters_sha256"] == new["parameters_sha256"]
    assert old["equity_curve"] == new["equity_curve"]
    assert old["campaigns"][0]["fills"] == new["campaigns"][0]["fills"]
    prefix = run(market, setup, dates[:12], params=params, exit_policy=policy)
    change(market, 20, open=8000, close=9000, high=10000)
    assert run(market, setup, dates[:12], params=params, exit_policy=policy) == prefix
    json.dumps(new, allow_nan=False)
