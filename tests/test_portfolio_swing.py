"""Synthetic portfolio accounting and causal price-rule checks; no real outcomes."""

import copy
from datetime import date, datetime, timedelta

import pytest

from twiz.portfolio_swing import (
    _clock,
    _features,
    _next,
    build_portfolio_swing_policy,
    simulate_portfolio_swing,
)
from twiz.sessions import is_us_equity_session
from twiz.swing_optimizer import _hash

PARAMS = {"support_sessions": 10, "trend_sessions": 63, "confirm_sessions": 2,
          "trim_to": .5, "trailing_atr": 2, "entry_steps": [.4, .7, 1]}


def fixture(closes=None):
    day, history = date(2023, 1, 3), []
    while len(history) < 260:
        if is_us_equity_session(day):
            history.append({"date": day.isoformat(), "open": 100.0, "high": 101.0,
                            "low": 99.0, "close": 100.0, "volume": 1000,
                            "provider_adjusted_close": 100.0, "source": "yfinance",
                            "session_complete": True, "price_basis": "test_nominal_consistent"})
        day += timedelta(days=1)
    start = history[-1]["date"]
    for close in closes or [100] * 10:
        append(history, close=close)
    return history, start, history[-1]["date"]


def append(rows, *, close, **values):
    day = _next(date.fromisoformat(rows[-1]["date"]))
    row = {**rows[-1], "date": day.isoformat(), "open": close, "high": close + 1,
           "low": close - 1, "close": close, "provider_adjusted_close": close, **values}
    rows.append(row)


def run(rows=None, start=None, end=None, *, parameters=None, asset="QQQ", **kwargs):
    if rows is None:
        rows, start, end = fixture()
    return simulate_portfolio_swing({asset: rows}, asset, start, end,
                                    parameters=parameters or PARAMS, **kwargs)


def top_fixture():
    rows, start, _ = fixture([100])
    rows.pop()
    append(rows, open=104, close=105, high=106, low=103)
    append(rows, open=105, close=106, high=108, low=104)
    append(rows, open=107, close=105.5, high=108.5, low=105)
    append(rows, open=104, close=104.4, high=105, low=103)
    return rows, start, rows[-1]["date"]


def test_initial_full_cap_matches_same_open_100pct_buy_hold_costs_without_churn():
    rows, start, end = fixture()
    result = run(rows, start, end)
    assert len(result["fills"]) == len(result["benchmark_fills"]) == 1
    buy, benchmark = result["fills"][0], result["benchmark_fills"][0]
    assert buy["at"] == benchmark["at"] == _clock(date.fromisoformat(rows[260]["date"]), True)
    assert buy["signal_at"] == _clock(date.fromisoformat(start))
    assert buy["qty"] == benchmark["qty"] == pytest.approx(100 / (100 * 1.0002 * 1.0001))
    assert buy["post_fill_exposure"] == pytest.approx(1)
    assert buy["cash"] == pytest.approx(0)
    assert result["equity_curve"] == result["benchmark_price_curve"]
    assert result["metrics"]["alpha_to_price_pp"] == 0
    assert result["metrics"]["alpha_to_total_return_pp"] == pytest.approx(0)
    assert result["metrics"]["net_account_return_pct"] < 0
    assert result["metrics"]["fill_count"] == 1
    assert all(not decision["order_scheduled"] for decision in result["decisions"][1:])


def test_rally_has_no_daily_weight_rebalancing_or_unilateral_small_risk_cap():
    rows, start, end = fixture([100, 102, 104, 106, 108])
    result = run(rows, start, end)
    assert result["metrics"]["fill_count"] == 1
    assert result["metrics"]["max_exposure_pct"] == pytest.approx(100)
    assert result["metrics"]["net_account_return_pct"] > 7
    assert result["metrics"]["turnover_pct"] > 99
    assert result["equity_curve"][-1]["qty"] == result["fills"][0]["qty"]


def test_full_trend_exit_works_before_first_top_and_fills_actual_gap_open():
    rows, start, end = fixture([98, 97, 90, 90])
    rows[262].update(open=90, high=91, low=89, close=90)
    result = run(rows, start, end)
    sell = result["fills"][1]
    assert sell["cause"] == "trend_exit" and sell["target_exposure"] == 0
    assert sell["signal_at"] == _clock(date.fromisoformat(rows[261]["date"]))
    assert sell["at"] == _clock(date.fromisoformat(rows[262]["date"]), True)
    assert sell["price"] == pytest.approx(90 * .9998)
    assert sell["qty"] == result["fills"][0]["qty"]
    assert sell["holdings_qty"] == 0 and sell["cost_basis"] == 0
    assert result["equity_curve"][-1]["cash"] == result["equity_curve"][-1]["equity"]


@pytest.mark.parametrize("trim", [0, .5, .7])
def test_profitable_bearish_rejection_partial_or_full_sells_to_cost_aware_target(trim):
    rows, start, end = top_fixture()
    result = run(rows, start, end, parameters={**PARAMS, "trim_to": trim})
    sell = result["fills"][1]
    assert sell["cause"] == "top_rejection"
    assert sell["target_exposure"] == trim
    assert sell["post_fill_exposure"] == pytest.approx(trim)
    assert sell["side"] == "sell" and sell["cash"] >= 0
    assert sell["signal_at"] == _clock(date.fromisoformat(rows[-2]["date"]))
    assert result["decisions"][-1]["trimmed"] is True


def test_repeated_top_rejection_does_not_repeat_partial_and_suppresses_adds():
    rows, start, _ = top_fixture()
    append(rows, open=105, close=104.2, high=108.5, low=103.5)
    append(rows, open=104.2, close=104.2, high=105, low=103.5)
    result = run(rows, start, rows[-1]["date"])
    assert len(result["fills"]) == 2
    rejection = result["decisions"][-2]
    assert rejection["cause"] == "top_rejection_already_trimmed"
    assert rejection["order_scheduled"] is False


def test_rejection_requires_profit_after_actual_entry_cost():
    rows, start, end = fixture([100, 99.99, 100])
    rows[261].update(open=100.5, high=101, low=99, close=99.99)
    result = run(rows, start, end)
    assert not any(fill["cause"] == "top_rejection" for fill in result["fills"])


def test_post_trim_trailing_stop_sells_next_gap_open():
    rows, start, _ = top_fixture()
    append(rows, open=103, close=99, high=104, low=98)
    append(rows, open=95, close=96, high=97, low=94)
    result = run(rows, start, rows[-1]["date"])
    sell = result["fills"][-1]
    assert sell["cause"] == "post_trim_trailing_exit"
    assert sell["raw_open"] == 95 and sell["target_exposure"] == 0
    assert sell["holdings_qty"] == 0


def test_bottom_takes_multiple_sessions_then_new_closes_stage_without_zone_retouch():
    rows, start, _ = fixture([98, 96])
    append(rows, open=95, close=97, low=94, high=98)
    append(rows, open=97, close=98, low=95, high=99)
    append(rows, open=100, close=100.5, low=97, high=101)
    append(rows, open=100.6, close=101, low=99.5, high=102)
    append(rows, open=101.3, close=102, low=100.5, high=103)
    append(rows, open=102.5, close=103, low=102, high=104)
    result = run(rows, start, rows[-1]["date"])
    assert [fill["target_exposure"] for fill in result["fills"]] == [1, 0, .4, .7, 1]
    adds = result["fills"][2:]
    assert all(fill["cause"] == "bottom_recovery" for fill in adds)
    first_signal = next(d for d in result["decisions"] if d["cause"] == "bottom_recovery")
    assert first_signal["features"]["anchor_age_sessions"] == 2
    assert first_signal["features"]["anchor_date"] == rows[262]["date"]
    assert all(rows[index]["low"] > 94 for index in (265, 266, 267))
    assert all(datetime.fromisoformat(fill["at"]) > datetime.fromisoformat(fill["signal_at"])
               for fill in adds)
    assert result["decisions"][-1]["trimmed"] is False
    assert result["equity_curve"][-1]["highwater"] == 103


def test_trend_breakout_reenters_cash_and_continuation_adds_without_fresh_bottom():
    rows, start, _ = fixture([98, 96, 95])
    append(rows, open=101, close=105, low=90, high=106)
    append(rows, open=105.5, close=106, low=91, high=107)
    append(rows, open=106.5, close=107, low=92, high=108)
    append(rows, open=107.5, close=108, low=93, high=109)
    result = run(rows, start, rows[-1]["date"])
    causes = [fill["cause"] for fill in result["fills"]]
    assert causes[:4] == ["initial_allocation", "trend_exit", "trend_reentry",
                          "recovery_continuation"]
    reentry = next(d for d in result["decisions"] if d["cause"] == "trend_reentry")
    assert reentry["features"]["bottom_confirmed"] is False


def test_exact_book_accounting_cash_quantity_cost_basis_and_costs():
    rows, start, end = top_fixture()
    result = run(rows, start, end)
    cash, qty, cost = 100.0, 0.0, 0.0
    for fill in result["fills"]:
        old_qty = qty
        if fill["side"] == "buy":
            cash -= fill["qty"] * fill["price"] + fill["fee"]
            qty += fill["qty"]
            cost += fill["qty"] * fill["price"] + fill["fee"]
        else:
            cash += fill["qty"] * fill["price"] - fill["fee"]
            cost *= 1 - fill["qty"] / old_qty
            qty -= fill["qty"]
        assert fill["cash"] == pytest.approx(cash, abs=1e-12)
        assert fill["holdings_qty"] == pytest.approx(qty)
        assert fill["cost_basis"] == pytest.approx(cost)
    assert result["metrics"]["fees_paid"] == sum(f["fee"] for f in result["fills"])
    assert result["metrics"]["slippage_paid"] == sum(f["slippage"] for f in result["fills"])
    assert all(p["cash"] >= 0 and p["qty"] >= 0 and 0 <= p["exposure"] <= 1 + 1e-12
               for p in result["equity_curve"])


def test_reduction_never_buys_after_a_gap_already_takes_weight_below_target():
    rows, start, end = top_fixture()
    rows[-1].update(open=20, close=20, high=21, low=19, provider_adjusted_close=20)
    result = run(rows, start, end, max_exposure=.7)
    assert len(result["fills"]) == 1
    assert result["unfilled_orders"][0]["direction"] == "sell"
    assert result["unfilled_orders"][0]["cause"] == "top_rejection"
    assert result["unfilled_orders"][0]["target_exposure"] == pytest.approx(.35)


def test_addition_never_sells_after_gap_already_takes_weight_above_target():
    rows, start, _ = fixture([98, 96])
    append(rows, open=95, close=97, low=94, high=98)
    append(rows, open=97, close=98, low=95, high=99)
    append(rows, open=100, close=100.5, low=97, high=101)
    append(rows, open=100.6, close=101, low=99.5, high=102)
    append(rows, open=1000, close=1000, low=999, high=1001)
    result = run(rows, start, rows[-1]["date"])
    assert [fill["target_exposure"] for fill in result["fills"]] == [1, 0, .4]
    assert result["unfilled_orders"][-1]["direction"] == "buy"
    assert result["unfilled_orders"][-1]["target_exposure"] == .7


def test_final_unexecuted_signal_remains_pending_no_imaginary_exit_fee():
    rows, start, end = fixture([98, 97])
    result = run(rows, start, end)
    assert len(result["fills"]) == len(result["benchmark_fills"]) == 1
    assert result["pending_order"]["cause"] == "trend_exit"
    assert result["equity_curve"][-1]["qty"] > 0
    assert result["metrics"]["fees_paid"] == result["fills"][0]["fee"]


def test_total_return_benchmark_uses_provider_adjusted_close_only_not_strategy_features():
    rows, start, end = fixture([100, 100, 100])
    before = run(rows, start, end)
    changed = copy.deepcopy(rows)
    changed[-1]["provider_adjusted_close"] = 105
    after = run(changed, start, end)
    assert after["fills"] == before["fills"]
    assert after["decisions"] == before["decisions"]
    assert after["equity_curve"] == before["equity_curve"]
    assert after["benchmark_price_curve"] == before["benchmark_price_curve"]
    expected = 100 * 105 / (100 * 1.0002 * 1.0001)
    assert after["benchmark_total_return_curve"][-1]["equity"] == pytest.approx(expected)
    assert after["metrics"]["alpha_to_total_return_pp"] < -4.9
    assert after["benchmark_total_return_basis"] == "provider_adjusted_close_not_strict_vintage"
    assert after["strategy_dividends_included"] is False
    assert after["total_return_comparison_like_for_like"] is False
    assert "conservative" in after["alpha_to_total_return_interpretation"]


@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), True])
def test_total_return_missing_is_explicit_not_replaced_with_price_return(value):
    rows, start, end = fixture()
    rows[-1]["provider_adjusted_close"] = value
    result = run(rows, start, end)
    assert result["benchmark_total_return_curve"] is None
    assert result["metrics"]["benchmark_total_return_pct"] is None
    assert result["metrics"]["alpha_to_total_return_pp"] is None
    assert result["benchmark_total_return_unavailable_reason"]
    assert result["metrics"]["benchmark_price_return_pct"] is not None


def test_policy_frozen_before_outcomes_and_configurable_100pct_ceiling_cost_stress():
    policy = build_portfolio_swing_policy(PARAMS)
    assert policy["policy_sha256"] == _hash(policy["policy"])
    assert policy["policy"]["max_exposure"] == 1
    assert "risk_cap" not in policy["policy"]
    assert set(policy["policy"]["source_code_sha256"]) == {
        "portfolio_swing.py", "structure.py", "sessions.py", "swing_optimizer.py"
    }
    assert build_portfolio_swing_policy(PARAMS, max_exposure=.7)["policy_sha256"] != policy[
        "policy_sha256"]
    assert build_portfolio_swing_policy(PARAMS, cost_multiplier=2)["policy_sha256"] != policy[
        "policy_sha256"]
    base, stress = run(), run(cost_multiplier=2)
    assert stress["metrics"]["fees_paid"] > base["metrics"]["fees_paid"]
    assert stress["metrics"]["slippage_paid"] > base["metrics"]["slippage_paid"]
    configured = run(max_exposure=.7)
    assert configured["fills"][0]["post_fill_exposure"] == pytest.approx(.7)
    assert configured["benchmark_fills"][0]["post_fill_exposure"] == pytest.approx(1)


def test_features_ignore_future_prices_and_full_result_ignores_rows_after_end():
    class Future(dict):
        def get(self, key, *args):
            assert key == "date", f"Read future numerical field {key}"
            return super().get(key, *args)

    rows, start, end = fixture()
    before = run(rows, start, end)
    future = Future(date=_next(date.fromisoformat(end)).isoformat(), close=object())
    assert run(rows + [future], start, end) == before
    changed = copy.deepcopy(rows)
    changed[-1].update(open=1000, high=1001, low=999, close=1000)
    after = run(changed, start, end)
    assert after["decisions"][:-1] == before["decisions"][:-1]
    assert after["fills"] == before["fills"]


def test_feature_anchor_equal_minimum_restarts_multi_day_clock():
    rows, _, _ = fixture([98, 99, 100])
    assert _features(rows, PARAMS)["bottom_confirmed"] is True
    rows[-1]["low"] = rows[-3]["low"]
    features = _features(rows, PARAMS)
    assert features["anchor_age_sessions"] == 0 and features["bottom_confirmed"] is False


@pytest.mark.parametrize("mode", ["missing", "duplicate", "incomplete", "basis", "nan"])
def test_complete_contiguous_uniform_basis_and_finite_source_prices_required(mode):
    rows, start, end = fixture()
    if mode == "missing":
        rows.pop(100)
    elif mode == "duplicate":
        rows.insert(100, copy.deepcopy(rows[100]))
    elif mode == "incomplete":
        rows[100]["session_complete"] = False
    elif mode == "basis":
        rows[100]["price_basis"] = "other_basis"
    else:
        rows[100]["close"] = float("nan")
    with pytest.raises(ValueError):
        run(rows, start, end)


def test_260_origin_history_required_and_input_not_mutated_and_spy_is_separate():
    rows, start, end = fixture()
    original = copy.deepcopy(rows)
    qqq, spy = run(rows, start, end), run(rows, start, end, asset="SPY")
    assert rows == original
    assert qqq["metrics"] == spy["metrics"] and qqq["asset"] != spy["asset"]
    with pytest.raises(ValueError, match="260"):
        run(rows[1:], start, end)


@pytest.mark.parametrize("change", [
    {"support_sessions": 11}, {"trend_sessions": 100}, {"confirm_sessions": 1},
    {"trim_to": .2}, {"trailing_atr": 1}, {"entry_steps": [.3, .6, 1]},
    {"confirm_sessions": True}, {"unused": 1},
])
def test_exact_parameters_reject_undeclared_or_invalid_changes(change):
    with pytest.raises(ValueError):
        build_portfolio_swing_policy({**PARAMS, **change})


@pytest.mark.parametrize("exposure", [0, -1, 1.01, True, float("nan")])
def test_leverage_and_invalid_exposure_not_allowed(exposure):
    with pytest.raises(ValueError):
        build_portfolio_swing_policy(PARAMS, max_exposure=exposure)
