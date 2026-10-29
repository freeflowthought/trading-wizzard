"""Synthetic core/satellite allocation, monthly causality and shared-cash accounting."""

import copy
import hashlib
import json
from datetime import date, timedelta

import pytest

from twiz.sector_rotation import (
    UNIVERSE,
    _book,
    _clock,
    _next,
    _rebalance,
    _targets,
    build_rotation_policy,
    simulate_sector_rotation,
)
from twiz.sessions import is_us_equity_session
from twiz.swing_optimizer import _hash

PARAMS = {"momentum_sessions": 63, "trend_sessions": 63, "core_weight": .5,
          "switch_buffer": .02, "fallback": "cash"}


def fixture(n=320, drifts=(.001, .0005, .002)):
    days, day = [], date(2023, 1, 3)
    while len(days) < n:
        if is_us_equity_session(day):
            days.append(day)
        day += timedelta(days=1)
    market = {}
    for asset, drift in zip(UNIVERSE, drifts):
        market[asset] = []
        for i, day in enumerate(days):
            close = 100*(1+drift)**i
            market[asset].append({
                "date": day.isoformat(), "open": close*.999, "high": close*1.005,
                "low": close*.995, "close": close, "volume": 1000,
                "provider_adjusted_close": close, "source": "yfinance",
                "session_complete": True, "price_basis": "test_provider_split_normalized",
            })
    return market, days[259].isoformat(), days[-1].isoformat()


def run(market=None, start=None, end=None, *, parameters=None, **kwargs):
    if market is None:
        market, start, end = fixture()
    return simulate_sector_rotation(market, start, end, parameters=parameters or PARAMS, **kwargs)


def test_one_shared_account_core_qqq_satellite_smh_and_separate_funded_benchmarks():
    market, start, end = fixture()
    result = run(market, start, end)
    initial = result["decisions"][0]
    assert initial["core"] == "QQQ" and initial["satellite"] == "SMH"
    assert initial["weights"] == {"QQQ": .5, "SPY": 0, "SMH": .5}
    assert result["equity_curve"][0]["equity"] == 100
    for curve in ("benchmark_price_curve", "benchmark_smh_price_curve",
                  "benchmark_static_basket_curve"):
        assert result[curve][0]["equity"] == 100
    first = result["rebalance_queues"][0]
    assert sum(first["actual_weights"].values()) == pytest.approx(1)
    assert result["equity_curve"][1]["equity"] < 101
    assert len(result["benchmark_fills"]["qqq"]) == len(result["benchmark_fills"]["smh"]) == 1
    assert len(result["benchmark_fills"]["static_basket"]) == 2
    expected_basis = "provider_split_normalized_units_not_historical_nominal_shares"
    assert result["quantity_basis"] == expected_basis


def test_qqq_core_and_qqq_satellite_combine_to_100_not_two_funded_positions():
    result = run(*fixture(drifts=(.003, .001, .002)))
    assert result["decisions"][0]["weights"] == {"QQQ": 1, "SPY": 0, "SMH": 0}
    assert result["metrics"]["alpha_to_price_pp"] == pytest.approx(0, abs=1e-10)
    assert len(result["fills"]) == 1


def test_decisions_only_initial_and_last_completed_month_session_next_actual_open():
    market, start, end = fixture()
    result = run(market, start, end)
    dates = [d["known_at"][:10] for d in result["decisions"]]
    assert dates[0] == start
    assert dates[1:] == ["2024-01-31", "2024-02-29", "2024-03-28"]
    for queue in result["rebalance_queues"]:
        signal = date.fromisoformat(queue["signal_at"][:10])
        assert queue["at"] == _clock(_next(signal), True)
    assert any(q["signal_at"].startswith("2024-03-28")
               and q["at"].startswith("2024-04-01") for q in result["rebalance_queues"])
    assert len(result["rebalance_queues"]) == len(result["decisions"])


def test_static_basket_really_holds_initial_units_while_rotation_rebalances():
    result = run()
    static = result["benchmark_static_basket_curve"]
    assert all(p["positions"] == static[1]["positions"] for p in static[1:])
    assert static[-1]["weights"]["SMH"] > static[1]["weights"]["SMH"]
    assert len(result["fills"]) > 2
    metrics = result["metrics"]
    assert metrics["rotation_excess_vs_static_basket_price_pp"] == pytest.approx(
        metrics["net_account_return_pct"]-metrics["benchmark_static_basket_price_return_pct"])
    assert metrics["rotation_excess_vs_static_basket_price_pp"] < 0


@pytest.mark.parametrize("fallback,spy_weight", [("cash", .3), ("spy", 1)])
def test_core_fallback_only_uses_eligible_spy_without_borrowed_capital(fallback, spy_weight):
    result = run(*fixture(drifts=(-.001, .001, -.002)),
                 parameters={**PARAMS, "core_weight": .7, "fallback": fallback})
    first = result["decisions"][0]
    assert first["weights"]["QQQ"] == first["weights"]["SMH"] == 0
    assert first["weights"]["SPY"] == pytest.approx(spy_weight)
    assert first["core"] == ("SPY" if fallback == "spy" else None)


def test_flat_or_negative_momentum_is_ineligible_and_cash_not_default_probability():
    result = run(*fixture(drifts=(0, -.001, -.002)), parameters={**PARAMS, "fallback": "spy"})
    assert all(sum(d["weights"].values()) == 0 for d in result["decisions"])
    assert result["fills"] == []
    assert result["metrics"]["net_account_return_pct"] == 0
    assert result["equity_curve"][-1]["cash"] == 100
    assert result["benchmark_fills"]["qqq"]


def endpoint_rows(values):
    market, _, _ = fixture(n=260, drifts=(0, 0, 0))
    for asset, close in zip(UNIVERSE, values):
        market[asset][-1].update(open=close, close=close, low=close-1, high=close+1)
    return market


def test_satellite_hysteresis_uses_absolute_roc_gap_inclusive_at_buffer():
    market = endpoint_rows((110, 105, 112))
    retained = _targets(market, 259, PARAMS, "QQQ", 1)
    assert retained["satellite"] == "QQQ" and retained["incumbent_retained"] is True
    switched = _targets(market, 259, {**PARAMS, "switch_buffer": 0}, "QQQ", 1)
    assert switched["satellite"] == "SMH"
    assert switched["best_eligible"] == "SMH"


def test_ineligible_incumbent_cannot_be_retained_and_initial_ties_follow_universe_order():
    market = endpoint_rows((110, 110, 110))
    assert _targets(market, 259, PARAMS, None, 1)["satellite"] == "QQQ"
    assert _targets(market, 259, PARAMS, "SMH", 1)["satellite"] == "SMH"
    market["SMH"][-1].update(close=90, open=90, high=91, low=89)
    assert _targets(market, 259, PARAMS, "SMH", 1)["satellite"] == "QQQ"


def test_common_postcost_nav_bisection_sells_first_and_independent_cash_rebuild():
    book = _book()
    book.update(cash=20, positions={"QQQ": .2, "SPY": .3, "SMH": .3},
                basis={"QQQ": 20, "SPY": 30, "SMH": 30})
    original = copy.deepcopy(book)
    prices, weights = dict.fromkeys(UNIVERSE, 100), {"QQQ": .2, "SPY": .5, "SMH": .3}
    result = _rebalance(book, weights, prices, "2024-01-31T16:15:00-05:00",
                        "2024-02-01T09:30:00-05:00", .0001, .0002)
    assert [f["side"] for f in result["fills"]] == ["sell", "sell", "buy"]
    assert result["actual_weights"] == pytest.approx(weights)
    assert abs(result["solver_residual"]) < 1e-12
    assert result["actual_postcost_nav"] == pytest.approx(result["solved_postcost_nav"])
    cash, positions = original["cash"], original["positions"]
    for fill in result["fills"]:
        if fill["side"] == "sell":
            cash += fill["units"]*fill["price"]-fill["fee"]
            positions[fill["asset"]] -= fill["units"]
        else:
            cash -= fill["units"]*fill["price"]+fill["fee"]
            positions[fill["asset"]] += fill["units"]
        assert fill["cash_after"] == pytest.approx(cash, abs=1e-12)
        assert fill["cash_after"] >= 0
    assert positions == pytest.approx(book["positions"])
    assert 100-result["actual_postcost_nav"] == pytest.approx(
        sum(f["fee"]+f["slippage"] for f in result["fills"]))


def test_unlevered_books_nav_and_wac_profit_are_physically_consistent():
    result = run()
    for point in result["equity_curve"]:
        assert point["cash"] >= 0
        assert all(value >= 0 for value in point["positions"].values())
        assert 0 <= point["exposure"] <= 1+1e-12
        assert sum(point["weights"].values()) == point["exposure"]
    metrics = result["metrics"]
    assert metrics["net_account_return_pct"] == pytest.approx(
        metrics["realized_account_pnl_pct"]+metrics["unrealized_account_pnl_pct"])
    assert metrics["fees_paid"] == pytest.approx(sum(f["fee"] for f in result["fills"]))
    assert metrics["slippage_paid"] == pytest.approx(sum(f["slippage"] for f in result["fills"]))


def test_initial_allin_costs_common_to_every_benchmark_and_same_actual_open():
    result = run()
    first_open = result["rebalance_queues"][0]["at"]
    post = 100/(1.0002*1.0001)
    assert result["rebalance_queues"][0]["solved_postcost_nav"] == pytest.approx(post)
    for fills in result["benchmark_fills"].values():
        assert {fill["at"] for fill in fills} == {first_open}
        assert sum(fill["units"]*fill["raw_open"] for fill in fills) == pytest.approx(post)
        assert sum(fill["fee"]+fill["slippage"] for fill in fills) == pytest.approx(100-post)


def test_end_month_signal_stays_pending_without_invented_end_exit_or_fee():
    market, start, _ = fixture()
    end = "2024-02-29"
    result = run(market, start, end)
    assert result["pending_order"]["known_at"].startswith(end)
    assert result["pending_order"]["scheduled_open_at"].startswith("2024-03-01")
    assert all(fill["at"][:10] <= end for fill in result["fills"])
    assert result["equity_curve"][-1]["positions"]["SMH"] > 0


def test_monthly_exit_to_cash_waits_for_month_end_then_sells_all_units_next_open():
    market, start, end = fixture()
    for rows in market.values():
        for row in rows:
            if row["date"] >= "2024-01-25":
                for key in ("open", "high", "low", "close", "provider_adjusted_close"):
                    row[key] *= .5
    result = run(market, start, end)
    sells = [fill for fill in result["fills"] if fill["side"] == "sell"]
    assert sells and all(fill["signal_at"].startswith("2024-01-31") for fill in sells)
    assert all(fill["at"].startswith("2024-02-01") for fill in sells)
    assert result["equity_curve"][-1]["exposure"] == 0


def test_no_future_numerics_and_changed_future_prices_cannot_change_earlier_decisions():
    class Future(dict):
        def get(self, key, *args):
            assert key == "date"
            return super().get(key, *args)

    market, start, end = fixture()
    before = run(market, start, end)
    for rows in market.values():
        rows.append(Future(date=_next(date.fromisoformat(end)).isoformat(), close=object()))
    assert run(market, start, end) == before
    changed = copy.deepcopy({a: rows[:-1] for a, rows in market.items()})
    for row in changed["SMH"]:
        if row["date"] > "2024-02-29":
            for key in ("open", "high", "low", "close"):
                row[key] *= 2
    after = run(changed, start, end)
    assert after["decisions"][:3] == before["decisions"][:3]


def test_provider_split_normalized_scale_changes_units_not_allocation_or_nav():
    market, start, end = fixture()
    before = run(market, start, end)
    for row in market["SMH"]:
        for key in ("open", "high", "low", "close", "provider_adjusted_close"):
            row[key] *= 2
        row["split_ratio"] = 2  # Informational only: no second split adjustment.
    after = run(market, start, end)
    assert after["metrics"]["net_account_return_pct"] == pytest.approx(
        before["metrics"]["net_account_return_pct"])
    assert after["equity_curve"][-1]["positions"]["SMH"] == pytest.approx(
        before["equity_curve"][-1]["positions"]["SMH"]/2)
    assert [d["weights"] for d in after["decisions"]] == [d["weights"] for d in before["decisions"]]
    for name in ("equity_curve", "benchmark_price_curve", "benchmark_smh_price_curve",
                 "benchmark_static_basket_curve", "benchmark_total_return_curve",
                 "benchmark_smh_total_return_curve"):
        actual = [p["equity"] for p in after[name]]
        assert actual == pytest.approx([p["equity"] for p in before[name]])


def test_adjusted_total_return_is_optional_and_never_used_for_price_allocation():
    market, start, end = fixture()
    before = run(market, start, end)
    market["QQQ"][-1]["provider_adjusted_close"] *= 1.05
    after = run(market, start, end)
    assert after["fills"] == before["fills"] and after["decisions"] == before["decisions"]
    assert after["metrics"]["benchmark_total_return_pct"] > before["metrics"][
        "benchmark_total_return_pct"]
    market["SMH"][-1]["provider_adjusted_close"] = None
    unavailable = run(market, start, end)
    assert unavailable["benchmark_smh_total_return_curve"] is None
    assert unavailable["metrics"]["benchmark_smh_total_return_pct"] is None
    assert unavailable["strategy_dividends_included"] is False
    assert unavailable["total_return_comparison_like_for_like"] is False


@pytest.mark.parametrize("mode", ["missing_asset", "missing_session", "duplicate", "incomplete",
                                  "basis", "nan", "different_grid"])
def test_all_assets_require_common_contiguous_completed_source_basis(mode):
    market, start, end = fixture()
    if mode == "missing_asset":
        del market["SMH"]
    elif mode == "missing_session":
        market["SMH"].pop(100)
    elif mode == "duplicate":
        market["QQQ"].insert(100, copy.deepcopy(market["QQQ"][100]))
    elif mode == "incomplete":
        market["SPY"][100]["session_complete"] = False
    elif mode == "basis":
        market["SPY"][100]["price_basis"] = "different"
    elif mode == "nan":
        market["SMH"][100]["close"] = float("nan")
    else:
        market["SMH"] = market["SMH"][1:]
    with pytest.raises(ValueError):
        run(market, start, end)


def test_maxlookback_fits_260row_warmup_and_259_does_not():
    market, start, end = fixture()
    assert run(market, start, end, parameters={**PARAMS, "momentum_sessions": 252})["fills"]
    with pytest.raises(ValueError):
        run({a: rows[1:] for a, rows in market.items()}, start, end)


def test_policy_hash_is_frozen_before_outcomes_and_cost_stress_ceiling_are_explicit():
    policy = build_rotation_policy(PARAMS)
    assert policy["policy_sha256"] == _hash(policy["policy"])
    assert set(policy["policy"]["source_code_sha256"]) == {
        "sector_rotation.py", "structure.py", "sessions.py", "swing_optimizer.py"}
    stress_policy = build_rotation_policy(PARAMS, cost_multiplier=2)
    assert stress_policy["policy_sha256"] != policy["policy_sha256"]
    base, stress = run(), run(cost_multiplier=2)
    assert stress["metrics"]["fees_paid"] > base["metrics"]["fees_paid"]
    capped = run(max_exposure=.7)
    assert sum(capped["decisions"][0]["weights"].values()) == pytest.approx(.7)
    assert capped["rebalance_queues"][0]["actual_weights"] == pytest.approx(
        capped["decisions"][0]["weights"])
    static_fills = capped["benchmark_fills"]["static_basket"]
    opening_value = sum(fill["units"]*fill["raw_open"] for fill in static_fills)
    opening_nav = static_fills[-1]["cash_after"]+opening_value
    assert opening_value/opening_nav == pytest.approx(.7)
    assert capped["benchmark_price_curve"][1]["exposure"] == pytest.approx(1)
    assert capped["benchmark_smh_price_curve"][1]["exposure"] == pytest.approx(1)


@pytest.mark.parametrize("change", [
    {"momentum_sessions": 20}, {"trend_sessions": 100}, {"core_weight": .3},
    {"switch_buffer": .01}, {"fallback": "qqq"}, {"momentum_sessions": True}, {"unused": 1},
])
def test_exact_parameters_reject_unbounded_or_unused_changes(change):
    with pytest.raises(ValueError):
        build_rotation_policy({**PARAMS, **change})


@pytest.mark.parametrize("weights", [{"QQQ": 1, "SPY": 1, "SMH": 0},
                                      {"QQQ": -1, "SPY": 0, "SMH": 0}])
def test_rebalance_rejects_borrowed_or_negative_targets(weights):
    with pytest.raises(ValueError):
        _rebalance(_book(), weights, dict.fromkeys(UNIVERSE, 100), "signal", "at", .0001, .0002)


def test_anchor_core_and_default_satellite_remain_qqq_even_below_sma_and_negative_roc():
    parameters = {**PARAMS, "fallback": "qqq_anchor", "core_weight": .7}
    market = endpoint_rows((90, 150, 95))
    decision = _targets(market, 259, parameters, None, 1)
    assert decision["features"]["QQQ"]["eligible"] is False
    assert decision["features"]["SPY"]["eligible"] is True
    assert decision["core"] == decision["satellite"] == "QQQ"
    assert decision["weights"] == {"QQQ": 1, "SPY": 0, "SMH": 0}


def test_anchor_qqq_core_is_unconditional_but_smh_satellite_still_needs_absolute_eligibility():
    parameters = {**PARAMS, "fallback": "qqq_anchor"}
    decision = _targets(endpoint_rows((90, 150, 112)), 259, parameters, None, 1)
    assert decision["core"] == "QQQ" and decision["satellite"] == "SMH"
    assert decision["weights"] == {"QQQ": .5, "SPY": 0, "SMH": .5}
    # SMH beats weak QQQ relatively, but its own negative ROC forbids enhancement.
    disabled = _targets(endpoint_rows((80, 150, 95)), 259, parameters, "SMH", 1)
    assert disabled["satellite"] == "QQQ"
    assert disabled["weights"]["SPY"] == disabled["weights"]["SMH"] == 0


@pytest.mark.parametrize("closes,incumbent,expected", [
    ((110, 105, 112), "QQQ", "QQQ"),  # At entry buffer: keep baseline.
    ((110, 105, 112.1), "QQQ", "SMH"),
    ((112, 105, 110), "SMH", "SMH"),  # At exit buffer: keep eligible incumbent.
    ((112.1, 105, 110), "SMH", "QQQ"),
    ((111, 105, 110), "SMH", "SMH"),  # Minor QQQ lead is not a forced switch.
    ((110, 105, 111), "QQQ", "QQQ"),
])
def test_anchor_uses_two_sided_relative_hysteresis(closes, incumbent, expected):
    parameters = {**PARAMS, "fallback": "qqq_anchor"}
    decision = _targets(endpoint_rows(closes), 259, parameters, incumbent, 1)
    assert decision["satellite"] == expected
    assert decision["weights"]["SPY"] == 0
    assert decision["incumbent_retained"] is (incumbent == expected)


def test_anchor_zero_buffer_entry_requires_strict_smh_lead_and_ineligible_smh_exits_now():
    parameters = {**PARAMS, "fallback": "qqq_anchor", "switch_buffer": 0}
    assert _targets(endpoint_rows((110, 150, 110)), 259, parameters, "QQQ", 1)[
        "satellite"] == "QQQ"
    assert _targets(endpoint_rows((110, 150, 110.1)), 259, parameters, "QQQ", 1)[
        "satellite"] == "SMH"
    assert _targets(endpoint_rows((80, 150, 99)), 259, parameters, "SMH", 1)[
        "satellite"] == "QQQ"


@pytest.mark.parametrize("cap", [1, .95, .9, .85])
def test_anchor_residual_cash_only_from_learnable_cap_and_no_spy_allocation(cap):
    result = run(*fixture(drifts=(-.001, .003, -.002)),
                 parameters={**PARAMS, "fallback": "qqq_anchor"}, max_exposure=cap)
    assert all(d["weights"] == {"QQQ": cap, "SPY": 0, "SMH": 0}
               for d in result["decisions"])
    first = result["rebalance_queues"][0]
    assert first["actual_weights"]["QQQ"] == pytest.approx(cap)
    assert all(point["positions"]["SPY"] == 0 for point in result["equity_curve"])
    assert result["metrics"]["fill_count"] >= 1


def test_anchor_has_version2_policy_while_old_mode_receipt_descriptions_remain_version1():
    old = build_rotation_policy(PARAMS)["policy"]
    anchor = build_rotation_policy({**PARAMS, "fallback": "qqq_anchor"})["policy"]
    assert old["version"] == "monthly-core-satellite-rotation-1"
    assert anchor["version"] == "monthly-core-satellite-rotation-2"
    assert "if eligible" in old["core"]
    assert "always QQQ" in anchor["core"]
    assert "SMH only" in anchor["eligibility"]


@pytest.mark.parametrize("mode,expected", [
    ("cash", "a349de463396093b2b15076d1767abde076ac3319da49299fac8bc734a81c703"),
    ("spy", "63e959c4b61e558da39edd3cd7aca05194c1775e51f1daf2a27e26b7c425c22c"),
])
def test_old_modes_numerical_books_and_decisions_match_pre_anchor_golden(mode, expected):
    # Captured from synthetic fixtures BEFORE the branch was added. Source/policy
    # fingerprints necessarily change; every numerical book/decision stays exact.
    market, start, end = fixture(drifts=(-.001, .001, -.002))
    result = run(market, start, end, parameters={**PARAMS, "core_weight": .7, "fallback": mode})
    fields = {key: result[key] for key in (
        "metrics", "equity_curve", "fills", "decisions", "rebalance_queues", "pending_order",
        "benchmark_price_curve", "benchmark_smh_price_curve", "benchmark_static_basket_curve",
        "benchmark_fills", "benchmark_total_return_curve", "benchmark_smh_total_return_curve",
    )}
    body = json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(body).hexdigest() == expected
