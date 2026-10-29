"""Structural V2 synthetic state-machine tests; no real outcomes or fitting."""

import copy
import hashlib
from datetime import datetime
from pathlib import Path

import pytest
from test_portfolio_swing import append, fixture, top_fixture

from twiz import portfolio_structure as module
from twiz.portfolio_structure import build_structural_policy, simulate_structural_portfolio
from twiz.portfolio_swing import simulate_portfolio_swing
from twiz.swing_optimizer import _hash

PARAMS = {"support_sessions": 20, "trend_sessions": 63, "confirm_sessions": 2,
          "trim_to": .7, "trailing_atr": 3, "entry_steps": [.4, .7, 1],
          "profit_atr": 3, "breakout_sessions": 3, "stop_buffer_atr": .5}


def run(rows=None, start=None, end=None, *, parameters=None, **kwargs):
    if rows is None:
        rows, start, end = fixture()
    return simulate_structural_portfolio({"QQQ": rows}, "QQQ", start, end,
                                         parameters=parameters or PARAMS, **kwargs)


def wave_path(*, full_recovery=False):
    rows, start, _ = fixture([100])
    rows.pop()
    for values in (
        {"open": 100, "close": 105, "high": 106, "low": 99},
        {"open": 105, "close": 110, "high": 111, "low": 104},
        {"open": 109, "close": 108, "high": 110, "low": 107},
        {"open": 108.5, "close": 109, "high": 110, "low": 108},
        {"open": 110, "close": 108, "high": 110, "low": 107},
        {"open": 107, "close": 105, "high": 108, "low": 104},
        {"open": 104, "close": 104, "high": 105, "low": 103},
        {"open": 104, "close": 104.5, "high": 105.2, "low": 103.5},
    ):
        append(rows, **values)
    if full_recovery:
        for values in (
            {"open": 105, "close": 105.5, "high": 106, "low": 104},
            {"open": 106, "close": 106.5, "high": 107, "low": 105},
            {"open": 106.6, "close": 107, "high": 108, "low": 106},
            {"open": 107.2, "close": 108, "high": 109, "low": 107},
            {"open": 108.3, "close": 109, "high": 110, "low": 108},
        ):
            append(rows, **values)
    return rows, start, rows[-1]["date"]


def below_ma_path():
    rows, start, _ = fixture([100])
    rows = rows[:260]
    for row in rows:
        row.update(open=200, high=201, low=199, close=200, provider_adjusted_close=200)
    for values in (
        {"open": 198, "close": 190, "high": 199, "low": 189},
        {"open": 190, "close": 180, "high": 191, "low": 179},
        {"open": 175, "close": 175, "high": 178, "low": 170},
        {"open": 175.5, "close": 176, "high": 177, "low": 172},
        {"open": 178, "close": 179, "high": 180, "low": 174},
        {"open": 180, "close": 181, "high": 182, "low": 176},
        {"open": 181.5, "close": 182, "high": 183, "low": 179},
        {"open": 182.5, "close": 183, "high": 184, "low": 180},
        {"open": 183.5, "close": 184, "high": 185, "low": 181},
    ):
        append(rows, **values)
    return rows, start, rows[-1]["date"]


def test_v1_single_red_bar_near_prior_high_does_not_make_v2_stage_top():
    rows, start, end = top_fixture()
    result = run(rows, start, end)
    assert len(result["fills"]) == 1
    assert not any(d["cause"] == "structural_top" for d in result["decisions"])


def test_top_requires_both_meaningful_wave_and_close_below_prior_n_low():
    rows, start, end = wave_path()
    result = run(rows, start, end)
    sell = result["fills"][1]
    assert sell["cause"] == "structural_top" and sell["target_exposure"] == .7
    decision = next(d for d in result["decisions"] if d["cause"] == "structural_top")
    assert decision["wave_profit_atr"] >= 3
    assert decision["features"]["close"] < decision["features"]["prior_breakout_low"]
    stricter = run(rows, start, end, parameters={**PARAMS, "profit_atr": 4})
    assert len(stricter["fills"]) == 1
    assert all(f["cause"] != "structural_top" for f in stricter["fills"])


def test_no_old_pre_top_anchor_authorizes_next_green_bar_repurchase():
    rows, start, end = wave_path()
    result = run(rows, start, end)
    sell = result["fills"][1]
    latest = result["decisions"][-1]
    assert latest["features"]["rising"] is True
    assert latest["episode_start"] == sell["signal_at"]
    assert latest["features"]["anchor_date"] >= sell["signal_at"][:10]
    assert latest["features"]["anchor_age_sessions"] == 1
    assert latest["order_scheduled"] is False
    assert len(result["fills"]) == 2


def test_new_post_sell_base_has_aged_confirmation_and_prior_high_breakout():
    rows, start, end = wave_path(full_recovery=True)
    result = run(rows, start, end)
    buy = result["fills"][2]
    signal = buy["recovery_signal"]
    assert buy["cause"] == "new_base_recovery"
    assert signal["anchor_date"] >= result["fills"][1]["signal_at"][:10]
    decision = next(d for d in result["decisions"] if d["proposed_recovery"])
    f = decision["features"]
    assert f["anchor_age_sessions"] >= PARAMS["confirm_sessions"]
    assert f["close"] > f["prior_breakout_high"]
    assert signal["signal_at"] == decision["known_at"]
    assert decision["structural_stop"] is None
    assert datetime.fromisoformat(buy["at"]) > datetime.fromisoformat(signal["signal_at"])


def test_below_slow_ma_reentry_stages_to_full_without_repeated_ma_cash_veto():
    rows, start, end = below_ma_path()
    result = run(rows, start, end)
    assert [f["target_exposure"] for f in result["fills"]] == [1, 0, .4, .7, 1]
    first = result["fills"][2]
    assert first["recovery_signal"]["risk_armed_on_fill"] is False
    assert first["recovery_signal"]["from_zero"] is True
    assert result["fills"][3]["cause"] == result["fills"][4]["cause"] == "recovery_continuation"
    for decision in result["decisions"]:
        if decision.get("first_recovery_signal"):
            assert decision["features"]["close"] < decision["features"]["slow_sma"]
            assert decision["risk_armed"] is False
            assert decision["structural_stop"] is not None
    stop = first["recovery_signal"]["structural_stop"]
    assert all(d["structural_stop"] == stop for d in result["decisions"]
               if d.get("first_recovery_signal"))
    assert result["decisions"][-1]["trimmed"] is False
    assert result["equity_curve"][-1]["highwater"] == rows[-1]["close"]


def test_n_day_trend_false_alone_does_not_rearm_below_ma_recovery(monkeypatch):
    rows, start, end = below_ma_path()
    first_fill_day = run(rows, start, end)["fills"][2]["at"][:10]
    original = module._features

    def isolate_predicate(history, parameters, episode):
        features = original(history, parameters, episode)
        if history[-1]["date"] == first_fill_day:
            # Isolate the latch branch: keep actual close/SMA but make the N-day
            # boolean false. False is not evidence that price regained the SMA.
            features["trend_broken"] = False
        return features

    monkeypatch.setattr(module, "_features", isolate_predicate)
    result = run(rows, start, end)
    decision = next(d for d in result["decisions"] if d.get("at", "")[:10] == first_fill_day)
    assert decision["features"]["trend_broken"] is False
    assert decision["features"]["close"] < decision["features"]["slow_sma"]
    assert decision["risk_armed"] is False


def test_completed_close_regains_slow_ma_then_confirmed_breach_rearms_exit():
    rows, start, _ = below_ma_path()
    append(rows, open=204, close=205, high=206, low=203)
    append(rows, open=196, close=195, high=197, low=194)
    append(rows, open=195, close=194, high=196, low=193)
    append(rows, open=193, close=193, high=194, low=192)
    result = run(rows, start, rows[-1]["date"])
    regained = next(d for d in result["decisions"] if d["features"]["close"] == 205)
    assert regained["risk_armed"] is True
    assert result["fills"][-1]["cause"] == "trend_exit"
    assert result["fills"][-1]["target_exposure"] == 0


def test_frozen_structural_stop_survives_full_recovery_and_exits_at_actual_gap():
    rows, start, end = below_ma_path()
    before = run(rows, start, end)
    stop = before["decisions"][-1]["structural_stop"]
    append(rows, open=stop+1, close=stop-1, high=stop+2, low=stop-2)
    append(rows, open=stop-5, close=stop-4, high=stop-3, low=stop-6)
    result = run(rows, start, rows[-1]["date"])
    sell = result["fills"][-1]
    assert sell["cause"] == "structural_stop_exit"
    assert sell["raw_open"] == stop-5
    assert sell["price"] < stop-5
    assert result["decisions"][-1]["structural_stop"] is None
    assert result["decisions"][-1]["episode_start"] == sell["signal_at"]


def test_pending_first_recovery_has_no_installed_stop_before_actual_fill():
    rows, start, end = below_ma_path()
    result = run(rows, start, end)
    signal = next(d for d in result["decisions"] if d["proposed_recovery"])
    assert signal["structural_stop"] is None and signal["first_recovery_signal"] is None
    assert signal["proposed_recovery"]["structural_stop"] > 0
    later = next(d for d in result["decisions"] if d.get("first_recovery_signal"))
    assert later["structural_stop"] == signal["proposed_recovery"]["structural_stop"]


def test_buy_gap_below_frozen_stop_cancels_without_consuming_stage_or_latch():
    rows, start, end = below_ma_path()
    before = run(rows, start, end)
    buy = before["fills"][2]
    stop, buy_day = buy["recovery_signal"]["structural_stop"], buy["at"][:10]
    for row in rows:
        if row["date"] == buy_day:
            row.update(open=stop-1, close=stop-1, high=stop, low=stop-2,
                       provider_adjusted_close=stop-1)
    result = run(rows, start, buy_day)
    assert [f["target_exposure"] for f in result["fills"]] == [1, 0]
    assert result["unfilled_orders"][-1]["reason"] == "buy_open_at_or_below_frozen_stop"
    latest = result["decisions"][-1]
    assert latest["structural_stop"] is None and latest["first_recovery_signal"] is None
    assert latest["target_exposure"] == 0
    assert result["equity_curve"][-1]["qty"] == 0


def test_initial_position_armed_even_if_first_origin_is_below_slow_ma():
    rows, start, _ = fixture([98, 97, 96])
    rows[259].update(close=99, open=99, high=100, low=98, provider_adjusted_close=99)
    result = run(rows, start, rows[-1]["date"])
    assert result["decisions"][0]["risk_armed"] is True
    assert result["fills"][1]["cause"] == "trend_exit"


@pytest.mark.parametrize("trim", [0, .5, .7, .9, 1])
def test_extended_trim_fractions_supported_and_one_does_not_create_fake_partial(trim):
    rows, start, end = wave_path()
    result = run(rows, start, end, parameters={**PARAMS, "trim_to": trim})
    if trim == 1:
        assert len(result["fills"]) == 1
        assert result["decisions"][-1]["episode_start"] is None
    else:
        assert result["fills"][1]["post_fill_exposure"] == pytest.approx(trim)


def test_partial_recovery_and_full_recovery_keep_stop_and_reset_highwater_only():
    rows, start, end = wave_path(full_recovery=True)
    result = run(rows, start, end, parameters={**PARAMS, "trim_to": 0})
    assert [f["target_exposure"] for f in result["fills"]] == [1, 0, .4, .7, 1]
    first_stop = result["fills"][2]["recovery_signal"]["structural_stop"]
    assert result["decisions"][-1]["structural_stop"] == first_stop
    assert result["decisions"][-1]["first_recovery_signal"]["signal_close"] == 106.5
    assert result["equity_curve"][-1]["highwater"] == rows[-1]["close"]


def test_recovery_stages_require_new_rising_close_above_original_signal_close():
    rows, start, end = below_ma_path()
    first_buy_day = run(rows, start, end)["fills"][2]["at"][:10]
    rows = [r for r in rows if r["date"] <= first_buy_day]
    rows[-1].update(close=180.5, low=179, high=183, provider_adjusted_close=180.5)
    result = run(rows, start, first_buy_day)
    assert result["fills"][-1]["target_exposure"] == .4
    assert result["decisions"][-1]["order_scheduled"] is False
    assert result["decisions"][-1]["target_exposure"] == .4


def test_benchmark_prices_costs_and_tr_are_identical_to_frozen_v1_helpers():
    rows, start, end = fixture()
    v1_params = {key: value for key, value in PARAMS.items() if key not in {
        "profit_atr", "breakout_sessions", "stop_buffer_atr"}}
    old = simulate_portfolio_swing({"QQQ": rows}, "QQQ", start, end, parameters=v1_params)
    result = run(rows, start, end)
    assert result["benchmark_price_curve"] == old["benchmark_price_curve"]
    assert result["benchmark_total_return_curve"] == old["benchmark_total_return_curve"]
    assert result["benchmark_fills"] == old["benchmark_fills"]
    assert result["metrics"]["net_account_return_pct"] == old["metrics"]["net_account_return_pct"]


def test_physical_cash_cost_basis_realized_unrealized_nav_conserve_unlevered_account():
    rows, start, end = wave_path(full_recovery=True)
    result = run(rows, start, end, parameters={**PARAMS, "trim_to": 0})
    cash, qty, basis, realized = 100.0, 0.0, 0.0, 0.0
    for fill in result["fills"]:
        if fill["side"] == "buy":
            cash -= fill["qty"]*fill["price"]+fill["fee"]
            basis += fill["qty"]*fill["price"]+fill["fee"]
            qty += fill["qty"]
        else:
            sold_basis = basis*fill["qty"]/qty
            proceeds = fill["qty"]*fill["price"]-fill["fee"]
            realized += proceeds-sold_basis
            cash += proceeds
            basis -= sold_basis
            qty -= fill["qty"]
        assert fill["cash"] == pytest.approx(cash, abs=1e-12)
        assert fill["holdings_qty"] == pytest.approx(qty, abs=1e-12)
        assert fill["cost_basis"] == pytest.approx(basis, abs=1e-12)
    metrics = result["metrics"]
    assert metrics["realized_account_pnl_pct"] == pytest.approx(realized)
    assert metrics["unrealized_account_pnl_pct"] == pytest.approx(qty*rows[-1]["close"]-basis)
    assert metrics["net_account_return_pct"] == pytest.approx(
        metrics["realized_account_pnl_pct"]+metrics["unrealized_account_pnl_pct"])
    assert all(p["cash"] >= 0 and p["qty"] >= 0 and p["exposure"] <= 1+1e-12
               for p in result["equity_curve"])


def test_future_numeric_values_not_read_and_rules_sources_are_policy_bound():
    class Future(dict):
        def get(self, key, *args):
            assert key == "date"
            return super().get(key, *args)

    rows, start, end = fixture()
    before = run(rows, start, end)
    assert run(rows+[Future(date="2026-01-02", close=object())], start, end) == before
    policy = build_structural_policy(PARAMS)
    assert policy["policy_sha256"] == _hash(policy["policy"])
    assert set(policy["policy"]["source_code_sha256"]) == {
        "portfolio_structure.py", "portfolio_swing.py", "structure.py", "sessions.py",
        "swing_optimizer.py"}
    v1_path = Path(module.__file__).with_name("portfolio_swing.py")
    expected = "771e920acec047706590a2b041d59a9818213e3fa55a39296f724bd8cf86bd03"
    assert hashlib.sha256(v1_path.read_bytes()).hexdigest() == expected


def test_missing_total_return_does_not_veto_price_book_or_invent_probability():
    rows, start, end = fixture()
    del rows[-1]["provider_adjusted_close"]
    result = run(rows, start, end)
    assert result["benchmark_total_return_curve"] is None
    assert result["metrics"]["alpha_to_total_return_pp"] is None
    assert result["strategy_dividends_included"] is False
    assert result["total_return_comparison_like_for_like"] is False
    assert result["fills"]


@pytest.mark.parametrize("change", [
    {"profit_atr": 1}, {"profit_atr": 5}, {"breakout_sessions": 2},
    {"stop_buffer_atr": .75}, {"trim_to": .2}, {"confirmation_sessions": 2},
    {"confirm_sessions": True}, {"entry_steps": [.2, .6, 1]},
])
def test_exact_frozen_policy_rejects_unbounded_or_unused_parameters(change):
    with pytest.raises(ValueError):
        build_structural_policy({**PARAMS, **change})


def test_policy_copies_config_exposes_cost_multiplier_and_full_cap_before_outcomes():
    original = copy.deepcopy(PARAMS)
    frozen = build_structural_policy(PARAMS)
    assert PARAMS == original and frozen["policy"]["max_exposure"] == 1
    stressed = build_structural_policy(PARAMS, cost_multiplier=2)
    assert stressed["policy_sha256"] != frozen["policy_sha256"]
    result = run(cost_multiplier=2)
    assert result["metrics"]["max_exposure_pct"] == pytest.approx(100)
    assert result["metrics"]["fees_paid"] > run()["metrics"]["fees_paid"]
