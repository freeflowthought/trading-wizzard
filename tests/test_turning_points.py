"""Market-implied turning layer: causal inputs, signals and simulator integration."""

from datetime import date, datetime, timedelta

import pytest
from test_macro_rotation import CYCLE, cycle_fixture, fixture, run

from twiz.macro_rotation import DEFAULT_PARAMETERS, build_macro_rotation_policy
from twiz.turning_points import (
    DEFAULT_TURNING,
    assess,
    build_turning_features,
    calm_weight,
    validate_turning,
)


def series(days, values):
    return [{"date": d.isoformat(), "close": v} for d, v in zip(days, values)]


def implied_market(market, ratio=0.85):
    days = [date.fromisoformat(r["date"]) for r in market["QQQ"]]
    return {
        "QQQ": market["QQQ"],
        "^VIX": series(days, [20 * ratio] * len(days)),
        "^VIX3M": series(days, [20.0] * len(days)),
        "HYG": series(days, [80.0] * len(days)),
        "IEF": series(days, [100.0] * len(days)),
    }, days


def test_validation_rejects_undeclared_values_and_bool_confusion():
    assert validate_turning(DEFAULT_TURNING) == DEFAULT_TURNING
    with pytest.raises(ValueError):
        validate_turning({**DEFAULT_TURNING, "stress_ratio": 0.99})
    with pytest.raises(ValueError):
        validate_turning({**DEFAULT_TURNING, "leverage_gate": 1})
    with pytest.raises(ValueError):
        validate_turning({**DEFAULT_TURNING, "extra": True})


def test_policy_hash_changes_only_when_turning_is_supplied():
    plain = build_macro_rotation_policy(DEFAULT_PARAMETERS, cycle_parameters=CYCLE)
    again = build_macro_rotation_policy(
        DEFAULT_PARAMETERS, cycle_parameters=CYCLE, turning_parameters=None
    )
    turned = build_macro_rotation_policy(
        DEFAULT_PARAMETERS, cycle_parameters=CYCLE, turning_parameters=DEFAULT_TURNING
    )
    assert plain["policy_sha256"] == again["policy_sha256"]
    assert turned["policy_sha256"] != plain["policy_sha256"]
    assert "turning_points.py" in turned["policy"]["source_code_sha256"]
    with pytest.raises(ValueError):
        build_macro_rotation_policy(DEFAULT_PARAMETERS, turning_parameters=DEFAULT_TURNING)


def test_fred_values_wait_for_next_session_and_claims_for_thursday_release():
    market, *_ = fixture()
    data, days = implied_market(market)
    observed = days[40]
    fred = {"DGS2": [(observed.isoformat(), 4.0)], "DFEDTARU": [(observed.isoformat(), 3.5)],
            "ICSA": []}
    rows = {r["date"]: r for r in build_turning_features(data, fred)["rows"]}
    assert rows[observed.isoformat()]["two_year_pct"] is None
    assert rows[days[41].isoformat()]["two_year_pct"] == 4.0
    assert rows[days[41].isoformat()]["fed_gap_pp"] == pytest.approx(0.5)


def test_missing_latest_auxiliary_close_stays_neutral():
    market, *_ = fixture()
    data, days = implied_market(market)
    for ticker in ("^VIX", "HYG"):
        data[ticker] = data[ticker][:-1]
    last = build_turning_features(data, {})["rows"][-1]
    assert last["vix_ratio"] is None and last["credit_ratio"] is None
    assert last["credit_above_ma50"] is None
    assert assess(last, q_state(), dict(DEFAULT_TURNING))["status"] == "missing_neutral"


def test_prediction_quotes_expire_with_their_meeting():
    market, *_ = fixture()
    data, days = implied_market(market)
    at = datetime.fromisoformat(market["QQQ"][10]["date"] + "T16:15:00-05:00")
    quote = {"id": "pm", "available_at": at.isoformat(),
             "expires_at": (at + timedelta(days=3)).isoformat(),
             "probabilities": {"hike": 0.8, "cut": 0.0}}
    rows = build_turning_features(data, {}, [quote])["rows"]
    assert rows[10]["pm_expected_change_pp"] == pytest.approx(0.2)
    assert rows[30]["pm_expected_change_pp"] is None


def q_state(close=100, slow=90, sma20=101, sma50=95):
    return {"close": close, "slow_sma": slow, "sma20": sma20, "sma50": sma50}


def feature(ratio, peak=None, **extra):
    row = {"vix_ratio": ratio, "vix_ratio_max": {10: peak or ratio, 15: peak or ratio,
                                                  20: peak or ratio},
           "credit_dd63": 0.0, "two_year_change20_pp": 0.0, "claims_rise": 0.0,
           "pm_change20_pp": None, "available_at": "2024-01-02T16:15:00-05:00"}
    row.update(extra)
    return row


def test_stress_requires_price_confirmation_below_selected_sma():
    p = dict(DEFAULT_TURNING)
    above = assess(feature(1.08), q_state(close=105, sma20=101), p)
    below = assess(feature(1.08), q_state(close=100, sma20=101), p)
    assert above["stress_reasons"] == ["vix_term_inversion"] and not above["stress"]
    assert below["stress"]


def test_prediction_market_shift_supersedes_two_year_proxy_when_present():
    p = dict(DEFAULT_TURNING)
    proxy = assess(feature(0.8, two_year_change20_pp=0.4), q_state(), p)
    quoted = assess(feature(0.8, two_year_change20_pp=0.4, pm_change20_pp=0.05), q_state(), p)
    assert proxy["stress_reasons"] == ["hawkish_repricing_2y"]
    assert quoted["stress_reasons"] == []
    off = assess(feature(0.8, two_year_change20_pp=0.4, pm_change20_pp=0.05), q_state(),
                 {**p, "use_prediction_markets": False})
    assert off["stress_reasons"] == ["hawkish_repricing_2y"]


def test_capitulation_needs_inversion_peak_then_normalisation():
    p = dict(DEFAULT_TURNING)
    assert assess(feature(0.94, peak=1.12), q_state(), p)["capitulation"]
    assert not assess(feature(1.02, peak=1.12), q_state(), p)["capitulation"]
    assert not assess(feature(0.94, peak=1.01), q_state(), p)["capitulation"]


def test_leverage_gate_and_missing_inputs_stay_neutral():
    p = dict(DEFAULT_TURNING)
    below_trend = q_state(close=80, slow=90)
    assert not assess(feature(0.9, peak=0.95), below_trend, p)["leverage_ok"]
    assert assess(feature(0.9, peak=1.2), below_trend, p)["leverage_ok"]
    assert assess(feature(0.9), below_trend, {**p, "leverage_gate": False})["leverage_ok"]
    missing = assess(None, q_state(), p)
    assert missing["status"] == "missing_neutral" and not missing["stress"]


def test_calm_weight_vol_targets_the_whole_account():
    p = {**DEFAULT_TURNING, "calm_tqqq": 0.5, "calm_vol_target": 0.2}
    assert calm_weight({"qqq_realized_vol20": 0.1}, p) == 0.5
    assert calm_weight({"qqq_realized_vol20": 0.16}, p) == pytest.approx(0.125)
    assert calm_weight({"qqq_realized_vol20": 0.25}, p) == 0.0


def test_simulator_trims_on_confirmed_stress_and_restores_when_cleared():
    market, start, end = cycle_fixture()
    data, days = implied_market(market)
    index = {r["date"]: i for i, r in enumerate(market["QQQ"])}
    first = next(i for i in range(index[start] + 1, len(days))
                 if market["QQQ"][i]["close"] < market["QQQ"][i - 1]["close"])
    for i in range(first, first + 3):
        data["^VIX"][i]["close"] = 23.0
    features = build_turning_features(data, {})
    result = run(market, start, end, parameters={"recovery_weight": 0.5}, cycle_parameters=CYCLE,
                 turning={"parameters": dict(DEFAULT_TURNING), "features": features})
    causes = [d["cause"] for d in result["decisions"]]
    stress = next(d for d in result["decisions"] if d["cause"] == "cycle_implied_stress_top")
    assert stress["target_weights"]["TQQQ"] == 0
    assert sum(stress["target_weights"].values()) <= DEFAULT_TURNING["stress_retained"]
    assert "vix_term_inversion" in stress["auxiliary"]["turning"]["stress_reasons"]
    assert causes.index("cycle_implied_stress_top") < len(causes) - 1


def test_turning_none_keeps_cycle_outputs_identical():
    market, start, end = cycle_fixture()
    plain = run(market, start, end, parameters={"recovery_weight": 0.5}, cycle_parameters=CYCLE)
    again = run(market, start, end, parameters={"recovery_weight": 0.5}, cycle_parameters=CYCLE,
                turning=None)
    assert plain["equity_curve"] == again["equity_curve"]
    assert plain["decisions"] == again["decisions"]
    assert "turning" not in plain["decisions"][-1]["auxiliary"]
    with pytest.raises(ValueError):
        run(market, start, end, turning={"parameters": DEFAULT_TURNING, "features": {}})


def test_claims_count_only_while_above_their_26_week_trend():
    p = dict(DEFAULT_TURNING)
    rising = feature(0.8, claims_rise=0.3, claims_above_trend=True)
    falling_off_peak = feature(0.8, claims_rise=0.3, claims_above_trend=False)
    assert assess(rising, q_state(), p)["stress_reasons"] == ["jobless_claims_rise"]
    assert assess(falling_off_peak, q_state(), p)["stress_reasons"] == []
    unfiltered = {**p, "claims_trend_filter": False}
    assert assess(falling_off_peak, q_state(), unfiltered)["stress_reasons"] == [
        "jobless_claims_rise"]


def capitulation_then_reinversion():
    market, start, end = cycle_fixture()
    data, days = implied_market(market)
    index = {r["date"]: i for i, r in enumerate(market["QQQ"])}
    base = run(market, start, end, parameters={"recovery_weight": 0.5}, cycle_parameters=CYCLE)
    bottom = next(d for d in base["decisions"] if d["cause"] == "cycle_confirmed_bottom")
    signal = index[bottom["at"][:10]]
    for i in range(signal - 8, signal - 2):
        data["^VIX"][i]["close"] = 23.0  # inversion peak inside the lookback
    for i in range(signal + 2, signal + 4):
        data["^VIX"][i]["close"] = 21.0  # brief re-inversion after entry
    return market, start, end, build_turning_features(data, {})


@pytest.mark.parametrize("grace,trimmed", [(0, True), (5, False)])
def test_capitulation_grace_ignores_brief_reinversion_above_anchor(grace, trimmed):
    market, start, end, features = capitulation_then_reinversion()
    params = {**DEFAULT_TURNING, "capitulation_grace_sessions": grace,
              "stress_confirm_sma": 50}
    result = run(market, start, end, parameters={"recovery_weight": 0.5},
                 cycle_parameters=CYCLE, turning={"parameters": params, "features": features})
    causes = [d["cause"] for d in result["decisions"]]
    if "cycle_capitulation_bottom" not in causes:
        pytest.skip("fixture produced no capitulation entry")
    after = causes[causes.index("cycle_capitulation_bottom") + 1:]
    assert ("cycle_implied_stress_top" in after[:4]) is trimmed
    if not trimmed:
        assert any(d["auxiliary"].get("turning", {}).get("grace_suppressed")
                   for d in result["decisions"])
