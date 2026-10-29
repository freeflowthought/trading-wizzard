"""Synthetic multi-sector and actual-series leveraged recovery tests only."""

import hashlib
import json
from datetime import date, timedelta

import pytest

from twiz.macro_rotation import (
    DEFAULT_PARAMETERS,
    DEFENSIVE,
    UNIVERSE,
    _auxiliary,
    _clock,
    _next,
    _snapshot,
    _utc,
    build_macro_rotation_policy,
    simulate_macro_rotation,
)
from twiz.sessions import is_us_equity_session
from twiz.swing_optimizer import _hash


def fixture(n=300):
    days, day = [], date(2023, 1, 3)
    while len(days) < n:
        if is_us_equity_session(day):
            days.append(day)
        day += timedelta(days=1)
    market = {}
    for asset in UNIVERSE:
        drift = 0.003 if asset == "SMH" else 0.001 if asset == "QQQ" else 0.0005
        market[asset] = []
        for i, day in enumerate(days):
            price = 100 * (1 + drift) ** i
            market[asset].append(
                {
                    "date": day.isoformat(),
                    "open": price * 0.999,
                    "high": price * 1.005,
                    "low": price * 0.995,
                    "close": price,
                    "volume": 1000,
                    "provider_adjusted_close": price,
                    "source": "yfinance",
                    "session_complete": True,
                    "price_basis": "test_split_normalized_units",
                }
            )
    return market, days[259].isoformat(), days[-1].isoformat()


def run(market=None, start=None, end=None, **kwargs):
    if market is None:
        market, start, end = fixture()
    parameters = {**DEFAULT_PARAMETERS, **kwargs.pop("parameters", {})}
    return simulate_macro_rotation(market, start, end, parameters=parameters, **kwargs)


def recovery_fixture(extra=0):
    market, start, _ = fixture(269 + extra)
    for i, row in enumerate(market["QQQ"]):
        price = 200
        row.update(
            open=price, high=price + 1, low=price - 1, close=price, provider_adjusted_close=price
        )
        if i >= 260:
            values = [
                (198, 199, 189, 190),
                (190, 191, 179, 180),
                (175, 178, 170, 175),
                (175.5, 177, 172, 176),
                (178, 180, 174, 179),
                (180, 182, 176, 181),
                (181.5, 183, 179, 182),
                (182.5, 184, 180, 183),
                (183.5, 185, 181, 184),
            ]
            o, h, lo, c = values[i - 260] if i - 260 < len(values) else (182, 183, 180, 182)
            row.update(open=o, high=h, low=lo, close=c, provider_adjusted_close=c)
    # Actual independent TQQQ input series, deliberately not three times QQQ returns.
    for row in market["TQQQ"]:
        row.update(open=10, high=10.2, low=9.8, close=10, provider_adjusted_close=10)
    return market, start, market["QQQ"][-1]["date"]


def fresh_issuers(available, spend=-10, revenue=-1, acceleration=None):
    result = {}
    for ticker in ("MSFT", "GOOGL"):
        facts = {
            kind: {"available_at_proxy": available, "refs": [ticker + kind]}
            for kind in ("capex", "revenue")
        }
        result[ticker] = {
            "status": "available",
            "current": facts,
            "prior": facts,
            "cash_capex_yoy_pct": spend,
            "revenue_yoy_pct": revenue,
        }
        if acceleration is not None:
            result[ticker].update(
                previous_same_duration=facts,
                cash_capex_yoy_acceleration_pp=acceleration[0],
                revenue_yoy_acceleration_pp=acceleration[1],
            )
    return result


def test_fixed_universe_shared_account_separate_benchmarks_and_costs():
    result = run()
    assert len(UNIVERSE) == 12 and "XLY" not in UNIVERSE
    assert result["equity_curve"][0]["equity"] == 100
    for name in (
        "benchmark_price_curve",
        "benchmark_tqqq_price_curve",
        "benchmark_static_blend_curve",
        "benchmark_sector_control_curve",
    ):
        assert result[name][0]["equity"] == 100
    for point in result["equity_curve"]:
        assert point["cash"] >= 0 and all(q >= 0 for q in point["positions"].values())
        assert point["exposure"] <= 1 + 1e-12
    assert all(f["at"] > f["signal_at"] for f in result["fills"])
    assert result["metrics"]["fees_paid"] == pytest.approx(sum(f["fee"] for f in result["fills"]))
    assert result["metrics"]["net_account_return_pct"] == pytest.approx(
        result["metrics"]["realized_account_pnl_pct"]
        + result["metrics"]["unrealized_account_pnl_pct"]
    )


def test_weekly_reviews_use_last_completed_session_not_calendar_future_prices():
    result = run()
    reviews = [d for d in result["decisions"] if d["weekly_review"]]
    assert reviews[0]["at"] == result["decisions"][0]["at"]
    for review in reviews[1:]:
        day = date.fromisoformat(review["at"][:10])
        assert _next(day).isocalendar()[:2] != day.isocalendar()[:2]


def test_bottom_requires_multiple_completed_days_higher_low_and_breakout_then_real_tqqq_fill():
    market, start, end = recovery_fixture()
    result = run(market, start, end, parameters={"recovery_weight": 1})
    signal = next(d for d in result["decisions"] if d["cause"] == "confirmed_bottom")
    q = signal["features"]["QQQ"]
    assert q["anchor_age"] >= 2 and q["higher_low"] and q["rising"]
    assert q["close"] > q["prior3_high"] and q["drawdown"] >= 0.08
    assert signal["recovery_stop"] is None
    buy = next(f for f in result["fills"] if f["asset"] == "TQQQ" and f["side"] == "buy")
    assert buy["price"] == pytest.approx(10 * 1.0002)
    assert buy["signal_at"] == signal["at"] and buy["at"] > signal["at"]
    assert result["equity_curve"][-1]["positions"]["TQQQ"] > 0
    assert result["metrics"]["benchmark_tqqq_price_return_pct"] == pytest.approx(
        100 / (1.0002 * 1.0001) - 100
    )


def test_tqqq_gain_migrates_to_qqq_next_open_not_same_day_and_does_not_reenter_old_campaign():
    market, start, end = recovery_fixture(extra=4)
    market["TQQQ"][267].update(close=11.5, high=11.7, low=9.8, provider_adjusted_close=11.5)
    for row in market["TQQQ"][268:]:
        row.update(open=12, close=12, high=12.2, low=11.8, provider_adjusted_close=12)
    result = run(market, start, end, parameters={"recovery_weight": 0.5})
    signal = next(d for d in result["decisions"] if d["cause"] == "tqqq_to_qqq_gain")
    sell = next(
        f
        for f in result["fills"]
        if f["asset"] == "TQQQ" and f["cause"] == "tqqq_to_qqq_gain" and f["side"] == "sell"
    )
    assert signal["tqqq_gain"] >= 0.1 and sell["at"] > signal["at"]
    assert sell["raw_open"] == 12
    assert any(
        f["asset"] == "QQQ" and f["side"] == "buy" and f["at"] == sell["at"]
        for f in result["fills"]
    )
    assert not any(
        f["asset"] == "TQQQ" and f["side"] == "buy" and f["at"] > sell["at"]
        for f in result["fills"]
    )


def test_tqqq_sma20_and_time_migration_have_separate_auditable_causes():
    market, start, end = recovery_fixture()
    market["QQQ"][267].update(open=204, close=205, high=206, low=203, provider_adjusted_close=205)
    result = run(market, start, end, parameters={"recovery_weight": 0.5})
    assert any(d["cause"] == "tqqq_to_qqq_sma20" for d in result["decisions"])
    market, start, end = recovery_fixture(extra=14)
    result = run(
        market, start, end, parameters={"recovery_weight": 0.5, "recovery_max_sessions": 10}
    )
    signal = next(d for d in result["decisions"] if d["cause"] == "tqqq_to_qqq_time")
    assert signal["holding_sessions"] == 10


def test_failed_recovery_gap_cancels_buy_and_does_not_install_stop_or_fake_fill():
    market, start, end = recovery_fixture()
    before = run(market, start, end, parameters={"recovery_weight": 1})
    signal = next(d for d in before["decisions"] if d["cause"] == "confirmed_bottom")
    stop = signal["proposed_recovery"]["qqq_stop"]
    fill_day = _next(date.fromisoformat(signal["at"][:10])).isoformat()
    row = next(r for r in market["QQQ"] if r["date"] == fill_day)
    row.update(
        open=stop - 1, close=stop - 1, high=stop, low=stop - 2, provider_adjusted_close=stop - 1
    )
    result = run(market, start, fill_day, parameters={"recovery_weight": 1})
    assert result["unfilled_orders"][-1]["reason"] == "QQQ_open_broke_frozen_recovery_stop"
    assert not any(f["asset"] == "TQQQ" for f in result["fills"])
    assert result["decisions"][-1]["recovery_stop"] is None
    assert result["decisions"][-1]["risk_off"] is True


def test_favorable_event_only_allows_bounded_probe_before_full_price_confirmation():
    market, start, end = recovery_fixture()
    day = market["QQQ"][263]["date"]
    events = [
        {
            "id": "positive",
            "available_at": _clock(date.fromisoformat(day)),
            "direction": 0.8,
            "source_refs": [{"url": "https://example.test/source"}],
        }
    ]
    result = run(
        market, start, end, parameters={"recovery_weight": 1, "use_events": True}, event_rows=events
    )
    probe = next(d for d in result["decisions"] if d["cause"] == "event_bounded_probe")
    assert probe["features"]["QQQ"]["bottom"] is False
    assert probe["target_weights"]["TQQQ"] == 0.25
    assert sum(probe["target_weights"].values()) <= 1


def test_adverse_event_can_rotate_early_then_carries_risk_until_price_repair_not_one_green():
    market, start, end = fixture()
    events = [
        {
            "id": "adverse",
            "available_at": _clock(date.fromisoformat(start)),
            "direction": -0.8,
            "source_refs": ["https://example.test/known-source"],
        }
    ]
    result = run(market, start, end, parameters={"use_events": True}, event_rows=events)
    first = result["decisions"][0]
    assert first["cause"] == "adverse_event" and first["features"]["QQQ"]["trend_broken"] is False
    assert first["target_weights"]["QQQ"] == first["target_weights"]["SMH"] == 0
    assert all(a in DEFENSIVE for a in UNIVERSE if first["target_weights"][a] > 0)
    assert sum(d["cause"] == "adverse_event" for d in result["decisions"]) == 1
    plain = run(market, start, end)
    assert result["benchmark_sector_control_curve"] == plain["benchmark_sector_control_curve"]


def test_future_and_stale_auxiliary_values_are_not_read_before_availability_checks():
    class UnreadValue(dict):
        def get(self, key, *args):
            assert key == "available_at", f"Read unqualified value {key}"
            return super().get(key, *args)

    origin = _utc("2026-06-26T16:15:00-04:00")
    future = UnreadValue(
        available_at="2027-01-01T00:00:00Z", direction=object(), aggregate=object()
    )
    stale = UnreadValue(available_at="2026-01-01T00:00:00Z", direction=object())
    p = {**DEFAULT_PARAMETERS, "use_events": True, "use_capex": True}
    result = _auxiliary([future, stale], [future], origin, p)
    assert result["event_status"] == "stale_neutral" and result["direction"] is None
    assert result["capex_status"] == "missing_neutral" and result["smh_restricted"] is False


def test_event_ttl_counts_completed_sessions_and_source_expiry_is_respected():
    event = {
        "available_at": "2026-06-18T16:15:00-04:00",
        "direction": -0.8,
        "source_refs": ["source"],
    }
    assert _snapshot([event], _utc("2026-06-26T16:15:00-04:00"), 5)[1] == "available"
    assert _snapshot([event], _utc("2026-06-29T16:15:00-04:00"), 5)[1] == "stale_neutral"
    event["expires_at"] = "2026-06-25T16:15:00-04:00"
    aux = _auxiliary(
        [event], None, _utc("2026-06-26T16:15:00-04:00"), {**DEFAULT_PARAMETERS, "use_events": True}
    )
    assert aux["direction"] is None and aux["event_status"] == "expired_or_invalid_expiry_neutral"


@pytest.mark.parametrize(
    "spend,revenue,blocked", [(-10, -1, True), (-10, 1, True), (10, -1, False)]
)
def test_capex_only_restricts_smh_eligibility_and_missing_is_neutral(spend, revenue, blocked):
    market, start, end = fixture()
    capex = [
        {
            "available_at": _clock(date.fromisoformat(start)),
            "source_refs": ["filing"],
            "issuers": fresh_issuers(_clock(date.fromisoformat(start)), spend, revenue),
            "aggregate": {
                "status": "available",
                "median_cash_capex_yoy_pct": spend,
                "median_revenue_yoy_pct": revenue,
            },
        }
    ]
    result = run(market, start, end, parameters={"use_capex": True}, capex_rows=capex)
    first = result["decisions"][0]
    assert first["auxiliary"]["smh_restricted"] is blocked
    assert first["risk_off"] is False and first["target_weights"]["QQQ"] >= 0.5
    if blocked:
        assert first["target_weights"]["SMH"] == 0
    assert result["auxiliary_sources_used"]


def test_tqqq_split_normalized_scaling_changes_units_not_signals_nav_or_benchmarks():
    market, start, end = recovery_fixture()
    before = run(market, start, end, parameters={"recovery_weight": 0.5})
    for row in market["TQQQ"]:
        for key in ("open", "high", "low", "close", "provider_adjusted_close"):
            row[key] *= 2
    after = run(market, start, end, parameters={"recovery_weight": 0.5})
    for key in (
        "equity_curve",
        "benchmark_price_curve",
        "benchmark_tqqq_price_curve",
        "benchmark_static_blend_curve",
    ):
        assert [p["equity"] for p in after[key]] == pytest.approx(
            [p["equity"] for p in before[key]]
        )
    assert after["equity_curve"][-1]["positions"]["TQQQ"] == pytest.approx(
        before["equity_curve"][-1]["positions"]["TQQQ"] / 2
    )


def test_actual_tqqq_history_is_required_and_future_prices_are_not_read():
    market, start, end = fixture()
    del market["TQQQ"]
    with pytest.raises(ValueError):
        run(market, start, end)

    class Future(dict):
        def get(self, key, *args):
            assert key == "date"
            return super().get(key, *args)

    market, start, end = fixture()
    before = run(market, start, end)
    for rows in market.values():
        rows.append(Future(date=_next(date.fromisoformat(end)).isoformat(), close=object()))
    assert run(market, start, end) == before


def test_policy_exact_hash_and_disabled_auxiliary_counterfactual():
    frozen = build_macro_rotation_policy(DEFAULT_PARAMETERS)
    assert frozen["policy_sha256"] == _hash(frozen["policy"])
    assert set(frozen["policy"]["source_code_sha256"]) == {
        "macro_rotation.py",
        "structure.py",
        "sessions.py",
        "swing_optimizer.py",
    }
    with pytest.raises(ValueError):
        build_macro_rotation_policy({**DEFAULT_PARAMETERS, "unknown": 1})

    class NeverRead(dict):
        def get(self, key, *args):
            raise AssertionError("Disabled auxiliary input inspected")

    baseline = run()
    result = run(event_rows=[NeverRead()], capex_rows=[NeverRead()])
    assert result == baseline


def test_event_families_keep_active_tariff_shock_despite_newer_weak_fed_quote():
    origin = _utc("2026-06-26T16:15:00-04:00")
    events = [
        {
            "id": "tariff",
            "correlation_family": "tariff_policy",
            "available_at": "2026-06-25T16:15:00-04:00",
            "direction": -0.5,
            "source_refs": ["known-tariff-source"],
        },
        {
            "id": "fed-quote",
            "episode_id": "fed-july",
            "correlation_family": "monetary_policy",
            "available_at": "2026-06-26T16:15:00-04:00",
            "direction": 0.03,
            "source_refs": ["quote"],
        },
    ]
    p = {**DEFAULT_PARAMETERS, "use_events": True, "event_threshold": 0.5}
    a = _auxiliary(events, None, origin, p)
    b = _auxiliary(events[::-1], None, origin, p)
    assert a == b and a["direction"] == -0.5 and a["adverse"] is True
    assert a["event_key"][0] == "tariff"


@pytest.mark.parametrize("same_family", [True, False])
def test_conflicting_equal_event_clocks_or_opposing_max_families_are_order_invariant_neutral(
    same_family,
):
    clock = "2026-06-26T16:15:00-04:00"
    events = [
        {
            "id": "a",
            "correlation_family": "one",
            "available_at": clock,
            "direction": -0.5,
            "source_refs": ["a"],
        },
        {
            "id": "b",
            "correlation_family": "one" if same_family else "two",
            "available_at": clock,
            "direction": 0.5,
            "source_refs": ["b"],
        },
    ]
    p = {**DEFAULT_PARAMETERS, "use_events": True, "event_threshold": 0.5}
    a, b = (_auxiliary(value, None, _utc(clock), p) for value in (events, events[::-1]))
    assert a == b and a["direction"] is None and not a["adverse"] and not a["favorable"]


def test_daily_unchanged_adverse_episode_does_not_restart_bottom_confirmation_clock():
    market, start, end = recovery_fixture()
    events = [
        {
            "id": "daily-" + row["date"],
            "episode_id": "same-fed-meeting",
            "correlation_family": "monetary_policy",
            "available_at": _clock(date.fromisoformat(row["date"])),
            "direction": -0.5,
            "source_refs": ["observed"],
        }
        for row in market["QQQ"][259:]
    ]
    result = run(
        market,
        start,
        end,
        event_rows=events,
        parameters={"use_events": True, "event_threshold": 0.5, "recovery_weight": 0.5},
    )
    assert sum(d["cause"] == "adverse_event" for d in result["decisions"]) == 1
    assert any(d["cause"] == "confirmed_bottom" for d in result["decisions"])


@pytest.mark.parametrize("age,future,count", [(365, False, 2), (0, True, 2), (0, False, 1)])
def test_daily_capex_snapshot_does_not_refresh_old_future_or_one_issuer_facts(age, future, count):
    clock = _utc("2026-06-26T16:15:00-04:00")
    available = (clock + timedelta(days=1) if future else clock - timedelta(days=age)).isoformat()
    issuers = fresh_issuers(available)
    if count == 1:
        issuers.pop("GOOGL")
    row = {
        "available_at": clock.isoformat(),
        "issuers": issuers,
        "aggregate": {
            "status": "available",
            "median_cash_capex_yoy_pct": -100,
            "median_revenue_yoy_pct": -100,
        },
    }
    result = _auxiliary(None, [row], clock, {**DEFAULT_PARAMETERS, "use_capex": True})
    assert result["smh_restricted"] is False
    assert result["capex_status"] == "insufficient_fresh_issuers_neutral"


@pytest.mark.parametrize(
    "acceleration,expected", [((-10, -1), True), ((10, -1), False), ((-10, 1), False)]
)
def test_positive_growth_with_joint_same_ytd_yoy_deceleration_is_smh_only_caution(
    acceleration, expected
):
    clock = "2026-06-26T16:15:00-04:00"
    row = {"available_at": clock, "issuers": fresh_issuers(clock, 30, 10, acceleration)}
    result = _auxiliary(None, [row], _utc(clock), {**DEFAULT_PARAMETERS, "use_capex": True})
    assert result["smh_restricted"] is expected
    assert result["smh_restriction_reason"] == (
        "joint_same_ytd_yoy_deceleration"
        if expected
        else "no_cash_capex_contraction_or_joint_deceleration"
    )


def test_unleveraged_below_long_ma_price_repair_has_own_frozen_stop_and_latch():
    market, start, _ = fixture(269)
    values = [190, 180, 181, 182, 183, 185, 190, 195, 196]
    for i, row in enumerate(market["QQQ"]):
        price = values[i - 260] if i >= 260 else 200
        row.update(
            open=price, high=price + 1, low=price - 1, close=price, provider_adjusted_close=price
        )
    result = run(market, start, market["QQQ"][-1]["date"], parameters={"drawdown_threshold": 0.12})
    repair = next(d for d in result["decisions"] if d["cause"] == "unleveraged_price_repair")
    assert repair["features"]["QQQ"]["close"] < repair["features"]["QQQ"]["slow_sma"]
    assert repair["proposed_recovery"] is not None
    after = result["decisions"][-1]
    assert after["features"]["QQQ"]["rising"] is True
    assert after["risk_armed"] is False and after["recovery_stop"] is not None
    assert after["cause"] != "long_trend_risk"


def test_missing_prior_filing_clock_is_neutral_even_if_current_is_fresh_and_growth_claimed():
    clock = "2026-06-26T16:15:00-04:00"
    issuers = fresh_issuers(clock)
    for issuer in issuers.values():
        issuer["prior"] = {"capex": {}, "revenue": {}}
    result = _auxiliary(
        None,
        [{"available_at": clock, "issuers": issuers}],
        _utc(clock),
        {**DEFAULT_PARAMETERS, "use_capex": True},
    )
    assert result["smh_restricted"] is False
    assert result["capex_status"] == "insufficient_fresh_issuers_neutral"


def test_filing_age_uses_130_calendar_days_inclusive_across_dst():
    clock = "2026-01-13T16:15:00-05:00"
    fact_clock = "2025-09-05T16:15:00-04:00"
    assert (_utc(clock) - _utc(fact_clock)).total_seconds() > 130 * 24 * 3600
    row = {"available_at": clock, "issuers": fresh_issuers(fact_clock)}
    result = _auxiliary(None, [row], _utc(clock), {**DEFAULT_PARAMETERS, "use_capex": True})
    assert result["smh_restricted"] is True
    assert result["capex_source"]["available_at"] == _utc(fact_clock).isoformat()


def test_local_top_carries_defensive_state_past_next_green_and_weekly_reviews():
    market, start, _ = fixture(268)
    values = [(100, 101, 99, 100)] * 260 + [
        (100, 106, 99, 105),
        (105, 111, 104, 110),
        (109, 110, 107, 108),
        (108.5, 110, 108, 109),
        (110, 110, 107, 108),
        (107, 108, 104, 105),
        (104, 105, 103, 104),
        (104, 105.2, 103.5, 104.5),
    ]
    for row, (o, h, lo, c) in zip(market["QQQ"], values):
        row.update(open=o, high=h, low=lo, close=c, provider_adjusted_close=c)
    result = run(market, start, market["QQQ"][-1]["date"], parameters={"top_breakout_sessions": 3})
    top = next(d for d in result["decisions"] if d["cause"] == "local_top")
    assert top["wave_profit_atr"] >= DEFAULT_PARAMETERS["top_profit_atr"]
    last = result["decisions"][-1]
    assert last["features"]["QQQ"]["rising"] is True and last["risk_off"] is True
    assert last["target_weights"]["QQQ"] == last["target_weights"]["SMH"] == 0


CYCLE = {
    "top_retained": 0.7,
    "bottom_min_age": 5,
    "bottom_trend_sessions": 20,
    "migrate_gain": 0.1,
    "max_holding_sessions": 20,
}


def cycle_fixture(extra=40):
    market, start, _ = fixture(269 + extra)
    early = [190, 180, 175, 176, 179, 181, 185, 190, 195]
    for i, row in enumerate(market["QQQ"]):
        price = (
            early[i - 260]
            if 260 <= i < 269
            else min(198, 195 + 0.5 * (i - 268))
            if i >= 269
            else 200
        )
        low = 170 if i == 262 else price - 0.4
        row.update(
            open=price, high=price + 0.2, low=low, close=price, provider_adjusted_close=price
        )
    for row in market["TQQQ"]:
        row.update(open=10, high=10.2, low=9.8, close=10, provider_adjusted_close=10)
    return market, start, market["QQQ"][-1]["date"]


def test_cycle_bull_baseline_is_qqq100_and_only_two_assets_can_be_funded():
    result = run(cycle_parameters=CYCLE)
    first = result["fills"][0]
    assert first["asset"] == "QQQ"
    assert result["decisions"][0]["target_weights"]["QQQ"] == 1
    assert all(f["asset"] in {"QQQ", "TQQQ"} for f in result["fills"])
    assert result["policy"]["version"] == "single-index-cycle-1"
    assert result["metrics"]["alpha_to_price_pp"] == pytest.approx(0, abs=1e-10)


def test_cycle_top_retains70pct_qqq_instead_of_defensive_sector_rotation():
    market, start, _ = fixture(268)
    values = [(100, 101, 99, 100)] * 260 + [
        (100, 106, 99, 105),
        (105, 111, 104, 110),
        (109, 110, 107, 108),
        (108.5, 110, 108, 109),
        (110, 110, 107, 108),
        (107, 108, 104, 105),
        (104, 105, 103, 104),
        (104, 105.2, 103.5, 104.5),
    ]
    for row, (o, h, lo, c) in zip(market["QQQ"], values):
        row.update(open=o, high=h, low=lo, close=c, provider_adjusted_close=c)
    result = run(
        market,
        start,
        market["QQQ"][-1]["date"],
        parameters={"top_breakout_sessions": 3},
        cycle_parameters=CYCLE,
    )
    top = next(d for d in result["decisions"] if d["cause"] == "cycle_local_top")
    assert top["target_weights"]["QQQ"] == 0.7
    assert sum(top["target_weights"].values()) == 0.7
    assert all(f["asset"] == "QQQ" for f in result["fills"])
    assert result["decisions"][-1]["target_weights"]["QQQ"] == 0.7


@pytest.mark.parametrize("age,trend", [(5, 20), (10, 50)])
def test_cycle_tqqq_bottom_waits_for_aged_base_and_selected_trend(age, trend):
    market, start, end = cycle_fixture()
    result = run(
        market,
        start,
        end,
        parameters={"recovery_weight": 0.5},
        cycle_parameters={**CYCLE, "bottom_min_age": age, "bottom_trend_sessions": trend},
    )
    signal = next(d for d in result["decisions"] if d["cause"] == "cycle_confirmed_bottom")
    q = signal["features"]["QQQ"]
    assert q["anchor_age"] >= age and q["close"] > q["cycle_bottom_sma"]
    assert q["close"] > q["prior3_high"] and q["higher_low"]
    assert signal["target_weights"]["TQQQ"] == 0.5
    assert sum(signal["target_weights"].values()) == 1


def test_cycle_does_not_immediately_migrate_on_sma20_and_uses20session_deadline():
    market, start, end = cycle_fixture()
    result = run(market, start, end, parameters={"recovery_weight": 0.5}, cycle_parameters=CYCLE)
    held = [d for d in result["decisions"] if d["tqqq_entry_at"] is not None]
    assert len(held) >= 20
    assert all(d["features"]["QQQ"]["close"] >= d["features"]["QQQ"]["sma20"] for d in held[:5])
    assert not any("sma20" in d["cause"] for d in result["decisions"])
    migration = next(d for d in result["decisions"] if d["cause"] == "cycle_tqqq_time_to_qqq")
    assert migration["holding_sessions"] == 20
    sell = next(
        f
        for f in result["fills"]
        if f["cause"] == "cycle_tqqq_time_to_qqq" and f["asset"] == "TQQQ"
    )
    assert sell["at"] > migration["at"]


def test_cycle_gain_migrates_to_qqq_and_failed_stop_exits_entire_cycle_to_cash():
    market, start, end = cycle_fixture()
    probe = run(market, start, end, parameters={"recovery_weight": 0.5}, cycle_parameters=CYCLE)
    first_buy = next(f for f in probe["fills"] if f["asset"] == "TQQQ" and f["side"] == "buy")
    day = next(i for i, r in enumerate(market["TQQQ"]) if r["date"] == first_buy["at"][:10])
    for row in market["TQQQ"][day + 2 :]:
        row.update(open=12, high=12.2, low=11.8, close=12, provider_adjusted_close=12)
    result = run(market, start, end, parameters={"recovery_weight": 0.5}, cycle_parameters=CYCLE)
    gain = next(d for d in result["decisions"] if d["cause"] == "cycle_tqqq_gain_to_qqq")
    assert gain["tqqq_gain"] >= 0.1 and gain["target_weights"]["QQQ"] == 1
    market, start, end = cycle_fixture()
    stop = probe["decisions"][day - 259]["recovery_stop"]
    assert stop is not None
    row = market["QQQ"][day + 2]
    row.update(
        open=stop, high=stop + 1, low=stop - 2, close=stop - 1, provider_adjusted_close=stop - 1
    )
    result = run(market, start, end, parameters={"recovery_weight": 1}, cycle_parameters=CYCLE)
    failure = next(d for d in result["decisions"] if d["cause"] == "cycle_structural_stop")
    assert sum(failure["target_weights"].values()) == 0
    after = next(
        p
        for p in result["equity_curve"]
        if p["at"][:10] == _next(date.fromisoformat(failure["at"][:10])).isoformat()
    )
    assert after["exposure"] == 0


def test_cycle_still_honors_adverse_events_and_capex_is_explicitly_not_applicable():
    market, start, end = fixture()
    events = [
        {
            "id": "shock",
            "available_at": _clock(date.fromisoformat(start)),
            "direction": -0.5,
            "source_refs": ["known"],
        }
    ]
    result = run(
        market,
        start,
        end,
        parameters={"use_events": True, "event_threshold": 0.5, "use_capex": True},
        event_rows=events,
        cycle_parameters=CYCLE,
    )
    assert result["decisions"][0]["cause"] == "cycle_adverse_event"
    assert result["decisions"][0]["target_weights"]["QQQ"] == 0.7
    assert all(
        d["auxiliary"]["capex_status"].startswith("not_applicable") for d in result["decisions"]
    )


@pytest.mark.parametrize(
    "case,expected",
    [
        ("plain", "cd99a586e0eeebaf79d183d11e1c56d8dc824d1abbd01ec3d9212517f538fd71"),
        ("tqqq", "edf0bbf46bfaadaad465cd7ddf184afc69cf17a3e64cfa61689163bf13fc5113"),
    ],
)
def test_cycle_none_preserves_pre_extension_all_numeric_outputs(case, expected):
    arguments = fixture() if case == "plain" else recovery_fixture(extra=14)
    result = run(
        *arguments,
        parameters={} if case == "plain" else {"recovery_weight": 0.5},
        cycle_parameters=None,
    )
    fields = {
        k: v
        for k, v in result.items()
        if k not in ("policy", "policy_sha256", "source_code_sha256")
    }
    digest = hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert digest == expected


def test_cycle_favorable_probe_is_bounded_before_five_day_price_confirmation():
    market, start, end = cycle_fixture(extra=1)
    clock = _clock(date.fromisoformat(market["QQQ"][263]["date"]))
    event = {"id": "favorable", "available_at": clock, "direction": 0.5, "source_refs": ["known"]}
    result = run(
        market,
        start,
        end,
        parameters={"use_events": True, "event_threshold": 0.5, "recovery_weight": 1},
        event_rows=[event],
        cycle_parameters=CYCLE,
    )
    probe = next(d for d in result["decisions"] if d["cause"] == "cycle_event_probe")
    assert probe["features"]["QQQ"]["cycle_bottom_ready"] is False
    assert probe["target_weights"]["TQQQ"] == 0.25
    assert sum(probe["target_weights"].values()) <= 1


def test_cycle_new_adverse_episode_does_not_rebalance_unchanged_retained_target():
    market, start, end = fixture(262)
    for i, row in enumerate(market["QQQ"]):
        price = 105 if i == 259 else 104.8 if i == 260 else 104.9 if i == 261 else 100
        row.update(
            open=price,
            high=price + 0.2,
            low=price - 0.4,
            close=price,
            provider_adjusted_close=price,
        )
    second = _clock(date.fromisoformat(market["QQQ"][261]["date"]))
    events = [
        {
            "id": "shock-a",
            "episode_id": "a",
            "correlation_family": "risk",
            "available_at": _clock(date.fromisoformat(start)),
            "direction": -0.5,
            "source_refs": ["a"],
        },
        {
            "id": "shock-b",
            "episode_id": "b",
            "correlation_family": "risk",
            "available_at": second,
            "direction": -0.5,
            "source_refs": ["b"],
        },
    ]
    result = run(
        market,
        start,
        end,
        parameters={"use_events": True, "event_threshold": 0.5},
        event_rows=events,
        cycle_parameters=CYCLE,
    )
    last = result["decisions"][-1]
    assert last["cause"] == "cycle_adverse_event"
    assert last["order_scheduled"] is False
    assert last["episode_start"] == second
    assert result["pending_order"] is None


HYBRID = {**CYCLE, "defensive_after_top": True}


def hybrid_top_fixture():
    market, start, _ = fixture(268)
    values = [(100, 101, 99, 100)] * 260 + [
        (100, 106, 99, 105),
        (105, 111, 104, 110),
        (109, 110, 107, 108),
        (108.5, 110, 108, 109),
        (110, 110, 107, 108),
        (107, 108, 104, 105),
        (104, 105, 103, 104),
        (104, 105.2, 103.5, 104.5),
    ]
    for row, (o, h, lo, c) in zip(market["QQQ"], values):
        row.update(open=o, high=h, low=lo, close=c, provider_adjusted_close=c)
    return market, start, market["QQQ"][-1]["date"]


def test_hybrid_top_fills_only_defensive30pct_sleeve_and_preserves_shared_cash_book():
    result = run(
        *hybrid_top_fixture(), parameters={"top_breakout_sessions": 3}, cycle_parameters=HYBRID
    )
    top = next(d for d in result["decisions"] if d["cause"] == "cycle_local_top")
    assert top["target_weights"]["QQQ"] == 0.7
    assert sum(top["target_weights"][a] for a in DEFENSIVE) == pytest.approx(0.3)
    assert top["target_weights"]["SPY"] == top["target_weights"]["SMH"] == 0
    assert result["policy"]["version"] == "single-index-cycle-defensive-2"
    assert any(f["asset"] in DEFENSIVE for f in result["fills"])
    for point in result["equity_curve"]:
        assert point["cash"] >= 0 and point["exposure"] <= 1 + 1e-12
    for queue in result["rebalance_queues"]:
        assert sum(queue["target_weights"].values()) <= 1 + 1e-12


def test_hybrid_longtrend_uses_eligible_defense_or_cash_never_forces_ineligible_hedges():
    market, start, end = cycle_fixture(extra=2)
    result = run(market, start, end, cycle_parameters=HYBRID)
    risk = next(d for d in result["decisions"] if d["cause"] == "cycle_long_trend_exit")
    assert risk["target_weights"]["QQQ"] == risk["target_weights"]["TQQQ"] == 0
    assert sum(risk["target_weights"][a] for a in DEFENSIVE) == 1
    for asset in DEFENSIVE:
        for i, row in enumerate(market[asset]):
            p = 100 * 0.999**i
            row.update(open=p, high=p * 1.01, low=p * 0.99, close=p, provider_adjusted_close=p)
    unavailable = run(market, start, end, cycle_parameters=HYBRID)
    risk = next(d for d in unavailable["decisions"] if d["cause"] == "cycle_long_trend_exit")
    assert sum(risk["target_weights"].values()) == 0
    assert not any(f["asset"] in DEFENSIVE for f in unavailable["fills"])


def test_hybrid_weekly_defensive_review_keeps_episode_and_only_rebalances_when_targets_change():
    result = run(*cycle_fixture(extra=8), cycle_parameters=HYBRID)
    reviews = [d for d in result["decisions"] if d["cause"] == "cycle_defensive_weekly_review"]
    assert reviews
    episode = reviews[0]["episode_start"]
    assert all(d["weekly_review"] and d["cycle_risk_off"] for d in reviews)
    assert all(d["episode_start"] == episode for d in reviews)
    assert all(d["target_weights"]["QQQ"] == d["target_weights"]["TQQQ"] == 0 for d in reviews)
    assert not any(d["order_scheduled"] for d in reviews[1:])


def test_hybrid_recovery_sells_defense_for_tqqq_then_gain_migrates_to_pure_qqq():
    market, start, end = cycle_fixture(extra=10)
    first = run(market, start, end, parameters={"recovery_weight": 0.25}, cycle_parameters=HYBRID)
    buy = next(f for f in first["fills"] if f["asset"] == "TQQQ" and f["side"] == "buy")
    day = next(i for i, row in enumerate(market["TQQQ"]) if row["date"] == buy["at"][:10])
    for row in market["TQQQ"][day + 2 :]:
        row.update(open=12, high=12.2, low=11.8, close=12, provider_adjusted_close=12)
    result = run(market, start, end, parameters={"recovery_weight": 0.25}, cycle_parameters=HYBRID)
    recovery = next(d for d in result["decisions"] if d["cause"] == "cycle_confirmed_bottom")
    assert recovery["target_weights"]["QQQ"] == 0.75
    assert recovery["target_weights"]["TQQQ"] == 0.25
    assert all(recovery["target_weights"][a] == 0 for a in DEFENSIVE)
    migration = next(d for d in result["decisions"] if d["cause"] == "cycle_tqqq_gain_to_qqq")
    assert migration["target_weights"]["QQQ"] == 1
    assert all(migration["target_weights"][a] == 0 for a in DEFENSIVE)
    assert any(
        f["asset"] in DEFENSIVE and f["side"] == "sell" and f["signal_at"] == recovery["at"]
        for f in result["fills"]
    )


def test_hybrid_structural_failure_goes_cash_before_later_weekly_defense():
    market, start, end = cycle_fixture(extra=10)
    first = run(market, start, end, parameters={"recovery_weight": 0.25}, cycle_parameters=HYBRID)
    buy = next(f for f in first["fills"] if f["asset"] == "TQQQ" and f["side"] == "buy")
    day = next(i for i, row in enumerate(market["QQQ"]) if row["date"] == buy["at"][:10])
    stop = first["decisions"][day - 259]["recovery_stop"]
    market["QQQ"][day + 2].update(
        open=stop, high=stop + 1, low=stop - 2, close=stop - 1, provider_adjusted_close=stop - 1
    )
    result = run(market, start, end, parameters={"recovery_weight": 0.25}, cycle_parameters=HYBRID)
    failure = next(d for d in result["decisions"] if d["cause"] == "cycle_structural_stop")
    assert sum(failure["target_weights"].values()) == 0
    next_day = _next(date.fromisoformat(failure["at"][:10])).isoformat()
    assert next(p for p in result["equity_curve"] if p["at"][:10] == next_day)["exposure"] == 0


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("plain", "3d7015d8b77137cef114e021db8f5953452c9d3c78eef01184db1d2199205036"),
        ("recovery", "4bda7f2680e6d5547a6999a08e3b4c34a33be02f867d6bc12c6d9152d3efb200"),
    ],
)
def test_five_key_cycle_keeps_pre_hybrid_numeric_golden(kind, expected):
    args = fixture() if kind == "plain" else cycle_fixture()
    params = {} if kind == "plain" else {"recovery_weight": 0.5}
    for cycle in (CYCLE, {**CYCLE, "defensive_after_top": False}):
        result = run(*args, parameters=params, cycle_parameters=cycle)
        fields = {
            k: v
            for k, v in result.items()
            if k not in ("policy", "policy_sha256", "source_code_sha256")
        }
        digest = hashlib.sha256(
            json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        assert digest == expected


def test_hybrid_switch_requires_boolean_and_rejects_unrelated_new_settings():
    for value in (0, 1, "true", None):
        with pytest.raises(ValueError):
            build_macro_rotation_policy(
                DEFAULT_PARAMETERS, cycle_parameters={**CYCLE, "defensive_after_top": value}
            )
    with pytest.raises(ValueError):
        build_macro_rotation_policy(
            DEFAULT_PARAMETERS, cycle_parameters={**HYBRID, "new_knob": True}
        )


def test_live_tail_clock_consumes_afterclose_news_without_changing_historical_book():
    from datetime import datetime, timedelta

    market, start, end = fixture()
    close = datetime.fromisoformat(_clock(date.fromisoformat(end)))
    tail = close + timedelta(hours=1)
    event = {
        "id": "afterclose",
        "available_at": (close + timedelta(minutes=30)).isoformat(),
        "direction": -0.5,
        "source_refs": ["official"],
    }
    baseline = run(
        market,
        start,
        end,
        parameters={"use_events": True, "event_threshold": 0.5},
        cycle_parameters=CYCLE,
        event_rows=[event],
    )
    result = run(
        market,
        start,
        end,
        parameters={"use_events": True, "event_threshold": 0.5},
        cycle_parameters=CYCLE,
        event_rows=[event],
        live_tail_clock=tail,
    )
    assert result["equity_curve"] == baseline["equity_curve"]
    assert result["fills"] == baseline["fills"]
    assert result["decisions"][:-1] == baseline["decisions"][:-1]
    assert result["decisions"][-1]["cause"] == "cycle_adverse_event"
    assert result["pending_order"]["known_at"] == tail.isoformat()
    for invalid in (
        close - timedelta(seconds=1),
        close.replace(tzinfo=None),
        datetime.fromisoformat(_clock(_next(date.fromisoformat(end)), True)),
    ):
        with pytest.raises(ValueError):
            run(market, start, end, cycle_parameters=CYCLE, live_tail_clock=invalid)
