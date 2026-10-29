"""Structural swing research: meaningful waves, new recovery bases, latched risk.

Execution/accounting helpers are unchanged from portfolio_swing. Only causal
close-time decisions differ; no author trades or probability model is inferred.
"""

import copy
import hashlib
import json
from datetime import date
from pathlib import Path
from statistics import mean

from .portfolio_swing import _clock, _drawdown, _fill, _point, _prices
from .sessions import is_us_equity_session
from .structure import _atr
from .swing_optimizer import _hash, _number

KEYS = {"support_sessions", "trend_sessions", "confirm_sessions", "trim_to", "trailing_atr",
        "entry_steps", "profit_atr", "breakout_sessions", "stop_buffer_atr"}


def build_structural_policy(parameters, max_exposure=1.0, cost_multiplier=1):
    """Create a rule/code/parameter receipt without inspecting market outcomes."""
    choices = {"support_sessions": (10, 20, 63), "trend_sessions": (63, 126, 200),
               "confirm_sessions": (2, 3), "trim_to": (0, .5, .7, .9, 1),
               "trailing_atr": (2, 3, 4), "profit_atr": (2, 3, 4),
               "breakout_sessions": (3, 5, 10), "stop_buffer_atr": (.5, 1)}
    if not isinstance(parameters, dict) or set(parameters) != KEYS or any(
            type(parameters[key]) not in (int, float) or parameters[key] not in allowed
            or key not in {"trim_to", "stop_buffer_atr"} and type(parameters[key]) is not int
            for key, allowed in choices.items()):
        raise ValueError("Structural policy requires the nine exact declared parameters.")
    steps = parameters["entry_steps"]
    if (not isinstance(steps, list) or any(type(x) not in (int, float) for x in steps)
            or steps not in ([.4, .7, 1], [.5, .75, 1])):
        raise ValueError("Unsupported cumulative entry ladder.")
    if (type(max_exposure) not in (int, float) or _number(max_exposure) is None
            or not 0 < max_exposure <= 1 or type(cost_multiplier) not in (int, float)
            or cost_multiplier not in (1, 2)):
        raise ValueError("Require unlevered exposure<=100% and cost multiplier1 or2.")
    code = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("portfolio_structure.py", "portfolio_swing.py", "structure.py",
                         "sessions.py", "swing_optimizer.py")}
    policy = {
        "version": "structural-portfolio-swing-2", "parameters": copy.deepcopy(parameters),
        "max_exposure": max_exposure, "cost_multiplier": cost_multiplier,
        "fee_bps_per_turn": cost_multiplier, "slippage_bps_per_turn": 2 * cost_multiplier,
        "execution": "start100cash; next open max strategy/100%benchmark; close signals/next open",
        "accounting": "unchanged V1 cost-aware target math; no leverage or passive-weight churn",
        "top": "completed highwater-WAC/share>=profit_atr*ATR14 and close<prior N lowest low",
        "recovery_episode": "starts at actual reduction's known SELL SIGNAL date; no older anchor",
        "first_recovery": "suffix low aged N confirmations; rising close>prior breakout high",
        "latch": "initial armed; fromcash signal/rearm require close>=slowSMA",
        "risk_priority": "held stop, armed N-day slow trend, posttrim trailing, top, adds",
        "structural_stop": "first recovery signal anchor-buffer*ATR; installed only actualbuy",
        "stop_retention": "retain across adds/fullrecovery; new episode can tighten, never widen",
        "gap_cancel": "buy open<=declaredstop cancels; unfilled restores prior target/state",
        "staging": "after actualfill, new risingclose>firstrecovery signalclose adds one step",
        "resets": "fullrecovery resets partial/highwater, keeps stop; fullsale clears stop/partial",
        "atr": "causal simple ATR14 from15 rows; >=260 complete contiguous samebasis origin bars",
        "benchmark": "same initial open/costs; pricehold and provider dividend reinvestment",
        "benchmark_total_return_formula": (
            "adjopen=open*adjclose/close;100*adjclose/adjopen/entrycost"
        ),
        "dividends": "strategy excludes dividends; TR reference is not like-for-like alpha",
        "source_hash_encoding": "compact sorted Python JSON, optional nonfinite metadata explicit",
        "source_code_sha256": code,
    }
    return {"policy": policy, "policy_sha256": _hash(policy)}


def _book():
    return {"cash": 100.0, "qty": 0.0, "cost_basis": 0.0, "highwater": None,
            "fees": 0.0, "slippage": 0.0, "turnover": 0.0}


def _features(history, parameters, episode):
    n, trend, breakout = (parameters[key] for key in (
        "confirm_sessions", "trend_sessions", "breakout_sessions"))
    close = history[-1]["close"]
    slow = mean(row["close"] for row in history[-trend:])
    broken = all(history[-offset - 1]["close"] < mean(
        row["close"] for row in history[len(history)-trend-offset:len(history)-offset])
        for offset in range(n))
    prior = history[-breakout-1:-1]
    candidates = [(index, row) for index, row in enumerate(history)
                  if index >= len(history)-parameters["support_sessions"]
                  and episode is not None and row["date"] >= episode[:10]]
    anchor = min(candidates, key=lambda item: (item[1]["low"], -item[0])) if candidates else None
    age = len(history)-1-anchor[0] if anchor else None
    rising = close > history[-2]["close"]
    return {
        "known_at": _clock(date.fromisoformat(history[-1]["date"])),
        "close": close, "previous_close": history[-2]["close"], "rising": rising,
        "atr14": _atr(history), "slow_sma": slow, "trend_broken": broken,
        "prior_breakout_low": min(row["low"] for row in prior),
        "prior_breakout_high": max(row["high"] for row in prior),
        "episode_start": episode, "anchor_date": anchor[1]["date"] if anchor else None,
        "anchor_at": _clock(date.fromisoformat(anchor[1]["date"])) if anchor else None,
        "anchor_low": anchor[1]["low"] if anchor else None, "anchor_age_sessions": age,
        "bottom_confirmed": bool(anchor and age >= n and rising
                                 and close > max(row["high"] for row in prior)),
        "confirmation_clocks": [_clock(date.fromisoformat(row["date"])) for row in history[-n:]],
    }


def simulate_structural_portfolio(market, asset, start, end, *, parameters, max_exposure=1.0,
                                  cost_multiplier=1):
    """Evaluate one QQQ or SPY account separately using completed-close information."""
    frozen = build_structural_policy(parameters, max_exposure, cost_multiplier)
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first >= last or not is_us_equity_session(first) or not is_us_equity_session(last):
        raise ValueError("Require an increasing completed-session window.")
    raw, rows, begin, basis = _prices(market, asset, first, last)
    fee, slip = cost_multiplier / 10000, cost_multiplier * 2 / 10000
    book, benchmark = _book(), _book()
    fills, benchmark_fills, unfilled, decisions = [], [], [], []
    curve, benchmark_curve = [_point(book, rows[begin])], [_point(benchmark, rows[begin])]
    target, trimmed, armed, recovery = max_exposure, False, True, False
    episode, stop, first_recovery = None, None, None
    pending = {"signal_at": _clock(first), "cause": "initial_allocation", "direction": "buy",
               "target_exposure": target, "prior_target_exposure": 0.0}
    decisions.append({**pending, "known_at": _clock(first), "risk_armed": armed,
                      "at": _clock(first), "structural_stop": stop,
                      "episode_start": None, "first_recovery_signal": None,
                      "proposed_recovery": None, "trimmed": False, "recovery_active": False,
                      "order_scheduled": True, "highwater": None, "wave_profit_atr": None,
                      "features": _features(rows[:begin+1], parameters, None)})
    for i in range(begin+1, len(rows)):
        bar, day = rows[i], date.fromisoformat(rows[i]["date"])
        if pending:
            declared = pending.get("recovery_signal", {}).get("structural_stop", stop)
            cancel = (pending["direction"] == "buy" and declared is not None
                      and bar["open"] <= declared)
            fill = None if cancel else _fill(book, pending, bar, fee, slip)
            if fill:
                fills.append(fill)
                if fill["side"] == "sell":
                    episode, first_recovery, recovery = pending["signal_at"], None, False
                    trimmed = bool(book["qty"])
                    if not book["qty"]:
                        stop, book["highwater"] = None, None
                else:
                    signal = pending.get("recovery_signal")
                    if signal:
                        stop, first_recovery = signal["structural_stop"], copy.deepcopy(signal)
                        if signal["from_zero"]:
                            armed = signal["risk_armed_on_fill"]
                    recovery = target < max_exposure
                    if target == max_exposure:
                        trimmed, book["highwater"] = False, None
            else:
                unfilled.append({**pending, "at": _clock(day, True), "raw_open": bar["open"],
                                 "reason": "buy_open_at_or_below_frozen_stop" if cancel else
                                           "open_target_already_crossed_or_zero_quantity"})
                target = pending["prior_target_exposure"]
            pending = None
        if i == begin+1:
            benchmark_fills.append(_fill(benchmark, {
                "signal_at": _clock(first), "target_exposure": 1.0, "direction": "buy",
                "cause": "buy_and_hold"}, bar, fee, slip))
        if book["qty"]:
            book["highwater"] = max(book["highwater"] or bar["close"], bar["close"])
        benchmark["highwater"] = max(benchmark["highwater"] or bar["close"], bar["close"])
        f = _features(rows[:i+1], parameters, episode)
        if book["qty"] and f["close"] >= f["slow_sma"]:
            armed = True
        wave = ((book["highwater"]-book["cost_basis"]/book["qty"])/f["atr14"]
                if book["qty"] and f["atr14"] > 0 else None)
        stop_failed = bool(book["qty"] and stop is not None and f["close"] < stop)
        trend_failed = bool(book["qty"] and armed and f["trend_broken"])
        trail_failed = bool(book["qty"] and trimmed and f["close"] <
                            book["highwater"]-parameters["trailing_atr"]*f["atr14"])
        top = bool(book["qty"] and not trimmed and wave is not None
                   and wave >= parameters["profit_atr"] and f["close"] < f["prior_breakout_low"])
        next_target, cause, recovery_signal = target, "hold", None
        if stop_failed or trend_failed or trail_failed:
            next_target = 0.0
            cause = ("structural_stop_exit" if stop_failed else "trend_exit" if trend_failed
                     else "post_trim_trailing_exit")
        elif top:
            next_target, cause = min(target, parameters["trim_to"]*max_exposure), "structural_top"
        elif target < max_exposure:
            if first_recovery is None and f["bottom_confirmed"] and f["atr14"] > 0:
                candidate_stop = f["anchor_low"]-parameters["stop_buffer_atr"]*f["atr14"]
                if candidate_stop > 0:
                    recovery_signal = {
                        "signal_at": _clock(day), "signal_close": f["close"],
                        "episode_start": episode, "anchor_date": f["anchor_date"],
                        "anchor_at": f["anchor_at"], "anchor_low": f["anchor_low"],
                        "anchor_age_sessions": f["anchor_age_sessions"],
                        "signal_atr14": f["atr14"], "calculated_stop": candidate_stop,
                        "signal_slow_sma": f["slow_sma"],
                        "structural_stop": max(stop or 0, candidate_stop),
                        "from_zero": book["qty"] == 0,
                        "risk_armed_on_fill": f["close"] >= f["slow_sma"]
                                              if book["qty"] == 0 else armed,
                    }
                    cause = "new_base_recovery"
            elif recovery and first_recovery and f["rising"] and f["close"] > first_recovery[
                    "signal_close"]:
                cause = "recovery_continuation"
            if recovery_signal or cause == "recovery_continuation":
                next_target = next(step*max_exposure for step in parameters["entry_steps"]
                                   if step > target/max_exposure+1e-12)
        before = target
        if next_target != target:
            target = next_target
            pending = {"signal_at": _clock(day), "target_exposure": target,
                       "prior_target_exposure": before,
                       "direction": "buy" if target > before else "sell",
                       "cause": cause}
            if recovery_signal:
                pending["recovery_signal"] = recovery_signal
        decisions.append({"at": _clock(day), "known_at": _clock(day), "features": f,
                          "cause": cause, "prior_target_exposure": before,
                          "target_exposure": target,
                          "order_scheduled": pending is not None, "risk_armed": armed,
                          "episode_start": episode,
                          "first_recovery_signal": copy.deepcopy(first_recovery),
                          "structural_stop": stop, "proposed_recovery": recovery_signal,
                          "wave_profit_atr": wave, "highwater": book["highwater"],
                          "trimmed": trimmed, "recovery_active": recovery})
        curve.append(_point(book, bar))
        benchmark_curve.append(_point(benchmark, bar))
    tr_curve, tr_reason = [{"at": _clock(first), "equity": 100.0}], None
    adjusted = [_number(row.get("provider_adjusted_close")) for row in rows[begin+1:]]
    if any(value is None or value <= 0 for value in adjusted):
        tr_curve, tr_reason = None, "missing_or_invalid_provider_adjusted_close"
    else:
        adjusted_open = rows[begin+1]["open"]*adjusted[0]/rows[begin+1]["close"]
        tr_curve += [{"at": _clock(date.fromisoformat(row["date"])),
                      "equity": 100*adj/(adjusted_open*(1+slip)*(1+fee))}
                     for row, adj in zip(rows[begin+1:], adjusted)]
    pnl, bh_pnl = curve[-1]["equity"]-100, benchmark_curve[-1]["equity"]-100
    tr_pnl = tr_curve[-1]["equity"]-100 if tr_curve else None
    realized = book["cash"]+book["cost_basis"]-100
    metrics = {
        "net_account_return_pct": pnl, "max_drawdown_pct": _drawdown(curve),
        "realized_account_pnl_pct": realized, "unrealized_account_pnl_pct": pnl-realized,
        "benchmark_price_return_pct": bh_pnl, "benchmark_total_return_pct": tr_pnl,
        "alpha_to_price_pp": pnl-bh_pnl,
        "alpha_to_total_return_pp": pnl-tr_pnl if tr_pnl is not None else None,
        "benchmark_price_max_drawdown_pct": _drawdown(benchmark_curve),
        "benchmark_total_return_max_drawdown_pct": _drawdown(tr_curve) if tr_curve else None,
        "turnover_pct": book["turnover"], "fill_count": len(fills),
        "avg_exposure_pct": 100*mean(point["exposure"] for point in curve),
        "max_exposure_pct": 100*max(point["exposure"] for point in curve),
        "fees_paid": book["fees"], "slippage_paid": book["slippage"],
    }
    source_bytes = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    return {**frozen, "asset": asset, "start": start, "end": end, "price_basis": basis,
            "source_history_sha256": source_hash,
            "source_code_sha256": frozen["policy"]["source_code_sha256"],
            "metrics": metrics, "equity_curve": curve, "benchmark_price_curve": benchmark_curve,
            "benchmark_total_return_curve": tr_curve,
            "benchmark_total_return_unavailable_reason": tr_reason,
            "benchmark_total_return_basis": "provider_adjusted_close_not_strict_vintage",
            "strategy_dividends_included": False, "total_return_comparison_like_for_like": False,
            "alpha_to_total_return_interpretation": (
                "conservative price-strategy vs reinvested benchmark reference"
            ),
            "fills": fills, "benchmark_fills": benchmark_fills, "decisions": decisions,
            "unfilled_orders": unfilled, "pending_order": pending,
            "historical_point_in_time_verified": False, "goal_completion_proven": False}
