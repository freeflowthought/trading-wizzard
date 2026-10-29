"""Single-ETF, unlevered price-first research portfolio and matched benchmarks.

Completed closes set targets; the following actual session open executes them.
No author trades, fitted probabilities, orders or historical-vintage data are implied.
"""

import bisect
import copy
import hashlib
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

from .sessions import is_us_equity_session
from .structure import _atr, _clean
from .swing_optimizer import _hash, _number

NY = ZoneInfo("America/New_York")
PARAMETERS = {"support_sessions", "trend_sessions", "confirm_sessions", "trim_to",
              "trailing_atr", "entry_steps"}


def _clock(day, opening=False):
    return datetime.combine(day, time(9, 30) if opening else time(16, 15), NY).isoformat()


def _next(day):
    day += timedelta(days=1)
    while not is_us_equity_session(day):
        day += timedelta(days=1)
    return day


def build_portfolio_swing_policy(parameters, *, max_exposure=1.0, cost_multiplier=1):
    """Freeze rules/code/targets/costs before reading any numerical market data."""
    if not isinstance(parameters, dict) or set(parameters) != PARAMETERS:
        raise ValueError("Portfolio parameters require exactly the six declared keys.")
    choices = {"support_sessions": {10, 20, 63}, "trend_sessions": {63, 126, 200},
               "confirm_sessions": {2, 3}, "trim_to": {0, .5, .7}, "trailing_atr": {2, 3, 4}}
    if any(type(parameters[key]) not in (int, float) or parameters[key] not in allowed
           or key != "trim_to" and type(parameters[key]) is not int
           for key, allowed in choices.items()):
        raise ValueError("Unsupported frozen portfolio parameter value.")
    steps = parameters["entry_steps"]
    if (not isinstance(steps, list) or any(type(x) not in (int, float) for x in steps)
            or steps not in ([.4, .7, 1], [.5, .75, 1])):
        raise ValueError("Entry steps must be one declared cumulative allocation ladder.")
    if (type(max_exposure) not in (int, float) or _number(max_exposure) is None
            or not 0 < max_exposure <= 1 or type(cost_multiplier) not in (int, float)
            or cost_multiplier not in (1, 2)):
        raise ValueError("Exposure must be >0 and <=100%; cost multiplier must be 1 or 2.")
    sources = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
               for name in ("portfolio_swing.py", "structure.py", "sessions.py",
                            "swing_optimizer.py")}
    policy = {
        "version": "price-first-portfolio-swing-1", "parameters": copy.deepcopy(parameters),
        "max_exposure": max_exposure, "cost_multiplier": cost_multiplier,
        "fee_bps_per_turn": cost_multiplier, "slippage_bps_per_turn": 2 * cost_multiplier,
        "initial": "100 cash at start close; strategy max target and benchmark100% next open",
        "execution": "completed-close decision; following actual session open; fractional shares",
        "target_math": "cost-aware post-trade marked exposure; no leverage or cash borrowing",
        "gap_policy": (
            "skip targets already crossed at open; reductions never buy/additions never sell"
        ),
        "rebalance": "new rule signals only; passive price drift does not cause daily rebalancing",
        "priority": "full risk exit, bearish rejection, then staged addition",
        "trend_exit": "last N closes each below its own causal slow SMA, including current close",
        "top": "positive held profit; high>=prior20high-.25ATR; close<open and prior close",
        "trailing_exit": "after actual trim, close<highest campaign close-trailing_atr*ATR14",
        "bottom": "most recent minimum low aged N; no later lower low; current close rises",
        "reentry": "fresh bottom or close>prior5high and slowSMA; thereafter rise+SMA20 or bottom",
        "reset": "partial/recovery/highwater reset upon actual full configured-target recovery",
        "atr": "simple mean last14 true ranges from15 completed rows; minimum260 origin history",
        "benchmark_price": "same initial open and costs; buy100% and hold; final mark, no exit fee",
        "benchmark_total_return": (
            "first adjusted open=raw open*provider_adjusted_close/raw close; "
            "wealth=100*adjusted close/(first adjusted open*(1+slip)*(1+fee))"
        ),
        "dividends": "strategy excludes dividend cash; benchmark reinvests by provider adjustment",
        "total_return_basis": "provider_adjusted_close_not_strict_vintage",
        "source_hash_encoding": (
            "compact sorted Python JSON; optional nonfinite metadata stays explicit"
        ),
        "source_code_sha256": sources,
    }
    return {"policy": policy, "policy_sha256": _hash(policy)}


def _prices(market, asset, first, last):
    if asset not in {"QQQ", "SPY"} or not isinstance(market, dict):
        raise ValueError("Evaluate one QQQ or SPY portfolio separately.")
    indexed = []
    for row in market.get(asset, []):
        if not isinstance(row, dict):
            raise ValueError("Prices require dated row objects.")
        day = date.fromisoformat(row.get("date"))
        if day <= last:  # Future numerical fields are neither validated nor read.
            indexed.append((day, row))
    indexed.sort(key=lambda item: item[0])
    raw = [item[1] for item in indexed]
    dates = [item[0] for item in indexed]
    origin_index = bisect.bisect_right(dates, first) - 1
    if origin_index < 259 or dates[origin_index] != first or not dates or dates[-1] != last:
        raise ValueError("Require260+ origin history and both exact completed endpoint sessions.")
    basis = raw[0].get("price_basis")
    if (not isinstance(basis, str) or not basis.strip()
            or basis.lower().strip() in {"unknown", "unverified", "none", "n/a", "na"}
            or "unknown" in basis.lower() or "未注明" in basis
            or any(row.get("price_basis") != basis or row.get("session_complete") is not True
                   for row in raw)):
        raise ValueError("Prices require one explicit verified basis and completed sessions.")
    clean = _clean(raw, last.isoformat())
    if len(clean) != len(raw) or any(_next(left) != right for left, right in zip(dates, dates[1:])):
        raise ValueError("Require finite complete unique contiguous source-verified OHLCV.")
    return raw, clean, origin_index, basis


def _features(history, parameters):
    n, trend = parameters["confirm_sessions"], parameters["trend_sessions"]
    anchor = min(enumerate(history[-parameters["support_sessions"]:]),
                 key=lambda item: (item[1]["low"], -item[0]))
    age = parameters["support_sessions"] - 1 - anchor[0]
    slow = mean(row["close"] for row in history[-trend:])
    prior5, prior20 = history[-6:-1], history[-21:-1]
    trend_broken = all(history[-offset - 1]["close"] < mean(
        row["close"] for row in history[len(history) - trend - offset:len(history) - offset])
        for offset in range(n))
    close, previous = history[-1]["close"], history[-2]["close"]
    return {"atr14": _atr(history), "slow_sma": slow,
            "sma20": mean(row["close"] for row in history[-20:]),
            "prior5_high": max(row["high"] for row in prior5),
            "prior20_high": max(row["high"] for row in prior20),
            "anchor_date": anchor[1]["date"], "anchor_low": anchor[1]["low"],
            "anchor_age_sessions": age, "trend_broken": trend_broken,
            "bottom_confirmed": age >= n and close > previous,
            "rising": close > previous, "close": close, "previous_close": previous,
            "known_at": _clock(date.fromisoformat(history[-1]["date"]))}


def _book():
    return {"cash": 100.0, "qty": 0.0, "cost_basis": 0.0, "highwater": None,
            "fees": 0.0, "slippage": 0.0, "turnover": 0.0}


def _fill(book, order, bar, fee, slip):
    price, target = bar["open"], order["target_exposure"]
    nav = book["cash"] + book["qty"] * price
    gap = target * nav - book["qty"] * price
    side = "buy" if gap > 0 else "sell"
    if side != order["direction"]:
        return None
    execution = price * (1 + slip if side == "buy" else 1 - slip)
    all_in, net = execution * (1 + fee), execution * (1 - fee)
    qty = (min(gap / (price + target * (all_in - price)), book["cash"] / all_in)
           if side == "buy" else min(-gap / (price - target * (price - net)), book["qty"]))
    if qty <= 1e-12:
        return None
    commission, slippage = qty * execution * fee, qty * abs(execution - price)
    if side == "buy":
        book["cash"] = max(0.0, book["cash"] - qty * all_in)
        book["qty"] += qty
        book["cost_basis"] += qty * all_in
    else:
        book["cost_basis"] *= max(0.0, 1 - qty / book["qty"])
        book["qty"] -= qty
        book["cash"] += qty * net
        if book["qty"] <= 1e-12:
            book.update(qty=0.0, cost_basis=0.0, highwater=None)
    book["fees"] += commission
    book["slippage"] += slippage
    book["turnover"] += qty * price
    nav = book["cash"] + book["qty"] * price
    return {**order, "at": _clock(date.fromisoformat(bar["date"]), True), "side": side,
            "qty": qty, "price": execution, "raw_open": price, "fee": commission,
            "slippage": slippage, "cash": book["cash"], "holdings_qty": book["qty"],
            "cost_basis": book["cost_basis"], "post_fill_nav": nav,
            "post_fill_exposure": book["qty"] * price / nav}


def _point(book, bar):
    equity = book["cash"] + book["qty"] * bar["close"]
    return {"at": _clock(date.fromisoformat(bar["date"])), "equity": equity,
            "cash": book["cash"], "qty": book["qty"],
            "exposure": book["qty"] * bar["close"] / equity,
            "cost_basis": book["cost_basis"], "highwater": book["highwater"]}


def _drawdown(curve):
    peak, worst = curve[0]["equity"], 0.0
    for point in curve:
        peak = max(peak, point["equity"])
        worst = max(worst, 100 * (1 - point["equity"] / peak))
    return worst


def simulate_portfolio_swing(market, asset, start, end, *, parameters, max_exposure=1.0,
                             cost_multiplier=1):
    """Simulate one ETF, never sum two independently funded100% accounts.

    The max exposure is an order-target ceiling; passive weight drift does not
    force rebalancing. All holdings stay unlevered. Final holdings are marked,
    not hypothetically liquidated. Strategy dividends are deliberately excluded.
    """
    frozen = build_portfolio_swing_policy(parameters, max_exposure=max_exposure,
                                        cost_multiplier=cost_multiplier)
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first >= last or not is_us_equity_session(first) or not is_us_equity_session(last):
        raise ValueError("Portfolio window requires two increasing completed session dates.")
    raw, rows, begin, basis = _prices(market, asset, first, last)
    fee, slip = cost_multiplier / 10000, cost_multiplier * 2 / 10000
    book, benchmark, fills, benchmark_fills, decisions = _book(), _book(), [], [], []
    unfilled = []
    curve, price_curve = [_point(book, rows[begin])], [_point(benchmark, rows[begin])]
    target, trimmed, recovery = max_exposure, False, False
    pending = {"signal_at": _clock(first), "target_exposure": target, "direction": "buy",
               "cause": "initial_allocation"}
    decisions.append({**pending, "features": _features(rows[:begin + 1], parameters)})
    for i in range(begin + 1, len(rows)):
        bar, day = rows[i], date.fromisoformat(rows[i]["date"])
        if pending:
            fill = _fill(book, pending, bar, fee, slip)
            if fill:
                fills.append(fill)
                if pending["cause"] == "top_rejection":
                    trimmed, recovery = True, False
                elif pending["cause"] in {"trend_exit", "post_trim_trailing_exit"}:
                    recovery = False
                elif fill["side"] == "buy":
                    recovery = target < max_exposure
                    if target == max_exposure:
                        trimmed, book["highwater"] = False, None
            else:
                unfilled.append({**pending, "at": _clock(day, True), "raw_open": bar["open"],
                                 "reason": "open_target_already_crossed_or_zero_quantity"})
            pending = None
        if i == begin + 1:
            benchmark_fills.append(_fill(benchmark, {
                "signal_at": _clock(first), "target_exposure": 1.0, "direction": "buy",
                "cause": "buy_and_hold"
            }, bar, fee, slip))
        if book["qty"]:
            book["highwater"] = max(book["highwater"] or bar["close"], bar["close"])
        benchmark["highwater"] = max(benchmark["highwater"] or bar["close"], bar["close"])
        features = _features(rows[:i + 1], parameters)
        atr, close = features["atr14"], bar["close"]
        trailing = bool(trimmed and book["qty"] and book["highwater"] is not None
                        and close < book["highwater"] - parameters["trailing_atr"] * atr)
        rejection = bool(book["qty"] and book["qty"] * close > book["cost_basis"]
                         and bar["high"] >= features["prior20_high"] - .25 * atr
                         and close < bar["open"] and close < features["previous_close"])
        cause, next_target = "hold", target
        if features["trend_broken"] or trailing:
            cause = "trend_exit" if features["trend_broken"] else "post_trim_trailing_exit"
            next_target = 0.0
        elif rejection:
            cause = "top_rejection_already_trimmed" if trimmed else "top_rejection"
            if not trimmed:
                next_target = min(target, parameters["trim_to"] * max_exposure)
        elif target < max_exposure:
            breakout = close > features["prior5_high"] and close > features["slow_sma"]
            continuation = recovery and features["rising"] and close >= features["sma20"]
            if features["bottom_confirmed"] or breakout or continuation:
                next_target = next(step * max_exposure for step in parameters["entry_steps"]
                                   if step > target / max_exposure + 1e-12)
                cause = ("bottom_recovery" if features["bottom_confirmed"] else
                         "trend_reentry" if breakout else "recovery_continuation")
        before = target
        if next_target != target:
            target = next_target
            pending = {"signal_at": _clock(day), "target_exposure": target, "cause": cause,
                       "direction": "buy" if target > before else "sell"}
        decisions.append({"at": _clock(day), "known_at": _clock(day), "cause": cause,
                          "prior_target_exposure": before, "target_exposure": target,
                          "order_scheduled": pending is not None, "features": features,
                          "trimmed": trimmed, "recovery_active": recovery,
                          "highwater": book["highwater"]})
        curve.append(_point(book, bar))
        price_curve.append(_point(benchmark, bar))
    tr_curve, tr_reason = [{"at": _clock(first), "equity": 100.0}], None
    adjusted = [_number(row.get("provider_adjusted_close")) for row in rows[begin + 1:]]
    if any(value is None or value <= 0 for value in adjusted):
        tr_curve, tr_reason = None, "missing_or_invalid_provider_adjusted_close"
    else:
        adjusted_open = rows[begin + 1]["open"] * adjusted[0] / rows[begin + 1]["close"]
        tr_curve += [{"at": _clock(date.fromisoformat(row["date"])),
                      "equity": 100 * adj / (adjusted_open * (1 + slip) * (1 + fee))}
                     for row, adj in zip(rows[begin + 1:], adjusted)]
    strategy_return, price_return = (curve[-1]["equity"] - 100, price_curve[-1]["equity"] - 100)
    tr_return = tr_curve[-1]["equity"] - 100 if tr_curve else None
    metrics = {
        "net_account_return_pct": strategy_return, "max_drawdown_pct": _drawdown(curve),
        "benchmark_price_return_pct": price_return, "benchmark_total_return_pct": tr_return,
        "alpha_to_price_pp": strategy_return - price_return,
        "alpha_to_total_return_pp": strategy_return - tr_return if tr_return is not None else None,
        "benchmark_price_max_drawdown_pct": _drawdown(price_curve),
        "benchmark_total_return_max_drawdown_pct": _drawdown(tr_curve) if tr_curve else None,
        "turnover_pct": book["turnover"], "fill_count": len(fills),
        "avg_exposure_pct": 100 * mean(point["exposure"] for point in curve),
        "max_exposure_pct": 100 * max(point["exposure"] for point in curve),
        "fees_paid": book["fees"], "slippage_paid": book["slippage"],
    }
    source_hash = hashlib.sha256(json.dumps(
        raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**frozen, "asset": asset, "start": start, "end": end, "price_basis": basis,
            "source_history_sha256": source_hash, "source_code_sha256": frozen["policy"][
                "source_code_sha256"], "metrics": metrics, "equity_curve": curve,
            "benchmark_price_curve": price_curve, "benchmark_total_return_curve": tr_curve,
            "benchmark_total_return_unavailable_reason": tr_reason,
            "benchmark_total_return_basis": "provider_adjusted_close_not_strict_vintage",
            "strategy_dividends_included": False,
            "total_return_comparison_like_for_like": False,
            "alpha_to_total_return_interpretation": (
                "conservative price-strategy minus reinvested benchmark reference; "
                "not like-for-like strategy total-return alpha"
            ),
            "fills": fills, "benchmark_fills": benchmark_fills, "decisions": decisions,
            "unfilled_orders": unfilled,
            "pending_order": pending, "historical_point_in_time_verified": False,
            "goal_completion_proven": False}
