"""Monthly price-first ETF core/satellite research in one self-financing account.

Quantities are provider split-normalized units, not historical nominal shares.
Corporate actions are not applied again. Benchmarks are separate comparison
accounts and are never added to the strategy's100 starting capital.
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
from .structure import _clean
from .swing_optimizer import _hash, _number

UNIVERSE = ("QQQ", "SPY", "SMH")
NY = ZoneInfo("America/New_York")


def _clock(day, opening=False):
    return datetime.combine(day, time(9, 30) if opening else time(16, 15), NY).isoformat()


def _next(day):
    day += timedelta(days=1)
    while not is_us_equity_session(day):
        day += timedelta(days=1)
    return day


def build_rotation_policy(parameters, max_exposure=1.0, cost_multiplier=1):
    choices = {"momentum_sessions": (63, 126, 252), "trend_sessions": (63, 126, 200),
               "core_weight": (0, .5, .7), "switch_buffer": (0, .02, .05),
               "fallback": ("cash", "spy", "qqq_anchor")}
    if not isinstance(parameters, dict) or set(parameters) != set(choices) or any(
            parameters[key] not in values or isinstance(parameters[key], bool)
            for key, values in choices.items()):
        raise ValueError("Rotation policy requires the five exact declared parameters.")
    if any(type(parameters[key]) is not int for key in ("momentum_sessions", "trend_sessions")):
        raise ValueError("Lookbacks must be integer session counts.")
    if (type(max_exposure) not in (int, float) or _number(max_exposure) is None
            or not 0 < max_exposure <= 1 or type(cost_multiplier) not in (int, float)
            or cost_multiplier not in (1, 2)):
        raise ValueError("Require0<max exposure<=1 and cost multiplier1 or2.")
    code = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ("sector_rotation.py", "structure.py", "sessions.py", "swing_optimizer.py")}
    policy = {
        "version": "monthly-core-satellite-rotation-1", "parameters": copy.deepcopy(parameters),
        "universe": list(UNIVERSE), "max_exposure": max_exposure,
        "cost_multiplier": cost_multiplier,
        "fee_bps_per_side": cost_multiplier, "slippage_bps_per_side": 2*cost_multiplier,
        "clock": "initial and last US session monthly close16:15NY; next actual session open",
        "eligibility": "raw split-normalized close>SMAtrend and causal momentum ROC>0",
        "core": "QQQ if eligible; otherwise eligible SPY only with spy fallback; otherwise cash",
        "satellite": "highest eligible ROC; retain eligible incumbent when challenger lead<=buffer",
        "buffer_units": "absolute ROC fraction; .02 means two percentage points",
        "floating_comparison_tolerance": 1e-12,
        "tie_order": list(UNIVERSE), "account": "one100cash account; positions+cash, no borrowing",
        "exposure_target": "cap applies to targets; passive intramonth weight drift is allowed",
        "rebalance": "monthly targetweights; solve common postcostNAV by60 bisections; sells first",
        "benchmarks": "separate100 accounts: full QQQ/SMH, cap-matched static QQQcore+SMHsat",
        "static_basket": "initial core*cap QQQ+(1-core)*cap SMH; rest cash; never rebalance",
        "quantity_basis": "provider_split_normalized_units_not_historical_nominal_shares",
        "corporate_actions": "do not split-adjust normalized input prices/units a second time",
        "dividends": "strategy excludes dividends; provider-adjusted TR is conservative reference",
        "source_code_sha256": code,
    }
    if parameters["fallback"] == "qqq_anchor":
        policy.update(
            version="monthly-core-satellite-rotation-2",
            eligibility="ROC>0 and close>SMA applies to SMH only; QQQ is unconditional baseline",
            core="always QQQ regardless of its absolute ROC/SMA; no SPY allocation",
            satellite=("default QQQ; eligible SMH enters when SMH-QQQ ROC>buffer; "
                       "incumbent SMH exits if ineligible or QQQ-SMH ROC>buffer"),
        )
    return {"policy": policy, "policy_sha256": _hash(policy)}


def _prices(market, first, last, minimum):
    if not isinstance(market, dict):
        raise ValueError("Market input must contain the three universe price series.")
    output, raw, grid, begin, basis = {}, {}, None, None, {}
    for asset in UNIVERSE:
        if not isinstance(market.get(asset), list):
            raise ValueError("Every universe asset needs an explicit dated price list.")
        source = []
        for row in market[asset]:
            if not isinstance(row, dict):
                raise ValueError("Prices require explicit dated row objects.")
            day = date.fromisoformat(row.get("date"))
            if day <= last:  # Never inspect/hash numerical data after the requested end.
                source.append((day, row))
        source.sort(key=lambda item: item[0])
        dates, rows = [item[0] for item in source], [item[1] for item in source]
        start_index = bisect.bisect_right(dates, first)-1
        if (start_index < minimum-1 or dates[start_index] != first or dates[-1] != last
                or any(_next(left) != right for left, right in zip(dates, dates[1:]))):
            raise ValueError("Require continuous completed endpoints and enough origin history.")
        price_basis = rows[0].get("price_basis")
        if (not isinstance(price_basis, str) or not price_basis.strip()
                or any(word in price_basis.lower() for word in ("unknown", "unverified", "未注明"))
                or price_basis.strip().lower() in {"none", "n/a", "na"}
                or any(row.get("price_basis") != price_basis
                       or row.get("session_complete") is not True for row in rows)):
            raise ValueError("Require one explicit complete price basis per asset.")
        clean = _clean(rows, last.isoformat())
        if len(clean) != len(rows) or grid is not None and dates != grid:
            raise ValueError("Require valid OHLCV and identical complete asset session grids.")
        grid, begin = dates, start_index
        output[asset], raw[asset], basis[asset] = clean, rows, price_basis
    return output, raw, grid, begin, basis


def _targets(rows, index, parameters, incumbent, cap):
    features = {}
    for asset in UNIVERSE:
        history = rows[asset][:index+1]
        momentum, trend = parameters["momentum_sessions"], parameters["trend_sessions"]
        close = history[-1]["close"]
        roc = close/history[-momentum-1]["close"]-1
        sma = mean(row["close"] for row in history[-trend:])
        features[asset] = {"roc": roc, "sma": sma, "close": close,
                           "eligible": close > sma and roc > 0,
                           "momentum_start_date": history[-momentum-1]["date"],
                           "known_at": _clock(date.fromisoformat(history[-1]["date"]))}
    eligible = [asset for asset in UNIVERSE if features[asset]["eligible"]]
    core = "QQQ" if "QQQ" in eligible else (
        "SPY" if parameters["fallback"] == "spy" and "SPY" in eligible else None)
    best = max(eligible, key=lambda asset: features[asset]["roc"]) if eligible else None
    retained = bool(incumbent in eligible and best is not None and
                    features[best]["roc"]-features[incumbent]["roc"]
                    <= parameters["switch_buffer"]+1e-12)
    satellite = incumbent if retained else best
    if parameters["fallback"] == "qqq_anchor":
        core = "QQQ"
        relative = features["SMH"]["roc"]-features["QQQ"]["roc"]
        smh_eligible = features["SMH"]["eligible"]
        best = "SMH" if smh_eligible and relative > parameters["switch_buffer"]+1e-12 else "QQQ"
        satellite = ("SMH" if incumbent == "SMH" and smh_eligible
                     and -relative <= parameters["switch_buffer"]+1e-12 else best)
        retained = incumbent == satellite
    weights = dict.fromkeys(UNIVERSE, 0.0)
    if core:
        weights[core] += cap*parameters["core_weight"]
    if satellite:
        weights[satellite] += cap*(1-parameters["core_weight"])
    return {"weights": weights, "core": core, "satellite": satellite,
            "prior_satellite": incumbent, "best_eligible": best,
            "incumbent_retained": retained, "features": features}


def _book():
    return {"cash": 100.0, "positions": dict.fromkeys(UNIVERSE, 0.0),
            "basis": dict.fromkeys(UNIVERSE, 0.0), "fees": 0.0, "slippage": 0.0, "turnover": 0.0}


def _point(book, prices, at):
    equity = book["cash"]+sum(book["positions"][a]*prices[a] for a in UNIVERSE)
    weights = {a: book["positions"][a]*prices[a]/equity for a in UNIVERSE}
    return {"at": at, "equity": equity, "cash": book["cash"],
            "positions": dict(book["positions"]), "weights": weights,
            "exposure": sum(weights.values()), "cost_basis": dict(book["basis"])}


def _rebalance(book, weights, prices, signal_at, at, fee, slip):
    """One common post-cost NAV, not three separately funded target calculations."""
    if (set(weights) != set(UNIVERSE)
            or any(_number(w) is None or w < 0 for w in weights.values())
            or sum(weights.values()) > 1+1e-12):
        raise ValueError("Rebalance targets must be nonnegative and sum to at most100%.")
    pre = book["cash"]+sum(book["positions"][a]*prices[a] for a in UNIVERSE)
    previous = dict(book["positions"])
    buy_cost, sell_cost = (1+slip)*(1+fee)-1, 1-(1-slip)*(1-fee)

    def costs(nav):
        return sum(abs(weights[a]*nav-previous[a]*prices[a]) * (
            buy_cost if weights[a]*nav >= previous[a]*prices[a] else sell_cost)
            for a in UNIVERSE)

    low, high = 0.0, pre
    for _ in range(60):
        mid = (low+high)/2
        if mid+costs(mid) <= pre:
            low = mid
        else:
            high = mid
    target_units = {a: weights[a]*low/prices[a] for a in UNIVERSE}
    fills = []
    for side in ("sell", "buy"):
        for asset in UNIVERSE:
            delta = target_units[asset]-book["positions"][asset]
            if (side == "sell" and delta >= 0 or side == "buy" and delta <= 0
                    or abs(delta)*prices[asset] <= 1e-12*pre):
                continue
            price = prices[asset]*(1+slip if side == "buy" else 1-slip)
            units = min(delta, book["cash"]/(price*(1+fee))) if side == "buy" else -delta
            commission, slippage = units*price*fee, units*abs(price-prices[asset])
            if side == "buy":
                book["cash"] = max(0.0, book["cash"]-units*price*(1+fee))
                book["positions"][asset] += units
                book["basis"][asset] += units*price*(1+fee)
            else:
                book["basis"][asset] *= max(0.0, 1-units/book["positions"][asset])
                book["cash"] += units*price*(1-fee)
                book["positions"][asset] = max(0.0, book["positions"][asset]-units)
            book["fees"] += commission
            book["slippage"] += slippage
            book["turnover"] += units*prices[asset]
            fills.append({"asset": asset, "side": side, "units": units, "price": price,
                          "raw_open": prices[asset], "signal_at": signal_at, "at": at,
                          "fee": commission, "slippage": slippage, "cash_after": book["cash"],
                          "units_after": book["positions"][asset]})
    point = _point(book, prices, at)
    return {"signal_at": signal_at, "at": at, "target_weights": dict(weights),
            "pre_nav": pre, "solved_postcost_nav": low, "actual_postcost_nav": point["equity"],
            "solver_residual": low+costs(low)-pre,
            "actual_weights": point["weights"], "fills": fills}


def _drawdown(curve):
    peak, worst = 100.0, 0.0
    for point in curve:
        peak = max(peak, point["equity"])
        worst = max(worst, 100*(1-point["equity"]/peak))
    return worst


def simulate_sector_rotation(market, start, end, *, parameters, max_exposure=1.0,
                             cost_multiplier=1):
    frozen = build_rotation_policy(parameters, max_exposure, cost_multiplier)
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first >= last or not is_us_equity_session(first) or not is_us_equity_session(last):
        raise ValueError("Require increasing completed session endpoints.")
    minimum = max(260, parameters["momentum_sessions"]+1, parameters["trend_sessions"])
    rows, raw, days, begin, basis = _prices(market, first, last, minimum)
    book = _book()
    benchmarks = {name: _book() for name in ("qqq", "smh", "static_basket")}
    curves = {name: [] for name in ("strategy", *benchmarks)}
    fills, benchmark_fills, queues, decisions, pending, incumbent = [], {}, [], [], None, None
    fee, slip = cost_multiplier/10000, cost_multiplier*2/10000
    for i in range(begin, len(days)):
        day = days[i]
        if pending and i > begin:
            prices = {a: rows[a][i]["open"] for a in UNIVERSE}
            queue = _rebalance(book, pending["weights"], prices, pending["known_at"],
                               _clock(day, True), fee, slip)
            queues.append(queue)
            fills.extend(queue["fills"])
            pending = None
            if i == begin+1:
                targets = {"qqq": {"QQQ": 1.0}, "smh": {"SMH": 1.0},
                           "static_basket": {"QQQ": parameters["core_weight"]*max_exposure,
                                             "SMH": (1-parameters["core_weight"])*max_exposure}}
                for name, weights in targets.items():
                    target = {a: weights.get(a, 0.0) for a in UNIVERSE}
                    result = _rebalance(benchmarks[name], target, prices, _clock(first),
                                        _clock(day, True), fee, slip)
                    benchmark_fills[name] = result["fills"]
        closes = {a: rows[a][i]["close"] for a in UNIVERSE}
        curves["strategy"].append(_point(book, closes, _clock(day)))
        for name, benchmark in benchmarks.items():
            curves[name].append(_point(benchmark, closes, _clock(day)))
        if i == begin or _next(day).month != day.month:
            selected = _targets(rows, i, parameters, incumbent, max_exposure)
            incumbent = selected["satellite"]
            pending = {**selected, "known_at": _clock(day),
                       "scheduled_open_at": _clock(_next(day), True)}
            decisions.append(copy.deepcopy(pending))
    references, unavailable = {}, {}
    for name, asset in (("qqq", "QQQ"), ("smh", "SMH")):
        adjusted = [_number(r.get("provider_adjusted_close")) for r in rows[asset][begin+1:]]
        if any(value is None or value <= 0 for value in adjusted):
            references[name], unavailable[name] = None, "missing_or_invalid_provider_adjusted_close"
        else:
            initial = rows[asset][begin+1]
            adjusted_open = initial["open"]*adjusted[0]/initial["close"]
            references[name] = [{"at": _clock(first), "equity": 100.0}]+[
                {"at": _clock(day), "equity": 100*adj/(adjusted_open*(1+slip)*(1+fee))}
                for day, adj in zip(days[begin+1:], adjusted)]
    pnl = curves["strategy"][-1]["equity"]-100
    returns = {name: curve[-1]["equity"]-100 for name, curve in curves.items()}
    tr = {name: curve[-1]["equity"]-100 if curve else None for name, curve in references.items()}
    realized = book["cash"]+sum(book["basis"].values())-100
    metrics = {
        "net_account_return_pct": pnl, "max_drawdown_pct": _drawdown(curves["strategy"]),
        "realized_account_pnl_pct": realized, "unrealized_account_pnl_pct": pnl-realized,
        "benchmark_price_return_pct": returns["qqq"], "benchmark_total_return_pct": tr["qqq"],
        "alpha_to_price_pp": pnl-returns["qqq"],
        "alpha_to_total_return_pp": pnl-tr["qqq"] if tr["qqq"] is not None else None,
        "benchmark_smh_price_return_pct": returns["smh"],
        "benchmark_smh_total_return_pct": tr["smh"],
        "benchmark_static_basket_price_return_pct": returns["static_basket"],
        "rotation_excess_vs_qqq_price_pp": pnl-returns["qqq"],
        "rotation_excess_vs_smh_price_pp": pnl-returns["smh"],
        "rotation_excess_vs_static_basket_price_pp": pnl-returns["static_basket"],
        "avg_exposure_pct": 100*mean(p["exposure"] for p in curves["strategy"]),
        "max_exposure_pct": 100*max(p["exposure"] for p in curves["strategy"]),
        "fees_paid": book["fees"], "slippage_paid": book["slippage"],
        "turnover_pct": book["turnover"], "fill_count": len(fills), "rebalance_count": len(queues),
    }
    source_bytes = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    return {**frozen, "start": start, "end": end, "source_history_sha256": source_hash,
            "price_basis_by_asset": basis,
            "source_code_sha256": frozen["policy"]["source_code_sha256"],
            "quantity_basis": "provider_split_normalized_units_not_historical_nominal_shares",
            "metrics": metrics, "equity_curve": curves["strategy"], "fills": fills,
            "rebalance_queues": queues, "decisions": decisions, "pending_order": pending,
            "benchmark_price_curve": curves["qqq"], "benchmark_smh_price_curve": curves["smh"],
            "benchmark_static_basket_curve": curves["static_basket"],
            "benchmark_fills": benchmark_fills,
            "benchmark_total_return_curve": references["qqq"],
            "benchmark_smh_total_return_curve": references["smh"],
            "benchmark_total_return_unavailable_reasons": unavailable,
            "strategy_dividends_included": False, "total_return_comparison_like_for_like": False,
            "benchmark_total_return_basis": "provider_adjusted_close_not_strict_vintage",
            "historical_point_in_time_verified": False, "goal_completion_proven": False}
