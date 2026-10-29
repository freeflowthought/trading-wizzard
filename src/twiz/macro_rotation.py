"""Weekly multi-sector allocation and optional actual-TQQQ recovery research.

One self-financing100 account; input-basis units are not historical nominal
shares. Corporate actions are never reapplied. Auxiliary timestamps are checked
before scores/facts; absent auxiliary inputs remain explicitly neutral.
"""

import bisect
import copy
import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

from .sessions import is_us_equity_session
from .structure import _atr, _clean
from .swing_optimizer import _hash, _number
from .turning_points import assess, calm_weight, index_features, validate_turning

UNIVERSE = ("QQQ", "SPY", "SMH", "XLU", "XLV", "XLP", "XLE", "XLF", "ITA", "TLT", "GLD", "TQQQ")
DEFENSIVE = ("XLU", "XLV", "XLP", "XLE", "XLF", "ITA", "TLT", "GLD")
NY = ZoneInfo("America/New_York")

DEFAULT_PARAMETERS = {
    "momentum_sessions": 126,
    "trend_sessions": 200,
    "leader_count": 2,
    "core_weight": 0.5,
    "top_profit_atr": 3,
    "top_breakout_sessions": 5,
    "support_sessions": 20,
    "confirm_sessions": 2,
    "drawdown_threshold": 0.08,
    "recovery_weight": 0,
    "recovery_take_profit": 0.1,
    "recovery_max_sessions": 10,
    "event_threshold": 0.65,
    "event_ttl_sessions": 5,
    "use_events": False,
    "use_capex": False,
}


def build_macro_rotation_policy(
    parameters, cost_multiplier=1, cycle_parameters=None, turning_parameters=None
):
    choices = {
        "momentum_sessions": (63, 126),
        "trend_sessions": (126, 200),
        "leader_count": (1, 2),
        "core_weight": (0.5, 0.7),
        "top_profit_atr": (2, 3),
        "top_breakout_sessions": (3, 5),
        "support_sessions": (20, 63),
        "confirm_sessions": (2, 3),
        "drawdown_threshold": (0.08, 0.12),
        "recovery_weight": (0, 0.25, 0.5, 1),
        "recovery_take_profit": (0.1, 0.2),
        "recovery_max_sessions": (10, 20),
        "event_threshold": (0.5, 0.65),
        "event_ttl_sessions": (5, 10),
        "use_events": (False, True),
        "use_capex": (False, True),
    }
    if (
        not isinstance(parameters, dict)
        or set(parameters) != set(choices)
        or any(parameters[key] not in values for key, values in choices.items())
    ):
        raise ValueError("Macro rotation requires the exact sixteen declared parameters.")
    for key in choices:
        kind = (
            bool
            if key.startswith("use_")
            else int
            if key.endswith("sessions") or key in {"leader_count", "top_profit_atr"}
            else (int, float)
        )
        if type(parameters[key]) not in ((kind,) if isinstance(kind, type) else kind):
            raise ValueError(f"Invalid parameter type: {key}")
    if type(cost_multiplier) not in (int, float) or cost_multiplier not in (1, 2):
        raise ValueError("Cost multiplier must be1 or2.")
    code = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("macro_rotation.py", "structure.py", "sessions.py", "swing_optimizer.py")
    }
    policy = {
        "version": "weekly-macro-sector-recovery-1",
        "parameters": copy.deepcopy(parameters),
        "universe": list(UNIVERSE),
        "defensive_candidates": list(DEFENSIVE),
        "capital": "one100cash account; target exposure<=100%; ETF internal leverage is not margin",
        "cost_multiplier": cost_multiplier,
        "fee_bps_per_side": cost_multiplier,
        "slippage_bps_per_side": 2 * cost_multiplier,
        "execution": "completedclose16:15NY to following actual open; commonpostcostNAV/sellsfirst",
        "review": "initial/lastUSsessionweekly; daily price/event/recovery transitions only",
        "leaders": "positive ROC/close>ownSMA; equal top1/2 eligible; universe-order ties",
        "core": "QQQcore while riskon; riskoff corezero, eligible defensive leaders or cash",
        "defensive": "candidates are not guaranteed protection; require own eligibility",
        "top": "growth QQQ-reference closewave>=profitATR and QQQclose<prior breakout low",
        "trend": "two QQQcloses below ownslowSMA; belowMA recovery latch protected",
        "bottom": "new episode support low aged N, currenthigherlow+risingclose>prior3high",
        "drawdown": "anchorlow vs maximum prior63completedQQQcloses; required for recovery/probe",
        "probe": "favorable event+higherlow/risingclose; min(.25,recovery_weight)",
        "event": "latest available score[-1,1]/source_refs; bounded TTL; never stockprobability",
        "event_families": "latest per family; conflicting clock/opposing strongest ties neutral",
        "event_repeat": "adverse episode triggers once, not per quote; price may override",
        "capex": "SMH: median cashcapexYoY<0 OR both same-YTDYoY accelerations<0; missing neutral",
        "capex_freshness": "currentcapex/revenue available_at_proxy<=origin and age<=130days",
        "stop": "QQQanchor-.5ATR frozen atsignal; install atfill; broken open cancels entry",
        "recovery": "actual TQQQ prices only; optional leveraged allocation within one account",
        "migration": "TQQQWAC gain, QQQ>=SMA20 or maxsessions -> nextopen replace TQQQ byQQQ",
        "campaign": "one TQQQ campaign per drawdown episode; new risk episode/new63high resets it",
        "benchmarks": "separate100 QQQ/TQQQ hold, fixed blend, weekly unleveraged sectorcontrol",
        "control": "samecore/leaders; weekly2barQQQtrend only, no localtop/events/capex/TQQQ",
        "quantity_basis": "units in each inputpricebasis, never asserted historical nominal shares",
        "dividends": "none added; adjustedclose reference/explicit syntheticTR preprocessing only",
        "source_code_sha256": code,
    }
    if cycle_parameters is not None:
        choices = {
            "top_retained": (0.5, 0.7, 1),
            "bottom_min_age": (5, 10),
            "bottom_trend_sessions": (20, 50),
            "migrate_gain": (0.1, 0.2),
            "max_holding_sessions": (20, 40),
        }
        if (
            not isinstance(cycle_parameters, dict)
            or set(cycle_parameters) not in (set(choices), set(choices) | {"defensive_after_top"})
            or (
                "defensive_after_top" in cycle_parameters
                and type(cycle_parameters["defensive_after_top"]) is not bool
            )
            or any(
                type(cycle_parameters[key]) not in (int, float)
                or cycle_parameters[key] not in values
                for key, values in choices.items()
            )
            or any(
                type(cycle_parameters[key]) is not int
                for key in ("bottom_min_age", "bottom_trend_sessions", "max_holding_sessions")
            )
        ):
            raise ValueError(
                "Cycle mode requires five bounded parameters and optional defensive_after_top bool."
            )
        policy.update(
            version="single-index-cycle-1",
            cycle_parameters=copy.deepcopy(cycle_parameters),
            allocation="ONLY QQQ/TQQQ/cash; bullbaseline100%QQQ; top retains configuredQQQ",
            bottom="aged5/10+close>SMA20/50+higherlow+prior3high+declared drawdown",
            migration="actualTQQQWACgain or20/40sessions orrisk; NO SMA20 migration",
            bull_restore="prior63completedclose newhigh restoresQQQ100 only",
            capex="notapplicable: noSMH cycleallocation; source not consumed",
            core="QQQ100 bullbaseline; configured QQq retained after localtop",
            leaders="unused by cycle; independent broadcontrol reference only",
            cycle_auxiliary="honor events; bounded probe; capex has no SMH position to restrict",
            cycle_campaign="one TQQQ campaign per dated anchor; no oldbase reentry after migration",
            cycle_rebalance="new targets only; new adverse episode resets base clock",
        )
        if cycle_parameters.get("defensive_after_top", False):
            policy.update(
                version="single-index-cycle-defensive-2",
                allocation="QQQ/TQQQ plus eligible DEFENSIVE sleeve while riskoff; one100 account",
                defensive_after_top="retainedQQQ; remaining sleeve to eligible DEFENSIVE/cash",
                defensive_review="weekly riskoff/noTQQQ; trend to defense; stop first to cash",
                leaders="existing positiveROC/ownSMA and leader_count; defensive sleeve only",
                capex="notapplicable: hybrid still has noSMH allocation; source not consumed",
            )
    if turning_parameters is not None:
        if cycle_parameters is None:
            raise ValueError("Turning layer is defined only for the cycle allocation.")
        policy.update(
            version=policy["version"] + "+market-implied-turning-1",
            turning_parameters=validate_turning(turning_parameters),
            turning="VIX/VIX3M, HYG/IEF, 2Y-Fed gap or PM Fed odds, claims; dated per session",
            turning_top="stress reason + QQQ below confirm SMA -> retain configured QQQ, no TQQQ",
            turning_bottom="term-structure inversion peak then normalisation + price turn",
            turning_calm="optional TQQQ sleeve only in calm uptrend; exits on stress/SMA50",
        )
        policy["source_code_sha256"]["turning_points.py"] = hashlib.sha256(
            (Path(__file__).parent / "turning_points.py").read_bytes()
        ).hexdigest()
    return {"policy": policy, "policy_sha256": _hash(policy)}


def _utc(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return (
            stamp.astimezone(timezone.utc)
            if stamp.tzinfo and stamp.utcoffset() is not None
            else None
        )
    except (TypeError, ValueError):
        return None


def _snapshot(rows, origin, ttl=None):
    """Availability is inspected before any score, fact, reference or other value."""
    candidates = []
    for index, row in enumerate(rows or []):
        stamp = _utc(row.get("available_at"))
        if stamp is not None and stamp <= origin:
            candidates.append((stamp, index, row))
    if not candidates:
        return None, "missing_neutral"
    stamp, _, row = max(candidates, key=lambda item: item[:2])
    if ttl is not None:
        day, age = stamp.astimezone(NY).date(), 0
        while day < origin.astimezone(NY).date() and age <= ttl:
            day = _next(day)
            if day <= origin.astimezone(NY).date():
                age += 1
        if age > ttl:
            return None, "stale_neutral"
    return row, "available"


def _event_context(events, origin, parameters):
    groups, reasons = {}, set()
    for row in events or []:
        candidate, status = _snapshot([row], origin, parameters["event_ttl_sessions"])
        if candidate is None:
            reasons.add(status)
            continue
        if row.get("expires_at") is not None:
            expiry = _utc(row.get("expires_at"))
            if expiry is None or expiry <= origin:
                reasons.add("expired_or_invalid_expiry_neutral")
                continue
        family = str(
            row.get("correlation_family") or row.get("driver") or row.get("id") or "ungrouped"
        )
        groups.setdefault(family, []).append(row)
    families, valid = {}, []
    for family, rows in sorted(groups.items()):
        latest = max(_utc(row["available_at"]) for row in rows)
        tied = [row for row in rows if _utc(row["available_at"]) == latest]
        scores = [_number(row.get("direction")) for row in tied]
        good = all(score is not None and -1 <= score <= 1 for score in scores) and all(
            isinstance(row.get("source_refs"), list) and row["source_refs"] for row in tied
        )
        if not good or len(set(scores)) != 1:
            families[family] = {
                "status": "conflicting_latest_neutral" if good else "invalid_neutral",
                "available_at": latest.isoformat(),
                "direction": None,
            }
            continue
        selected = min(
            tied, key=lambda row: (str(row.get("id", row.get("event_id", ""))), _hash(row))
        )
        families[family] = {
            "status": "available",
            "available_at": selected["available_at"],
            "direction": scores[0],
            "id": selected.get("id"),
            "source_refs": copy.deepcopy(selected["source_refs"]),
        }
        valid.append((family, scores[0], selected))
    reason = next(
        (
            value
            for value in ("expired_or_invalid_expiry_neutral", "stale_neutral", "missing_neutral")
            if value in reasons
        ),
        "missing_neutral",
    )
    result = {
        "event_status": reason,
        "direction": None,
        "event_key": None,
        "event_episode_key": None,
        "event_source": None,
        "event_families": families,
    }
    if valid:
        strength = max(abs(value) for _, value, _ in valid)
        strongest = [entry for entry in valid if abs(abs(entry[1]) - strength) <= 1e-12]
        if any(value > 0 for _, value, _ in strongest) and any(
            value < 0 for _, value, _ in strongest
        ):
            result["event_status"] = "opposing_families_neutral"
        else:
            family, value, selected = min(strongest, key=lambda entry: entry[0])
            identity = str(
                selected.get("id") or selected.get("event_id") or selected["available_at"]
            )
            episode = str(selected.get("episode_id") or selected.get("event_id") or identity)
            result.update(
                event_status="available",
                direction=value,
                event_key=(identity, selected["available_at"]),
                event_episode_key=(family, episode),
                event_source=copy.deepcopy(selected),
            )
    elif families:
        result["event_status"] = "conflicting_or_invalid_families_neutral"
    return result


def _fresh_capex(row, origin):
    fresh, excluded = {}, {}
    for ticker, issuer in row.get("issuers", {}).items():
        current = issuer.get("current", {})
        clocks = [
            _utc(current.get(kind, {}).get("available_at_proxy")) for kind in ("capex", "revenue")
        ]
        if any(
            clock is None
            or clock > origin
            or (origin.astimezone(NY).date() - clock.astimezone(NY).date()).days > 130
            for clock in clocks
        ):
            excluded[ticker] = "current_fact_future_missing_or_older_than130days"
            continue
        prior = issuer.get("prior", {})
        prior_clocks = [
            _utc(prior.get(kind, {}).get("available_at_proxy")) for kind in ("capex", "revenue")
        ]
        if any(clock is None or clock > origin for clock in prior_clocks):
            excluded[ticker] = "prior_fact_unavailable_at_origin"
            continue
        spend, revenue = (
            _number(issuer.get(key)) for key in ("cash_capex_yoy_pct", "revenue_yoy_pct")
        )
        if issuer.get("status") != "available" or spend is None or revenue is None:
            excluded[ticker] = "missing_comparable_growth"
            continue
        older = issuer.get("previous_same_duration", {})
        older_clocks = [
            _utc(older.get(kind, {}).get("available_at_proxy")) for kind in ("capex", "revenue")
        ]
        accelerations = [None, None]
        if all(clock is not None and clock <= origin for clock in older_clocks):
            values = [
                _number(issuer.get(key))
                for key in ("cash_capex_yoy_acceleration_pp", "revenue_yoy_acceleration_pp")
            ]
            if all(value is not None for value in values):
                accelerations = values
        fresh[ticker] = {
            "current": copy.deepcopy(current),
            "prior": copy.deepcopy(prior),
            "cash_capex_yoy_pct": spend,
            "revenue_yoy_pct": revenue,
            "cash_capex_yoy_acceleration_pp": accelerations[0],
            "revenue_yoy_acceleration_pp": accelerations[1],
        }
        if accelerations[0] is not None:
            fresh[ticker]["previous_same_duration"] = copy.deepcopy(older)
    if len(fresh) < 2:
        return False, None, "insufficient_fresh_issuers_neutral", excluded
    medians = {}
    for key in (
        "cash_capex_yoy_pct",
        "revenue_yoy_pct",
        "cash_capex_yoy_acceleration_pp",
        "revenue_yoy_acceleration_pp",
    ):
        values = sorted(issuer[key] for issuer in fresh.values() if issuer[key] is not None)
        mid = len(values) // 2
        medians[key] = (
            (values[mid] if len(values) % 2 else values[mid - 1] / 2 + values[mid] / 2)
            if len(values) >= 2
            else None
        )
    used_refs = {
        ref
        for issuer in fresh.values()
        for name in ("current", "prior", "previous_same_duration")
        for kind in ("capex", "revenue")
        for ref in issuer.get(name, {}).get(kind, {}).get("refs", [])
    }
    row_refs = row.get("source_refs", {})
    source_refs = (
        {key: copy.deepcopy(row_refs[key]) for key in sorted(used_refs) if key in row_refs}
        if isinstance(row_refs, dict)
        else copy.deepcopy(row_refs)
    )
    negative = medians["cash_capex_yoy_pct"] < 0
    decelerating = all(
        medians[key] is not None and medians[key] < 0
        for key in ("cash_capex_yoy_acceleration_pp", "revenue_yoy_acceleration_pp")
    )
    restriction = (
        "negative_cash_capex_yoy"
        if negative
        else "joint_same_ytd_yoy_deceleration"
        if decelerating
        else "no_cash_capex_contraction_or_joint_deceleration"
    )
    proof = {
        "available_at": max(
            _utc(issuer[name][kind]["available_at_proxy"])
            for issuer in fresh.values()
            for name in ("current", "prior", "previous_same_duration")
            if name in issuer
            for kind in ("capex", "revenue")
        ).isoformat(),
        "issuers": fresh,
        "fresh_medians": medians,
        "source_refs": source_refs,
        "restriction_reason": restriction,
        "acceleration_definition": "difference of YoY growth rates for same fiscal YTD duration",
        "excluded_issuers": excluded,
    }
    return negative or decelerating, proof, "available_fresh", excluded


def _auxiliary(events, capex, origin, parameters):
    result = (
        _event_context(events, origin, parameters)
        if parameters["use_events"]
        else {
            "event_status": "disabled",
            "direction": None,
            "event_key": None,
            "event_episode_key": None,
            "event_source": None,
            "event_families": {},
        }
    )
    direction = result["direction"]
    row, status = _snapshot(capex, origin) if parameters["use_capex"] else (None, "disabled")
    blocked, proof, excluded = False, None, {}
    if row is not None:
        blocked, proof, status, excluded = _fresh_capex(row, origin)
    return {
        **result,
        "adverse": direction is not None and direction <= -parameters["event_threshold"],
        "favorable": direction is not None and direction >= parameters["event_threshold"],
        "capex_status": status,
        "smh_restricted": blocked,
        "capex_source": proof,
        "capex_excluded_issuers": excluded,
        "capex_snapshot_at": row["available_at"] if row is not None else None,
        "smh_restriction_reason": proof["restriction_reason"] if proof else status,
    }


def _features(rows, index, parameters, episode):
    features = {}
    m, trend = parameters["momentum_sessions"], parameters["trend_sessions"]
    for asset in UNIVERSE:
        history = rows[asset][: index + 1]
        close = history[-1]["close"]
        roc, sma = close / history[-m - 1]["close"] - 1, mean(r["close"] for r in history[-trend:])
        features[asset] = {
            "close": close,
            "roc": roc,
            "slow_sma": sma,
            "eligible": asset != "TQQQ" and roc > 0 and close > sma,
        }
    q = rows["QQQ"][: index + 1]
    lower = max(0, len(q) - parameters["support_sessions"])
    candidates = [
        (i, r)
        for i, r in enumerate(q)
        if i >= lower and (episode is None or r["date"] >= episode[:10])
    ]
    anchor_i, anchor = min(candidates, key=lambda item: (item[1]["low"], -item[0]))
    qf = features["QQQ"]
    qf.update(
        atr14=_atr(q),
        sma20=mean(r["close"] for r in q[-20:]),
        prior3_high=max(r["high"] for r in q[-4:-1]),
        prior_low=min(r["low"] for r in q[-parameters["top_breakout_sessions"] - 1 : -1]),
        anchor_date=anchor["date"],
        anchor_low=anchor["low"],
        anchor_age=index - anchor_i,
        higher_low=q[-1]["low"] > q[-2]["low"],
        rising=q[-1]["close"] > q[-2]["close"],
        drawdown=1 - anchor["low"] / max(r["close"] for r in q[-64:-1]),
        new_high=q[-1]["close"] >= max(r["close"] for r in q[-64:-1]),
        trend_broken=all(
            q[-j - 1]["close"] < mean(r["close"] for r in q[len(q) - trend - j : len(q) - j])
            for j in range(2)
        ),
    )
    qf["bottom"] = (
        qf["anchor_age"] >= parameters["confirm_sessions"]
        and qf["higher_low"]
        and qf["rising"]
        and qf["close"] > qf["prior3_high"]
        and qf["drawdown"] >= parameters["drawdown_threshold"]
    )
    return features


def _allocation(features, parameters, risk_off, smh_restricted=False, recovery=0):
    candidates = DEFENSIVE if risk_off else tuple(a for a in UNIVERSE if a != "TQQQ")
    eligible = [
        a for a in candidates if features[a]["eligible"] and not (a == "SMH" and smh_restricted)
    ]
    leaders = sorted(eligible, key=lambda a: -features[a]["roc"])[: parameters["leader_count"]]
    weights = dict.fromkeys(UNIVERSE, 0.0)
    core = 0 if risk_off else parameters["core_weight"]
    weights["QQQ"] = core
    for asset in leaders:
        weights[asset] += (1 - core) / len(leaders)
    weights = {a: w * (1 - recovery) for a, w in weights.items()}
    weights["TQQQ"] = recovery
    return weights


def _clock(day, opening=False):
    return datetime.combine(day, time(9, 30) if opening else time(16, 15), NY).isoformat()


def _next(day):
    day += timedelta(days=1)
    while not is_us_equity_session(day):
        day += timedelta(days=1)
    return day


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
        start_index = bisect.bisect_right(dates, first) - 1
        if (
            start_index < minimum - 1
            or dates[start_index] != first
            or dates[-1] != last
            or any(_next(left) != right for left, right in zip(dates, dates[1:]))
        ):
            raise ValueError("Require continuous completed endpoints and enough origin history.")
        price_basis = rows[0].get("price_basis")
        if (
            not isinstance(price_basis, str)
            or not price_basis.strip()
            or any(word in price_basis.lower() for word in ("unknown", "unverified", "未注明"))
            or price_basis.strip().lower() in {"none", "n/a", "na"}
            or any(
                row.get("price_basis") != price_basis or row.get("session_complete") is not True
                for row in rows
            )
        ):
            raise ValueError("Require one explicit complete price basis per asset.")
        clean = _clean(rows, last.isoformat())
        if len(clean) != len(rows) or grid is not None and dates != grid:
            raise ValueError("Require valid OHLCV and identical complete asset session grids.")
        grid, begin = dates, start_index
        output[asset], raw[asset], basis[asset] = clean, rows, price_basis
    return output, raw, grid, begin, basis


def _book():
    return {
        "cash": 100.0,
        "positions": dict.fromkeys(UNIVERSE, 0.0),
        "basis": dict.fromkeys(UNIVERSE, 0.0),
        "fees": 0.0,
        "slippage": 0.0,
        "turnover": 0.0,
    }


def _point(book, prices, at):
    equity = book["cash"] + sum(book["positions"][a] * prices[a] for a in UNIVERSE)
    weights = {a: book["positions"][a] * prices[a] / equity for a in UNIVERSE}
    return {
        "at": at,
        "equity": equity,
        "cash": book["cash"],
        "positions": dict(book["positions"]),
        "weights": weights,
        "exposure": sum(weights.values()),
        "cost_basis": dict(book["basis"]),
    }


def _rebalance(book, weights, prices, signal_at, at, fee, slip):
    """One common post-cost NAV, not three separately funded target calculations."""
    if (
        set(weights) != set(UNIVERSE)
        or any(_number(w) is None or w < 0 for w in weights.values())
        or sum(weights.values()) > 1 + 1e-12
    ):
        raise ValueError("Rebalance targets must be nonnegative and sum to at most100%.")
    pre = book["cash"] + sum(book["positions"][a] * prices[a] for a in UNIVERSE)
    previous = dict(book["positions"])
    buy_cost, sell_cost = (1 + slip) * (1 + fee) - 1, 1 - (1 - slip) * (1 - fee)

    def costs(nav):
        return sum(
            abs(weights[a] * nav - previous[a] * prices[a])
            * (buy_cost if weights[a] * nav >= previous[a] * prices[a] else sell_cost)
            for a in UNIVERSE
        )

    low, high = 0.0, pre
    for _ in range(60):
        mid = (low + high) / 2
        if mid + costs(mid) <= pre:
            low = mid
        else:
            high = mid
    target_units = {a: weights[a] * low / prices[a] for a in UNIVERSE}
    fills = []
    for side in ("sell", "buy"):
        for asset in UNIVERSE:
            delta = target_units[asset] - book["positions"][asset]
            if (
                side == "sell"
                and delta >= 0
                or side == "buy"
                and delta <= 0
                or abs(delta) * prices[asset] <= 1e-12 * pre
            ):
                continue
            price = prices[asset] * (1 + slip if side == "buy" else 1 - slip)
            units = min(delta, book["cash"] / (price * (1 + fee))) if side == "buy" else -delta
            commission, slippage = units * price * fee, units * abs(price - prices[asset])
            if side == "buy":
                book["cash"] = max(0.0, book["cash"] - units * price * (1 + fee))
                book["positions"][asset] += units
                book["basis"][asset] += units * price * (1 + fee)
            else:
                book["basis"][asset] *= max(0.0, 1 - units / book["positions"][asset])
                book["cash"] += units * price * (1 - fee)
                book["positions"][asset] = max(0.0, book["positions"][asset] - units)
            book["fees"] += commission
            book["slippage"] += slippage
            book["turnover"] += units * prices[asset]
            fills.append(
                {
                    "asset": asset,
                    "side": side,
                    "units": units,
                    "price": price,
                    "raw_open": prices[asset],
                    "signal_at": signal_at,
                    "at": at,
                    "fee": commission,
                    "slippage": slippage,
                    "cash_after": book["cash"],
                    "units_after": book["positions"][asset],
                }
            )
    point = _point(book, prices, at)
    return {
        "signal_at": signal_at,
        "at": at,
        "target_weights": dict(weights),
        "pre_nav": pre,
        "solved_postcost_nav": low,
        "actual_postcost_nav": point["equity"],
        "solver_residual": low + costs(low) - pre,
        "actual_weights": point["weights"],
        "fills": fills,
    }


def _drawdown(curve):
    peak, worst = 100.0, 0.0
    for point in curve:
        peak = max(peak, point["equity"])
        worst = max(worst, 100 * (1 - point["equity"] / peak))
    return worst


def simulate_macro_rotation(
    market,
    start,
    end,
    *,
    parameters,
    event_rows=None,
    capex_rows=None,
    cost_multiplier=1,
    cycle_parameters=None,
    live_tail_clock=None,
    turning=None,
):
    if cycle_parameters is not None:
        return _simulate_cycle(
            market,
            start,
            end,
            parameters,
            cycle_parameters,
            event_rows,
            cost_multiplier,
            live_tail_clock,
            turning,
        )
    if turning is not None:
        raise ValueError("Turning layer is defined only for the cycle allocation.")
    if live_tail_clock is not None:
        raise ValueError("Live tail clock is supported only for the cycle delivery plan.")
    frozen = build_macro_rotation_policy(parameters, cost_multiplier)
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first >= last or not is_us_equity_session(first) or not is_us_equity_session(last):
        raise ValueError("Require increasing completed-session endpoints.")
    rows, raw, days, begin, basis = _prices(market, first, last, 260)
    fee, slip = cost_multiplier / 10000, cost_multiplier * 2 / 10000
    books = {
        name: _book() for name in ("strategy", "qqq", "tqqq", "static_blend", "sector_control")
    }
    curves, fills = {name: [] for name in books}, {name: [] for name in books}
    decisions, queues, unfilled, sources = [], [], [], {}
    pending, control_pending, target = None, None, dict.fromkeys(UNIVERSE, 0.0)
    risk_off, armed, episode, stop, stop_at = False, True, None, None, None
    growth_reference, growth_peak, entry_index, stage = None, None, None, 0.0
    used_recovery, adverse_episodes = False, set()
    for i in range(begin, len(days)):
        day, book = days[i], books["strategy"]
        opens = {a: rows[a][i]["open"] for a in UNIVERSE}
        if i > begin:
            if pending:
                info = pending.get("recovery_info")
                cancel = info is not None and opens["QQQ"] <= info["qqq_stop"]
                if cancel:
                    risk_off, target = pending["prior_risk_off"], pending["prior_target"]
                    unfilled.append(
                        {
                            **pending,
                            "at": _clock(day, True),
                            "reason": "QQQ_open_broke_frozen_recovery_stop",
                        }
                    )
                else:
                    had_growth = any(
                        book["positions"][a] > 0 for a in ("QQQ", "SPY", "SMH", "TQQQ")
                    )
                    had_tqqq = book["positions"]["TQQQ"] > 0
                    execution = _rebalance(
                        book,
                        pending["weights"],
                        opens,
                        pending["known_at"],
                        _clock(day, True),
                        fee,
                        slip,
                    )
                    execution["cause"] = pending["cause"]
                    queues.append(execution)
                    for fill in execution["fills"]:
                        fills["strategy"].append({**fill, "cause": pending["cause"]})
                    has_growth = any(
                        book["positions"][a] > 0 for a in ("QQQ", "SPY", "SMH", "TQQQ")
                    )
                    if has_growth and not had_growth:
                        growth_reference, growth_peak = opens["QQQ"], None
                    if not has_growth:
                        growth_reference, growth_peak, stop, stop_at = None, None, None, None
                    if info and has_growth and execution["fills"]:
                        stop = stop if stop is not None else info["qqq_stop"]
                        stop_at = stop_at or _clock(day, True)
                        armed = info["signal_close"] >= info["signal_slow_sma"]
                    if book["positions"]["TQQQ"] > 0:
                        stage = pending.get("recovery_stage", stage)
                        if not had_tqqq:
                            entry_index, used_recovery = i, True
                    else:
                        entry_index, stage = None, 0.0
                pending = None
            if control_pending:
                execution = _rebalance(
                    books["sector_control"],
                    control_pending["weights"],
                    opens,
                    control_pending["known_at"],
                    _clock(day, True),
                    fee,
                    slip,
                )
                fills["sector_control"].extend(execution["fills"])
                control_pending = None
            if i == begin + 1:
                r = parameters["recovery_weight"]
                for name, allocation in (
                    ("qqq", {"QQQ": 1}),
                    ("tqqq", {"TQQQ": 1}),
                    ("static_blend", {"QQQ": 1 - r, "TQQQ": r}),
                ):
                    weights = {a: allocation.get(a, 0.0) for a in UNIVERSE}
                    execution = _rebalance(
                        books[name], weights, opens, _clock(first), _clock(day, True), fee, slip
                    )
                    fills[name].extend(execution["fills"])
        at = _clock(day)
        origin = _utc(at)
        auxiliary = _auxiliary(event_rows, capex_rows, origin, parameters)
        # Deduplicate large filing/quote proofs instead of copying each into every daily trace.
        for key in ("event_source", "capex_source"):
            if auxiliary[key] is not None:
                digest = _hash(auxiliary[key])
                sources[digest] = auxiliary[key]
                auxiliary[key] = {"sha256": digest, "available_at": auxiliary[key]["available_at"]}
        features = _features(rows, i, parameters, episode)
        q = features["QQQ"]
        if i == begin:
            risk_off = q["trend_broken"]
            episode = at if risk_off else None
            if episode:
                features = _features(rows, i, parameters, episode)
                q = features["QQQ"]
        if any(book["positions"][a] > 0 for a in ("QQQ", "SPY", "SMH", "TQQQ")):
            growth_peak = max(growth_peak or q["close"], q["close"])
        if q["close"] >= q["slow_sma"]:
            armed = True
        weekly = i == begin or _next(day).isocalendar()[:2] != day.isocalendar()[:2]
        wave = (
            (growth_peak - growth_reference) / q["atr14"]
            if growth_reference is not None and q["atr14"] > 0
            else None
        )
        stop_failed = stop is not None and q["close"] < stop
        trend_failed = armed and q["trend_broken"] and not risk_off
        top = (
            wave is not None
            and wave >= parameters["top_profit_atr"]
            and q["close"] < q["prior_low"]
        )
        adverse = auxiliary["adverse"] and auxiliary["event_episode_key"] not in adverse_episodes
        t_units = book["positions"]["TQQQ"]
        t_cost = book["basis"]["TQQQ"] / t_units if t_units else None
        t_gain = features["TQQQ"]["close"] / t_cost - 1 if t_cost else None
        holding = i - entry_index + 1 if entry_index is not None else 0
        weights, info, cause, desired_stage = None, None, "hold", stage
        prior_risk = risk_off
        if stop_failed or trend_failed or top or adverse:
            risk_off, episode, used_recovery = True, at, False
            if adverse:
                adverse_episodes.add(auxiliary["event_episode_key"])
            cause = (
                "recovery_structure_failed"
                if stop_failed
                else "adverse_event"
                if adverse
                else "local_top"
                if top
                else "long_trend_risk"
            )
            weights = (
                dict.fromkeys(UNIVERSE, 0.0)
                if stop_failed
                else _allocation(features, parameters, True, auxiliary["smh_restricted"])
            )
            desired_stage = 0.0
            features = _features(rows, i, parameters, episode)
            q = features["QQQ"]
        elif t_units and (
            t_gain >= parameters["recovery_take_profit"]
            or q["close"] >= q["sma20"]
            or holding >= parameters["recovery_max_sessions"]
        ):
            cause = (
                "tqqq_to_qqq_gain"
                if t_gain >= parameters["recovery_take_profit"]
                else "tqqq_to_qqq_sma20"
                if q["close"] >= q["sma20"]
                else "tqqq_to_qqq_time"
            )
            weights = dict(target)
            weights["QQQ"] += weights["TQQQ"]
            weights["TQQQ"], desired_stage = 0.0, 0.0
            if q["close"] >= q["sma20"]:
                risk_off = False
        elif q["bottom"] and (
            risk_off
            or parameters["recovery_weight"] > 0
            and not used_recovery
            or t_units
            and stage < parameters["recovery_weight"]
        ):
            risk_off, desired_stage, cause = (
                False,
                parameters["recovery_weight"],
                "confirmed_bottom",
            )
            weights = _allocation(
                features, parameters, False, auxiliary["smh_restricted"], desired_stage
            )
            info = {
                "qqq_stop": stop or q["anchor_low"] - 0.5 * q["atr14"],
                "anchor_date": q["anchor_date"],
                "known_at": at,
                "signal_close": q["close"],
                "signal_slow_sma": q["slow_sma"],
            }
        elif (
            parameters["recovery_weight"] > 0
            and not used_recovery
            and auxiliary["favorable"]
            and q["drawdown"] >= parameters["drawdown_threshold"]
            and q["anchor_age"] >= 1
            and q["higher_low"]
            and q["rising"]
        ):
            desired_stage, cause = min(0.25, parameters["recovery_weight"]), "event_bounded_probe"
            weights = _allocation(
                features, parameters, risk_off, auxiliary["smh_restricted"], desired_stage
            )
            info = {
                "qqq_stop": stop or q["anchor_low"] - 0.5 * q["atr14"],
                "anchor_date": q["anchor_date"],
                "known_at": at,
                "signal_close": q["close"],
                "signal_slow_sma": q["slow_sma"],
            }
        elif (
            risk_off and q["close"] >= q["sma20"] and q["rising"] and q["close"] > q["prior3_high"]
        ):
            risk_off, cause = False, "unleveraged_price_repair"
            weights = _allocation(features, parameters, False, auxiliary["smh_restricted"])
            info = {
                "qqq_stop": stop or q["anchor_low"] - 0.5 * q["atr14"],
                "anchor_date": q["anchor_date"],
                "known_at": at,
                "signal_close": q["close"],
                "signal_slow_sma": q["slow_sma"],
            }
        elif weekly:
            cause = "initial_allocation" if i == begin else "weekly_rotation"
            weights = _allocation(
                features, parameters, risk_off, auxiliary["smh_restricted"], stage
            )
        if q["new_high"] and not t_units:
            used_recovery = False
        if weights is not None:
            if info is None or info["qqq_stop"] > 0:
                pending = {
                    "weights": weights,
                    "known_at": at,
                    "cause": cause,
                    "prior_risk_off": prior_risk,
                    "prior_target": dict(target),
                    "recovery_stage": desired_stage,
                    "recovery_info": info,
                }
                target = weights
        if weekly:
            control_pending = {
                "weights": _allocation(features, parameters, q["trend_broken"]),
                "known_at": at,
            }
        closes = {a: rows[a][i]["close"] for a in UNIVERSE}
        for name, value in books.items():
            curves[name].append(_point(value, closes, at))
        decisions.append(
            {
                "at": at,
                "cause": cause,
                "weekly_review": weekly,
                "features": features,
                "auxiliary": auxiliary,
                "target_weights": dict(target),
                "risk_off": risk_off,
                "risk_armed": armed,
                "episode_start": episode,
                "growth_reference": growth_reference,
                "growth_peak": growth_peak,
                "wave_profit_atr": wave,
                "recovery_stop": stop,
                "stop_installed_at": stop_at,
                "proposed_recovery": info,
                "recovery_stage": stage,
                "tqqq_entry_at": _clock(days[entry_index], True)
                if entry_index is not None
                else None,
                "tqqq_entry_wac": t_cost,
                "tqqq_gain": t_gain,
                "holding_sessions": holding,
                "recovery_campaign_used": used_recovery,
                "order_scheduled": pending is not None,
            }
        )
    references, unavailable = {}, {}
    for name, asset in (("qqq", "QQQ"), ("tqqq", "TQQQ")):
        adjusted = [_number(r.get("provider_adjusted_close")) for r in rows[asset][begin + 1 :]]
        if any(value is None or value <= 0 for value in adjusted):
            references[name], unavailable[name] = None, "missing_or_invalid_provider_adjusted_close"
        else:
            initial = rows[asset][begin + 1]
            adjusted_open = initial["open"] * adjusted[0] / initial["close"]
            references[name] = [{"at": _clock(first), "equity": 100.0}] + [
                {"at": _clock(day), "equity": 100 * adj / (adjusted_open * (1 + slip) * (1 + fee))}
                for day, adj in zip(days[begin + 1 :], adjusted)
            ]
    returns = {name: curve[-1]["equity"] - 100 for name, curve in curves.items()}
    pnl = returns["strategy"]
    qtr = references["qqq"][-1]["equity"] - 100 if references["qqq"] else None
    realized = books["strategy"]["cash"] + sum(books["strategy"]["basis"].values()) - 100
    metrics = {
        "net_account_return_pct": pnl,
        "max_drawdown_pct": _drawdown(curves["strategy"]),
        "realized_account_pnl_pct": realized,
        "unrealized_account_pnl_pct": pnl - realized,
        "benchmark_price_return_pct": returns["qqq"],
        "benchmark_total_return_pct": qtr,
        "alpha_to_price_pp": pnl - returns["qqq"],
        "alpha_to_total_return_pp": pnl - qtr if qtr is not None else None,
        "benchmark_tqqq_price_return_pct": returns["tqqq"],
        "benchmark_static_blend_return_pct": returns["static_blend"],
        "benchmark_sector_control_return_pct": returns["sector_control"],
        "excess_vs_tqqq_pp": pnl - returns["tqqq"],
        "excess_vs_static_blend_pp": pnl - returns["static_blend"],
        "excess_vs_sector_control_pp": pnl - returns["sector_control"],
        "turnover_pct": books["strategy"]["turnover"],
        "fill_count": len(fills["strategy"]),
        "fees_paid": books["strategy"]["fees"],
        "slippage_paid": books["strategy"]["slippage"],
        "avg_exposure_pct": 100 * mean(p["exposure"] for p in curves["strategy"]),
        "max_exposure_pct": 100 * max(p["exposure"] for p in curves["strategy"]),
    }
    return {
        **frozen,
        "start": start,
        "end": end,
        "metrics": metrics,
        "source_code_sha256": frozen["policy"]["source_code_sha256"],
        "source_history_sha256": hashlib.sha256(
            json.dumps(raw, sort_keys=True).encode()
        ).hexdigest(),
        "price_basis_by_asset": basis,
        "quantity_basis": "input_price_basis_units_not_nominal_shares",
        "equity_curve": curves["strategy"],
        "fills": fills["strategy"],
        "decisions": decisions,
        "rebalance_queues": queues,
        "unfilled_orders": unfilled,
        "pending_order": pending,
        "benchmark_price_curve": curves["qqq"],
        "benchmark_tqqq_price_curve": curves["tqqq"],
        "benchmark_static_blend_curve": curves["static_blend"],
        "benchmark_sector_control_curve": curves["sector_control"],
        "benchmark_fills": {name: value for name, value in fills.items() if name != "strategy"},
        "benchmark_total_return_curve": references["qqq"],
        "benchmark_tqqq_total_return_curve": references["tqqq"],
        "benchmark_total_return_unavailable_reasons": unavailable,
        "auxiliary_sources_used": sources,
        "auxiliary_sources_used_sha256": _hash(sources),
        "strategy_dividends_included": False,
        "total_return_comparison_like_for_like": False,
        "benchmark_total_return_basis": "provider_adjusted_close_not_strict_vintage",
        "historical_point_in_time_verified": False,
        "goal_completion_proven": False,
    }


def _cycle_weights(qqq=0.0, tqqq=0.0):
    weights = dict.fromkeys(UNIVERSE, 0.0)
    weights.update(QQQ=qqq, TQQQ=tqqq)
    return weights


def _cycle_allocation(features, parameters, retained):
    """Fill only the unused cycle sleeve; ineligible defensive assets remain cash."""
    weights = _cycle_weights(retained)
    eligible = [asset for asset in DEFENSIVE if features[asset]["eligible"]]
    leaders = sorted(eligible, key=lambda asset: -features[asset]["roc"])[
        : parameters["leader_count"]
    ]
    for asset in leaders:
        weights[asset] = (1 - retained) / len(leaders)
    return weights


def _simulate_cycle(
    market,
    start,
    end,
    parameters,
    cycle,
    event_rows,
    cost_multiplier,
    live_tail_clock=None,
    turning=None,
):
    tp = turning["parameters"] if turning is not None else None
    frozen = build_macro_rotation_policy(parameters, cost_multiplier, cycle, tp)
    turn_rows = index_features(turning["features"]) if turning is not None else {}
    calm_on, stress_trim, capitulation_index = False, False, None
    hybrid = cycle.get("defensive_after_top", False)
    cycle_riskoff = False
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first >= last or not is_us_equity_session(first) or not is_us_equity_session(last):
        raise ValueError("Require increasing completed session endpoints.")
    rows, raw, days, begin, basis = _prices(market, first, last, 260)
    if live_tail_clock is not None:
        if (
            not isinstance(live_tail_clock, datetime)
            or live_tail_clock.tzinfo is None
            or live_tail_clock.utcoffset() is None
            or not _utc(_clock(last)) <= live_tail_clock < _utc(_clock(_next(last), True))
        ):
            raise ValueError(
                "Tail clock must be aware and between last completed close and next open."
            )
    fee, slip = cost_multiplier / 10000, cost_multiplier * 2 / 10000
    books = {
        name: _book() for name in ("strategy", "qqq", "tqqq", "static_blend", "sector_control")
    }
    curves, fills = {name: [] for name in books}, {name: [] for name in books}
    decisions, queues, unfilled, sources = [], [], [], {}
    target = _cycle_weights(1)
    pending = {
        "weights": target,
        "known_at": _clock(first),
        "cause": "cycle_initial_qqq",
        "prior_target": _cycle_weights(),
        "recovery_info": None,
    }
    control_pending = None
    armed, trimmed, episode, stop, stop_at = True, False, None, None, None
    reference, peak, entry_index, stage = None, None, None, 0.0
    adverse_episodes, consumed_anchors = set(), set()
    for i in range(begin, len(days)):
        day, book = days[i], books["strategy"]
        opens = {a: rows[a][i]["open"] for a in UNIVERSE}
        if i > begin:
            if pending:
                info = pending.get("recovery_info")
                if info is not None and opens["QQQ"] <= info["qqq_stop"]:
                    target = pending["prior_target"]
                    unfilled.append(
                        {
                            **pending,
                            "at": _clock(day, True),
                            "reason": "QQQ_open_broke_frozen_recovery_stop",
                        }
                    )
                else:
                    had_t = book["positions"]["TQQQ"] > 0
                    had_growth = book["positions"]["QQQ"] + book["positions"]["TQQQ"] > 0
                    execution = _rebalance(
                        book,
                        pending["weights"],
                        opens,
                        pending["known_at"],
                        _clock(day, True),
                        fee,
                        slip,
                    )
                    queues.append({**execution, "cause": pending["cause"]})
                    fills["strategy"].extend(
                        {**fill, "cause": pending["cause"]} for fill in execution["fills"]
                    )
                    if execution["fills"]:
                        if not book["positions"]["QQQ"] and not book["positions"]["TQQQ"]:
                            if (
                                not hybrid
                                or had_growth
                                or pending["cause"]
                                in ("cycle_long_trend_exit", "cycle_structural_stop")
                            ):
                                stop, stop_at, reference, peak = None, None, None, None
                                episode, trimmed = pending["known_at"], False
                        elif pending["cause"] in (
                            "cycle_local_top",
                            "cycle_adverse_event",
                            "cycle_implied_stress_top",
                        ):
                            episode, trimmed = (
                                pending["known_at"],
                                target["QQQ"] < 1 if hybrid else sum(target.values()) < 1,
                            )
                        elif target["QQQ"] == 1 and target["TQQQ"] == 0:
                            trimmed, reference, peak = False, opens["QQQ"], None
                        if hybrid and (target["TQQQ"] > 0 or target["QQQ"] == 1):
                            cycle_riskoff = False
                        if (
                            not had_growth
                            and sum(book["positions"].values()) > 0
                            and (
                                not hybrid
                                or book["positions"]["QQQ"] + book["positions"]["TQQQ"] > 0
                            )
                        ):
                            reference, peak = opens["QQQ"], None
                        if info and (book["positions"]["QQQ"] > 0 or book["positions"]["TQQQ"] > 0):
                            stop, stop_at = stop or info["qqq_stop"], stop_at or _clock(day, True)
                            armed = info["signal_close"] >= info["signal_slow_sma"]
                        if book["positions"]["TQQQ"] > 0:
                            if not had_t:
                                entry_index = i
                                if info is not None:  # calm sleeve has no dated anchor
                                    consumed_anchors.add(info["anchor_date"])
                            stage = target["TQQQ"]
                        else:
                            entry_index, stage = None, 0.0
                pending = None
            if control_pending:
                execution = _rebalance(
                    books["sector_control"],
                    control_pending["weights"],
                    opens,
                    control_pending["known_at"],
                    _clock(day, True),
                    fee,
                    slip,
                )
                fills["sector_control"].extend(execution["fills"])
                control_pending = None
            if i == begin + 1:
                r = parameters["recovery_weight"]
                for name, allocation in (
                    ("qqq", _cycle_weights(1)),
                    ("tqqq", _cycle_weights(0, 1)),
                    ("static_blend", _cycle_weights(1 - r, r)),
                ):
                    execution = _rebalance(
                        books[name], allocation, opens, _clock(first), _clock(day, True), fee, slip
                    )
                    fills[name].extend(execution["fills"])
        price_at = _clock(day)
        at = (
            live_tail_clock.isoformat() if live_tail_clock is not None and day == last else price_at
        )
        origin = _utc(at)
        auxiliary = _auxiliary(event_rows, None, origin, {**parameters, "use_capex": False})
        auxiliary["capex_status"] = "not_applicable_no_SMH_cycle_allocation"
        if auxiliary["event_source"] is not None:
            digest = _hash(auxiliary["event_source"])
            sources[digest] = auxiliary["event_source"]
            auxiliary["event_source"] = {
                "sha256": digest,
                "available_at": auxiliary["event_source"]["available_at"],
            }
        feature_episode = episode
        features = _features(rows, i, parameters, feature_episode)
        q = features["QQQ"]
        q["cycle_bottom_sma"] = mean(
            r["close"] for r in rows["QQQ"][i - cycle["bottom_trend_sessions"] + 1 : i + 1]
        )
        q["cycle_bottom_ready"] = (
            q["anchor_age"] >= cycle["bottom_min_age"]
            and q["higher_low"]
            and q["rising"]
            and q["close"] > q["prior3_high"]
            and q["close"] > q["cycle_bottom_sma"]
            and q["drawdown"] >= parameters["drawdown_threshold"]
        )
        held = book["positions"]["QQQ"] + book["positions"]["TQQQ"] > 0
        if held:
            peak = max(peak or q["close"], q["close"])
        if q["close"] >= q["slow_sma"]:
            armed = True
        wave = (peak - reference) / q["atr14"] if reference is not None and q["atr14"] > 0 else None
        t_units = book["positions"]["TQQQ"]
        t_cost = book["basis"]["TQQQ"] / t_units if t_units else None
        t_gain = features["TQQQ"]["close"] / t_cost - 1 if t_cost else None
        holding = i - entry_index + 1 if entry_index is not None else 0
        stop_fail = held and stop is not None and q["close"] < stop
        trend_fail = held and armed and q["trend_broken"]
        top = (
            held
            and not trimmed
            and wave is not None
            and wave >= parameters["top_profit_atr"]
            and q["close"] < q["prior_low"]
        )
        adverse = auxiliary["adverse"] and auxiliary["event_episode_key"] not in adverse_episodes
        signal = None
        if tp is not None:
            closes_q = [r["close"] for r in rows["QQQ"][max(0, i - 49) : i + 1]]
            q["sma50"], q["sma10"] = mean(closes_q), mean(closes_q[-10:])
            signal = assess(turn_rows.get(day.isoformat()), q, tp)
            if t_units and not calm_on and signal["stress"]:
                # Inside a bottom campaign only a renewed term-structure inversion counts.
                signal["stress"] = "vix_term_inversion" in signal["stress_reasons"]
            if (
                signal["stress"]
                and t_units
                and capitulation_index is not None
                and 0 < i - capitulation_index <= tp["capitulation_grace_sessions"]
                and q["close"] >= q["anchor_low"]
            ):
                # Fresh capitulation entries are judged by the anchor low, not a re-inversion.
                signal["stress"], signal["grace_suppressed"] = False, True
            auxiliary["turning"] = signal
        exposure_now = sum(target.values())
        weights, info, cause = None, None, "cycle_hold"
        if i == begin and not adverse:
            cause = "cycle_initial_qqq"
        elif stop_fail or trend_fail:
            weights, cause = (
                _cycle_allocation(features, parameters, 0)
                if hybrid and not stop_fail
                else _cycle_weights(),
                "cycle_structural_stop" if stop_fail else "cycle_long_trend_exit",
            )
            if hybrid:
                cycle_riskoff = True
        elif signal is not None and held and signal["stress"] and (not trimmed or t_units):
            retained = min(exposure_now, tp["stress_retained"])
            weights, cause = _cycle_weights(retained), "cycle_implied_stress_top"
            stress_trim = True
        elif adverse or top:
            exposure = target["QQQ"] + target["TQQQ"] if hybrid else sum(target.values())
            retained = min(exposure, cycle["top_retained"])
            weights, cause = (
                _cycle_allocation(features, parameters, retained)
                if hybrid
                else _cycle_weights(retained),
                "cycle_adverse_event" if adverse else "cycle_local_top",
            )
            if hybrid:
                cycle_riskoff = True
            if adverse:
                adverse_episodes.add(auxiliary["event_episode_key"])
                episode = at
        elif calm_on and t_units and signal is not None and signal["calm_exit"]:
            weights, cause = _cycle_weights(exposure_now), "cycle_calm_exit"
        elif (
            signal is not None
            and not t_units
            and signal["capitulation"]
            and q["drawdown"] >= parameters["drawdown_threshold"]
            and q["higher_low"]
            and q["rising"]
            and q["close"] > q["prior3_high"]
            and q["anchor_date"] not in consumed_anchors
        ):
            w = tp["capitulation_weight"]
            weights, cause = _cycle_weights(1 - w, w), "cycle_capitulation_bottom"
        elif not calm_on and t_units and (
            t_gain >= cycle["migrate_gain"] or holding >= cycle["max_holding_sessions"]
        ):
            weights = _cycle_weights(sum(target.values()))
            cause = (
                "cycle_tqqq_gain_to_qqq"
                if t_gain >= cycle["migrate_gain"]
                else "cycle_tqqq_time_to_qqq"
            )
        elif q["cycle_bottom_ready"] and (
            sum(target.values()) < 1
            or hybrid
            and cycle_riskoff
            or parameters["recovery_weight"] > stage
            and (t_units or q["anchor_date"] not in consumed_anchors)
        ):
            recovery = (
                parameters["recovery_weight"]
                if t_units or q["anchor_date"] not in consumed_anchors
                else 0
            )
            weights, cause = _cycle_weights(1 - recovery, recovery), "cycle_confirmed_bottom"
        elif not t_units and q["new_high"] and target["QQQ"] < 1:
            weights, cause = _cycle_weights(1), "cycle_bull_qqq_restore"
        elif (
            signal is not None
            and stress_trim
            and not t_units
            and exposure_now < 1
            and signal["status"] == "available"
            and not signal["stress_reasons"]
            and signal["evidence"]["vix_ratio"] <= tp["normalized_ratio"]
            and q["close"] > q["sma20"]
            and q["close"] > q["prior3_high"]
        ):
            weights, cause = _cycle_weights(1), "cycle_implied_stress_cleared"
        elif (
            signal is not None
            and tp["calm_tqqq"] > 0
            and not t_units
            and exposure_now >= 1 - 1e-9
            and signal["calm"]
            and calm_weight(turn_rows.get(day.isoformat()), tp) >= 0.05
        ):
            w = calm_weight(turn_rows.get(day.isoformat()), tp)
            weights, cause = _cycle_weights(1 - w, w), "cycle_calm_leverage"
        elif (
            parameters["recovery_weight"] > 0
            and not t_units
            and q["anchor_date"] not in consumed_anchors
            and auxiliary["favorable"]
            and q["anchor_age"] >= 1
            and q["higher_low"]
            and q["rising"]
            and q["drawdown"] >= parameters["drawdown_threshold"]
        ):
            probe = min(0.25, parameters["recovery_weight"])
            weights, cause = (
                _cycle_weights(min(target["QQQ"], 1 - probe), probe),
                "cycle_event_probe",
            )
        if (
            signal is not None
            and weights is not None
            and weights["TQQQ"] > 0
            and cause in ("cycle_confirmed_bottom", "cycle_event_probe")
            and (not signal["leverage_ok"] or "vix_term_inversion" in signal["stress_reasons"])
        ):
            # Price-only bottoms below the long trend restore QQQ notional, not 3x exposure.
            weights = _cycle_weights(min(1.0, weights["QQQ"] + weights["TQQQ"]))
        if cause in (
            "cycle_confirmed_bottom",
            "cycle_bull_qqq_restore",
            "cycle_event_probe",
            "cycle_capitulation_bottom",
            "cycle_implied_stress_cleared",
        ):
            info = {
                "qqq_stop": stop or q["anchor_low"] - 0.5 * q["atr14"],
                "anchor_date": q["anchor_date"],
                "known_at": at,
                "signal_close": q["close"],
                "signal_slow_sma": q["slow_sma"],
            }
        weekly = i == begin or _next(day).isocalendar()[:2] != day.isocalendar()[:2]
        if hybrid and weights is None and weekly and cycle_riskoff and not t_units:
            weights = _cycle_allocation(features, parameters, target["QQQ"])
            cause = "cycle_defensive_weekly_review"
        if weights is not None and weights != target and (info is None or info["qqq_stop"] > 0):
            if tp is not None:
                calm_on = cause == "cycle_calm_leverage" or (calm_on and weights["TQQQ"] > 0)
                stress_trim = cause == "cycle_implied_stress_top"
                if cause == "cycle_capitulation_bottom":
                    capitulation_index = i
            pending = {
                "weights": weights,
                "known_at": at,
                "cause": cause,
                "prior_target": dict(target),
                "recovery_info": info,
            }
            target = weights
        weekly = i == begin or _next(day).isocalendar()[:2] != day.isocalendar()[:2]
        if weekly:
            control_pending = {
                "weights": _allocation(features, parameters, q["trend_broken"]),
                "known_at": at,
            }
        closes = {a: rows[a][i]["close"] for a in UNIVERSE}
        for name, value in books.items():
            curves[name].append(_point(value, closes, price_at))
        decisions.append(
            {
                "at": at,
                **(
                    {"price_at": price_at, "live_tail_clock": at}
                    if live_tail_clock is not None and day == last
                    else {}
                ),
                "cause": cause,
                "features": features,
                "auxiliary": auxiliary,
                "target_weights": dict(target),
                "risk_armed": armed,
                "trimmed": trimmed,
                "episode_start": episode,
                "feature_episode_start": feature_episode,
                "growth_reference": reference,
                "growth_peak": peak,
                "wave_profit_atr": wave,
                "recovery_stop": stop,
                "stop_installed_at": stop_at,
                "proposed_recovery": info,
                "recovery_stage": stage,
                "tqqq_entry_at": _clock(days[entry_index], True)
                if entry_index is not None
                else None,
                "tqqq_entry_wac": t_cost,
                "tqqq_gain": t_gain,
                "holding_sessions": holding,
                "consumed_recovery_anchors": sorted(consumed_anchors),
                **({"cycle_risk_off": cycle_riskoff, "weekly_review": weekly} if hybrid else {}),
                "order_scheduled": pending is not None,
            }
        )
    references, unavailable = {}, {}
    for name, asset in (("qqq", "QQQ"), ("tqqq", "TQQQ")):
        adjusted = [_number(r.get("provider_adjusted_close")) for r in rows[asset][begin + 1 :]]
        if any(value is None or value <= 0 for value in adjusted):
            references[name], unavailable[name] = None, "missing_or_invalid_provider_adjusted_close"
        else:
            initial = rows[asset][begin + 1]
            adjusted_open = initial["open"] * adjusted[0] / initial["close"]
            references[name] = [{"at": _clock(first), "equity": 100.0}] + [
                {"at": _clock(day), "equity": 100 * adj / (adjusted_open * (1 + slip) * (1 + fee))}
                for day, adj in zip(days[begin + 1 :], adjusted)
            ]
    returns = {name: curve[-1]["equity"] - 100 for name, curve in curves.items()}
    pnl = returns["strategy"]
    qtr = references["qqq"][-1]["equity"] - 100 if references["qqq"] else None
    realized = books["strategy"]["cash"] + sum(books["strategy"]["basis"].values()) - 100
    metrics = {
        "net_account_return_pct": pnl,
        "max_drawdown_pct": _drawdown(curves["strategy"]),
        "realized_account_pnl_pct": realized,
        "unrealized_account_pnl_pct": pnl - realized,
        "benchmark_price_return_pct": returns["qqq"],
        "benchmark_total_return_pct": qtr,
        "alpha_to_price_pp": pnl - returns["qqq"],
        "alpha_to_total_return_pp": pnl - qtr if qtr is not None else None,
        "benchmark_tqqq_price_return_pct": returns["tqqq"],
        "benchmark_static_blend_return_pct": returns["static_blend"],
        "benchmark_sector_control_return_pct": returns["sector_control"],
        "excess_vs_tqqq_pp": pnl - returns["tqqq"],
        "excess_vs_static_blend_pp": pnl - returns["static_blend"],
        "excess_vs_sector_control_pp": pnl - returns["sector_control"],
        "turnover_pct": books["strategy"]["turnover"],
        "fill_count": len(fills["strategy"]),
        "fees_paid": books["strategy"]["fees"],
        "slippage_paid": books["strategy"]["slippage"],
        "avg_exposure_pct": 100 * mean(p["exposure"] for p in curves["strategy"]),
        "max_exposure_pct": 100 * max(p["exposure"] for p in curves["strategy"]),
    }
    return {
        **frozen,
        "start": start,
        "end": end,
        "metrics": metrics,
        "source_code_sha256": frozen["policy"]["source_code_sha256"],
        "source_history_sha256": hashlib.sha256(
            json.dumps(raw, sort_keys=True).encode()
        ).hexdigest(),
        "price_basis_by_asset": basis,
        "quantity_basis": "input_price_basis_units_not_nominal_shares",
        "equity_curve": curves["strategy"],
        "fills": fills["strategy"],
        "decisions": decisions,
        "rebalance_queues": queues,
        "unfilled_orders": unfilled,
        "pending_order": pending,
        "benchmark_price_curve": curves["qqq"],
        "benchmark_tqqq_price_curve": curves["tqqq"],
        "benchmark_static_blend_curve": curves["static_blend"],
        "benchmark_sector_control_curve": curves["sector_control"],
        "benchmark_fills": {name: value for name, value in fills.items() if name != "strategy"},
        "benchmark_total_return_curve": references["qqq"],
        "benchmark_tqqq_total_return_curve": references["tqqq"],
        "benchmark_total_return_unavailable_reasons": unavailable,
        "auxiliary_sources_used": sources,
        "auxiliary_sources_used_sha256": _hash(sources),
        "strategy_dividends_included": False,
        "total_return_comparison_like_for_like": False,
        "benchmark_total_return_basis": "provider_adjusted_close_not_strict_vintage",
        "historical_point_in_time_verified": False,
        "goal_completion_proven": False,
    }
