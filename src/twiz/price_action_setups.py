"""Causal daily bottom hypotheses: a low followed by multiple completed sessions.

This is an explicit price-only research rule, not a reconstructed author trade.
Setup generation never reads subsequent prices or requires event probabilities.
"""

import bisect
import copy
import hashlib
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .sessions import is_us_equity_session
from .structure import _atr, _clean, validate_structure_parameters
from .swing_optimizer import (
    CORE_KEYS,
    STRATEGY_KEYS,
    VARIANTS,
    _hash,
    _number,
    validate_candidate_parameters,
)

VERSION = "daily-confirmed-bottom-setups-1"
NY = ZoneInfo("America/New_York")
POLICY_BOUNDS = {
    "support_lookback_sessions": (10, 63, int),
    "confirmation_sessions": (2, 5, int),
    "min_pullback_atr": (1, 6, float),
    "max_rebound_atr": (1, 4, float),
    "watch_sessions": (1, 10, int),
}
METHOD = {
    "minimum_history_sessions": 260,
    "atr": "simple mean of last 14 true ranges; previous close from last 15 rows",
    "prior_high_sessions": 63,
    "pullback": "prior 63-session highest high minus current close, divided by causal ATR14",
    "anchor": "most recent equal minimum low in declared support lookback",
    "confirmation": "anchor age >= N; all later lows >= anchor; last N closes > anchor",
    "last_close": "strictly above prior completed close",
    "raw_zone": "anchor low to anchor low + 0.5 ATR; center anchor + 0.25 ATR",
    "execution_zone": "simulator replaces total width with candidate zone_width_atr",
    "origin_clock": "completed US equity session at 16:15 America/New_York",
    "event_policy": "auxiliary event inputs absent; never veto price setup generation",
    "denominator": "daily asset-session dispositions, distinct from weekly origin controls",
}


def _clock(day):
    return datetime.combine(day, time(16, 15), NY).isoformat()


def _advance(day, count):
    for _ in range(count):
        day += timedelta(days=1)
        while not is_us_equity_session(day):
            day += timedelta(days=1)
    return day


def _validate(policy, candidate):
    if not isinstance(policy, dict) or set(policy) != set(POLICY_BOUNDS):
        raise ValueError("Price-action policy requires exactly the five declared keys.")
    for key, (low, high, kind) in POLICY_BOUNDS.items():
        value = policy[key]
        if (type(value) not in ((int,) if kind is int else (int, float))
                or _number(value) is None or not low <= value <= high):
            raise ValueError(f"Invalid price-action policy {key}.")
    try:
        content = {key: candidate[key] for key in (
            "parameters", "structure_parameters", "event_variant")}
        parameters = validate_candidate_parameters(candidate["parameters"])
        validate_structure_parameters(candidate["structure_parameters"])
        if (set(parameters) != CORE_KEYS | STRATEGY_KEYS
                or _hash(content) != candidate["parameters_sha256"]
                or not isinstance(candidate["id"], str) or not candidate["id"]
                or candidate["event_variant"] not in VARIANTS):
            raise ValueError("A bound frozen version-two candidate is required.")
        if policy["watch_sessions"] != parameters["watch_sessions"]:
            raise ValueError("Policy and frozen candidate watch sessions must match.")
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed frozen price-action candidate.") from exc


def _history(raw, day):
    if len(raw) < 260:
        return None, "insufficient_predecision_history"
    bases = [row.get("price_basis") for row in raw]
    if (any(not isinstance(basis, str) or not basis.strip()
            or basis.strip().lower() in {"none", "n/a", "na", "unverified"}
            or "unknown" in basis.lower() or "未注明" in basis for basis in bases)
            or len(set(bases)) != 1
            or any(row.get("session_complete") is not True for row in raw)):
        return None, "unverified_or_inconsistent_price_basis_or_sessions"
    clean = _clean(raw, day.isoformat())
    if len(clean) != len(raw):
        return None, "invalid_or_duplicate_predecision_price_observation"
    if (clean[-1]["date"] != day.isoformat() or any(
            _advance(date.fromisoformat(left["date"]), 1).isoformat() != right["date"]
            for left, right in zip(clean, clean[1:]))):
        return None, "missing_completed_session_in_predecision_history"
    return clean, None


def _bottom(history, policy):
    atr, close = _atr(history, 14), history[-1]["close"]
    if atr is None or atr <= 0:
        return None, "nonpositive_causal_atr"
    first = len(history) - policy["support_lookback_sessions"]
    # The most recent tie restarts the clock instead of inheriting an older low's age.
    anchor_index = min(range(first, len(history)), key=lambda i: (history[i]["low"], -i))
    anchor = history[anchor_index]
    later, confirmations = history[anchor_index + 1:], history[-policy["confirmation_sessions"]:]
    prior = history[-64:-1]
    peak = max(prior, key=lambda row: (row["high"], row["date"]))
    conditions = {
        "anchor_old_enough": len(later) >= policy["confirmation_sessions"],
        "subsequent_lows_hold": all(row["low"] >= anchor["low"] for row in later),
        "confirmation_closes_above_anchor": all(row["close"] > anchor["low"]
                                                for row in confirmations),
        "final_close_rising": close > history[-2]["close"],
        "sufficient_pullback": (peak["high"] - close) / atr >= policy["min_pullback_atr"],
        "rebound_not_extended": (close - anchor["low"]) / atr <= policy["max_rebound_atr"],
    }
    return {
        "anchor_date": anchor["date"], "anchor_low": anchor["low"],
        "anchor_at": _clock(date.fromisoformat(anchor["date"])),
        "anchor_age_sessions": len(later), "atr14": atr, "close": close,
        "prior_high": peak["high"], "prior_high_date": peak["date"],
        "pullback_atr": (peak["high"] - close) / atr,
        "rebound_atr": (close - anchor["low"]) / atr,
        "confirmation_clocks": [_clock(date.fromisoformat(row["date"]))
                                for row in confirmations],
        "conditions": conditions,
    }, next((key for key, passed in conditions.items() if not passed), None)


def generate_price_action_setups(market, start, end, candidate, *, policy):
    """Return daily dispositions and once-per-anchor setups, using only origin prefixes.

    ``start``/``end`` bound completed decision dates, not a future outcome window.
    The input can contain later rows; their numerical fields are never inspected.
    Histories must explicitly declare completion and one continuous price basis.
    Anchor deduplication is local to this call; no persistent campaign state is
    implied. Generate a full span before partitioning if cross-block dedup is needed.
    Event variants remain bound to the candidate; use ``price_only`` to execute
    without the simulator's separate event-model gate.
    """
    _validate(policy, candidate)
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or not isinstance(market, dict) or any(
            not isinstance(asset, str) or not asset for asset in market):
        raise ValueError("Invalid daily setup dates or asset inventory.")
    policy = copy.deepcopy(policy)
    code = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("price_action_setups.py", "structure.py", "sessions.py",
                         "swing_optimizer.py")}
    method_hash, policy_hash = _hash(METHOD), _hash(policy)
    dispositions, setups, latest, emitted = [], [], {}, {}
    for asset in sorted(market):
        rows = market[asset]
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Price histories require explicit dated row lists.")
        # Index dates alone; do not normalize, hash or inspect future numeric fields.
        indexed = []
        for row in rows:
            try:
                day = date.fromisoformat(row.get("date"))
            except (TypeError, ValueError) as exc:
                raise ValueError("Price row requires an ISO session date.") from exc
            indexed.append((day, row))
        indexed.sort(key=lambda item: item[0])
        dates, ordered = [item[0] for item in indexed], [item[1] for item in indexed]
        day = first
        while day <= last:
            if not is_us_equity_session(day):
                day += timedelta(days=1)
                continue
            origin = {
                "opportunity_id": f"{asset}|{day.isoformat()}", "asset": asset,
                "origin_at": _clock(day),
                "watch_expires_at": _clock(_advance(day, policy["watch_sessions"])),
                "comparison_available_at": _clock(_advance(
                    day, policy["watch_sessions"]
                    + candidate["parameters"]["holding_sessions"] + 1)),
                "author_explicit_deadline": None,
            }
            decision = {**origin, "status": "unavailable", "reason": None,
                        "policy_sha256": policy_hash, "method_sha256": method_hash}
            raw = ordered[:bisect.bisect_right(dates, day)]
            history, reason = _history(raw, day)
            if history is not None:
                diagnostic, reason = _bottom(history, policy)
                try:
                    source_hash = _hash(raw)
                except (ValueError, TypeError):
                    diagnostic, reason = None, "noncanonical_predecision_source_record"
                if diagnostic is not None:
                    decision.update(diagnostic=diagnostic, source_history_sha256=source_hash)
                    identity = (asset, diagnostic["anchor_date"])
                    if reason is None and identity in emitted:
                        decision.update(status="already_emitted",
                                        reason="same_anchor_already_emitted",
                                        first_emitted_at=emitted[identity])
                    elif reason is None:
                        generation = {
                            "origin": origin, "policy": policy, "method": METHOD,
                            "candidate_sha256": candidate["parameters_sha256"],
                            "source_history_sha256": source_hash, "source_code_sha256": code,
                            "diagnostic": diagnostic,
                        }
                        atr, low = diagnostic["atr14"], diagnostic["anchor_low"]
                        setup = {
                            **origin, "available_at": origin["origin_at"],
                            "features_available_at": origin["origin_at"],
                            "candidate_id": candidate["id"],
                            "parameters_sha256": candidate["parameters_sha256"],
                            "zone_lower": low, "zone_upper": low + .5 * atr,
                            "zone_center": low + .25 * atr, "atr": atr,
                            "price_basis": history[-1]["price_basis"],
                            "data_source": history[-1]["source"],
                            "source_history_sha256": source_hash,
                            "source_code_sha256": code, "policy_sha256": policy_hash,
                            "generation_parameters": policy, "method_sha256": method_hash,
                            "upstream_generation_hash": _hash(generation),
                            "causal_bottom": diagnostic,
                            "hypothesis_tier": "confirmed_daily_bottom",
                            "event_context": None,
                            "historical_point_in_time_verified": False,
                            "author_model_reconstructed": False,
                        }
                        setups.append(setup)
                        emitted[identity] = origin["origin_at"]
                        decision.update(status="ready", reason=None,
                                        upstream_generation_hash=setup["upstream_generation_hash"])
                    else:
                        decision.update(status="nonqualifying", reason=reason)
            if decision["reason"] is None and decision["status"] == "unavailable":
                decision["reason"] = reason
            dispositions.append(decision)
            latest[asset] = decision
            day += timedelta(days=1)
    return copy.deepcopy({
        "schema_version": VERSION, "start": start, "end": end,
        "policy": policy, "policy_sha256": policy_hash, "method": METHOD,
        "method_sha256": method_hash, "source_code_sha256": code,
        "candidate_id": candidate["id"], "parameters_sha256": candidate["parameters_sha256"],
        "execution_parameters": {key: value for key, value in candidate["parameters"].items()
                                 if key != "lookback_sessions"},
        "opportunity_count": len(dispositions), "ready_count": len(setups),
        "setups": setups, "dispositions": dispositions, "latest_decision": latest,
        "goal_completion_proven": False,
    })
