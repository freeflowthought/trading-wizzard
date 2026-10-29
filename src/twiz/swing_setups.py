"""Generate dated price-region hypotheses without reading subsequent outcomes.

The structure rules are an explicit Shunge/Eagle-inspired proxy. They do not
reconstruct undisclosed author indicators, trades or the claimed five-year score.
"""

import bisect
import copy
import hashlib
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .sessions import is_us_equity_session
from .structure import _clean, _one_structure, validate_structure_parameters
from .swing_optimizer import _hash, _time

NY = ZoneInfo("America/New_York")


def _advance(day, sessions):
    for _ in range(sessions):
        day += timedelta(days=1)
        while not is_us_equity_session(day):
            day += timedelta(days=1)
    return day


def _close(day):
    return datetime.combine(day, time(16, 15), NY).isoformat()


def freeze_weekly_origins(start, end, *, assets=("QQQ", "SPY"), watch_sessions=20,
                          maximum_holding_sessions=40):
    """Freeze every scheduled opportunity, including ones later missing data.

    One origin per asset at the last scheduled US session of each whole week.
    Watch expiry and common maturity are independent of candidate profit/exit.
    Calendar emergency closures discovered later remain a source limitation.
    """
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if (first > last or not assets or len(set(assets)) != len(assets)
            or any(not isinstance(asset, str) or not asset for asset in assets)
            or type(watch_sessions) is not int or not 1 <= watch_sessions <= 252
            or type(maximum_holding_sessions) is not int
            or not 1 <= maximum_holding_sessions <= 252):
        raise ValueError("Invalid swing opportunity schedule.")
    friday = first+timedelta(days=(4-first.weekday()) % 7)
    origins = []
    while friday <= last+timedelta(days=6):
        origin = friday
        while not is_us_equity_session(origin):
            origin -= timedelta(days=1)
        if first <= origin <= last:
            expiry = _advance(origin, watch_sessions)
            mature = _advance(expiry, maximum_holding_sessions+1)
            for asset in assets:
                origins.append({"opportunity_id": f"{asset}|{origin.isoformat()}",
                                "asset": asset, "origin_at": _close(origin),
                                "watch_expires_at": _close(expiry),
                                "comparison_available_at": _close(mature),
                                "author_explicit_deadline": None})
        friday += timedelta(days=7)
    return {"schema_version": "weekly-swing-origins-1", "start": start, "end": end,
            "assets": list(assets), "watch_sessions": watch_sessions,
            "maximum_holding_sessions": maximum_holding_sessions,
            "origins": origins, "sha256": _hash(origins)}


def generate_swing_setups(market_data, origins, candidate, *, tier="support",
                           event_context_by_opportunity=None, minimum_history_sessions=260):
    """Return a disposition for every origin and ready setups for the simulator.

    ``lookback_sessions`` limits confirmed pivot history. Long moving averages
    and weekly context still use preceding history; a 63-session pivot window
    must not turn a 200-day average into a fabricated 63-day average.
    ``tier`` is fixed by the study, never chosen after seeing future extrema.
    Deep/extreme hypotheses remain separate conditional studies, not guaranteed
    floors. No absent event probability is replaced by a default estimate.
    """
    if tier not in {"support", "deep", "extreme"}:
        raise ValueError("Study must predeclare a valid structure tier.")
    if type(minimum_history_sessions) is not int or minimum_history_sessions < 260:
        raise ValueError("Daily/weekly structure needs at least 260 historical sessions.")
    content = {key: candidate[key] for key in (
        "parameters", "structure_parameters", "event_variant")}
    if _hash(content) != candidate["parameters_sha256"]:
        raise ValueError("Candidate parameters changed after registry freeze.")
    structure_params = validate_structure_parameters(candidate["structure_parameters"])
    lookback = candidate["parameters"]["lookback_sessions"]
    if type(lookback) is not int or not 20 <= lookback <= 2520:
        raise ValueError("Invalid pivot history lookback.")
    policy = {"schema": "dated-structure-setups-1", "tier": tier,
              "minimum_history_sessions": minimum_history_sessions,
              "pivot_lookback_sessions": lookback,
              "structure_parameters": structure_params}
    source_directory = Path(__file__).parent
    upstream_code = {name: hashlib.sha256((source_directory/name).read_bytes()).hexdigest()
                     for name in ("swing_setups.py", "structure.py", "sessions.py")}
    indexed = {}
    for asset, rows in market_data.items():
        valid = [row for row in rows if isinstance(row, dict)
                 and isinstance(row.get("date"), str)]
        valid.sort(key=lambda row: row["date"])
        indexed[asset] = ([row["date"] for row in valid], valid)
    dispositions, setups, seen = [], [], set()
    event_inputs = event_context_by_opportunity or {}
    for origin in origins:
        identity, asset = origin.get("opportunity_id"), origin.get("asset")
        decision = _time(origin.get("origin_at"))
        expiry = _time(origin.get("watch_expires_at"))
        common = _time(origin.get("comparison_available_at"))
        if (not isinstance(identity, str) or not identity or identity in seen or not asset
                or not decision or not expiry or not common or not decision < expiry < common):
            raise ValueError("Malformed, duplicate or nonchronological opportunity origin.")
        local = decision.astimezone(NY)
        if not is_us_equity_session(local.date()) or local.time() < time(16, 15):
            raise ValueError("Region decisions require a completed US regular session.")
        seen.add(identity)
        day = local.date().isoformat()
        dates, all_rows = indexed.get(asset, ([], []))
        history = all_rows[:bisect.bisect_right(dates, day)]
        disposition = {"opportunity_id": identity, "asset": asset,
                       "origin_at": origin["origin_at"], "status": "unavailable",
                       "comparison_available_at": origin["comparison_available_at"],
                       "watch_expires_at": origin["watch_expires_at"],
                       "reason": None}
        if len(history) < minimum_history_sessions:
            disposition["reason"] = "insufficient_predecision_history"
            dispositions.append(disposition)
            continue
        bases = [row.get("price_basis") for row in history]
        if (any(not isinstance(basis, str) or not basis.strip()
                or basis.strip().lower() in {"none", "n/a", "na", "unverified"}
                or "未注明" in basis or "unknown" in basis.lower() for basis in bases)
                or len(set(bases)) != 1
                or any(row.get("session_complete") is not True for row in history)
                or len({row["date"] for row in history}) != len(history)):
            disposition["reason"] = "unverified_or_inconsistent_price_basis_or_sessions"
            dispositions.append(disposition)
            continue
        if len(_clean(history, day)) != len(history):
            disposition["reason"] = "invalid_predecision_price_observation"
            dispositions.append(disposition)
            continue
        try:
            source_hash = _hash(history)
        except (ValueError, TypeError):
            disposition["reason"] = "noncanonical_source_record"
            dispositions.append(disposition)
            continue
        context = copy.deepcopy(event_inputs.get(identity))
        if context is not None:
            clocks = ("available_at", "features_available_at", "model_as_of", "model_fit_at")
            invalid = (not isinstance(context, dict) or not _time(context.get("available_at")))
            if not invalid:
                invalid = any(key in context and (
                    not _time(context[key]) or _time(context[key]) > decision) for key in clocks)
            try:
                json.dumps(context, allow_nan=False)
            except (ValueError, TypeError):
                invalid = True
            if invalid:
                disposition["reason"] = "unavailable_or_unverifiable_event_context"
                dispositions.append(disposition)
                continue
        structure = _one_structure(asset, history, day, structure_params, lookback)
        zone = structure.get("zones", {}).get(tier)
        if structure.get("status") != "ready" or not zone or not zone.get("credible"):
            disposition["reason"] = "no_credible_predeclared_region"
            dispositions.append(disposition)
            continue
        generation = {"policy": policy, "candidate_hash": candidate["parameters_sha256"],
                      "origin": origin, "used_history_sha256": source_hash,
                      "upstream_source_sha256": upstream_code, "structure": structure,
                      "dated_event_context": context}
        setup = {**origin, "available_at": origin["origin_at"],
                 "candidate_id": candidate["id"],
                 "parameters_sha256": candidate["parameters_sha256"],
                 "features_available_at": origin["origin_at"],
                 "zone_lower": zone["lower"], "zone_upper": zone["upper"],
                 "atr": structure["features"]["decision_atr"],
                 "price_basis": structure["features"]["price_basis"],
                 "upstream_generation_hash": _hash(generation),
                 "source_history_sha256": source_hash, "data_source": structure["data_source"],
                 "hypothesis_tier": tier, "reference_zones": structure["zones"],
                 "generation_parameters": policy,
                 "event_context": context,
                 "historical_point_in_time_verified": False,
                 "author_model_reconstructed": False}
        setups.append(setup)
        disposition.update(status="ready", reason=None,
                           upstream_generation_hash=setup["upstream_generation_hash"])
        dispositions.append(disposition)
    return {"setups": setups, "dispositions": dispositions, "policy": policy,
            "candidate_id": candidate["id"], "parameters_sha256": candidate["parameters_sha256"],
            "execution_parameters": {key: value for key, value in candidate["parameters"].items()
                                     if key != "lookback_sessions"},
            "opportunity_count": len(origins), "ready_count": len(setups),
            "goal_completion_proven": False}
