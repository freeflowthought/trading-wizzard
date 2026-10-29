"""Freeze bounded swing searches and select only on mature paired research records.

This module does not fetch data or rebuild fills. Supplied campaign PnL remains
explicitly unverified until a separate price-based simulator validates it.
"""

import hashlib
import json
import math
import random
from copy import deepcopy
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict
from zoneinfo import ZoneInfo

from .sessions import is_us_equity_session
from .structure import PARAMETERS, validate_structure_parameters

CORE_KEYS = {"event_threshold", "main_case_only", "zone_width_atr", "confirm_sessions",
             "stop_atr", "take_profit_r", "holding_sessions", "lookback_sessions",
             "entry_fractions", "trim_fractions", "trade_probability_threshold"}
VARIANTS = ("price_only", "signed_weighted_event_score", "learned_event_bridge")
DEFAULTS = {"event_threshold": .65, "main_case_only": False, "zone_width_atr": .5,
            "trade_probability_threshold": .65,
            "confirm_sessions": 2, "stop_atr": 1.0, "take_profit_r": 2.0,
            "holding_sessions": 20, "lookback_sessions": 126,
            "entry_fractions": [.4, .35, .25], "trim_fractions": [.4, .3, .3]}
STRATEGY_DEFAULTS = {"entry_rule": "retest", "entry_ladder": "zone_edges",
                     "watch_sessions": 20, "max_entry_extension_atr": 1.0}
STRATEGY_KEYS = set(STRATEGY_DEFAULTS)
EXECUTION = {"max_new_capital_pct": 10.0, "max_campaign_loss_pct": .5,
             "max_concurrent_campaigns": 1, "fee_bps_per_side": 1.0,
             "slippage_bps_per_side": 2.0,
             "execution": "daily_close_decision_next_session_open_fill",
             "funding_order": "ready_time_then_source_time_then_asset_then_candidate_id",
             "pending_watch_reserves_capital": False,
             "cancel_remaining_entries_after_first_trim": True,
             "cost_stress_multipliers": [1.0, 2.0, 3.0]}
FINAL_STATES = {"closed", "unfilled", "expired", "blocked", "unavailable", "cancelled"}


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _time(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp.astimezone(timezone.utc) if stamp.tzinfo else None
    except (TypeError, ValueError, OverflowError):
        return None


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _validate_parameter(key, value):
    if key == "entry_rule":
        valid = isinstance(value, str) and value in {"retest", "close_hold", "sweep_reclaim"}
    elif key == "entry_ladder":
        valid = value in {"zone_edges", "atr_approach"} if isinstance(value, str) else False
    elif key == "watch_sessions":
        valid = type(value) is int and 1 <= value <= 20
    elif key == "main_case_only":
        valid = type(value) is bool
    elif key in {"entry_fractions", "trim_fractions"}:
        valid = (isinstance(value, (list, tuple)) and len(value) == 3
                 and all(type(x) in (int, float) and _number(x) is not None and 0 < x <= 1
                         for x in value) and abs(sum(value)-1) <= 1e-9)
    elif key in {"confirm_sessions", "holding_sessions", "lookback_sessions"}:
        upper = {"confirm_sessions": 20, "holding_sessions": 252, "lookback_sessions": 2520}[key]
        lower = 20 if key == "lookback_sessions" else 1
        valid = type(value) is int and lower <= value <= upper
    else:
        probability = key in {"event_threshold", "trade_probability_threshold"}
        upper = 1 if probability else 20 if key == "take_profit_r" else 10
        lower = .5 if probability else 0
        valid = (type(value) in (int, float) and _number(value) is not None
                 and lower < value <= upper)
    if not valid:
        raise ValueError(f"Invalid swing parameter {key}: {value!r}")
    return list(value) if key in {"entry_fractions", "trim_fractions"} else value


def validate_candidate_parameters(parameters):
    """Legacy eleven knobs or the complete version-two strategy schema; never a partial hybrid."""
    if not isinstance(parameters, dict) or set(parameters) not in (
            CORE_KEYS, CORE_KEYS | STRATEGY_KEYS):
        raise ValueError(
            "Candidate requires exactly legacy parameters or all four V2 strategy knobs."
        )
    return {key: _validate_parameter(key, value) for key, value in parameters.items()}


def freeze_candidate_search(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deterministic bounded candidate registry, including default and ablations.

    Core knobs live in ``parameters``; all structure knobs in
    ``structure_parameters``. Unsupported declared search fields return no
    candidates, rather than being silently ignored. Invalid supported values raise.
    """
    defaults = config.get("defaults", {})
    version = config.get("protocol_version", "author-swing-v1")
    if version not in {"author-swing-v1", "author-swing-v2"}:
        raise ValueError("Unsupported swing strategy protocol version.")
    supported = CORE_KEYS | STRATEGY_KEYS if version == "author-swing-v2" else CORE_KEYS
    protocol_defaults = (
        {**DEFAULTS, **STRATEGY_DEFAULTS} if version == "author-swing-v2" else DEFAULTS
    )
    core_space = config.get("search_space", {})
    structure_space = config.get("structure_search_space", {})
    direction = config.get("event_direction", {})
    fixed = {**EXECUTION, **config.get("fixed_constraints", {})}
    unsupported = (["defaults."+key for key in set(defaults)-supported-{"event_variant"}]
                   + ["search_space."+key for key in set(core_space)-supported]
                   + ["structure_search_space."+key for key in set(structure_space)-set(PARAMETERS)]
                   + ["fixed_constraints."+key for key in set(fixed)-set(EXECUTION)])
    if unsupported:
        return {"status": "unsupported_fields", "candidates": [], "candidate_sha256": None,
                "unsupported_declared_fields": sorted(unsupported), "default_id": None}
    if fixed != EXECUTION:
        raise ValueError("The shared 10%/.5% close-to-next-open execution and costs are fixed.")
    core = {key: _validate_parameter(key, defaults.get(key, value))
            for key, value in protocol_defaults.items()}
    variant = defaults.get("event_variant", "learned_event_bridge")
    variants = direction.get("variants", list(VARIANTS))
    if (variant not in VARIANTS or not isinstance(variants, list) or not variants
            or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants)
            or variant not in variants):
        raise ValueError("Invalid event variant registry or default variant.")
    axes = []
    for key in sorted(supported):
        values = core_space.get(key, [core[key]])
        if not isinstance(values, list) or not values:
            raise ValueError(f"Empty or invalid search axis {key}.")
        normalized = [_validate_parameter(key, value) for value in values]
        if (core[key] not in normalized
                or len({_hash(value) for value in normalized}) != len(normalized)):
            raise ValueError(f"Search axis {key} must include default without duplicates.")
        axes.append(("core", key, normalized))
    for key in sorted(structure_space):
        values = structure_space[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Empty or invalid structure axis {key}.")
        for value in values:
            validate_structure_parameters({key: value})
        if PARAMETERS[key] not in values or len({_hash(v) for v in values}) != len(values):
            raise ValueError(f"Structure axis {key} must include default without duplicates.")
        axes.append(("structure", key, values))
    selection = config.get("selection", {})
    maximum, seed = selection.get("maximum_candidates", 96), selection.get("seed", 20261028)
    if type(maximum) is not int or not 1 <= maximum <= 96 or type(seed) is not int:
        raise ValueError("Candidate count must be at most 96 and seed must be an integer.")
    candidates, seen = [], set()

    def add(parameters, structure, event_variant, default=False):
        content = {"parameters": parameters, "structure_parameters": structure,
                   "event_variant": event_variant}
        digest = _hash(content)
        if digest not in seen and len(candidates) < maximum:
            candidates.append({"id": "swing-"+digest[:20], **content,
                               "parameters_sha256": digest, "is_default": default})
            seen.add(digest)

    add(core, dict(PARAMETERS), variant, True)
    for other in variants:
        add(dict(core), dict(PARAMETERS), other)
    if version == "author-swing-v2":
        # Matched one-factor strategy controls precede joint sampling. Keep V1 bytes stable.
        for family, key, values in axes:
            if family == "core" and key in STRATEGY_KEYS:
                for value in values:
                    if value != core[key]:
                        for other in variants:
                            add({**core, key: value}, dict(PARAMETERS), other)
    rng = random.Random(seed)
    combinations = len(variants)
    for _, _, values in axes:
        combinations *= len(values)
    for _ in range(maximum*100):
        if len(candidates) >= min(maximum, combinations):
            break
        chosen, structure = dict(core), dict(PARAMETERS)
        for family, key, values in axes:
            (chosen if family == "core" else structure)[key] = rng.choice(values)
        add(chosen, structure, rng.choice(variants))
    return {"status": "frozen", "candidates": candidates, "candidate_sha256": _hash(candidates),
            "unsupported_declared_fields": [], "default_id": candidates[0]["id"], "seed": seed,
            "selection_policy": dict(selection), "execution_policy": deepcopy(EXECUTION),
            "execution_policy_id": _hash(EXECUTION),
            "simulator_status": "not_implemented_external_evaluations_required",
            "author_deadlines_optimized": False}


def _embargo_at(stamp, sessions):
    local = stamp.astimezone(ZoneInfo("America/New_York"))
    for _ in range(sessions):
        local += timedelta(days=1)
        while not is_us_equity_session(local.date()):
            local += timedelta(days=1)
    return local.astimezone(timezone.utc)


def _next_open(decision):
    if decision is None:
        return None
    local = decision.astimezone(ZoneInfo("America/New_York"))
    if not is_us_equity_session(local.date()) or local.time() < time(16):
        return None
    next_day = local.date()+timedelta(days=1)
    while not is_us_equity_session(next_day):
        next_day += timedelta(days=1)
    return datetime.combine(next_day, time(9, 30), tzinfo=ZoneInfo("America/New_York"))


def freeze_opportunity_universe(block_metadata):
    """Freeze the comparison denominator and common maturity BEFORE joining trade outcomes.

    Each block supplies block_id/partition/block_start/block_end/outcome_available_at
    and opportunities [{opportunity_id, origin_at, outcome_available_at}]. Common
    maturity must include the longest candidate holding window and required fills.
    """
    blocks, identities, opportunity_ids = [], set(), set()
    for block in block_metadata:
        identity, partition = block.get("block_id"), block.get("partition")
        start, end, available = (_time(block.get(key)) for key in (
            "block_start", "block_end", "outcome_available_at"))
        if (not isinstance(identity, str) or not identity or partition not in {
                "train", "validation", "test"} or (partition, identity) in identities
                or not start or not end or not available or not start <= end <= available):
            raise ValueError("Invalid or duplicate frozen opportunity block.")
        identities.add((partition, identity))
        opportunities = []
        for row in block.get("opportunities", []):
            key = row.get("opportunity_id")
            origin, mature = _time(row.get("origin_at")), _time(row.get("outcome_available_at"))
            if (not isinstance(key, str) or not key or key in opportunity_ids
                    or not origin or not mature or not start <= origin <= end
                    or not origin < mature <= available):
                raise ValueError("Invalid, duplicate or unmatured opportunity metadata.")
            opportunity_ids.add(key)
            opportunities.append({"opportunity_id": key, "origin_at": origin.isoformat(),
                                  "outcome_available_at": mature.isoformat()})
        if not opportunities:
            raise ValueError("Frozen opportunity blocks cannot be empty.")
        blocks.append({"block_id": identity, "partition": partition,
                       "block_start": start.isoformat(), "block_end": end.isoformat(),
                       "outcome_available_at": available.isoformat(),
                       "opportunities": sorted(opportunities,
                                               key=lambda row: row["opportunity_id"])})
    blocks.sort(key=lambda row: (row["block_start"], row["partition"], row["block_id"]))
    return {"blocks": blocks, "sha256": _hash(blocks)}


def _validate_block(block, candidate, execution_id):
    if block.get("execution_policy_id") != execution_id:
        raise ValueError("different_common_execution")
    if candidate and block.get("parameters_sha256") != candidate["parameters_sha256"]:
        raise ValueError("candidate_parameters_hash_mismatch")
    start, end, available = (_time(block.get(key)) for key in (
        "block_start", "block_end", "outcome_available_at"))
    if not all((start, end, available)) or not start <= end <= available:
        raise ValueError("invalid_block_chronology")
    campaigns = block.get("campaigns")
    if not isinstance(campaigns, list) or not campaigns:
        raise ValueError("missing_campaign_opportunity_panel")
    identities, intervals = set(), []
    for row in campaigns:
        identity, status = row.get("opportunity_id"), row.get("status")
        origin, mature = _time(row.get("origin_at")), _time(row.get("outcome_available_at"))
        if (not isinstance(identity, str) or not identity or identity in identities
                or not origin or not mature or not start <= origin <= end
                or not origin < mature <= available):
            raise ValueError("invalid_or_duplicate_campaign_chronology")
        identities.add(identity)
        if status not in FINAL_STATES:
            raise ValueError("pending_or_open_campaign_not_mature")
        pnl, capital, risk = (_number(row.get(key)) for key in (
            "net_account_pnl_pct", "new_capital_pct", "modeled_loss_pct"))
        if (pnl is None or capital is None or risk is None or not 0 <= capital <= 10
                or not 0 <= risk <= .5):
            raise ValueError("missing_or_unsafe_capital_risk_or_pnl")
        entered, exited = _time(row.get("entered_at")), _time(row.get("exited_at"))
        if status == "closed":
            entry_signal, exit_signal = (_time(row.get(key)) for key in (
                "entry_signal_at", "exit_signal_at"))
            if (not entered or not exited or not origin <= entered < exited <= mature
                    or exited > end or capital <= 0):
                raise ValueError("invalid_filled_trade_times")
            if (entry_signal is None or exit_signal is None or not origin <= entry_signal < entered
                    or not entered <= exit_signal < exited
                    or _next_open(entry_signal) != entered or _next_open(exit_signal) != exited):
                raise ValueError("filled_trade_not_close_decision_next_session_open")
            intervals.append((entered, exited, identity))
        elif entered or exited or capital != 0 or pnl != 0:
            raise ValueError("unfilled_campaign_claims_funding_or_pnl")
    intervals.sort()
    if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("overlapping_funded_campaigns")
    marks = block.get("equity_curve")
    if not isinstance(marks, list) or len(marks) < 2:
        raise ValueError("missing_daily_nav_curve")
    curve = []
    for point in marks:
        stamp, equity = _time(point.get("at")), _number(point.get("equity"))
        if not stamp or equity is None or equity <= 0 or not start <= stamp <= end:
            raise ValueError("invalid_nav_point")
        curve.append((stamp, equity))
    if (any(b[0] <= a[0] for a, b in zip(curve, curve[1:])) or abs(curve[0][1]-100) > 1e-8
            or curve[0][0] != start or curve[-1][0] != end):
        raise ValueError("unordered_nav_or_nonstandard_initial_account")
    local_start, local_end = (stamp.astimezone(ZoneInfo("America/New_York"))
                              for stamp in (start, end))
    needed, day = set(), local_start.date()
    while day <= local_end.date():
        close = datetime.combine(day, time(16, 15), tzinfo=ZoneInfo("America/New_York"))
        if is_us_equity_session(day) and start <= close <= end:
            needed.add(day)
        day += timedelta(days=1)
    observed = {stamp.astimezone(ZoneInfo("America/New_York")).date() for stamp, _ in curve
                if stamp.astimezone(ZoneInfo("America/New_York")).time() == time(16, 15)}
    if not needed <= observed:
        raise ValueError("missing_common_daily_nav_close_grid")
    pnl = sum(float(row["net_account_pnl_pct"]) for row in campaigns)
    if abs(curve[-1][1]-100-pnl) > 1e-6:
        raise ValueError("nav_does_not_reconcile_with_campaign_pnl")
    return {**block, "start": start, "end": end, "available": available,
            "curve": curve, "opportunities": identities, "intervals": intervals}


def _metrics(blocks):
    blocks = sorted(blocks, key=lambda block: block["start"])
    all_campaigns, wealth, peak, drawdown = [], 100.0, 100.0, 0.0
    intervals = []
    for block in blocks:
        all_campaigns.extend(block["campaigns"])
        intervals.extend(block["intervals"])
        opening = wealth
        for _, equity in block["curve"]:
            wealth = opening*equity/100
            if not math.isfinite(wealth) or wealth <= 0:
                raise ValueError("nonfinite_or_nonpositive_aggregate_nav")
            peak = max(peak, wealth)
            drawdown = max(drawdown, (peak-wealth)/peak*100)
    identities = [row["opportunity_id"] for row in all_campaigns]
    if len(identities) != len(set(identities)):
        raise ValueError("opportunity_reused_across_blocks")
    intervals.sort()
    if any(b[0] < a[1] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("overlapping_campaigns_across_blocks")
    closed = [row for row in all_campaigns if row["status"] == "closed"]
    wins = sum(float(row["net_account_pnl_pct"]) > 0 for row in closed)
    return {"blocks": len(blocks), "opportunities": len(all_campaigns),
            "closed_campaigns": len(closed),
            "wins": wins, "filled_trade_win_rate": wins/len(closed) if closed else None,
            "opportunity_success": wins/len(all_campaigns) if all_campaigns else None,
            "entry_coverage": len(closed)/len(all_campaigns) if all_campaigns else None,
            "net_account_return_pct": wealth-100, "max_drawdown_pct": drawdown,
            "status_counts": {state: sum(row["status"] == state for row in all_campaigns)
                              for state in sorted(FINAL_STATES)}}


def select_swing_parameters(candidates, evaluations, *, selection_at, min_closed_campaigns=20):
    """Select supplied paired campaign evaluations; this function does NOT reconstruct trades.

    ``evaluations`` is a list of blocks with candidate_id (or ``baseline``),
    partition=train/validation/test, block_id/start/end/outcome_available_at,
    execution_policy_id, parameters_sha256, campaigns and an ordered daily NAV
    equity_curve starting at100. Campaigns contain opportunity_id, origin_at,
    outcome_available_at, status, net_account_pnl_pct, new_capital_pct,
    modeled_loss_pct and, for closed trades, entry_signal_at/entered_at and
    exit_signal_at/exited_at. NAV must include each completed session's16:15NY mark.
    Test records are ignored before reading any campaign outcomes.
    """
    cutoff = _time(selection_at)
    if not cutoff or type(min_closed_campaigns) is not int or min_closed_campaigns < 1:
        raise ValueError("Selection needs an aware cutoff and positive minimum sample count.")
    frozen = candidates if isinstance(candidates, dict) else {"candidates": candidates}
    rows = frozen.get("candidates", [])
    registry = {row["id"]: row for row in rows}
    defaults = [row["id"] for row in registry.values() if row.get("is_default")]
    if not registry or len(defaults) != 1 or len(registry) != len(rows):
        raise ValueError("A frozen registry must have one unique default candidate.")
    if frozen.get("candidate_sha256", _hash(rows)) != _hash(rows):
        raise ValueError("Frozen candidate list hash mismatch.")
    for row in registry.values():
        validate_candidate_parameters(row["parameters"])
        content = {key: row[key] for key in ("parameters", "structure_parameters", "event_variant")}
        if _hash(content) != row["parameters_sha256"]:
            raise ValueError("Candidate registry parameter hash mismatch.")
    execution_id = frozen.get("execution_policy_id", _hash(EXECUTION))
    if execution_id != _hash(EXECUTION):
        raise ValueError("The fixed execution policy hash differs.")
    policy = frozen.get("selection_policy", {})
    embargo, required_blocks = policy.get("embargo_sessions", 5), policy.get("validation_blocks", 3)
    if (type(embargo) is not int or embargo < 0
            or type(required_blocks) is not int or required_blocks < 1):
        raise ValueError("Invalid embargo or validation-block requirements.")
    universe = frozen.get("opportunity_universe", {})
    if not universe.get("blocks") or universe.get("sha256") != _hash(universe["blocks"]):
        return {"status": "insufficient", "selected_id": defaults[0],
                "selected": registry[defaults[0]], "default_id": defaults[0],
                "reason": "missing_or_unsealed_opportunity_universe",
                "goal_completion_proven": False, "parameter_application_verified": False,
                "evaluation_basis": "supplied_campaign_records_until_simulator_validates",
                "simulator_status": "not_implemented_external_evaluations_required",
                "production_eligible": False, "author_comparison_proven": False}
    if freeze_opportunity_universe(universe["blocks"]) != universe:
        raise ValueError("Opportunity universe is not in canonical validated form.")
    expected = {(row["partition"], row["block_id"]): row for row in universe["blocks"]
                if row["partition"] in {"train", "validation"}}
    parsed, excluded, seen_blocks, tainted_blocks = {}, [], set(), set()
    for block in evaluations:
        if not isinstance(block, dict):
            excluded.append({"reason": "invalid_evaluation_block"})
            continue
        who, partition = block.get("candidate_id"), block.get("partition")
        if partition not in {"train", "validation"}:
            excluded.append({"candidate_id": who, "reason": "test_or_unknown_partition_ignored"})
            continue
        if not isinstance(who, str) or who not in registry and who != "baseline":
            excluded.append({"candidate_id": who, "reason": "unknown_candidate"})
            continue
        identity = block.get("block_id")
        panel = expected.get((partition, identity)) if isinstance(identity, str) else None
        if panel is None:
            excluded.append({"candidate_id": who, "block_id": identity,
                             "reason": "not_in_frozen_opportunity_universe"})
            continue
        if _embargo_at(_time(panel["outcome_available_at"]), embargo) >= cutoff:
            excluded.append({"candidate_id": who, "block_id": identity,
                             "reason": "common_panel_unmatured_or_embargoed"})
            continue
        key = (who, partition, identity) if isinstance(identity, str) else None
        if key is not None:
            if key in seen_blocks or key in tainted_blocks:
                tainted_blocks.add(key)
                parsed.pop(key, None)
                excluded.append({"candidate_id": who, "block_id": identity,
                                 "reason": "duplicate_or_conflicting_block_permanently_tainted"})
                continue
            seen_blocks.add(key)
        try:
            if (_time(block.get("block_start")) != _time(panel["block_start"])
                    or _time(block.get("block_end")) != _time(panel["block_end"])):
                raise ValueError("different_frozen_comparison_window")
            value = _validate_block(block, registry.get(who), execution_id)
            if not isinstance(identity, str) or not identity:
                raise ValueError("missing_or_duplicate_block_id")
            key = (who, partition, identity)
            parsed[key] = value
            panel = expected.get((partition, identity))
            if (not panel or value["start"] != _time(panel["block_start"])
                    or value["end"] != _time(panel["block_end"])
                    or value["available"] > _time(panel["outcome_available_at"])):
                raise ValueError("different_frozen_comparison_window")
            panel_rows = {row["opportunity_id"]: row for row in panel["opportunities"]}
            if value["opportunities"] != set(panel_rows):
                raise ValueError("different_frozen_opportunity_panel")
            for row in value["campaigns"]:
                declared = panel_rows[row["opportunity_id"]]
                if (_time(row["origin_at"]) != _time(declared["origin_at"])
                        or _time(row["outcome_available_at"])
                        > _time(declared["outcome_available_at"])):
                    raise ValueError("different_frozen_opportunity_chronology")
        except (ValueError, KeyError, TypeError) as error:
            if isinstance(block.get("block_id"), str):
                parsed.pop((who, partition, block["block_id"]), None)
            excluded.append({"candidate_id": who, "block_id": block.get("block_id"),
                             "reason": str(error)})
    eligible, diagnostics, sample_sufficient = [], [], False
    for identity, candidate in registry.items():
        paired = {"train": [], "validation": []}
        baselines = {"train": [], "validation": []}
        reasons = []
        incomplete = False
        for (partition, block_id), panel in expected.items():
            maturity = _time(panel["outcome_available_at"])
            if _embargo_at(maturity, embargo) >= cutoff:
                reasons.append("common_panel_unmatured_or_embargoed")
                continue
            block = parsed.get((identity, partition, block_id))
            baseline = parsed.get(("baseline", partition, block_id))
            if not baseline or not block:
                reasons.append("missing_required_candidate_or_baseline_block")
                incomplete = True
                continue
            paired[partition].append(block)
            baselines[partition].append(baseline)
        try:
            if (incomplete or not paired["train"] or not paired["validation"]
                    or max(_embargo_at(_time(expected[("train", b["block_id"])][
                        "outcome_available_at"]), embargo) for b in paired["train"])
                    >= min(b["start"] for b in paired["validation"]+baselines["validation"])):
                raise ValueError("missing_or_nonchronological_train_validation")
            for partition in ("train", "validation"):
                ordered = sorted(paired[partition], key=lambda b: b["start"])
                if any(b["start"] <= a["end"] for a, b in zip(ordered, ordered[1:])):
                    raise ValueError("overlapping_evaluation_blocks")
            train, validation = (_metrics(paired[part]) for part in ("train", "validation"))
            baseline_train, baseline_validation = (_metrics(baselines[part])
                                                    for part in ("train", "validation"))
            if ({row["opportunity_id"] for block in paired["train"] for row in block["campaigns"]}
                    & {row["opportunity_id"] for block in paired["validation"]
                       for row in block["campaigns"]}):
                raise ValueError("train_validation_opportunity_leak")
            enough = (min(train["closed_campaigns"], validation["closed_campaigns"],
                          baseline_train["closed_campaigns"],
                          baseline_validation["closed_campaigns"])
                      >= min_closed_campaigns and validation["blocks"] >= required_blocks)
            sample_sufficient |= enough
            passed = (enough and validation["filled_trade_win_rate"]
                      >= baseline_validation["filled_trade_win_rate"]
                      and validation["max_drawdown_pct"]
                      <= baseline_validation["max_drawdown_pct"]+1e-9
                      and validation["net_account_return_pct"]
                      > baseline_validation["net_account_return_pct"]+1e-9)
            diagnostic = {"candidate_id": identity, "enough_samples": enough, "passed": passed,
                          "train": train, "validation": validation,
                          "baseline_validation": baseline_validation, "reasons": reasons}
            diagnostics.append(diagnostic)
            if passed:
                default = registry[defaults[0]]
                changes = sum(candidate["parameters"][key] != default["parameters"].get(key)
                              for key in candidate["parameters"])
                changes += sum(candidate["structure_parameters"][key]
                               != default["structure_parameters"][key] for key in PARAMETERS)
                changes += candidate["event_variant"] != default["event_variant"]
                eligible.append((-validation["net_account_return_pct"],
                                 -validation["opportunity_success"], changes, identity))
        except ValueError as error:
            diagnostics.append({"candidate_id": identity, "passed": False,
                                "reasons": reasons+[str(error)]})
    chosen = min(eligible)[-1] if eligible else defaults[0]
    return {"status": "selected_research_candidate" if eligible else "target_not_met"
            if sample_sufficient else "insufficient", "selected_id": chosen,
            "selected": registry[chosen], "default_id": defaults[0],
            "selection_at": cutoff.isoformat(), "diagnostics": diagnostics, "excluded": excluded,
            "evaluation_basis": "supplied_campaign_records_until_simulator_validates",
            "simulator_status": "not_implemented_external_evaluations_required",
            "parameter_application_verified": False,
            "opportunity_universe_sha256": universe["sha256"],
            "selection_opportunities": {
                "declared": sum(len(row["opportunities"]) for row in expected.values()),
                "mature": sum(len(row["opportunities"]) for row in expected.values()
                              if _embargo_at(_time(row["outcome_available_at"]), embargo)
                              < cutoff)},
            "goal_completion_proven": False, "author_comparison_proven": False,
            "production_eligible": False, "literal_author_deadlines_modified": False}
