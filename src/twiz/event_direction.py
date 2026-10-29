"""Auditable event-impact features for swing research, not equity probabilities.

P(event) is an input. P(profitable equity swing | information) is a different
quantity and needs dated equity outcomes and out-of-period calibration. This
module deliberately emits an impact score and its contributions, never relabels
an average of unrelated contract probabilities as an equity probability.
"""

import math
from datetime import datetime, timezone


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _timestamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except (TypeError, ValueError):
        return None


def aggregate_event_direction(components, *, as_of, horizon_end, channel_weights,
                              event_threshold=0.65, max_age_hours=24.0,
                              main_case_only=False):
    """Aggregate conditional effects, with at most one input per common risk.

    Each component must supply both outcome effects in [-1, 1]. In particular,
    "No rate cut" cannot silently become "rate hike". Effects and nonnegative
    channel weights are explicit research parameters, not inferred from party
    names or raw question strings. Alternative contracts for the same risk must
    share a risk_group, including joint/marginal election questions and opposite
    wordings. A missing group is rejected rather than treated as independence.

    Within a risk group choose the nearest in-horizon deadline, then freshest
    observation and stable event ID. Repeated copies do not gain weight. Multiple
    groups in one channel share that channel's budget. Unknown channels retain
    zero authority. No historical current-liquidity backfill occurs here.
    """
    cutoff, end = _timestamp(as_of), _timestamp(horizon_end)
    threshold, age = _number(event_threshold), _number(max_age_hours)
    if (cutoff is None or end is None or end <= cutoff or threshold is None
            or not 0.5 < threshold <= 1 or age is None or age <= 0
            or type(main_case_only) is not bool):
        raise ValueError("Dated horizon, event threshold and freshness must be valid.")
    if not isinstance(channel_weights, dict) or any(
            not isinstance(key, str) or not key or _number(value) is None
            or _number(value) < 0 for key, value in channel_weights.items()):
        raise ValueError("Channel weights must be explicit, finite and nonnegative.")
    weights = {key: float(value) for key, value in channel_weights.items()}
    if not math.isfinite(sum(weights.values())):
        raise ValueError("The sum of channel weights must be finite.")
    eligible, rejected = [], []
    for index, row in enumerate(components):
        if not isinstance(row, dict):
            rejected.append({"index": index, "reason": "invalid_component"})
            continue
        event_id, group, channel = (row.get(key) for key in (
            "event_id", "risk_group", "channel"))
        p, q, yes_effect, no_effect = (_number(row.get(key)) for key in (
            "yes_probability", "no_probability", "yes_equity_effect", "no_equity_effect"))
        known, deadline = (_timestamp(row.get(key)) for key in ("available_at", "end_at"))
        reason = None
        if not all(isinstance(value, str) and value for value in (event_id, group, channel)):
            reason = "missing_identity_or_shared_risk_group"
        elif channel not in weights or weights[channel] <= 0:
            reason = "channel_has_no_weight"
        elif (p is None or q is None or not 0 <= p <= 1 or not 0 <= q <= 1
              or abs(p+q-1) > 0.02+1e-12):
            reason = "invalid_binary_probability"
        elif (yes_effect is None or no_effect is None or not -1 <= yes_effect <= 1
              or not -1 <= no_effect <= 1):
            reason = "both_conditional_effects_required"
        elif known is None or deadline is None or known > cutoff or known >= deadline:
            reason = "unavailable_or_invalid_timestamp"
        elif (cutoff-known).total_seconds() > age*3600:
            reason = "stale"
        elif not cutoff < deadline <= end:
            reason = "outside_swing_horizon"
        if reason:
            rejected.append({"index": index, "event_id": event_id, "reason": reason})
            continue
        yes_main, no_main = p >= threshold, q >= threshold
        if main_case_only and yes_main == no_main:
            rejected.append({"index": index, "event_id": event_id,
                             "reason": ("ambiguous_binary_main_case" if yes_main else
                                        "below_main_case_threshold")})
            continue
        # Use the canonical Yes probability plus its complement, not two slightly
        # inconsistent independently quoted tokens whose sum can exceed one.
        impact = p*yes_effect+(1-p)*no_effect
        reference = _number(row.get("reference_yes_probability"))
        reference_at = _timestamp(row.get("reference_available_at"))
        revision = ((p-reference)*(yes_effect-no_effect)
                    if reference is not None and 0 <= reference <= 1
                    and reference_at is not None and reference_at < known else None)
        eligible.append({"index": index, "event_id": event_id, "risk_group": group,
                         "channel": channel, "available_at": known.isoformat(),
                         "end_at": deadline.isoformat(), "yes_probability": p,
                         "main_case": ("Yes" if yes_main else "No")
                         if yes_main != no_main else None,
                         "yes_equity_effect": yes_effect, "no_equity_effect": no_effect,
                         "expected_impact": impact, "repricing_impact": revision,
                         "reference_available_at": (reference_at.isoformat()
                                                    if revision is not None else None),
                         "repricing_status": "available" if revision is not None else
                                             "missing_valid_preceding_reference",
                         "source_ref": row.get("source_ref")})
    identities, conflicting = {}, set()
    for row in eligible:
        signature = (row["risk_group"], row["channel"], row["end_at"],
                     row["yes_equity_effect"], row["no_equity_effect"])
        if row["event_id"] in identities and identities[row["event_id"]] != signature:
            conflicting.add(row["event_id"])
        identities[row["event_id"]] = signature
    for row in eligible:
        if row["event_id"] in conflicting:
            rejected.append({"index": row["index"], "event_id": row["event_id"],
                             "reason": "conflicting_event_identity_mapping"})
    eligible = [row for row in eligible if row["event_id"] not in conflicting]
    groups = {}
    for row in sorted(eligible, key=lambda value: (
            value["end_at"], -_timestamp(value["available_at"]).timestamp(),
            value["event_id"], value["index"])):
        if row["risk_group"] in groups:
            rejected.append({"index": row["index"], "event_id": row["event_id"],
                             "reason": "same_risk_already_represented"})
        else:
            groups[row["risk_group"]] = row
    selected = list(groups.values())
    counts = {channel: sum(row["channel"] == channel for row in selected)
              for channel in weights}
    configured = sum(weights.values())
    represented = sum(weight for channel, weight in weights.items() if counts[channel])
    # Fixed configured denominator: missing channels do not inflate remaining
    # event authority. Coverage is a separate feature, not an assumed neutral fact.
    for row in selected:
        weight = weights[row["channel"]]/counts[row["channel"]]
        row["weight"] = weight
        row["contribution"] = weight*row["expected_impact"]/configured
    score = sum(row["contribution"] for row in selected) if configured else None
    return {"status": "ready" if selected else "unavailable", "as_of": cutoff.isoformat(),
            "horizon_end": end.isoformat(), "event_threshold": threshold,
            "aggregation_mode": "main_case_only" if main_case_only else "continuous",
            "direction_score": score if selected else None,
            "equity_up_probability": None,
            "probability_status": "requires_separate_out_of_period_equity_calibration",
            "channel_coverage": represented/configured if configured else 0.0,
            "channel_weights": weights, "contributions": selected, "excluded": rejected,
            "score_definition": "sum(channel_weight*conditional_expected_impact)/configured_weight",
            "limitations": ["Outcome effects and risk groups require an explicit dated mapping.",
                            "Missing channels are unknown, not evidence of neutral conditions.",
                            "Event level and probability revisions remain separate features.",
                            "Event probabilities are not equity-direction probabilities."]}
