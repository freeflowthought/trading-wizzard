"""Dated event log-odds offsets to an unchanged price prior; research only."""

import math
from collections import Counter

import numpy as np

from .swing_optimizer import _hash, _number, _time

TARGET = "close_20_sessions_ahead_above_origin_completed_close"
VERSION = "event-residual-offset-v1"


def _weights(rows):
    counts = Counter(row["episode_id"] for row in rows)
    weights = np.asarray([1 / counts[row["episode_id"]] for row in rows])
    return weights / weights.sum()


def _sigmoid(logits):
    values = np.asarray(logits, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1 + exponent)
    return output


def _design(rows, names, mean, scale):
    # Missing raw values are never transformed or inspected as observations.
    return np.asarray([
        [float(np.clip((row["event_features"][name] - mean[i]) / scale[i], -8, 8))
         if row["observed_mask"][name] else 0.0 for i, name in enumerate(names)]
        for row in rows
    ])


def _fit(rows, names, penalty):
    weights = _weights(rows)
    mean, scale, counts = [], [], []
    for name in names:
        observed = [i for i, row in enumerate(rows) if row["observed_mask"][name]]
        counts.append(len(observed))
        if not observed:
            mean.append(0.0)
            scale.append(1.0)
            continue
        values = np.asarray([rows[i]["event_features"][name] for i in observed])
        center = float(np.average(values, weights=weights[observed]))
        variance = float(np.average((values - center) ** 2, weights=weights[observed]))
        mean.append(center)
        scale.append(math.sqrt(variance) if variance > 1e-20 else 1.0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Nonfinite observed-only feature scale.")
    x = _design(rows, names, mean, scale)
    y = np.asarray([row["label"] for row in rows])
    prior = np.asarray([row["baseline_probability"] for row in rows])
    offset = np.log(prior) - np.log1p(-prior)
    beta = np.zeros(len(names))  # Deliberately no fitted intercept or missingness term.
    ridge = np.eye(len(names)) * penalty
    for _iteration in range(80):
        p = _sigmoid(offset + x @ beta)
        gradient = x.T @ (weights * (p - y)) + ridge @ beta
        hessian = (x.T * (weights * p * (1 - p))) @ x + ridge
        step = np.linalg.solve(hessian, gradient)
        if not np.isfinite(step).all():
            raise ValueError("Nonfinite residual fit.")
        step *= min(1.0, 5 / max(float(np.max(np.abs(step))), 1e-12))
        beta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    if not np.isfinite(beta).all():
        raise ValueError("Nonfinite residual coefficients.")
    return {"feature_names": list(names), "mean": mean, "scale": scale,
            "observed_counts": dict(zip(names, counts)), "coefficients": beta.tolist(),
            "intercept": 0.0, "penalty": penalty, "iterations": _iteration + 1,
            "training_samples": len(rows), "training_episodes": len(set(
                row["episode_id"] for row in rows)),
            "training_origin_last": max(row["origin_at"] for row in rows),
            "training_outcome_cutoff": max(row["outcome_available_at"] for row in rows),
            "loss_weighting": "mean_loss_equal_episode_then_equal_rows"}


def _probabilities(model, rows):
    prior = np.asarray([row["baseline_probability"] for row in rows])
    x = _design(rows, model["feature_names"], model["mean"], model["scale"])
    adjustments = x @ np.asarray(model["coefficients"])
    values = _sigmoid(np.log(prior) - np.log1p(-prior) + adjustments)
    values[adjustments == 0] = prior[adjustments == 0]
    return values.tolist()


def _metrics(rows, probabilities):
    probabilities = np.asarray(probabilities)
    labels, weights = np.asarray([row["label"] for row in rows]), _weights(rows)
    bounded = np.clip(probabilities, 1e-12, 1 - 1e-12)
    return {"brier": float(weights @ ((probabilities - labels) ** 2)),
            "log_loss": float(weights @ (-labels * np.log(bounded)
                                         - (1 - labels) * np.log1p(-bounded))),
            "samples": len(rows), "episodes": len(set(row["episode_id"] for row in rows))}


def fit_event_residual(rows, *, as_of, event_feature_names, penalties=(1.0, 10.0, 100.0),
                       label_definition=TARGET, min_samples=60, min_event_groups=12):
    """Select an offset on preceding calibration data and gate on untouched validation.

    Required row fields: id/episode_id, origin_at/features_available_at/
    outcome_available_at, label/label_definition, baseline_probability/
    baseline_model_fit_at, event_features and observed_mask (exact named keys).
    Optional baseline_feature_available_at and baseline_label_cutoff are checked.
    Missing values have zero design/contribution; there is no free intercept.
    """
    cutoff, names = _time(as_of), tuple(event_feature_names)
    if (not cutoff or not names or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)
            or label_definition != TARGET
            or type(min_samples) is not int or min_samples < 12
            or type(min_event_groups) is not int or min_event_groups < 12
            or not penalties or any(_number(p) is None or _number(p) <= 0 for p in penalties)):
        raise ValueError("Invalid residual protocol, clocks, features or positive penalties.")
    penalties = tuple(sorted({float(p) for p in penalties}))
    timed, excluded = [], []
    for row in rows:
        if not isinstance(row, dict):
            excluded.append({"reason": "invalid_row"})
            continue
        identity, episode = row.get("id"), row.get("episode_id")
        origin, known, mature, fitted = (_time(row.get(key)) for key in (
            "origin_at", "features_available_at", "outcome_available_at", "baseline_model_fit_at"))
        if (not all((origin, known, mature, fitted)) or not known <= origin < mature < cutoff
                or fitted > origin):
            excluded.append({"id": identity, "reason": "unavailable_or_unmatured_clocks"})
            continue  # No labels, prior probabilities or event values inspected above this line.
        if (not isinstance(identity, str) or not identity or not isinstance(episode, str)
                or not episode):
            excluded.append({"id": identity, "reason": "invalid_identity"})
            continue
        optional = {key: _time(row.get(key)) for key in (
            "baseline_feature_available_at", "baseline_label_cutoff") if key in row}
        if any(not stamp or stamp > (fitted if key == "baseline_label_cutoff" else origin)
               for key, stamp in optional.items()):
            excluded.append({"id": identity, "reason": "invalid_baseline_proof_clock"})
            continue
        timed.append((row, {"id": identity, "episode_id": episode,
                           "origin_at": origin.isoformat(),
                           "features_available_at": known.isoformat(),
                           "outcome_available_at": mature.isoformat(),
                           "baseline_model_fit_at": fitted.isoformat(),
                           **{key: value.isoformat() for key, value in optional.items()}}))
    identities = Counter(metadata["id"] for _, metadata in timed)
    accepted = []
    for row, metadata in timed:
        features, mask = row.get("event_features"), row.get("observed_mask")
        prior = _number(row.get("baseline_probability"))
        reason = None
        if identities[metadata["id"]] != 1:
            reason = "duplicate_identity"
        elif (row.get("label_definition") != label_definition
              or row.get("baseline_label_definition", TARGET) != TARGET):
            reason = "different_probability_target"
        elif type(row.get("label")) is not int or row["label"] not in (0, 1):
            reason = "invalid_label"
        elif prior is None or not 0 < prior < 1:
            reason = "invalid_price_prior"
        elif (not isinstance(features, dict) or not isinstance(mask, dict)
              or set(features) != set(names) or set(mask) != set(names)
              or any(type(mask[name]) is not bool for name in names)
              or any(mask[name] and _number(features[name]) is None for name in names)):
            reason = "invalid_event_features_or_mask"
        if reason:
            excluded.append({"id": metadata["id"], "reason": reason})
            continue
        accepted.append({**metadata, "label_definition": label_definition, "label": row["label"],
                         "baseline_probability": prior, "observed_mask": dict(mask),
                         "event_features": {name: float(features[name]) if mask[name] else None
                                            for name in names}})
    accepted.sort(key=lambda row: (row["origin_at"], row["id"]))
    grouped = {}
    for row in accepted:
        grouped.setdefault(row["episode_id"], []).append(row)
    groups = sorted(grouped, key=lambda g: (min(row["origin_at"] for row in grouped[g]), g))
    audit = {"version": VERSION, "as_of": cutoff.isoformat(), "feature_names": list(names),
             "label_definition": label_definition, "accepted_samples": len(accepted),
             "event_episodes": len(groups), "excluded": excluded, "input_sha256": _hash(accepted),
             "validated": False, "event_authority_enabled": False, "production_eligible": False,
             "probability_calibrated": False, "goal_completion_proven": False,
             "algorithm": "ridge_logistic_offset_observed_only_no_intercept",
             "fallback": "unchanged_price_prior", "penalties": list(penalties),
             "split_policy": "chronological_episodes_50_25_25_purged_at_next_origin",
             "selection_rule": "calibration_Brier_then_log_loss_then_penalty",
             "minimum_samples": min_samples, "minimum_event_episodes": min_event_groups}
    if len(accepted) < min_samples or len(groups) < min_event_groups:
        return {**audit, "status": "insufficient_history"}
    first, second = len(groups) // 2, len(groups) * 3 // 4
    blocks = [[row for group in ids for row in grouped[group]] for ids in (
        groups[:first], groups[first:second], groups[second:])]
    purged = []
    for i in (0, 1):
        next_origin = min(row["origin_at"] for row in blocks[i + 1])
        purged.extend(row["id"] for row in blocks[i] if row["outcome_available_at"] >= next_origin)
        blocks[i] = [row for row in blocks[i] if row["outcome_available_at"] < next_origin]
    audit["purged_ids"] = purged
    audit["blocks"] = {name: {"episodes": sorted({r["episode_id"] for r in block}),
                               "ids": [r["id"] for r in block], "samples": len(block),
                               "label_counts": dict(Counter(str(r["label"]) for r in block))}
                       for name, block in zip(("train", "calibration", "validation"), blocks)}
    train, calibration, validation = blocks
    if (sum(len(block) for block in blocks) < min_samples
            or len({r["episode_id"] for block in blocks for r in block}) < min_event_groups
            or any(len(block) < 4 or len({r["episode_id"] for r in block}) < 3
                   for block in blocks)):
        return {**audit, "status": "insufficient_purged_blocks"}
    try:
        choices = []
        for penalty in penalties:
            fitted = _fit(train, names, penalty)
            scores = _metrics(calibration, _probabilities(fitted, calibration))
            choices.append({"penalty": penalty, "calibration": scores,
                            "training_outcome_cutoff": fitted["training_outcome_cutoff"]})
        chosen = min(choices, key=lambda x: (
            x["calibration"]["brier"], x["calibration"]["log_loss"], x["penalty"]))
        refit = train + calibration
        validation_start = min(row["origin_at"] for row in validation)
        assert all(row["outcome_available_at"] < validation_start for row in refit)
        model = _fit(refit, names, chosen["penalty"])
        scores = _metrics(validation, _probabilities(model, validation))
        baseline = _metrics(validation, [row["baseline_probability"] for row in validation])
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        return {**audit, "status": "fit_unavailable", "reason": str(exc)}
    validated = (scores["brier"] < baseline["brier"] - 1e-12
                 and scores["log_loss"] < baseline["log_loss"] - 1e-12)
    return {**audit, "status": (
                "validated_research_candidate" if validated else "price_prior_fallback"),
            "validated": validated, "event_authority_enabled": validated, "model": model,
            "penalty_selection": choices, "selected_penalty": chosen["penalty"],
            "calibration_cutoff": max(r["outcome_available_at"] for r in calibration),
            "calibration_has_both_classes": len({r["label"] for r in calibration}) == 2,
            "independent_probability_calibration": False,
            "validation": scores, "price_prior_validation": baseline,
            "gate": "both_validation_Brier_and_log_loss_strictly_improve_unchanged_prior"}


def predict_event_residual(artifact, *, baseline_probability, event_features, observed_mask,
                           as_of, baseline_model_fit_at, baseline_feature_available_at,
                           features_available_at, baseline_label_definition=TARGET):
    """Return the supplied prior unchanged whenever event authority is absent."""
    cutoff, fitted, known, price_known = (
        _time(as_of), _time(baseline_model_fit_at),
        _time(features_available_at), _time(baseline_feature_available_at))
    if (not cutoff or not fitted or fitted > cutoff
            or not price_known or price_known > cutoff):
        raise ValueError("Price prior and its model must be available at prediction time.")
    if artifact.get("label_definition") != TARGET or baseline_label_definition != TARGET:
        raise ValueError(
            "Residual and price prior must use the fixed twenty-session equity target.")
    prior = _number(baseline_probability)
    if prior is None or not 0 < prior < 1:
        raise ValueError("Price prior must be a finite probability strictly between zero and one.")
    model = artifact.get("model", {})
    names = artifact.get("feature_names", [])
    clock = _time(artifact.get("as_of"))
    validated = bool(artifact.get("validated") and clock and clock <= cutoff)
    enabled = bool(validated and artifact.get("event_authority_enabled")
                   and known and known <= cutoff)
    contributions, adjustment = [], 0.0
    if enabled:
        if (not isinstance(event_features, dict) or not isinstance(observed_mask, dict)
                or set(event_features) != set(names) or set(observed_mask) != set(names)
                or any(type(observed_mask[name]) is not bool for name in names)
                or any(observed_mask[name] and _number(event_features[name]) is None
                       for name in names)):
            raise ValueError("Event features and explicit observed masks must match the model.")
        for i, name in enumerate(names):
            observed = observed_mask[name]
            value = _number(event_features[name]) if observed else None
            z = (float(np.clip((value - model["mean"][i]) / model["scale"][i], -8, 8))
                 if observed else 0.0)
            contribution = z * model["coefficients"][i]
            adjustment += contribution
            contributions.append({"feature": name, "observed": observed, "value": value,
                                  "standardized_value": z, "weight": model["coefficients"][i],
                                  "log_odds_contribution": contribution})
        enabled = any(observed_mask.values())
    probability = prior if adjustment == 0 else float(_sigmoid([
        math.log(prior) - math.log1p(-prior) + adjustment])[0])
    return {"status": "event_adjusted_research" if enabled else "price_prior_only",
            "probability": probability, "equity_up_probability": probability,
            "label_definition": TARGET,
            "baseline_probability": prior, "event_authority_enabled": enabled,
            "validated": validated, "contributions": contributions,
            "event_log_odds_adjustment": adjustment, "intercept": 0.0,
            "probability_calibrated": False, "automatic_policy_promotion": False,
            "reason": None if enabled else "event_authority_unavailable_or_all_events_missing"}
