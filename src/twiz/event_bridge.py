"""Learn dated event/market weights and validate their directional probability.

This is an auxiliary probability model for swing decisions. Its validation is
not proof of better bottom/top levels or trading returns. Rows must come from a
separately audited historical adapter; this module fetches no provider data.
"""

import hashlib
import json
import math

import numpy as np

from .evaluation import _apply_calibrator, _calibrate, _probability
from .event_direction import _number, _timestamp


def _fit(rows, names, penalty):
    x = np.asarray([[row["features"][name] for name in names] for row in rows], dtype=float)
    y = np.asarray([row["label"] for row in rows], dtype=float)
    counts = {group: sum(row["episode_id"] == group for row in rows)
              for group in {row["episode_id"] for row in rows}}
    # A thousand quotes from one FOMC do not count as a thousand independent
    # events. Each episode has total sample weight one, across assets as well.
    weights = np.asarray([1/counts[row["episode_id"]] for row in rows])
    center = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-center)**2, axis=0, weights=weights))
    scale[scale < 1e-10] = 1
    design = np.column_stack((np.ones(len(x)), np.clip((x-center)/scale, -8, 8)))
    beta = np.zeros(design.shape[1])
    prior = (float(weights@y)+0.5)/(float(weights.sum())+1)
    beta[0] = math.log(prior/(1-prior))
    ridge = np.eye(len(beta))*penalty
    ridge[0, 0] = 1e-8
    for _iteration in range(60):
        p = 1/(1+np.exp(-np.clip(design@beta, -35, 35)))
        gradient = design.T@(weights*(p-y))+ridge@beta
        hessian = (design.T*(weights*p*(1-p)))@design+ridge
        step = np.linalg.solve(hessian, gradient)
        step *= min(1, 5/max(float(np.max(np.abs(step))), 1e-8))
        beta -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return {"feature_names": list(names), "mean": center.tolist(), "scale": scale.tolist(),
            "coefficients": beta.tolist(), "penalty": penalty,
            "training_samples": len(rows), "training_episodes": len(counts),
            "iterations": _iteration+1, "prior_probability": prior}


def _metrics(rows, probabilities):
    groups = {row["episode_id"] for row in rows}
    brier, logloss = [], []
    for group in sorted(groups):
        pairs = [(row["label"], min(1-1e-12, max(1e-12, p)))
                 for row, p in zip(rows, probabilities) if row["episode_id"] == group]
        brier.append(sum((p-y)**2 for y, p in pairs)/len(pairs))
        logloss.append(sum(-y*math.log(p)-(1-y)*math.log(1-p) for y, p in pairs)/len(pairs))
    return {"brier": sum(brier)/len(brier), "log_loss": sum(logloss)/len(logloss),
            "samples": len(rows), "independent_event_episodes": len(groups),
            "weighting": "equal_episode_then_equal_rows_within_episode"}


def fit_event_bridge(rows, *, as_of, feature_names, label_definition,
                     baseline_feature_names=(), penalties=(0.1, 1.0, 10.0),
                     min_event_groups=20, min_samples=100):
    """Fit, calibrate and compare on three disjoint chronological event blocks.

    Each row supplies id, episode_id, origin_at, features_available_at,
    outcome_available_at, features, label and label_definition. A label may only
    enter a preceding block once fully observed before the following block's
    first origin. No event episode appears in two blocks. Candidate penalties
    are selected with the calibration block; the final validation block is never
    reused for fitting or calibration. A failed gate returns no live probability.
    """
    cutoff = _timestamp(as_of)
    names = tuple(feature_names)
    base_names = tuple(baseline_feature_names)
    if (cutoff is None or not names or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)
            or not isinstance(label_definition, str) or not label_definition
            or not set(base_names) <= set(names)
            or len(set(base_names)) != len(base_names)
            or type(min_event_groups) is not int or min_event_groups < 5
            or type(min_samples) is not int or min_samples < 10
            or not penalties or any(_number(p) is None or _number(p) <= 0 for p in penalties)):
        raise ValueError("Invalid event bridge protocol, dates, features or penalties.")
    accepted, excluded, ids = [], [], set()
    for row in rows:
        if not isinstance(row, dict):
            excluded.append({"reason": "invalid_row"})
            continue
        identity, episode = row.get("id"), row.get("episode_id")
        origin, known, matured = (_timestamp(row.get(key)) for key in (
            "origin_at", "features_available_at", "outcome_available_at"))
        features = row.get("features")
        reason = None
        if (not isinstance(identity, str) or not identity or not isinstance(episode, str)
                or not episode or identity in ids):
            reason = "missing_or_duplicate_identity"
        elif (origin is None or known is None or matured is None or known > origin
              or origin >= matured or matured >= cutoff):
            reason = "unavailable_features_or_unmatured_outcome"
        elif row.get("label_definition") != label_definition:
            reason = "different_probability_target"
        elif type(row.get("label")) is not int or row["label"] not in (0, 1):
            reason = "invalid_outcome"
        elif (not isinstance(features, dict)
              or any(_number(features.get(name)) is None for name in names)):
            reason = "missing_or_invalid_feature"
        if reason:
            excluded.append({"id": identity, "reason": reason})
            continue
        ids.add(identity)
        accepted.append({**row, "origin_at": origin.isoformat(),
                         "outcome_available_at": matured.isoformat(),
                         "features": {name: float(features[name]) for name in names}})
    grouped = {}
    for row in accepted:
        grouped.setdefault(row["episode_id"], []).append(row)
    groups = sorted(grouped, key=lambda group: (min(row["origin_at"] for row in grouped[group]),
                                               group))
    audit = {"as_of": cutoff.isoformat(), "label_definition": label_definition,
             "feature_names": list(names), "accepted_samples": len(accepted),
             "independent_event_episodes": len(groups), "excluded": excluded,
             "production_eligible": False, "equity_up_probability": None,
             "probability_estimate_eligible": False,
             "algorithm": "episode_weighted_regularized_logistic_then_heldout_sigmoid",
             "input_sha256": hashlib.sha256(json.dumps(accepted, sort_keys=True,
                                                       allow_nan=False).encode()).hexdigest()}
    if len(accepted) < min_samples or len(groups) < min_event_groups:
        return {**audit, "status": "insufficient_history",
                "required_samples": min_samples, "required_event_episodes": min_event_groups}
    first, second = max(1, len(groups)*3//5), max(2, len(groups)*4//5)
    splits = [groups[:first], groups[first:second], groups[second:]]
    blocks = [[row for group in block for row in grouped[group]] for block in splits]
    for i in (0, 1):
        next_origin = min(row["origin_at"] for row in blocks[i+1])
        blocks[i] = [row for row in blocks[i] if row["outcome_available_at"] < next_origin]
    train, calibration, validation = blocks
    audit["blocks"] = {name: {"episodes": sorted({row["episode_id"] for row in block}),
                              "samples": len(block),
                              "label_counts": {str(label): sum(
                                  row["label"] == label for row in block)
                                               for label in (0, 1)}} for name, block in zip(
                                  ("fit", "calibration", "validation"), blocks)}
    if (any(len(block) < 20 for block in blocks)
            or len({row["label"] for row in train}) < 2
            or len({row["episode_id"] for row in validation}) < 3):
        return {**audit, "status": "insufficient_purged_blocks"}

    def select(columns):
        choices = []
        for penalty in penalties:
            model = _fit(train, columns, float(penalty))
            p = [_probability(model, [row["features"][name] for name in columns])
                 for row in calibration]
            metrics = _metrics(calibration, p)
            choices.append((metrics["brier"], metrics["log_loss"], float(penalty), model, p))
        best = min(choices, key=lambda choice: choice[:3])
        return best[3], best[4]

    model, calibration_p = select(names)
    def calibrate(probabilities):
        # One predeclared last-origin case per episode, not a chosen winner.
        # A thousand quotes are not a thousand independent calibration events.
        cal_p, cal_y = [], []
        for group in sorted({row["episode_id"] for row in calibration}):
            positions = [i for i, row in enumerate(calibration) if row["episode_id"] == group]
            i = max(positions, key=lambda j: (calibration[j]["origin_at"], calibration[j]["id"]))
            cal_p.append(probabilities[i])
            cal_y.append(calibration[i]["label"])
        return _calibrate(cal_p, cal_y)

    calibrator = calibrate(calibration_p)
    raw = [_probability(model, [row["features"][name] for name in names]) for row in validation]
    calibrated = ([_apply_calibrator(p, calibrator) for p in raw]
                  if calibrator["available"] else raw)
    metrics = _metrics(validation, calibrated)
    prior_metrics = _metrics(validation, [model["prior_probability"]]*len(validation))
    references = {"event_free_prior": prior_metrics}
    if base_names:
        baseline, baseline_cal_p = select(base_names)
        baseline_calibrator = calibrate(baseline_cal_p)
        baseline_p = [
            _probability(baseline, [row["features"][name] for name in base_names])
            for row in validation]
        if baseline_calibrator["available"]:
            baseline_p = [_apply_calibrator(p, baseline_calibrator) for p in baseline_p]
        references["price_macro_only"] = _metrics(validation, baseline_p)
    improves = all(metrics["brier"] < ref["brier"] and metrics["log_loss"] < ref["log_loss"]
                   for ref in references.values())
    eligible = improves and calibrator["available"] and bool(base_names)
    return {**audit, "status": "validated_research_candidate" if eligible else "not_validated",
            "probability_estimate_eligible": eligible, "model": model, "calibrator": calibrator,
            "validation": metrics, "references": references,
            "gate": {"improves_all_references": improves,
                     "independent_calibration_available": calibrator["available"],
                     "price_macro_ablation_included": bool(base_names)},
            "limitations": ["Directional probability validation is not swing-profit validation.",
                            "A later untouched period is required before automatic promotion.",
                            "Current contract rules do not prove historical rule availability."]}


def predict_event_bridge(artifact, features, *, as_of, features_available_at):
    """Expose a validated estimate with additive log-odds contributions."""
    cutoff, known = _timestamp(as_of), _timestamp(features_available_at)
    fitted = _timestamp(artifact.get("as_of"))
    if (cutoff is None or known is None or fitted is None or known > cutoff or fitted > cutoff):
        raise ValueError("Prediction cannot precede its model or input availability.")
    if not artifact.get("probability_estimate_eligible"):
        return {"status": "unavailable", "equity_up_probability": None,
                "reason": artifact.get("status", "not_validated")}
    model = artifact["model"]
    names = model["feature_names"]
    values = [_number(features.get(name)) for name in names]
    if any(value is None for value in values):
        return {"status": "unavailable", "equity_up_probability": None,
                "reason": "missing_feature"}
    standardized = np.clip((np.asarray(values)-model["mean"])/model["scale"], -8, 8)
    contributions = [{"feature": name, "value": value, "standardized_value": float(z),
                      "learned_weight": weight, "log_odds_contribution": float(z*weight)}
                     for name, value, z, weight in zip(
                         names, values, standardized, model["coefficients"][1:])]
    raw = _probability(model, values)
    return {"status": "research_estimate", "label_definition": artifact["label_definition"],
            "as_of": cutoff.isoformat(), "raw_probability": raw,
            "equity_up_probability": _apply_calibrator(raw, artifact["calibrator"]),
            "intercept": model["coefficients"][0], "contributions": contributions,
            "validation": artifact["validation"], "automatic_policy_promotion": False}
