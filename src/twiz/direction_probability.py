"""Chronological, price-first 20-session direction estimates for the daily brief.

Event quotations are features, never substituted for equity probabilities. Missing
historical option chains are deliberately excluded. Downloaded history is not a
certificate of historical publication times or future model performance.
"""

import math
from datetime import date, datetime
from statistics import median

import numpy as np

from .sector_rotation import _clock, _next
from .sessions import is_us_equity_session
from .structure import _atr, _clean
from .swing_forecast import FEATURES, _feature
from .swing_optimizer import _hash

PRICE_NAMES = (*FEATURES[:9], "candle_body_atr", "close_in_daily_range")
AUX_NAMES = ("month_sin", "month_cos", "capex_yoy", "revenue_yoy",
             "capex_yoy_acceleration", "yield10", "vix", "fed_upper", "event_direction")
PENALTIES = (.1, 1., 10.)
SHRINKAGES = (0., .25, .5, .75, 1.)


def _time(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp if stamp.tzinfo and stamp.utcoffset() is not None else None
    except (TypeError, ValueError):
        return None


def _number(value):
    return (float(value) if type(value) in (int, float) and math.isfinite(value) else None)


def _available(records, cutoff):
    """Only metadata of future records is inspected, not their numeric payload."""
    eligible = []
    for row in records or []:
        if not isinstance(row, dict):
            continue
        known = _time(row.get("available_at"))
        if known and known <= cutoff:
            eligible.append((known, row))
    return sorted(eligible, key=lambda pair: pair[0])


def _auxiliary(day, capex_rows, event_rows, macro_rows):
    cutoff = _time(_clock(day))
    values = dict.fromkeys(AUX_NAMES, None)
    values.update(month_sin=math.sin(2*math.pi*day.month/12),
                  month_cos=math.cos(2*math.pi*day.month/12))
    capex = _available(capex_rows, cutoff)
    if capex:
        _, row = capex[-1]
        agg = row.get("aggregate", {})
        fresh = []
        for name in agg.get("included_issuers", []):
            issuer = row.get("issuers", {}).get(name, {})
            current = issuer.get("current", {})
            clocks = [_time(current.get(kind, {}).get("available_at_proxy"))
                      for kind in ("capex", "revenue")]
            if (issuer.get("status") == "available" and all(
                    clock and 0 <= (cutoff-clock).days <= 130 for clock in clocks)):
                fresh.append(issuer)
        if agg.get("status") == "available" and len(fresh) >= 2:
            for target, key in (("capex_yoy", "median_cash_capex_yoy_pct"),
                                ("revenue_yoy", "median_revenue_yoy_pct"),
                                ("capex_yoy_acceleration",
                                 "median_cash_capex_yoy_acceleration_pp")):
                observed = [_number(issuer.get(key.removeprefix("median_"))) for issuer in fresh]
                observed = [value for value in observed if value is not None]
                values[target] = median(observed) if len(observed) >= 2 else None
    macro = _available(macro_rows, cutoff)
    if macro:
        known, row = macro[-1]
        if (cutoff-known).days <= 10:
            for target, key in (("yield10", "yield10_pct"), ("vix", "vix"),
                                ("fed_upper", "fed_upper_pct")):
                proof = row.get("source_observations", {}).get(key, {})
                clock = _time(proof.get("available_at"))
                if clock and 0 <= (cutoff-clock).days <= 10:
                    values[target] = _number(row.get("values", {}).get(key))
    events = _available(event_rows, cutoff)
    live = [(known, row) for known, row in events if (cutoff-known).days <= 14]
    families = {}
    for known, row in live:
        family = row.get("correlation_family") or row.get("driver") or row.get("id")
        expires = _time(row.get("expires_at"))
        if (not family or not row.get("source_refs") or expires and expires < cutoff):
            continue
        if family not in families or known > families[family][0]:
            families[family] = (known, [row])
        elif known == families[family][0]:
            families[family][1].append(row)
    scores = []
    for _, tied in families.values():
        directions = {_number(row.get("direction")) for row in tied}
        if len(directions) == 1:  # Conflicting same-clock versions carry no authority.
            scores.extend(directions)
    scores = [score for score in scores if score is not None and -1 <= score <= 1]
    if scores:
        values["event_direction"] = max(-1., min(1., sum(scores)))
    return [values[name] for name in AUX_NAMES]


def _sigmoid(value):
    return 1/(1+np.exp(-np.clip(value, -35, 35)))


def _fit(records, size, penalty):
    raw = np.asarray([[np.nan if v is None else v for v in r["x"][:size]] for r in records])
    observed = np.isfinite(raw)
    counts = observed.sum(axis=0)
    center = np.nansum(raw, axis=0)/np.maximum(counts, 1)
    scale = np.sqrt(np.nansum((raw-center)**2, axis=0)/np.maximum(counts, 1))
    scale[scale < 1e-8] = 1
    x = np.column_stack([np.ones(len(records)),
                         np.where(observed, np.clip((raw-center)/scale, -5, 5), 0.)])
    y = np.asarray([r["label"] for r in records], dtype=float)
    prior = float(np.clip(y.mean(), .02, .98))
    beta = np.zeros(size+1)
    beta[0] = math.log(prior/(1-prior))
    ridge = np.eye(size+1)*penalty
    ridge[0, 0] = 1e-8
    for _ in range(50):
        p = _sigmoid(x@beta)
        step = np.linalg.solve((x.T*(p*(1-p)))@x+ridge, x.T@(p-y)+ridge@beta)
        step *= min(1., 3/max(float(np.max(np.abs(step))), 1e-12))
        beta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return {"center": center.tolist(), "scale": scale.tolist(), "beta": beta.tolist(),
            "prior": prior, "penalty": penalty, "observed_counts": counts.tolist(), "size": size}


def _predict(model, records):
    raw = np.asarray([[np.nan if v is None else v for v in r["x"][:model["size"]]]
                      for r in records])
    x = np.where(np.isfinite(raw), np.clip(
        (raw-np.asarray(model["center"]))/np.asarray(model["scale"]), -5, 5), 0.)
    if model.get("method") == "analog":
        train, masks = np.asarray(model["training_x"]), np.asarray(model["training_mask"])
        weights = np.ones(model["size"])/model["size"]
        if model["size"] > len(PRICE_NAMES):
            weights[:len(PRICE_NAMES)] = .7/len(PRICE_NAMES)
            weights[len(PRICE_NAMES):] = .3/(model["size"]-len(PRICE_NAMES))
        probabilities = []
        for row, mask in zip(x, np.isfinite(raw)):
            shared = masks*mask*weights
            distance = np.sum((train-row)**2*shared, axis=1)/np.maximum(shared.sum(axis=1), 1e-12)
            selected = np.argsort(distance, kind="stable")[:model["neighbors"]]
            labels = np.asarray(model["training_labels"])[selected]
            probabilities.append((float(labels.sum())+2*model["prior"])/(len(labels)+2))
        p = np.asarray(probabilities)
    else:
        p = _sigmoid(model["beta"][0]+x@np.asarray(model["beta"][1:]))
    return np.clip(model.get("shrinkage", 1.)*p+
                   (1-model.get("shrinkage", 1.))*model["prior"], .02, .98)


def _select(train, calibration, validation, size):
    y = np.asarray([r["label"] for r in calibration])
    trials = []
    for penalty in PENALTIES:
        model = _fit(train, size, penalty)
        for shrink in SHRINKAGES:
            trial = {**model, "shrinkage": shrink}
            brier = float(np.mean((_predict(trial, calibration)-y)**2))
            trials.append((brier, -penalty, shrink, trial))
    model = min(trials, key=lambda row: row[:3])[3]
    predictions = _predict(model, validation)
    labels = np.asarray([r["label"] for r in validation])
    brier = float(np.mean((predictions-labels)**2))
    prior_brier = float(np.mean((model["prior"]-labels)**2))
    return model, {"brier": brier, "prior_brier": prior_brier, "samples": len(validation),
                   "calibration_brier": min(trials, key=lambda row: row[:3])[0],
                   "accuracy": float(np.mean((predictions >= .5) == labels)),
                   "mean_probability_up": float(np.mean(predictions)),
                   "observed_up_frequency": float(np.mean(labels)),
                   "origin_first": validation[0]["date"], "origin_last": validation[-1]["date"]}


def _fit_analog(records, size, neighbors):
    model = _fit(records, size, 10.)
    raw = np.asarray([[np.nan if v is None else v for v in r["x"][:size]] for r in records])
    mask = np.isfinite(raw)
    normalized = np.where(mask, np.clip(
        (raw-np.asarray(model["center"]))/np.asarray(model["scale"]), -5, 5), 0.)
    return {"method": "analog", "size": size, "center": model["center"],
            "scale": model["scale"], "prior": model["prior"], "neighbors": neighbors,
            "training_x": normalized.tolist(), "training_mask": mask.tolist(),
            "training_labels": [r["label"] for r in records],
            "observed_counts": model["observed_counts"],
            "feature_family_budget": "price70percent_aux30percent; shared observed dimensions only"}


def _select_analog(train, calibration, validation, size):
    trials, y = [], np.asarray([r["label"] for r in calibration])
    for neighbors in (20, 40, 80):
        model = _fit_analog(train, size, neighbors)
        for shrinkage in SHRINKAGES:
            trial = {**model, "shrinkage": shrinkage}
            brier = float(np.mean((_predict(trial, calibration)-y)**2))
            trials.append((brier, -neighbors, shrinkage, trial))
    chosen = min(trials, key=lambda row: row[:3])
    model, labels = chosen[3], np.asarray([r["label"] for r in validation])
    probabilities = _predict(model, validation)
    return model, {"brier": float(np.mean((probabilities-labels)**2)),
                   "prior_brier": float(np.mean((model["prior"]-labels)**2)),
                   "samples": len(validation), "calibration_brier": chosen[0],
                   "accuracy": float(np.mean((probabilities >= .5) == labels)),
                   "mean_probability_up": float(probabilities.mean()),
                   "observed_up_frequency": float(labels.mean()),
                   "origin_first": validation[0]["date"], "origin_last": validation[-1]["date"]}


def estimate_direction(rows, origin_date, *, capex_rows=None, event_rows=None, macro_rows=None,
                       horizon_sessions=20, training_sessions=1260):
    """A dated research estimate with an out-of-sample prior/price ablation gate.

Weekly observations still have overlapping forward labels; validation sample
counts are not independent trials. A constant historical prior is explicitly
identified when conditional features fail, rather than presented as an event call.
"""
    if horizon_sessions != 20 or type(training_sessions) is not int or training_sessions < 504:
        raise ValueError("Require the declared20-session horizon and504 training sessions.")
    day = date.fromisoformat(origin_date)
    prefix = [r for r in rows if isinstance(r, dict) and isinstance(r.get("date"), str)
              and r["date"] <= origin_date]
    clean = _clean(prefix, origin_date)
    audit = {"origin_date": origin_date, "horizon_sessions": 20, "probability_up": None,
             "probability_down": None, "historical_pit_verified": False,
             "excluded_inputs": ["unavailable_historical_option_chain"],
             "target": "completed_close_20_US_sessions_ahead_above_origin_close",
             "down_includes_unchanged": True, "goal_completion_proven": False}
    if (len(clean) != len(prefix) or len(clean) < 800 or clean[-1]["date"] != origin_date
            or any(r.get("session_complete") is not True for r in clean)
            or any(not is_us_equity_session(date.fromisoformat(r["date"])) for r in clean)
            or any(not isinstance(r.get("price_basis"), str) or not r["price_basis"].strip()
                   or "unknown" in r["price_basis"].lower() for r in clean)
            or len({r.get("price_basis") for r in clean}) != 1
            or any(date.fromisoformat(b["date"]) != _next(date.fromisoformat(a["date"]))
                   for a, b in zip(clean, clean[1:]))):
        return {**audit, "status": "unavailable",
                "reason": "insufficient_complete_continuous_history"}
    first, final = max(200, len(clean)-training_sessions), len(clean)-26
    records = []
    for i in range(first, final+1):
        point_day = date.fromisoformat(clean[i]["date"])
        if _next(point_day).weekday() > point_day.weekday():
            continue  # Last actual US session in its trading week, including holiday weeks.
        feature = _feature(clean, i)
        atr = _atr(clean[:i+1])
        if feature is None or not atr:
            continue
        row = clean[i]
        candle = [(row["close"]-row["open"])/atr,
                  (row["close"]-row["low"])/(row["high"]-row["low"])
                  if row["high"] > row["low"] else .5]
        x = feature[:9]+candle+_auxiliary(point_day, capex_rows, event_rows, macro_rows)
        records.append({"date": row["date"], "mature_date": clean[i+20]["date"], "x": x,
                        "label": int(clean[i+20]["close"] > row["close"])})
    if len(records) < 120:
        return {**audit, "status": "unavailable", "reason": "fewer_than_120_mature_weekly_cases"}
    a, b = int(len(records)*.60), int(len(records)*.80)
    train = [r for r in records[:a] if r["mature_date"] < records[a]["date"]]
    calibration = [r for r in records[a:b] if r["mature_date"] < records[b]["date"]]
    validation = records[b:]
    if min(len(train), len(calibration), len(validation)) < 20:
        return {**audit, "status": "unavailable", "reason": "insufficient_purged_partitions"}
    price, price_score = _select(train, calibration, validation, len(PRICE_NAMES))
    combined, combined_score = _select(train, calibration, validation, len(PRICE_NAMES+AUX_NAMES))
    price_analog = _select_analog(train, calibration, validation, len(PRICE_NAMES))
    combined_analog = _select_analog(train, calibration, validation, len(PRICE_NAMES+AUX_NAMES))
    price, price_score = min([(price, price_score), price_analog],
                             key=lambda pair: pair[1]["calibration_brier"])
    combined, combined_score = min([(combined, combined_score), combined_analog],
                                   key=lambda pair: pair[1]["calibration_brier"])
    use_combined = combined_score["calibration_brier"] < price_score["calibration_brier"]-1e-10
    chosen, score = (combined, combined_score) if use_combined else (price, price_score)
    candidate_score = dict(score)
    kind = "price_calendar_macro_capex_events" if use_combined else "price_only"
    if chosen.get("method") == "analog":
        kind += "_analog"
    if score["brier"] >= score["prior_brier"]-1e-5 or chosen["shrinkage"] == 0:
        chosen = {**chosen, "shrinkage": 0.}
        kind = "historical_base_rate"
        labels = np.asarray([r["label"] for r in validation])
        score = {**score, "brier": score["prior_brier"],
                 "mean_probability_up": chosen["prior"],
                 "accuracy": float(np.mean((chosen["prior"] >= .5) == labels))}
    # Refit only after frozen chronological calibration/validation selected the specification.
    model = (_fit_analog(records, chosen["size"], chosen["neighbors"])
             if chosen.get("method") == "analog"
             else _fit(records, chosen["size"], chosen["penalty"]))
    model["shrinkage"] = chosen["shrinkage"]
    current = clean[-1]
    atr = _atr(clean)
    feature = _feature(clean, len(clean)-1)
    if not feature or not atr:
        return {**audit, "status": "unavailable", "reason": "missing_current_atr"}
    x = feature[:9]+[(current["close"]-current["open"])/atr,
                     (current["close"]-current["low"])/(current["high"]-current["low"])
                     if current["high"] > current["low"] else .5]
    x += _auxiliary(day, capex_rows, event_rows, macro_rows)
    probability = float(np.clip(_predict(model, [{"x": x}])[0], .02, .98))
    names = (PRICE_NAMES+AUX_NAMES)[:model["size"]]
    contributions = []
    for i, name in enumerate(names):
        value = x[i]
        if model.get("method") == "analog" and value is not None:
            without = list(x)
            without[i] = None
            p = float(_predict(model, [{"x": without}])[0])
            contribution = math.log(probability/(1-probability))-math.log(p/(1-p))
        else:
            contribution = (float(np.clip((value-model["center"][i])/model["scale"][i], -5, 5))
                            *model["beta"][i+1] if value is not None else 0.)
        contributions.append({"factor": name, "observed": value is not None,
                              "value": value, "log_odds_contribution": contribution,
                              "active": model["shrinkage"] > 0})
    return {**audit, "status": "available", "model_kind": kind,
            "probability_up": probability, "probability_down": 1-probability,
            "calibration_shrinkage": model["shrinkage"], "model": model,
            "conditional_model_beats_prior": kind != "historical_base_rate",
            "validation": score, "price_validation": price_score,
            "conditional_candidate_validation": candidate_score,
            "combined_validation": combined_score, "contributions": contributions,
            "model_family_selection": "calibration_only; validation_used_only_as_prior_edge_gate",
            "contribution_method": ("nonadditive_leave_one_current_feature_out_log_odds"
                                    if model.get("method") == "analog" else "linear_log_odds"),
            "training_samples": len(records), "train_samples": len(train),
            "calibration_samples": len(calibration),
            "training_label_last": records[-1]["mature_date"],
            "training_origin_first": records[0]["date"], "refit_at": _clock(day),
            "used_price_prefix_sha256": _hash(prefix),
            "limitations": ["Weekly20-session targets overlap; cases are not independent.",
                            "Validation selects the specification; this is adaptive research.",
                            "Filing/macro availability proxies are not certified timestamps.",
                            "Factor contributions are log odds before probability shrinkage."]}
