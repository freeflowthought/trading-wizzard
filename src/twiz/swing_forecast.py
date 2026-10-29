"""Dated conditional price-range forecasts, using only matured historical swings.

Neighbor quantiles are research estimates, not calibrated stock probabilities or
reconstructions of either author's private indicators. Trading validation is
separate from forecasting future lows/highs.
"""

import math
from datetime import date, timedelta

import numpy as np

from .sessions import is_us_equity_session
from .structure import _atr, _clean
from .swing_optimizer import _hash

FEATURES = ("return5_atr", "return20_atr", "sma20_distance_atr", "sma60_distance_atr",
            "sma200_distance_atr", "drawdown126_atr", "range_position63",
            "atr_pct", "volatility20", "month_sin", "month_cos")
GROUPS = {"trend": [0, 1, 2, 3, 4], "stress": [5, 6],
          "volatility": [7, 8], "seasonality": [9, 10]}


def _feature(rows, index):
    history = rows[max(0, index-200):index+1]
    atr = _atr(history)
    if index < 200 or not atr or atr <= 0:
        return None
    closes = [row["close"] for row in history]
    close = closes[-1]
    recent = history[-63:]
    low, high = min(row["low"] for row in recent), max(row["high"] for row in recent)
    returns = [math.log(b)-math.log(a) for a, b in zip(closes[-21:-1], closes[-20:])]
    month = date.fromisoformat(rows[index]["date"]).month
    features = [
        (close-closes[-6])/atr, (close-closes[-21])/atr,
        *[(close-float(np.mean(closes[-n:])))/atr for n in (20, 60, 200)],
        (close-max(closes[-126:]))/atr, (close-low)/(high-low) if high > low else .5,
        atr/close, float(np.std(returns)),
        math.sin(2*math.pi*month/12), math.cos(2*math.pi*month/12),
    ]
    return features if all(math.isfinite(value) for value in features) else None


def forecast_swing_range(rows, origin_date, *, horizon_sessions=20, training_sessions=1260,
                         neighbors=80, embargo_sessions=5, group_weights=None):
    """Forecast typical and stress price ranges from preceding similar market states.

    Training targets are actual future lows/highs normalized by the ATR known at
    each historical origin. Their entire windows must end before the embargo at
    the forecast origin. Long averages never use future pivot confirmation.
    Feature families share one budget, so five trend measures do not get five
    times the authority of the volatility family.
    """
    day = date.fromisoformat(origin_date)
    for key, value, lower, upper in (
            ("horizon", horizon_sessions, 1, 126), ("training", training_sessions, 60, 2520),
            ("neighbors", neighbors, 20, 504), ("embargo", embargo_sessions, 1, 63)):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"Invalid {key} session count.")
    weights = dict.fromkeys(GROUPS, 1.0)
    if group_weights is not None:
        if not isinstance(group_weights, dict) or set(group_weights) != set(GROUPS):
            raise ValueError("Exactly the four declared feature-family weights are required.")
        weights = dict(group_weights)
    if (any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 10
            for v in weights.values()) or sum(weights.values()) <= 0):
        raise ValueError("Feature weights must be finite, nonnegative and not all zero.")
    # Inspect dates before any numeric fields: later prices cannot influence the
    # forecast or even be read through an invalid-future-data fallback.
    prefix = [row for row in rows if isinstance(row, dict)
              and isinstance(row.get("date"), str) and row["date"] <= origin_date]
    clean = _clean(prefix, origin_date)
    audit = {"origin_date": day.isoformat(), "horizon_sessions": horizon_sessions,
             "training_sessions": training_sessions, "neighbors_requested": neighbors,
             "embargo_sessions": embargo_sessions, "feature_names": list(FEATURES),
             "group_weights": weights, "equity_up_probability": None,
             "goal_completion_proven": False, "historical_pit_verified": False}
    if (len(clean) != len(prefix) or len(clean) < 260 or clean[-1]["date"] != origin_date
            or any(row.get("session_complete") is not True for row in clean)
            or any(not isinstance(row.get("price_basis"), str)
                   or not row["price_basis"].strip()
                   or row["price_basis"].strip().lower() in {"unknown", "none", "n/a", "unverified"}
                   or "未注明" in row["price_basis"] or "unknown" in row["price_basis"].lower()
                   for row in clean)
            or len({row["price_basis"] for row in clean}) != 1):
        return {**audit, "status": "unavailable", "reason": "invalid_or_insufficient_price_history"}
    try:
        source_hash = _hash(prefix)
    except (ValueError, TypeError):
        return {**audit, "status": "unavailable", "reason": "noncanonical_source_record"}
    scheduled, cursor = [], date.fromisoformat(clean[0]["date"])
    while cursor <= day:
        if is_us_equity_session(cursor):
            scheduled.append(cursor.isoformat())
        cursor += timedelta(days=1)
    if scheduled != [row["date"] for row in clean]:
        return {**audit, "status": "unavailable", "reason": "missing_complete_session"}
    feature_now, atr_now = _feature(clean, len(clean)-1), _atr(clean)
    if feature_now is None or not atr_now:
        return {**audit, "status": "unavailable", "reason": "zero_or_missing_current_atr"}
    last = len(clean)-1
    final_origin = last-horizon_sessions-embargo_sessions
    first_origin = max(200, last-training_sessions)
    training = []
    for index in range(first_origin, final_origin+1):
        feature = _feature(clean, index)
        atr = _atr(clean[:index+1])
        if feature is None or not atr:
            return {**audit, "status": "unavailable", "reason": "nonfinite_historical_feature"}
        future = clean[index+1:index+horizon_sessions+1]
        close = clean[index]["close"]
        training.append({"index": index, "origin_date": clean[index]["date"],
                         "label_available_date": future[-1]["date"], "x": feature,
                         "low": (min(row["low"] for row in future)-close)/atr,
                         "high": (max(row["high"] for row in future)-close)/atr,
                         "up": int(future[-1]["close"] > close)})
        if any(not math.isfinite(training[-1][key]) for key in ("low", "high")):
            return {**audit, "status": "unavailable", "reason": "nonfinite_training_target"}
    if len(training) < neighbors:
        return {**audit, "status": "insufficient_training",
                "mature_training_samples": len(training)}
    matrix = np.asarray([row["x"] for row in training])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        center, scale = matrix.mean(axis=0), matrix.std(axis=0)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        return {**audit, "status": "unavailable", "reason": "nonfinite_training_scale"}
    scale[scale < 1e-10] = 1
    current = np.asarray(feature_now)
    metric = np.zeros(len(FEATURES))
    for group, positions in GROUPS.items():
        metric[positions] = weights[group]/len(positions)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        distance = np.sum(np.clip((matrix-current)/scale, -8, 8)**2*metric, axis=1)
    if not np.isfinite(distance).all():
        return {**audit, "status": "unavailable", "reason": "nonfinite_neighbor_distance"}
    chosen_indices = sorted(range(len(training)), key=lambda i: (
        float(distance[i]), training[i]["origin_date"]))[:neighbors]
    chosen = [training[i] for i in chosen_indices]
    atr, close = _atr(clean), clean[-1]["close"]
    low = np.quantile([row["low"] for row in chosen], [.1, .5, .9], method="linear")
    high = np.quantile([row["high"] for row in chosen], [.1, .5, .9], method="linear")
    prices = {"stress_low": close+atr*float(low[0]), "typical_low": close+atr*float(low[1]),
              "low_upper_quantile": close+atr*float(low[2]),
              "high_lower_quantile": close+atr*float(high[0]),
              "typical_high": close+atr*float(high[1]), "stretch_high": close+atr*float(high[2])}
    prices = {key: value if math.isfinite(value) and value > 0 else None
              for key, value in prices.items()}
    proof = [{key: row[key] for key in ("origin_date", "label_available_date", "low", "high", "up")}
             for row in chosen]
    current_z = np.clip((current-center)/scale, -8, 8)
    return {**audit, "status": "ready", "close": close, "atr": atr,
            "price_basis": clean[-1]["price_basis"], "prices": prices,
            "feature_values": dict(zip(FEATURES, current.tolist())),
            "standardized_current_features": current_z.tolist(),
            "training_origin_first": training[0]["origin_date"],
            "training_origin_last": training[-1]["origin_date"],
            "training_label_last": max(row["label_available_date"] for row in training),
            "mature_training_samples": len(training), "neighbors_used": neighbors,
            "raw_neighbor_up_frequency": sum(row["up"] for row in chosen)/neighbors,
            "raw_frequency_is_calibrated_probability": False,
            "neighbor_proof": proof, "used_history_sha256": source_hash,
            "usable_take_profit": prices["typical_high"] if prices["typical_high"]
            and prices["typical_high"] > close+atr*.5 else None,
            "limitations": [
                "Historical neighbor label windows overlap; cases are not independent.",
                "Quantiles need chronological validation; they are not guaranteed bounds.",
                "This price/calendar model excludes unverified historical event features.",
                "A missing forward target is not raised artificially above current price."]}
