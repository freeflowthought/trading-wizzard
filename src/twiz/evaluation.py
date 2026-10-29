"""Chronological directional research, separate from executable trading allocations.

This is an explicit proxy experiment, not a reconstruction of either author's
undisclosed system. Test labels never select hyperparameters or thresholds.
NumPy is an evaluator-only runtime requirement; no provider calls occur here.
"""

import bisect
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from .sessions import is_us_equity_session

FEATURES = (
    "return1", "return5", "return20", "return60", "close_sma20", "close_sma50",
    "close_sma200", "sma20_sma50", "sma50_sma200", "volatility20", "atr14_pct",
    "drawdown126", "rsi14", "prior20_breakout", "support_reclaim", "relative_spy20",
    "semis_minus_software20", "rotation_missing", "yield10", "yield_change20",
    "yield_missing", "hy_spread", "hy_change20", "hy_missing", "oil_return20",
    "oil_missing",
)
MODEL_VERSION = "eagle-shunge-proxy-ridge-logistic-v1"
BASELINE_RULES = (
    "DGS10>=5 or HY>=5: down; else close>SMA20>SMA50: up; else "
    "DGS10<5 and HY<4 and drawdown126<=-8% and 5-session-support reclaim: up; "
    "else close<SMA20<SMA50: down; else close>=SMA20: up, otherwise down. "
    "Missing macro does not assert stress or authorize the dip exception."
)


@dataclass(frozen=True)
class EvaluationProtocol:
    evaluation_start: str = "2026-04-27"
    evaluation_end: str = "2026-10-27"
    primary_horizon_sessions: int = 5
    secondary_horizons_sessions: Tuple[int, ...] = (1, 20)
    primary_tickers: Tuple[str, ...] = ("QQQ", "SPY")
    individual_tickers: Tuple[str, ...] = ()
    window_sessions: int = 1260
    refit_every_sessions: int = 5
    embargo_sessions: int = 5
    pretest_validation_sessions: int = 63
    penalties: Tuple[float, ...] = (0.1, 1.0, 10.0)
    stock_direction_thresholds: Tuple[float, ...] = (0.45, 0.5, 0.55, 0.6)
    paired_date_blocks_sessions: int = 5
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 20261027
    minimum_paired_gain_percentage_points: float = 3.0
    lower_confidence_gain_must_exceed: float = 0.0
    minimum_positive_month_fraction: float = 2 / 3
    minimum_complete_cohort_coverage: float = 1.0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EvaluationProtocol":
        for key, supported in (
            ("prediction_time", "after_completed_US_regular_session"),
            ("label", "close_to_close_direction"), ("flat_return_label", "up"),
        ):
            if key in config and config[key] != supported:
                raise ValueError(f"Unsupported {key}; this evaluator requires {supported}.")
        merged = {**config, **config.get("training", {}), **config.get("scoring", {}),
                  **config.get("target", {})}
        fields = cls.__dataclass_fields__
        values = {key: merged[key] for key in fields if key in merged}
        for key in ("secondary_horizons_sessions", "primary_tickers", "individual_tickers",
                    "penalties", "stock_direction_thresholds"):
            if key in values:
                values[key] = tuple(values[key])
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if date.fromisoformat(self.evaluation_start) > date.fromisoformat(self.evaluation_end):
            raise ValueError("Evaluation start must precede its end.")
        for value in (self.primary_horizon_sessions, *self.secondary_horizons_sessions,
                      self.window_sessions, self.refit_every_sessions,
                      self.pretest_validation_sessions, self.paired_date_blocks_sessions,
                      self.bootstrap_resamples):
            if type(value) is not int or value <= 0:
                raise ValueError("Session counts must be positive integers.")
        if type(self.embargo_sessions) is not int or self.embargo_sessions < 0:
            raise ValueError("Embargo must be a nonnegative session count.")
        if self.pretest_validation_sessions < 12:
            raise ValueError("At least twelve pretest validation sessions are required.")
        if (not self.penalties
                or any(type(x) not in (int, float) or _num(x) is None or x <= 0
                       for x in self.penalties)
                or not self.stock_direction_thresholds
                or any(type(x) not in (int, float) or _num(x) is None or not 0 < x < 1
                       for x in self.stock_direction_thresholds)):
            raise ValueError("Regularization and direction thresholds must be finite and valid.")
        if (not self.primary_tickers
                or len(set(self.primary_tickers)) != len(self.primary_tickers)
                or len(set(self.individual_tickers)) != len(self.individual_tickers)
                or set(self.primary_tickers) & set(self.individual_tickers)):
            raise ValueError("Predeclared cohorts must be nonempty, distinct and disjoint.")


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _calendar(start: str, end: str) -> List[str]:
    day, last, result = date.fromisoformat(start), date.fromisoformat(end), []
    while day <= last:
        if is_us_equity_session(day):
            result.append(day.isoformat())
        day += timedelta(days=1)
    return result


def _prices(market: Dict[str, List[dict]], end: str) -> Dict[str, Dict[str, dict]]:
    result = {}
    for ticker, rows in market.items():
        clean, duplicates = {}, set()
        for row in rows:
            stamp = str(row.get("date", ""))
            try:
                valid_date = date.fromisoformat(stamp)
            except ValueError:
                continue
            if stamp > end or not is_us_equity_session(valid_date):
                continue
            values = {key: _num(row.get(key)) for key in ("open", "high", "low", "close")}
            volume = _num(row.get("volume"))
            if (row.get("source") != "yfinance" or row.get("session_complete") is not True
                    or not row.get("price_basis") or volume is None or volume < 0
                    or any(v is None or v <= 0 for v in values.values())
                    or not values["low"] <= min(values["open"], values["close"])
                    <= max(values["open"], values["close"]) <= values["high"]):
                continue
            if stamp in clean:
                duplicates.add(stamp)
            clean[stamp] = {**values, "date": stamp, "price_basis": row.get("price_basis")}
        result[ticker] = {day: row for day, row in clean.items() if day not in duplicates}
    return result


def _publication_clock(value: Any) -> Optional[datetime]:
    try:
        if isinstance(value, str) and len(value) == 10:
            return datetime.combine(date.fromisoformat(value), time.max,
                                    tzinfo=ZoneInfo("America/New_York"))
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp.astimezone(timezone.utc) if stamp.tzinfo else None
    except (TypeError, ValueError):
        return None


def _macro_index(rows: List[dict]) -> Dict[str, dict]:
    observations = {}
    for row in rows:
        series = row.get("series_id")
        source = str(row.get("source", ""))
        if not series or not source.startswith(("FRED", "U.S. Treasury")):
            continue
        for item in row.get("history", []):
            stamp, number = str(item.get("date", "")), _num(item.get("value"))
            try:
                date.fromisoformat(stamp)
            except ValueError:
                continue
            available = next((source[key] for source in (item, row)
                              for key in ("available_at", "released_at", "published_at", "known_at")
                              if source.get(key) is not None), None)
            known = _publication_clock(available) if available is not None else None
            if number is None or available is not None and known is None:
                continue
            observations.setdefault(series, {})[(stamp, known)] = number
    indexed = {}
    for series, versions in observations.items():
        explicit_dates = {stamp for stamp, known in versions if known is not None}
        versions = {key: value for key, value in versions.items()
                    if key[1] is not None or key[0] not in explicit_dates}
        ordered = sorted(versions, key=lambda key: (
            key[0], key[1] or datetime.min.replace(tzinfo=timezone.utc)))
        indexed[series] = {"days": [key[0] for key in ordered],
                           "values": [versions[key] for key in ordered],
                           "known_at": [key[1] for key in ordered]}
    return indexed


def _macro_at(indexed: dict, cutoff: str, available_by: Optional[datetime] = None) -> dict:
    available_by = available_by or datetime.combine(date.fromisoformat(cutoff), time.max,
                                                   tzinfo=ZoneInfo("America/New_York"))
    if available_by.tzinfo is None:
        raise ValueError("Macro availability cutoff requires a timezone.")
    result = {"cutoff": cutoff, "available_by": available_by.isoformat()}
    for series, key in (("DGS10", "yield"), ("BAMLH0A0HYM2", "hy"),
                        ("DCOILBRENTEU", "oil")):
        records = indexed.get(series, {"days": [], "values": [], "known_at": []})
        stop = bisect.bisect_right(records["days"], cutoff)
        eligible = {}
        for stamp, value, known in zip(records["days"][:stop], records["values"][:stop],
                                        records["known_at"][:stop]):
            if known is None or known <= available_by:
                eligible[stamp] = (value, known)
        days = sorted(eligible)
        values = [eligible[day][0] for day in days]
        pos = len(days)-1
        current = values[pos] if pos >= 0 else None
        if pos < 0 or (date.fromisoformat(cutoff)-date.fromisoformat(days[pos])).days > 7:
            current = None
        result[key] = current
        result[key+"_date"] = days[pos] if current is not None else None
        known = eligible[days[pos]][1] if current is not None else None
        result[key+"_known_at"] = known.isoformat() if known is not None else None
        result[key+"_availability_basis"] = ("explicit_publication_time" if known else
                                             "revised_history_with_one_session_lag_assumption")
        result[key+"_prior_date"] = days[pos-20] if current is not None and pos >= 20 else None
        result[key+"_change20"] = (
            current-values[pos-20] if current is not None and pos >= 20 else None)
        if key == "oil":
            result[key+"_change20"] = (
                current/values[pos-20]-1 if current is not None and pos >= 20
                and values[pos-20] > 0 else None)
    return result


def _return(prices: dict, ticker: str, days: List[str], i: int, n: int) -> Optional[float]:
    rows = prices.get(ticker, {})
    now, previous = rows.get(days[i]), rows.get(days[i-n]) if i >= n else None
    return now["close"]/previous["close"]-1 if now and previous else None


def _features(prices: dict, ticker: str, days: List[str], i: int, macro: dict) -> Optional[dict]:
    if i < 200:
        return None
    rows = prices.get(ticker, {})
    history = [rows.get(day) for day in days[i-200:i+1]]
    if any(row is None for row in history) or len({r["price_basis"] for r in history}) != 1:
        return None
    closes = np.array([r["close"] for r in history])
    close = float(closes[-1])
    sma20, sma50, sma200 = [float(closes[-n:].mean()) for n in (20, 50, 200)]
    returns = np.diff(closes[-21:])/closes[-21:-1]
    changes = np.diff(closes[-15:])
    gain, loss = np.maximum(changes, 0).mean(), np.maximum(-changes, 0).mean()
    rsi = 0.5 if gain == loss == 0 else gain/(gain+loss)
    atr = np.mean([max(r["high"]-r["low"], abs(r["high"]-p["close"]),
                       abs(r["low"]-p["close"])) for r, p in zip(history[-14:], history[-15:-1])])
    support = min(r["low"] for r in history[-6:-1])
    reclaim = float(history[-1]["low"] <= support and close > support)
    ret20 = close/closes[-21]-1
    spy = _return(prices, "SPY", days, i, 20)
    semi, software = [_return(prices, asset, days, i, 20) for asset in ("SMH", "IGV")]
    rotation = semi-software if semi is not None and software is not None else None
    values = [close/closes[-1-n]-1 for n in (1, 5, 20, 60)] + [
        close/sma20-1, close/sma50-1, close/sma200-1, sma20/sma50-1, sma50/sma200-1,
        float(returns.std()), float(atr/close), close/max(r["high"] for r in history[-126:])-1,
        float(rsi), close/max(r["high"] for r in history[-21:-1])-1, reclaim,
        ret20-spy if spy is not None else 0, rotation or 0, float(rotation is None),
        macro["yield"] or 0, macro["yield_change20"] or 0, float(macro["yield"] is None),
        macro["hy"] or 0, macro["hy_change20"] or 0, float(macro["hy"] is None),
        macro["oil_change20"] or 0, float(macro["oil_change20"] is None),
    ]
    if not all(math.isfinite(value) for value in values):
        return None
    return {"x": values, "close": close, "sma20": sma20, "sma50": sma50,
            "macro": macro, "history_start": history[0]["date"],
            "history_end": history[-1]["date"], "price_basis": history[-1]["price_basis"]}


def _shunge_proxy(feature: Optional[dict]) -> Optional[int]:
    if feature is None:
        return None
    close, short, long = feature["close"], feature["sma20"], feature["sma50"]
    rate, hy = feature["macro"]["yield"], feature["macro"]["hy"]
    if (rate is not None and rate >= 5) or (hy is not None and hy >= 5):
        return 0
    if close > short > long:
        return 1
    if (rate is not None and rate < 5 and hy is not None and hy < 4
            and feature["x"][11] <= -0.08 and feature["x"][14] == 1):
        return 1
    if close < short < long:
        return 0
    return int(close >= short)


def _sigmoid(value):
    return 1/(1+np.exp(-np.clip(value, -30, 30)))


def _logistic(x: np.ndarray, y: np.ndarray, penalty: float) -> dict:
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-10] = 1
    design = np.column_stack((np.ones(len(x)), np.clip((x-mean)/scale, -8, 8)))
    beta = np.zeros(design.shape[1])
    beta[0] = math.log((y.sum()+0.5)/(len(y)-y.sum()+0.5))
    ridge = np.eye(len(beta))*penalty
    ridge[0, 0] = 1e-8
    for _iteration in range(25):
        probability = _sigmoid(design@beta)
        gradient = design.T@(probability-y)+ridge@beta
        hessian = (design.T*(probability*(1-probability)))@design+ridge
        step = np.linalg.solve(hessian, gradient)
        step *= min(1.0, 5/max(float(np.max(np.abs(step))), 1e-8))
        beta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coefficients": beta.tolist(),
            "penalty": penalty, "iterations": _iteration+1, "training_samples": len(y)}


def _probability(model: dict, x: List[float]) -> float:
    z = np.clip((np.asarray(x)-model["mean"])/model["scale"], -8, 8)
    return float(_sigmoid(model["coefficients"][0]+z@model["coefficients"][1:]))


def _calibrate(probabilities: list, labels: list) -> dict:
    if len(labels) < 20 or len(set(labels)) < 2:
        return {"available": False, "slope": 1.0, "intercept": 0.0,
                "reason": "Too few calibration cases or only one outcome class."}
    logits = np.array([math.log(p/(1-p)) for p in probabilities]).reshape(-1, 1)
    fitted = _logistic(logits, np.asarray(labels), 1.0)
    slope = fitted["coefficients"][1]/fitted["scale"][0]
    intercept = fitted["coefficients"][0]-slope*fitted["mean"][0]
    # A negative slope reverses the learned model; don't silently turn this into another strategy.
    if slope <= 0:
        return {"available": False, "slope": 1.0, "intercept": 0.0,
                "reason": "Calibration relationship not monotone; raw probabilities retained."}
    return {"available": True, "slope": slope, "intercept": intercept,
            "samples": len(labels), "method": "pretest_heldout_Platt",
            "transform_model": fitted, "standardized_input_clip": [-8, 8]}


def _apply_calibrator(probability: float, calibrator: dict) -> float:
    logit = math.log(probability/(1-probability))
    if calibrator.get("transform_model"):
        return _probability(calibrator["transform_model"], [logit])
    return float(_sigmoid(calibrator["intercept"]+calibrator["slope"]*logit))


def _outcome_label(current: Optional[dict], future: Optional[dict]) -> Optional[int]:
    if not current or not future:
        return None
    first, last = _num(current.get("close")), _num(future.get("close"))
    if (first is None or last is None or min(first, last) <= 0
            or not current.get("price_basis")
            or current.get("price_basis") != future.get("price_basis")):
        return None
    return int(last >= first)


def _training(features: dict, prices: dict, days: list, tickers: tuple, fit_i: int,
              horizon: int, protocol: EvaluationProtocol) -> tuple:
    last = fit_i-horizon-protocol.embargo_sessions
    samples, labels, origins, ends = [], [], [], []
    for i in range(max(200, fit_i-protocol.window_sessions), last+1):
        for ticker in tickers:
            feature = features.get((ticker, i))
            row, future = prices.get(ticker, {}).get(days[i]), prices.get(ticker, {}).get(
                days[i+horizon])
            label = _outcome_label(row, future)
            if feature and label is not None:
                samples.append(feature["x"])
                labels.append(label)
                origins.append(days[i])
                ends.append(days[i+horizon])
    return np.asarray(samples), np.asarray(labels), origins, ends


def _fit(features: dict, prices: dict, days: list, tickers: tuple, fit_i: int,
         horizon: int, protocol: EvaluationProtocol, penalty: float) -> Optional[dict]:
    x, y, origins, ends = _training(features, prices, days, tickers, fit_i, horizon, protocol)
    if len(y) < max(50, len(FEATURES)*2) or len(set(y.tolist())) < 2:
        return None
    model = _logistic(x, y, penalty)
    model.update(fit_date=days[fit_i], training_origin_start=min(origins),
                 training_origin_cutoff=max(origins), training_label_cutoff=max(ends),
                 embargo_sessions=protocol.embargo_sessions, horizon_sessions=horizon,
                 feature_names=list(FEATURES), tickers=list(tickers))
    return model


def _validate(features: dict, prices: dict, days: list, tickers: tuple, start_i: int,
              horizon: int, protocol: EvaluationProtocol) -> dict:
    end_i = start_i-horizon-protocol.embargo_sessions-1
    first_i = end_i-protocol.pretest_validation_sessions+1
    result = {"horizon": horizon, "tickers": list(tickers), "start": days[max(first_i, 0)],
              "end": days[max(end_i, 0)], "candidates": [], "selected": None}
    if first_i < 201:
        result["reason"] = "Insufficient preceding history for untouched pretest validation."
        return result
    split_i = first_i+protocol.pretest_validation_sessions*2//3
    calibration_end = split_i-horizon-protocol.embargo_sessions-1
    best = None
    for penalty in protocol.penalties:
        model = _fit(features, prices, days, tickers, first_i, horizon, protocol, penalty)
        if model is None:
            continue
        calibration_p, calibration_y, selection = [], [], []
        for i in range(first_i, end_i+1):
            for ticker in tickers:
                feature = features.get((ticker, i))
                row, future = prices.get(ticker, {}).get(days[i]), prices.get(ticker, {}).get(
                    days[i+horizon])
                label = _outcome_label(row, future)
                if feature and label is not None:
                    p = _probability(model, feature["x"])
                    y = label
                    if i <= calibration_end:
                        calibration_p.append(p)
                        calibration_y.append(y)
                    elif i >= split_i:
                        selection.append((p, y))
        calibrator = _calibrate(calibration_p, calibration_y)
        for threshold in protocol.stock_direction_thresholds:
            correct = sum(int(_apply_calibrator(p, calibrator) >= threshold) == y
                          for p, y in selection)
            accuracy = correct/len(selection) if selection else None
            candidate = {"penalty": penalty, "threshold": threshold, "accuracy": accuracy,
                         "selection_cases": len(selection), "calibrator": calibrator,
                         "training_label_cutoff": model["training_label_cutoff"]}
            result["candidates"].append(candidate)
            # Stable ties prefer the threshold closest to .5, then the stronger penalty.
            rank = (accuracy or -1, -abs(threshold-0.5), penalty)
            if selection and (best is None or rank > best[0]):
                best = (rank, candidate)
    if best:
        result["selected"] = best[1]
    else:
        result["reason"] = "No usable chronological pretest selection cases."
    result["selection_start"] = days[split_i]
    result["calibration_label_cutoff"] = days[calibration_end+horizon]
    result["latest_label_date"] = days[end_i+horizon]
    return result


def _metrics(rows: list, protocol: EvaluationProtocol, horizon: int) -> dict:
    known = [r for r in rows if r["actual_up"] is not None]
    count = len(known)
    result = {"scheduled_cases": len(rows), "known_outcomes": count,
              "prediction_cases": sum(r["prediction_up"] is not None for r in rows),
              "outcome_coverage": count/len(rows) if rows else 0,
              "prediction_coverage": sum(r["prediction_up"] is not None for r in rows)/len(rows)
              if rows else 0}
    if not known:
        return {**result, "status": "unavailable", "improvement_gate": False}
    def correct(row, key):
        return int(row[key] is not None and row[key] == row["actual_up"])

    accuracy = sum(correct(r, "prediction_up") for r in known)/count
    proxy = sum(correct(r, "shunge_proxy_up") for r in known)/count
    always = sum(r["actual_up"] for r in known)/count
    daily = {}
    months = {}
    for row in known:
        delta = correct(row, "prediction_up")-correct(row, "shunge_proxy_up")
        daily.setdefault(row["origin_date"], []).append(delta)
        months.setdefault(row["origin_date"][:7], []).append(delta)
    ordered = [daily[key] for key in sorted(daily)]
    block = max(protocol.paired_date_blocks_sessions, horizon)
    rng = np.random.default_rng(protocol.bootstrap_seed+horizon)
    values = []
    for _ in range(protocol.bootstrap_resamples):
        sample = []
        while len(sample) < len(ordered):
            start = int(rng.integers(0, len(ordered)))
            sample.extend(ordered[(start+j) % len(ordered)] for j in range(block))
        flat = [value for day in sample[:len(ordered)] for value in day]
        values.append(sum(flat)/len(flat))
    low, high = [float(x) for x in np.quantile(values, [0.025, 0.975])]
    folds = [{"month": key, "cases": len(values), "paired_gain": sum(values)/len(values)}
             for key, values in sorted(months.items())]
    positive = sum(fold["paired_gain"] > 0 for fold in folds)/len(folds)
    coverage = sum(r["prediction_up"] is not None and r["shunge_proxy_up"] is not None
                   and r["actual_up"] is not None for r in rows)/len(rows)
    checks = {"paired_gain": accuracy-proxy >=
              protocol.minimum_paired_gain_percentage_points/100,
              "beats_both_baselines": accuracy > max(proxy, always),
              "confidence_interval": low > protocol.lower_confidence_gain_must_exceed,
              "complete_cohort_coverage": coverage >= protocol.minimum_complete_cohort_coverage,
              "positive_months": positive >= protocol.minimum_positive_month_fraction,
              "adequate_dates_and_cases": len(daily) >= 80 and count >= 100 and len(folds) >= 4}
    probabilities = [r for r in known if r["probability_up"] is not None]
    result.update(status="evaluated", accuracy=accuracy, shunge_proxy_accuracy=proxy,
                  always_up_accuracy=always, paired_gain=accuracy-proxy,
                  gain_vs_always_up=accuracy-always, paired_gain_ci95=[low, high],
                  bootstrap_block_sessions=block, monthly_folds=folds,
                  positive_month_fraction=positive, complete_cohort_coverage=coverage,
                  brier_score=sum((r["probability_up"]-r["actual_up"])**2
                                  for r in probabilities)/len(probabilities)
                  if probabilities else None,
                  checks=checks, improvement_gate=all(checks.values()),
                  beats_both_baselines=accuracy > max(proxy, always))
    return result


def run_walk_forward_evaluation(
    market_data: Dict[str, List[dict]], macro_history: Optional[List[dict]] = None,
    protocol: Optional[EvaluationProtocol] = None, archived_options: Optional[Any] = None,
    archived_prediction_markets: Optional[Any] = None,
) -> dict:
    """Evaluate every mature scheduled origin, retaining missing cases and auditable models."""
    protocol = protocol or EvaluationProtocol()
    protocol.validate()
    prices = _prices(market_data, protocol.evaluation_end)
    starts = [stamp for rows in prices.values() for stamp in rows]
    if not starts:
        raise ValueError("No verified completed daily prices are available.")
    days = _calendar(min(starts), protocol.evaluation_end)
    start_i = bisect.bisect_left(days, protocol.evaluation_start)
    macro_index = _macro_index(macro_history or [])
    macro = {i: _macro_at(macro_index, days[i-1], available_by=datetime.combine(
        date.fromisoformat(days[i]), time(16, 15), tzinfo=ZoneInfo("America/New_York")))
        for i in range(1, len(days))}
    tickers = protocol.primary_tickers+protocol.individual_tickers
    features = {(ticker, i): value for ticker in tickers for i in range(200, len(days))
                if (value := _features(prices, ticker, days, i, macro[i])) is not None}
    records, models, validation, cohorts = [], [], [], {}
    horizons = tuple(dict.fromkeys((protocol.primary_horizon_sessions,
                                   *protocol.secondary_horizons_sessions)))
    for group, universe in (("primary", protocol.primary_tickers),
                            ("individual", protocol.individual_tickers)):
        cohort = {"tickers": list(universe), "horizons": {}}
        for horizon in horizons:
            if not universe:
                continue
            validated = _validate(features, prices, days, universe, start_i, horizon, protocol)
            validated["cohort"] = group
            validation.append(validated)
            selected = validated["selected"]
            model, model_id, rows = None, None, []
            for i in range(start_i, len(days)-horizon):
                if (i-start_i) % protocol.refit_every_sessions == 0 and selected:
                    model = _fit(features, prices, days, universe, i, horizon, protocol,
                                 selected["penalty"])
                    model_id = f"{group}-{horizon}-{days[i]}" if model else None
                    if model:
                        models.append({"id": model_id, "cohort": group, **model,
                                       "threshold": selected["threshold"],
                                       "calibrator": selected["calibrator"]})
                for ticker in universe:
                    feature = features.get((ticker, i))
                    current = prices.get(ticker, {}).get(days[i])
                    future = prices.get(ticker, {}).get(days[i+horizon])
                    probability, prediction = None, None
                    if feature and model and selected:
                        probability = _apply_calibrator(_probability(model, feature["x"]),
                                                        selected["calibrator"])
                        prediction = int(probability >= selected["threshold"])
                    actual = _outcome_label(current, future)
                    missing = []
                    if feature is None:
                        missing.append("No complete consistent 201-session feature history.")
                    if model is None:
                        missing.append("No preceding validated fitted model.")
                    if actual is None:
                        missing.append("Missing origin or exact horizon-session outcome close.")
                    row = {"cohort": group, "ticker": ticker, "horizon_sessions": horizon,
                           "origin_date": days[i], "outcome_date": days[i+horizon],
                           "origin_close": current["close"] if current else None,
                           "outcome_close": future["close"] if future else None,
                           "actual_up": actual, "prediction_up": prediction,
                           "probability_up": probability, "shunge_proxy_up": _shunge_proxy(feature),
                           "always_up": 1, "model_id": model_id,
                           "training_origin_cutoff": model["training_origin_cutoff"]
                           if model else None,
                           "training_label_cutoff": model["training_label_cutoff"]
                           if model else None,
                           "feature_values": dict(zip(FEATURES, feature["x"])) if feature else {},
                           "input_dates": {"price_start": feature["history_start"],
                                           "price_end": feature["history_end"],
                                           "macro": feature["macro"]} if feature else {},
                           "blocked_missing_data": missing,
                           "historical_options_used": False,
                           "historical_event_probabilities_used": False}
                    rows.append(row)
            cohort["horizons"][str(horizon)] = _metrics(rows, protocol, horizon)
            records.extend(rows)
        cohort["primary_horizon"] = cohort["horizons"].get(str(protocol.primary_horizon_sessions))
        cohorts[group] = cohort
    passed = [cohorts[group]["primary_horizon"] for group in cohorts
              if cohorts[group]["primary_horizon"] is not None]
    specification = {"version": MODEL_VERSION, "features": FEATURES,
                     "baseline": BASELINE_RULES, "protocol": asdict(protocol)}
    return {"status": "proxy_research_gate_passed" if passed and all(
                m["improvement_gate"] for m in passed)
            else "target_not_met", "protocol": asdict(protocol),
            "goal_completion_proven": False,
            "goal_status": "unproven_author_baseline_and_point_in_time_evidence",
            "specification_sha256": hashlib.sha256(json.dumps(specification, sort_keys=True)
                                                   .encode()).hexdigest(),
            "engine_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_version": MODEL_VERSION, "baseline_rules": BASELINE_RULES,
            **cohorts, "predictions": records, "models": models, "validation": validation,
            "reference_claims": {"index_accuracy": 0.72, "individual_accuracy": 0.68,
                                 "status": "user_reported_unverified_not_same_protocol",
                                 "aspirational_thresholds": {"index": 0.75, "individual": 0.71}},
            "input_availability": {
                "prices": "downloaded_reconstructed_history_not_contemporaneous_archive",
                "macro": "revised_history_assumed_available_one_completed_session_later",
                "options": "not_used_no_validated_point_in_time_adapter",
                "prediction_markets": "not_used_no_validated_point_in_time_adapter",
                "supplied_option_archive": archived_options is not None,
                "supplied_event_archive": archived_prediction_markets is not None},
            "limitations": [
                "The fixed Shunge proxy is not the author's actual recorded forecast accuracy.",
                "Fixed present-day watchlist cohort has survivorship and selection limitations.",
                "Revised macro and reconstructed split-adjusted history are not strict PIT.",
                "Current options/probabilities never backfill missing historical observations.",
                "Pretest calibration is empirical, not a guarantee of probability reliability.",
                "Directional accuracy is not return; commissions and allocations are separate."]}
