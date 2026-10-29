"""Research v2: chronological selection and rolling OOF calibration, never live orders.

The v1 half-year result was already observed. A v2 half-year run is an adaptive
retest, not an untouched holdout or proof of the authors' reported win rates.
"""

import bisect
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from . import evaluation as v1

VERSION = "eagle-shunge-research-v2"
EPSILON = 1e-12


@dataclass(frozen=True)
class EvaluationV2Protocol(v1.EvaluationProtocol):
    pretest_validation_sessions: int = 126
    stock_direction_thresholds: Tuple[float, ...] = (0.5,)
    candidate_windows_sessions: Tuple[int, ...] = (252, 504, 1260)
    mean_loss_penalties: Tuple[float, ...] = (0.001, 0.01)
    calibration_window_sessions: int = 126
    prior_window_sessions: int = 1260

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EvaluationV2Protocol":
        # Keep timing/label semantics and shared validation identical to the frozen v1 proxy.
        v1.EvaluationProtocol.from_config(config)
        merged = {**config, **config.get("training", {}), **config.get("scoring", {}),
                  **config.get("target", {})}
        fields = cls.__dataclass_fields__
        values = {key: merged[key] for key in fields if key in merged}
        for key in ("secondary_horizons_sessions", "primary_tickers", "individual_tickers",
                    "penalties", "stock_direction_thresholds", "candidate_windows_sessions",
                    "mean_loss_penalties"):
            if key in values:
                values[key] = tuple(values[key])
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        super().validate()
        if self.stock_direction_thresholds != (0.5,):
            raise ValueError("V2 freezes the direction threshold at .5.")
        if self.pretest_validation_sessions % 3:
            raise ValueError("Validation must split into three equal chronological blocks.")
        for value in (*self.candidate_windows_sessions, self.calibration_window_sessions,
                      self.prior_window_sessions):
            if type(value) is not int or value <= 0:
                raise ValueError("Candidate, prior and calibration windows must be positive.")
        if (not self.candidate_windows_sessions or not self.mean_loss_penalties
                or any(type(value) not in (int, float) or v1._num(value) is None or value <= 0
                       for value in self.mean_loss_penalties)):
            raise ValueError("V2 requires finite positive mean-loss penalties and windows.")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _source_hashes():
    return {"engine": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "support": hashlib.sha256(Path(v1.__file__).read_bytes()).hexdigest(),
            "sessions": hashlib.sha256(Path(v1.is_us_equity_session.__code__.co_filename)
                                       .read_bytes()).hexdigest()}


def _prepare(market, macro_rows, protocol, selection_only=False):
    end = ((date.fromisoformat(protocol.evaluation_start)-timedelta(days=1)).isoformat()
           if selection_only else protocol.evaluation_end)
    prices = v1._prices(market, end)
    starts = [stamp for rows in prices.values() for stamp in rows]
    if not starts:
        raise ValueError("No verified completed daily price evidence is available.")
    days = v1._calendar(min(starts), protocol.evaluation_start if selection_only else end)
    start_i = bisect.bisect_left(days, protocol.evaluation_start)
    indexed = v1._macro_index(macro_rows or [])
    features = {}
    for i in range(200, len(days)):
        forecast = datetime.combine(date.fromisoformat(days[i]), time(16, 15),
                                    tzinfo=ZoneInfo("America/New_York"))
        macro = v1._macro_at(indexed, days[i-1], available_by=forecast)
        for ticker in protocol.primary_tickers+protocol.individual_tickers:
            feature = v1._features(prices, ticker, days, i, macro)
            if feature and all(v1._num(value) is not None for value in feature["x"]):
                features[ticker, i] = feature
    return {"days": days, "prices": prices, "features": features, "start_i": start_i}


def _pretest_input_hash(prepared, protocol):
    """Bind choices to effective dated inputs, independent of later rows or retrieval metadata."""
    cutoff = protocol.evaluation_start
    used = set(protocol.primary_tickers+protocol.individual_tickers+("SPY", "SMH", "IGV"))
    prices = {ticker: {stamp: row for stamp, row in prepared["prices"].get(ticker, {}).items()
                       if stamp < cutoff} for ticker in sorted(used)}
    features = [{"ticker": ticker, "origin_date": prepared["days"][i], **feature}
                for (ticker, i), feature in sorted(prepared["features"].items())
                if prepared["days"][i] < cutoff]
    return _hash({"prices": prices, "features": features,
                  "sessions": [stamp for stamp in prepared["days"] if stamp < cutoff]})


class _Cohort:
    """A date-indexed sample matrix; every fit masks both origin and label availability."""

    def __init__(self, prepared, tickers, horizon, protocol):
        self.days, self.prices = prepared["days"], prepared["prices"]
        self.features, self.tickers = prepared["features"], tickers
        self.horizon, self.protocol = horizon, protocol
        origins, labels, vectors, prior_origins, prior_labels = [], [], [], [], []
        for i in range(max(0, len(self.days)-horizon)):
            for ticker in tickers:
                current = self.prices.get(ticker, {}).get(self.days[i])
                future = self.prices.get(ticker, {}).get(self.days[i+horizon])
                label = v1._outcome_label(current, future)
                if label is None:
                    continue
                prior_origins.append(i)
                prior_labels.append(label)
                feature = self.features.get((ticker, i))
                if feature:
                    origins.append(i)
                    labels.append(label)
                    vectors.append(feature["x"])
        self.ix, self.y, self.x = np.asarray(origins), np.asarray(labels), np.asarray(vectors)
        self.prior_ix, self.prior_y = np.asarray(prior_origins), np.asarray(prior_labels)

    def fit(self, i, config):
        prior = config["kind"] == "prior_only"
        origins, labels = (self.prior_ix, self.prior_y) if prior else (self.ix, self.y)
        last = i-self.horizon-self.protocol.embargo_sessions
        mask = (origins >= i-config["window_sessions"]) & (origins <= last)
        used, y = origins[mask], labels[mask]
        if len(y) < (20 if prior else max(50, len(v1.FEATURES)*2)):
            return None
        if prior:
            model = {"prior_probability": float((y.sum()+0.5)/(len(y)+1)),
                     "training_samples": len(y), "kind": "prior_only"}
        else:
            if len(set(y.tolist())) < 2:
                return None
            penalty = config["mean_loss_penalty"]*len(y)
            model = v1._logistic(self.x[mask], y, penalty)
            model.update(kind="ridge_logistic", mean_loss_penalty=config["mean_loss_penalty"],
                         summed_loss_penalty=penalty)
        model.update(fit_date=self.days[i], training_origin_start=self.days[int(used.min())],
                     training_origin_cutoff=self.days[int(used.max())],
                     training_label_cutoff=self.days[int(used.max())+self.horizon],
                     embargo_sessions=self.protocol.embargo_sessions,
                     horizon_sessions=self.horizon, window_sessions=config["window_sessions"],
                     tickers=list(self.tickers))
        return model


def _prior_config(protocol):
    return {"kind": "prior_only", "window_sessions": protocol.prior_window_sessions,
            "mean_loss_penalty": None, "calibration": "raw"}


def _identity(reason):
    return {"available": False, "slope": 1.0, "intercept": 0.0, "reason": reason}


def _simulate(data, first, last, config, prefix):
    protocol, horizon, days = data.protocol, data.horizon, data.days
    warm = max(0, first-protocol.calibration_window_sessions-horizon
               -protocol.embargo_sessions-protocol.refit_every_sessions)
    warm -= warm % protocol.refit_every_sessions
    history, emitted, models = [], [], []
    model, model_id = None, None
    calibrator = _identity("No prior OOF calibration labels.")
    for i in range(warm, last+1):
        if i % protocol.refit_every_sessions == 0:
            model = data.fit(i, config)
            model_id = f"{prefix}-{days[i]}" if model else None
            mature_end = i-horizon-protocol.embargo_sessions
            mature = [row for row in history
                      if mature_end-protocol.calibration_window_sessions < row["i"] <= mature_end
                      and row["actual_up"] is not None and row["raw_probability"] is not None]
            calibrator = (v1._calibrate([r["raw_probability"] for r in mature],
                                       [r["actual_up"] for r in mature])
                          if config["kind"] != "prior_only" else
                          _identity("Prior-only baseline is not Platt calibrated."))
            calibrator = {**calibrator,
                          "origin_start": days[min(r["i"] for r in mature)] if mature else None,
                          "origin_cutoff": days[max(r["i"] for r in mature)] if mature else None,
                          "label_cutoff": days[max(r["i"] for r in mature)+horizon]
                          if mature else None,
                          "source_model_ids": sorted({r["model_id"] for r in mature
                                                      if r["model_id"]}),
                          "origin_sessions": len({r["i"] for r in mature})}
            if model:
                models.append({"id": model_id, **model, "calibrator": calibrator,
                               "phase": "warmup_oof" if i < first else "prediction",
                               "feature_names": list(v1.FEATURES)
                               if config["kind"] != "prior_only" else []})
        for ticker in data.tickers:
            feature = data.features.get((ticker, i))
            current = data.prices.get(ticker, {}).get(days[i])
            future = data.prices.get(ticker, {}).get(days[i+horizon])
            raw = None
            if model and current:
                raw = model["prior_probability"] if config["kind"] == "prior_only" else (
                    v1._probability(model, feature["x"]) if feature else None)
            calibrated = v1._apply_calibrator(raw, calibrator) if raw is not None else None
            actual = v1._outcome_label(current, future)
            history.append({"i": i, "ticker": ticker, "raw_probability": raw,
                            "actual_up": actual, "model_id": model_id})
            if i >= first:
                emitted.append({"i": i, "ticker": ticker, "origin_date": days[i],
                                "outcome_date": days[i+horizon], "actual_up": actual,
                                "origin_close": current["close"] if current else None,
                                "outcome_close": future["close"] if future else None,
                                "raw_probability": raw, "calibrated_probability": calibrated,
                                "calibration_available": calibrator["available"],
                                "calibration_origin_cutoff": calibrator.get("origin_cutoff"),
                                "calibration_label_cutoff": calibrator.get("label_cutoff"),
                                "model_id": model_id,
                                "training_origin_cutoff": model["training_origin_cutoff"]
                                if model else None,
                                "training_label_cutoff": model["training_label_cutoff"]
                                if model else None})
    return emitted, models


def _score(rows, key):
    known = [row for row in rows if row["actual_up"] is not None]
    brier, logloss, correct, predictions = 0.0, 0.0, 0, 0
    for row in known:
        probability = v1._num(row.get(key))
        if probability is None or not 0 <= probability <= 1:
            brier += 1
            logloss -= math.log(EPSILON)
            continue
        predictions += 1
        probability = min(1-EPSILON, max(EPSILON, probability))
        y = row["actual_up"]
        brier += (probability-y)**2
        logloss -= y*math.log(probability)+(1-y)*math.log(1-probability)
        correct += int(probability >= .5) == y
    return {"scheduled_cases": len(rows), "known_outcomes": len(known),
            "prediction_cases": predictions, "brier": brier/len(known) if known else None,
            "logloss": logloss/len(known) if known else None,
            "accuracy": correct/len(known) if known else None}


def _summary(rows, key, first, protocol):
    width = protocol.pretest_validation_sessions//3
    return {"aggregate": _score(rows, key),
            "blocks": [_score([row for row in rows if first+n*width <= row["i"]
                               < first+(n+1)*width], key) for n in range(3)]}


def _better(candidate, reference, strict_logloss):
    actual, baseline = candidate["aggregate"], reference["aggregate"]
    if any(value is None for value in (actual["brier"], actual["logloss"],
                                       baseline["brier"], baseline["logloss"])):
        return False
    stable = sum(a["brier"] is not None and b["brier"] is not None
                 and a["brier"] < b["brier"]-EPSILON
                 for a, b in zip(candidate["blocks"], reference["blocks"])) >= 2
    loss_ok = (actual["logloss"] < baseline["logloss"]-EPSILON if strict_logloss else
               actual["logloss"] <= baseline["logloss"]+EPSILON)
    return bool(actual["brier"] < baseline["brier"]-EPSILON and loss_ok and stable)


def _select(data, group, start, protocol):
    end = start-data.horizon-protocol.embargo_sessions-1
    first = end-protocol.pretest_validation_sessions+1
    prior_config = _prior_config(protocol)
    result = {"cohort": group, "horizon": data.horizon, "selected": prior_config,
              "candidates": [], "models": [], "traces": [],
              "fallback_reason": "No stable pretest improvement over the prior-only baseline."}
    if first < 201:
        result["fallback_reason"] = "Insufficient pretest history; prior baseline is a fallback."
        return result
    result.update(start=data.days[first], end=data.days[end],
                  latest_label_date=data.days[end+data.horizon])
    prior, models = _simulate(data, first, end, prior_config, f"{group}-{data.horizon}-prior")
    result["models"].extend(models)
    result["prior_scores"] = _summary(prior, "raw_probability", first, protocol)
    result["traces"].extend({**row, "candidate": "prior"} for row in prior)
    eligible = []
    for window in protocol.candidate_windows_sessions:
        for penalty in protocol.mean_loss_penalties:
            config = {"kind": "ridge_logistic", "window_sessions": window,
                      "mean_loss_penalty": penalty, "calibration": "raw"}
            identity = f"{group}-{data.horizon}-w{window}-p{penalty}"
            rows, models = _simulate(data, first, end, config, identity)
            result["models"].extend(models)
            raw = _summary(rows, "raw_probability", first, protocol)
            calibrated = _summary(rows, "calibrated_probability", first, protocol)
            calibration_ok = _better(calibrated, raw, strict_logloss=False)
            for mode, score in (("raw", raw), ("rolling_oof_platt", calibrated)):
                allowed = mode == "raw" or calibration_ok
                beats_prior = _better(score, result["prior_scores"], strict_logloss=True)
                candidate = {"config": {**config, "calibration": mode}, "scores": score,
                             "calibration_eligible": allowed, "beats_prior_stably": beats_prior,
                             "eligible": allowed and beats_prior}
                result["candidates"].append(candidate)
                if candidate["eligible"]:
                    eligible.append(candidate)
            result["traces"].extend({**row, "candidate": identity} for row in rows)
    if eligible:
        selected = min(eligible, key=lambda row: (row["scores"]["aggregate"]["brier"],
                                                 row["scores"]["aggregate"]["logloss"]))
        result["selected"], result["fallback_reason"] = selected["config"], None
    return result


def select_pretest_models_v2(market_data, macro_history=None, protocol=None):
    """Create and save this artifact before opening the adaptive-retest dataset."""
    protocol = protocol or EvaluationV2Protocol()
    protocol.validate()
    prepared = _prepare(market_data, macro_history, protocol, selection_only=True)
    horizons = tuple(dict.fromkeys((protocol.primary_horizon_sessions,
                                   *protocol.secondary_horizons_sessions)))
    selections = []
    for group, universe in (("primary", protocol.primary_tickers),
                            ("individual", protocol.individual_tickers)):
        if not universe:
            continue
        for horizon in horizons:
            data = _Cohort(prepared, universe, horizon, protocol)
            selections.append(_select(data, group, prepared["start_i"], protocol))
    return {"version": VERSION, "protocol": asdict(protocol),
            "protocol_sha256": _hash(asdict(protocol)), "source_sha256": _source_hashes(),
            "pretest_input_sha256": _pretest_input_hash(prepared, protocol),
            "selection_cutoff_exclusive": protocol.evaluation_start,
            "validation": selections, "probability_threshold": .5,
            "halfyear_result_type": "adaptive_retest_not_untouched_holdout"}


def _check_selection(artifact, protocol):
    if (artifact.get("version") != VERSION
            or artifact.get("protocol_sha256") != _hash(asdict(protocol))
            or artifact.get("source_sha256") != _source_hashes()
            or artifact.get("selection_cutoff_exclusive") != protocol.evaluation_start):
        raise ValueError("Pretest selection protocol or source hashes differ; reseal selection.")
    expected = {(group, h) for group, universe in (("primary", protocol.primary_tickers),
                ("individual", protocol.individual_tickers)) if universe
                for h in (protocol.primary_horizon_sessions, *protocol.secondary_horizons_sessions)}
    found = [(row.get("cohort"), row.get("horizon")) for row in artifact.get("validation", [])]
    if len(found) != len(set(found)) or set(found) != expected:
        raise ValueError("Selection must cover exactly the predeclared cohorts/horizons.")
    for row in artifact["validation"]:
        config = row.get("selected") or {}
        if config.get("kind") == "prior_only":
            allowed = config == _prior_config(protocol)
        else:
            allowed = (config.get("kind") == "ridge_logistic"
                       and config.get("window_sessions") in protocol.candidate_windows_sessions
                       and config.get("mean_loss_penalty") in protocol.mean_loss_penalties
                       and config.get("calibration") in {"raw", "rolling_oof_platt"}
                       and any(candidate["eligible"] and candidate["config"] == config
                               for candidate in row.get("candidates", [])))
        if row.get("latest_label_date", "0000") >= protocol.evaluation_start:
            allowed = False
        if not allowed:
            raise ValueError("Selected configuration lacks its pretest eligibility evidence.")


def run_walk_forward_evaluation_v2(
    market_data, macro_history=None, protocol=None, pretest_selection=None,
    archived_options: Optional[Any] = None, archived_prediction_markets: Optional[Any] = None,
):
    """Run a research retest using sealed choices; no parameters are chosen on retest labels."""
    protocol = protocol or EvaluationV2Protocol()
    protocol.validate()
    selection = pretest_selection or select_pretest_models_v2(market_data, macro_history, protocol)
    _check_selection(selection, protocol)
    prepared = _prepare(market_data, macro_history, protocol)
    if selection.get("pretest_input_sha256") != _pretest_input_hash(prepared, protocol):
        raise ValueError("Pretest price/macro inputs differ from the sealed selection evidence.")
    predictions, registry, cohorts = [], [], {}
    for group, universe in (("primary", protocol.primary_tickers),
                            ("individual", protocol.individual_tickers)):
        cohort = {"tickers": list(universe), "horizons": {}}
        for validated in (row for row in selection["validation"] if row["cohort"] == group):
            horizon, config = validated["horizon"], validated["selected"]
            data = _Cohort(prepared, universe, horizon, protocol)
            rows, models = _simulate(data, prepared["start_i"], len(data.days)-horizon-1,
                                     config, f"deployed-{group}-{horizon}")
            registry.extend({**model, "cohort": group} for model in models)
            records = []
            for row in rows:
                i, ticker = row["i"], row["ticker"]
                feature = data.features.get((ticker, i))
                calibrated = config["calibration"] == "rolling_oof_platt"
                probability = (row["calibrated_probability"] if calibrated else
                               row["raw_probability"])
                authorship = ("historical_pooled_prior_only" if config["kind"] == "prior_only" else
                              "rolling_OOF_Platt_pretest_supported_not_guaranteed"
                              if calibrated and row["calibration_available"] else
                              "raw_logistic_score_uncalibrated")
                missing = []
                if feature is None:
                    missing.append("No complete consistent 201-session feature history.")
                if probability is None:
                    missing.append("No authorized preceding fitted forecast.")
                if row["actual_up"] is None:
                    missing.append("Missing outcome or mismatched origin/outcome price basis.")
                records.append({**row, "cohort": group, "horizon_sessions": horizon,
                                "probability_up": probability,
                                "prediction_up": int(probability >= .5)
                                if probability is not None else None,
                                "probability_authorship": authorship, "selected_config": config,
                                "shunge_proxy_up": v1._shunge_proxy(feature), "always_up": 1,
                                "feature_values": dict(zip(v1.FEATURES, feature["x"]))
                                if feature else {},
                                "feature_values_used": config["kind"] != "prior_only",
                                "input_dates": {"price_start": feature["history_start"],
                                                "price_end": feature["history_end"],
                                                "macro": feature["macro"]} if feature else {},
                                "blocked_missing_data": missing,
                                "historical_options_used": False,
                                "historical_event_probabilities_used": False})
            metrics = v1._metrics(records, protocol, horizon)
            metrics["forecast_score_with_missing_penalty"] = _score(records, "probability_up")
            metrics["selected_config"] = config
            metrics["prior_only_fallback"] = config["kind"] == "prior_only"
            metrics.setdefault("checks", {})["not_prior_only"] = config["kind"] != "prior_only"
            # A baseline selected as fallback cannot count as a successful merged prediction model.
            if config["kind"] == "prior_only":
                metrics["improvement_gate"] = False
            cohort["horizons"][str(horizon)] = metrics
            predictions.extend(records)
        cohort["primary_horizon"] = cohort["horizons"].get(str(protocol.primary_horizon_sessions))
        cohorts[group] = cohort
    measured = [row["primary_horizon"] for row in cohorts.values() if row["primary_horizon"]]
    passed = bool(measured and all(row["improvement_gate"] for row in measured))
    return {"model_version": VERSION, "protocol": asdict(protocol),
            "protocol_sha256": _hash(asdict(protocol)), "source_sha256": _source_hashes(),
            "status": "proxy_research_gate_passed" if passed else "target_not_met",
            "goal_completion_proven": False,
            "goal_status": "unproven_author_baseline_and_point_in_time_evidence",
            "halfyear_result_type": "adaptive_retest_not_untouched_holdout",
            **cohorts, "predictions": predictions, "models": registry,
            "validation": selection["validation"], "pretest_selection_sha256": _hash(selection),
            "baseline_rules": v1.BASELINE_RULES,
            "reference_claims": {"index_accuracy": .72, "individual_accuracy": .68,
                                 "status": "user_reported_unverified_not_same_protocol"},
            "input_availability": {
                "prices": "reconstructed_provider_history_not_strict_PIT",
                "macro": "revised_history_with_lag_and_explicit_availability_guard",
                "options": "not_used_no_validated_point_in_time_adapter",
                "prediction_markets": "not_used_no_validated_point_in_time_adapter",
                "supplied_option_archive": archived_options is not None,
                "supplied_event_archive": archived_prediction_markets is not None},
            "limitations": [
                "V1 metrics were observed; this adaptive retest is not an untouched holdout.",
                "Authors' actual 72/68 baselines and undefined indicators are not reproduced.",
                "All 26 fixed features remain; no new author indicator formula is assumed.",
                "Raw logistic output is uncalibrated; Platt requires a pretest quality gate.",
                "Prior-only fallback is a research baseline, never an improvement claim.",
                "Revised macro, reconstructed prices and fixed-universe survivorship limit proof.",
                "No current options or event probability is inserted into historical features."]}
