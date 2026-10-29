"""Local, causal refresh of daily direction and cloud-spending context; no network."""

import copy
import gzip
import hashlib
import json
import math
from datetime import date, datetime, time, timezone
from statistics import median
from zoneinfo import ZoneInfo

from .cloud_capex import METRICS, _aggregate
from .direction_probability import estimate_direction
from .sessions import expected_complete_session, is_us_equity_session
from .storage import write_json

BASIS = "provider_split_adjusted_OHLC_no_dividend_direction"
NY = ZoneInfo("America/New_York")


def _time(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result if result.tzinfo is not None and result.utcoffset() is not None else None
    except (TypeError, ValueError):
        return None


def _number(value):
    if type(value) not in (int, float):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def _load(path, fingerprints, warnings):
    if not path.is_file():
        fingerprints[path.name] = {"status": "missing"}
        return {}
    try:
        raw = path.read_bytes()
        fingerprints[path.name] = {"status": "captured", "sha256": hashlib.sha256(raw).hexdigest()}
        return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    except (ValueError, OSError, EOFError):
        warnings.append(path.name + ": unreadable local cache")
        return {}


def _records(value):
    rows = (value if isinstance(value, list) else
            value.get("rows") if isinstance(value, dict) else [])
    return rows if isinstance(rows, list) else []


def _normalise(rows, cutoff, clock):
    result = {}
    for row in rows:
        day = row.get("date") if isinstance(row, dict) else None
        if not isinstance(day, str) or day > cutoff:
            continue  # Future prices are not inspected, even when malformed.
        parsed = date.fromisoformat(day)
        if day != parsed.isoformat() or not is_us_equity_session(parsed):
            raise ValueError("invalid_price_session")
        fetched = _time(row.get("fetched_at"))
        if (row.get("source") not in {"yfinance", "Yahoo chart"}
                or row.get("session_complete") is not True
                or row.get("fetched_at") is not None and fetched is None
                or fetched and fetched > clock):
            raise ValueError("unverified_incomplete_or_future_price")
        basis = str(row.get("price_basis") or "")
        if basis == BASIS:
            keys = ("open", "high", "low", "close")
        elif any("raw_" + key in row for key in ("open", "high", "low", "close")):
            keys = ("raw_open", "raw_high", "raw_low", "raw_close")
        elif "split-adjusted ohlc" in basis.lower():
            keys = ("open", "high", "low", "close")
        else:
            raise ValueError("legacy_adjusted_prices_without_raw_OHLC")
        values = [_number(row.get(key)) for key in keys]
        volume = _number(row.get("volume"))
        if (any(v is None or v <= 0 for v in values) or volume is None or volume < 0
                or not values[2] <= min(values[0], values[3])
                <= max(values[0], values[3]) <= values[1]):
            raise ValueError("invalid_raw_OHLCV")
        normal = {key: value for key, value in zip(("open", "high", "low", "close"), values)}
        normal.update(date=day, volume=volume, source=row["source"], session_complete=True,
                      price_basis=BASIS, provider_adjusted_close=_number(
                          row.get("provider_adjusted_close")))
        for key in ("fetched_at", "raw_body_file", "raw_json_pointer", "raw_timestamp_pointer",
                    "raw_array_index", "ohlc_reconciliation_factor"):
            if key in row:
                normal[key] = row[key]
        if day in result and any(result[day][key] != normal[key] for key in
                                 ("open", "high", "low", "close", "volume")):
            raise ValueError("conflicting_same_day_prices")
        result[day] = normal
    return [result[day] for day in sorted(result)]


def _merge(old, new, allow_recent_revisions=False):
    if not new:
        raise ValueError("no_current_verified_raw_prices")
    previous, incoming = ({r["date"]: r for r in rows} for rows in (old, new))
    overlap = sorted(set(previous) & set(incoming))
    ratio = 1.
    revisions = []
    if old:
        if not overlap:
            raise ValueError("no_overlap_to_reconcile_provider_price_basis")
        ratios = [incoming[day][key]/previous[day][key] for day in overlap
                  for key in ("open", "high", "low", "close")]
        ratio = median(ratios)
        if (not math.isfinite(ratio) or ratio <= 0
                or max(abs(r/ratio-1) for r in ratios) > 1e-4):
            if not allow_recent_revisions:
                raise ValueError("inconsistent_provider_overlap_price_ratios")
            recent = {row["date"] for row in new[-2:]}
            stable = [day for day in overlap if day not in recent]
            if len(stable) < 20:
                raise ValueError("recent_revision_requires20stable_overlap_sessions")
            stable_ratios = [incoming[day][key]/previous[day][key] for day in stable
                             for key in ("open", "high", "low", "close")]
            ratio = median(stable_ratios)
            if (not math.isfinite(ratio) or ratio <= 0
                    or max(abs(value/ratio-1) for value in stable_ratios) > 1e-4):
                raise ValueError("inconsistent_older_provider_overlap_price_ratios")
            for day in overlap:
                differences = {key: abs(incoming[day][key]/previous[day][key]/ratio-1)
                               for key in ("open", "high", "low", "close")}
                if max(differences.values()) <= 1e-4:
                    continue
                older, newer = (_time(rows[day].get("fetched_at"))
                                for rows in (previous, incoming))
                if (day not in recent or max(differences.values()) > 1e-3
                        or older is None or newer is None or newer <= older):
                    raise ValueError("unverified_or_excessive_recent_provider_revision")
                revisions.append({"date": day,
                                  "fields": [key for key,value in differences.items()
                                             if value > 1e-4],
                                  "old": {key: previous[day][key] for key in differences},
                                  "new": {key: incoming[day][key] for key in differences},
                                  "old_fetched_at": older.isoformat(),
                                  "new_fetched_at": newer.isoformat(),
                                  "maximum_relative_residual": max(differences.values()),
                                  "basis_factor_from_stable_overlap": ratio})
    merged = {}
    for day, row in previous.items():
        scaled = dict(row)
        if day < new[0]["date"]:
            for key in ("open", "high", "low", "close"):
                scaled[key] *= ratio
                if not math.isfinite(scaled[key]) or scaled[key] <= 0:
                    raise ValueError("nonfinite_reconciled_price_prefix")
            scaled["ohlc_reconciliation_factor"] = ratio
        elif day not in incoming and abs(ratio-1) > 1e-4:
            raise ValueError("new_provider_history_gap_during_unit_reconciliation")
        merged[day] = scaled
    merged.update(incoming)
    return [merged[day] for day in sorted(merged)], {
        "overlap_sessions": len(overlap), "older_OHLC_factor": ratio,
        "maximum_relative_ratio_deviation": max((abs(r/ratio-1) for r in ratios), default=0.)
        if old else 0., "corporate_split_verified": False, "volume_rescaled": False,
        **({"accepted_recent_provider_revisions": revisions,
            "recent_revision_policy": "latest2sessions<=0.1%;20stableoverlap;newerretrieval"}
           if revisions else {})}


def _causal(records, cutoff):
    return [row for row in _records(records) if isinstance(row, dict)
            and _time(row.get("available_at")) is not None
            and _time(row["available_at"]) <= cutoff]


def _capex_snapshot(bundle, report_date, cutoff, generated):
    eligible = [row for row in _causal(bundle, cutoff)
                if str(row.get("date") or "9999") <= cutoff.date().isoformat()]
    if not eligible:
        return {"date": report_date, "available_at": generated.isoformat(), "issuers": {},
                "aggregate": {"status": "insufficient_issuers", "included_issuers": []},
                "reason": "no_causal_cloud_capex_cache", "source_capture_date": "unknown"}
    result = copy.deepcopy(max(eligible, key=lambda row: (row["date"], row["available_at"])))
    result.update(date=report_date, snapshot_as_of=cutoff.isoformat(),
                  source_snapshot_date=result["date"], source_capture_date=(
                      bundle.get("source_capture_date", "unknown") if isinstance(bundle, dict)
                      else "unknown"))
    for issuer in result.get("issuers", {}).values():
        clocks = [_time(issuer.get("current", {}).get(kind, {}).get("available_at_proxy"))
                  for kind in ("capex", "revenue")]
        if not all(c and 0 <= (cutoff-c).total_seconds() <= 130*86400 for c in clocks):
            issuer.update(status="unknown",
                          reason="current_filing_missing_future_or_older_than_130_days",
                          **dict.fromkeys(METRICS))
    result["aggregate"] = _aggregate(result.get("issuers", {}))
    return result


def _live_macro(rows, cutoff):
    """Only explicitly dated provider availability is usable; observation date is insufficient."""
    result = []
    names = {"DGS10": "yield10_pct", "UST10Y": "yield10_pct", "VIXCLS": "vix",
             "DFEDTARU": "fed_upper_pct"}
    for series in _records(rows):
        if not isinstance(series, dict):
            continue
        key = names.get(series.get("series_id"))
        if key is None:
            continue
        for point in series.get("history") or [series]:
            if not isinstance(point, dict):
                continue
            available = _time(point.get("available_at"))
            if available is None or available > cutoff or not series.get("source"):
                continue
            try:
                if date.fromisoformat(point.get("date")) > cutoff.date():
                    continue
            except (TypeError, ValueError):
                continue
            value = _number(point.get("value"))
            if value is not None:
                result.append({"date": point.get("date"), "available_at": available.isoformat(),
                               "values": {key: value}, "source_observations": {key: {
                                   "available_at": available.isoformat(),
                                   "source": series["source"], "series_id": series["series_id"],
                                   "observed_date": point.get("date")}}})
    return result


def _refresh_daily_context(paths, report_date, market_data, analysis_clock=None):
    """Write dated contexts and exact price/refit traces before analysis archiving.

    Probability availability is actual generation time; refit_at in the model is
    only its logical historical cutoff. Missing caches never stop the daily report.
    """
    clock = analysis_clock or datetime.now(timezone.utc)
    if not isinstance(clock, datetime) or clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError("Analysis clock must include a timezone.")
    expected = expected_complete_session(report_date, now=clock)
    cutoff = datetime.combine(date.fromisoformat(expected), time(16, 15), NY)
    fingerprints, warnings = {}, []
    history = _load(paths.processed / "direction-history.json", fingerprints, warnings)
    capex = _load(paths.processed / "cloud-capex-history.json.gz", fingerprints, warnings)
    events = _load(paths.processed / "direction-event-history.json", fingerprints, warnings)
    macro = _load(paths.processed / "direction-macro-history.json", fingerprints, warnings)
    live_macro = _load(paths.processed / f"macro-{report_date}.json", fingerprints, warnings)
    capex_rows, event_rows = _causal(capex, cutoff), _causal(events, cutoff)
    macro_rows = _causal(macro, cutoff) + _live_macro(live_macro, cutoff)
    # Carry forward field-specific clocks when a live snapshot updates only one field.
    combined, values, observations = [], {}, {}
    for row in sorted(macro_rows, key=lambda row: _time(row["available_at"])):
        for key in row.get("values", {}):
            proof = row.get("source_observations", {}).get(key, {})
            available = _time(proof.get("available_at"))
            if available is not None and available <= cutoff:
                values[key], observations[key] = row["values"][key], proof
        combined.append({**row, "values": dict(values),
                         "source_observations": copy.deepcopy(observations)})
    probabilities, used, reconciliation, updated = {}, {}, {}, {}
    for ticker in ("QQQ", "SPY"):
        try:
            old = _normalise(history.get(ticker, []) if isinstance(history, dict) else [],
                             expected, clock)
            new = _normalise(market_data.get(ticker, []), expected, clock)
            merged, proof = _merge(old, new)
            if merged[-1]["date"] != expected or not new or new[-1]["date"] != expected:
                raise ValueError("latest_completed_session_missing")
            updated[ticker], used[ticker], reconciliation[ticker] = merged, merged, proof
            result = estimate_direction(merged, expected, capex_rows=capex_rows,
                                        event_rows=event_rows, macro_rows=combined)
            json.dumps(result, allow_nan=False)
            result.update(price_date=expected, price_close=merged[-1]["close"])
            if result.get("model_kind") == "historical_base_rate":
                result["reason"] = "条件模型未优于验证基准，当前仅为历史频率参考"
            probabilities[ticker] = result
        except Exception as error:
            probabilities[ticker] = {"status": "unavailable", "probability_up": None,
                                     "probability_down": None, "price_date": expected,
                                     "reason": str(error) or type(error).__name__}
    # Timestamp AFTER all fitting, never at the historical refit cutoff.
    generated = datetime.now(timezone.utc) if analysis_clock is None else clock
    for row in probabilities.values():
        row["available_at"] = generated.isoformat()
    directional = {"date": report_date, "horizon_sessions": 20, "by_ticker": probabilities,
                   "available_at": generated.isoformat(), "options_in_historical_model": False}
    cloud = _capex_snapshot(capex, report_date, cutoff, generated)
    trace = {"date": report_date, "generated_at": generated.isoformat(),
             "expected_completed_session": expected, "used_price_history": used,
             "price_basis": BASIS, "source_fingerprints": fingerprints,
             "provider_reconciliation": reconciliation,
             "auxiliary_source_rows": {"capex": len(capex_rows), "events": len(event_rows),
                                       "macro": len(combined)},
             "training_caps": {ticker: {key: row.get(key) for key in (
                 "training_samples", "training_label_last", "training_origin_first", "refit_at",
                 "used_price_prefix_sha256", "model_kind", "validation", "model")}
                 for ticker, row in probabilities.items()},
             "historical_pit_verified": False,
             "auxiliary_cache_bytes_archived_here": False,
             "live_event_quotes_directly_assigned_to_stock_probability": False}
    for name, payload in (("direction-probabilities", directional), ("cloud-capex", cloud),
                          ("direction-training", trace)):
        write_json(paths.processed / f"{name}-{report_date}.json", payload)
    if updated:
        preserved = dict(history) if isinstance(history, dict) else {}
        for ticker, rows in updated.items():
            # A historical replay must not truncate or rescale a later persistent cache.
            future = [r for r in _records(preserved.get(ticker, []))
                      if isinstance(r, dict) and str(r.get("date") or "") > expected]
            if not future:
                preserved[ticker] = rows
        write_json(paths.processed / "direction-history.json", preserved)
    return {"directional_context": directional, "cloud_capex_context": cloud, "warnings": warnings}


def refresh_daily_context(paths, report_date, market_data, analysis_clock=None):
    """Fail explicitly unknown, replacing any old dated forecasts after cache errors."""
    if date.fromisoformat(report_date).isoformat() != report_date:
        raise ValueError("An ISO report date is required.")
    if analysis_clock is not None and (
            not isinstance(analysis_clock, datetime) or analysis_clock.tzinfo is None
            or analysis_clock.utcoffset() is None):
        raise ValueError("Analysis clock must include a timezone.")
    try:
        return _refresh_daily_context(paths, report_date, market_data, analysis_clock)
    except Exception as error:
        generated = analysis_clock or datetime.now(timezone.utc)
        reason = "本地历史上下文不完整；本次不复用旧方向概率。"
        unavailable = {"status": "unavailable", "probability_up": None, "probability_down": None,
                       "available_at": generated.isoformat(), "reason": reason}
        directional = {"date": report_date, "horizon_sessions": 20,
                       "by_ticker": {ticker: dict(unavailable) for ticker in ("QQQ", "SPY")}}
        cloud = {"date": report_date, "available_at": generated.isoformat(), "issuers": {},
                 "reason": reason, "aggregate": {"status": "insufficient_issuers",
                                                 "included_issuers": []}}
        trace = {"date": report_date, "generated_at": generated.isoformat(),
                 "status": "unavailable", "error_type": type(error).__name__, "reason": reason,
                 "partial_fit_outputs_discarded": True, "old_probabilities_reused": False}
        for prefix, payload in (("direction-probabilities", directional), ("cloud-capex", cloud),
                                ("direction-training", trace)):
            write_json(paths.processed / f"{prefix}-{report_date}.json", payload)
        return {"directional_context": directional, "cloud_capex_context": cloud,
                "warnings": [reason + " (" + type(error).__name__ + ")"]}
