"""Pretest-only paired event/price samples. No fitting or model selection occurs here."""

import copy
import hashlib
import json
import math
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from . import sessions, structure, swing_forecast
from .ingest import historical_event_features
from .ingest.historical_event_features import BENCHMARKS, FEATURE_VERSION
from .sessions import is_us_equity_session
from .structure import _atr
from .swing_forecast import FEATURES, _feature

VERSION = "pretest-event-bridge-samples-v1"
FIT_ROW_ADAPTER_VERSION = "event-bridge-fit-row-shape-v1"
PRETEST_END_EXCLUSIVE = "2026-04-27"
NY = ZoneInfo("America/New_York")
FAMILY_CHANNELS = {
    "cpi_headline_mom": "inflation_cpi",
    "real_gdp_annualized_qoq": "jobs_growth",
    "payroll_jobs": "jobs_growth",
    "unemployment_u3": "jobs_growth",
}
SCALES = {family: 100000.0 if family == "payroll_jobs" else 1.0 for family in FAMILY_CHANNELS}
RELATIONS = {"lt": -1.0, "le": -1.0, "gt": 1.0, "ge": 1.0, "eq": 0.0, "interval": 2.0}
EVENT_FEATURES = ("fed_cut", "fed_hold", "fed_hike", "fed_missing") + tuple(
    f"{family}_{suffix}"
    for family in FAMILY_CHANNELS
    for suffix in (
        "p_yes",
        "threshold_offset",
        "relation_code",
        "interval_width",
        "missing",
        "p_yes_relation",
    )
)
FEATURE_NAMES = tuple(FEATURES) + EVENT_FEATURES


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _utc(value):
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _clock(day):
    return datetime.combine(date.fromisoformat(day), time(16, 15), NY).astimezone(timezone.utc)


def _sessions(start, end):
    days, cursor = [], date.fromisoformat(start)
    last = date.fromisoformat(end)
    while cursor <= last:
        if is_us_equity_session(cursor):
            days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days


def _future_sessions(origin, n):
    result, cursor = [], date.fromisoformat(origin)
    while len(result) < n:
        cursor += timedelta(days=1)
        if is_us_equity_session(cursor):
            result.append(cursor.isoformat())
    return result


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _date_index(rows):
    """Dates are the only fields inspected before temporal eligibility is known."""
    index, duplicates, rejected = {}, set(), []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            rejected.append({"source_position": position, "reason": "invalid_row_date_metadata"})
            continue
        stamp = row.get("date")
        if isinstance(stamp, str) and stamp >= PRETEST_END_EXCLUSIVE:
            continue
        try:
            if not isinstance(stamp, str) or date.fromisoformat(stamp).isoformat() != stamp:
                raise ValueError("noncanonical date")
            if not is_us_equity_session(date.fromisoformat(stamp)):
                raise ValueError("not a US session")
        except (ValueError, TypeError):
            rejected.append({"source_position": position, "reason": "invalid_row_date_metadata"})
            continue
        if stamp in index:
            duplicates.add(stamp)
        index[stamp] = row
    return index, duplicates, rejected


def _price(row, expected_basis=None):
    if row.get("session_complete") is not True:
        return None, "price_session_not_confirmed_complete"
    basis = row.get("price_basis")
    if (
        not isinstance(basis, str)
        or not basis.strip()
        or basis.lower().strip() in {"none", "n/a", "unverified"}
        or "unknown" in basis.lower()
        or "未注明" in basis
    ):
        return None, "unknown_price_basis"
    if expected_basis is not None and basis != expected_basis:
        return None, "price_basis_changed"
    source = row.get("source")
    if source not in {"yfinance", "Yahoo chart"}:
        return None, "unsupported_or_sample_price_source"
    trace = {
        key: row.get(key)
        for key in (
            "raw_body_file",
            "raw_json_pointer",
            "raw_timestamp_pointer",
            "raw_array_index",
            "source_timestamp",
            "fetched_at",
        )
        if row.get(key) is not None
    }
    if source == "Yahoo chart" and (
        not all(
            isinstance(trace.get(key), str) and trace[key]
            for key in ("raw_body_file", "raw_json_pointer", "raw_timestamp_pointer")
        )
        or type(trace.get("raw_array_index")) is not int
        or trace["raw_array_index"] < 0
        or _utc(trace.get("fetched_at")) is None
    ):
        return None, "untraceable_yahoo_chart_row"
    values = {key: _number(row.get(key)) for key in ("open", "high", "low", "close", "volume")}
    if (
        any(value is None for value in values.values())
        or values["volume"] < 0
        or min(values[key] for key in ("open", "high", "low", "close")) <= 0
        or not values["low"]
        <= min(values["open"], values["close"])
        <= max(values["open"], values["close"])
        <= values["high"]
    ):
        return None, "invalid_ohlcv"
    return {
        "date": row["date"],
        **values,
        "source": source,
        "price_basis": basis,
        "session_complete": True,
        **trace,
    }, None


def _encode_channels(payload, origin):
    if not isinstance(payload, dict) or not isinstance(payload.get("channels"), dict):
        return None, "invalid_event_feature_payload"
    channels, available = payload["channels"], []
    for channel, context in channels.items():
        if not isinstance(context, dict) or context.get("status") != "available":
            continue
        end = _utc(context.get("declared_event_end"))
        if (
            channel not in ("fed", "inflation_cpi", "jobs_growth")
            or not isinstance(context.get("group_id"), str)
            or not context["group_id"].strip()
            or end is None
            or end <= origin
            or end > origin + timedelta(days=21)
        ):
            return None, "invalid_available_event_context"
        available.append((end, channel, context["group_id"]))
    if not available:
        return None, "all_event_channels_missing"
    proofs, quote_clocks = [], []
    # Cohort membership itself uses all available channels, including unencoded families.
    for _, channel, _ in available:
        references = channels[channel].get("source_refs")
        if not isinstance(references, list) or not references:
            return None, "missing_event_source_proof"
        for ref in references:
            if not isinstance(ref, dict):
                return None, "missing_event_source_proof"
            clock = _utc(ref.get("quote_available_at"))
            if clock is None or clock > origin:
                return None, "event_features_not_available_at_origin"
            quote_clocks.append(clock)
            proofs.append(ref)
    fed = channels.get("fed", {})
    if fed.get("status") == "available":
        probabilities = fed.get("probabilities")
        if not isinstance(probabilities, dict):
            return None, "invalid_available_fed_vector"
        values = [_number(probabilities.get(key)) for key in ("cut", "hold", "hike")]
        if (
            any(value is None or not 0 <= value <= 1 for value in values)
            or abs(sum(values) - 1) > 0.05 + 1e-12
        ):
            return None, "invalid_available_fed_vector"
        features = values + [0.0]
    else:
        features = [0.0, 0.0, 0.0, 1.0]
    specifications = {}
    for family, channel in FAMILY_CHANNELS.items():
        context = channels.get(channel, {})
        source_measurements = context.get("measurements") or {}
        if not isinstance(source_measurements, dict):
            return None, "invalid_event_feature_payload"
        measurement = source_measurements.get(family, {})
        if not isinstance(measurement, dict):
            return None, "invalid_event_feature_payload"
        if context.get("status") != "available" or measurement.get("status") != "available":
            features.extend([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            continue
        spec = measurement.get("specification") or {}
        if not isinstance(spec, dict):
            return None, "invalid_available_measurement_specification"
        references = measurement.get("source_refs")
        if not isinstance(references, list) or not references:
            return None, "missing_event_source_proof"
        for ref in references:
            available_at = _utc(ref.get("quote_available_at")) if isinstance(ref, dict) else None
            if available_at is None or available_at > origin:
                return None, "event_features_not_available_at_origin"
            quote_clocks.append(available_at)
            if ref not in proofs:
                proofs.append(ref)
        yes, benchmark = (
            _number(measurement.get("yes_probability")),
            _number(measurement.get("benchmark")),
        )
        lower, upper = _number(spec.get("lower")), _number(spec.get("upper"))
        relation = spec.get("relation")
        if (
            measurement.get("family") != family
            or spec.get("family") != family
            or spec.get("unit") != ("jobs" if family == "payroll_jobs" else "percent")
            or yes is None
            or not 0 <= yes <= 1
            or benchmark != BENCHMARKS[family]
            or relation not in RELATIONS
            or spec.get("lower") is not None
            and lower is None
            or spec.get("upper") is not None
            and upper is None
            or lower is None
            and upper is None
            or lower is not None
            and upper is not None
            and lower > upper
            or relation in {"eq", "interval"}
            and (lower is None or upper is None)
            or relation == "eq"
            and lower != upper
            or relation in {"lt", "le"}
            and upper is None
            or relation in {"gt", "ge"}
            and lower is None
        ):
            return None, "invalid_available_measurement_specification"
        bounds = [value for value in (lower, upper) if value is not None]
        offset = (sum(bounds) / len(bounds) - benchmark) / SCALES[family]
        width = (upper - lower) / SCALES[family] if lower is not None and upper is not None else 0.0
        code = RELATIONS[relation]
        features.extend([yes, offset, code, width, 0.0, yes * code])
        specifications[family] = measurement
    if len(features) != len(EVENT_FEATURES) or not all(math.isfinite(value) for value in features):
        return None, "nonfinite_encoded_event_features"
    return {
        "features": features,
        "episode_id": min(available)[2],
        "underlying_group_ids": sorted({item[2] for item in available}),
        "quote_available_at": max(quote_clocks),
        "proofs": proofs,
        "measurement_specifications": specifications,
    }, None


def build_event_bridge_samples(market, adapter, *, start, end, as_of, horizon_sessions=20):
    """Return paired rows on the same event-present pretest cohort, without tuning."""
    start, end = date.fromisoformat(start).isoformat(), date.fromisoformat(end).isoformat()
    cutoff = datetime.combine(date.fromisoformat(PRETEST_END_EXCLUSIVE), time(), NY).astimezone(
        timezone.utc
    )
    requested_clock = _utc(as_of)
    if (
        requested_clock is None
        or start > end
        or end >= PRETEST_END_EXCLUSIVE
        or type(horizon_sessions) is not int
        or not 1 <= horizon_sessions <= 126
    ):
        raise ValueError(
            "Aware as_of, valid pretest dates and a bounded integer horizon are required."
        )
    if not isinstance(market, dict) or not callable(getattr(adapter, "lookup", None)):
        raise ValueError("Market mapping and audited event adapter are required.")
    manifest_sha = getattr(adapter, "manifest_sha256", None)
    if (
        not isinstance(manifest_sha, str)
        or len(manifest_sha) != 64
        or not set(manifest_sha) <= set("0123456789abcdef")
    ):
        raise ValueError("The event adapter must expose its sealed manifest hash.")
    effective_clock = min(requested_clock, cutoff)
    label_definition = f"close_{horizon_sessions}_sessions_ahead_above_origin_completed_close"
    source_modules = {
        "samples": Path(__file__),
        "price_features": Path(swing_forecast.__file__),
        "structure": Path(structure.__file__),
        "sessions": Path(sessions.__file__),
        "event_adapter": Path(historical_event_features.__file__),
    }
    policy = {
        "version": VERSION,
        "start": start,
        "end": end,
        "pretest_end_exclusive": PRETEST_END_EXCLUSIVE,
        "as_of": effective_clock.isoformat(),
        "origin_clock": "16:15 America/New_York completed-session buffer",
        "horizon_sessions": horizon_sessions,
        "label_definition": label_definition,
        "label_available_rule": "all future sessions and their close buffers strictly before as_of",
        "cohort": "any available event channel; event+price and price-only use identical rows",
        "event_archive_manifest_sha256": manifest_sha,
        "adapter_version": FEATURE_VERSION,
        "event_policy": {
            "lookahead_calendar_days": 21,
            "max_observation_age_hours": 24,
            "availability": "conservative bucket-end proxy",
        },
        "benchmarks": {key: BENCHMARKS[key] for key in FAMILY_CHANNELS},
        "unit_scales": dict(SCALES),
        "relation_codes": dict(RELATIONS),
        "event_encoding": "p_yes, mean-finite-bound benchmark offset/scale, relation_code, "
        "finite interval width/scale, missing_flag, p_yes*relation_code",
        "missing_policy": "ML zero imputation with missing=1, not observed probability zero",
        "price_prefix": "contiguous completed sessions from first supplied pretest date to origin",
        "price_hash_method": "canonical normalized row hashes, calendar ordered",
        "source_code_sha256": {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in source_modules.items()
        },
        "feature_names": list(FEATURE_NAMES),
        "baseline_feature_names": list(FEATURES),
        "historical_pit_verified": False,
        "stock_total_return": False,
    }
    policy_hash = _hash(policy)
    origins = _sessions(start, end)
    if len(origins) > 6000:
        raise ValueError("Origin count budget exceeded.")
    samples, exclusions = [], []
    event_cache = {}
    for asset in ("QQQ", "SPY"):
        rows = market.get(asset, [])
        index, duplicates, metadata_rejections = _date_index(rows)
        exclusions.extend(
            {"asset": asset, "origin_at": None, **item} for item in metadata_rejections
        )
        first = min(index) if index else None
        cache, digests = {}, {}

        def price_at(day, basis, _cache=cache, _index=index, _digests=digests):
            if day not in _cache:
                _cache[day] = _price(_index[day])
                if _cache[day][0] is not None:
                    _digests[day] = _hash(_cache[day][0])
            clean, reason = _cache[day]
            if clean is not None and basis is not None and clean["price_basis"] != basis:
                return None, "price_basis_changed"
            return clean, reason

        for day in origins:
            origin = _clock(day)
            identity = f"{asset}|{origin.isoformat()}"

            def exclude(reason, _identity=identity, _asset=asset, _origin=origin, **details):
                exclusions.append(
                    {
                        "id": _identity,
                        "asset": _asset,
                        "origin_at": _origin.isoformat(),
                        "reason": reason,
                        **details,
                    }
                )

            future_dates = _future_sessions(day, horizon_sessions)
            outcome_clock = _clock(future_dates[-1])
            # These checks must precede future-row access, validation, hashing, or labels.
            if origin >= effective_clock:
                exclude("origin_not_available_at_as_of")
                continue
            if future_dates[-1] >= PRETEST_END_EXCLUSIVE or outcome_clock >= effective_clock:
                exclude(
                    "label_window_unmatured_or_outside_pretest",
                    target_date=future_dates[-1],
                    outcome_available_at=outcome_clock.isoformat(),
                )
                continue
            if day not in event_cache:
                payload = adapter.lookup(origin)
                try:
                    json.dumps(payload, allow_nan=False)
                except (ValueError, TypeError) as exc:
                    raise ValueError("Noncanonical event feature payload.") from exc
                if (
                    not isinstance(payload, dict)
                    or payload.get("source_manifest_sha256") != manifest_sha
                    or payload.get("feature_version") != FEATURE_VERSION
                    or _utc(payload.get("origin")) != origin
                ):
                    raise ValueError("Adapter origin/version/manifest binding changed.")
                event_cache[day] = (payload, *_encode_channels(payload, origin))
            raw_events, encoded, event_error = event_cache[day]
            if event_error:
                exclude(event_error, raw_channels=raw_events.get("channels"))
                continue
            prefix_dates = _sessions(first, day) if first is not None and first <= day else []
            if len(prefix_dates) < 201:
                exclude("insufficient_preorigin_price_history")
                continue
            required = prefix_dates + future_dates
            missing = [stamp for stamp in required if stamp not in index or stamp in duplicates]
            if missing:
                exclude("missing_or_duplicate_complete_session", affected_dates=missing)
                continue
            prefix, basis, price_error = [], None, None
            for stamp in prefix_dates:
                value, reason = price_at(stamp, basis)
                if reason:
                    price_error = reason
                    break
                basis = value["price_basis"] if basis is None else basis
                prefix.append(value)
            if price_error:
                exclude("invalid_price_prefix", detail=price_error)
                continue
            price_features = _feature(prefix, len(prefix) - 1)
            atr = _atr(prefix)
            if price_features is None or len(price_features) != len(FEATURES) or not atr:
                exclude("unavailable_price_feature_vector")
                continue
            label_rows = []
            for stamp in future_dates:
                value, reason = price_at(stamp, basis)
                if reason:
                    price_error = reason
                    break
                label_rows.append(value)
            if price_error:
                exclude("invalid_label_window", detail=price_error)
                continue
            values = list(price_features) + encoded["features"]
            if not all(_number(value) is not None for value in values):
                exclude("nonfinite_joined_features")
                continue
            close, future_close = prefix[-1]["close"], label_rows[-1]["close"]
            available_clock = max(origin, encoded["quote_available_at"])
            if available_clock > origin:
                exclude("features_not_available_at_origin")
                continue
            samples.append(
                {
                    "id": identity,
                    "asset": asset,
                    "origin_at": origin.isoformat(),
                    "origin_date": day,
                    "features": values,
                    "baseline_features": list(price_features),
                    "feature_values": dict(zip(FEATURE_NAMES, values)),
                    "label": int(future_close > close),
                    "label_definition": label_definition,
                    "label_available_at": outcome_clock.isoformat(),
                    "outcome_available_at": outcome_clock.isoformat(),
                    "features_available_at": available_clock.isoformat(),
                    "episode_id": encoded["episode_id"],
                    "underlying_group_ids": encoded["underlying_group_ids"],
                    "price_context": {
                        "close": close,
                        "atr14": atr,
                        "price_basis": basis,
                        "history_start": prefix_dates[0],
                        "history_end": day,
                        "history_rows": len(prefix),
                        "history_sha256": _hash(
                            [(stamp, digests[stamp]) for stamp in prefix_dates]
                        ),
                        "origin_row": prefix[-1],
                        "price_available_at": origin.isoformat(),
                    },
                    "label_context": {
                        "future_session_dates": future_dates,
                        "terminal_close": future_close,
                        "window_sha256": _hash([(stamp, digests[stamp]) for stamp in future_dates]),
                        "terminal_row": label_rows[-1],
                        "strict_comparison": ">",
                    },
                    "raw_channels": raw_events["channels"],
                    "event_source_proofs": encoded["proofs"],
                    "measurement_specifications": encoded["measurement_specifications"],
                    "source_policy_sha256": policy_hash,
                    "historical_pit_verified": False,
                }
            )
    return {
        "rows": sorted(samples, key=lambda row: (row["origin_at"], row["asset"])),
        "exclusions": exclusions,
        "feature_names": list(FEATURE_NAMES),
        "baseline_feature_names": list(FEATURES),
        "source_policy_sha256": policy_hash,
        "source_policy": policy,
        "summary": {
            "rows": len(samples),
            "assets": dict(Counter(row["asset"] for row in samples)),
            "exclusion_reasons": dict(Counter(row["reason"] for row in exclusions)),
            "unique_underlying_release_groups": len(
                {group for row in samples for group in row["underlying_group_ids"]}
            ),
            "model_fitting_performed": False,
            "performance_computed": False,
        },
    }


def as_event_bridge_fit_rows(sample_bundle):
    """Convert verified vector exports to the fitter's named-feature row contract.

    This changes shape only. It never drops rows, changes labels/proofs, rescales
    values, or modifies the supplied bundle. The original vector is retained.
    """
    if not isinstance(sample_bundle, dict):
        raise ValueError("A sample bundle is required.")
    names = sample_bundle.get("feature_names")
    rows = sample_bundle.get("rows")
    baseline_names = sample_bundle.get("baseline_feature_names", [])
    if (
        not isinstance(names, (list, tuple))
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or not isinstance(rows, list)
        or not isinstance(baseline_names, (list, tuple))
        or list(names[: len(baseline_names)]) != list(baseline_names)
    ):
        raise ValueError("Invalid or misaligned feature-name inventory.")
    converted = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each sample row must be a mapping.")
        vector, values = row.get("features"), row.get("feature_values")
        if (
            not isinstance(vector, list)
            or len(vector) != len(names)
            or not isinstance(values, dict)
            or set(values) != set(names)
            or any(type(value) not in (int, float) or _number(value) is None for value in vector)
            or any(
                type(values[name]) is not type(value) or values[name] != value
                for name, value in zip(names, vector)
            )
        ):
            raise ValueError("Feature vector/name/feature_values alignment failed.")
        if "feature_vector" in row and row["feature_vector"] != vector:
            raise ValueError("Existing feature_vector disagrees with the original export.")
        if baseline_names:
            baseline = row.get("baseline_features")
            if (
                not isinstance(baseline, list)
                or baseline != vector[: len(baseline_names)]
                or any(type(a) is not type(b) for a, b in zip(baseline, vector))
            ):
                raise ValueError("Price-only baseline does not match the feature prefix.")
        result = copy.deepcopy(row)
        result["feature_vector"] = result["features"]
        result["features"] = dict(zip(names, result["feature_vector"]))
        converted.append(result)
    return converted
