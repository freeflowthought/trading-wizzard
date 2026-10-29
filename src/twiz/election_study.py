"""Descriptive US midterm event windows; election outcomes are retrospective groups.

No historical prediction-market data is required or inferred. Thirteen elections
cannot identify a calibrated market probability or isolate an election's causal effect.
"""

import math
from datetime import date, timedelta
from statistics import mean, median, stdev
from typing import Any, Dict, Iterable, List

from .sessions import is_us_equity_session

MIDTERM_YEARS = tuple(range(1974, 2023, 4))
DEFAULT_SECONDARY = ("^NDX", "SPY", "QQQ")
UNKNOWN_SOURCES = {"sample", "unverified", "synthetic", "unknown", "demo", "placeholder",
                   "unavailable", "none", "n/a"}


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _election_date(year):
    first = date(year, 11, 1)
    monday = first+timedelta(days=(0-first.weekday()) % 7)
    return monday+timedelta(days=1)


def _window_days(election):
    before, after = [], []
    for direction, output in ((-1, before), (1, after)):
        cursor = election
        while len(output) < 10:
            cursor += timedelta(days=direction)
            if is_us_equity_session(cursor):
                output.append(cursor.isoformat())
    return list(reversed(before))+[election.isoformat()]+after


def _price_index(rows):
    values, duplicates = {}, set()
    for row in rows:
        stamp = str(row.get("date", ""))
        try:
            day = date.fromisoformat(stamp)
        except ValueError:
            continue
        if not is_us_equity_session(day):
            continue
        if stamp in values:
            duplicates.add(stamp)
        values[stamp] = row
    return values, duplicates


def _metrics(rows, pre=False):
    closes = [float(row["close"]) for row in rows]
    daily = [(b/a-1)*100 for a, b in zip(closes, closes[1:])]
    logs = [math.log(value) for value in closes]
    x_mean, y_mean = 5.0, mean(logs)
    slope = sum((i-x_mean)*(y-y_mean) for i, y in enumerate(logs))/110
    fitted = [y_mean+slope*(i-x_mean) for i in range(11)]
    total = sum((y-y_mean)**2 for y in logs)
    residual = sum((y-fit)**2 for y, fit in zip(logs, fitted))
    r2 = max(0.0, min(1.0, 1-residual/total)) if total > 1e-24 else None
    returns = {str(n): (closes[-1]/closes[-1-n]-1)*100 if pre else
               (closes[n]/closes[0]-1)*100 for n in (1, 5, 10)}
    if not all(math.isfinite(value) for value in (*daily, *returns.values())):
        raise ValueError("Nonfinite derived returns.")
    peak, peak_date, drawdown = closes[0], rows[0]["date"], 0.0
    worst_peak, trough = peak_date, peak_date
    for row, close in zip(rows, closes):
        if close > peak:
            peak, peak_date = close, row["date"]
        fall = (close/peak-1)*100
        if fall < drawdown:
            drawdown, worst_peak, trough = fall, peak_date, row["date"]
    result = {"start_date": rows[0]["date"], "end_date": rows[-1]["date"],
              "close_count": 11, "return_count": 10, "returns_pct": returns,
              "return_10d_pct": returns["10"], "daily_returns_pct": daily,
              "daily_volatility_sample_pct": stdev(daily),
              "log_ols_slope_per_session": slope, "log_ols_r2": r2,
              "max_drawdown_pct": drawdown,
              "drawdown_peak_date": worst_peak, "drawdown_trough_date": trough}
    numeric = [*daily, *returns.values(), slope, result["daily_volatility_sample_pct"], drawdown]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("Nonfinite derived window statistics.")
    return result


def _party(value):
    if not isinstance(value, str):
        return "unknown"
    return {"D": "Democratic", "Democrat": "Democratic", "Democratic": "Democratic",
            "R": "Republican", "Republican": "Republican"}.get(value, "unknown")


def _politics(record, expected_date):
    empty = {"president_party": "unknown", "chambers_flipped": "unknown",
             "resulting_control": "unknown", "opposition_control": "unknown",
             "opposition_takeover": "unknown", "new_opposition_controls_both": "unknown",
             "power_shift_severity": "unknown"}
    if not record:
        return {**empty, "metadata_status": "unknown", "sources": []}
    supplied = record.get("election_date", record.get("date"))
    if supplied is not None and supplied != expected_date:
        return {**empty, "metadata_status": "event_date_mismatch", "sources": []}
    if record.get("verified") is False:
        return {**empty, "metadata_status": "explicitly_unverified",
                "sources": record.get("sources", record.get("source_urls", []))}
    president = _party(record.get("president_party"))
    before = [_party(record.get(key)) for key in ("house_before", "senate_before")]
    after = [_party(record.get(key)) for key in ("house_after", "senate_after")]
    flips = sum(a != b for a, b in zip(before, after)) if "unknown" not in before+after else None
    result = {**empty, "president_party": president, "metadata_status": "caller_supplied",
              "sources": record.get("sources", record.get("source_urls",
                          [record["source"]] if record.get("source") else [])),
              "result_effective_date": record.get("result_effective_date"),
              "results_known_at": record.get("results_known_at"),
              "congress_before": record.get("congress_before"),
              "congress_after": record.get("congress_after"),
              "supplied_controls": {key: record.get(key) for key in (
                  "house_before", "senate_before", "house_after", "senate_after")}}
    if flips is not None:
        result["chambers_flipped"] = flips
    elif type(record.get("chambers_flipped")) is int and record["chambers_flipped"] in (0, 1, 2):
        result["chambers_flipped"] = record["chambers_flipped"]
    if president != "unknown" and "unknown" not in after:
        opposition = [a != president for a in after]
        result["resulting_control"] = "unified" if not any(opposition) else "divided"
        result["opposition_control"] = ("both" if all(opposition) else "house" if opposition[0]
                                        else "senate" if opposition[1] else "neither")
        if "unknown" not in before:
            takes = [a == president and b != president for a, b in zip(before, after)]
            result["opposition_takeover"] = ("both" if all(takes) else "house" if takes[0]
                                             else "senate" if takes[1] else "neither")
            result["new_opposition_controls_both"] = bool(
                all(opposition) and not all(party != president for party in before))
    else:
        for key, allowed in (("resulting_control", {"unified", "divided"}),
                             ("opposition_control", {"house", "senate", "both", "neither"})):
            if record.get(key) in allowed:
                result[key] = record[key]
        if record.get("government_after") in {"unified", "divided"}:
            result["resulting_control"] = record["government_after"]
    change = result["chambers_flipped"]
    result["power_shift_severity"] = (
        "double_flip" if change == 2 else "new_full_opposition_congress"
        if result["new_opposition_controls_both"] is True else "single_chamber_only"
        if change == 1 else "unchanged" if change == 0 else "unknown")
    return result


def _describe(values):
    return {"n": len(values), "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "positive_fraction": sum(value > 0 for value in values)/len(values) if values else None}


def _summarize_rows(rows):
    metrics = {}
    for side in ("pre", "post"):
        for key in ("return_10d_pct", "daily_volatility_sample_pct",
                    "log_ols_slope_per_session", "log_ols_r2", "max_drawdown_pct"):
            values = [row[side][key] for row in rows if row["status"] == "available"
                      and row[side][key] is not None]
            metrics[side+"_"+key] = _describe(values)
    for horizon in (1, 5, 10):
        values = [row["post"]["returns_pct"][str(horizon)] for row in rows
                  if row["status"] == "available"]
        metrics[f"post_return_{horizon}d_pct"] = _describe(values)
    return {"cycles": len(rows), "available_cycles": sum(row["status"] == "available"
                                                         for row in rows), "metrics": metrics}


def _groups(cycles, assets):
    dimensions = {
        "chambers_flipped": (0, 1, 2, "unknown"),
        "president_party": ("Democratic", "Republican", "unknown"),
        "resulting_control": ("unified", "divided", "unknown"),
        "opposition_control": ("house", "senate", "both", "neither", "unknown"),
        "opposition_takeover": ("house", "senate", "both", "neither", "unknown"),
        "new_opposition_controls_both": (True, False, "unknown"),
        "power_shift_severity": ("double_flip", "new_full_opposition_congress",
                                 "single_chamber_only", "unchanged", "unknown"),
    }
    result = []
    for asset in assets:
        for dimension, categories in dimensions.items():
            for category in categories:
                rows = [row for row in cycles if row["asset"] == asset
                        and row["politics"][dimension] == category]
                result.append({"asset": asset, "dimension": dimension, "category": category,
                               **_summarize_rows(rows)})
    return result


def _joint_groups(cycles, assets):
    dimensions = ("chambers_flipped", "president_party", "resulting_control",
                  "opposition_control", "opposition_takeover", "new_opposition_controls_both",
                  "power_shift_severity")
    result = []
    for asset in assets:
        grouped = {}
        for row in cycles:
            if row["asset"] == asset:
                key = tuple(row["politics"][dimension] for dimension in dimensions)
                grouped.setdefault(key, []).append(row)
        for key, rows in grouped.items():
            result.append({"asset": asset, "context": dict(zip(dimensions, key)),
                           "years": [row["year"] for row in rows], **_summarize_rows(rows)})
    return result


def build_election_study(
    market_data: Dict[str, List[Dict[str, Any]]], event_records: Iterable[Dict[str, Any]],
    primary_asset: str = "^GSPC", secondary_assets=DEFAULT_SECONDARY,
) -> Dict[str, Any]:
    """Retain every declared election/asset cell, including missing histories and controls."""
    assets = tuple(dict.fromkeys((primary_asset, *secondary_assets)))
    indexed = {asset: _price_index(market_data.get(asset, [])) for asset in assets}
    events, duplicates = {}, set()
    for record in event_records:
        year = record.get("year")
        if type(year) is not int:
            try:
                year = date.fromisoformat(str(record.get("election_date", record.get("date")))).year
            except ValueError:
                continue
        if year in events:
            duplicates.add(year)
        events[year] = record
    cycles = []
    for year in MIDTERM_YEARS:
        election = _election_date(year)
        days = _window_days(election)
        political = _politics(None if year in duplicates else events.get(year),
                              election.isoformat())
        if year in duplicates:
            political["metadata_status"] = "duplicate_event_records"
        for asset in assets:
            prices, duplicate_dates = indexed[asset]
            row = {"year": year, "asset": asset, "event_date": election.isoformat(),
                   "status": "unknown", "politics": political,
                   "control_information_timing": "retrospective_after_election_not_known_at_E0",
                   "window_dates": days, "pre": None, "post": None, "prices": [],
                   "price_basis": None, "price_sources": [], "reasons": []}
            missing = [stamp for stamp in days if stamp not in prices]
            if missing:
                row["reasons"].append("Missing exact session closes: "+", ".join(missing))
            elif duplicate_dates.intersection(days):
                row["reasons"].append("Duplicate session rows prevent an unambiguous price window.")
            else:
                observations = [prices[stamp] for stamp in days]
                supplied_basis = [r.get("price_basis") for r in observations]
                basis = (set(supplied_basis) if all(isinstance(value, str) and value.strip()
                                                   and value.strip().lower() not in {
                                                       "unknown", "none", "n/a", "mixed"}
                                                   for value in supplied_basis) else set())
                valid = (len(basis) == 1
                         and all(_number(r.get("close")) is not None and float(r["close"]) > 0
                                 and r.get("session_complete") is True
                                 and isinstance(r.get("source"), str) and bool(r["source"].strip())
                                 and r["source"].strip().lower() not in UNKNOWN_SOURCES
                                 for r in observations))
                if not valid:
                    row["reasons"].append("Requires positive finite closes, one nonempty basis, "
                                          "a named source and explicit completed-session flags.")
                else:
                    source_rows = [{"date": stamp, "close": float(prices[stamp]["close"])}
                                   for stamp in days]
                    try:
                        row.update(pre=_metrics(source_rows[:11], pre=True),
                                   post=_metrics(source_rows[10:]), status="available",
                                   price_basis=next(iter(basis)), prices=source_rows,
                                   price_sources=sorted({str(r.get("source", "unknown"))
                                                         for r in observations}))
                    except (OverflowError, ValueError):
                        row["reasons"].append("Derived window statistics are nonfinite.")
            cycles.append(row)
    return {"study": "US_midterms_1974_2022_descriptive", "years": list(MIDTERM_YEARS),
            "primary_asset": primary_asset, "secondary_assets": list(assets[1:]),
            "political_group_definitions": {
                "new_opposition_controls_both": "Opposition controls both after, not both before; "
                                                "either party, including one final chamber flip.",
                "opposition_takeover_both": "Both chambers newly switch from president's party "
                                             "to opposition; strict subgroup.",
                "power_shift_severity": "Descriptive control changes only, not market direction."},
            "window_definition": {"pre": "E-10 close to E0 close", "post": "E0 close to E+10 close",
                                  "pre_short_returns": "E-n close to E0 close",
                                  "post_short_returns": "E0 close to E+n close",
                                  "volatility": "sample SD of ten simple returns; unannualized",
                                  "slope": "OLS of eleven log closes on session index 0..10",
                                  "drawdown": "ordered running peak-to-trough closes"},
            "cycles": cycles, "groups": _groups(cycles, assets),
            "joint_groups": _joint_groups(cycles, assets),
            "coverage": {asset: {"declared_cycles": len(MIDTERM_YEARS),
                                  "available_cycles": sum(row["status"] == "available"
                                                          for row in cycles
                                                          if row["asset"] == asset)}
                         for asset in assets},
            "limitations": [
                "Descriptive only; no calibrated market probability or election causal effect.",
                "Final chamber control is retrospective, may follow E+10, and is not known at E0.",
                "E0 closes precede election-night information; overnight and daily moves differ.",
                "Policy, earnings, crises and macro changes confound these short windows.",
                "Price returns do not add dividend reinvestment; provider bases require checking.",
                "Assets stay separate; related ETFs and indices are not independent events.",
                "Caller-supplied political facts are not independently verified by this module.",
                "Missing history stays unknown; no historical prediction-market data is invented."]}
