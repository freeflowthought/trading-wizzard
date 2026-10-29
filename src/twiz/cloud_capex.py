"""Dated SEC cash-PP&E/revenue comparisons; no prices or inferred cloud guidance."""

import copy
import hashlib
import math
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from .sessions import is_us_equity_session

NY = ZoneInfo("America/New_York")
ISSUERS = {"MSFT": (789019, 7), "GOOGL": (1652044, 1), "META": (1326801, 1)}
CAPEX = "PaymentsToAcquirePropertyPlantAndEquipment"
REVENUE = ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet")
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
METRICS = ("cash_capex_yoy_pct", "revenue_yoy_pct", "capex_revenue_intensity_pct",
           "cash_capex_yoy_acceleration_pp", "revenue_yoy_acceleration_pp")


def _available(filed):
    day = date.fromisoformat(filed) + timedelta(days=1)
    while not is_us_equity_session(day):
        day += timedelta(days=1)
    return datetime.combine(day, time(16, 15), NY)


def _period(start, end, fiscal_month):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if (start != first.isoformat() or end != last.isoformat()
            or (first.month, first.day) != (fiscal_month, 1)):
        return None
    for months in (3, 6, 9, 12):
        total = first.year * 12 + first.month - 1 + months
        boundary = date(total // 12, total % 12 + 1, 1) - timedelta(days=1)
        if boundary == last:
            return {"start": start, "end": end, "cumulative_months": months}
    return None


def _prior(period):
    # Exact endpoints only. No silent leap-day, week-calendar or duration tolerance.
    if not period:
        return None
    try:
        return {**period, **{k: date.fromisoformat(period[k]).replace(
            year=date.fromisoformat(period[k]).year - 1).isoformat() for k in ("start", "end")}}
    except ValueError:
        return None


def _index(company, ticker):
    """Read metadata only, retaining raw fact references without accessing val."""
    events = []
    facts_root = company.get("facts")
    concepts = facts_root.get("us-gaap", {}) if isinstance(facts_root, dict) else {}
    if not isinstance(concepts, dict):
        return []
    for tag in (CAPEX, *REVENUE):
        concept = concepts.get(tag)
        units = concept.get("units") if isinstance(concept, dict) else None
        facts = units.get("USD", []) if isinstance(units, dict) else []
        if not isinstance(facts, list):
            continue
        for index, fact in enumerate(facts):
            if not isinstance(fact, dict) or fact.get("form") not in FORMS:
                continue
            try:
                period = _period(fact.get("start"), fact.get("end"), ISSUERS[ticker][1])
                available = _available(fact.get("filed"))
                if (not period or date.fromisoformat(fact["filed"]).isoformat() != fact["filed"]
                        or period["end"] > fact["filed"]):
                    continue
            except (TypeError, ValueError, OverflowError):
                continue
            events.append({"kind": "capex" if tag == CAPEX else "revenue", "tag": tag,
                           "period": period, "available": available, "filed": fact["filed"],
                           "pointer": f"/facts/us-gaap/{tag}/units/USD/{index}", "fact": fact})
    return sorted(events, key=lambda r: (r["available"], r["filed"], r["pointer"]))


def _select(records, ticker, source_hash, refs):
    if not records:
        return {"status": "unknown", "value": None, "reason": "no_available_exact_period_fact",
                "refs": []}
    # The event state already contains only the latest eligible FILED date,
    # across revenue aliases too. Bad latest versions never fall back to older ones.
    values, ids, invalid = [], [], False
    for record in records:
        fact = record["fact"]
        ref_id = hashlib.sha256((source_hash + record["pointer"]).encode()).hexdigest()
        refs[ref_id] = {"ticker": ticker, "source_sha256": source_hash,
                        "json_pointer": record["pointer"], "tag": record["tag"], "unit": "USD",
                        **record["period"], "filed": record["filed"],
                        "available_at_proxy": record["available"].isoformat(),
                        "accn": fact.get("accn"), "form": fact.get("form"),
                        "reported_fy": fact.get("fy"), "reported_fp": fact.get("fp"),
                        "frame": fact.get("frame")}
        ids.append(ref_id)
        value = fact.get("val")  # Only reached AFTER eligible availability was processed.
        if (type(value) not in (int, float) or not isinstance(fact.get("accn"), str)
                or not fact["accn"].strip()):
            invalid = True
            continue
        try:
            number = float(value)
        except (ValueError, OverflowError):
            number = math.nan
        if not math.isfinite(number) or number < 0:
            invalid = True
        values.append(number)
    reason = ("invalid_latest_fact" if invalid else "conflicting_latest_versions_or_aliases"
              if len(set(values)) != 1 else None)
    return {"status": "unknown" if reason else "available", "reason": reason,
            "value": None if reason else values[0], "refs": ids,
            "filed": records[0]["filed"],
            "available_at_proxy": records[0]["available"].isoformat()}


def _pair(state, period, ticker, source_hash, refs):
    pair = {"period": period}
    for kind in ("capex", "revenue"):
        records = state.get((kind, period["start"], period["end"]), []) if period else []
        pair[kind] = _select(records, ticker, source_hash, refs)
    pair["status"] = "available" if all(pair[k]["status"] == "available"
                                        for k in ("capex", "revenue")) else "unknown"
    if pair["status"] == "available" and pair["revenue"]["value"] <= 0:
        pair["status"] = "unknown"
        pair["revenue"].update(status="unknown", reason="nonpositive_revenue", value=None)
    return pair


def _growth(current, prior):
    if current["status"] != "available" or prior["status"] != "available":
        return None
    if prior["capex"]["value"] <= 0:
        return None
    values = {"cash_capex_yoy_pct": 100 * (current["capex"]["value"] /
                                         prior["capex"]["value"] - 1),
              "revenue_yoy_pct": 100 * (current["revenue"]["value"] /
                                       prior["revenue"]["value"] - 1)}
    return values if all(math.isfinite(v) for v in values.values()) else None


def _snapshot(state, ticker, source_hash, refs):
    # A newer known revenue period also makes an older cash period incomplete;
    # do not present the previous quarter as the current filing's comparison.
    latest = max((records[0]["period"] for records in state.values()),
                 key=lambda p: p["end"], default=None)
    if not latest:
        return {"status": "unknown", "reason": "no_available_cumulative_cash_ppe",
                **dict.fromkeys(METRICS)}
    current = _pair(state, latest, ticker, source_hash, refs)
    prior = _pair(state, _prior(latest), ticker, source_hash, refs)
    growth = _growth(current, prior)
    result = {"status": "available" if growth else "partial" if current["status"] == "available"
              else "unknown",
              "reason": None if growth else "latest_period_or_exact_prior_unavailable",
              "period": latest, "current": current, "prior": prior,
              "acceleration_reason": "same_duration_three_year_comparison_unavailable",
              **dict.fromkeys(METRICS)}
    if current["status"] == "available":
        intensity = 100 * current["capex"]["value"] / current["revenue"]["value"]
        if math.isfinite(intensity):
            result["capex_revenue_intensity_pct"] = intensity
        else:
            result.update(status="unknown", reason="nonfinite_derived_intensity")
    if growth:
        result.update(growth)
        older = _pair(state, _prior(_prior(latest)), ticker, source_hash, refs)
        previous_growth = _growth(prior, older)
        if previous_growth:
            accelerations = {name.replace("_pct", "_acceleration_pp"): growth[name] - value
                             for name, value in previous_growth.items()}
            if all(math.isfinite(v) for v in accelerations.values()):
                result.update(accelerations, previous_same_duration=older, acceleration_reason=None)
    return result


def _aggregate(issuers):
    included = [name for name, row in issuers.items() if row["status"] == "available"]
    result = {"status": "available" if len(included) >= 2 else "insufficient_issuers",
              "included_issuers": included,
              "excluded_issuers": {name: row["reason"] for name, row in issuers.items()
                                   if name not in included}, "issuer_count": len(included)}
    for metric in METRICS:
        values = [issuers[name][metric] for name in included if issuers[name][metric] is not None]
        ordered = sorted(values)
        mid = len(ordered) // 2
        result["median_" + metric] = (
            ordered[mid] if len(ordered) % 2 else ordered[mid-1]/2 + ordered[mid]/2
        ) if len(values) >= 2 else None
        result[metric + "_issuer_count"] = len(values)
    return result


def build_cloud_capex_timeline(companyfacts_by_ticker, sessions, source_sha256_by_ticker):
    """Return one as-of row per declared completed US session and exact source refs.

    Next-session 16:15 NY is a conservative FILED-DATE proxy, not a verified SEC
    acceptance clock or historical API vintage. Calendarized totals, quarter
    differencing, finance leases, Amazon productive assets and cloud guidance
    are deliberately outside this cash-PP&E/total-issuer-revenue definition.
    """
    if (not isinstance(companyfacts_by_ticker, dict)
            or not isinstance(source_sha256_by_ticker, dict)):
        raise ValueError("Companyfacts and captured body hashes must be mappings.")
    if not isinstance(sessions, list) or any(type(day) is not str for day in sessions):
        raise ValueError("Sessions must be a sorted unique list of ISO dates.")
    try:
        days = [date.fromisoformat(day) for day in sessions]
    except ValueError as error:
        raise ValueError("Invalid session date.") from error
    if (sessions != sorted(set(sessions)) or any(day.isoformat() != text
                                               or not is_us_equity_session(day)
                                               for day, text in zip(days, sessions))):
        raise ValueError("Sessions must be sorted, unique completed US session dates.")
    indexed, invalid = {}, {}
    for ticker, (cik, _) in ISSUERS.items():
        company = companyfacts_by_ticker.get(ticker)
        if company is None:
            invalid[ticker] = "missing_companyfacts"
            continue
        source_hash = source_sha256_by_ticker.get(ticker)
        if (not isinstance(source_hash, str) or len(source_hash) != 64
                or not set(source_hash) <= set("0123456789abcdef")):
            raise ValueError("Captured raw body SHA256 required for " + ticker)
        if (not isinstance(company, dict) or type(company.get("cik")) is not int
                or company["cik"] != cik):
            invalid[ticker] = "issuer_cik_mismatch_or_missing"
            continue
        indexed[ticker] = _index(company, ticker)
    states = {ticker: {} for ticker in indexed}
    cursors, cached, refs, output = dict.fromkeys(indexed, 0), {}, {}, []
    for day in days:
        clock = datetime.combine(day, time(16, 15), NY)
        for ticker, events in indexed.items():
            cursor, changed = cursors[ticker], False
            while cursor < len(events) and events[cursor]["available"] <= clock:
                event = events[cursor]
                key = (event["kind"], event["period"]["start"], event["period"]["end"])
                previous = states[ticker].get(key)
                if not previous or event["filed"] > previous[0]["filed"]:
                    states[ticker][key] = [event]
                elif event["filed"] == previous[0]["filed"]:
                    previous.append(event)
                cursor, changed = cursor + 1, True
            cursors[ticker] = cursor
            if changed or ticker not in cached:
                cached[ticker] = _snapshot(
                    states[ticker], ticker, source_sha256_by_ticker[ticker], refs)
        issuers = {ticker: copy.deepcopy(cached[ticker]) if ticker in cached else {
            "status": "unknown", "reason": invalid[ticker], **dict.fromkeys(METRICS)}
            for ticker in ISSUERS}
        used_refs = {ref for issuer in issuers.values()
                     for name in ("current", "prior", "previous_same_duration")
                     for kind in ("capex", "revenue")
                     for ref in issuer.get(name, {}).get(kind, {}).get("refs", [])}
        output.append({"date": day.isoformat(), "as_of": clock.isoformat(),
                       "available_at": clock.isoformat(), "issuers": issuers,
                       "source_refs": {key: copy.deepcopy(refs[key]) for key in sorted(used_refs)},
                       "aggregate": _aggregate(issuers)})
    return {"schema_version": "sec-cloud-cash-ppe-timeline-v1", "rows": output,
            "source_refs": refs, "source_sha256_by_ticker": dict(source_sha256_by_ticker),
            "excluded_issuers": {
                "AMZN": "strict_cash_PPE_missing_after_2017_broader_productive_assets_excluded"},
            "method": {"availability": "next_US_session_after_filed_date_at_16:15_New_York",
                       "periods": "exact_fiscal_YTD_3_6_9_12_months; no leap/date tolerance",
                       "versions": "latest_available_filed_date_across_aliases; conflicts_unknown",
                       "acceleration": "YoY_difference_vs_prior_year_same_YTD_duration_"
                                       "using_as_of_versions",
                       "aggregate": "median_of_at_least_two_comparable_issuer_growths; "
                                    "no fiscal-dollar sum"},
            "historical_pit_verified": False,
            "limitations": [
                "Filed-date proxy is not verified acceptance/publication time or API vintage.",
                "Latest missing/invalid period is not replaced with an older convenient period.",
                "Total issuer revenue/cash PP&E are not cloud revenue, AI spend or lease guidance.",
                "Median constituent periods differ across fiscal calendars; no dollar total.",
                "Acceleration compares same-YTD-duration as-of versions, not sequential quarters."]}
