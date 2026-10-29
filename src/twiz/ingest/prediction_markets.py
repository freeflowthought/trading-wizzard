"""Read-only Polymarket event beliefs. Market prices are neither facts nor stock forecasts."""
import hashlib
import json
import math
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from ..config import DEFAULT_EVENT_MAIN_CASE_THRESHOLD, validate_event_threshold

API_ROOT = "https://gamma-api.polymarket.com"
MAX_REQUESTS = 10
BUDGET_SECONDS = 20
REQUEST_TIMEOUT = 5


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _timestamp(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def _array(value: Any) -> List[Any]:
    value = json.loads(value) if isinstance(value, str) else value
    return value if isinstance(value, list) else []


def _fetch_market(market_id: str, timeout: float) -> Dict[str, Any]:
    request = Request(f"{API_ROOT}/markets/{quote(market_id, safe='')}",
                      headers={"User-Agent": "trading-wizzard/0.1 personal research"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Market endpoint did not return an object.")
    return payload


def _contract_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {}
    for key, source in (
        ("description", "description"), ("created_at", "createdAt"),
        ("start_date", "startDate"), ("end_date", "endDate"),
        ("resolution_source", "resolutionSource"), ("market_slug", "slug"),
    ):
        metadata[key] = payload.get(source) if isinstance(payload.get(source), str) else None
    for key, source in (("clob_token_ids", "clobTokenIds"), ("outcomes", "outcomes")):
        try:
            metadata[key] = [str(value) for value in _array(payload.get(source))
                             if isinstance(value, (str, int)) and not isinstance(value, bool)]
        except (TypeError, ValueError):
            metadata[key] = []
    metadata["raw_response_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()).hexdigest()
    return metadata


def _market_row(
    item: Dict[str, Any], payload: Dict[str, Any], now: datetime,
    threshold: float = DEFAULT_EVENT_MAIN_CASE_THRESHOLD,
) -> Dict[str, Any]:
    row = {key: item.get(key) for key in (
        "id", "label", "topic", "driver", "yes_scenario", "no_scenario", "settlement_scope_note"
    )}
    row.update(question=payload.get("question"), yes_probability=None, no_probability=None,
               status="unavailable", reason="", market_updated_at=None,
               fetched_at=now.isoformat(), end_date=None, liquidity_usd=None, spread=None,
               url=None, source="Polymarket Gamma",
               price_type="outcomePrices_market_implied_belief",
               classification_threshold=threshold,
               contract_metadata=_contract_metadata(payload))
    updated, end = _timestamp(payload.get("updatedAt")), _timestamp(payload.get("endDate"))
    row["market_updated_at"] = updated.isoformat() if updated else None
    row["end_date"] = end.isoformat() if end else None
    liquidity = _number(payload.get("liquidityNum"))
    if liquidity is None:
        liquidity = _number(payload.get("liquidity"))
    row["liquidity_usd"] = liquidity
    spread = _number(payload.get("spread"))
    bid, ask = _number(payload.get("bestBid")), _number(payload.get("bestAsk"))
    if bid is not None and ask is not None and 0 <= bid <= ask <= 1:
        spread = max(ask-bid, spread) if spread is not None else ask-bid
    row["spread"] = spread
    events = payload.get("events") or []
    slug = events[0].get("slug") if events and isinstance(events[0], dict) else payload.get("slug")
    if isinstance(slug, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-]*", slug):
        row["url"] = f"https://polymarket.com/event/{slug}"
    reason = ""
    if str(payload.get("id", "")) != str(item.get("id", "")):
        reason = "Returned market ID does not match the configured market."
    elif not isinstance(payload.get("question"), str) or not payload["question"].strip():
        reason = "Exact market question is unavailable."
    elif item.get("expected_question") and payload["question"] != item["expected_question"]:
        reason = "Exact market question does not match the configured contract."
    elif not (payload.get("active") is True and payload.get("closed") is False
              and payload.get("archived") is False):
        reason = "Market is closed, archived, inactive, or its state is unknown."
    elif updated is None or updated > now or now-updated > timedelta(hours=24):
        reason = "Market update time is missing, unzoned, future-dated, or older than 24 hours."
    elif end is None or end <= now:
        reason = "Market end date is missing, unzoned, or expired."
    elif liquidity is None or liquidity < 1000:
        reason = "Market liquidity is unavailable or below $1,000."
    elif payload.get("spread") is not None and (
        _number(payload["spread"]) is None or _number(payload["spread"]) < 0
    ):
        reason = "Reported market spread is invalid."
    elif any(payload.get(key) is not None and (value is None or not 0 <= value <= 1)
             for key, value in (("bestBid", bid), ("bestAsk", ask))):
        reason = "Market bid or ask is invalid."
    elif bid is not None and ask is not None and bid > ask:
        reason = "Market quotes are crossed."
    elif spread is None or not 0 <= spread <= 0.10 + 1e-12:
        reason = "Market spread is unavailable, invalid, or wider than 0.10."
    try:
        outcomes, prices = _array(payload.get("outcomes")), _array(payload.get("outcomePrices"))
        labels = [str(value).strip().lower() for value in outcomes]
        values = [_number(value) for value in prices]
        if (len(labels) != 2 or set(labels) != {"yes", "no"} or len(values) != 2
                or any(value is None or not 0 <= value <= 1 for value in values)
                or abs(sum(values)-1) > 0.02 + 1e-12):
            raise ValueError("invalid binary prices")
    except (TypeError, ValueError):
        reason = reason or "Market must have two valid Yes/No prices summing approximately to one."
        labels, values = [], []
    if not reason:
        yes, no = values[labels.index("yes")], values[labels.index("no")]
        yes_main, no_main = yes >= threshold, no >= threshold
        status = "yes_leaning" if yes_main and not no_main else (
            "no_leaning" if no_main and not yes_main else "uncertain")
        row.update(yes_probability=yes, no_probability=no, status=status,
                   reason="Market-implied belief for this exact question; not a verified event.")
    else:
        row["reason"] = reason
    return row


def fetch_prediction_markets(
    config: Dict[str, Any], report_date: str, live: bool = False,
    now: Optional[datetime] = None,
    discovery_audit: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Discover equity events when enabled; legacy explicit watchlists remain supported."""
    if not live:
        return [], ["Polymarket live collection skipped; no probabilities inferred offline."]
    threshold = validate_event_threshold(config.get(
        "main_case_threshold", DEFAULT_EVENT_MAIN_CASE_THRESHOLD
    ))
    injected_clock = now
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        return [], ["Polymarket requires a timezone-aware current timestamp."]
    now = now.astimezone(timezone.utc)
    try:
        local_day = now.astimezone(ZoneInfo(config.get("timezone", "America/Toronto"))).date()
        if date.fromisoformat(report_date) != local_day:
            return [], ["Polymarket current beliefs cannot be backfilled into a historical report."]
    except (TypeError, ValueError, KeyError):
        return [], ["Polymarket report date or local timezone is invalid."]
    if (config.get("discovery") or {}).get("enabled") is True:
        from .prediction_discovery import discover_prediction_markets

        return discover_prediction_markets(config, report_date, now=injected_clock,
                                          audit=discovery_audit)
    deadline = time.monotonic() + BUDGET_SECONDS
    rows, warnings, cache = [], [], {}
    requests = 0
    for item in config.get("markets", []):
        market_id = str(item.get("id", ""))
        error, payload = "", {}
        if not re.fullmatch(r"\d+", market_id):
            error = "Configured market ID must be a numeric Gamma market ID."
        elif market_id in cache:
            payload, error = cache[market_id]
        elif requests >= MAX_REQUESTS or time.monotonic() >= deadline:
            error = "Polymarket request count or collection time budget exhausted."
        else:
            requests += 1
            try:
                timeout = max(0.001, min(REQUEST_TIMEOUT, deadline-time.monotonic()))
                payload = _fetch_market(market_id, timeout=timeout)
            except (OSError, ValueError) as exc:
                error = f"Polymarket fetch failed: {exc}"
            cache[market_id] = (payload, error)
        row = _market_row(item, payload, now, threshold)
        if error:
            row.update(status="unavailable", reason=error,
                       yes_probability=None, no_probability=None)
        if row["status"] == "unavailable":
            warnings.append(f"Polymarket {market_id or 'unknown'}: {row['reason']}")
        rows.append(row)
    return rows, warnings
