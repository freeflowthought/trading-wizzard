"""Append-only Yahoo daily bars and action evidence for a frozen research universe.

Price and action ledgers come from the same response. Nothing here joins forecasts
to outcomes, adjusts prices, infers fills, or certifies historical announcement time.
"""

import hashlib
import io
import json
import math
import re
import sys
import time
import uuid
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from ..sessions import expected_complete_session, is_us_equity_session

SCHEMA_VERSION = "swing-market-actions-v1"
CORE = {"^GSPC", "^SOX", "QQQ", "SPY"}
AMBIGUOUS = {"CBRS", "SPCX"}
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
PRICE_BASIS = ("Yahoo historical split-adjusted OHLC; no dividend adjustment "
               "or local split adjustment")
LIMITATIONS = [
    "Downloaded reconstructed history, not a contemporaneous publication archive.",
    "Yahoo historical equity/ETF prices already reflect provider split adjustments.",
    "Do not blindly apply the action ratios again to these historical price series.",
    "Dividend amounts retain provider-reported basis; historical nominal amount unverified.",
    "Effective/ex-dividend date is not a declaration, payment, or known-at timestamp.",
    "Publication time, original action announcement, and prior data revisions are unknown.",
    "This is not a complete merger, spinoff, delisting, tax, or all-corporate-actions tape.",
    "Short histories remain short; no prices are filled before listing or across missing sessions.",
    "No forecast outcomes, trading fills, campaign profit, or parameter optimization performed.",
]


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def _fetch_bytes(url, timeout):
    """Use Yahoo's existing provider transport when installed, no account credentials."""
    try:
        from curl_cffi import requests
    except ImportError:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=timeout) as response:
            return response.read(MAX_RESPONSE_BYTES + 1)
    try:
        response = requests.get(url, timeout=timeout, impersonate="chrome")
    except requests.exceptions.RequestException as exc:
        raise OSError(f"Yahoo public transport failed: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise HTTPError(url, response.status_code, "Yahoo chart HTTP error", {},
                        io.BytesIO(response.content))
    return response.content


def build_swing_universe(existing_market: dict, annotation: dict) -> dict:
    """Literal symbols/explicit same-post antecedents only; ambiguous aliases stay out."""
    universe = {ticker: {"ticker": ticker, "origins": ["existing_dataset"], "evidence": []}
                for ticker in existing_market if re.fullmatch(r"\^?[A-Z][A-Z0-9.-]{0,9}", ticker)
                and ticker not in AMBIGUOUS}
    for ticker in CORE:
        row = universe.setdefault(ticker, {"ticker": ticker, "origins": [], "evidence": []})
        row["origins"].append("core_requested_asset")
    excluded = []
    for item in annotation.get("candidates", []):
        ticker = item.get("ticker")
        if ticker is None:
            continue
        evidence, context = item.get("evidence") or {}, item.get("same_post_context") or ""
        clause = str(evidence.get("exact_clause", ""))
        pattern = r"(?<![A-Za-z0-9])" + re.escape(str(ticker)) + r"(?![A-Za-z0-9])"
        # OPEN must be literal uppercase: 'Open AI' is not the real-estate ticker.
        flags = 0 if ticker == "OPEN" else re.I
        literal = bool(re.search(pattern, clause + "\n" + context, flags))
        basis = item.get("instrument_mapping_basis", "")
        valid_basis = basis in {
            "literal_index_or_symbol",
            "same_post_explicit_antecedent_or_prior_user_confirmed_SPX_level_convention",
        }
        if (ticker in AMBIGUOUS or not isinstance(ticker, str)
                or not re.fullmatch(r"\^?[A-Z][A-Z0-9.-]{0,9}", ticker)
                or not valid_basis or not literal):
            excluded.append({"candidate_id": item.get("id"), "ticker": ticker,
                             "reason": "Ambiguous alias or no literal/same-post symbol proof."})
            continue
        row = universe.setdefault(ticker, {"ticker": ticker, "origins": [], "evidence": []})
        if "author_candidate_literal_symbol" not in row["origins"]:
            row["origins"].append("author_candidate_literal_symbol")
        row["evidence"].append({"candidate_id": item.get("id"), "source": evidence.get("path"),
                                "line_start": evidence.get("line_start"), "mapping_basis": basis})
    return {"assets": [universe[ticker] for ticker in sorted(universe)],
            "excluded_candidate_mappings": excluded,
            "selection": "Core four + existing dataset + source-annotated explicit symbols; "
                         "not selected using price outcomes."}


def _timestamp(value, tz):
    number = _number(value)
    if number is None or number != int(number) or number <= 0:
        return None
    try:
        return datetime.fromtimestamp(number, tz)
    except (OSError, ValueError, OverflowError):
        return None


def parse_chart(payload: dict, ticker: str, start: str, cutoff: str) -> Dict[str, Any]:
    """Use exchange-local event/bar dates, never UTC-date shortcuts or local price repair."""
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if (not isinstance(chart, dict) or chart.get("error") is not None
            or not isinstance(chart.get("result"), list) or len(chart["result"]) != 1):
        raise ValueError("Yahoo chart response has an error or no unique result.")
    result = chart["result"][0]
    meta = result.get("meta") if isinstance(result, dict) else None
    if (not isinstance(meta, dict) or meta.get("symbol") != ticker
            or meta.get("instrumentType") not in ("EQUITY", "ETF", "INDEX")):
        raise ValueError("Yahoo chart identity/instrument type mismatch.")
    if meta.get("exchangeTimezoneName") not in ("America/New_York", "US/Eastern"):
        raise ValueError("Expected verified US listing exchange timezone metadata.")
    tz = ZoneInfo(meta["exchangeTimezoneName"])
    timestamps = result.get("timestamp")
    indicators = result.get("indicators") or {}
    if not isinstance(indicators, dict):
        raise ValueError("Malformed Yahoo chart indicators.")
    quotes = indicators.get("quote")
    if (not isinstance(timestamps, list) or not isinstance(quotes, list) or len(quotes) != 1
            or not isinstance(quotes[0], dict)):
        raise ValueError("Yahoo chart is missing its timestamp/OHLC arrays.")
    quote_rows = quotes[0]
    adjusted = indicators.get("adjclose") or []
    adjusted = adjusted[0].get("adjclose", []) if adjusted and isinstance(adjusted[0], dict) else []
    rejected, rows, duplicates = Counter(), {}, set()
    for offset, timestamp in enumerate(timestamps):
        observed = _timestamp(timestamp, tz)
        if observed is None:
            rejected["invalid_timestamp"] += 1
            continue
        day = observed.date().isoformat()
        if not start <= day <= cutoff or not is_us_equity_session(observed.date()):
            rejected["outside_completed_session_window"] += 1
            continue
        values = {key: _number(quote_rows.get(key, [])[offset])
                  if isinstance(quote_rows.get(key), list)
                  and offset < len(quote_rows[key]) else None
                  for key in ("open", "high", "low", "close", "volume")}
        if (any(value is None for value in values.values()) or values["volume"] < 0
                or min(values[key] for key in ("open", "high", "low", "close")) <= 0
                or not values["low"] <= min(values["open"], values["close"])
                <= max(values["open"], values["close"]) <= values["high"]):
            rejected["invalid_ohlcv"] += 1
            continue
        row = {"date": day, **values, "source_timestamp": timestamp,
               "source": "Yahoo chart", "session_complete": True,
               "price_basis": "Yahoo index level" if meta["instrumentType"] == "INDEX"
               else PRICE_BASIS,
               "provider_adjusted_close": _number(adjusted[offset])
               if isinstance(adjusted, list) and offset < len(adjusted) else None,
               "raw_json_pointer": "/chart/result/0/indicators/quote/0",
               "raw_array_index": offset,
               "raw_timestamp_pointer": f"/chart/result/0/timestamp/{offset}"}
        if day in rows:
            duplicates.add(day)
            rejected["duplicate_session"] += 1
        rows[day] = row
    bars = [rows[day] for day in sorted(rows) if day not in duplicates]
    actions, action_rejected = [], Counter()
    events = result.get("events", {})
    if not isinstance(events, dict):
        raise ValueError("Malformed Yahoo corporate action envelope.")
    for native, kind in (("splits", "split"), ("dividends", "dividend")):
        event_rows = events.get(native, {})
        if not isinstance(event_rows, dict):
            raise ValueError(f"Malformed Yahoo {native} action map.")
        for key, event in event_rows.items():
            if not isinstance(event, dict):
                action_rejected["malformed_event"] += 1
                continue
            observed = _timestamp(event.get("date"), tz)
            if observed is None or str(event.get("date")) != str(key):
                action_rejected["invalid_or_mismatched_event_timestamp"] += 1
                continue
            day = observed.date().isoformat()
            if not start <= day <= cutoff or not is_us_equity_session(observed.date()):
                action_rejected["outside_completed_session_window"] += 1
                continue
            action = {"ticker": ticker, "type": kind, "effective_date": day,
                      "source_timestamp": event["date"],
                      "exchange_timezone": meta["exchangeTimezoneName"],
                      "known_at": None, "historical_publication_time_verified": False,
                      "raw_json_pointer": f"/chart/result/0/events/{native}/{key}"}
            if kind == "split":
                numerator, denominator = _number(event.get("numerator")), _number(
                    event.get("denominator"))
                if numerator is None or denominator is None or min(numerator, denominator) <= 0:
                    action_rejected["invalid_split_ratio"] += 1
                    continue
                ratio = numerator / denominator
                if not math.isfinite(ratio) or ratio <= 0:
                    action_rejected["invalid_split_ratio"] += 1
                    continue
                reported = event.get("splitRatio")
                if reported is not None:
                    match = re.fullmatch(r"([0-9.]+)[:/]([0-9.]+)", str(reported))
                    numbers = [_number(value) for value in match.groups()] if match else []
                    if (not numbers or any(value is None or value <= 0 for value in numbers)
                            or not math.isclose(numbers[0] / numbers[1], ratio, rel_tol=1e-9)):
                        action_rejected["conflicting_split_ratio"] += 1
                        continue
                action.update(numerator=numerator, denominator=denominator,
                              ratio=ratio, provider_split_ratio=event.get(
                                  "splitRatio"), price_adjustment_applied=False)
            else:
                amount = _number(event.get("amount"))
                if amount is None or amount < 0:
                    action_rejected["invalid_dividend_amount"] += 1
                    continue
                action.update(amount=amount, amount_basis="Yahoo provider-reported historical "
                              "per-share amount; historical nominal basis unverified",
                              currency=event.get("currency"), listing_currency=meta.get("currency"),
                              payment_date=None, declaration_date=None)
            actions.append(action)
    return {"ticker": ticker, "exchange_timezone": meta["exchangeTimezoneName"],
            "instrument_type": meta["instrumentType"], "currency": meta.get("currency"),
            "first_trade_date_provider": meta.get("firstTradeDate"), "rows": bars,
            "actions": sorted(actions, key=lambda row: (row["effective_date"], row["type"])),
            "rejected_bars": dict(rejected), "rejected_actions": dict(action_rejected),
            "action_status": "partial_invalid_events" if action_rejected else
            ("not_applicable_index" if meta["instrumentType"] == "INDEX" and not actions else
             "observed_events" if actions else "no_events_returned"),
            "coverage_start": bars[0]["date"] if bars else None,
            "coverage_end": bars[-1]["date"] if bars else None,
            "latest_completed_session_present": bool(bars and bars[-1]["date"] == cutoff),
            "historical_pit_verified": False}


def collect_swing_data(
    existing_market_path: Path, annotation_path: Path, output_root: Path, *,
    start: str = "2016-01-01", now: Optional[datetime] = None,
    max_requests: int = 108, total_budget_seconds: float = 900,
    request_timeout: float = 15,
    fetcher: Optional[Callable[[str, float], bytes]] = None,
    progress: Optional[Callable[[dict], None]] = None,
) -> Dict[str, Any]:
    as_of = now or datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        raise ValueError("Swing research clock requires a timezone.")
    as_of = as_of.astimezone(timezone.utc)
    start_date = date.fromisoformat(start)
    if (isinstance(max_requests, bool) or not isinstance(max_requests, int)
            or not 0 <= max_requests <= 500 or not 0 < total_budget_seconds <= 3600
            or not 0 < request_timeout <= 30):
        raise ValueError("Invalid swing data collection budget.")
    local_day = as_of.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    cutoff = expected_complete_session(local_day, now=as_of)
    if start > cutoff:
        raise ValueError("Requested start is after the completed-session cutoff.")
    existing_body, annotation_body = Path(existing_market_path).read_bytes(), Path(
        annotation_path).read_bytes()
    universe = build_swing_universe(json.loads(existing_body), json.loads(annotation_body))
    run = Path(output_root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-")
                               + uuid.uuid4().hex[:12])
    run.mkdir(parents=True, exist_ok=False)
    files = {}

    def save(relative, body):
        target = run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(body)
        files[relative] = hashlib.sha256(body).hexdigest()
        return files[relative]

    save("collector.py", Path(__file__).read_bytes())
    from .. import sessions

    save("sessions.py", Path(sessions.__file__).read_bytes())
    save("inputs/annotation.json", annotation_body)
    save("inputs/previous-market-reference.json", _json({
        "path": str(Path(existing_market_path).resolve()),
        "sha256": hashlib.sha256(existing_body).hexdigest(), "bytes": len(existing_body),
        "use": "Universe/hash reference only; none of these OHLC rows are mixed into new series."}))
    save("universe.json", _json(universe))
    deadline, requests = time.monotonic() + total_budget_seconds, 0
    assets, market, actions = [], {}, []
    params = {"period1": int(datetime.combine(start_date, datetime.min.time(), ZoneInfo(
        "America/New_York")).timestamp()), "period2": int(datetime.combine(
            date.fromisoformat(cutoff) + timedelta(days=1), datetime.min.time(),
            ZoneInfo("America/New_York")).timestamp()),
        "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"}
    get = fetcher or _fetch_bytes
    for entry in universe["assets"]:
        ticker, parsed, attempts, error = entry["ticker"], None, [], ""
        stem = ticker.replace("^", "index-")
        for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
            remaining = deadline - time.monotonic()
            if requests >= max_requests or remaining <= 0:
                error = "Request count or elapsed collection budget exhausted."
                break
            url = f"https://{host}/v8/finance/chart/{quote(ticker, safe='')}?{urlencode(params)}"
            receipt = {"request_url": url, "query": params,
                       "request_started_at": datetime.now(timezone.utc).isoformat()}
            requests += 1
            body = None
            try:
                body = get(url, min(request_timeout, remaining))
                receipt["fetched_at"] = datetime.now(timezone.utc).isoformat()
                receipt["http_status"] = 200
                if not isinstance(body, bytes) or len(body) > MAX_RESPONSE_BYTES:
                    raise ValueError("Invalid response bytes or byte budget exceeded.")
                parsed = parse_chart(json.loads(body), ticker, start, cutoff)
                error = ""
            except HTTPError as exc:
                body = exc.read(MAX_RESPONSE_BYTES + 1)
                receipt["http_status"] = exc.code
                error = f"Yahoo HTTP {exc.code}; data unknown."
            except (ValueError, TypeError, KeyError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            receipt.setdefault("fetched_at", datetime.now(timezone.utc).isoformat())
            receipt["error"] = error or None
            if isinstance(body, bytes):
                relative = f"raw/{stem}-{len(attempts)}.body"
                receipt.update(body_file=relative, body_sha256=save(
                    relative, body[:MAX_RESPONSE_BYTES + 1]),
                    body_truncated=len(body) > MAX_RESPONSE_BYTES)
            save(f"raw/{stem}-{len(attempts)}.json", _json(receipt))
            attempts.append(receipt)
            if parsed is not None:
                break
        if parsed is None:
            parsed = {"ticker": ticker, "rows": [], "actions": [], "action_status": "unknown",
                      "status": "unavailable", "error": error}
        else:
            parsed.update(status="downloaded" if parsed["latest_completed_session_present"]
                          else "incomplete_latest_session", error=None)
        parsed.update(source="Yahoo chart public API", requested_start=start,
                      completed_session_cutoff=cutoff, fetched_at=attempts[-1]["fetched_at"]
                      if attempts else None, attempts=attempts)
        raw_file = next((item.get("body_file") for item in reversed(attempts)
                         if item.get("error") is None), None)
        for row in parsed["rows"] + parsed["actions"]:
            row.update(raw_body_file=raw_file, fetched_at=parsed["fetched_at"],
                       historical_pit_verified=False)
        save(f"assets/{stem}.json", _json(parsed))
        market[ticker] = parsed["rows"]
        actions.extend(parsed["actions"])
        asset = {key: value for key, value in parsed.items() if key not in ("rows", "actions")}
        asset.update(ohlc_rows=len(parsed["rows"]), action_rows=len(parsed["actions"]),
                     asset_file=f"assets/{stem}.json")
        assets.append(asset)
        if progress:
            progress({key: asset[key] for key in ("ticker", "status", "ohlc_rows", "action_rows")})
    save("market.json", _json(market))
    save("actions.json", _json(actions))
    summary = {"assets": len(assets), "requests": requests,
               "asset_statuses": dict(Counter(row["status"] for row in assets)),
               "action_statuses": dict(Counter(row["action_status"] for row in assets)),
               "ohlc_rows": sum(len(rows) for rows in market.values()),
               "actions": len(actions),
               "action_types": dict(Counter(row["type"] for row in actions))}
    save("index.json", _json({"schema_version": SCHEMA_VERSION, "assets": assets,
                              "summary": summary, "limitations": LIMITATIONS}))
    manifest = {"schema_version": SCHEMA_VERSION, "analysis_clock_utc": as_of.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(), "requested_start": start,
                "completed_session_cutoff": cutoff, "query": params, "files": files,
                "runtime_python": sys.version, "summary": summary, "limitations": LIMITATIONS,
                "historical_pit_verified": False, "outcome_join_performed": False}
    digest = hashlib.sha256(_json(manifest)).hexdigest()
    with (run / "manifest.json").open("xb") as handle:
        handle.write(_json(manifest))
    with (run / "manifest.sha256").open("xb") as handle:
        handle.write((digest + "\n").encode())
    return {"path": str(run), "manifest_sha256": digest, "summary": summary}
