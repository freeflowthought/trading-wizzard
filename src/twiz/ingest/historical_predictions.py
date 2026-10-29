"""Archive observed token history, never current prices masquerading as past beliefs.

The Data API's coarse series is downloaded retrospectively. Bucket-end availability
is a conservative proxy, not proof that today's contract wording existed then.
"""

import hashlib
import json
import math
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

HISTORY_ENDPOINT = "https://data-api.polymarket.com/v2/prices-history"
HISTORY_DOCUMENTATION = (
    "https://docs.polymarket.com/api-reference/markets/get-a-tokens-price-history"
)
SCHEMA_VERSION = "historical-predictions-v1"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
LIMITATIONS = [
    "Retrospectively downloaded provider history, not a contemporaneous archive.",
    "available_at is a conservative bucket-end proxy, not verified publication time.",
    "Current contract rules are not independently verified at each historical origin.",
    "Creation/start gates use current Gamma metadata; prior revisions are unavailable.",
    "Source endDate is a conservative cutoff, not a verified FOMC announcement time.",
    "Zero-resolution terminal points are excluded; they can include settlement payouts.",
    "Current outcomePrices/liquidity/spread/volume are never historical feature inputs.",
    "No interpolation, complementary-probability imputation, or event-sum normalization.",
]


def _utc(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _array(value: Any) -> list:
    result = json.loads(value) if isinstance(value, str) else value
    return result if isinstance(result, list) else []


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode()


def _write_new(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def _fetch_bytes(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": "trading-wizzard/0.1 historical research"})
    with urlopen(request, timeout=timeout) as response:
        return response.read(MAX_RESPONSE_BYTES + 1)


def _source_contracts(
    wrapper: dict, filename: str, source_sha256: str, as_of: datetime,
) -> List[dict]:
    """Bind exact token/outcome pairs to one saved official Gamma event response."""
    if not isinstance(wrapper, dict):
        raise ValueError("Gamma source wrapper must be an object.")
    parsed = urlparse(str(wrapper.get("request_url", "")))
    slug = parse_qs(parsed.query).get("slug", [])
    fetched = _utc(wrapper.get("fetched_at"))
    if (parsed.scheme != "https" or parsed.netloc != "gamma-api.polymarket.com"
            or parsed.path != "/events" or len(slug) != 1 or fetched is None
            or fetched > as_of):
        raise ValueError("Unverified Gamma source URL or source capture chronology.")
    events = wrapper.get("response")
    if not isinstance(events, list):
        raise ValueError("Gamma source response must be an event array.")
    if not events:
        return []
    if len(events) != 1 or not isinstance(events[0], dict):
        raise ValueError("Expected one exact event per slug source file.")
    event = events[0]
    if (event.get("slug") != slug[0] or not re.fullmatch(r"\d+", str(event.get("id", "")))
            or not re.search(r"\bfed decision\b", str(event.get("title", "")), re.I)):
        raise ValueError("Source event slug, numeric ID, or Fed identity mismatch.")
    markets = event.get("markets")
    if not isinstance(markets, list) or not markets:
        raise ValueError("Source event has no bracket markets.")
    contracts = []
    for market in markets:
        if not isinstance(market, dict):
            raise ValueError("Malformed market in source event.")
        labels = _array(market.get("outcomes"))
        tokens = _array(market.get("clobTokenIds"))
        if (len(labels) != 2 or set(labels) != {"Yes", "No"} or len(tokens) != 2
                or len(set(str(token) for token in tokens)) != 2
                or any(not re.fullmatch(r"\d+", str(token)) for token in tokens)
                or not re.fullmatch(r"\d+", str(market.get("id", "")))
                or not re.search(r"\bfed\b", str(market.get("question", "")), re.I)
                or not re.search(r"\brates?\b", str(market.get("question", "")), re.I)):
            raise ValueError("Invalid binary Fed market identity or token/outcome binding.")
        starts, ends, closure_dates = [], [], []
        for entity in (event, market):
            for key in ("createdAt", "startDate"):
                if entity.get(key) is not None:
                    timestamp = _utc(entity[key])
                    if timestamp is None:
                        raise ValueError(f"Missing timezone or invalid {key} gate.")
                    starts.append(timestamp)
            end = _utc(entity.get("endDate"))
            if end is None:
                raise ValueError("Missing timezone or invalid source endDate gate.")
            ends.append(end)
            for key in ("closedTime", "umaEndDate", "resolvedAt"):
                if entity.get(key) is not None:
                    timestamp = _utc(entity[key])
                    if timestamp is None:
                        raise ValueError(f"Missing timezone or invalid {key} closure gate.")
                    closure_dates.append(timestamp)
        if (_utc(market.get("createdAt")) is None or not starts
                or min(ends).year not in (2025, 2026)
                or max(starts) >= min(ends + closure_dates) or max(starts) > fetched):
            raise ValueError("Invalid 2025–2026 market creation/end chronology.")
        for label, token in zip(labels, tokens):
            contracts.append({
                "event_id": str(event["id"]), "event_slug": event["slug"],
                "event_title": event.get("title"), "market_id": str(market["id"]),
                "question": market.get("question"), "outcome": label, "token_id": str(token),
                "price_type": "historical_outcome_token_market_price",
                "source_file": f"sources/{filename}", "source_sha256": source_sha256,
                "source_fetched_at": fetched.isoformat(),
                "valid_from": max(starts).isoformat(),
                "source_end_date": min(ends).isoformat(),
                "cutoff": min(ends + closure_dates + [as_of]).isoformat(),
                "source_closed_or_resolved_at": [value.isoformat() for value in closure_dates],
                "end_date_precision": "source_timestamp_not_verified_announcement_time",
                "contract_metadata": {key: market.get(native) for key, native in (
                    ("description", "description"), ("created_at", "createdAt"),
                    ("start_date", "startDate"), ("end_date", "endDate"),
                    ("resolution_source", "resolutionSource"), ("condition_id", "conditionId"))},
                "rules_version": "current_download_not_verified_historical",
            })
    return contracts


def _terminal_cutoff(points: List[Any], start: datetime, end: datetime) -> Optional[datetime]:
    cutoffs = []
    for point in points:
        if not isinstance(point, dict):
            continue
        timestamp, price = _number(point.get("timestamp")), _number(point.get("price"))
        if (_number(point.get("resolution_seconds")) != 0 or timestamp is None
                or timestamp != int(timestamp) or price is None or not 0 <= price <= 1):
            continue
        try:
            observed = datetime.fromtimestamp(timestamp, timezone.utc)
            if start <= observed < end:
                cutoffs.append(observed)
        except (ValueError, OverflowError, OSError):
            continue
    return min(cutoffs) if cutoffs else None


def normalize_history(
    points: List[Any], contract: dict, requested_bucket_seconds: int,
) -> tuple:
    """Reject terminal, impossible, conflicting, and not-yet-available observations."""
    start, end = _utc(contract["valid_from"]), _utc(contract["cutoff"])
    terminal = _terminal_cutoff(points, start, end)
    counts, by_timestamp, conflicts = Counter(), {}, set()
    for point in points:
        if not isinstance(point, dict):
            counts["malformed"] += 1
            continue
        timestamp, price = _number(point.get("timestamp")), _number(point.get("price"))
        resolution = _number(point.get("resolution_seconds"))
        if (timestamp is None or timestamp != int(timestamp) or timestamp <= 0
                or price is None or not 0 <= price <= 1 or resolution is None
                or resolution != int(resolution) or not 0 <= resolution <= 86400):
            counts["malformed"] += 1
            continue
        if resolution == 0:
            counts["zero_resolution_terminal"] += 1
            continue
        try:
            observed = datetime.fromtimestamp(timestamp, timezone.utc)
            available = observed + timedelta(seconds=max(resolution, requested_bucket_seconds))
        except (ValueError, OverflowError, OSError):
            counts["malformed"] += 1
            continue
        if observed < start:
            counts["before_creation_or_start"] += 1
            continue
        if terminal is not None and (observed >= terminal or available > terminal):
            counts["after_terminal_or_bucket_not_available"] += 1
            continue
        if observed >= end or available > end:
            counts["after_cutoff_or_bucket_not_available"] += 1
            continue
        row = {"t": int(timestamp), "p": price, "resolution_seconds": int(resolution),
               "observed_at": observed.isoformat(), "available_at": available.isoformat(),
               "available_at_method": "conservative_bucket_end_proxy",
               "historical_publication_time_verified": False,
               "raw_references": [point["_raw_reference"]] if "_raw_reference" in point else []}
        if timestamp in by_timestamp:
            prior = by_timestamp[timestamp]
            if any(row[key] != prior[key] for key in row if key != "raw_references"):
                conflicts.add(timestamp)
                counts["conflicting_timestamp"] += 1
            else:
                counts["duplicate"] += 1
                row["raw_references"] = prior["raw_references"] + row["raw_references"]
        by_timestamp[timestamp] = row
    return [by_timestamp[t] for t in sorted(by_timestamp) if t not in conflicts], dict(counts)


def _event_diagnostics(contracts: List[dict]) -> List[dict]:
    diagnostics = []
    for event_id in sorted({row["event_id"] for row in contracts}):
        yes = [row for row in contracts if row["event_id"] == event_id and row["outcome"] == "Yes"]
        maps = [{(point["t"], point["available_at"]): point for point in row["history"]}
                for row in yes]
        common = set.intersection(*(set(points) for points in maps)) if maps else set()
        sums = [sum(points[t]["p"] for points in maps) for t in sorted(common)]
        diagnostics.append({
            "event_id": event_id, "rate_brackets": len(yes),
            "brackets_with_history": sum(bool(row["history"]) for row in yes),
            "exact_common_timestamps": len(common),
            "yes_sum_min": min(sums) if sums else None,
            "yes_sum_max": max(sums) if sums else None,
            "yes_sum_outside_0_95_to_1_05": sum(abs(value - 1) > 0.05 for value in sums),
            "interpretation": "Diagnostic only: exact shared observation/availability times; "
                              "no normalization "
                              "or assumed historical mutual-exclusivity validation.",
        })
    return diagnostics


def collect_historical_predictions(
    source_dir: Path, output_root: Path, *, bucket_seconds: int = 10800,
    max_requests: int = 120, max_pages_per_token: int = 3,
    total_budget_seconds: float = 600, request_timeout: float = 10,
    now: Optional[datetime] = None,
    fetcher: Optional[Callable[[str, float], bytes]] = None,
    progress: Optional[Callable[[dict], None]] = None,
) -> Dict[str, Any]:
    """Capture a unique append-only run. Failure/empty history means unknown, not zero."""
    as_of = now or datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        raise ValueError("Analysis clock must include a timezone.")
    as_of = as_of.astimezone(timezone.utc)
    if (isinstance(bucket_seconds, bool) or bucket_seconds not in (10800, 43200)
            or isinstance(max_requests, bool) or not isinstance(max_requests, int)
            or not 0 <= max_requests <= 1000
            or isinstance(max_pages_per_token, bool) or not isinstance(max_pages_per_token, int)
            or not 1 <= max_pages_per_token <= 10
            or _number(total_budget_seconds) is None or not 0 < total_budget_seconds <= 3600
            or _number(request_timeout) is None or not 0 < request_timeout <= 30):
        raise ValueError("Invalid history density, request count, pagination, or time budget.")
    sources = sorted(Path(source_dir).glob("*.json"))
    if not sources:
        raise ValueError("No source event files were found.")
    run = Path(output_root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-")
                               + uuid.uuid4().hex[:12])
    run.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def save(relative, content):
        hashes[relative] = _write_new(run / relative, content)
        return hashes[relative]

    save("collector.py", Path(__file__).read_bytes())
    candidates, source_audit = [], []
    for path in sources:
        body = path.read_bytes()
        digest = save(f"sources/{path.name}", body)
        receipt = {"file": path.name, "sha256": digest}
        try:
            found = _source_contracts(json.loads(body), path.name, digest, as_of)
            candidates.extend(found)
            receipt.update(status="verified" if found else "empty_response", tokens=len(found))
        except (ValueError, TypeError, KeyError) as exc:
            receipt.update(status="invalid_source", reason=str(exc), tokens=0)
        source_audit.append(receipt)
    token_counts = Counter(row["token_id"] for row in candidates)
    market_counts = Counter((row["market_id"], row["outcome"]) for row in candidates)
    contracts = [row for row in candidates if token_counts[row["token_id"]] == 1
                 and market_counts[(row["market_id"], row["outcome"])] == 1]
    rejected_duplicates = len(candidates) - len(contracts)
    deadline, request_count = time.monotonic() + total_budget_seconds, 0
    get = fetcher or _fetch_bytes
    for contract in contracts:
        points, receipts, cursor, cursors, complete, error = [], [], None, set(), False, ""
        for page in range(max_pages_per_token):
            remaining = deadline - time.monotonic()
            if request_count >= max_requests or remaining <= 0:
                error = "Collection request count or elapsed time budget exhausted."
                break
            query = {"token_id": contract["token_id"], "interval": "max",
                     "bucket_seconds": bucket_seconds, "limit": 10000}
            if cursor:
                query["cursor"] = cursor
            url = HISTORY_ENDPOINT + "?" + urlencode(query)
            receipt = {"request_url": url, "query": query,
                       "request_started_at": datetime.now(timezone.utc).isoformat()}
            request_count += 1
            body, payload = None, None
            try:
                body = get(url, min(request_timeout, remaining))
                receipt["fetched_at"] = datetime.now(timezone.utc).isoformat()
                receipt["http_status"] = 200
                if not isinstance(body, bytes):
                    raise ValueError("History transport must return exact response bytes.")
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ValueError("History response exceeded byte budget; stored prefix only.")
                payload = json.loads(body)
                if (not isinstance(payload, dict) or not isinstance(payload.get("data"), list)
                        or not isinstance(payload.get("pagination"), dict)):
                    raise ValueError("Invalid history data/pagination response envelope.")
            except HTTPError as exc:
                body = exc.read(MAX_RESPONSE_BYTES + 1)
                receipt.update(http_status=exc.code, error_type=type(exc).__name__)
                error = f"HTTP status {exc.code}; history unknown."
            except (OSError, ValueError, TypeError) as exc:
                receipt["error_type"] = type(exc).__name__
                error = str(exc)
            receipt.setdefault("fetched_at", datetime.now(timezone.utc).isoformat())
            stem = f"raw/{contract['market_id']}-{contract['outcome'].lower()}-{page:02d}"
            if isinstance(body, bytes):
                receipt["body_file"] = stem + ".body"
                receipt["body_sha256"] = save(stem + ".body", body[:MAX_RESPONSE_BYTES + 1])
                receipt["body_truncated"] = len(body) > MAX_RESPONSE_BYTES
            receipt["error"] = error or None
            save(stem + ".json", _json_bytes(receipt))
            receipts.append(receipt)
            if error:
                break
            points.extend({**item, "_raw_reference": {
                "body_file": receipt["body_file"], "data_index": offset,
            }} if isinstance(item, dict) else item for offset, item in enumerate(payload["data"]))
            pagination = payload["pagination"]
            cursor = pagination.get("next_cursor")
            if pagination.get("has_more") is False and cursor is None:
                complete = True
                break
            if (pagination.get("has_more") is not True or not isinstance(cursor, str)
                    or not cursor or cursor in cursors or len(cursor) > 8192):
                error = "Invalid or repeated history pagination cursor."
                break
            cursors.add(cursor)
        if not complete and not error:
            error = "Per-token pagination budget exhausted; series may be incomplete."
        history, rejected = normalize_history(points, contract, bucket_seconds)
        terminal = _terminal_cutoff(points, _utc(contract["valid_from"]), _utc(contract["cutoff"]))
        contract.update(history=history, status=("downloaded" if complete else "partial")
                        if history else "unknown", pagination_complete=complete,
                        error=error or None, points_received=len(points), rejected_points=rejected,
                        response_files=[receipt.get("body_file") for receipt in receipts],
                        fetched_at=receipts[-1]["fetched_at"] if receipts else None,
                        request_count=len(receipts), historical_pit_verified=False,
                        terminal_cutoff=terminal.isoformat() if terminal else None,
                        coverage_start=history[0]["available_at"] if history else None,
                        coverage_end=history[-1]["available_at"] if history else None)
        relative = f"history/{contract['market_id']}-{contract['outcome'].lower()}.json"
        contract["history_file"] = relative
        save(relative, _json_bytes(contract))
        if progress:
            progress({key: contract[key] for key in ("market_id", "outcome", "status")}
                     | {"observations": len(history)})
    summary = {
        "source_files": len(sources), "source_statuses": dict(Counter(
            row["status"] for row in source_audit)),
        "events": len({row["event_id"] for row in contracts}),
        "markets": len({row["market_id"] for row in contracts}), "tokens": len(contracts),
        "tokens_with_history": sum(bool(row["history"]) for row in contracts),
        "observations": sum(len(row["history"]) for row in contracts),
        "status_counts": dict(Counter(row["status"] for row in contracts)),
        "requests": request_count, "duplicate_bindings_excluded": rejected_duplicates,
    }
    index = {"schema_version": SCHEMA_VERSION, "analysis_clock_utc": as_of.isoformat(),
             "source_audit": source_audit, "summary": summary,
             "contracts": [{key: value for key, value in row.items() if key != "history"}
                           for row in contracts],
             "event_sum_diagnostics": _event_diagnostics(contracts), "limitations": LIMITATIONS}
    save("index.json", _json_bytes(index))
    manifest = {"schema_version": SCHEMA_VERSION, "completed_at": datetime.now(
        timezone.utc).isoformat(), "analysis_clock_utc": as_of.isoformat(),
        "provider": "Polymarket Data API v2", "endpoint": HISTORY_ENDPOINT,
        "documentation": HISTORY_DOCUMENTATION, "bucket_seconds": bucket_seconds,
        "budgets": {"max_requests": max_requests, "max_pages_per_token": max_pages_per_token,
                    "total_budget_seconds": total_budget_seconds,
                    "request_timeout": request_timeout},
        "historical_pit_verified": False, "summary": summary, "files": hashes,
        "limitations": LIMITATIONS}
    manifest_hash = _write_new(run / "manifest.json", _json_bytes(manifest))
    _write_new(run / "manifest.sha256", (manifest_hash + "\n").encode())
    return {"path": str(run), "manifest_sha256": manifest_hash, "summary": summary}


def verify_event_history_run(
    path: Path, expected_manifest_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Read-only integrity check; a valid hash never certifies historical availability."""
    path = Path(path)
    errors = []
    try:
        body = (path / "manifest.json").read_bytes()
        expected = (path / "manifest.sha256").read_text().strip()
        if hashlib.sha256(body).hexdigest() != expected:
            errors.append("manifest hash mismatch")
        if (expected_manifest_sha256
                and hashlib.sha256(body).hexdigest() != expected_manifest_sha256):
            errors.append("manifest does not match the externally recorded receipt")
        manifest = json.loads(body)
        for relative, digest in manifest["files"].items():
            target = path / relative
            if (target.is_symlink() or not target.resolve().is_relative_to(path.resolve())
                    or not target.is_file()
                    or hashlib.sha256(target.read_bytes()).hexdigest() != digest):
                errors.append(f"file missing, unsafe, or hash mismatch: {relative}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    return {"valid": not errors, "errors": errors, "historical_pit_verified": False}
