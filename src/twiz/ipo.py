"""User-maintained IPO terms and illustrative playbook scenarios.

Only rows with verified=True and a source_url can produce price context or
supply events. Release windows must be supplied explicitly from the issuer's
terms; there is no universal 15-day release or 180-day lockup assumption.
"""

import math
from datetime import date, timedelta
from typing import Any, Dict, List, Optional


def _verified(row: Dict[str, Any]) -> bool:
    return row.get("verified") is True and bool(str(row.get("source_url") or "").strip())


def _offset(row: Dict[str, Any], key: str) -> Optional[int]:
    try:
        value = int(row[key])
        return value if value >= 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def ipo_stage(days_since: int, early_release_days: int, lockup_days: int) -> str:
    if days_since < 0:
        return "pre_ipo"
    if days_since <= 5:
        return "debut_window"
    if days_since < early_release_days:
        return "pre_early_release"
    if days_since < lockup_days:
        return "early_release_open"
    return "post_lockup"


def ipo_events(
    ipos: List[Dict[str, Any]], start: date, horizon_days: int = 45
) -> List[Dict[str, Any]]:
    """Calendar events derived from tracked IPOs (for events.py consumers)."""
    end = start + timedelta(days=horizon_days)
    events = []
    for row in ipos or []:
        if not _verified(row):
            continue
        try:
            ipo_day = date.fromisoformat(str(row.get("ipo_date", ""))[:10])
        except ValueError:
            continue
        ticker = str(row.get("ticker", "")).upper()
        for offset, kind, name, rule in [
            (
                _offset(row, "early_release_days"),
                "ipo_early_release",
                f"{ticker} early-release window opens",
                "Early holders can start selling; expect supply pressure and "
                "volatility. His rule: do not chase strength into this window.",
            ),
            (
                _offset(row, "lockup_days"),
                "ipo_lockup_expiry",
                f"{ticker} lockup expiry",
                "Full lockup expiry: the largest supply unlock. Watch for the "
                "fall toward the IPO price as the long-term entry zone.",
            ),
        ]:
            if offset is None:
                continue
            when = ipo_day + timedelta(days=offset)
            if start <= when <= end:
                events.append(
                    {
                        "date": when.isoformat(),
                        "kind": kind,
                        "name": name,
                        "importance": 4,
                        "tickers": [ticker],
                        "precision": "verified_config",
                        "source_url": row["source_url"],
                        "playbook_rule": rule,
                        "days_away": (when - start).days,
                    }
                )
    return events


def build_ipo_snapshot(
    row: Dict[str, Any],
    close: Optional[float],
    as_of: date,
) -> Dict[str, Any]:
    if not _verified(row):
        return {
            "ticker": str(row.get("ticker", "")).upper(),
            "stage": "unverified", "verified": False, "verdicts": [],
            "warnings": [
                "IPO terms have no verified source; supply events and price signals withheld."
            ],
        }
    ipo_day = date.fromisoformat(str(row.get("ipo_date"))[:10])
    ipo_price = float(row.get("ipo_price") or 0)
    if not math.isfinite(ipo_price) or ipo_price <= 0:
        raise ValueError("IPO price must be positive and finite")
    early = _offset(row, "early_release_days")
    lockup = _offset(row, "lockup_days")
    days_since = (as_of - ipo_day).days
    pop_pct = None
    if close and math.isfinite(close) and ipo_price:
        pop_pct = round((close / ipo_price - 1.0) * 100.0, 1)
    verdicts: List[str] = []
    stage = (
        ipo_stage(days_since, early, lockup)
        if early is not None and lockup is not None else "terms_incomplete"
    )
    if pop_pct is not None:
        if stage in ("debut_window", "pre_early_release") and pop_pct >= 50:
            verdicts.append(
                "Debut pop >= 50%: his rule is sell into strength, do not chase."
            )
        if stage == "pre_early_release":
            verdicts.append(
                f"Early-release window opens in {early - days_since}d — supply shock risk."
            )
        if pop_pct >= 100:
            verdicts.append(
                "Price is 2x+ the IPO price in the hype phase; the historical "
                "pattern is a return to valuation."
            )
        if -10 <= pop_pct <= 25 and stage in ("early_release_open", "post_lockup"):
            verdicts.append(
                "Back near the IPO price after the unlock — his long-term entry zone."
            )
    return {
        "ticker": str(row.get("ticker", "")).upper(),
        "verified": True,
        "source_url": row["source_url"],
        "name": row.get("name", row.get("ticker", "")),
        "ipo_date": ipo_day.isoformat(),
        "ipo_price": ipo_price or None,
        "days_since_ipo": days_since,
        "stage": stage,
        "close": close,
        "vs_ipo_price_pct": pop_pct,
        "early_release_date": (
            (ipo_day + timedelta(days=early)).isoformat() if early is not None else None
        ),
        "lockup_expiry_date": (
            (ipo_day + timedelta(days=lockup)).isoformat() if lockup is not None else None
        ),
        "verdicts": verdicts,
    }


def build_ipo_snapshots(
    ipos: List[Dict[str, Any]],
    market_data: Dict[str, List[Dict[str, Any]]],
    as_of: date,
) -> Dict[str, Dict[str, Any]]:
    snapshots: Dict[str, Dict[str, Any]] = {}
    for row in ipos or []:
        ticker = str(row.get("ticker", "")).upper()
        rows = market_data.get(ticker, [])
        close = float(rows[-1]["close"]) if rows else None
        try:
            snapshots[ticker] = build_ipo_snapshot(row, close, as_of)
        except (TypeError, ValueError):
            continue
    return snapshots
