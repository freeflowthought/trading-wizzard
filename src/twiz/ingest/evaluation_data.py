"""Download evidence for a frozen research cohort without claiming vintage archives."""

import hashlib
import json
import math
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode

from ..sessions import expected_complete_session, is_us_equity_session
from ..storage import write_json
from .public_macro import FRED_CSV, _get


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def fetch_research_asset(
    ticker: str, start: str, end: str, now: Optional[datetime] = None,
) -> Dict[str, Any]:
    if now is not None and now.tzinfo is None:
        raise ValueError("Research analysis time must include a timezone.")
    import yfinance as yf

    requested_at = datetime.now(timezone.utc).isoformat()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        history = yf.Ticker(ticker).history(
            start=start, end=(date.fromisoformat(end)+timedelta(days=1)).isoformat(),
            interval="1d", auto_adjust=False, timeout=15, raise_errors=True,
        )
    captured = datetime.now(timezone.utc)
    analysis_clock = now or captured
    cutoff = expected_complete_session(end, now=analysis_clock)
    rows = []
    for index, item in history.iterrows():
        day = index.date().isoformat()
        values = {key: _finite(item.get(native)) for key, native in (
            ("open", "Open"), ("high", "High"), ("low", "Low"),
            ("close", "Close"), ("volume", "Volume"))}
        if (not start <= day <= min(end, cutoff)
                or not is_us_equity_session(date.fromisoformat(day))
                or any(v is None for v in values.values())
                or values["volume"] < 0
                or min(values[key] for key in ("open", "high", "low", "close")) <= 0
                or not values["low"] <= min(values["open"], values["close"])
                <= max(values["open"], values["close"]) <= values["high"]):
            continue
        rows.append({"date": day, **values, "source": "yfinance", "session_complete": True,
                     "price_basis": "Yahoo Close auto_adjust=False; "
                                    "provider historical split basis",
                     "fetched_at": captured.isoformat(),
                     "pit_status": "downloaded_reconstructed_history"})
    rows.sort(key=lambda row: row["date"])
    return {"ticker": ticker, "source": "yfinance", "fetched_at": captured.isoformat(),
            "request_started_at": requested_at,
            "analysis_clock_utc": analysis_clock.astimezone(timezone.utc).isoformat(),
            "completed_session_cutoff": cutoff,
            "requested_start": start, "requested_end": end, "rows": rows,
            "coverage_start": rows[0]["date"] if rows else None,
            "coverage_end": rows[-1]["date"] if rows else None,
            "pit_status": "downloaded_history_not_contemporaneous_archive"}


def fetch_research_prices(
    tickers: Iterable[str], folder: Path, start: str, end: str, workers: int = 4,
    now: Optional[datetime] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """One evidence file per asset; failed assets remain explicit, never generated."""
    if now is not None and now.tzinfo is None:
        raise ValueError("Research analysis time must include a timezone.")
    folder.mkdir(parents=True, exist_ok=True)
    market, sources = {}, []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as executor:
        pending = {executor.submit(fetch_research_asset, ticker, start, end, now=now): ticker
                   for ticker in dict.fromkeys(tickers)}
        for future in as_completed(pending):
            ticker = pending[future]
            try:
                asset = future.result()
            except Exception as exc:
                asset = {"ticker": ticker, "rows": [], "status": "unavailable",
                         "error_type": type(exc).__name__, "requested_start": start,
                         "requested_end": end}
            payload = json.dumps(asset, sort_keys=True, allow_nan=False).encode()
            digest = hashlib.sha256(payload).hexdigest()
            path = folder / f"{ticker.replace('^', 'index-')}-{digest[:16]}.json"
            if not path.exists():
                path.write_bytes(payload)
            market[ticker] = asset["rows"]
            sources.append({"ticker": ticker, "path": str(path), "sha256": digest,
                            "rows": len(asset["rows"]), "start": asset.get("coverage_start"),
                            "end": asset.get("coverage_end"),
                            "status": asset.get("status", "downloaded"),
                            "pit_status": asset.get("pit_status", "unavailable")})
            print(f"{ticker}: {len(asset['rows'])} observed bars", flush=True)
    write_json(folder.parent / "market.json", market)
    write_json(folder.parent / "market-manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "assets": sources,
        "availability": "Reconstructed daily history; historical publication time unverified.",
        "raw_close_definition": "No dividend adjustment; provider historical splits apply.",
    })
    return market


def fetch_research_macro(folder: Path, start: str, end: str) -> List[Dict[str, Any]]:
    import csv
    import io

    folder.mkdir(parents=True, exist_ok=True)
    rows, sources = [], []
    for series in ("DGS10", "DGS2", "DGS30", "BAMLH0A0HYM2", "DCOILBRENTEU"):
        url = FRED_CSV+"?"+urlencode({"id": series, "cosd": start, "coed": end})
        requested_at = datetime.now(timezone.utc).isoformat()
        try:
            payload = _get(url)
            captured = datetime.now(timezone.utc).isoformat()
            parsed = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
            history = []
            for item in parsed:
                day = str(item.get("observation_date") or item.get("DATE") or "")
                value = _finite(item.get(series))
                if start <= day <= end and value is not None:
                    history.append({"date": day, "value": value})
            if not history:
                raise ValueError("No dated numeric observations.")
            history.sort(key=lambda item: item["date"])
            digest = hashlib.sha256(payload).hexdigest()
            path = folder / f"{series}-{digest[:16]}.csv"
            if not path.exists():
                path.write_bytes(payload)
            rows.append({"series_id": series, "source": "FRED public CSV", "url": url,
                         "fetched_at": captured, "history": history, **history[-1],
                         "request_started_at": requested_at,
                         "unit": "USD/barrel" if series == "DCOILBRENTEU" else "percent",
                         "pit_status": "revised_history_not_historical_vintage",
                         "availability_assumption": ("One completed session lag; historical "
                                                     "publication time unverified.")})
            sources.append({"series_id": series, "path": str(path), "sha256": digest,
                            "rows": len(history), "start": history[0]["date"],
                            "end": history[-1]["date"], "status": "downloaded"})
        except (OSError, ValueError) as exc:
            sources.append({"series_id": series, "status": "unavailable",
                            "error_type": type(exc).__name__})
        print(f"{series}: {sources[-1]['status']}", flush=True)
    write_json(folder.parent / "macro.json", rows)
    write_json(folder.parent / "macro-manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "pit_certified": False, "no_historical_publication_times_in_download": True,
    })
    return rows
