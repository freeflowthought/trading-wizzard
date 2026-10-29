import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

from ..config import WatchedAsset
from ..sessions import expected_complete_session


def _confirmed_metadata_close(
    instrument: Any, bar_date: date, row: Any, now: datetime
) -> Optional[float]:
    """Repair only a missing daily close from a timestamped, completed regular session."""
    try:
        metadata = instrument.history_metadata
        regular_end = datetime.fromtimestamp(
            int(metadata["currentTradingPeriod"]["regular"]["end"]), timezone.utc
        )
        quote_time = datetime.fromtimestamp(int(metadata["regularMarketTime"]), timezone.utc)
        exchange_zone = ZoneInfo(metadata["exchangeTimezoneName"])
        close = float(metadata["regularMarketPrice"])
        low, high = float(row.get("Low")), float(row.get("High"))
        if (
            regular_end.astimezone(exchange_zone).date() != bar_date
            or quote_time.astimezone(exchange_zone).date() != bar_date
            or not regular_end <= quote_time <= regular_end + timedelta(minutes=5)
            or now < regular_end + timedelta(minutes=15)
            or quote_time > now
            or not all(math.isfinite(value) and value > 0 for value in (close, low, high))
            or not low <= close <= high
        ):
            return None
        return close
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, OSError):
        return None


def synthetic_ohlcv(
    ticker: str, days: int = 260, as_of: Optional[date] = None
) -> List[Dict[str, Any]]:
    seed = sum(ord(char) for char in ticker)
    base = 80.0 + seed % 240
    rows: List[Dict[str, Any]] = []
    current = base
    trading_days = []
    cursor = as_of or date.today()
    while len(trading_days) < days:
        if cursor.weekday() < 5:
            trading_days.append(cursor)
        cursor -= timedelta(days=1)
    trading_days.reverse()

    for index, trading_day in enumerate(trading_days):
        drift = 0.00045 + ((seed % 13) - 6) / 100000
        cycle = math.sin((index + seed % 19) / 8.0) * 0.012
        shock = math.sin((index + seed) / 17.0) * 0.006
        previous = current
        current = max(5.0, current * (1.0 + drift + cycle + shock))
        high = max(previous, current) * 1.006
        low = min(previous, current) * 0.994
        rows.append(
            {
                "date": trading_day.isoformat(),
                "source": "sample",
                "open": round(previous, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(current, 2),
                "volume": float(1_000_000 + (seed % 100) * 10_000 + index * 750),
            }
        )
    return rows


def fetch_yfinance_ohlcv(
    ticker: str,
    period: str = "2y",
    as_of: Optional[date] = None,
    now: Optional[datetime] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    try:
        import yfinance as yf
    except Exception as exc:
        return [], [f"yfinance unavailable for {ticker}: {exc}"]

    try:
        kwargs = {
            # Preserve OHLC when Yahoo's latest Close/Adj Close is null, then adjust
            # valid historical bars locally. auto_adjust=True propagates that null
            # into Open/High/Low before we can validate a closing metadata quote.
            "period": period, "interval": "1d", "auto_adjust": False,
            "timeout": 10, "raise_errors": True,
        }
        if as_of is not None:
            # Yahoo end is exclusive. Never include bars after the report date.
            kwargs["end"] = (as_of + timedelta(days=1)).isoformat()
            days = {"1y": 366, "2y": 732, "5y": 1830}.get(period, 366)
            kwargs["start"] = (as_of - timedelta(days=days)).isoformat()
            kwargs.pop("period")
        instrument = yf.Ticker(ticker)
        history = instrument.history(**kwargs)
    except Exception as exc:
        return [], [f"yfinance fetch failed for {ticker}: {exc}"]

    if history is None or history.empty:
        return [], [f"yfinance returned no OHLCV rows for {ticker}"]

    rows: List[Dict[str, Any]] = []
    now = now or datetime.now(timezone.utc)
    complete_through = expected_complete_session((as_of or now.date()).isoformat(), now=now)
    dropped = 0
    for index, row in history.iterrows():
        if index.date().isoformat() > complete_through:
            continue
        close_source = "daily_history"
        try:
            missing_close = not math.isfinite(float(row.get("Close")))
        except (TypeError, ValueError):
            missing_close = True
        if missing_close and index.date().isoformat() == complete_through:
            confirmed = _confirmed_metadata_close(instrument, index.date(), row, now)
            if confirmed is not None:
                row = dict(row)
                row["Close"] = confirmed
                close_source = "regular_market_metadata"
                warnings.append(
                    f"{ticker}: daily close missing; used regular-session metadata "
                    f"after its confirmed close on {index.date().isoformat()}."
                )
        try:
            valid = all(
                math.isfinite(float(row.get(key, 0)))
                and float(row.get(key, 0)) > 0
                for key in ("Open", "High", "Low", "Close")
            ) and math.isfinite(float(row.get("Volume", 0))) and float(row.get("Volume", 0)) >= 0
        except (ValueError, TypeError, OverflowError):
            valid = False
        if not valid:
            dropped += 1
            continue
        adjustment = 1.0
        try:
            adjusted_close = float(row.get("Adj Close"))
            if (
                close_source == "daily_history"
                and math.isfinite(adjusted_close) and adjusted_close > 0
            ):
                adjustment = adjusted_close / float(row["Close"])
        except (TypeError, ValueError, OverflowError):
            pass
        rows.append(
            {
                "date": index.date().isoformat(),
                "source": "yfinance",
                "close_source": close_source,
                "fetched_at": now.isoformat(),
                "session_complete": True,
                "open": float(row.get("Open", 0.0)) * adjustment,
                "high": float(row.get("High", 0.0)) * adjustment,
                "low": float(row.get("Low", 0.0)) * adjustment,
                "close": float(row.get("Close", 0.0)) * adjustment,
                "raw_open": float(row.get("Open", 0.0)),
                "raw_high": float(row.get("High", 0.0)),
                "raw_low": float(row.get("Low", 0.0)),
                "raw_close": float(row.get("Close", 0.0)),
                "provider_adjusted_close": float(row.get("Close", 0.0)) * adjustment,
                "price_basis": "yahoo_adjusted_history",
                "volume": float(row.get("Volume", 0.0)),
            }
        )
    if dropped:
        latest = rows[-1]["date"] if rows else "unavailable"
        warnings.append(
            f"{ticker}: discarded {dropped} invalid OHLCV bars; latest valid bar {latest}."
        )
    return rows, warnings


def fetch_market_data(
    watchlist: Iterable[WatchedAsset],
    live: bool = False,
    period: str = "2y",
    as_of: Optional[date] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    data: Dict[str, List[Dict[str, Any]]] = {}
    warnings: List[str] = []
    for asset in watchlist:
        rows: List[Dict[str, float]] = []
        if live:
            rows, provider_warnings = fetch_yfinance_ohlcv(
                asset.ticker, period=period, as_of=as_of
            )
            warnings.extend(provider_warnings)
        if not rows and not live:
            rows = synthetic_ohlcv(asset.ticker, as_of=as_of)
            warnings.append(
                f"Using deterministic sample OHLCV for {asset.ticker}; "
                "pass --live with a configured provider for current market data."
            )
        if live and not rows:
            warnings.append(
                f"Market data unavailable for {asset.ticker}; no sample prices substituted."
            )
        data[asset.ticker] = rows
    return data, warnings


def market_source(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "unavailable"
    sources = {row.get("source", "unverified") for row in rows}
    if "sample" in sources:
        return "sample"
    if sources == {"yfinance"}:
        return "yfinance"
    return "unverified"


def valid_market_rows(
    rows: List[Dict[str, Any]], as_of: date
) -> List[Dict[str, Any]]:
    """Validate cached bars too, including artifacts written by older versions."""
    valid = {}
    for row in rows:
        try:
            day = date.fromisoformat(str(row.get("date", ""))[:10])
            if day > as_of or "close" not in row:
                continue
            if not all(
                math.isfinite(float(row[key])) and float(row[key]) > 0
                for key in ("open", "high", "low", "close") if key in row
            ):
                continue
            if "volume" in row and (
                not math.isfinite(float(row["volume"])) or float(row["volume"]) < 0
            ):
                continue
        except (TypeError, ValueError, OverflowError):
            continue
        valid[day] = row
    return [valid[day] for day in sorted(valid)]
