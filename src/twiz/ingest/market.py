import math
from datetime import date, timedelta
from typing import Dict, Iterable, List, Tuple

from ..config import WatchedAsset


def synthetic_ohlcv(ticker: str, days: int = 260) -> List[Dict[str, float]]:
    seed = sum(ord(char) for char in ticker)
    base = 80.0 + seed % 240
    rows: List[Dict[str, float]] = []
    current = base
    start = date.today() - timedelta(days=days * 2)
    trading_days = []
    cursor = start
    while len(trading_days) < days:
        if cursor.weekday() < 5:
            trading_days.append(cursor)
        cursor += timedelta(days=1)

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
    period: str = "1y",
) -> Tuple[List[Dict[str, float]], List[str]]:
    warnings: List[str] = []
    try:
        import yfinance as yf
    except Exception as exc:
        return [], [f"yfinance unavailable for {ticker}: {exc}"]

    try:
        history = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    except Exception as exc:
        return [], [f"yfinance fetch failed for {ticker}: {exc}"]

    if history is None or history.empty:
        return [], [f"yfinance returned no OHLCV rows for {ticker}"]

    rows: List[Dict[str, float]] = []
    for index, row in history.iterrows():
        rows.append(
            {
                "date": index.date().isoformat(),
                "open": float(row.get("Open", 0.0)),
                "high": float(row.get("High", 0.0)),
                "low": float(row.get("Low", 0.0)),
                "close": float(row.get("Close", 0.0)),
                "volume": float(row.get("Volume", 0.0)),
            }
        )
    return rows, warnings


def fetch_market_data(
    watchlist: Iterable[WatchedAsset],
    live: bool = False,
    period: str = "1y",
) -> Tuple[Dict[str, List[Dict[str, float]]], List[str]]:
    data: Dict[str, List[Dict[str, float]]] = {}
    warnings: List[str] = []
    for asset in watchlist:
        rows: List[Dict[str, float]] = []
        if live:
            rows, provider_warnings = fetch_yfinance_ohlcv(asset.ticker, period=period)
            warnings.extend(provider_warnings)
        if not rows:
            rows = synthetic_ohlcv(asset.ticker)
            warnings.append(
                f"Using deterministic sample OHLCV for {asset.ticker}; "
                "pass --live with a configured provider for current market data."
            )
        data[asset.ticker] = rows
    return data, warnings
