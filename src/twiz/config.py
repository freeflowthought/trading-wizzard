import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover - exercised when PyYAML is absent
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WatchedAsset:
    ticker: str
    name: str
    aliases: List[str]
    reddit_terms: List[str]


DEFAULT_WATCHLIST = [
    WatchedAsset("SPY", "SPDR S&P 500 ETF Trust", ["$SPY", "SPY", "S&P 500"], ["SPY", "$SPY"]),
    WatchedAsset("QQQ", "Invesco QQQ Trust", ["$QQQ", "QQQ", "Nasdaq 100"], ["QQQ", "$QQQ"]),
    WatchedAsset("SMH", "VanEck Semiconductor ETF", ["$SMH", "SMH", "semis"], ["SMH", "$SMH"]),
    WatchedAsset("SOXX", "iShares Semiconductor ETF", ["$SOXX", "SOXX"], ["SOXX", "$SOXX"]),
    WatchedAsset("NVDA", "NVIDIA", ["$NVDA", "NVDA", "NVIDIA", "Nvidia"], ["NVDA", "$NVDA"]),
    WatchedAsset("AMD", "Advanced Micro Devices", ["$AMD", "AMD"], ["AMD", "$AMD"]),
    WatchedAsset("AVGO", "Broadcom", ["$AVGO", "AVGO", "Broadcom"], ["AVGO", "$AVGO"]),
    WatchedAsset("TSM", "Taiwan Semiconductor", ["$TSM", "TSM", "TSMC"], ["TSM", "$TSM"]),
    WatchedAsset("MSFT", "Microsoft", ["$MSFT", "MSFT", "Microsoft"], ["MSFT", "$MSFT"]),
    WatchedAsset("AAPL", "Apple", ["$AAPL", "AAPL", "Apple"], ["AAPL", "$AAPL"]),
]

DEFAULT_SUBREDDITS = ["stocks", "investing", "wallstreetbets", "options", "SecurityAnalysis"]
DEFAULT_IGNORE_TERMS = ["AI", "GDP", "CPI", "Fed"]

DEFAULT_FEEDS = {
    "rss_feeds": [
        {
            "name": "Fed press releases",
            "source_type": "fed_rss",
            "url": "https://www.federalreserve.gov/feeds/press_all.xml",
        },
        {
            "name": "Fed speeches",
            "source_type": "fed_rss",
            "url": "https://www.federalreserve.gov/feeds/speeches.xml",
        },
    ],
    "fred_series": [
        {"id": "DFF", "name": "Effective Federal Funds Rate"},
        {"id": "DGS10", "name": "10-Year Treasury Constant Maturity"},
        {"id": "CPIAUCSL", "name": "Consumer Price Index"},
        {"id": "UNRATE", "name": "Unemployment Rate"},
    ],
    "news_keywords": ["FOMC", "Federal Reserve", "Treasury", "CPI", "PCE", "IPO"],
}


def load_env_file(path: Optional[Path] = None, override: bool = False) -> None:
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and (override or key not in os.environ):
            os.environ[key] = value


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


def load_watchlist(path: Optional[Path] = None) -> List[WatchedAsset]:
    path = path or PROJECT_ROOT / "config" / "watchlist.yml"
    data = _read_yaml(path)
    assets = data.get("watchlist") if data else None
    if not assets:
        return list(DEFAULT_WATCHLIST)

    watchlist = []
    for row in assets:
        ticker = str(row["ticker"]).upper()
        watchlist.append(
            WatchedAsset(
                ticker=ticker,
                name=str(row.get("name", ticker)),
                aliases=[str(alias) for alias in row.get("aliases", [ticker])],
                reddit_terms=[str(term) for term in row.get("reddit_terms", [ticker])],
            )
        )
    return watchlist


def load_watchlist_config(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or PROJECT_ROOT / "config" / "watchlist.yml"
    data = _read_yaml(path)
    return {
        "watchlist": load_watchlist(path),
        "subreddits": data.get("subreddits", DEFAULT_SUBREDDITS) if data else DEFAULT_SUBREDDITS,
        "ignore_terms": (
            data.get("ignore_terms", DEFAULT_IGNORE_TERMS) if data else DEFAULT_IGNORE_TERMS
        ),
    }


def load_feeds_config(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or PROJECT_ROOT / "config" / "feeds.yml"
    data = _read_yaml(path)
    if not data:
        return dict(DEFAULT_FEEDS)
    merged = dict(DEFAULT_FEEDS)
    merged.update(data)
    return merged
