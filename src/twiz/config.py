import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover - exercised when PyYAML is absent
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENT_MAIN_CASE_THRESHOLD = 0.65


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
    WatchedAsset("UMC", "United Microelectronics Corporation", ["$UMC", "UMC"], ["UMC", "$UMC"]),
    WatchedAsset("TSLA", "Tesla", ["$TSLA", "TSLA", "Tesla"], ["TSLA", "$TSLA"]),
    WatchedAsset("ORCL", "Oracle", ["$ORCL", "ORCL", "Oracle"], ["ORCL", "$ORCL"]),
    WatchedAsset("CRM", "Salesforce", ["$CRM", "CRM", "Salesforce"], ["CRM", "$CRM"]),
    WatchedAsset("NOW", "ServiceNow", ["$NOW", "NOW", "ServiceNow"], ["NOW", "$NOW"]),
    WatchedAsset("ADBE", "Adobe", ["$ADBE", "ADBE", "Adobe"], ["ADBE", "$ADBE"]),
    WatchedAsset("SNOW", "Snowflake", ["$SNOW", "SNOW", "Snowflake"], ["SNOW", "$SNOW"]),
    WatchedAsset("TTMI", "TTM Technologies", ["$TTMI", "TTMI"], ["TTMI", "$TTMI"]),
    WatchedAsset("COHR", "Coherent", ["$COHR", "COHR", "Coherent"], ["COHR", "$COHR"]),
    WatchedAsset("PFE", "Pfizer", ["$PFE", "PFE", "Pfizer"], ["PFE", "$PFE"]),
    WatchedAsset("IGV", "iShares Expanded Tech-Software ETF", ["$IGV", "IGV"], ["IGV", "$IGV"]),
    WatchedAsset("MU", "Micron", ["$MU", "MU", "Micron"], ["MU", "$MU"]),
    WatchedAsset("SNDK", "SanDisk", ["$SNDK", "SNDK", "SanDisk"], ["SNDK", "$SNDK"]),
    WatchedAsset("QCOM", "Qualcomm", ["$QCOM", "QCOM", "Qualcomm"], ["QCOM", "$QCOM"]),
    WatchedAsset("CEG", "Constellation Energy", ["$CEG", "CEG"], ["CEG", "$CEG"]),
    WatchedAsset("VST", "Vistra", ["$VST", "VST", "Vistra"], ["VST", "$VST"]),
    WatchedAsset("RCL", "Royal Caribbean", ["$RCL", "RCL"], ["RCL", "$RCL"]),
    WatchedAsset("HOOD", "Robinhood", ["$HOOD", "HOOD", "Robinhood"], ["HOOD", "$HOOD"]),
    WatchedAsset("PLTR", "Palantir", ["$PLTR", "PLTR", "Palantir"], ["PLTR", "$PLTR"]),
    WatchedAsset(
        "SOXL", "Direxion Daily Semiconductor Bull 3X", ["$SOXL", "SOXL"], ["SOXL", "$SOXL"]
    ),
    WatchedAsset("SOXQ", "Invesco PHLX Semiconductor ETF", ["$SOXQ", "SOXQ"], ["SOXQ", "$SOXQ"]),
    WatchedAsset("SPCX", "SpaceX", ["$SPCX", "SPCX", "SpaceX"], ["SPCX", "$SPCX", "SpaceX"]),
    WatchedAsset("INTC", "Intel", ["$INTC", "INTC", "Intel"], ["INTC", "$INTC"]),
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
        {"id": "BAMLH0A0HYM2", "name": "ICE BofA High Yield OAS"},
        {"id": "SOFR", "name": "Secured Overnight Financing Rate"},
        {"id": "WTREGEN", "name": "Treasury General Account"},
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


def validate_event_threshold(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("Event threshold must be a finite ratio greater than 0.5 and at most 1.")
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Event threshold must be greater than 0.5 and at most 1.") from error
    if not math.isfinite(threshold) or not 0.5 < threshold <= 1:
        raise ValueError("Event threshold must be a finite ratio greater than 0.5 and at most 1.")
    return threshold


def load_prediction_markets_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config = _read_yaml(path or PROJECT_ROOT / "config" / "prediction-markets.yml")
    config["main_case_threshold"] = validate_event_threshold(
        config.get("main_case_threshold", DEFAULT_EVENT_MAIN_CASE_THRESHOLD)
    )
    return config


def load_allocation_config(path: Optional[Path] = None) -> Dict[str, Any]:
    defaults = {
        "primary_ticker": "QQQ", "max_new_capital_pct": 100,
        "entry_fractions": [0.40, 0.35, 0.25], "trim_fractions": [0.40, 0.30, 0.30],
        "unknown_macro_initial_cap_pct": 2, "max_campaign_loss_pct": 100,
        "min_entry_spacing_pct": 0.75,
    }
    defaults.update(_read_yaml(path or PROJECT_ROOT / "config" / "allocation.yml"))
    for key in ("max_new_capital_pct", "max_campaign_loss_pct", "unknown_macro_initial_cap_pct"):
        raw = defaults[key]
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Allocation parameter {key} must be finite.") from error
        upper = defaults["max_new_capital_pct"] if key == "unknown_macro_initial_cap_pct" else 100
        if (isinstance(raw, bool) or not math.isfinite(value) or value > upper
                or (value < 0 if key == "unknown_macro_initial_cap_pct" else value <= 0)):
            raise ValueError(f"Allocation parameter {key} is outside its account budget.")
        defaults[key] = value
    return defaults


def load_shunge_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return _read_yaml(path or PROJECT_ROOT / "config" / "shunge.yml")


def load_analyst_patterns_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return _read_yaml(path or PROJECT_ROOT / "config" / "analyst-patterns.yml")


def load_events_config(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Manual forward events from config/events.yml (product launches,
    lockup expiries, investor days — events with no public calendar)."""
    path = path or PROJECT_ROOT / "config" / "events.yml"
    data = _read_yaml(path)
    rows = data.get("events") if data else None
    if not rows:
        return []
    return [row for row in rows if isinstance(row, dict)]


def load_ipos_config(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Tracked recent IPOs from config/ipos.yml (ipo_date, ipo_price,
    early-release and lockup windows) for the IPO tracker."""
    path = path or PROJECT_ROOT / "config" / "ipos.yml"
    data = _read_yaml(path)
    rows = data.get("ipos") if data else None
    if not rows:
        return []
    return [row for row in rows if isinstance(row, dict)]
