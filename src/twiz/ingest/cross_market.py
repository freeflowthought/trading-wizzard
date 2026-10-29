"""Independent crypto references and exact SPY/QQQ equity-perpetual references."""
import json
import math
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

API_URL = "https://mainnet.zklighter.elliot.ai/api/v1/orderBookDetails"
SPEC_URL = "https://apidocs.lighter.xyz/reference/orderbookdetails"
MARKET_IDS = {"SPY": 128, "QQQ": 129}
MIN_DAY_VOLUME_USD = 10_000


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _fetch_details() -> Any:
    request = Request(API_URL, headers={"User-Agent": "trading-wizzard/0.1 personal research"})
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _empty(symbol: str, now: datetime) -> Dict[str, Any]:
    return {"symbol": symbol, "reference_for": symbol, "instrument_name": symbol,
            "market_id": MARKET_IDS[symbol], "underlying_label": "对应ETF的链上永续参考",
            "asset_class": "equity", "asset_type": "equity_perpetual", "source": "Lighter",
            "mapping_verified": False, "equity_mapping_verified": False,
            "mapping_basis": (f"Lighter永续列表中精确匹配{symbol}/market_id={MARKET_IDS[symbol]}；"
                              "非现货或指数代替。"),
            "source_url": SPEC_URL, "url": API_URL, "fetched_at": now.isoformat(),
            "exchange_updated_at": None, "quote_age_verified": False,
            "mark_price": None, "index_price": None, "oracle_price": None, "last_trade_price": None,
            "price_type": "mark_price", "price_label": "永续合约标记价",
            "day_change_pct": None, "basis_pct": None,
            "funding_rate_raw": None, "funding_unit": "unavailable",
            "open_interest_base": None, "open_interest_usd": None,
            "open_interest_notional_usd": None, "day_volume_usd": None,
            "status": "unavailable", "reason": ""}


def fetch_cross_market(
    report_date: str, live: bool = False, now: Optional[datetime] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Two bounded public requests; each source fails independently."""
    if not live:
        return [], ["Cross-market reference collection skipped offline."]
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        return [], ["Reference collection requires a timezone-aware current timestamp."]
    now = now.astimezone(timezone.utc)
    try:
        if date.fromisoformat(report_date) != now.astimezone(ZoneInfo("America/Toronto")).date():
            return [], ["Current perpetual references cannot be backfilled into another date."]
    except (TypeError, ValueError):
        return [], ["Reference report date is invalid."]
    crypto_rows, crypto_warnings = _fetch_crypto_rows(now)
    equity_rows, equity_warnings = _fetch_equity_rows(now)
    return crypto_rows + equity_rows, crypto_warnings + equity_warnings


def _fetch_equity_rows(now: datetime) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows, warnings = [_empty(symbol, now) for symbol in MARKET_IDS], []
    try:
        payload = _fetch_details()
        if (not isinstance(payload, dict) or payload.get("code") != 200
                or not isinstance(payload.get("order_book_details"), list)):
            raise ValueError("Expected a successful perpetual order-book details response.")
        # Deliberately never inspect spot_order_book_details.
        markets = payload["order_book_details"]
    except (OSError, TypeError, ValueError) as exc:
        reason = f"Lighter reference unavailable: {exc}"
        for row in rows:
            row["reason"] = reason
        return rows, [reason]
    for row in rows:
        exact = [m for m in markets if isinstance(m, dict)
                 and m.get("symbol") == row["symbol"] and m.get("market_id") == row["market_id"]]
        if len(exact) != 1:
            row["reason"] = "Exact symbol/market ID missing or duplicated; no substitute inferred."
        else:
            market = exact[0]
            if not (market.get("market_type") == "perp" and market.get("status") == "active"
                    and market.get("is_frozen") is False):
                row["reason"] = "Expected an active, unfrozen perpetual market."
            else:
                mark, index, last, change, volume, oi = (_number(market.get(key)) for key in (
                    "mark_price", "index_price", "last_trade_price", "daily_price_change",
                    "daily_quote_token_volume", "open_interest_usd",
                ))
                if (any(v is None or v <= 0 for v in (mark, index, last)) or change is None
                        or volume is None or volume < MIN_DAY_VOLUME_USD or oi is None or oi < 0):
                    row["reason"] = "Invalid market numbers or daily volume below $10k."
                elif not math.isfinite((mark/index-1)*100):
                    row["reason"] = "Mark/index basis exceeds finite numeric bounds."
                else:
                    row.update(mark_price=mark, index_price=index, last_trade_price=last,
                               day_change_pct=change, basis_pct=(mark/index-1)*100,
                               open_interest_usd=oi, open_interest_notional_usd=oi,
                               day_volume_usd=volume, mapping_verified=True,
                               equity_mapping_verified=True,
                               status="reference_only",
                               reason="Lighter perpetual reference, not official ETF close. "
                               "Quote timestamp absent; age unknown. Funding unavailable.")
        if row["status"] == "unavailable":
            warnings.append(f"Lighter {row['symbol']}: {row['reason']}")
    return rows, warnings


CRYPTO_SPEC_URL = (
    "https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/api/info/assetctxs.yaml"
)


def _fetch_crypto_contexts() -> Any:
    request = Request("https://api.hyperliquid.xyz/info", method="POST",
                      data=json.dumps({"type": "metaAndAssetCtxs"}).encode("utf-8"),
                      headers={"Content-Type": "application/json",
                               "User-Agent": "trading-wizzard/0.1 personal research"})
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_crypto_rows(now: datetime) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows, warnings = [], []
    for symbol in ("BTC", "ETH"):
        row = _empty("SPY", now)
        row.update(symbol=symbol, reference_for=symbol, instrument_name=symbol, market_id=None,
                   underlying_label=f"{symbol}加密资产永续参考", asset_class="crypto",
                   asset_type="crypto_perpetual", source="Hyperliquid",
                   equity_mapping_verified=False,
                   mapping_basis="精确资产名与universe/context相同索引；不映射为股票价格。",
                   source_url=CRYPTO_SPEC_URL, url=f"https://app.hyperliquid.xyz/trade/{symbol}",
                   funding_unit="provider interval")
        rows.append(row)
    try:
        payload = _fetch_crypto_contexts()
        if (not isinstance(payload, list) or len(payload) != 2
                or not isinstance(payload[0], dict) or not isinstance(payload[1], list)):
            raise ValueError("Expected metadata and asset-context arrays.")
        universe, contexts = payload[0].get("universe"), payload[1]
        if not isinstance(universe, list) or len(universe) != len(contexts):
            raise ValueError("Universe/context lengths differ; index mapping unsafe.")
    except (OSError, TypeError, ValueError) as exc:
        reason = f"Hyperliquid reference unavailable: {exc}"
        for row in rows:
            row["reason"] = reason
        return rows, [reason]
    for row in rows:
        matches = [i for i, asset in enumerate(universe)
                   if isinstance(asset, dict) and asset.get("name") == row["symbol"]]
        if len(matches) != 1:
            row["reason"] = "Exact crypto asset is missing or duplicated."
        else:
            index = matches[0]
            asset, context = universe[index], contexts[index]
            if asset.get("isDelisted", False) is not False or not isinstance(context, dict):
                row["reason"] = "Crypto asset is delisted or context invalid."
            else:
                mark, oracle, previous, funding, oi, volume = (
                    _number(context.get(key)) for key in
                    ("markPx", "oraclePx", "prevDayPx", "funding", "openInterest", "dayNtlVlm")
                )
                if (any(v is None or v <= 0 for v in (mark, oracle, previous))
                        or funding is None or any(v is None or v < 0 for v in (oi, volume))):
                    row["reason"] = "Crypto prices, funding, USD volume, or base OI are invalid."
                else:
                    change, basis, notional = (mark/previous-1)*100, (mark/oracle-1)*100, oi*mark
                    if not all(math.isfinite(v) for v in (change, basis, notional)):
                        row["reason"] = "Crypto derived metrics exceed finite numeric bounds."
                    else:
                        row.update(mark_price=mark, oracle_price=oracle, day_change_pct=change,
                                   basis_pct=basis, funding_rate_raw=funding, open_interest_base=oi,
                                   open_interest_usd=notional, open_interest_notional_usd=notional,
                                   day_volume_usd=volume, mapping_verified=True,
                                   status="reference_only",
                                   reason="Crypto perpetual reference only. "
                                   "Quote timestamp absent; age unknown; not an equity signal.")
        if row["status"] == "unavailable":
            warnings.append(f"Hyperliquid {row['symbol']}: {row['reason']}")
    return rows, warnings
