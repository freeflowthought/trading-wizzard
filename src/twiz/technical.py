import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import TechnicalSnapshot
from .storage import write_json

IMMEDIATE_LEVEL_DISTANCE_PCT = 0.02
NEAR_LEVEL_DISTANCE_PCT = 0.06
NEAR_LEVEL_MERGE_PERCENT = 0.0035
RECENT_PRICE_ACTION_LOOKBACK = 60
DEFAULT_MAX_LEVELS = 24


def closes(rows: Iterable[Dict[str, Any]]) -> List[float]:
    return [float(row["close"]) for row in rows]


def rolling_mean(values: List[float], window: int) -> List[Optional[float]]:
    output: List[Optional[float]] = []
    for index in range(len(values)):
        if index + 1 < window:
            output.append(None)
        else:
            output.append(mean(values[index + 1 - window : index + 1]))
    return output


def ema(values: List[float], span: int) -> List[Optional[float]]:
    if not values:
        return []
    alpha = 2 / (span + 1)
    output: List[Optional[float]] = []
    current: Optional[float] = None
    for index, value in enumerate(values):
        if index + 1 < span:
            output.append(None)
            continue
        if current is None:
            current = mean(values[index + 1 - span : index + 1])
        else:
            current = value * alpha + current * (1 - alpha)
        output.append(current)
    return output


def rsi(values: List[float], window: int = 14) -> List[Optional[float]]:
    output: List[Optional[float]] = [None] * len(values)
    if len(values) <= window:
        return output
    gains: List[float] = []
    losses: List[float] = []
    for index in range(1, window + 1):
        change = values[index] - values[index - 1]
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))
    avg_gain = mean(gains)
    avg_loss = mean(losses)
    output[window] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for index in range(window + 1, len(values)):
        change = values[index] - values[index - 1]
        gain = max(change, 0.0)
        loss = abs(min(change, 0.0))
        avg_gain = (avg_gain * (window - 1) + gain) / window
        avg_loss = (avg_loss * (window - 1) + loss) / window
        output[index] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return output


def macd(values: List[float]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    ema12 = ema(values, 12)
    ema26 = ema(values, 26)
    macd_line: List[Optional[float]] = []
    for fast, slow in zip(ema12, ema26):
        macd_line.append(None if fast is None or slow is None else fast - slow)
    compact = [value for value in macd_line if value is not None]
    signal_compact = ema(compact, 9)
    signal_line: List[Optional[float]] = (
        [None] * (len(macd_line) - len(signal_compact)) + signal_compact
    )
    return macd_line, signal_line


def latest_indicators(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    close_values = closes(rows)
    if not close_values:
        return {}
    sma20 = rolling_mean(close_values, 20)
    sma50 = rolling_mean(close_values, 50)
    rsi14 = rsi(close_values, 14)
    macd_line, signal_line = macd(close_values)
    returns = [
        close_values[index] / close_values[index - 1] - 1.0
        for index in range(1, len(close_values))
        if close_values[index - 1] != 0
    ]
    volatility = pstdev(returns[-20:]) * math.sqrt(252) if len(returns) >= 2 else None
    latest = {
        "close": close_values[-1],
        "sma20": sma20[-1],
        "sma50": sma50[-1],
        "rsi14": rsi14[-1],
        "macd": macd_line[-1],
        "macd_signal": signal_line[-1],
        "annualized_volatility_20d": volatility,
    }
    return {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in latest.items()
    }


def _pivot_points(rows: List[Dict[str, Any]], bars_for_peak: int = 3) -> List[Dict[str, Any]]:
    pivots: List[Dict[str, Any]] = []
    if len(rows) < bars_for_peak * 2 + 1:
        return pivots
    for index in range(bars_for_peak, len(rows) - bars_for_peak):
        window = rows[index - bars_for_peak : index + bars_for_peak + 1]
        high = float(rows[index]["high"])
        low = float(rows[index]["low"])
        if high == max(float(row["high"]) for row in window):
            pivots.append({"price": high, "kind": "resistance"})
        if low == min(float(row["low"]) for row in window):
            pivots.append({"price": low, "kind": "support"})
    return pivots


def _recent_range_levels(
    rows: List[Dict[str, Any]],
    windows: Tuple[int, ...] = (5, 10, 20, 50, 100),
) -> List[Dict[str, Any]]:
    levels: List[Dict[str, Any]] = []
    for window in windows:
        sample = rows[-min(window, len(rows)) :]
        if not sample:
            continue
        low = min(float(row["low"]) for row in sample)
        high = max(float(row["high"]) for row in sample)
        levels.append({"price": low, "touches": 1, "source": f"{window}d_low"})
        levels.append({"price": high, "touches": 1, "source": f"{window}d_high"})
    return levels


def _recent_price_action_levels(
    rows: List[Dict[str, Any]],
    current: float,
    lookback: int = RECENT_PRICE_ACTION_LOOKBACK,
    near_distance_pct: float = NEAR_LEVEL_DISTANCE_PCT,
) -> List[Dict[str, Any]]:
    if current <= 0:
        return []
    lower_bound = current * (1.0 - near_distance_pct)
    upper_bound = current * (1.0 + near_distance_pct)
    levels: List[Dict[str, Any]] = []
    for row in rows[-min(lookback, len(rows)) :]:
        for field, source in (("low", "recent_low"), ("high", "recent_high")):
            price = float(row[field])
            if lower_bound <= price <= upper_bound:
                levels.append({"price": price, "touches": 1, "source": source})
    return levels


def _merge_limit(price: float, cluster_price: float, current: float, merge_percent: float) -> float:
    near_current = (
        abs(price / current - 1.0) <= NEAR_LEVEL_DISTANCE_PCT
        or abs(cluster_price / current - 1.0) <= NEAR_LEVEL_DISTANCE_PCT
    )
    if near_current:
        return min(merge_percent, NEAR_LEVEL_MERGE_PERCENT)
    return merge_percent


def _can_merge_price(
    price: float,
    cluster: List[Dict[str, Any]],
    current: float,
    merge_percent: float,
) -> bool:
    cluster_prices = [float(row["price"]) for row in cluster]
    cluster_price = mean(cluster_prices)
    limit = _merge_limit(price, cluster_price, current, merge_percent)
    proposed_prices = cluster_prices + [price]
    proposed_width = (max(proposed_prices) - min(proposed_prices)) / cluster_price
    return (
        abs(price - cluster_price) / cluster_price <= limit
        and proposed_width <= limit * 2.0
    )


def _cluster_price_levels(
    candidates: List[Dict[str, Any]],
    current: float,
    merge_percent: float,
    max_levels: int,
) -> List[Dict[str, Any]]:
    if not candidates or current <= 0:
        return []

    clusters: List[List[Dict[str, Any]]] = []
    for candidate in sorted(candidates, key=lambda row: float(row["price"])):
        price = float(candidate["price"])
        if price <= 0:
            continue
        if not clusters:
            clusters.append([candidate])
            continue
        if _can_merge_price(price, clusters[-1], current, merge_percent):
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])

    levels = []
    for cluster in clusters:
        price = mean(float(row["price"]) for row in cluster)
        distance_pct = (price / current - 1.0) * 100
        levels.append(
            {
                "price": round(price, 2),
                "kind": "support" if price <= current else "resistance",
                "touches": int(sum(int(row.get("touches", 1)) for row in cluster)),
                "distance_pct": round(distance_pct, 2),
                "sources": sorted({str(row.get("source", "pivot")) for row in cluster}),
            }
        )

    per_side = max(1, max_levels // 2)
    supports = _select_ranked_levels(levels, "support", per_side)
    resistances = _select_ranked_levels(levels, "resistance", per_side)
    return supports + resistances


def _select_ranked_levels(
    levels: List[Dict[str, Any]], kind: str, max_count: int
) -> List[Dict[str, Any]]:
    def rank_key(row: Dict[str, Any]) -> Tuple[float, int]:
        return (abs(float(row["distance_pct"])), -int(row["touches"]))

    side = sorted([level for level in levels if level["kind"] == kind], key=rank_key)
    immediate = [
        level
        for level in side
        if abs(float(level["distance_pct"])) <= IMMEDIATE_LEVEL_DISTANCE_PCT * 100.0
    ]
    intermediate = [
        level
        for level in side
        if IMMEDIATE_LEVEL_DISTANCE_PCT * 100.0
        < abs(float(level["distance_pct"]))
        <= NEAR_LEVEL_DISTANCE_PCT * 100.0
    ]
    deep = [
        level
        for level in side
        if abs(float(level["distance_pct"])) > NEAR_LEVEL_DISTANCE_PCT * 100.0
    ]

    tier_target = max(1, max_count // 3)
    selected = immediate[:tier_target] + intermediate[:tier_target] + deep[:tier_target]
    if len(selected) < max_count:
        selected_ids = {id(level) for level in selected}
        selected.extend(
            level for level in side if id(level) not in selected_ids
        )
    return sorted(selected[:max_count], key=rank_key)


def cluster_support_resistance(
    rows: List[Dict[str, Any]],
    merge_percent: float = 0.0125,
    bars_for_peak: int = 3,
    max_levels: int = DEFAULT_MAX_LEVELS,
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    current = float(rows[-1]["close"])
    recent_rows = rows[-180:]
    candidates = _pivot_points(recent_rows, bars_for_peak=bars_for_peak)
    candidates.extend(_recent_range_levels(rows))
    candidates.extend(_recent_price_action_levels(rows, current))
    return _cluster_price_levels(candidates, current, merge_percent, max_levels)


def classify_regime(indicators: Dict[str, Optional[float]]) -> str:
    close = indicators.get("close")
    sma20 = indicators.get("sma20")
    sma50 = indicators.get("sma50")
    rsi14 = indicators.get("rsi14")
    if close is None or sma20 is None or sma50 is None:
        return "insufficient_history"
    if close > sma20 > sma50 and (rsi14 or 0) < 75:
        return "uptrend"
    if close < sma20 < sma50:
        return "downtrend"
    if rsi14 is not None and rsi14 > 75:
        return "overbought"
    if rsi14 is not None and rsi14 < 30:
        return "oversold"
    return "mixed"


def analyze_technical_snapshot(
    ticker: str,
    rows: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
) -> TechnicalSnapshot:
    indicators = latest_indicators(rows)
    levels = cluster_support_resistance(rows)
    regime = classify_regime(indicators)
    chart_path = None
    if output_dir is not None:
        chart_path = str(output_dir / f"{ticker}-chart-data.json")
        write_json(Path(chart_path), {"ticker": ticker, "rows": rows[-160:], "levels": levels})
    return TechnicalSnapshot(
        ticker=ticker,
        timeframe="1d",
        levels=levels,
        indicators=indicators,
        regime=regime,
        chart_path=chart_path,
    )


def analyze_technical_snapshots(
    market_data: Dict[str, List[Dict[str, Any]]],
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    return {
        ticker: analyze_technical_snapshot(ticker, rows, output_dir=output_dir).to_dict()
        for ticker, rows in market_data.items()
    }
