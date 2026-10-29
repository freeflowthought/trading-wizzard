import html
import math
import re
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from .config import (
    DEFAULT_EVENT_MAIN_CASE_THRESHOLD,
    WatchedAsset,
    _read_yaml,
    load_allocation_config,
    load_analyst_patterns_config,
    load_prediction_markets_config,
    load_shunge_config,
    validate_event_threshold,
)
from .decision import build_daily_brief
from .ingest.market import market_source, valid_market_rows
from .models import ReportBundle
from .scenarios import build_scenario_context
from .sentiment import parse_datetime
from .sessions import expected_complete_session
from .shunge import build_shunge_context
from .storage import ProjectPaths, read_json, read_jsonl
from .structure import build_dynamic_action_plan, build_market_structure
from .technical import cluster_support_resistance

IMMEDIATE_LEVEL_DISTANCE_PCT = 2.0
NEAR_LEVEL_DISTANCE_PCT = 6.0


def _latest_close(market_rows: List[Dict[str, Any]]) -> float:
    return float(market_rows[-1]["close"]) if market_rows else 0.0


def _macro_from_raw_items(paths: ProjectPaths, report_date: str) -> List[Dict[str, Any]]:
    rows = []
    cutoff = min(
        datetime.now(timezone.utc),
        datetime.combine(date.fromisoformat(report_date), time.max, tzinfo=timezone.utc),
    )
    for item in read_jsonl(paths.raw / f"items-{report_date}.jsonl"):
        source_type = str(item.get("source_type", ""))
        published = parse_datetime(str(item.get("published_at") or ""))
        if source_type in {"fed_rss", "sec_rss", "macro_rss", "rss"} and (
            published is not None and published <= cutoff
        ):
            rows.append(
                {
                    "source": item.get("source"),
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "published_at": item.get("published_at"),
                }
            )
    return rows


def _levels_by_kind(
    levels: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    supports = sorted(
        [level for level in levels if level.get("kind") == "support"],
        key=lambda row: (abs(float(row.get("distance_pct", 0))), -int(row.get("touches", 0))),
    )
    resistances = sorted(
        [level for level in levels if level.get("kind") == "resistance"],
        key=lambda row: (abs(float(row.get("distance_pct", 0))), -int(row.get("touches", 0))),
    )
    return supports, resistances


def _format_levels(levels: List[Dict[str, Any]], limit: int = 3) -> str:
    if not levels:
        return "n/a"
    parts = []
    for level in levels[:limit]:
        parts.append(
            f"{float(level['price']):.2f} ({float(level['distance_pct']):+.2f}%, "
            f"{int(level.get('touches', 1))} candidates)"
        )
    return "; ".join(parts)


def _split_level_tiers(
    levels: List[Dict[str, Any]],
    immediate_distance_pct: float = IMMEDIATE_LEVEL_DISTANCE_PCT,
    near_distance_pct: float = NEAR_LEVEL_DISTANCE_PCT,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    immediate = [
        level
        for level in levels
        if abs(float(level.get("distance_pct", 0.0))) <= immediate_distance_pct
    ]
    intermediate = [
        level
        for level in levels
        if immediate_distance_pct
        < abs(float(level.get("distance_pct", 0.0)))
        <= near_distance_pct
    ]
    deep = [
        level
        for level in levels
        if abs(float(level.get("distance_pct", 0.0))) > near_distance_pct
    ]
    return immediate, intermediate, deep


def _format_near_levels_with_fallback(levels: List[Dict[str, Any]], limit: int = 2) -> str:
    immediate, intermediate, deep = _split_level_tiers(levels)
    if immediate:
        return _format_levels(immediate, limit=limit)
    if intermediate:
        return f"no immediate level; next {_format_levels(intermediate, limit=1)}"
    if deep:
        return f"no nearby level; next {_format_levels(deep, limit=1)}"
    return "n/a"


def _sentiment_interpretation(sentiment: Dict[str, Any]) -> Dict[str, Any]:
    if not sentiment:
        return {
            "summary": "No sentiment data",
            "detail": "No ticker-matched Reddit/news discussions were found.",
            "condition": "Unknown",
        }
    positive = int(sentiment.get("positive", 0))
    negative = int(sentiment.get("negative", 0))
    neutral = int(sentiment.get("neutral", 0))
    total = positive + negative + neutral
    score = float(sentiment.get("weighted_score", 0.0))
    pos_ratio = positive / total if total else 0.0
    neg_ratio = negative / total if total else 0.0

    if total < 3:
        condition = "Low sample"
        meaning = "Too few discussions to trust the sentiment signal."
    elif score >= 0.55 and pos_ratio >= 0.60:
        condition = "Overheated bullish sentiment"
        meaning = "Discussion is strongly one-sided bullish; watch for crowded optimism."
    elif score >= 0.20:
        condition = "Bullish sentiment"
        meaning = "Discussion leans positive, but it is not one-sided enough to call overheated."
    elif score <= -0.55 and neg_ratio >= 0.60:
        condition = "Fear-heavy sentiment"
        meaning = "Discussion is strongly one-sided bearish; watch for capitulation risk."
    elif score <= -0.20:
        condition = "Bearish sentiment"
        meaning = "Discussion leans negative, but it is not one-sided enough to call panic."
    else:
        condition = "Mixed sentiment"
        meaning = "Positive and negative discussion is balanced or mostly neutral."

    return {
        "summary": f"{condition} ({positive}+ / {negative}- / {neutral} neutral)",
        "detail": (
            f"{total} ticker-matched discussions: {positive} positive "
            f"({pos_ratio:.0%}), {negative} negative ({neg_ratio:.0%}), "
            f"{neutral} neutral. {meaning}"
        ),
        "condition": condition,
    }


def _technical_temperature(indicators: Dict[str, Any]) -> str:
    rsi = indicators.get("rsi14")
    close = indicators.get("close")
    sma20 = indicators.get("sma20")
    sma50 = indicators.get("sma50")
    if rsi is None:
        return "Technical temperature: not enough history for RSI."
    trend = ""
    if close is not None and sma20 is not None and sma50 is not None:
        if close > sma20 > sma50:
            trend = " Price is above the 20-day and 50-day averages."
        elif close < sma20 < sma50:
            trend = " Price is below the 20-day and 50-day averages."
    if rsi >= 70:
        return f"Technical temperature: overheated by RSI ({rsi:.1f}).{trend}"
    if rsi <= 30:
        return f"Technical temperature: oversold / fear-heavy by RSI ({rsi:.1f}).{trend}"
    return f"Technical temperature: neutral RSI ({rsi:.1f}).{trend}"


def _backtest_interpretation(backtest: Dict[str, Any]) -> str:
    if not backtest:
        return "Backtest: no result available."
    metrics = backtest.get("metrics", {})
    strategy = float(metrics.get("total_return", 0.0))
    benchmark = float(metrics.get("benchmark_return", 0.0))
    diff = strategy - benchmark
    direction = "outperformed" if diff >= 0 else "underperformed"
    return (
        f"Backtest: the simple 20/50-day moving-average trend strategy returned "
        f"{strategy:.1%}; buy-and-hold returned {benchmark:.1%}. The strategy "
        f"{direction} buy-and-hold by {abs(diff):.1%}. This is historical context, "
        f"not a forecast."
    )


def _warning_explanation(warning: str) -> Dict[str, str]:
    if "FRED_API_KEY" in warning:
        return {
            "area": "FRED macro data",
            "meaning": ("Authenticated FRED collection was skipped; public yield/credit/FX "
                        "fallback coverage is listed separately. Inflation/labor may be missing."),
            "fix": "Add FRED_API_KEY to .env.",
        }
    if (
        "SEC EDGAR identity" in warning
        or "SEC latest filings" in warning
        or ("sec.gov" in warning.lower() and "HTTP Error 403" in warning)
    ):
        return {
            "area": "SEC filings feed",
            "meaning": (
                "The SEC blocked or limited the EDGAR feed request, usually because "
                "the request needs a descriptive identity."
            ),
            "fix": "Add EDGAR_IDENTITY='Your Name your.email@example.com' to .env.",
        }
    if "Reddit credentials missing" in warning:
        return {
            "area": "Reddit sentiment",
            "meaning": (
                "The report used Reddit RSS fallback. It can find some posts, but "
                "coverage and engagement metadata are limited."
            ),
            "fix": "Add REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT to .env.",
        }
    if "Jina Search fallback" in warning or "JINA_API_KEY" in warning:
        return {
            "area": "Optional web-search fallback",
            "meaning": (
                "Jina Search was requested but could not run. Reddit RSS fallback still "
                "works without a key."
            ),
            "fix": "Leave JINA_SEARCH_ENABLED=false, or add JINA_API_KEY to .env.",
        }
    return {
        "area": "Data quality",
        "meaning": warning,
        "fix": "Review configuration or provider access.",
    }


def _playbook_notes(pb: Dict[str, Any]) -> List[str]:
    """Render observable playbook context without inherited categorical trading claims."""
    if not pb:
        return []
    notes = []
    if pb.get("drawdown_pct") is not None and pb.get("peak"):
        notes.append(
            f"距近期高点 {float(pb['peak']):g} 回撤 {float(pb['drawdown_pct']):+.1f}%；"
            f"相邻整数位 {float(pb.get('round_below') or 0):g} / "
            f"{float(pb.get('round_above') or 0):g}。"
        )
    if pb.get("targets"):
        notes.append(
            "技术阻力参考（不另设仓位或止盈指令）："
            + " / ".join(f"{float(target):g}" for target in pb["targets"])
            + "；这些是条件价位，不是预期一定到达的目标。"
        )
    event = pb.get("next_event") or {}
    if event:
        precision = event.get("precision", "unknown")
        notes.append(
            f"下一事件：{event.get('name')}，{event.get('date')} "
            f"（{event.get('days_away')}天后；日期口径 {precision}）。"
        )
    return notes


def _table_text(value: Any) -> str:
    return str(value or "待确认").replace("|", " / ").replace("\n", " ")


def _event_number(value: Any) -> Any:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _event_time(value: Any) -> Any:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def _event_quote(
    row: Dict[str, Any], report_date: str = "",
    event_threshold: float = DEFAULT_EVENT_MAIN_CASE_THRESHOLD,
) -> Tuple[Any, str]:
    """Recheck cached quotes, including local report date and expiry, before display."""
    if row.get("selection_mode") == "equity-event-screen-v1":
        from .ingest.prediction_discovery import discovery_contract

        if discovery_contract(row) is None:
            return None, "自动筛选合约与原始记录的身份或报价绑定不匹配"
    if row.get("status") == "unavailable":
        return None, str(row.get("reason") or "暂无可用报价")
    if row.get("status") not in {"yes_leaning", "no_leaning", "uncertain"}:
        return None, "状态未验证"
    yes, no = _event_number(row.get("yes_probability")), _event_number(row.get("no_probability"))
    if (yes is None or no is None or not 0 <= yes <= 1 or not 0 <= no <= 1
            or abs(yes + no - 1) > 0.02 + 1e-12):
        return None, "Yes/No概率缺失、无效或不一致"
    now = datetime.now(timezone.utc)
    updated, fetched, end = (_event_time(row.get(key)) for key in (
        "market_updated_at", "fetched_at", "end_date"
    ))
    if updated is None or fetched is None:
        return None, "更新或抓取时间未验证"
    if updated > fetched or fetched > now or (now - updated).total_seconds() > 24 * 3600:
        return None, "报价时间在未来或已超过24小时，需重新获取"
    if (report_date
            and fetched.astimezone(ZoneInfo("America/Toronto")).date().isoformat() != report_date):
        return None, "快照日期与报告日期不符，不回填历史事件概率"
    if end is None or end <= now:
        return None, "合约期限未验证或已经到期，当前概率未知"
    liquidity, spread = _event_number(row.get("liquidity_usd")), _event_number(row.get("spread"))
    if liquidity is None or liquidity < 1000 or spread is None or not 0 <= spread <= 0.10 + 1e-12:
        return None, "流动性不足或买卖价差未通过质量门槛"
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        return None, "原始合约问题未验证"
    yes_main, no_main = yes >= event_threshold, no >= event_threshold
    quality = "偏Yes" if yes_main and not no_main else (
        "偏No" if no_main and not yes_main else "分歧/未明显倾向")
    return yes, f"{quality}；距更新{(now - updated).total_seconds() / 3600:.1f}小时"


def _render_prediction_markets(
    rows: List[Dict[str, Any]], report_date: str = "",
    event_threshold: float = DEFAULT_EVENT_MAIN_CASE_THRESHOLD,
) -> List[str]:
    lines = ["### 事件温度计（Polymarket）", ""]
    if not rows:
        lines.extend(["暂无可用事件市场快照；未知不等于事件不会发生。", ""])
        return lines
    from .ingest.prediction_discovery import discovery_contract

    discovered = [row for row in rows if discovery_contract(row) is not None]
    if discovered:
        scan = max((row.get("discovery", {}).get("scanned_events", 0) for row in discovered),
                   default=0)
        lines.extend([f"本次从有界扫描及主题搜索取得的{scan}个事件中，按股票关联、传导渠道、"
                      "期限及报价质量自动筛选；每个事件保留一份代表合约。"
                      "重要程度不是股票涨跌概率。", ""])
    else:
        lines.extend(["本快照来自指定合约清单，未包含自动发现的候选池；"
                      "不能据此声称已经覆盖当日所有重要事件。", ""])
    lines.extend([
        "| 事件与原始问题 | Yes隐含概率 | 合约期限 | 更新时间与质量 | 条件解读 |",
        "| --- | --- | --- | --- | --- |",
    ])
    for row in rows:
        probability, quality = _event_quote(row, report_date, event_threshold)
        yes_text = "未知" if probability is None else f"{probability * 100:.1f}%"
        question = str(row.get("question") or "原始问题暂无数据")
        url = str(row.get("url") or "")
        try:
            parsed = urlsplit(url)
            trusted_url = parsed.scheme == "https" and parsed.hostname in {
                "polymarket.com", "www.polymarket.com"
            }
        except ValueError:
            trusted_url = False
        if trusted_url:
            question = f"[{question}]({url})"
        label = str(row.get("label") or row.get("topic") or "事件")
        liquidity, spread = row.get("liquidity_usd"), row.get("spread")
        details = [quality]
        if isinstance(liquidity, (int, float)) and math.isfinite(liquidity):
            details.append(f"流动性${liquidity:,.0f}")
        if isinstance(spread, (int, float)) and math.isfinite(spread):
            details.append(f"价差{spread:.3f}")
        def local_stamp(value: Any) -> str:
            stamp = _event_time(value)
            return (stamp.astimezone(ZoneInfo("America/Toronto")).strftime("%Y-%m-%d %H:%M %Z")
                    if stamp else "未知")

        updated = local_stamp(row.get("market_updated_at"))
        fetched = local_stamp(row.get("fetched_at"))
        yes_scenario = str(row.get("yes_scenario") or "按原始问题的结算条件解释").rstrip("。；; ")
        scenarios = (
            f"若Yes：{yes_scenario}；"
            f"若No：{row.get('no_scenario') or '仅表示本合约未满足Yes结算条件'}"
        )
        if row.get("settlement_scope_note"):
            scenarios += "；结算范围：" + str(row["settlement_scope_note"])
        selection = row.get("discovery")
        if isinstance(selection, dict) and selection.get("selection_reason"):
            scenarios += "；入选依据：" + str(selection["selection_reason"])
        lines.append("| " + " | ".join(_table_text(value) for value in (
            f"{label}：{question}", yes_text, local_stamp(row.get("end_date")),
            f"{updated}；抓取{fetched}；" + "；".join(details), scenarios,
        )) + " |")
    lines.extend([
        "", "这是市场报价隐含的事件概率，不是QQQ上涨概率，也不代表事件已经发生。"
        "不同问题与期限不能直接横比；缺失或过期报价记为未知，不自动改变交易方向。", "",
    ])
    return lines


def _chain_usable(row: Dict[str, Any], report_date: str) -> bool:
    fetched = _event_time(row.get("fetched_at"))
    mark = _event_number(row.get("mark_price"))
    return bool(
        row.get("reference_for") in {"SPY", "QQQ"}
        and row.get("mapping_verified") is True and row.get("equity_mapping_verified") is True
        and row.get("source") == "Lighter"
        and row.get("symbol") == row.get("reference_for")
        and row.get("instrument_name") == row.get("reference_for")
        and row.get("market_id") == {"SPY": 128, "QQQ": 129}.get(row.get("reference_for"))
        and row.get("mapping_basis")
        and row.get("source_url") and row.get("instrument_name")
        and row.get("asset_type") == "equity_perpetual"
        and row.get("status") == "reference_only" and mark is not None and mark > 0
        and fetched is not None and fetched <= datetime.now(timezone.utc)
        and fetched.astimezone(ZoneInfo("America/Toronto")).date().isoformat() == report_date
    )


def _crypto_reference(row: Dict[str, Any], report_date: str) -> bool:
    fetched = _event_time(row.get("fetched_at"))
    mark = _event_number(row.get("mark_price"))
    return bool(
        row.get("asset_type") == "crypto_perpetual" and row.get("asset_class") != "equity"
        and row.get("symbol") not in {"SPY", "QQQ"}
        and row.get("status") == "reference_only" and row.get("symbol")
        and mark is not None and mark > 0 and fetched is not None
        and fetched <= datetime.now(timezone.utc)
        and fetched.astimezone(ZoneInfo("America/Toronto")).date().isoformat() == report_date
    )


def _render_cross_market(rows: List[Dict[str, Any]], report_date: str) -> List[str]:
    rows = [row for row in rows if row.get("asset_type") == "crypto_perpetual"
            or row.get("reference_for") in {"SPY", "QQQ"}]
    if not rows:
        return []
    lines = [
        "### 盘后／周末风险参考", "",
        "| 标的 / 类型 | 标记价 | 滚动24h变化 | 资金费率原始值 / 区间 | "
        "未平仓美元估算 | 标记价-预言机偏离 | 抓取时间 / 报价年龄 |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        fetched = _event_time(row.get("fetched_at"))
        usable = _chain_usable(row, report_date) or _crypto_reference(row, report_date)
        values = {key: _event_number(row.get(key)) if usable else None for key in (
            "mark_price", "day_change_pct", "funding_rate_raw",
            "open_interest_notional_usd", "basis_pct"
        )}
        mark, change = values["mark_price"], values["day_change_pct"]
        funding, oi, basis = (values[key] for key in (
            "funding_rate_raw", "open_interest_notional_usd", "basis_pct"
        ))
        kind = {"equity_perpetual": "ETF永续代理", "crypto_perpetual": "加密永续"}.get(
            row.get("asset_type"), "映射未核验"
        )
        stamp = (fetched.astimezone(ZoneInfo("America/Toronto")).strftime("%Y-%m-%d %H:%M:%S %Z")
                 if fetched else "抓取时间未知")
        source = str(row.get("source") or "来源未验证")
        quality = "交易所报价时间/年龄未知"
        if not usable:
            quality += "；" + str(row.get("reason") or "快照日期不符或暂无可用数据")
        lines.append("| " + " | ".join(_table_text(item) for item in (
            f"{row.get('reference_for') or '未知'} / {kind}（{source}）",
            f"${mark:,.2f}" if mark is not None and mark > 0 else "未知",
            f"{change:+.2f}%" if change is not None else "未知",
            f"{funding:.12g} / {row.get('funding_unit') or '区间未知'}"
            if funding is not None else "未知",
            f"${oi:,.0f}" if oi is not None and oi >= 0 else "未知",
            f"{basis:+.3f}%" if basis is not None else "未知",
            stamp + "；" + quality,
        )) + " |")
    lines.extend([
        "", "加密资产参考独立保留；链上股票参考仅限经核验的SPY/QQQ永续。"
        "抓取时间不等于交易所报价时间，这些不是ETF正式收盘价，"
        "不自动改变股票方向或仓位预算。", "",
    ])
    return lines


def _aligned_options(snapshot: Dict[str, Any], row: Dict[str, Any], report_date: str) -> bool:
    fetched = _event_time(snapshot.get("fetched_at"))
    spot, close = _event_number(snapshot.get("underlying_spot")), _event_number(row.get("close"))
    return bool(
        row.get("data_source") == "yfinance" and not row.get("stale")
        and snapshot.get("underlying_date") == row.get("data_date")
        and spot is not None and close is not None and close > 0
        and abs(spot - close) <= max(0.011, close * 0.00001)
        and fetched is not None and fetched <= datetime.now(timezone.utc)
        and fetched.astimezone(ZoneInfo("America/Toronto")).date().isoformat() == report_date
    )


def _report_clock(report_date: str) -> datetime:
    return min(datetime.now(timezone.utc), datetime.combine(
        date.fromisoformat(report_date), time.max, ZoneInfo("America/Toronto")))


def _bound_spy_range(rows: List[Dict[str, Any]], end: str) -> Dict[str, Any]:
    """Observe twenty completed raw sessions from the already hash-bound strategy input."""
    from datetime import timedelta

    from .daily_context import BASIS
    from .sessions import is_us_equity_session

    tail = rows[-20:]
    expected, cursor = [], date.fromisoformat(end)
    while len(expected) < 20:
        if is_us_equity_session(cursor):
            expected.append(cursor.isoformat())
        cursor -= timedelta(days=1)
    if (len(tail) != 20 or [r.get("date") for r in tail] != list(reversed(expected))
            or any(r.get("session_complete") is not True or r.get("price_basis") != BASIS
                   or r.get("source") not in {"yfinance", "Yahoo chart"} for r in tail)):
        return {"status": "unknown", "reason": "缺少连续20个已完成原始价格交易日"}
    for row in tail:
        if any(type(row.get(k)) not in (int, float) for k in ("open", "high", "low", "close")):
            return {"status": "unknown", "reason": "原始价格窗口OHLC类型无效"}
        prices = [_event_number(row.get(k)) for k in ("open", "high", "low", "close")]
        if (any(p is None or p <= 0 for p in prices)
                or not prices[2] <= min(prices[0], prices[3])
                <= max(prices[0], prices[3]) <= prices[1]):
            return {"status": "unknown", "reason": "原始价格窗口OHLC无效"}
    return {"status": "available", "start_date": tail[0]["date"], "end_date": end,
            "low": min(r["low"] for r in tail), "high": max(r["high"] for r in tail),
            "basis": BASIS, "role": "macro_observation_only"}


def _current_cluster_prices(rows: List[Dict[str, Any]], end: str) -> Dict[str, List[float]]:
    """Observe using cmd_analyze's unchanged default algorithm and source price basis."""
    empty = {"support": [], "resistance": []}
    if not rows or rows[-1].get("date") != end or rows[-1].get("session_complete") is not True:
        return empty
    for row in rows:
        if row.get("session_complete") is not True:
            return empty
        if any(type(row.get(k)) not in (int, float) for k in ("open", "high", "low", "close")):
            return empty
        prices = [_event_number(row.get(k)) for k in ("open", "high", "low", "close")]
        if (any(p is None or p <= 0 for p in prices)
                or not prices[2] <= min(prices[0], prices[3])
                <= max(prices[0], prices[3]) <= prices[1]):
            return empty
    try:
        levels = cluster_support_resistance(rows)
        close = float(rows[-1]["close"])
        return {kind: [float(r["price"]) for r in sorted(
            (r for r in levels if r.get("kind") == kind
             and _event_number(r.get("price")) is not None and float(r["price"]) > 0),
            key=lambda r: abs(float(r["price"])-close))[:2]] for kind in empty}
    except (TypeError, ValueError, OverflowError):
        return empty


def _observation_prices(values: Any) -> str:
    if not isinstance(values, list):
        return "未知"
    numbers = [_event_number(value) for value in values[:2]]
    return " / ".join(f"{p:.2f}" for p in numbers if p is not None and p > 0) or "未知"


def _portfolio_proof_reason(paths: ProjectPaths, report_date: str, plan: Dict[str, Any]) -> str:
    """Bind displayed targets to the archived accepted policy and replay, without refitting."""
    from .macro_rotation import _clock, _next, build_macro_rotation_policy
    from .swing_optimizer import _hash

    try:
        trace = read_json(paths.processed / f"portfolio-training-{report_date}.json", {})
        config_path = paths.config / "portfolio-strategy.yml"
        config = _read_yaml(config_path)
        if not config.get("enabled") or config.get("automatic_orders") is not False:
            return "已接受的模型配置未启用或订单权限不匹配"
        activated = _event_time(config.get("activated_at"))
        available = _event_time(plan.get("available_at"))
        if (activated is None or available is None or activated > available
                or activated.astimezone(ZoneInfo("America/New_York")).date().isoformat()
                > report_date):
            return "模型计划早于已接受配置的启用时间"
        active = build_macro_rotation_policy(
            config["parameters"], cycle_parameters=config["cycle_parameters"],
            turning_parameters=config.get("turning_parameters"))
        policy = trace["policy"]
        signature = _hash(policy)
        if not (signature == trace["policy_sha256"] == plan["policy_sha256"]
                == active["policy_sha256"]):
            return "模型计划与当前已接受规则的指纹不匹配"
        import hashlib

        if (trace["source_fingerprints"] != plan["source_fingerprints"]
                or trace["source_fingerprints"]["portfolio-strategy.yml"]["sha256"]
                != hashlib.sha256(config_path.read_bytes()).hexdigest()
                or trace["date"] != report_date or trace["status"] != plan["status"]
                or trace["available_at"] != plan["available_at"]):
            return "模型计划与已归档输入或生成时点不匹配"
        replay = trace["replay"]
        if (_hash(replay["policy"]) != signature or replay["policy_sha256"] != signature
                or replay["source_code_sha256"] != plan["model_source_sha256"]
                or replay["source_code_sha256"] != policy["source_code_sha256"]
                or trace["used_price_history_sha256"] != _hash(trace["used_price_history"])
                or trace["used_news_sha256"] != _hash(trace["used_news"])):
            return "模型回放与价格/新闻输入指纹不匹配"
        for ticker in ("QQQ", "SPY", "TQQQ"):
            source = trace["used_price_history"][ticker][-1]
            if (source["date"] != plan["price_date"]
                    or source["close"] != plan["price_closes"][ticker]):
                return "模型回放使用的行情与展示价格不匹配"
        last, point = replay["decisions"][-1], replay["equity_curve"][-1]
        weights = {key: last["target_weights"][key] for key in ("QQQ", "TQQQ")}
        weights["cash"] = max(0., 1-sum(weights.values()))
        paper = {key: point["weights"][key] for key in ("QQQ", "TQQQ")}
        paper["cash"] = max(0., 1-sum(paper.values()))
        gain = config["cycle_parameters"]["migrate_gain"]
        wac = last["tqqq_entry_wac"]
        expected = {"target_weights": weights, "paper_weights": paper, "cause": last["cause"],
                    "next_open_at": _clock(_next(date.fromisoformat(plan["price_date"])), True),
                    "decision_at": last["at"], "price_at": point["at"],
                    "qqq_features": last["features"]["QQQ"],
                    "structural_stop": last["recovery_stop"],
                    "tqqq_entry_wac": wac, "tqqq_gain": last["tqqq_gain"],
                    "holding_sessions": last["holding_sessions"],
                    "pending_order": replay["pending_order"],
                    "profit_target_gain": gain,
                    "profit_target_price": wac*(1+gain) if wac else None,
                    "time_limit_sessions": config["cycle_parameters"]["max_holding_sessions"],
                    "reference_source": config.get("reference_source")}
        if any(plan.get(key) != value for key, value in expected.items()):
            return "展示目标、原因或退出条件与模型回放末值不匹配"
        if type(last.get("risk_armed")) is not bool:
            return "模型长期风控启用状态缺失"
        plan["validated_display_policy"] = {
            "parameters": dict(policy["parameters"]),
            "cycle_parameters": dict(policy["cycle_parameters"]),
            "trend_exit_sessions": 2 if policy["trend"].startswith("two QQQcloses") else None,
            "prior_high_sessions": 3 if "prior3high" in policy["bottom"] else None,
            "drawdown_lookback": 63 if "prior63" in policy["drawdown"] else None,
            "risk_armed": last["risk_armed"],
            "turning_parameters": dict(policy.get("turning_parameters") or {}),
        }
        plan["validated_observation_levels"] = {
            "SPY": _bound_spy_range(trace["used_price_history"]["SPY"], plan["price_date"])}
        return ""
    except (KeyError, TypeError, ValueError, OSError, IndexError, AttributeError):
        return "缺少可核验的已接受规则与模型回放凭据"


def _direction_reason(context: Dict[str, Any], row: Dict[str, Any], market: Dict[str, Any],
                      report_date: str) -> str:
    if context.get("date") != report_date or context.get("horizon_sessions") != 20:
        return "缺少本报告日期的20交易日模型结果"
    if row.get("status") != "available":
        return str(row.get("reason") or "模型证据不足")
    clock = _report_clock(report_date)
    available = _event_time(row.get("available_at", context.get("available_at")))
    expires = _event_time(row.get("expires_at", context.get("expires_at")))
    raw_expiry = row.get("expires_at", context.get("expires_at"))
    if (available is None or available > clock or raw_expiry is not None and expires is None
            or expires is not None and expires <= clock):
        return "模型信息时点未核验、在未来或已过期"
    if (market.get("data_source") != "yfinance" or market.get("stale", True)
            or market.get("session_complete") is not True
            or market.get("data_date") != expected_complete_session(report_date)
            or row.get("price_date") != market.get("data_date")):
        return "模型行情日期与当前完整交易日不匹配"
    if row.get("origin_date", row["price_date"]) != row["price_date"]:
        return "模型预测原点与行情日期不匹配"
    if row.get("refit_at") is not None:
        refit = _event_time(row["refit_at"])
        origin_close = datetime.combine(date.fromisoformat(row["price_date"]), time(16, 15),
                                        ZoneInfo("America/New_York"))
        if (refit is None or refit > origin_close
                or refit.astimezone(ZoneInfo("America/New_York")).date() != origin_close.date()):
            return "模型训练截止时点与预测原点不匹配"
    if row.get("price_close") is not None:
        source_close, close = _event_number(row["price_close"]), _event_number(market.get("close"))
        if (source_close is None or close is None
                or abs(source_close-close) > max(.011, close*.00001)):
            return "模型与日报使用的同日价格不一致"
    up, down = (_event_number(row.get(key)) for key in ("probability_up", "probability_down"))
    if (up is None or down is None or not 0 <= up <= 1 or not 0 <= down <= 1
            or abs(up + down - 1) > 1e-6 or not row.get("model_kind")):
        return "上涨/下跌概率或模型身份未通过核验"
    return ""


def _render_directional_context(bundle: ReportBundle, compact: bool = False) -> List[str]:
    context = bundle.directional_context
    market = {row["ticker"]: row for row in bundle.watchlist_table}
    lines = ["### 未来20个交易日方向概率", "",
             "| 标的 | 上涨 | 下跌／持平 | 数据日与验证依据 |", "| --- | ---: | ---: | --- |"]
    for ticker in ("QQQ", "SPY"):
        row = context.get("by_ticker", {}).get(ticker, {})
        reason = _direction_reason(context, row, market.get(ticker, {}), bundle.date)
        if reason:
            lines.append(f"| {ticker} | 未知 | 未知 | {_table_text(reason)} |")
            continue
        validation = row.get("validation") or {}
        kind = {"historical_base_rate": "历史频率参考", "price_only": "价格/波动条件模型",
                "price_calendar_macro_capex_events": "价格＋宏观/财报/事件模型"}.get(
                    row["model_kind"], str(row["model_kind"]))
        evidence = f"{row['price_date']}；{kind}"
        brier = _event_number(validation.get("brier"))
        prior = _event_number(validation.get("prior_brier"))
        if not compact and brier is not None and prior is not None:
            evidence += f"；验证Brier {brier:.3f} / 历史频率基准 {prior:.3f}"
        if not compact and validation.get("samples") is not None:
            evidence += f"；样本{validation['samples']}"
        if row["model_kind"] == "historical_base_rate":
            evidence += "；条件模型未优于基准，当前仅为历史频率参考"
        contributions = [r for r in row.get("contributions", [])
                         if r.get("active") is True and row["model_kind"] != "historical_base_rate"
                         and _event_number(r.get("log_odds_contribution")) is not None]
        if contributions and not compact:
            strongest = sorted(contributions, key=lambda r: abs(float(
                r["log_odds_contribution"])), reverse=True)[:2]
            evidence += "；主要调整：" + "、".join(
                f"{r.get('factor')} {'偏多' if float(r['log_odds_contribution']) > 0 else '偏空'}"
                for r in strongest if float(r["log_odds_contribution"]) != 0)
        lines.append(f"| {ticker} | {float(row['probability_up']):.1%} | "
                     f"{float(row['probability_down']):.1%} | {_table_text(evidence)} |")
    return lines + ["", "这是指定期限的模型估计；事件市场Yes/No另列，"
                    "不能直接当作股指上涨概率。", ""]


def _cloud_fact_reason(row: Dict[str, Any], refs: Dict[str, Any], clock: datetime,
                       ticker: str) -> str:
    from .cloud_capex import CAPEX, ISSUERS, REVENUE, _available, _period, _prior

    if row.get("status") != "available":
        return str(row.get("reason") or "可比财报尚未齐备")
    if row.get("period") != row.get("current", {}).get("period"):
        return "财报显示期间与来源不匹配"
    period = row["period"]
    try:
        valid_period = _period(period.get("start"), period.get("end"), ISSUERS[ticker][1])
    except (TypeError, ValueError):
        valid_period = None
    if valid_period != period or row.get("prior", {}).get("period") != _prior(period):
        return "财报累计期间或上年可比期间不匹配"
    names = ["current", "prior"]
    if row.get("cash_capex_yoy_acceleration_pp") is not None:
        if row.get("previous_same_duration", {}).get("period") != _prior(_prior(period)):
            return "同比增速变化缺少同累计期间的前两年来源"
        names.append("previous_same_duration")
    for name in names:
        pair = row.get(name, {})
        period = pair.get("period") or {}
        for kind in ("capex", "revenue"):
            fact = pair.get(kind, {})
            ids = fact.get("refs") or []
            if fact.get("status") != "available" or not ids:
                return "财报数值缺少可追溯来源"
            for key in ids:
                ref = refs.get(key, {})
                available = _event_time(ref.get("available_at_proxy"))
                try:
                    declared = _available(ref.get("filed"))
                    period_valid = (date.fromisoformat(ref.get("end")) <=
                                    date.fromisoformat(ref.get("filed")))
                except (TypeError, ValueError, OverflowError):
                    return "申报日期未核验"
                if (ref.get("ticker") != ticker or not period_valid
                        or available is None or available != declared
                        or available > clock
                        or ref.get("start") != period.get("start")
                        or ref.get("end") != period.get("end")
                        or ref.get("unit") != "USD"
                        or not re.fullmatch(r"\d{10}-\d{2}-\d{6}", str(ref.get("accn", "")))
                        or ref.get("tag") not in ((CAPEX,) if kind == "capex" else REVENUE)
                        or not re.fullmatch(r"[0-9a-f]{64}", str(ref.get("source_sha256", "")))
                        or not str(ref.get("json_pointer", "")).startswith("/facts/us-gaap/")):
                    return "财报来源、期间或可用时点未通过核验"
    return ""


def _render_cloud_capex(bundle: ReportBundle) -> List[str]:
    context = bundle.cloud_capex_context
    clock = _report_clock(bundle.date)
    available = _event_time(context.get("available_at"))
    if (context.get("date") not in {bundle.date, expected_complete_session(bundle.date)}
            or available is None or available > clock):
        return ["### 云客户资本开支与收入", "", "尚无本日可核验的历史可比财报快照。", ""]
    lines = ["### 云客户资本开支与收入", "",
             "| 公司 | 可比累计期间 | 现金资本开支同比 | 收入同比 | 开支同比变化 | 来源 |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    known = []
    for ticker in ("MSFT", "GOOGL", "META"):
        row = context.get("issuers", {}).get(ticker, {})
        reason = _cloud_fact_reason(row, context.get("source_refs", {}), available, ticker)
        if reason:
            lines.append(f"| {ticker} | {_table_text(reason)} | 未知 | 未知 | 未知 | — |")
            continue
        cash, revenue, acceleration = (_event_number(row.get(key)) for key in (
            "cash_capex_yoy_pct", "revenue_yoy_pct", "cash_capex_yoy_acceleration_pp"))
        if cash is None or revenue is None:
            lines.append(f"| {ticker} | 同比数值未通过核验 | 未知 | 未知 | 未知 | — |")
            continue
        known.append((cash, revenue))
        period = row.get("period", {})
        ids = row["current"]["capex"]["refs"]
        source = context["source_refs"][ids[0]]
        filing = source["accn"]
        cik = {"MSFT": 789019, "GOOGL": 1652044, "META": 1326801}[ticker]
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/" + filing.replace("-", "")
        lines.append(f"| {ticker} | {period.get('start')}至{period.get('end')} | {cash:+.1f}% | "
                     f"{revenue:+.1f}% | "
                     f"{f'{acceleration:+.1f}个百分点' if acceleration is not None else '未知'} | "
                     f"[SEC {source['filed']}]({url}) |")
    if len(known) >= 2:
        weak = sum(cash < 0 and revenue < 0 for cash, revenue in known)
        expanding = sum(cash > 0 and revenue > 0 for cash, revenue in known)
        reason = ("多家客户开支与收入同时收缩，半导体新增仓位需更严格的价格确认。"
                  if weak >= 2 else "多家客户开支与收入同增，投入背景偏扩张；买入仍等价格确认。"
                  if expanding >= 2 else "客户投入与收入分化，操作以价格和相对强弱为主。")
        lines.extend(["", "操作含义：" + reason])
    refresh = read_json(ProjectPaths().processed / f"cloud-capex-refresh-{bundle.date}.json", {})
    if refresh.get("issuers"):
        newest = "；".join(
            f"{ticker} {row['latest_sec_filing']['form']} {row['latest_sec_filing']['filed']}"
            + ("（财报数据库尚未收录）" if row.get("status") == "filing_pending_in_companyfacts"
               else "（已下载，按提交日次一交易日收盘计入）"
               if row.get("status") == "refreshed" else "")
            for ticker, row in refresh["issuers"].items() if row.get("latest_sec_filing"))
        lines.extend(["", f"SEC最新定期报告核对（{refresh['checked_at'][:16]}Z）：{newest}。"])
    return lines + ["", "同比变化是同一累计期间同比增速相对前一年的差值，非环比或物理加速度。"
                    "总公司收入/现金PP&E不等于云收入、AI采购或含租赁指引；"
                    "Amazon口径缺口保持缺失。", ""]


def _render_options_reasons(bundle: ReportBundle) -> List[str]:
    market = {row["ticker"]: row for row in bundle.watchlist_table}
    lines = ["### 期权怎样影响执行", ""]
    for ticker in ("QQQ", "SPY"):
        snapshot = bundle.options_context.get(ticker, {})
        expiry = str(snapshot.get("expiration") or "")
        try:
            expiry_valid = date.fromisoformat(expiry).isoformat() == expiry
        except ValueError:
            expiry_valid = False
        if (not _aligned_options(snapshot, market.get(ticker, {}), bundle.date)
                or market.get(ticker, {}).get("session_complete") is not True
                or not expiry_valid or not snapshot.get("contracts")
                or expiry <= max(bundle.date, str(market.get(ticker, {}).get("data_date") or ""))):
            lines.append(f"- {ticker}：当前期权数据缺失、到期或与行情未对齐，操作不依赖该快照。")
            continue
        skew = snapshot.get("skew_summary") or {}
        move, ratio = (_event_number(skew.get(key)) for key in (
            "expected_move_pct", "put_call_oi_ratio"))
        evidence = []
        if move is not None and move > 0:
            evidence.append(f"至{expiry}跨式隐含幅度约±{move:.2f}%")
        if ratio is not None and ratio >= 0:
            evidence.append(f"该快照P/C持仓比{ratio:.2f}")
        reading = "；".join(evidence) if evidence else "缺少有效波动/持仓统计"
        lines.append(f"- {ticker}：" + reading
                     + "。用于核查止盈空间与事件波动，不能从持仓比推导必涨或加仓概率。")
    return lines + [""]


def _add_event_context(
    brief: Dict[str, Any], rows: List[Dict[str, Any]], gate: Dict[str, Any]
) -> None:
    usable = [(row, _event_quote(row, str(brief.get("date", "")))[0]) for row in rows]
    usable = [(row, probability) for row, probability in usable if probability is not None]
    brief["event_evidence"] = [
        f"{row.get('label') or row.get('topic')}: Yes {probability * 100:.1f}%"
        for row, probability in usable
    ]
    if len(usable) < len(rows) or not rows:
        brief.setdefault("missing", []).append(
            "部分或全部事件市场报价缺失/过期；未知不等于No，不据此改写大盘方向。"
        )
    macro = {"risk_on": "利率/信用门槛未触发防守", "risk_off": "利率/信用门槛偏防守",
             "caution": "利率/信用门槛需谨慎", "unknown": "未知，等待有效利率与信用数据"}
    measured = [row for row in brief.get("sectors", [])
                if isinstance(row.get("relative_return_20d_pct"), (float, int))
                and math.isfinite(row["relative_return_20d_pct"])
                and row.get("ticker") not in {"SPY", "QQQ", "DIA", "IWM"}]
    sector_text = "同区间板块强弱证据不足，先等待。"
    if measured:
        measured.sort(key=lambda row: row["relative_return_20d_pct"], reverse=True)
        chosen = [measured[0], measured[-1]] if len(measured) > 1 else measured
        sector_text = "；".join(
            f"{row['ticker']} 较SPY {row['relative_return_20d_pct']:+.2f}个百分点"
            for row in chosen
        ) + "（同区间20日）"
    event_text = "；".join(brief["event_evidence"]) or "暂无可用预测市场报价"
    brief["framework"] = [
        {"step": "大盘方向", "read": brief.get("stance", "等待") + "；以价格和均线为主"},
        {"step": "利率环境", "read": macro.get(gate.get("state"), macro["unknown"])},
        {"step": "事件风险", "read": event_text + "；策略见上方事件主情景与行动卡"},
        {"step": "板块选择", "read": sector_text},
    ]


def _action_pct(value: Any) -> str:
    number = _event_number(value)
    return f"{number:g}%" if number is not None and 0 <= number <= 100 else "待确认"


def _action_price(value: Any) -> str:
    number = _event_number(value)
    return f"{number:.2f}" if number is not None and number > 0 else "待确认"


def _render_action_plan(plan: Dict[str, Any]) -> List[str]:
    lines = ["### 今日行动卡：抄底与止盈", ""]
    if not plan or plan.get("status") == "unavailable":
        lines.extend([
            "**现在买不买：** " + str(plan.get("current_action") or
                               "当前新增买入0%；行情、仓位政策或关键价位尚未通过校验。"), "",
        ])
        for reason in plan.get("limitations", []) if plan else []:
            lines.append(f"- {reason}")
        return lines + [""]
    budget = plan.get("budget", {})
    ticker = str(budget.get("ticker") or "QQQ")
    lines.extend([
        f"**现在买不买：{plan.get('current_action') or '等待确认，当前新增买入0%。'}**", "",
        f"本轮只给 **{ticker}** 一个新增抄底预算，上限为**总账户的"
        f"{_action_pct(budget.get('max_new_capital_pct'))}**，不是每只标的各用这个比例。"
        f"当前计划合计 {_action_pct(budget.get('planned_capital_pct'))}；"
        f"现在可执行买入 **{_action_pct(plan.get('buy_now_pct'))}**。", "",
        f"满足触发条件后最多可启用 {_action_pct(budget.get('currently_enabled_pct'))}；"
        f"本轮暂不启用预算 {_action_pct(budget.get('uncommitted_cash_pct'))}。"
        "这不是实际现金或现有仓位；本报告未接入账户，也未记录已成交订单。", "",
    ])
    inputs = plan.get("model_inputs", {})
    if plan.get("price_date") and inputs.get("close") is not None:
        close_source = inputs.get("close_source") or inputs.get("source")
        source_text = {
            "regular_market_metadata": "Yahoo常规时段收盘报价（regular_market_metadata）",
            "yahoo_daily_history": "Yahoo日线收盘",
            "daily_history": "Yahoo日线收盘",
        }.get(close_source, str(close_source or "待确认"))
        lines.extend([
            f"价位依据：{ticker} {plan['price_date']} 日线值 {_action_price(inputs['close'])}；"
            f"收盘来源 {source_text}。", "",
        ])
    entries = plan.get("entries", [])
    if entries:
        if plan.get("entry_reference"):
            lines.extend(["价梯口径：" + str(plan["entry_reference"]), ""])
        spacing = plan.get("spacing_assessment", {})
        if spacing:
            first_distance = _event_number(spacing.get("first_entry_distance_atr"))
            target_distance = _event_number(spacing.get("first_target_distance_atr"))
            distances = []
            if first_distance is not None:
                distances.append(f"首档距当前价{first_distance:.2f}倍日均真实波幅（ATR）")
            if target_distance is not None:
                distances.append(f"首个止盈在当前价上方{target_distance:.2f}ATR")
            lines.extend([
                "价距评估：" + "；".join(distances) + "。"
                + str(spacing.get("verdict") or "") + str(spacing.get("target_note") or ""), "",
            ])
        lines.extend([
            "**什么时候抄底、买多少（占总账户）：**", "",
            "| 批次 | 观察价 | 本批计划 | 条件满足后可启用 | 必须满足的条件 |",
            "| --- | ---: | ---: | ---: | --- |",
        ])
        for row in entries:
            condition = str(row.get("trigger") or "等待有效确认").rstrip("。；; ")
            if row.get("price_origin"):
                condition = str(row["price_origin"]) + "；" + condition
            if row.get("status"):
                condition += "；状态：" + str(row["status"])
            lines.append("| " + " | ".join(_table_text(value) for value in (
                row.get("batch"), _action_price(row.get("price")),
                _action_pct(row.get("account_pct")), _action_pct(row.get("currently_enabled_pct")),
                condition,
            )) + " |")
        lines.append("")
    exits = plan.get("exits", [])
    if exits:
        lines.extend([
            "**买入后什么时候卖、卖多少：** 只有本轮实际建立抄底仓后，下面的止盈才适用。"
            "每批都按**本轮抄底仓原始份额**计算，不是按剩余仓位，也不包含已有核心仓。", "",
            "| 止盈批次 | 观察价 | 卖出本轮原始抄底仓 | 执行条件 |",
            "| --- | ---: | ---: | --- |",
        ])
        for row in exits:
            condition = str(row.get("trigger") or row.get("status") or "按已建立的仓位逐批复核")
            reward_risk = _event_number(row.get("reward_risk_multiple"))
            if reward_risk is not None:
                condition += f"；按最高首档成本及止损计算约{reward_risk:.2f}R"
            if row.get("price") is not None and row.get("source_date"):
                kind = {"confirmed_price_pivot": "已确认价格枢轴",
                        "current_completed_daily_SMA": "已完成日线均线"}.get(
                            row.get("reference_kind"), "来源参考位")
                condition += f"；来源：{row['source_date']} {kind}"
            lines.append("| " + " | ".join(_table_text(value) for value in (
                row.get("stage"), _action_price(row.get("price")),
                _action_pct(row.get("trim_original_dip_pct")),
                condition,
            )) + " |")
        lines.append("")
    stop = plan.get("stop") or {}
    if stop.get("price") is not None:
        lines.extend([
            f"**何时认错：日收盘跌破 {ticker} {_action_price(stop['price'])}，"
            "取消剩余加仓，复核退出本轮抄底仓；不把这一规则自动套到未知的核心仓。**", "",
        ])
        modeled = _event_number(stop.get("modeled_loss_pct"))
        if modeled is not None:
            lines.extend([
                "若全部计划批次成交并恰好在该价退出，模型损失约占总账户 "
                f"{modeled:.2f}%；跳空、滑点或执行延迟可使实际损失超过这个数。", "",
            ])
    lines.extend([f"本轮新增预算只给{ticker}；其他标的新增分配为0%，不新增杠杆。", ""])
    for reason in plan.get("limitations", []):
        lines.append(f"- {reason}")
    lines.append("")
    return lines


def _render_shunge_context(
    context: Dict[str, Any], expected_date: str = "", study: bool = False
) -> List[str]:
    # Author price scenarios are retained only for source study, never live execution.
    if not study:
        return []
    patterns, case = context.get("patterns", []), context.get("case", {})
    if not patterns and not case.get("available"):
        return []
    lines = [
        "## 顺哥模式与历史案例复核", "",
        "以下仅为有日期的历史学习案例；作者当时的价格不参与当前区域、买入、止盈或仓位计算。", "",
    ]
    if case.get("available"):
        support = case.get("source_support", [])
        lines.extend([
            f"**QQQ为什么会从704–705谈到650（截至{case.get('as_of')}的案例复核）**", "",
            f"作者原文目标为 {_action_price(case.get('source_target'))}；"
            f"数据口径：{case.get('price_basis')}。", "",
        ])
        friday = case.get("prior_friday") or {}
        if friday.get("low") is not None:
            lines.extend([
                f"**原文与行情并非精确一致：{friday.get('date')}前周五实际低点为"
                f"{_action_price(friday['low'])}；704–705应视为作者判断线，"
                "不能写成已核验的精确前周五最低价。**", "",
            ])
        reactions = case.get("support_reactions", [])
        if reactions:
            sample = "；".join(
                f"{row.get('date')}低点{_action_price(row.get('low'))}、"
                f"收盘{_action_price(row.get('close'))}" for row in reactions[:4]
            )
            lines.extend([
                "此前704附近反复出现低点并收盘守住：" + sample
                + "。这说明附近是一片值得观察的区域，不是精确704的独立触碰统计。", "",
            ])
        if len(support) == 2 and case.get("range_high") is not None:
            midpoint = (float(support[0]) + float(support[1])) / 2
            lines.extend([
                "一种事后可复核的解释是区间高度投影："
                f"2 × {midpoint:.2f} − {float(case['range_high']):.2f}"
                f"（{case.get('range_high_date')}区间高点） = "
                f"{_action_price(case.get('measured_target'))}，与原文650相近。"
                "这不能证明作者用了该公式，也不能证明650必达或必守。", "",
            ])
        method_url = str(case.get("method_source_url") or "")
        if method_url.startswith("https://"):
            lines.extend([
                f"[区间高度投影的方法参考]({method_url})"
                "（仅说明方法，不证明作者采用或目标会实现）。", "",
            ])
        checkpoint_values = sorted(
            [(float(case[key]), label) for key, label in (
                ("sma200", "200日线"), ("weekly_sma30", "30周线"), ("weekly_sma50", "50周线")
            ) if case.get(key) is not None], reverse=True,
        )
        checkpoints = [f"{label} {_action_price(value)}" for value, label in checkpoint_values]
        if checkpoints:
            lines.extend(["中间仍应逐级观察：" + "；".join(checkpoints)
                          + "，不是破位后直接跳到650。", ""])
        hits = case.get("prior_zone_hits", [])
        if hits:
            sample = "、".join(str(row.get("date")) for row in hits[:4])
            lines.extend([
                f"此前645–655价格区域在 {sample} 等{len(hits)}个日线区间出现过；"
                "这里只说明历史交易价格曾经过该区域，不代表成交量密集区或确定底部。", "",
            ])
        for note in case.get("limitations", []):
            lines.append(f"- {note}")
        lines.append("")
    if patterns:
        lines.extend([
            "**样本中的七类模式（有日期的来源归纳，不是训练出的胜率模型）**", "",
            "| 模式 | 含义 | 原文证据 |", "| --- | --- | --- |",
        ])
        for pattern in patterns:
            evidence = "; ".join(
                f"{row.get('date')} {row.get('message_time') or ''} — {row.get('summary')}"
                for row in pattern.get("evidence", [])
            )
            lines.append("| " + " | ".join(_table_text(value) for value in (
                pattern.get("name"), pattern.get("interpretation"), evidence,
            )) + " |")
        lines.append("")
    for note in context.get("source_notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return lines


def _zone_text(zone: Any) -> str:
    if not isinstance(zone, dict) or not zone.get("credible"):
        return "依据不足，暂不指定"
    return f"{_action_price(zone.get('lower'))}–{_action_price(zone.get('upper'))}"


def _render_global_context(context: Dict[str, Any], report_date: str) -> List[str]:
    if not context:
        return []
    if context.get("as_of") != report_date:
        return [f"海外政策快照仅核验至{context.get('as_of') or '未知'}，本日背景待复核。", ""]
    rates = []
    for row in context.get("policy_rates", []):
        value = _event_number(row.get("value"))
        url = str(row.get("source_url") or "")
        if (value is None or value < 0 or row.get("verified_as_of") != report_date
                or str(row.get("announced_on") or "9999") > report_date
                or not url.startswith("https://")):
            continue
        timing = f"{row.get('announced_on')}公布"
        if row.get("effective_from") and str(row["effective_from"]) <= report_date:
            timing += f"、{row['effective_from']}生效"
        rates.append(f"[{row.get('country')}]({url}) {value:g}%"
                     f"（{row.get('rate_type')}；{timing}）")
    lines = ["海外利率背景：" + "；".join(rates) + "。", ""] if rates else []
    lines.extend(_render_index_exposure(context, report_date))
    for row in context.get("policy_events", [])[:1]:
        url = str(row.get("source_url") or "")
        if (str(row.get("published_on") or "9999") <= report_date
                and str(row.get("signed_on") or "9999") <= report_date
                and url.startswith("https://")):
            lines.extend([
                f"政策事实：[{row.get('person')} {row.get('document')}]({url})"
                f"（{row.get('signed_on')}签署、{row.get('published_on')}公开）："
                f"{row.get('fact')} 不单凭文件标题生成股市方向。", "",
            ])
    for row in context.get("disclosures", [])[:1]:
        url = str(row.get("source_url") or "")
        if row.get("status") == "unavailable" and url.startswith("https://"):
            interpretation = row.get("interpretation") or (
                "当前具体申报未核验，不作为实时持仓或买入信号。"
            )
            lines.extend([
                f"人物申报：[{row.get('person')} 官方披露入口]({url})："
                f"{interpretation}", "",
            ])
    if lines and context.get("interpretation"):
        lines.extend([str(context["interpretation"]), ""])
    return lines


def _render_index_exposure(context: Dict[str, Any], report_date: str) -> List[str]:
    rows = []
    for row in context.get("semiconductor_exposures", []):
        weight = _event_number(row.get("weight_pct"))
        holdings_date = str(row.get("holdings_date") or "")
        try:
            age = (date.fromisoformat(report_date)-date.fromisoformat(holdings_date)).days
        except ValueError:
            continue
        if (row.get("ticker") not in {"QQQ", "SPY"} or row.get("coverage") != "complete"
                or weight is None or not 0 < weight <= 100 or not 0 <= age <= 7
                or not str(row.get("source_url") or "").startswith("https://")):
            continue
        rows.append(row)
    if not rows:
        return []
    lines = ["**半导体怎样传到指数：** 以下按发行方完整持仓汇总狭义半导体及设备公司；"
             "SNDK等存储链、EDA软件、光学元件及光伏另列，不计入此口径。"
             "这是按公司清单计算的参考值，非发行方直接发布的行业合计。"
             "ETF权重只用于解释直接价格贡献，不等于事件权重或上涨概率。", "",
             "| 指数ETF | 半导体及设备权重 | 持仓日期及来源 | 板块跌5%的直接拖累示例 |",
             "| --- | --- | --- | --- |"]
    for row in rows:
        weight = float(row["weight_pct"])
        lines.append(f"| {row['ticker']} | {weight:.2f}% | "
                     f"[{row['holdings_date']}]({row['source_url']}) | "
                     f"约-{weight*0.05:.2f}个百分点 |")
    for row in rows:
        total = _event_number(row.get("source_weight_sum_pct"))
        if row.get("source_weight_sum_reconciled_to_100") is False and total is not None:
            lines.extend(["", f"{row['ticker']}原始披露权重合计{total:.3f}%，"
                          "尚有对账差异；保留原始权重、未重新归一化，以上仅作敞口参考。"])
    lines.extend(["", "示例假设指数内这一组股票整体跌5%、其余成分不变；"
                  "不直接用SMH涨跌代替指数内半导体组合，也不估计其他行业的连带反应。"
                  "持仓快照的取得时间另存于输入记录，不能倒推为当日已知。", ""])
    return lines


def _render_strategy_context(
    scenario: Dict[str, Any], structure: Dict[str, Any], plan: Dict[str, Any]
) -> List[str]:
    lines = []
    chain = scenario.get("causal_chain", [])
    if chain:
        lines.extend(["**为什么这样判断：** " + "；".join(str(row).rstrip("。") for row in chain)
                      + "。", ""])
    if scenario.get("events"):
        threshold = scenario.get("event_threshold", DEFAULT_EVENT_MAIN_CASE_THRESHOLD)
        lines.extend([
            f"事件Yes或No达到{threshold*100:g}%"
            "就作为该事件的主情景，未达到则保留分支。"
            "这是事件结果的市场定价，不是QQQ上涨概率；高概率维持利率不变意味着预期已偏向兑现，"
            "仍要由收益率、信用和价格反应决定是否加仓。", "",
        ])
    lines.extend(_render_regional_transmission(scenario))
    for ticker, item in structure.get("by_ticker", {}).items():
        if item.get("status") != "ready":
            continue
        zones = item.get("zones", {})
        lines.extend([
            f"**{ticker}当前结构（{item.get('price_date')}，日线价"
            f"{_action_price(item.get('close'))}）：** "
            f"近端支撑 {_zone_text(zones.get('support'))}；"
            f"深回调区 {_zone_text(zones.get('deep'))}；"
            f"极限观察区 {_zone_text(zones.get('extreme'))}。", "",
        ])
        primary_selected = ticker == plan.get("ticker") and plan.get("selected_zone")
        selected = primary_selected or zones.get("extreme") or zones.get("deep")
        if selected:
            labels = list(dict.fromkeys(str(origin.get("label")) for origin in
                                        selected.get("origins", []) if origin.get("label")))
            labels = [{"sma200": "200日均线", "weekly_sma30": "30周均线",
                       "weekly_sma20": "20周均线", "weekly_sma50": "50周均线",
                       "confirmed_pivot_low": "已确认摆动低点",
                       "prior_traded_weekly_close": "历史周收盘区域",
                       "conditional_range_height": "条件式区间高度投影",
                       "derived_recent_support": "近期结构低点",
                       "range_low": "历史区间低点"}.get(label, label) for label in labels]
            lines.extend([
                ("当前采用区域依据：" if primary_selected else "极限/深回调区域依据：")
                + "、".join(labels[:5])
                + "。先收复区域上沿，再等后续完整日线回踩守住；触价本身不构成买入。", "",
            ])
    if structure and not any(item.get("status") == "ready" for item in
                             structure.get("by_ticker", {}).values()):
        lines.extend(["价格操作暂缓：当前历史长度、完整交易日或来源不足以构建动态区域；"
                      "上面的宏观主判断保留。", ""])
    policy = scenario.get("policy", {})
    if policy:
        lines.extend([
            f"**事件怎样影响仓位：** 当前事件阶段的首批政策上限为总账户"
            f"{_action_pct(policy.get('initial_cap_pct'))}；事件落地并再次确认后，"
            f"才可另行复核合计 {_action_pct(policy.get('post_event_cap_pct'))} 的预案。"
            "这两个额度不相加，且还要受结构、盈亏空间和临近事件限制；"
            "实际可启用额以行动卡为准。", "",
        ])
    if plan.get("selected_zone"):
        tier_name = {"support": "近端支撑的修复试仓区", "deep": "深回调区",
                     "extreme": "极限观察区"}.get(plan.get("selected_tier"), "待确认区域")
        lines.extend([
            "当前主情景的行动卡选择" + tier_name
            + "；另外两层保留为改判后的备选，不重复计算账户预算。", "",
        ])
    return lines


def _render_regional_transmission(scenario: Dict[str, Any]) -> List[str]:
    rows = scenario.get("regional_transmission", [])
    if not rows:
        return []
    lines = [
        "**事件影响的先后顺序：** 美国利率、信用与直接政策为核心；"
        "日本利率为辅助，重点看日元融资压力；韩国对美股大盘权重更低，"
        "重点看存储与设备产业链。这里的高低是判断优先级，尚非回测校准的数值权重。", "",
        "| 辅助因素 | 事件主情景 | 已观察传导 | 怎样影响操作 |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        usable = row.get("event_quality") == "usable"
        event = (row.get("event_semantics") if usable else
                 "本报告日期无通过校验的事件概率，未回填后续报价")
        probability = _event_number(row.get("event_probability")) if usable else None
        if usable and row.get("event_main_case") == "分支":
            yes = _event_number(row.get("event_yes_probability"))
            no = _event_number(row.get("event_no_probability"))
            if yes is not None and no is not None and 0 <= yes <= 1 and 0 <= no <= 1:
                event = f"{event}（Yes {yes*100:.1f}% / No {no*100:.1f}%）"
        elif probability is not None and 0 <= probability <= 1:
            event = f"{event}（{probability*100:.1f}%）"
        fx = row.get("fx") or {}
        semi = row.get("semiconductor") or {}
        evidence = []
        move = _event_number(fx.get("change_5_pct"))
        if move is not None:
            evidence.append(f"{fx.get('series_id')}近5个观察间隔{move:+.2f}%"
                            f"（{fx.get('date')}）")
        strength = _event_number(fx.get("local_currency_strength_5_pct"))
        if row.get("region") == "Japan" and strength is not None:
            evidence.append(f"对应日元价值{strength:+.2f}%")
        if fx.get("alignment") in {"misaligned", "unavailable"}:
            evidence.append("汇率窗口未对齐，暂停传导判定")
        lag = _event_number(semi.get("vs_spy_5d_pp"))
        if lag is not None:
            evidence.append(f"SMH近5个完整交易日相对SPY {lag:+.2f}个百分点"
                            f"（截至{semi.get('end_date')}）")
        if not evidence:
            evidence.append("汇率或半导体价格传导待核验")
        lines.append("| " + " | ".join(_table_text(value) for value in (
            {"Japan": "日本", "Korea": "韩国"}.get(row.get("region"),
                                                   row.get("region", "海外")),
            event or "事件方向待核验", "；".join(evidence),
            row.get("action") or "仅作辅助观察，不单独触发买入",
        )) + " |")
    lines.extend(["", "日韩利率预期先经过汇率、融资或产业链，再由半导体价格验证；"
                  "不能把加息概率直接当作半导体下跌概率。日本若出现日元急升与半导体同步走弱，"
                  "才升级为大盘仓位风险；韩国的传导先用于半导体及QQQ的优先级判断。", ""])
    lines.extend(["当前风控观察门槛：日元近5个观察间隔升值≥2%，或美元兑韩元上升≥3%；"
                  "同时SMH近5个完整交易日落后SPY≥2个百分点。这些是待验证的风控规则，"
                  "不表示已确认因果关系或胜率。", ""])
    return lines


def _render_scenario_branches(scenario: Dict[str, Any]) -> List[str]:
    branches = scenario.get("branches", [])
    if not branches:
        return []
    lines = [
        "### 主情景如何转向行动", "",
        "以下额度是未来复核的政策上限；只有新价格确认且行动卡允许时才启用，"
        "不随事件概率自动成交。", "",
        "| 情景 | 需要看到什么 | 对市场的判断 | 行动预案 |",
        "| --- | --- | --- | --- |",
    ]
    for row in branches:
        lines.append("| " + " | ".join(_table_text(row.get(key)) for key in
                                      ("name", "trigger", "market_implication", "action")) + " |")
    return lines + [""]


def _build_analyst_study(paths: ProjectPaths, report_date: str) -> Dict[str, Any]:
    corpus = read_json(paths.processed / "eagle-corpus.json", default={}) or {}
    profile = load_analyst_patterns_config(paths.config / "analyst-patterns.yml")
    comparisons = []
    for row in profile.get("comparisons", []):
        if str(row.get("available_from") or "9999") > report_date:
            continue
        evidence = [item for item in row.get("evidence", [])
                    if item.get("date") and str(item["date"]) <= report_date]
        if evidence:
            comparisons.append({**row, "evidence": evidence})
    dated = [row for row in corpus.get("messages", []) if row.get("date")
             and str(row["date"]) <= report_date]
    replays = read_json(paths.processed / f"analyst-replays-{report_date}.json", default=[]) or []
    if isinstance(replays, dict):
        replays = replays.get("cases", [])
    replays = [row for row in replays if row.get("published_date")
               and str(row["published_date"]) <= report_date
               and str(row.get("price_cutoff") or "9999") < str(row["published_date"])]
    return {
        "as_of": report_date, "coverage": corpus.get("coverage", {}),
        "dated_posts_as_of": len(dated), "comparisons": comparisons,
        "resources": profile.get("resources", []), "notes": profile.get("notes", []),
        "limitations": corpus.get("limitations", []),
        "replays": replays,
    }


def _render_analyst_study(study: Dict[str, Any]) -> List[str]:
    coverage = study.get("coverage", {})
    comparisons = study.get("comparisons", [])
    if not coverage and not comparisons and not study.get("replays"):
        return []
    lines = ["## 鹰哥与顺哥：学判断方法，不照抄旧价位", ""]
    if coverage:
        lines.extend([
            f"材料覆盖 {coverage.get('first_date')}–{coverage.get('last_date')}，"
            f"共{coverage.get('unique_posts')}篇唯一帖子，剔除"
            f"{coverage.get('duplicates_removed')}篇重复，"
            f"{coverage.get('unresolved_dates')}篇日期未解。"
            f"截至本报告日期可定日的帖子为{study.get('dated_posts_as_of')}篇。"
            "这是选摘材料，不是完整一年；原文发布时区未知。", "",
        ])
    if comparisons:
        lines.extend([
            "| 可复用判断 | 两位作者的共同点与差别 | 有日期的来源证据 | 何时失效／不能推断什么 |",
            "| --- | --- | --- | --- |",
        ])
        for row in comparisons:
            comparison = (f"{row.get('common')} 鹰哥：{row.get('eagle')} "
                          f"顺哥：{row.get('shunge')} 差别：{row.get('difference')}")
            evidence = "；".join(f"{item.get('author')} {item.get('date')} "
                                 f"{item.get('time') or ''}：{item.get('summary')}"
                                 for item in row.get("evidence", []))
            lines.append("| " + " | ".join(_table_text(value) for value in (
                row.get("name"), comparison, evidence, row.get("failure_conditions")
            )) + " |")
        lines.append("")
    lines.extend([
        "**如何落到今天：** 事件先定主情景，再看利率、油价、信用与板块价格是否响应，"
        "最后由当前数据形成区域、收复和回踩信号。动态区域不是作者旧数字的外推。"
        "配置中的本轮资金上限、分批比例和止盈比例是本系统的政策，不能归为作者验证过的配方。", "",
    ])
    replays = study.get("replays", [])
    if replays:
        lines.extend([
            "**站在发言前一完整交易日：** 以下用事后下载的历史价格重建条件，不是当时留存快照，"
            "不使用发言日尚未完成的价格，也不据此宣称预测命中。", "",
            "| 发言日期／标的 | 作者当时条件 | 此前可观察环境 | 能推断什么 |",
            "| --- | --- | --- | --- |",
        ])
        for row in replays:
            features = row.get("features", {})
            environment = "缺少精确标的的完整历史，未以其他ETF冒充。"
            if row.get("status") == "reconstructed" and features:
                environment = (
                    f"截至{row.get('price_cutoff')}，日线价{_action_price(features.get('close'))}；"
                    f"SMA20代理{_action_price(features.get('sma20_proxy'))}、"
                    f"SMA60代理{_action_price(features.get('sma60_proxy'))}；"
                    f"此前40日区间{_action_price(features.get('prior40_low'))}–"
                    f"{_action_price(features.get('prior40_high'))}。"
                )
                counter = features.get("public_9bar_proxy", {})
                if counter:
                    environment += (f"公开计数代理：买序{counter.get('buy_count')}、"
                                    f"卖序{counter.get('sell_count')}；作者版本未核实。")
            lines.append("| " + " | ".join(_table_text(value) for value in (
                f"{row.get('published_date')} / {row.get('ticker')}", row.get("author_rule"),
                environment, row.get("limits") or "只重建前置条件，不等同于未来结果。",
            )) + " |")
        lines.append("")
    for resource in study.get("resources", []):
        url = str(resource.get("url") or "")
        if url.startswith("https://"):
            access = "仅核验标题／说明，未取得字幕正文" if resource.get("access_level") == (
                "description_only"
            ) else "公开原始资料"
            lines.append(f"- [{resource.get('name')}]({url})（{access}）："
                         f"{resource.get('finding')}")
    for note in study.get("notes", []):
        lines.append(f"- {note}")
    for note in study.get("limitations", []):
        lines.append(f"- {note}")
    return lines + [""]


def _render_daily_brief(
    brief: Dict[str, Any], watchlist_table: List[Dict[str, Any]],
    prediction_markets: List[Dict[str, Any]], action_plan: Dict[str, Any],
    cross_market: List[Dict[str, Any]], scenario_context: Dict[str, Any],
    market_structure: Dict[str, Any], bundle: Any = None,
) -> List[str]:
    if not brief:
        return ["## 今日结论", "", "等待：尚无可核验的方向判断。", ""]
    portfolio_mode = bundle is not None and bool(bundle.portfolio_plan)
    elapsed_fed = bundle is not None and _expired_today(bundle)
    if elapsed_fed:
        action_plan = {**action_plan, "current_action": str(action_plan.get("current_action") or "")
                       .replace("关键事件临近", "关键事件结果待核验")}
    headline = scenario_context.get("headline") or brief.get("headline", "等待可核验数据")
    if elapsed_fed:
        headline = str(headline).replace("关键事件临近", "关键事件结果待核验")
    if portfolio_mode:
        headline = ("当前组合模型待核验，不据此新增。" if _portfolio_reason(bundle)
                    else _portfolio_title(bundle.portfolio_plan))
    lines = [
        "## 今日结论", "",
        f"**{headline}**", "",
    ]
    if portfolio_mode:
        lines.extend(_render_portfolio_reference(bundle))
    elif brief.get("price_data_pending"):
        lines.extend(["**今天怎么做：当前新增买入信号 0%；先补齐最新完整交易日行情。**", ""])
    elif action_plan.get("current_action"):
        lines.extend([
            f"**今天怎么做：当前新增买入信号 {_action_pct(action_plan.get('buy_now_pct'))}。"
            f"{action_plan['current_action']}**", "",
        ])
    if portfolio_mode:
        pass
    elif brief.get("price_data_pending"):
        lines.extend(_render_action_plan({
            "status": "unavailable", "limitations": ["先补齐最新完整交易日行情，再重算操作价位。"],
        }))
    else:
        lines.extend(_render_action_plan(action_plan))
    if bundle is not None:
        lines.extend(_render_directional_context(bundle))
        lines.extend(_render_options_reasons(bundle))
        lines.extend(_render_cloud_capex(bundle))
    if not portfolio_mode:
        lines.extend(_render_strategy_context(scenario_context, market_structure, action_plan))
        lines.extend(_render_scenario_branches(scenario_context))
    lines.extend(_render_global_context(
        scenario_context.get("global_context", {}), str(scenario_context.get("date"))
    ))
    if scenario_context:
        price_view = str(brief.get("headline") or "等待可核验价格").rstrip("。")
        lines.extend([f"价格确认层：{price_view}。", ""])
    macro_features = scenario_context.get("macro_features", {})
    macro_observed = bool(macro_features.get("yield10") and macro_features.get("credit"))
    usable_tickers = {
        row["ticker"] for row in brief.get("ticker_decisions", []) if row.get("usable")
    }
    snapshots = [
        f"{row['ticker']} {row['data_date']}" for row in watchlist_table
        if row.get("ticker") in {"QQQ", "SPY"} & usable_tickers
    ]
    if snapshots:
        lines.extend(["行情快照（日线）：" + "；".join(snapshots) + "。", ""])
    horizon = brief.get("horizon", {})
    lines.append(f"**未来1–5个交易日：** {horizon.get('short', '等待数据。')}")
    lines.extend(["", f"**未来1–4周：** {horizon.get('medium', '等待条件确认。')}", ""])
    evidence = brief.get("evidence", [])
    if evidence:
        lines.append("判断依据：" + "；".join(str(item).rstrip("。") for item in evidence) + "。")
        lines.append("")
    scenarios = brief.get("market_scenarios", [])
    if scenarios:
        lines.extend([
            "### 大盘条件与失效位", "",
            "| 标的 | 当前判断 | 上行确认与观察目标 | 下行确认与观察目标 | 判断失效 |",
            "| --- | --- | --- | --- | --- |",
        ])
        for row in scenarios:
            lines.append(
                "| " + " | ".join(_table_text(row.get(key)) for key in (
                    "ticker", "view", "bullish", "bearish", "invalidation"
                )) + " |"
            )
        lines.append("")
    actions = brief.get("actions", [])
    if actions and not portfolio_mode:
        lines.extend(["### 按持仓状态应对", ""])
        for row in actions:
            lines.append(f"- **{row.get('profile')}：** {row.get('action')}")
        lines.append("")
    lines.extend(_render_prediction_markets(
        prediction_markets, str(brief.get("date", "")),
        scenario_context.get("event_threshold", DEFAULT_EVENT_MAIN_CASE_THRESHOLD),
    ))
    lines.extend(_render_cross_market(cross_market, str(brief.get("date", ""))))
    sectors = brief.get("sectors", [])
    if sectors:
        lines.extend([
            "### 板块观察（同区间数据才比较相对强弱）", "",
            "| 观察标的 | 相对强弱 | 证据与执行条件 |",
            "| --- | --- | --- |",
        ])
        for row in sectors:
            lines.append(
                "| " + " | ".join(_table_text(row.get(key)) for key in (
                    "ticker", "view", "condition"
                )) + " |"
            )
        lines.append("")
    missing = brief.get("missing", [])
    if missing:
        missing = [
            "独立宏观风险门槛未完整确认；已有利率与信用观察，仍不自动提高额度或安排卖PUT"
            if macro_observed and str(item).startswith("宏观未知") else item for item in missing
        ]
        lines.append(
            "**判断边界：** " + "；".join(str(item).rstrip("。") for item in missing) + "。"
        )
        lines.append("")
    decisions = brief.get("ticker_decisions", [])
    if decisions:
        lines.extend([
            "### 个股方向与现在怎么做", "",
            ("以下为个股技术观察，不另设资金；组合配置以上方模型参考为准。" if portfolio_mode else
             "以下是方向观察，不另设新增资金；本轮仓位安排以上面的单一行动卡为准。"), "",
            "| 标的 | 方向 | 现在怎么做 | 转强条件 | 转弱与判断失效 |",
            "| --- | --- | --- | --- | --- |",
        ])
        for row in decisions:
            invalidation = " ".join(
                str(value) for value in (row.get("bearish"), row.get("invalidation")) if value
            )
            lines.append(
                "| " + " | ".join(_table_text(value) for value in (
                    row.get("ticker"), row.get("view"), row.get("action"),
                    row.get("bullish"), invalidation,
                )) + " |"
            )
        lines.append("")
    return lines


def build_report_bundle(
    paths: ProjectPaths,
    report_date: str,
    watchlist: Iterable[WatchedAsset],
    event_threshold: Any = None,
    analysis_clock: Any = None,
) -> ReportBundle:
    prediction_config = load_prediction_markets_config(paths.config / "prediction-markets.yml")
    threshold = validate_event_threshold(
        prediction_config["main_case_threshold"] if event_threshold is None else event_threshold
    )
    expected_date = expected_complete_session(report_date)
    market_data = read_json(paths.raw / f"market-{report_date}.json", default={}) or {}
    sentiment_summary = (
        read_json(paths.processed / f"sentiment-summary-{report_date}.json", default={}) or {}
    )
    technical = read_json(paths.processed / f"technical-{report_date}.json", default={}) or {}
    backtests = read_json(paths.processed / f"backtests-{report_date}.json", default={}) or {}
    options = read_json(paths.processed / f"options-{report_date}.json", default={}) or {}
    macro = read_json(paths.processed / f"macro-{report_date}.json", default=[]) or []
    warnings = read_json(paths.processed / f"warnings-{report_date}.json", default=[]) or []
    events = read_json(paths.processed / f"events-{report_date}.json", default=[]) or []
    playbook = read_json(paths.processed / f"playbook-{report_date}.json", default={}) or {}
    gate = read_json(paths.processed / f"macro-gate-{report_date}.json", default={}) or {}
    leadership = read_json(paths.processed / f"leadership-{report_date}.json", default={}) or {}
    ipo_watch = read_json(paths.processed / f"ipos-{report_date}.json", default={}) or {}
    fundamentals = (
        read_json(paths.processed / f"fundamentals-{report_date}.json", default={}) or {}
    )
    cross_market = read_json(paths.processed / f"cross-market-{report_date}.json", default=[]) or []
    directional_context = read_json(
        paths.processed / f"direction-probabilities-{report_date}.json", default={}
    ) or {}
    cloud_capex_context = read_json(
        paths.processed / f"cloud-capex-{report_date}.json", default={}
    ) or {}
    portfolio_plan = read_json(
        paths.processed / f"portfolio-plan-{report_date}.json", default={}
    ) or {}
    prediction_markets = read_json(
        paths.processed / f"prediction-markets-{report_date}.json", default=[]
    ) or []
    registered = {str(row.get("id")): row for row in prediction_config.get("markets", [])}
    for index, row in enumerate(prediction_markets):
        contract = registered.get(str(row.get("id")), {})
        if contract.get("expected_question") == row.get("question"):
            row = {**row, **{key: contract[key] for key in (
                "label", "yes_scenario", "no_scenario", "settlement_scope_note"
            ) if key in contract}}
            prediction_markets[index] = row
        probability, reason = _event_quote(row, report_date)
        if probability is None:
            prediction_markets[index] = {
                **row, "status": "unavailable", "yes_probability": None,
                "no_probability": None, "reason": reason,
            }

    # Normalize legacy labels without altering the cached statement inputs or scores.
    for snapshot in fundamentals.values():
        try:
            fcf = float(snapshot.get("free_cash_flow"))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(fcf) and fcf >= 0 and snapshot.get("label") in {"burner", "distress"}:
            snapshot["label"] = "pressured"

    # Files created by earlier versions did not identify generated sample rows.
    for ticker, rows in market_data.items():
        for row in rows:
            if any(f"sample OHLCV for {ticker};" in warning for warning in warnings):
                row.setdefault("source", "sample")
        market_data[ticker] = valid_market_rows(rows, date.fromisoformat(report_date))
    watchlist_table = []
    per_ticker_notes: Dict[str, List[str]] = {}
    for asset in watchlist:
        ticker = asset.ticker
        rows = market_data.get(ticker, [])
        source = market_source(rows)
        observed = str(rows[-1].get("date", ""))[:10] if rows else None
        stale = observed is None or observed < expected_date
        usable = bool(rows) and source != "sample"
        actionable = source == "yfinance" and not stale
        tech = technical.get(ticker, {}) if usable else {}
        sentiment = sentiment_summary.get(ticker, {})
        option_context = options.get(ticker, {}) if actionable else {}
        backtest = backtests.get(ticker, {}) if usable else {}
        option_warnings = (
            option_context.get("warnings", []) if isinstance(option_context, dict) else []
        )
        if option_warnings:
            warnings.extend([f"{ticker}: {warning}" for warning in option_warnings])
        for warning in fundamentals.get(ticker, {}).get("warnings", []):
            warnings.append(f"{ticker}: {warning}")
        latest_price = _latest_close(rows)
        supports, resistances = _levels_by_kind(tech.get("levels", []))
        row = {
            "ticker": ticker,
            "name": asset.name,
            "data_source": source,
            "data_date": observed,
            "expected_data_date": expected_date,
            "close_source": rows[-1].get("close_source", "yahoo_daily_history") if rows else None,
            "session_complete": rows[-1].get("session_complete") if rows else None,
            "history_start": rows[0].get("date") if rows else None,
            "return_20d_pct": (
                round((float(rows[-1]["close"]) / float(rows[-21]["close"]) - 1) * 100, 2)
                if source == "yfinance" and len(rows) >= 21 else None
            ),
            "return_20d_start": rows[-21]["date"] if len(rows) >= 21 else None,
            "stale": stale,
            "close": round(latest_price, 2) if latest_price else None,
            "regime": tech.get("regime", "unknown"),
            "sentiment_score": sentiment.get("weighted_score", 0.0),
            "sentiment_weight": sentiment.get("total_weight", 0.0),
            "sentiment_view": _sentiment_interpretation(sentiment)["summary"],
            "support_levels": _format_near_levels_with_fallback(supports, limit=2),
            "resistance_levels": _format_near_levels_with_fallback(resistances, limit=2),
            "backtest_return": backtest.get("metrics", {}).get("total_return"),
            "option_status": "available" if option_context.get("contracts") else "unavailable",
        }
        observation = _current_cluster_prices(rows, expected_date) if actionable else {}
        row["observation_support_prices"] = observation.get("support", [])
        row["observation_resistance_prices"] = observation.get("resistance", [])
        if (option_context.get("contracts")
                and not _aligned_options(option_context, row, report_date)):
            warnings.append(
                f"{ticker}: option context withheld; underlying date/spot or fetch provenance "
                "does not match the report's current market snapshot."
            )
            row["option_status"] = "unavailable"
            options.pop(ticker, None)
        watchlist_table.append(row)
        notes = [f"Market data: {source}; latest bar {observed or 'unavailable'}."]
        if rows and rows[-1].get("close_source"):
            notes.append(f"Close provenance: {rows[-1]['close_source']}.")
        if source == "sample":
            notes.append(
                "Generated demonstration prices; trading levels, backtests and playbook withheld."
            )
        elif source == "unverified":
            notes.append(
                "Cached prices have no recorded provider; "
                "actionable playbook withheld until refreshed live."
            )
        elif source == "unavailable":
            notes.append("No market prices available; trading analysis withheld.")
        if stale:
            notes.append(
                "Market prices are stale for this report date; actionable playbook withheld."
            )
            warnings.append(
                f"{ticker}: latest market bar {observed} is stale; "
                f"expected completed session {expected_date}."
            )
        if supports:
            immediate_supports, intermediate_supports, deep_supports = _split_level_tiers(
                supports
            )
            if immediate_supports:
                notes.append(
                    "Immediate support levels: "
                    f"{_format_levels(immediate_supports, limit=4)}."
                )
            if intermediate_supports:
                notes.append(
                    "Intermediate downside checkpoints: "
                    f"{_format_levels(intermediate_supports, limit=4)}."
                )
            else:
                notes.append(
                    "Intermediate downside checkpoints: none found between the immediate "
                    "and deeper ranges in the current lookback."
                )
            if deep_supports:
                notes.append(
                    "Deeper downside checkpoints, not immediate support: "
                    f"{_format_levels(deep_supports, limit=4)}."
                )
            else:
                notes.append(
                    "Deeper downside checkpoints: none found beyond the intermediate "
                    "range in the current lookback."
                )
        if resistances:
            immediate_resistances, intermediate_resistances, deep_resistances = (
                _split_level_tiers(resistances)
            )
            if immediate_resistances:
                notes.append(
                    "Immediate resistance / possible ceiling levels: "
                    f"{_format_levels(immediate_resistances, limit=4)}."
                )
            if intermediate_resistances:
                notes.append(
                    "Intermediate upside checkpoints: "
                    f"{_format_levels(intermediate_resistances, limit=4)}."
                )
            else:
                notes.append(
                    "Intermediate upside checkpoints: none found between the immediate "
                    "and higher ranges in the current lookback."
                )
            if deep_resistances:
                notes.append(
                    "Higher upside checkpoints, not nearby resistance: "
                    f"{_format_levels(deep_resistances, limit=4)}."
                )
            else:
                notes.append(
                    "Higher upside checkpoints: none found beyond the intermediate "
                    "range in the current lookback."
                )
        notes.append(_technical_temperature(tech.get("indicators", {})))
        notes.append(f"Sentiment: {_sentiment_interpretation(sentiment)['detail']}")
        notes.append(_backtest_interpretation(backtest))
        if actionable:
            notes.extend(_playbook_notes(playbook.get(ticker, {})))
        ipo_info = ipo_watch.get(ticker, {}) if actionable else {}
        for verdict in ipo_info.get("verdicts", []):
            notes.append(f"IPO watch ({ipo_info.get('stage')}): {verdict}")
        per_ticker_notes[ticker] = notes or ["No high-confidence notes generated."]

    by_ticker = {row["ticker"]: row for row in watchlist_table}
    options = {
        ticker: snapshot for ticker, snapshot in options.items()
        if _aligned_options(snapshot, by_ticker.get(ticker, {}), report_date)
    }
    trusted = {
        row["ticker"] for row in watchlist_table
        if row["data_source"] == "yfinance" and not row["stale"]
    }
    playbook = {ticker: pb for ticker, pb in playbook.items() if ticker in trusted}
    if leadership and (leadership.get("benchmark") not in trusted or any(
        ticker not in trusted for ticker in leadership.get("leaders", {})
    )):
        leadership = {
            "state": "unknown",
            "rule": "Leadership signal withheld: verified current market data required.",
        }
    for ticker, snapshot in ipo_watch.items():
        if ticker not in trusted:
            snapshot = dict(snapshot)
            snapshot.update(close=None, vs_ipo_price_pct=None, verdicts=[])
            ipo_watch[ticker] = snapshot
    macro_as_of = []
    for row in macro:
        observations = [
            obs for obs in row.get("history", [])
            if obs.get("date") and str(obs["date"])[:10] <= report_date
        ]
        if observations:
            latest = max(observations, key=lambda obs: str(obs["date"]))
            macro_as_of.append({**row, **latest, "history": observations})
        elif row.get("date") and str(row["date"])[:10] <= report_date:
            macro_as_of.append(row)
    macro_radar = macro_as_of + _macro_from_raw_items(paths, report_date)
    risk_flags = sorted(set(str(warning) for warning in warnings if warning))
    daily_brief = build_daily_brief(
        report_date=report_date, watchlist_table=watchlist_table, technical=technical,
        playbook=playbook, macro_gate=gate, leadership=leadership, events=events,
        fundamentals=fundamentals,
    )
    anchor_rows = {row["ticker"]: row for row in watchlist_table if row["ticker"] in {"QQQ", "SPY"}}
    pending_prices = [
        f"{ticker}：最新{anchor_rows.get(ticker, {}).get('data_date') or '缺失'}"
        for ticker in ("QQQ", "SPY")
        if anchor_rows.get(ticker, {}).get("data_source") != "yfinance"
        or anchor_rows.get(ticker, {}).get("stale", True)
    ]
    daily_brief["expected_data_date"] = expected_date
    daily_brief["price_data_pending"] = pending_prices
    supporting_market = dict(market_data)
    extra_market = read_json(paths.raw / f"market-context-{report_date}.json", {}) or {}
    for ticker, rows in extra_market.items():
        if ticker not in supporting_market:
            supporting_market[ticker] = valid_market_rows(rows, date.fromisoformat(report_date))
    if portfolio_plan and isinstance(portfolio_plan, dict):
        portfolio_plan["report_policy_proof_error"] = (
            _portfolio_proof_reason(paths, report_date, portfolio_plan)
            if portfolio_plan.get("status") in {"available", "reference_only"} else "")
        # Rebuild this comparison from archived market inputs, never from a cached plan's assertion.
        portfolio_plan["report_price_inputs"] = {}
        for ticker in ("QQQ", "SPY", "TQQQ"):
            rows = supporting_market.get(ticker, [])
            completed = [r for r in rows if str(r.get("date") or "9999") <= expected_date
                         and r.get("session_complete") is True]
            latest = completed[-1] if completed else {}
            portfolio_plan["report_price_inputs"][ticker] = {
                "date": latest.get("date"), "close": latest.get("raw_close", latest.get("close")),
                "source": market_source(completed),
                "session_complete": latest.get("session_complete"),
                "fetched_at": latest.get("fetched_at"),
            }
    policy = dict(load_allocation_config(paths.config / "allocation.yml"))
    scenario_context = build_scenario_context(
        report_date, macro_as_of, prediction_markets, supporting_market,
        global_context=read_json(
            paths.processed / f"global-context-{report_date}.json", default={}
        ) or {},
        contracts=prediction_config.get("markets", []), event_threshold=threshold,
        now=analysis_clock,
        allocation_policy=policy,
    )
    market_structure = build_market_structure(report_date, market_data, scenario_context)
    scenario_policy = scenario_context.get("policy", {})
    for key in ("initial_cap_pct", "post_event_cap_pct", "before_event_cap_pct",
                "event_main_case_confirmed", "preferred_entry_tier", "reason"):
        if key in scenario_policy:
            policy[key] = scenario_policy[key]
    if scenario_policy.get("max_new_capital_pct") is not None:
        policy["max_new_capital_pct"] = min(
            policy.get("max_new_capital_pct", 10), scenario_policy["max_new_capital_pct"]
        )
    action_plan = build_dynamic_action_plan(
        structure=market_structure, macro_gate=gate, policy=policy, events=events,
    )
    primary = str(policy.get("primary_ticker") or "QQQ")
    action_plan.setdefault("model_inputs", {})["close_source"] = anchor_rows.get(
        primary, {}
    ).get("close_source")
    # Historical author cases are retained for study, never passed into live decisions.
    shunge_context = build_shunge_context(
        report_date, market_data, load_shunge_config(paths.config / "shunge.yml"),
        case_history=read_json(
            paths.processed / "shunge-case-QQQ-2026-10-21.json", default=None
        ),
    )
    shunge_context["scenario"] = {"active": False, "study_only": True}
    if pending_prices:
        action_plan.update(
            status="unavailable", buy_now_pct=0.0, entries=[], exits=[], stop={},
            selected_zone=None, selected_tier=None,
            current_action="最新交易日行情未齐，当前暂停新增买入。",
        )
        budget = action_plan.setdefault("budget", {})
        budget.update(
            planned_capital_pct=0.0, currently_enabled_pct=0.0,
            estimated_weighted_entry=None, uncommitted_cash_pct=budget.get("max_new_capital_pct"),
        )
        if action_plan.get("signal"):
            action_plan["signal"] = {**action_plan["signal"], "buy_signal": False}
        action_plan.setdefault("limitations", []).append(
            f"应有完整交易日{expected_date}；" + "；".join(pending_prices)
        )
    _add_event_context(daily_brief, prediction_markets, gate)
    return ReportBundle(
        date=report_date,
        watchlist_table=watchlist_table,
        macro_radar=macro_radar,
        per_ticker_notes=per_ticker_notes,
        sentiment_deltas=sentiment_summary,
        technical_setup=technical,
        options_context=options,
        risk_flags=risk_flags,
        events=events,
        playbook=playbook,
        macro_gate=gate,
        leadership=leadership,
        ipo_watch=ipo_watch,
        fundamentals=fundamentals,
        daily_brief=daily_brief,
        prediction_markets=prediction_markets,
        action_plan=action_plan,
        cross_market=cross_market,
        shunge_context=shunge_context,
        market_structure=market_structure,
        scenario_context=scenario_context,
        analyst_study=_build_analyst_study(paths, report_date),
        directional_context=directional_context,
        cloud_capex_context=cloud_capex_context,
        portfolio_plan=portfolio_plan,
    )


def _render_run_audit(bundle: ReportBundle) -> List[str]:
    run = bundle.analysis_run
    if not run:
        return []
    run_id = str(run.get("run_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        return ["## 本次分析追溯", "", "来源记录保存在本次分析归档。", ""]
    link = f"../data/runs/{bundle.date}/{run_id}/manifest.json"
    return ["## 本次分析追溯", "", f"[查看本次来源、时点与验证记录]({link})。", ""]


def render_research_markdown(bundle: ReportBundle) -> str:
    """Explicit historical research view, excluded from every daily default."""
    lines = [f"# 历史研究附录 — {bundle.date}", ""]
    lines.extend(_render_analyst_study(bundle.analyst_study))
    lines.extend(_render_shunge_context(
        bundle.shunge_context, str(bundle.daily_brief.get("expected_data_date", "")), study=True
    ))
    return "\n".join(lines)


def render_detailed_markdown(bundle: ReportBundle) -> str:
    lines = [f"# Trading Wizzard 日报 — {bundle.date}", ""]
    pending = bundle.daily_brief.get("price_data_pending", [])
    if pending:
        lines.extend([
            "**最新交易日行情未齐，暂停抄底/加仓/止盈价建议。**", "",
            f"应有完整交易日：{bundle.daily_brief.get('expected_data_date')}；"
            + "；".join(pending) + "。",
            "",
        ])
    lines.extend(_render_daily_brief(
        bundle.daily_brief, bundle.watchlist_table, bundle.prediction_markets, bundle.action_plan,
        bundle.cross_market, bundle.scenario_context, bundle.market_structure, bundle
    ))
    lines.extend([
        "## 数据与技术明细", "",
        "以下保留行情、技术位、回测和来源，供核验上面的结论。"
        "本报告未接入个人账户，持仓应对按情景区分。", "",
        "情绪口径：最近7天、有日期且去重的讨论；未来和无效日期不参与。"
        "默认使用金融词典评分，只有显式启用且可用时才使用FinBERT。", "",
        "历史报告按观察日期筛选，但无法重建当时版本的数据。", "",
        "## Data Coverage", "",
    ])
    sample_count = sum(row.get("data_source") == "sample" for row in bundle.watchlist_table)
    verified_count = sum(row.get("data_source") == "yfinance" for row in bundle.watchlist_table)
    lines.append(
        f"Verified provider prices: {verified_count}/{len(bundle.watchlist_table)} tickers. "
        f"Generated sample prices: {sample_count}."
    )
    if sample_count:
        lines.append(
            "**DEMO DATA PRESENT: generated prices are not market observations; "
            "trading signals are withheld for these tickers.**"
        )
    if verified_count < len(bundle.watchlist_table):
        lines.append(
            "Refresh with `twiz report --live` after installing the providers extra "
            "to obtain verified market data."
        )
    lines.extend([
        "", "## Watchlist", "",
        "| Ticker | Close | Source / Last Bar | Regime | Sentiment Read | Support | "
        "Resistance | Backtest | Options |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in bundle.watchlist_table:
        backtest = row.get("backtest_return")
        backtest_text = "" if backtest is None else f"{float(backtest) * 100:.1f}% strategy"
        close = "" if row.get("close") is None else f"{float(row['close']):.2f}"
        lines.append(
            f"| {row['ticker']} | {close} | {row.get('data_source', 'unverified')} / "
            f"{row.get('data_date') or 'unavailable'}{' (stale)' if row.get('stale') else ''} | "
            f"{row['regime']} | {row['sentiment_view']} | "
            f"{row['support_levels']} | {row['resistance_levels']} | "
            f"{backtest_text} | {row['option_status']} |"
        )

    lines.extend(["", "## Macro Radar", ""])
    if bundle.macro_radar:
        for item in bundle.macro_radar[:20]:
            label = item.get("name") or item.get("title") or item.get("series_id") or "Macro item"
            value = item.get("value")
            suffix = f": {value}" if value not in (None, "") else ""
            observed = item.get("date") or item.get("published_at")
            stamp = f" (observed {observed})" if observed else ""
            lines.append(f"- {item.get('source', 'source')}: {label}{suffix}{stamp}")
    else:
        lines.append("- No macro/news items available in local artifacts.")

    gate = bundle.macro_gate or {}
    if gate:
        lines.extend(["", "## Macro Gate (10Y master switch)", ""])
        lines.append(f"- State: **{gate.get('state', 'unknown')}**")
        for metric in ("dgs10", "hy_oas"):
            if gate.get(metric) is not None:
                lines.append(
                    f"- {metric}: {gate[metric]} (observed {gate.get(metric + '_date')})"
                )
        for reason in gate.get("reasons", []):
            lines.append(f"- {reason}")
    if bundle.events:
        lines.extend(["", "## Upcoming Events (next 45 days)", ""])
        for item in bundle.events[:15]:
            tickers = (
                f" [{', '.join(item['tickers'])}]" if item.get("tickers") else ""
            )
            lines.append(
                f"- {item.get('date')} (+{item.get('days_away')}d) "
                f"{item.get('name')}{tickers} [{item.get('precision', 'unverified')}]"
            )

    leadership = bundle.leadership or {}
    if leadership:
        lines.extend(["", "## Leadership Rotation (hardware cycle switch)", ""])
        lines.append(f"- State: **{leadership.get('state', 'unknown')}**")
        if leadership.get("benchmark_return_20d_pct") is not None:
            lines.append(
                f"- Benchmark {leadership.get('benchmark')} 20d: "
                f"{leadership.get('benchmark_return_20d_pct')}%"
            )
        for ticker, info in (leadership.get("leaders") or {}).items():
            lines.append(
                f"- {ticker}: 20d {info.get('return_20d_pct')}% "
                f"(vs benchmark {info.get('vs_benchmark_pct')}%)"
                + (" — weakening" if info.get("weakening") else "")
            )
        state = leadership.get("state", "unknown")
        descriptions = {
            "intact": "所列半导体龙头尚未触发本规则的转弱条件。",
            "weakening": "部分所列半导体龙头已转弱，需观察是否扩散。",
            "broken": "所列半导体龙头中转弱范围较广，半导体仓位应优先检查失效位。",
            "unknown": "半导体龙头数据不足，暂不作轮动判断。",
        }
        lines.append("- 解读：" + descriptions.get(state, descriptions["unknown"]))
        lines.append("- 范围：这项指标只覆盖所列半导体龙头，不代表全市场风险偏好。")
    total_fundamentals = len(bundle.watchlist_table)
    known_fundamentals = sum(
        info.get("score") is not None for info in bundle.fundamentals.values()
    )
    lines.extend(["", "## Cash-Flow Data Coverage", ""])
    lines.append(
        f"Scored statement snapshots: {known_fundamentals}/{total_fundamentals}; "
        "unavailable or incomplete statements carry no score."
    )
    missing_statements = [
        row["ticker"] for row in bundle.watchlist_table
        if bundle.fundamentals.get(row["ticker"], {}).get("score") is None
    ]
    if missing_statements:
        lines.append("Unavailable/incomplete: " + ", ".join(missing_statements) + ".")
    scored = [
        (ticker, info)
        for ticker, info in (bundle.fundamentals or {}).items()
        if info.get("score") is not None
    ]
    if scored:
        lines.extend(["", "## Cash-Flow Quality Screen", ""])
        for ticker, info in sorted(scored, key=lambda pair: pair[1]["score"]):
            flags = ", ".join(info.get("flags", [])) or "no flags"
            lines.append(
                f"- {ticker}: **{info.get('label')}** (score {info.get('score')}) — {flags}"
            )
    if bundle.ipo_watch:
        lines.extend(["", "## IPO Watch", ""])
        for ticker, info in bundle.ipo_watch.items():
            if info.get("verified") is False:
                lines.append(
                    f"- {ticker}: unverified configuration; "
                    "IPO prices, supply dates and trading signals withheld."
                )
                continue
            pop = info.get("vs_ipo_price_pct")
            pop_text = f"{pop}%" if pop is not None else "unavailable"
            lines.append(
                f"- {ticker}: IPO {info.get('ipo_date')} at {info.get('ipo_price')}; "
                f"day {info.get('days_since_ipo')} ({info.get('stage')}), "
                f"vs IPO price {pop_text}; "
                f"early release {info.get('early_release_date')}, "
                f"lockup expiry {info.get('lockup_expiry_date')}"
            )

    lines.extend(["", "## Per-Ticker Notes", ""])
    for ticker, notes in bundle.per_ticker_notes.items():
        lines.append(f"### {ticker}")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(["## Setup And Data Quality Notes", ""])
    if bundle.risk_flags:
        lines.extend([
            "| Area | What it means | How to fix |",
            "| --- | --- | --- |",
        ])
        seen_notes = set()
        for warning in bundle.risk_flags:
            note = _warning_explanation(warning)
            key = (note["area"], note["meaning"], note["fix"])
            if key in seen_notes:
                continue
            seen_notes.add(key)
            lines.append(f"| {note['area']} | {note['meaning']} | {note['fix']} |")
    else:
        lines.append("All configured data sources completed without setup warnings.")
    lines.append("")
    lines.extend(_render_run_audit(bundle))
    return "\n".join(lines)


def _macro_action(plan: Dict[str, Any], blocked: bool) -> List[str]:
    lines = ["## 操作计划", ""]
    if blocked or not plan or plan.get("status") == "unavailable":
        return lines + ["**当前不新增买入。** 行情或有效价位尚未齐备，先补齐后再设操作价。", ""]
    budget = plan.get("budget", {})
    ticker = str(budget.get("ticker") or "QQQ")
    lines.extend([
        f"**现在新增买入 {_action_pct(plan.get('buy_now_pct'))}。** "
        f"{ticker}本轮最多使用**总账户 {_action_pct(budget.get('max_new_capital_pct'))}**，"
        "不是每只ETF各分配这一比例。", "",
        "| 回踩观察价 | 本批计划占总账户 | 确认后当前最多可用 |",
        "| ---: | ---: | ---: |",
    ])
    for entry in plan.get("entries", []):
        lines.append(
            f"| {_action_price(entry.get('price'))} | {_action_pct(entry.get('account_pct'))} | "
            f"{_action_pct(entry.get('currently_enabled_pct'))} |"
        )
    first = next(iter(plan.get("entries", [])), {})
    allowed = _event_number(first.get("currently_enabled_pct"))
    if allowed is not None and allowed > 0:
        entry_text = (f"未来回踩 {_action_price(first.get('price'))} 后，日收盘守住或收复，"
                      f"首批才最多投入总账户 {_action_pct(allowed)}。")
    else:
        entry_text = "当前没有启用买入预算，先等待利率环境和价格确认。"
    lines.extend(["", entry_text + "未启用的后续批次继续等待。", ""])
    exits = plan.get("exits", [])
    if exits:
        parts = [
            f"{_action_price(row.get('price'))}卖出{_action_pct(row.get('trim_original_dip_pct'))}"
            if row.get("price") is not None else "后续价位待确认" for row in exits
        ]
        lines.extend([
            "实际建仓后，分批止盈：**" + "；".join(parts) + "**。"
            "比例均按本轮实际建立的**原始抄底仓**计算，不按剩余仓位，也不涉及原有核心仓。"
            "首次止盈后取消尚未成交的加仓。", "",
        ])
    stop = plan.get("stop") or {}
    if stop.get("price") is not None:
        lines.extend([
            f"**日收盘跌破 {_action_price(stop['price'])}："
            "取消剩余加仓，复核退出本轮抄底仓。**", "",
        ])
    lines.extend(["以上是新增资金预算，未接入账户持仓；默认不新增杠杆。", ""])
    return lines


def _main_event_rows(bundle: ReportBundle) -> List[Dict[str, Any]]:
    selected = []
    groups = [
        [row for row in bundle.prediction_markets
         if "fed" in str(row.get("question", "")).lower()],
        [row for row in bundle.prediction_markets
         if "iran" in str(row.get("question", "")).lower()],
    ]
    for rows in groups:
        if rows:
            rows.sort(key=lambda row: (
                _event_quote(row, bundle.date)[0] is None, str(row.get("end_date") or "9999")
            ))
            selected.append(rows[0])
    return selected


def _macro_chain(rows: List[Dict[str, Any]], report_date: str) -> List[str]:
    verified = {row["reference_for"]: row for row in rows if _chain_usable(row, report_date)}
    if not verified:
        return ["## 链上参考", "", "SPY、QQQ的链上映射尚未核验或报价不可用，当前记为未知。", ""]
    lines = ["## 链上参考", "", "| ETF | 永续参考价 | 24h变化 | 抓取时间 |",
             "| --- | ---: | ---: | --- |"]
    for ticker in ("SPY", "QQQ"):
        row = verified.get(ticker)
        if row is None:
            lines.append(f"| {ticker} | 未知 | 未知 | 未知 |")
            continue
        stamp = _event_time(row.get("fetched_at"))
        change = _event_number(row.get("day_change_pct"))
        label = f"{ticker}（{row.get('source') or '链上'}）"
        lines.append("| " + " | ".join(_table_text(value) for value in (
            label, f"{float(row['mark_price']):,.2f}",
            f"{change:+.2f}%" if change is not None else "未知",
            stamp.astimezone(ZoneInfo("America/Toronto")).strftime("%m-%d %H:%M %Z"),
        )) + " |")
    return lines + ["", "链上是衍生品代理；抓取时间不是交易所报价时间，报价年龄未知。"
                    "它不替代ETF收盘价，也不自动改变本轮仓位计划。", ""]


def render_markdown(bundle: ReportBundle, detailed: bool = True) -> str:
    """Render the comprehensive report by default; compact macro view is explicit."""
    if detailed:
        return render_detailed_markdown(bundle)
    brief = bundle.daily_brief
    pending = brief.get("price_data_pending", [])
    lines = [f"# 宏观交易简报 — {bundle.date}", ""]
    if pending:
        lines.extend(["**最新交易日行情未齐，暂停抄底/加仓/止盈价建议。**", ""])
    lines.extend([str(brief.get("headline") or "等待完整行情后再判断。"), ""])
    rows = {row["ticker"]: row for row in bundle.watchlist_table if row["ticker"] in {"SPY", "QQQ"}}
    lines.extend(["| ETF | 正式收盘 | 交易日 |", "| --- | ---: | --- |"] )
    for ticker in ("SPY", "QQQ"):
        row = rows.get(ticker, {})
        current = row.get("data_source") == "yfinance" and not row.get("stale", True)
        price = _action_price(row.get("close")) if current else "待更新"
        lines.append(f"| {ticker} | {price} | {row.get('data_date') or '未知'} |")
    states = {"unknown": "利率与信用数据未齐，先保持谨慎，后续批次暂不启用。",
              "risk_off": "利率与信用环境偏防守，暂停新增买入。",
              "caution": "利率与信用环境仍需谨慎，控制新增预算。",
              "risk_on": "当前利率与信用数据未触发本计划的防守条件。"}
    lines.extend(["", states.get(bundle.macro_gate.get("state"), states["unknown"]), ""])
    lines.extend(_macro_action(bundle.action_plan, bool(pending)))
    lines.extend(_render_directional_context(bundle))
    lines.extend(_render_options_reasons(bundle))
    lines.extend(_render_cloud_capex(bundle))
    lines.extend(["## 关键宏观事件", ""])
    events = _main_event_rows(bundle)
    if not events:
        lines.extend(["美联储与美伊事件报价暂无可用快照，先看正式消息。", ""])
    for row in events:
        probability, _ = _event_quote(
            row, bundle.date, bundle.scenario_context.get(
                "event_threshold", DEFAULT_EVENT_MAIN_CASE_THRESHOLD
            ),
        )
        number = f"{probability:.1%}" if probability is not None else "未知"
        end, updated = _event_time(row.get("end_date")), _event_time(row.get("market_updated_at"))
        deadline = end.astimezone(ZoneInfo("America/Toronto")).strftime("%m-%d") if end else "未知"
        stamp = (updated.astimezone(ZoneInfo("America/Toronto")).strftime("%m-%d %H:%M %Z")
                 if updated else "未知")
        label = str(row.get("label") or "事件预期")
        url = str(row.get("url") or "")
        if url.startswith("https://polymarket.com/"):
            label = f"[{label}]({url})"
        lines.extend([f"- {label}：Yes **{number}**；合约期限{deadline}，更新{stamp}。"] )
    if events:
        lines.extend(["", "这些是事件市场的报价，不是股指上涨概率。"
                      "不同期限分开看，不据此自动加仓。", ""])
    lines.extend(_macro_chain(bundle.cross_market, bundle.date))
    lines.extend([f"[查看完整报告]({bundle.date}-report.html)", ""])
    lines.extend(_render_run_audit(bundle))
    return "\n".join(lines)


def _brief_text(value: Any, limit: int = 80) -> str:
    text = _table_text(value)
    return text if len(text) <= limit else text[:limit-1] + "…"


def _portfolio_reason(bundle: ReportBundle) -> str:
    plan = bundle.portfolio_plan
    if not isinstance(plan, dict) or plan.get("status") not in {"available", "reference_only"}:
        return (str(plan.get("reason") or "模型计划尚不可用")
                if isinstance(plan, dict) else "模型格式无效")
    if plan.get("report_policy_proof_error") != "":
        return str(plan.get("report_policy_proof_error") or "缺少当前规则/回放核验记录")
    clock = _report_clock(bundle.date)
    available = _event_time(plan.get("available_at"))
    if (plan.get("date") != bundle.date or available is None or available > clock
            or plan.get("price_date") != expected_complete_session(bundle.date)):
        return "模型日期、信息时点或行情日期不匹配"
    if (plan.get("news_only") is not True or plan.get("prediction_markets_trigger") is not False
            or plan.get("orders_executed") is not False
            or plan.get("paper_holdings_only") is not True
            or not plan.get("source_fingerprints")
            or not re.fullmatch(r"[a-f0-9]{64}", str(plan.get("policy_sha256") or ""))):
        return "模型规则、新闻来源或参考仓位身份未核验"
    next_open = _event_time(plan.get("next_open_at"))
    if plan["status"] == "available" and (next_open is None or next_open <= clock):
        return "下一开盘执行窗口已过，不能补记成交"
    for ticker in ("QQQ", "SPY", "TQQQ"):
        actual = plan.get("report_price_inputs", {}).get(ticker, {})
        close, observed = (_event_number(value) for value in (
            plan.get("price_closes", {}).get(ticker), actual.get("close")))
        fetched = _event_time(actual.get("fetched_at"))
        if (actual.get("date") != plan["price_date"] or actual.get("source") != "yfinance"
                or actual.get("session_complete") is not True
                or plan.get("price_dates", {}).get(ticker) != plan["price_date"]
                or close is None or observed is None or min(close, observed) <= 0
                or abs(close-observed) > max(.011, observed*.00001)
                or actual.get("fetched_at") is not None
                and (fetched is None or fetched > available)):
            return ticker + "模型价格与当前完整行情未对齐"
    for key in ("target_weights", "paper_weights"):
        weights = plan.get(key)
        if not isinstance(weights, dict) or set(weights) != {"QQQ", "TQQQ", "cash"}:
            return "模型缺少统一账户资金表"
        numbers = [_event_number(v) for v in weights.values()]
        if any(v is None or not 0 <= v <= 1 for v in numbers) or abs(sum(numbers)-1) > 1e-6:
            return "模型配置超出单一账户100%资金边界"
    increment = _event_number(plan.get("new_tqqq_increment"))
    upper = max(0., float(plan["target_weights"]["TQQQ"]) - float(plan["paper_weights"]["TQQQ"]))
    if increment is None or not 0 <= increment <= upper + 1e-6:
        return "新增TQQQ参考量与模型仓位差额不一致"
    gain = _event_number(plan.get("profit_target_gain"))
    if (type(plan.get("time_limit_sessions")) is not int or plan["time_limit_sessions"] <= 0
            or gain is None or gain <= 0):
        return "模型退出条件未完整记录"
    return ""


def _portfolio_title(plan: Dict[str, Any]) -> str:
    return {"cycle_hold": "模型保持当前参考组合，等待新的价格/新闻条件。",
            "cycle_initial_qqq": "模型以QQQ为核心参考，TQQQ等待新的底部确认。",
            "cycle_structural_stop": "QQQ结构失效，模型转向现金防守。",
            "cycle_long_trend_exit": "长期趋势失效，模型先退出风险仓位。",
            "cycle_local_top": "顶部条件出现，模型降低风险敞口。",
            "cycle_adverse_event": "已观察新闻风险触发降仓参考。",
            "cycle_tqqq_gain_to_qqq": "TQQQ达到获利条件，模型转回QQQ。",
            "cycle_tqqq_time_to_qqq": "TQQQ达到持有期限，模型转回QQQ。",
            "cycle_confirmed_bottom": "新的底部已完成模型确认，可复核TQQQ阶段。",
            "cycle_bull_qqq_restore": "价格恢复强势，模型恢复QQQ参考。",
            "cycle_event_probe": "已观察新闻与价格条件支持有限TQQQ试探。"}.get(
                plan.get("cause"), "按已核验的新闻与价格周期模型参考组合观察。")


def _render_dynamic_levels(plan: Dict[str, Any]) -> List[str]:
    policy, features = plan["validated_display_policy"], plan["qqq_features"]
    parameters, cycle = policy["parameters"], policy["cycle_parameters"]
    close = float(plan["price_closes"]["QQQ"])
    age, anchor_date = features.get("anchor_age"), features.get("anchor_date")
    try:
        anchor_valid = (type(age) is int and age >= 0
                        and date.fromisoformat(anchor_date).isoformat() == anchor_date
                        and anchor_date <= plan["price_date"])
    except (TypeError, ValueError):
        anchor_valid = False
    rows = [("QQQ", "锚定支撑低点", features.get("anchor_low") if anchor_valid else None,
             f"{anchor_date}；形成{age}交易日" if anchor_valid else "锚点日期或年龄未核验")]
    for label, key in ((f"底部确认{cycle['bottom_trend_sessions']}日线", "cycle_bottom_sma"),
                       (f"前{policy['prior_high_sessions']}日高点", "prior3_high")):
        level = _event_number(features.get(key))
        state = ("已高于" if close > level else "尚未高于") if level and level > 0 else "未知"
        rows.append(("QQQ", label, level, f"日收盘{state}；不是未来买入挂单价"))
    rows.extend([
        ("QQQ", f"此前{parameters['top_breakout_sessions']}日低点", features.get("prior_low"),
         "与本轮涨幅等顶部条件共同判断，不单独触发卖出"),
        ("QQQ", f"长期{parameters['trend_sessions']}日线", features.get("slow_sma"),
         "长期趋势退出参考，需结合风控启用状态与连续收盘条件"),
    ])
    spy = plan.get("validated_observation_levels", {}).get("SPY", {})
    for key, label in (("low", "低点"), ("high", "高点")):
        rows.append(("SPY", "近20交易日" + label,
                     spy.get(key) if spy.get("status") == "available" else None,
                     (f"{spy['start_date']}至{spy['end_date']}；仅宏观观察，无新增SPY规则"
                      if spy.get("status") == "available" else "缺少已核验的完整20日原始OHLC")))
    lines = ["### 动态关键价位（策略原始价格）", "",
             f"价格截至{plan['price_date']}；QQQ策略价{close:.2f}。", "",
             "| 标的 | 参考位 | 数值 | 当前状态与用途 |", "| --- | --- | ---: | --- |"]
    for ticker, label, value, purpose in rows:
        number = _event_number(value)
        lines.append(f"| {ticker} | {label} | "
                     f"{f'{number:.2f}' if number is not None and number > 0 else '未知'} | "
                     f"{_table_text(purpose)} |")
    missing = []
    if anchor_valid and age < cycle["bottom_min_age"]:
        missing.append(f"低点年龄{age}日不足{cycle['bottom_min_age']}日")
    drawdown = _event_number(features.get("drawdown"))
    if drawdown is not None and drawdown < parameters["drawdown_threshold"]:
        missing.append(f"低点回撤{drawdown:.2%}不足{parameters['drawdown_threshold']:.0%}")
    if features.get("cycle_bottom_ready") is False and missing:
        lines.extend(["", "**底部尚未完整确认：** " + "；".join(missing) + "。"])
    return lines + ["", "以上不做分红复权，与通用技术均线可能不同。"
                    "触价不是买入；须完整底部确认。", ""]


def _render_portfolio_reference(bundle: ReportBundle) -> List[str]:
    reason = _portfolio_reason(bundle)
    if reason:
        return ["## 当前模型组合", "", "**本次模型计划不可用，暂停据此新增。** "
                + _brief_text(reason, 120), ""]
    plan = bundle.portfolio_plan
    lines = ["## 当前模型组合", "", f"**{_portfolio_title(plan)}**", "",
             "| 标的 | 单一账户目标参考 |", "| --- | ---: |"]
    for ticker, label in (("QQQ", "QQQ"), ("TQQQ", "TQQQ"), ("cash", "现金")):
        lines.append(f"| {label} | {float(plan['target_weights'][ticker]):.1%} |")
    lines.extend(["", "总资金100%，不是每只标的各用100%。这是模型目标，不是今天的买入金额；"
                  "未接入你的账户，纸面仓位不代表你的持仓，也没有执行订单。", ""])
    if plan["status"] == "reference_only":
        lines.extend(["下一开盘窗口已经开始：只保留收盘模型参考，不提供补记过去开盘的指令。", ""])
    increment = _event_number(plan.get("new_tqqq_increment")) or 0.
    if _expired_today(bundle):
        lines.extend(["**当日FOMC结果待核验：当前暂停新开或新增TQQQ。** "
                      "这是当前交付提醒，不改写上表模型参考，也不虚构已有仓位清仓。", ""])
    elif plan["status"] == "available" and increment > 0:
        lines.extend([f"模型待确认的新TQQQ增量参考：总资金{increment:.1%}；"
                      "仅供下一实际开盘前复核，须先核对真实持仓差额。", ""])
    else:
        lines.extend(["当前没有可据此直接执行的新TQQQ增量；持有与新开仓分开判断。", ""])
    lines.extend(_render_dynamic_levels(plan))
    gain = _event_number(plan.get("profit_target_gain"))
    gain_text = f"{gain:.0%}" if gain is not None and gain > 0 else "配置目标"
    target = _event_number(plan.get("profit_target_price"))
    profit = f"模型成本浮盈达{gain_text}后转回QQQ"
    if target is not None and target > 0:
        profit += f"；纸面成本对应参考价{target:.2f}，个人成本未知"
    stop = _event_number(plan.get("structural_stop"))
    policy = plan["validated_display_policy"]
    parameters, cycle = policy["parameters"], policy["cycle_parameters"]
    features = plan.get("qqq_features", {})
    slow = _event_number(features.get("slow_sma"))
    exit_rule = (f"长期风控启用时：连续{policy['trend_exit_sessions']}日收盘低于各自"
                 f"{parameters['trend_sessions']}日线"
                 + (f"（当前{slow:.2f}）" if slow is not None else ""))
    exit_rule += ("，当前已启用" if policy["risk_armed"] else
                  "，当前未启用，须先日收盘达到/高于长期线后启用")
    if stop is not None and stop > 0:
        exit_rule += (f"；独立恢复结构位{stop:.2f}：日收盘跌破即触发，"
                      "不受上述启用状态限制")
    exit_rule += "；任一已生效条件满足，下一实际开盘复核退出"
    ready = features.get("cycle_bottom_ready")
    bottom = (f"支撑低点形成满{cycle['bottom_min_age']}交易日，低点抬高且收盘回升；"
              f"收盘站上{cycle['bottom_trend_sessions']}日线及前{policy['prior_high_sessions']}日高点；"
              f"该低点较前{policy['drawdown_lookback']}日最高收盘回撤≥"
              f"{parameters['drawdown_threshold']:.0%}。"
              f"当前{'已满足' if ready is True else '未满足'}")
    top = (f"持有策略仓且本轮尚未顶部减仓时，QQQ本轮参考价至最高收盘涨幅达"
           f"{parameters['top_profit_atr']:g}ATR，且日收盘跌破此前"
           f"{parameters['top_breakout_sessions']}日最低价：下一实际开盘降为QQQ最多"
           f"{cycle['top_retained']:.0%}（不超过当前总风险敞口），其余现金，TQQQ转出")
    lines.extend(["| 条件 | 动作 |", "| --- | --- |", f"| 新底部 | {bottom} |",
                  f"| 确认后阶段 | TQQQ模型目标{parameters['recovery_weight']:.0%}，"
                  "替代相应QQQ资金，不增加账户总资金 |",
                  f"| QQQ阶段顶部 | {top} |",
                  f"| TQQQ获利 | {profit} |",
                  f"| TQQQ时间 | 最长{plan.get('time_limit_sessions')}个交易日；"
                  "未先触发风险退出时转回QQQ |",
                  f"| QQQ失效 | {exit_rule} |"])
    turning = policy.get("turning_parameters")
    if turning:
        lines.extend([
            f"| 市场隐含压力顶部 | VIX/VIX3M≥{turning['stress_ratio']:g}、2年美债20日升"
            f"≥{turning['fed_shock_pp']:g}个百分点（有预测市场美联储赔率时以其变化替代）或"
            f"初请4周均值较52周低点升≥{turning['claims_rise']:.0%}"
            + ("且高于26周均值" if turning.get("claims_trend_filter") else "")
            + "，且QQQ收盘低于"
            f"{turning['stress_confirm_sma']}日线：降为QQQ最多{turning['stress_retained']:.0%}，"
            "TQQQ转出 |",
            f"| 波动率投降底部 | 近{turning['capitulation_lookback']}日VIX/VIX3M曾≥"
            f"{turning['capitulation_ratio']:g}，现回落至≤{turning['normalized_ratio']:g}，回撤达门槛且"
            f"低点抬高、突破前3日高点：TQQQ{turning['capitulation_weight']:.0%}，其余QQQ"
            + (f"；入场后{turning['capitulation_grace_sessions']}日内只要收盘守住锚定低点，"
               "VIX再倒挂不单独触发减仓" if turning.get("capitulation_grace_sessions") else "")
            + " |"])
    lines.extend(["", "结构退出/风险降仓优先于获利转回；TQQQ为杠杆ETF，阶段风险高于QQQ。"
                  + (("市场隐含层与经交叉印证的新闻可触发上述规则；"
                       "单一来源、传闻和威胁性表述只作背景。"
                      if (plan.get("classified_news_receipt") or {}).get("role") == "trigger"
                      else "市场隐含层可触发上述规则；经交叉印证的一般新闻只作背景，不触发交易。")
                     if turning else
                     "本计划使用已观察新闻，预测市场赔率仅作旁证，不额外触发周期交易。"), ""])
    signal = plan.get("turning_signal") or {}
    evidence = signal.get("evidence") or {}
    if turning and signal.get("status") == "available":
        ratio = _event_number(evidence.get("vix_ratio"))
        reasons = "、".join({"vix_term_inversion": "VIX期限倒挂",
                            "hawkish_repricing_2y": "2年美债鹰派重定价",
                            "hawkish_repricing_prediction_market": "预测市场鹰派重定价",
                            "jobless_claims_rise": "初请上升",
                            "credit_drawdown": "信用回撤"}.get(r, r)
                           for r in signal.get("stress_reasons") or []) or "无"
        state = ("压力已由价格确认" if signal.get("stress") else
                 "投降后回落已出现" if signal.get("capitulation") else "未触发")
        lines.extend([f"市场隐含层：VIX/VIX3M {ratio:.2f}；压力来源：{reasons}；状态：{state}。"
                      if ratio is not None else f"市场隐含层：压力来源：{reasons}；状态：{state}。",
                      ""])
    reference = plan.get("reference_source") or {}
    drawdown, base = (_event_number(reference.get(key)) for key in (
        "recent_max_drawdown_pct", "QQQ_recent_max_drawdown_pct"))
    if drawdown is not None and base is not None:
        lines.extend([f"风险对照（调整价代理回测口径）：该参考区间模型最大回撤{drawdown:.2f}%，"
                      f"同期QQQ为{base:.2f}%；模型回撤更大。", ""])
    return lines


def _expired_today(bundle: ReportBundle) -> bool:
    clock = _report_clock(bundle.date)
    local = clock.astimezone(ZoneInfo("America/New_York"))
    if (local.date().isoformat() == bundle.date and (local.hour, local.minute) >= (16, 15)
            and any(row.get("date") == bundle.date and row.get("kind") == "fomc"
                    and row.get("precision") in {"scheduled", "verified", "confirmed"}
                    and not row.get("resolution_verified") for row in bundle.events)):
        return True
    return any(end and end <= clock and end.astimezone(ZoneInfo("America/Toronto"))
               .date().isoformat() == bundle.date and not row.get("resolution_verified")
               and re.search(r"\bFed\b|FOMC|美联储", str(row.get("label") or "") + " "
                             + str(row.get("question") or ""), re.I)
               for row in bundle.prediction_markets
               for end in [_event_time(row.get("end_date"))])


def _render_primary_causes(bundle: ReportBundle) -> List[str]:
    """At most five material, sourced factors; event prices retain event semantics."""
    rows = []
    features = bundle.scenario_context.get("macro_features", {})
    for key, label, channel in (("yield10", "10年美债", "融资成本与成长股估值"),
                                 ("credit", "高收益债利差", "信用与风险承受能力")):
        item = features.get(key) or {}
        value, change = _event_number(item.get("value")), _event_number(item.get("change_5"))
        if value is not None:
            observation = f"{label} {value:.2f}%（{item.get('date')}）"
            if change is not None:
                observation += f"；5观察间隔{change:+.2f}个百分点"
            action = ("利率/利差反向上升时先复核未启用批次，执行上方价格条件"
                      if change is None or change > 0 else "压力有所缓和，仍等收复/回踩确认")
            rows.append((observation, channel, action))
    source_by_id = {str(row.get("id")): row for row in bundle.scenario_context.get("events", [])}
    channel_names = {
        "us_rates": "政策→美债收益率→估值", "us_growth_inflation": "增长/通胀→利率与盈利",
        "us_fiscal_trade": "财政/贸易→成本与风险溢价", "oil_geopolitics": "冲突→油价/信用",
        "oil_inflation": "能源→通胀/利润率", "chip_supply_earnings": "供应/业绩→半导体",
        "mega_cap_earnings_policy": "权重股盈利/监管→指数",
        "us_policy_background": "政策背景→价格验证", "regional_funding_fx": "融资/汇率→风险资产",
        "regional_memory_chain": "韩元/存储链→半导体"}
    def event_priority(row):
        priority = _event_number(source_by_id.get(str(row.get("id")), row).get("importance_rank"))
        return (priority if priority is not None else 9,
                str(row.get("end_date") or "9999"), str(row.get("id")))

    events = sorted(bundle.prediction_markets, key=event_priority)
    seen = set()
    for event in events:
        if len(rows) >= 5:
            break
        known = source_by_id.get(str(event.get("id")), event)
        identity = str(event.get("id"))
        if identity in seen:
            continue
        probability, _ = _event_quote(event, bundle.date)
        end = _event_time(event.get("end_date"))
        elapsed = bool(end and end <= _report_clock(bundle.date)
                       and end.astimezone(ZoneInfo("America/Toronto")).date().isoformat()
                       == bundle.date)
        if probability is None and not elapsed:
            continue
        seen.add(identity)
        question = str(event.get("question") or known.get("question") or "").strip()
        exact_question = bool(question) and len(question) <= 140
        label = question if exact_question else "复杂合约：完整条件见来源"
        url = str(event.get("url") or "")
        if url.startswith("https://polymarket.com/"):
            label = f"[{label}]({url})"
        no_probability = _event_number(event.get("no_probability"))
        no_text = f"{no_probability:.1%}" if no_probability is not None else "未知"
        observation = label + ("：已到合约期限，结果待核验" if elapsed else
                               f"：Yes {probability:.1%} / No {no_text}" if exact_question else
                               "：主文不简写该合约赔率")
        channel = known.get("impact_channel") or event.get("topic")
        transmission = channel_names.get(channel, "经收益率、盈利或风险溢价传导，仍需价格验证")
        action = ("等待正式结果及市场反应，不按过期报价加仓" if elapsed else
                  "主情景改变则复核条件单；事件概率不直接等于股指上涨概率")
        rows.append((observation, transmission, action))
    lines = ["## 现在这样做的依据", "", "| 当前观察 | 传导 | 操作影响 |", "| --- | --- | --- |"]
    if not rows:
        rows.append(("有效宏观/事件观察不足", "未验证", "按价格条件等待，不推测事件结果"))
    for row in rows:
        lines.append("| " + " | ".join(_table_text(value) for value in row) + " |")
    return lines + [""]


def _render_primary_references(bundle: ReportBundle) -> List[str]:
    lines = ["## 24小时市场参考", "", "| 标的 / 类型 | 永续参考价 | 24h变化 | 来源 / 抓取时间 |",
             "| --- | ---: | ---: | --- |"]
    for ticker in ("BTC", "ETH", "SPY", "QQQ"):
        candidates = [row for row in bundle.cross_market
                      if (row.get("reference_for") or row.get("symbol")) == ticker]
        usable = next((row for row in candidates if (
            _crypto_reference(row, bundle.date) if ticker in {"BTC", "ETH"}
            else _chain_usable(row, bundle.date))), None)
        kind = "加密永续" if ticker in {"BTC", "ETH"} else "ETF永续代理"
        if usable is None:
            lines.append(f"| {ticker} / {kind} | 未知 | 未知 | 无有效快照 |")
            continue
        change = _event_number(usable.get("day_change_pct"))
        stamp = _event_time(usable.get("fetched_at"))
        when = stamp.astimezone(ZoneInfo("America/Toronto")).strftime("%m-%d %H:%M %Z")
        lines.append(f"| {ticker} / {kind} | ${float(usable['mark_price']):,.2f} | "
                     f"{f'{change:+.2f}%' if change is not None else '未知'} | "
                     f"{_table_text(usable.get('source'))} / {when} |")
    return lines + ["", "抓取时间不是交易所报价时间，报价年龄未知；永续参考不替代ETF正式收盘。", ""]


def render_daily_markdown(bundle: ReportBundle) -> str:
    """Compact primary decisions; complete observations remain in the linked details."""
    plan = bundle.action_plan
    portfolio_mode = bool(bundle.portfolio_plan)
    if portfolio_mode:
        plan = {}
    if bundle.daily_brief.get("price_data_pending"):
        plan = {"buy_now_pct": 0, "current_action": "完整行情或有效价位未齐，暂不新增买入。"}
    elif plan.get("status") == "unavailable":
        plan = {"buy_now_pct": 0, "current_action": plan.get("current_action")
                or "有效价位或盈亏空间不足，暂不新增买入。"}
    headline = (bundle.scenario_context.get("headline") or bundle.daily_brief.get("headline")
                or "等待数据")
    action = str(plan.get("current_action") or "行情或价位未齐，暂不新增买入。")
    if _expired_today(bundle):
        headline = str(headline).replace("关键事件临近", "关键事件结果待核验")
        action = action.replace("关键事件临近", "关键事件结果待核验")
    lines = [f"# Trading Wizzard 日报 — {bundle.date}", "", "## 今日判断与操作", "",
             f"**{_brief_text(headline, 160)}**", "",
             f"**现在新增买入：{_action_pct(plan.get('buy_now_pct', 0))}。** "
             + _brief_text(action, 160), ""]
    if portfolio_mode:
        lines = [f"# Trading Wizzard 日报 — {bundle.date}", ""]
        lines.extend(_render_portfolio_reference(bundle))
    budget = plan.get("budget", {})
    if budget:
        lines.extend([f"本轮标的 {plan.get('ticker') or budget.get('ticker') or 'QQQ'}；"
                      f"账户资金上限 {_action_pct(budget.get('max_new_capital_pct'))}，"
                      f"当前确认后可启用 {_action_pct(budget.get('currently_enabled_pct'))}。", ""])
    entries, exits = plan.get("entries", []), plan.get("exits", [])
    if entries:
        lines.extend(["| 买入批次 | 回踩观察价 | 占总账户计划 | 条件满足后可启用 |",
                      "| --- | ---: | ---: | ---: |"])
        for row in entries:
            lines.append(f"| {row.get('batch')} | {_action_price(row.get('price'))} | "
                         f"{_action_pct(row.get('account_pct'))} | "
                         f"{_action_pct(row.get('currently_enabled_pct'))} |")
        lines.extend(["", "须日收盘收复/守住并完成回踩确认；"
                      "观察价不是挂单，后续批次逐次复核。", ""])
    if exits:
        lines.extend(["| 止盈批次 | 观察价 | 卖出本轮初始份额 |", "| --- | ---: | ---: |"])
        for row in exits:
            lines.append(f"| {row.get('stage')} | {_action_price(row.get('price'))} | "
                         f"{_action_pct(row.get('trim_original_dip_pct'))} |")
        lines.extend(["", "仅对本轮实际建立的仓位适用；首次止盈后取消未成交加仓。", ""])
    stop = plan.get("stop", {})
    if stop.get("price") is not None:
        lines.extend([f"**失效：日收盘跌破 {_action_price(stop['price'])}，"
                      "取消剩余加仓并复核退出本轮仓位。**", ""])
    lines.extend(_render_directional_context(bundle, compact=True))
    lines.extend(_render_primary_causes(bundle))
    lines.extend(_render_options_reasons(bundle))
    lines.extend(_render_cloud_capex(bundle))
    lines.extend(["## 板块相对强弱", "",
                  "| 板块 | 近20交易日涨跌 | 同期相对SPY | 截止日 |",
                  "| --- | ---: | ---: | --- |"])
    rotation = bundle.scenario_context.get("rotation", [])
    valid_rotation = [row for row in rotation
                      if row.get("end_date") == expected_complete_session(bundle.date)
                      and _event_number(row.get("return_20d_pct")) is not None
                      and _event_number(row.get("vs_spy_pp")) is not None]
    for row in valid_rotation:
        label = _table_text(row.get("name") or row.get("ticker"))
        lines.append(f"| {label}（{row.get('ticker')}） | "
                     f"{float(row['return_20d_pct']):+.2f}% | "
                     f"{float(row['vs_spy_pp']):+.2f}个百分点 | "
                     f"{row['end_date']} |")
    if not valid_rotation:
        lines.append("| 同区间数据不足 | 未知 | 未知 | 待更新 |")
    lines.extend(["", "## 全部观察标的", "",
                  "通用聚类按当前行情日线口径只读重算，可能含分红复权；"
                  "仅观察，不替代上方策略原始价触发位。", "",
                  "| 标的 | 价格 | 交易日 | 方向 | 支撑参考 | 阻力参考 |",
                  "| --- | ---: | --- | --- | --- | --- |"])
    decisions = {row.get("ticker"): row for row in bundle.daily_brief.get("ticker_decisions", [])}
    for row in bundle.watchlist_table:
        current = (row.get("data_source") == "yfinance" and not row.get("stale", True)
                   and row.get("session_complete") is True)
        view = decisions.get(row["ticker"], {}).get("view") or row.get("regime", "未知")
        lines.append("| " + " | ".join(_table_text(value) for value in (
            row["ticker"], _action_price(row.get("close")) if current else "待更新",
            row.get("data_date"), view if current else "待更新",
            _observation_prices(row.get("observation_support_prices")) if current else "待更新",
            _observation_prices(row.get("observation_resistance_prices")) if current else "待更新"
        )) + " |")
    lines.append("")
    lines.extend(_render_primary_references(bundle))
    lines.extend(["## 风险与完整来源", "",
                  "日收盘条件不保证盘中或次日成交价，跳空可能越过退出价。"
                  "未接入账户，不假设已有仓位。", ""])
    if bundle.daily_brief.get("price_data_pending"):
        lines.extend(["**最新完整交易日行情未齐，当前操作价位暂停使用。**", ""])
    if bundle.risk_flags:
        lines.extend([f"另有{len(bundle.risk_flags)}项数据/来源提示，见完整明细。", ""])
    lines.extend([f"[完整数据与来源]({bundle.date}-details.html)", ""])
    lines.extend(_render_run_audit(bundle))
    return "\n".join(lines)


def render_html(markdown: str, title: str) -> str:
    """Render this report's Markdown subset without executing provider-supplied HTML."""
    def inline(value: str) -> str:
        escaped = html.escape(value)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        return re.sub(
            r"\[([^\]]+)\]\((https://[^\s)]+|\d{4}-\d{2}-\d{2}-(?:details|report)\.html|"
            r"\.\./data/runs/\d{4}-\d{2}-\d{2}/[A-Za-z0-9_-]+/manifest\.json)\)",
            r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', escaped,
        )

    output: List[str] = []
    in_table = False
    in_list = False
    for line in markdown.splitlines():
        is_table = line.startswith("|")
        is_list = line.startswith("- ")
        if in_table and not is_table:
            output.append("</tbody></table></div>")
            in_table = False
        if in_list and not is_list:
            output.append("</ul>")
            in_list = False
        if is_table:
            if re.fullmatch(r"[| :\-]+", line):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            tag = "td" if in_table else "th"
            rendered = "<tr>" + "".join(
                f"<{tag}>{inline(cell)}</{tag}>" for cell in cells
            ) + "</tr>"
            if not in_table:
                output.append(
                    '<div class="table-wrap"><table><thead>' + rendered + "</thead><tbody>"
                )
                in_table = True
            else:
                output.append(rendered)
        elif is_list:
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append(f"<li>{inline(line[2:])}</li>")
        elif line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 6)
            output.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
        elif line.strip():
            output.append(f"<p>{inline(line)}</p>")
    if in_table:
        output.append("</tbody></table></div>")
    if in_list:
        output.append("</ul>")
    return (
        '<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "max-width:1440px;margin:36px auto;padding:0 24px;color:#202a35;line-height:1.75}"
        "h1{font-size:28px}h2{margin-top:40px;border-bottom:1px solid #d9e0e6;padding-bottom:8px}"
        "h3{margin-top:28px}p{max-width:1100px}li{margin:5px 0}"
        ".table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;"
        "line-height:1.65}th,td{padding:10px 12px;text-align:left;vertical-align:top;"
        "border-bottom:1px solid #dfe5eb;min-width:70px}th{background:#eef3f7}"
        "tbody tr:nth-child(even){background:#fafbfd}code{background:#edf1f4;padding:2px 5px}"
        "strong{color:#152d46}@media print{body{margin:0;font-size:11px}h2{break-after:avoid}"
        "tr{break-inside:avoid}.table-wrap{overflow:visible}}</style></head><body><article>"
        + "\n".join(output) + "</article></body></html>\n"
    )


def write_report_files(paths: ProjectPaths, bundle: ReportBundle) -> Tuple[Path, Path]:
    markdown = render_daily_markdown(bundle)
    markdown_path = paths.reports / f"{bundle.date}-report.md"
    html_path = paths.reports / f"{bundle.date}-report.html"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(
        render_html(markdown, f"Trading Wizzard 综合日报 {bundle.date}"),
        encoding="utf-8",
    )
    details = render_detailed_markdown(bundle)
    (paths.reports / f"{bundle.date}-details.md").write_text(details, encoding="utf-8")
    (paths.reports / f"{bundle.date}-details.html").write_text(
        render_html(details, f"Trading Wizzard 数据明细 {bundle.date}"), encoding="utf-8"
    )
    return markdown_path, html_path
