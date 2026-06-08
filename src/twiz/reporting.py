import html
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .config import WatchedAsset
from .models import ReportBundle
from .storage import ProjectPaths, read_json, read_jsonl

IMMEDIATE_LEVEL_DISTANCE_PCT = 2.0
NEAR_LEVEL_DISTANCE_PCT = 6.0


def _latest_close(market_rows: List[Dict[str, Any]]) -> float:
    return float(market_rows[-1]["close"]) if market_rows else 0.0


def _macro_from_raw_items(paths: ProjectPaths, report_date: str) -> List[Dict[str, Any]]:
    rows = []
    for item in read_jsonl(paths.raw / f"items-{report_date}.jsonl"):
        source_type = str(item.get("source_type", ""))
        if source_type in {"fed_rss", "sec_rss", "macro_rss", "rss"}:
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
            f"{int(level.get('touches', 1))}x)"
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
            "meaning": "Interest-rate, inflation, labor, and credit series were skipped.",
            "fix": "Add FRED_API_KEY to .env.",
        }
    if (
        "SEC EDGAR identity" in warning
        or "SEC latest filings" in warning
        or "HTTP Error 403" in warning
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
    return {
        "area": "Data quality",
        "meaning": warning,
        "fix": "Review configuration or provider access.",
    }


def build_report_bundle(
    paths: ProjectPaths,
    report_date: str,
    watchlist: Iterable[WatchedAsset],
) -> ReportBundle:
    market_data = read_json(paths.raw / f"market-{report_date}.json", default={}) or {}
    sentiment_summary = (
        read_json(paths.processed / f"sentiment-summary-{report_date}.json", default={}) or {}
    )
    technical = read_json(paths.processed / f"technical-{report_date}.json", default={}) or {}
    backtests = read_json(paths.processed / f"backtests-{report_date}.json", default={}) or {}
    options = read_json(paths.processed / f"options-{report_date}.json", default={}) or {}
    macro = read_json(paths.processed / f"macro-{report_date}.json", default=[]) or []
    warnings = read_json(paths.processed / f"warnings-{report_date}.json", default=[]) or []

    watchlist_table = []
    per_ticker_notes: Dict[str, List[str]] = {}
    for asset in watchlist:
        ticker = asset.ticker
        tech = technical.get(ticker, {})
        sentiment = sentiment_summary.get(ticker, {})
        option_context = options.get(ticker, {})
        backtest = backtests.get(ticker, {})
        option_warnings = (
            option_context.get("warnings", []) if isinstance(option_context, dict) else []
        )
        if option_warnings:
            warnings.extend([f"{ticker}: {warning}" for warning in option_warnings])
        latest_price = _latest_close(market_data.get(ticker, []))
        supports, resistances = _levels_by_kind(tech.get("levels", []))
        row = {
            "ticker": ticker,
            "name": asset.name,
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
        watchlist_table.append(row)
        notes = []
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
        per_ticker_notes[ticker] = notes or ["No high-confidence notes generated."]

    macro_radar = list(macro) + _macro_from_raw_items(paths, report_date)
    risk_flags = sorted(set(str(warning) for warning in warnings if warning))
    return ReportBundle(
        date=report_date,
        watchlist_table=watchlist_table,
        macro_radar=macro_radar,
        per_ticker_notes=per_ticker_notes,
        sentiment_deltas=sentiment_summary,
        technical_setup=technical,
        options_context=options,
        risk_flags=risk_flags,
    )


def render_markdown(bundle: ReportBundle) -> str:
    lines = [
        f"# Trading Wizzard Report - {bundle.date}",
        "",
        "Decision-support only. This report is not financial advice and does not place trades.",
        "",
        "## Watchlist",
        "",
        "| Ticker | Close | Regime | Sentiment Read | Support | Resistance | Backtest | Options |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in bundle.watchlist_table:
        backtest = row.get("backtest_return")
        backtest_text = "" if backtest is None else f"{float(backtest) * 100:.1f}% strategy"
        close = "" if row.get("close") is None else f"{float(row['close']):.2f}"
        lines.append(
            f"| {row['ticker']} | {close} | {row['regime']} | {row['sentiment_view']} | "
            f"{row['support_levels']} | {row['resistance_levels']} | "
            f"{backtest_text} | {row['option_status']} |"
        )

    lines.extend(["", "## Macro Radar", ""])
    if bundle.macro_radar:
        for item in bundle.macro_radar[:20]:
            label = item.get("name") or item.get("title") or item.get("series_id") or "Macro item"
            value = item.get("value")
            suffix = f": {value}" if value not in (None, "") else ""
            lines.append(f"- {item.get('source', 'source')}: {label}{suffix}")
    else:
        lines.append("- No macro/news items available in local artifacts.")

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
    return "\n".join(lines)


def render_html(markdown: str, title: str) -> str:
    escaped = html.escape(markdown)
    return (
        "<!doctype html>\n"
        "<html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
        "max-width:1100px;margin:32px auto;padding:0 16px;line-height:1.45}"
        "pre{white-space:pre-wrap;background:#f7f7f7;padding:16px;border-radius:8px}</style>"
        "</head><body>"
        f"<pre>{escaped}</pre>"
        "</body></html>\n"
    )


def write_report_files(paths: ProjectPaths, bundle: ReportBundle) -> Tuple[Path, Path]:
    markdown = render_markdown(bundle)
    markdown_path = paths.reports / f"{bundle.date}-report.md"
    html_path = paths.reports / f"{bundle.date}-report.html"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(
        render_html(markdown, f"Trading Wizzard Report {bundle.date}"),
        encoding="utf-8",
    )
    return markdown_path, html_path
