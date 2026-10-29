"""Chinese decision commentary derived only from the report's dated evidence."""

import math
from datetime import date
from typing import Any, Dict, List, Optional

from .sessions import expected_complete_session

UNKNOWN = "数据不足等待"
ROLES = {"SPY": "大盘", "QQQ": "科技成长", "IWM": "小盘", "DIA": "道指",
         "SMH": "半导体", "SOXX": "半导体", "SOXQ": "半导体", "IGV": "软件"}
LEVERAGED = {"SOXL", "SOXS", "TQQQ", "SQQQ", "UPRO", "SPXU"}


def _num(value: Any) -> Optional[float]:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _positive(value: Any) -> Optional[float]:
    value = _num(value)
    return value if value is not None and value > 0 else None


def _fresh(row: Dict[str, Any], as_of: date) -> bool:
    try:
        age = (as_of - date.fromisoformat(str(row.get("data_date"))[:10])).days
        expected = expected_complete_session(as_of.isoformat())
        return (row.get("data_source") == "yfinance" and not row.get("stale") and 0 <= age <= 4
                and str(row.get("data_date"))[:10] == expected
                and row.get("session_complete") is not False)
    except ValueError:
        return False


def _levels(technical: Dict[str, Any], close: float, side: str) -> List[float]:
    values = {_positive(level.get("price")) for level in technical.get("levels", [])
              if level.get("kind") == side}
    return sorted((p for p in values if p and (p < close if side == "support" else p > close)),
                  reverse=side == "support")


def _conditions(view: str, supports: List[float], resistances: List[float],
                sma20: float, sma50: float):
    repair = view in {"偏弱防守", "反弹修复未确认反转"}
    confirmation = max(sma20, sma50, resistances[0] if resistances else 0)
    if repair:
        bull = (f"日收盘站上 {resistances[0]:.2f} 只是反弹第一关；"
                if resistances else "近端阻力尚未识别；")
        bull += f"转强需收复 {confirmation:.2f}，并观察均线结构改善；"
        later = [p for p in resistances if p > confirmation]
    else:
        bull = (f"日收盘站上 {resistances[0]:.2f} 才确认突破近端阻力；"
                if resistances else "有效突破位待确认；")
        later = resistances[1:]
    bull += f"其后观察 {later[0]:.2f}。" if later else "不外推更高目标。"
    bear = (f"日收盘跌破 {supports[0]:.2f}，转短线防守；"
            if supports else "尚无有效支撑位，不给出破位目标；")
    bear += f"下一支撑 {supports[1]:.2f}。" if len(supports) > 1 else "更低支撑待确认。"
    if view == "偏弱防守":
        invalidation = f"日收盘收复 {confirmation:.2f}，当前偏弱结构需重评；近端反弹不等于反转。"
    elif view in {"偏强延续", "偏强整理"}:
        invalidation = f"日收盘跌破50日线 {sma50:.2f}，当前偏强结构需重评。"
    elif view == "震荡等待":
        invalidation = "日收盘走出上述有效区间后，重新判断方向。"
    else:
        floor = supports[0] if supports else min(sma20, sma50)
        invalidation = f"日收盘跌破 {floor:.2f}，当前修复判断失效。"
    return bull, bear, invalidation


def _ticker_decision(row: Dict[str, Any], technical: Dict[str, Any],
                     fundamentals: Dict[str, Any], as_of: date) -> Dict[str, Any]:
    ticker = str(row.get("ticker", ""))
    close = _positive(row.get("close"))
    ind = technical.get("indicators") or {}
    sma20, sma50, tech_close = (_positive(ind.get(k)) for k in ("sma20", "sma50", "close"))
    result = {"ticker": ticker, "view": UNKNOWN, "action": "等待有效日线数据再作方向判断。",
              "bullish": "暂无可验证触发位。", "bearish": "暂无可验证触发位。",
              "invalidation": "数据补齐前不建立交易判断。", "evidence": "", "usable": False}
    if not (_fresh(row, as_of) and close and sma20 and sma50 and tech_close
            and abs(close - tech_close) <= max(0.011, close * 0.00001)):
        return result
    if close > sma20 and sma20 >= sma50:
        view, action = "偏强延续", "沿偏强结构观察；等待回踩守住支撑或收盘突破，不追盘中拉升。"
    elif close < sma20 and close < sma50:
        view, action = "偏弱防守", "不把下跌直接当便宜；先等收盘收复阻力再评估修复。"
    elif close >= sma20 and sma20 < sma50:
        view = "反弹修复未确认反转"
        action = "按反弹修复处理；收复50日线并改善均线结构前，不升级为主升。"
    elif close > sma50 and sma20 > sma50:
        view, action = "偏强整理", "短线低于20日线，50日线尚守；先看回撤能否止住。"
    else:
        view, action = "震荡等待", "均线与价格未形成清晰同向结构，等待收盘走出区间。"
    supports = _levels(technical, close, "support")
    resistances = _levels(technical, close, "resistance")
    bull, bear, invalidation = _conditions(view, supports, resistances, sma20, sma50)
    flags = fundamentals.get("flags") or []
    if any(flag in flags for flag in ("negative_free_cash_flow", "share_issuance_dilution",
                                      "capex_exceeds_operating_cash_flow")):
        action += "现金流或稀释有警示，新增暴露前核对报表；该标签本身不构成卖出指令。"
    if ticker in LEVERAGED:
        action += "这是杠杆ETF，须另行评估每日重置与波动损耗。"
    return {**result, "view": view, "action": action, "bullish": bull, "bearish": bear,
            "invalidation": invalidation, "usable": True,
            "evidence": (f"{ticker} {row['data_date']} "
                         + ("已完成常规交易日收盘 " if row.get("session_complete") is True
                            else "日线最新值 ") + f"{close:.2f}；"
                         f"20日线 {sma20:.2f}，50日线 {sma50:.2f}（Yahoo日线）。")}


def build_daily_brief(
    report_date: str,
    watchlist_table: List[Dict[str, Any]],
    technical: Dict[str, Dict[str, Any]],
    playbook: Dict[str, Dict[str, Any]],
    macro_gate: Dict[str, Any],
    leadership: Dict[str, Any],
    events: List[Dict[str, Any]],
    fundamentals: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build conditional views, never a probability model or intraday forecast."""
    as_of = date.fromisoformat(report_date)
    decisions = [_ticker_decision(row, technical.get(str(row.get("ticker")), {}),
                                 fundamentals.get(str(row.get("ticker")), {}), as_of)
                 for row in watchlist_table]
    by_ticker = {row["ticker"]: row for row in decisions}
    anchors = [by_ticker[t] for t in ("QQQ", "SPY") if t in by_ticker and by_ticker[t]["usable"]]
    incomplete = [row["ticker"] for row in decisions if not row["usable"]]
    missing = ([f"{', '.join(incomplete)}：来源、日期或均线证据不足，暂缓方向判断。"]
               if incomplete else [])
    if len(anchors) == 2 and anchors[0]["view"] == anchors[1]["view"]:
        stance = anchors[0]["view"]
        headlines = {
            "偏弱防守": "先按防守看待：QQQ与SPY均在20/50日线下，反弹未确认反转。",
            "偏强延续": "短线偏强延续：QQQ与SPY均在20/50日线上，守住支撑再逐级看阻力。",
            "反弹修复未确认反转": "先按反弹修复处理：QQQ与SPY均线结构尚未转强，不升级为主升。",
            "偏强整理": "先看偏强整理：QQQ与SPY短线回撤，50日线仍是结构观察线。",
            "震荡等待": "指数方向尚未确认，先等日收盘走出区间。",
        }
        headline = headlines[stance]
    elif anchors:
        stance = anchors[0]["view"] if len(anchors) == 1 else "指数分化"
        headline = ("指数分化：" if len(anchors) == 2 else "仅单一指数证据完整：")
        headline += "；".join(f"{row['ticker']} {row['view']}" for row in anchors)
        headline += "，按各自收盘门槛处理。"
        if len(anchors) == 1:
            missing.append("QQQ/SPY仅一个有完整证据，不能据此代表整个市场。")
    else:
        stance, headline = UNKNOWN, "大盘锚点证据不足，当前先等待，不给出涨跌路线。"
    evidence = [row["evidence"] for row in anchors]
    for row in anchors:
        pb = playbook.get(row["ticker"], {})
        dd = _num(pb.get("drawdown_pct"))
        raw_close = next(item["close"] for item in watchlist_table
                         if item["ticker"] == row["ticker"])
        if (dd is not None and -100 <= dd <= 0 and pb.get("as_of") == report_date
                and _positive(pb.get("close")) is not None
                and abs(float(pb["close"]) - raw_close) <= max(0.011, raw_close * 0.00001)):
            evidence.append(
                f"{row['ticker']} 较区间高点回撤 {dd:.1f}%；位置背景本身不确认反转。"
            )
    macro_state = macro_gate.get("state", "unknown")
    if macro_state == "risk_off":
        evidence.append("宏观门槛偏防守，技术反弹不能单独解除这一约束。")
    elif macro_state == "caution":
        evidence.append("宏观门槛处于谨慎状态，突破需要收盘确认。")
    elif macro_state != "risk_on":
        missing.append("宏观未知，仅允许用户预算内的保守条件试仓，不自动提高额度或安排卖PUT。")
    imminent = []
    for event in events:
        try:
            days = (date.fromisoformat(str(event.get("date"))[:10]) - as_of).days
        except ValueError:
            continue
        if 0 <= days <= 3 and (_num(event.get("importance")) or 0) >= 4:
            imminent.append((bool(event.get("tickers")), days, event))
    for _, days, event in sorted(imminent, key=lambda item: item[:2])[:3]:
        estimated = event.get("precision") in {"approximate", "rule_based"}
        evidence.append(f"未来{days}天：{event.get('name')}（{event.get('date')}"
                        + ("，估计日期，须核对正式日程）。" if estimated else "）。"))
    if len(imminent) > 3:
        evidence.append(f"另有{len(imminent) - 3}项临近事件，详见事件表。")
    scenarios = [{key: row[key] for key in ("ticker", "view", "bullish", "bearish", "invalidation")}
                 for row in anchors]
    raw_rows = {str(row.get("ticker")): row for row in watchlist_table}
    benchmark = raw_rows.get("SPY", {})
    benchmark_return = _num(benchmark.get("return_20d_pct"))
    sectors = []
    for row in decisions:
        if not row["usable"] or row["ticker"] not in ROLES:
            continue
        raw = raw_rows[row["ticker"]]
        measured = _num(raw.get("return_20d_pct"))
        aligned = (by_ticker.get("SPY", {}).get("usable") and benchmark_return is not None
                   and measured is not None and raw.get("data_date") == benchmark.get("data_date")
                   and raw.get("return_20d_start")
                   and raw.get("return_20d_start") == benchmark.get("return_20d_start"))
        relative = round(measured - benchmark_return, 2) if aligned else None
        condition = (f"同区间20日价格变动 {measured:+.2f}%，较SPY {relative:+.2f}个百分点；"
                     if relative is not None else "同区间相对强弱待验证；")
        sectors.append({"ticker": row["ticker"], "view": f"{ROLES[row['ticker']]}：{row['view']}",
                        "condition": condition + row["invalidation"],
                        "relative_return_20d_pct": relative})
    sectors.sort(key=lambda row: (row["relative_return_20d_pct"] is None,
                                 -(row["relative_return_20d_pct"] or 0)))
    compared = [row for row in sectors if row["ticker"] in {"SMH", "SOXX", "SOXQ", "IGV"}
                and row["relative_return_20d_pct"] is not None]
    if compared:
        evidence.append("板块观察顺序（同区间20日相对SPY）：" + "；".join(
            f"{row['ticker']} {row['relative_return_20d_pct']:+.2f}个百分点" for row in compared
        ) + "。按已发生的相对强弱排序，后续仍需各自突破确认。")
    if leadership.get("state") in {"weakening", "broken"}:
        sectors.append({"ticker": "硬件领导股", "view": "领导股转弱，半导体扩散仍需修复",
                        "condition": "这是行业约束，不据此改写SPY的大盘方向。"})
    if stance == UNKNOWN:
        flat, held, lever = ("数据补齐前暂停新增判断。", "先核对数据与原有风控条件。",
                             "先核验风险暴露，不能依据缺失数据延长持有。")
    elif stance == "偏弱防守":
        flat, held, lever = ("暂不追反弹，等待收盘收复转强门槛。",
                             "反弹到阻力时复核持仓并控制风险；支撑失守不沿用原修复判断。",
                             "优先控制暴露，先核对破位与事件窗口，再讨论方向。")
    elif stance in {"偏强延续", "偏强整理"}:
        flat, held, lever = ("优先等回踩支撑确认或日收盘突破，不追盘中拉升。",
                             "沿支撑复核持仓，接近阻力逐级评估；50日线失守重评结构。",
                             "偏强也需控制波动风险，按收盘失效门槛复核暴露。")
    elif stance == "指数分化":
        flat, held, lever = ("分别等待QQQ与SPY触发，不把单一指数强弱推广到全市场。",
                             "按各自标的结构复核，不按统一大盘方向处理全部持仓。",
                             "分化阶段先核对标的与杠杆方向，优先控制不匹配暴露。")
    else:
        flat, held, lever = ("按修复或区间观察，未确认转强前不追拉升。",
                             "阻力处复核修复进度，跌破失效线则重评原判断。",
                             "修复未确认阶段优先控制暴露，不把反弹预设为主升。")
    actions = [{"profile": name, "action": action} for name, action in
               zip(("空仓", "已有普通持仓", "已有杠杆持仓"), (flat, held, lever))]
    if any(row["usable"] and not raw_rows[row["ticker"]].get("session_complete")
           for row in decisions):
        missing.append("部分日线未记录收盘完成标志；突破与失效仍需正式日收盘确认。")
    return {
        "date": report_date, "headline": headline, "stance": stance,
        "horizon": {"short": f"{stance}；以最近支撑与阻力的日收盘确认作为行动开关。",
                    "medium": "只作条件推演：收盘突破后逐级观察下一阻力；跌破支撑则转防守。"
                              "未确认的反弹不直接外推为趋势反转。"},
        "evidence": evidence, "market_scenarios": scenarios, "sectors": sectors,
        "ticker_decisions": decisions, "actions": actions, "missing": missing,
    }
