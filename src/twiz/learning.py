"""Preserve dated analyst evidence without scoring self-reported trading wins."""

import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .sessions import is_us_equity_session

HEADER = re.compile(r"^(?:canEagle\s*\[量化交易\],?|royal flush 皇者顺888)\s*[—-]\s*(.+)$")
STAMP = re.compile(r"(?P<day>20\d\d-\d\d-\d\d),\s*(?P<time>\d{1,2}:\d\d\s*[AP]M)")


def parse_timeline(text: str, source: str, author: str = "Eagle") -> List[Dict[str, Any]]:
    lines = text.splitlines()
    headers = [(index, HEADER.match(line)) for index, line in enumerate(lines)
               if HEADER.match(line)]
    rows = []
    for position, (index, header) in enumerate(headers):
        end = headers[position+1][0] if position+1 < len(headers) else len(lines)
        body = "\n".join(lines[index+1:end]).strip()
        stamp = STAMP.search(header.group(1))
        day = stamp.group("day") if stamp else None
        if day:
            try:
                date.fromisoformat(day)
            except ValueError:
                day = None
        normalized = re.sub(r"\s+", "", re.sub(r"(?m)^Image\s*$", "", body))
        identity = hashlib.sha256(f"{author}|{day}|{stamp.group('time') if stamp else ''}|"
                                  f"{normalized}".encode()).hexdigest()[:20]
        tags = []
        if re.search(r"预计|预判|如果|若|将会|会继续|可能|等待|支撑|支持|阻力|压力", body):
            tags.append("conditional_view")
        if re.search(r"我(?:今天|刚刚|已经|计划|会)|买入|减仓|止盈|止损|清仓", body):
            tags.append("stated_action_or_plan")
        if re.search(r"预判兑现|成功预测|成功预|利润|恭喜|赚了|盈利", body):
            tags.append("self_reported_result")
        rows.append({"id": identity, "author": author, "source": source,
                     "source_line": index+1, "header": header.group(1),
                     "date": day, "time_text": stamp.group("time") if stamp else None,
                     "timezone": None, "date_status": "explicit" if day else "unresolved",
                     "text": body, "tags": tags,
                     "score_eligible": False,
                     "scoring_note": "原文含混合判断及自述结果，需标注期限、标的、条件后再评价。"})
    return rows


def study_corpus(paths: Iterable[Path]) -> Dict[str, Any]:
    messages, duplicates = {}, 0
    sources = []
    for path in paths:
        rows = parse_timeline(path.read_text(encoding="utf-8"), str(path))
        sources.append({"path": str(path), "posts": len(rows)})
        for row in rows:
            if row["id"] in messages:
                duplicates += 1
                messages[row["id"]].setdefault("also_in", []).append(
                    {"source": row["source"], "line": row["source_line"]})
            else:
                messages[row["id"]] = row
    ordered = sorted(messages.values(), key=lambda r: (r["date"] or "9999", r["source_line"]))
    dated = [row["date"] for row in ordered if row["date"]]
    return {"author": "Eagle", "sources": sources, "messages": ordered,
            "coverage": {"first_date": min(dated) if dated else None,
                         "last_date": max(dated) if dated else None,
                         "unique_posts": len(ordered), "duplicates_removed": duplicates,
                         "unresolved_dates": sum(row["date"] is None for row in ordered)},
            "limitations": ["材料是选摘，未证明覆盖完整一年；视频正文不在原文里。",
                            "原文没有发布时区；盘前重建保守使用之前已完成交易日。",
                            "作者所述宏观数据、资金流和交易获利不是独立验证事实。",
                            "重复帖子、事后自述及没有明确日期/期限的判断不计预测命中率。"]}


def build_learning_replays(
    market_data: Dict[str, List[Dict[str, Any]]], report_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Conservative price-only reconstructions, never a claimed forecast hit rate."""
    from datetime import timedelta
    from statistics import mean

    cases = [
        ("2026-06-08", "^GSPC", "站回20天线并守3天才改善趋势",
         "data/raw/timelines/eagle-june-july-2026.txt:9"),
        ("2026-07-09", "^SOX", "费半60天线是观察支撑，不能当QQQ的同一价位",
         "data/raw/timelines/eagle-june-july-2026.txt:119"),
        ("2026-10-21", "SMH", "接近两月低位、低9与财报条件共同考虑分批建仓",
         "data/raw/timelines/eagle-october-2026.txt:77"),
    ]
    output = []
    for published, ticker, rule, source in cases:
        if report_date and published > report_date:
            continue
        cutoff = date.fromisoformat(published)-timedelta(days=1)
        while not is_us_equity_session(cutoff):
            cutoff -= timedelta(days=1)
        rows = [row for row in market_data.get(ticker, [])
                if row.get("source") == "yfinance" and row.get("session_complete") is not False
                and row.get("date", "9999") <= cutoff.isoformat()]
        rows.sort(key=lambda r: r["date"])
        features = {}
        if len(rows) >= 60 and rows[-1]["date"] == cutoff.isoformat():
            features = {"close": rows[-1]["close"],
                        "sma20_proxy": mean(r["close"] for r in rows[-20:]),
                        "sma60_proxy": mean(r["close"] for r in rows[-60:]),
                        "prior40_low": min(r["low"] for r in rows[-40:]),
                        "prior40_high": max(r["high"] for r in rows[-40:])}
            buy, sell = 0, 0
            for i in range(4, len(rows)):
                buy = buy+1 if rows[i]["close"] < rows[i-4]["close"] else 0
                sell = sell+1 if rows[i]["close"] > rows[i-4]["close"] else 0
            features["public_9bar_proxy"] = {
                "buy_count": min(buy, 9), "sell_count": min(sell, 9),
                "raw_buy_streak": buy, "raw_sell_streak": sell,
                "definition": "每根收盘与4根前比较，连续9根计数代理；无完整Price Flip/Countdown。",
                "source_url": "https://demark.com/sequential-indicator/",
                "author_variant_verified": False,
            }
        status = "reconstructed" if features else "missing_exact_instrument_history"
        output.append({"published_date": published, "ticker": ticker, "source": source,
                       "author_rule": rule, "price_cutoff": cutoff.isoformat(),
                       "status": status,
                       "features": features,
                       "limits": "时区未核实，保守用前一完整交易日；SMA为代理，非作者公式。"})
    return output
