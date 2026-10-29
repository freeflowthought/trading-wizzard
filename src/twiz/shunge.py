"""A dated, auditable reading of the supplied commentary, not a trained model."""

import math
from datetime import date, timedelta
from statistics import mean
from typing import Any, Dict, List, Optional

from .sessions import expected_complete_session


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError):
        return None


def _history(rows: List[Dict[str, Any]], cutoff: str) -> List[Dict[str, Any]]:
    """Reject ambiguous duplicate dates and non-provider rows before calculating."""
    valid = {}
    duplicates = set()
    for row in rows:
        try:
            stamp = date.fromisoformat(str(row.get("date"))[:10]).isoformat()
        except ValueError:
            continue
        values = {key: _number(row.get(key)) for key in ("close", "high", "low")}
        if (stamp > cutoff or row.get("source") != "yfinance"
                or row.get("session_complete") is False
                or any(v is None for v in values.values())
                or not values["low"] <= values["close"] <= values["high"]):
            continue
        if stamp in valid:
            duplicates.add(stamp)
        valid[stamp] = {**row, **values, "date": stamp}
    return [valid[key] for key in sorted(valid) if key not in duplicates]


def _weekly_average(rows: List[Dict[str, Any]], cutoff: str, window: int) -> Optional[float]:
    # Friday must have elapsed. Holiday weeks use their last observed session.
    end = date.fromisoformat(cutoff)
    weeks = {}
    for row in rows:
        day = date.fromisoformat(row["date"])
        friday = day + timedelta(days=4-day.weekday())
        if friday <= end:
            weeks[friday] = row["close"]
    closes = [weeks[key] for key in sorted(weeks)]
    return round(mean(closes[-window:]), 2) if len(closes) >= window else None


def build_shunge_context(
    report_date: str, market_data: Dict[str, List[Dict[str, Any]]],
    profile: Dict[str, Any], case_history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Freeze case calculations at the author's date; never look ahead to outcomes."""
    result = {"patterns": [], "case": {"available": False}, "scenario": {"active": False},
              "source_notes": [
                  "归纳来自用户提供的6月、8–9月及10月样本；8–9月存档为选摘，非完整记录。",
                  "这些是作者的条件判断，尚未验证预测胜率；42%/58%等不作为本系统概率。",
                  "650推导是我们的解释假设，不能证明作者使用了该公式，也不能证明650会到达或守住。",
                  "本轮10%预算、分批比例与日收盘确认是用户偏好及本系统规则，不是作者原始仓位公式。",
              ]}
    for pattern in profile.get("patterns", []):
        evidence = [row for row in pattern.get("evidence", [])
                    if str(row.get("date", "9999")) <= report_date]
        if evidence:
            result["patterns"].append({**pattern, "evidence": evidence})
    spec = profile.get("case", {})
    as_of = str(spec.get("as_of", ""))
    try:
        if date.fromisoformat(as_of) > date.fromisoformat(report_date):
            return result
        review_after = date.fromisoformat(str(spec["review_after"]))
        floor, ceiling = [_number(v) for v in spec["source_support"]]
        target = _number(spec["source_target"])
        confirmation, reversal = [_number(spec[k]) for k in ("confirmation", "reversal")]
        low, high = [_number(v) for v in spec["prior_zone"]]
        if (any(v is None for v in (floor, ceiling, target, confirmation, reversal, low, high))
                or not low < high < floor <= ceiling < confirmation < reversal
                or not low <= target <= high):
            return result
    except (ValueError, TypeError, KeyError):
        return result
    ticker = str(spec.get("ticker", "QQQ"))
    basis = "本次Yahoo历史复权数据；可能与作者当时屏幕报价略有差异"
    history = market_data.get(ticker, [])
    if (case_history and case_history.get("source") == "yfinance"
            and case_history.get("ticker") == ticker
            and case_history.get("as_of") == as_of
            and case_history.get("price_basis") == "unadjusted"):
        history = [{**row, "source": "yfinance"} for row in case_history.get("rows", [])]
        basis = ("Yahoo日线（auto_adjust=False，拆股按Yahoo历史口径）；"
                 "事后下载并截断至作者发言日，非当时留存快照")
    rows = _history(history, as_of)
    case = {"available": False, "ticker": ticker, "as_of": as_of,
            "source": spec.get("source"), "source_line": spec.get("source_line"),
            "source_support": [floor, ceiling], "source_target": target,
            "prior_zone": [low, high], "price_basis": basis,
            "method_source_url": spec.get("method_source_url"),
            "limitations": ["区间高度投影只用于破位压力测试；未识别或证明标准矩形/双顶形态。",
                            "历史成交区域不是成交量密集区；没有用日线数据推断筹码分布。"]}
    if rows and rows[-1]["date"] == as_of:
        lookback = int(spec.get("high_lookback_sessions", 63))
        if len(rows) >= max(lookback, 200):
            peak = max(rows[-lookback:], key=lambda row: row["high"])
            midpoint = (floor+ceiling)/2
            projected = 2*midpoint-peak["high"]
            hits = [r for r in rows if r["date"].startswith(str(spec.get("prior_zone_month")))
                    and r["low"] <= high and r["high"] >= low]
            friday = date.fromisoformat(as_of)
            friday -= timedelta(days=(friday.weekday()-4) % 7 or 7)
            prior_friday = next((r for r in rows if r["date"] == friday.isoformat()), None)
            case.update(
                available=projected > 0, range_high=round(peak["high"], 2),
                range_high_date=peak["date"], measured_target=round(projected, 2),
                sma200=round(mean(r["close"] for r in rows[-200:]), 2),
                weekly_sma30=_weekly_average(rows, as_of, 30),
                weekly_sma50=_weekly_average(rows, as_of, 50),
                prior_zone_hits=[{k: r[k] for k in ("date", "low", "high", "close")}
                                 for r in hits],
                prior_friday=({k: prior_friday[k] for k in ("date", "low", "close")}
                              if prior_friday else None),
                support_reactions=[
                    {k: r[k] for k in ("date", "low", "close")} for r in rows[-lookback:]
                    if abs(r["low"]/midpoint-1) <= 0.005 and r["close"] >= floor
                ],
            )
            if prior_friday and abs(prior_friday["low"]-midpoint) > 1:
                case["limitations"].append(
                    f"源文另称704为上周五低位，但Yahoo {friday.isoformat()}最低"
                    f"{prior_friday['low']:.2f}；两者不完全吻合，704–705保留为作者判断线，"
                    "不伪称已精确复刻其行情依据。"
                )
    result["case"] = case
    current = _history(market_data.get(ticker, []), report_date)
    latest = current[-1] if current else {}
    close = latest.get("close")
    active = bool(as_of <= report_date <= review_after.isoformat() and close
                  and latest.get("date") == expected_complete_session(report_date)
                  and latest.get("session_complete") is not False)
    breaches = [{"date": row["date"], "close": row["close"]} for row in current
                if as_of < row["date"] <= report_date and row["close"] < floor
                and row.get("session_complete") is not False]
    state = ("below_floor" if close and close < floor else
             "reversal_watch" if close and close >= reversal else
             "repair_watch" if close and close >= confirmation else "defensive_watch")
    result["scenario"] = {
        "active": active, "ticker": ticker, "review_after": review_after.isoformat(),
        "source_date": as_of, "data_date": latest.get("date"), "close": close,
        "data_source": latest.get("source"), "session_complete": latest.get("session_complete"),
        "source": spec.get("source"), "source_line": spec.get("source_line"),
        "floor_breach": breaches[0] if breaches else None,
        "latest_floor_breach": breaches[-1] if breaches else None,
        "reassessment": spec.get("reassessment"),
        "support_floor": floor, "support_ceiling": ceiling, "confirmation": confirmation,
        "reversal": reversal, "extreme_zone": [low, high], "state": state,
        "operational_rule": spec.get("operational_rule"),
        "actions": [
            f"{floor:g}–{ceiling:g}守住：只评估反弹试仓，不把长线目标当作短线保证。",
            f"日收盘低于{floor:g}：原试仓剧本失效，取消下方连续加仓；先观察中间支撑。",
            f"{low:g}–{high:g}：独立的深跌观察区，触价不自动买；需止跌、收复区间上沿、"
            "后续回踩不破并重新核对宏观，才能另建方案；仍共享本轮10%上限。",
            f"收复{confirmation:g}：修复第一关；进一步站稳{reversal:g}且形成更高低点，"
            "才重新评估反转。",
        ],
    }
    return result


def _nonnegative(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _reviewed_after(scenario: Dict[str, Any], breach: Dict[str, Any], report_date: str) -> bool:
    review = scenario.get("reassessment") or {}
    try:
        stamp = date.fromisoformat(str(review.get("date"))).isoformat()
        return bool(review.get("source") and breach["date"] < stamp <= report_date)
    except (TypeError, ValueError, KeyError):
        return False


def apply_structure_guard(plan: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """A dated author scenario may tighten a fresh plan, never resurrect a broken one."""
    scenario = context.get("scenario", {})
    if (not scenario.get("active") or scenario.get("ticker") != plan.get("ticker")
            or plan.get("status") == "unavailable"):
        return plan
    try:
        report = date.fromisoformat(str(plan.get("date"))).isoformat()
        source_date = date.fromisoformat(str(scenario.get("source_date"))).isoformat()
        expires = date.fromisoformat(str(scenario.get("review_after"))).isoformat()
        quote_date = date.fromisoformat(str(scenario.get("data_date"))).isoformat()
        inputs = plan.get("model_inputs") or {}
        close, plan_close = _number(scenario.get("close")), _number(inputs.get("close"))
        floor, first_exit, second_exit = [_number(scenario.get(key))
                                          for key in ("support_floor", "confirmation", "reversal")]
        cap = _number(plan.get("budget", {}).get("max_new_capital_pct"))
        if (not source_date <= report <= expires or quote_date != plan.get("price_date")
                or quote_date != expected_complete_session(report)
                or inputs.get("source") != "yfinance"
                or scenario.get("data_source") != "yfinance"
                or scenario.get("session_complete") is False
                or any(value is None for value in
                       (close, plan_close, floor, first_exit, second_exit, cap))
                or abs(close-plan_close) > 0.011 or not floor < first_exit < second_exit):
            return plan
    except (TypeError, ValueError):
        return plan
    cap = min(10.0, cap)
    original = [row for row in plan.get("entries", [])
                if _number(row.get("price")) is not None
                and _number(row.get("account_pct")) is not None]
    original_stop = _number(plan.get("stop", {}).get("price"))
    old_loss = _nonnegative(plan.get("stop", {}).get("modeled_loss_pct"))
    if old_loss is None:
        if original_stop is None or not original:
            return plan
        old_loss = sum(row["account_pct"]*max(0, 1-original_stop/row["price"]) for row in original)
    loss_cap = min(0.5, old_loss)
    old_enabled = _nonnegative(plan.get("budget", {}).get("currently_enabled_pct"))
    if old_enabled is None:
        old_enabled = sum(_nonnegative(row.get("currently_enabled_pct")) or 0 for row in original)
    macro_block = plan.get("status") == "blocked"
    breach = scenario.get("latest_floor_breach") or scenario.get("floor_breach")
    invalidated = False
    if breach:
        try:
            breach_date = date.fromisoformat(str(breach["date"])).isoformat()
            breach_close = _number(breach.get("close"))
            if not (source_date < breach_date <= quote_date
                    and breach_close and breach_close < floor):
                return plan
            invalidated = not _reviewed_after(scenario, breach, report)
        except (TypeError, ValueError, KeyError):
            return plan
    below_floor = close < floor
    entries = [dict(row) for row in original
               if floor < row["price"] < first_exit
               and first_exit-row["price"] >= row["price"]-floor-1e-9]
    entries.sort(key=lambda row: -row["price"])
    if below_floor or invalidated:
        entries = []
    funded = sum(row["account_pct"] for row in entries)
    loss = sum(row["account_pct"]*(1-floor/row["price"]) for row in entries)
    scale = min(1.0, cap/funded if funded else 1, loss_cap/loss if loss else 1)
    for index, row in enumerate(entries):
        row.update(original_batch=row.get("batch"), batch=index+1,
                   account_pct=math.floor(row["account_pct"]*scale*10_000)/10_000)
        row["currently_enabled_pct"] = (min(row["account_pct"], 2.0, old_enabled)
                                         if index == 0 and not macro_block else 0.0)
        row["status"] = ("仅在结构守住及回踩确认后试仓" if row["currently_enabled_pct"]
                         else "暂停，等待重新评估")
        row["trigger"] = (f"未来触及{row['price']:.2f}且日收盘守住或收复；"
                          f"{floor:g}结构线失守即取消，不表示已经成交。")
    entries = [row for row in entries if row["account_pct"] > 0]
    funded = sum(row["account_pct"] for row in entries)
    enabled = sum(row["currently_enabled_pct"] for row in entries)
    budget = {**plan["budget"], "max_new_capital_pct": cap, "planned_capital_pct": funded,
              "currently_enabled_pct": enabled, "uncommitted_cash_pct": cap-enabled,
              "estimated_weighted_entry": (funded/sum(r["account_pct"]/r["price"] for r in entries)
                                           if funded else None)}
    exits = []
    if entries:
        technical_targets = [_number(row.get("price")) for row in plan.get("exits", [])]
        third = min((p for p in technical_targets if p and p > second_exit), default=None)
        targets = zip((first_exit, second_exit, third), (40, 30, 30))
        for stage, (price, trim) in enumerate(targets, 1):
            basis = "author_scenario" if stage < 3 else "technical_candidate"
            reference_return = (price/entries[0]["price"]-1)*100 if price else None
            source = (scenario.get("source") or context.get("case", {}).get("source")
                      if stage < 3 else "original_action_plan.technical_candidate")
            status = ("作者有日期的情景兑现参考，不是新观察阻力" if stage < 3 else
                      "原方案更高技术候选" if price else "没有更高技术候选，暂不设目标")
            exits.append({"stage": stage, "price": price, "trim_original_dip_pct": trim,
                          "reference_return_pct": reference_return, "reference_kind": basis,
                          "source_date": source_date if stage < 3 else quote_date,
                          "source": source, "status": status})
    stop = {**plan.get("stop", {}), "price": floor,
            "modeled_loss_pct": sum(r["account_pct"]*(1-floor/r["price"]) for r in entries),
            "currently_enabled_loss_pct": sum(r["currently_enabled_pct"]*(1-floor/r["price"])
                                               for r in entries),
            "trigger": (f"本系统按日收盘低于{floor:g}取消未成交加仓，复核退出本轮抄底仓；"
                        "不指代核心仓。")}
    if below_floor:
        action = f"当前有效收盘低于{floor:g}，取消原方案，暂停新增买入。"
    elif invalidated:
        action = (f"{breach['date']}日收盘已失守{floor:g}；即使随后反弹，"
                  "旧方案仍停用，须明确重新评估。")
    elif not entries:
        action = "现有候选不满足结构线以上及到首个兑现参考至少1R，暂停新增买入。"
    elif macro_block or not enabled:
        action = "结构筛选保留候选，但原风险门槛仍阻止新增买入，不自动解除暂停。"
    else:
        action = (f"先等{entries[0]['price']:.2f}回踩及日收盘确认，最多启用{enabled:g}%账户资金；"
                  f"{floor:g}失守即取消，其他批次待重新评估。")
    limitations = [note for note in plan.get("limitations", [])
                   if not note.startswith("按完整三批成交模型")]
    limitations.extend([
        f"结构层依据{source_date}作者情景，至{expires}需重评；日收盘触发是本系统口径。",
        "第一保留候选最多启用账户2%，其余暂不启用；所有情景仍共享本轮10%上限与原更严格损失预算。",
        f"{first_exit:g}/{second_exit:g}为作者有日期的情景参考，不是本日已观察阻力或保证到达。",
        "历史日收盘失守后，即使价格回升也不自动恢复旧方案；须有带日期和出处的明确重评。",
    ])
    return {**plan, "entries": entries, "exits": exits, "budget": budget, "stop": stop,
            "buy_now_pct": 0.0,
            "status": "blocked" if not entries or macro_block else plan["status"],
            "headline": (f"{plan['ticker']} 结构先行：守{floor:g}，"
                         f"先看{first_exit:g}/{second_exit:g}兑现"),
            "current_action": action, "limitations": list(dict.fromkeys(limitations)),
            "entry_reference": (f"收益率以首个保留候选{entries[0]['price']:.2f}为参考；"
                                "实际成交成本尚未知。"
                                if entries else "原方案暂停，没有可启用的新入场成本参考。")}
