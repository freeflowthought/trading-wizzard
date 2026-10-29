"""One conditional dip campaign, measured in percent of the total account."""
import math
from datetime import date
from typing import Any, Dict, List, Optional

from .sessions import expected_complete_session


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _fresh(row: Dict[str, Any], technical: Dict[str, Any], as_of: date) -> bool:
    try:
        observed = date.fromisoformat(str(row.get("data_date"))[:10])
        age = (as_of-observed).days
        expected = expected_complete_session(as_of.isoformat())
        close = _num(row.get("close"))
        ind = technical.get("indicators") or {}
        values = [_num(ind.get(key)) for key in ("close", "sma20", "sma50")]
        return bool(row.get("data_source") == "yfinance" and not row.get("stale")
                    and row.get("session_complete") is not False
                    and observed.isoformat() == expected
                    and 0 <= age <= 4 and close and close > 0
                    and all(v is not None and v > 0 for v in values)
                    and abs(close-values[0]) <= max(0.011, close*0.00001))
    except ValueError:
        return False


def _levels(technical: Dict[str, Any], close: float, kind: str) -> List[float]:
    values = {_num(row.get("price")) for row in technical.get("levels", [])
              if row.get("kind") == kind}
    return sorted((p for p in values if p is not None and 0 < p
                   and (p < close if kind == "support" else p > close)), reverse=kind == "support")


def build_action_plan(
    report_date: str, watchlist_table: List[Dict[str, Any]],
    technical: Dict[str, Dict[str, Any]], macro_gate: Dict[str, Any], policy: Dict[str, Any],
) -> Dict[str, Any]:
    """This is a policy template, never an instruction claiming a fill or known holding."""
    ticker = str(policy.get("primary_ticker", "QQQ")).upper()
    ceiling = _num(policy.get("max_new_capital_pct", 10))
    fractions = policy.get("entry_fractions", [0.4, 0.35, 0.25])
    trims = policy.get("trim_fractions", [0.4, 0.3, 0.3])
    loss_cap = _num(policy.get("max_campaign_loss_pct", 0.5))
    spacing = _num(policy.get("min_entry_spacing_pct", 0.75))
    initial_cap = _num(policy.get("unknown_macro_initial_cap_pct", 2))
    plan = {
        "ticker": ticker, "date": report_date, "status": "unavailable", "buy_now_pct": 0.0,
        "price_date": None, "model_inputs": {},
        "headline": f"{ticker} 本轮抄底方案暂缓", "current_action": "现在不新增买入。",
        "budget": {"ticker": ticker, "max_new_capital_pct": ceiling,
                   "planned_capital_pct": 0.0, "currently_enabled_pct": 0.0,
                   "uncommitted_cash_pct": ceiling, "estimated_weighted_entry": None},
        "entries": [], "exits": [], "stop": {"price": None, "modeled_loss_pct": None},
        "other_assets": [{"ticker": row["ticker"], "new_allocation_pct": 0.0}
                         for row in watchlist_table if row.get("ticker") != ticker],
        "leverage": False, "limitations": [],
        "entry_reference": "止盈收益率以首批计划入场价为参考；实际成交成本尚未知。",
        "trim_basis": ("首档止盈前锁定本轮实际买入的抄底股数为100%；"
                       "各档比例均按该原始股数，非剩余股数或核心仓。"),
        "budget_scope": "仅本轮新增资金；未启用金额是本轮预算余额，不是账户实际现金。",
    }

    def unavailable(reason):
        plan["limitations"].append(reason)
        return plan

    weights = [_num(v) for v in fractions] if isinstance(fractions, list) else []
    trim_weights = [_num(v) for v in trims] if isinstance(trims, list) else []
    if (ceiling is None or not 0 < ceiling <= 100 or loss_cap is None or not 0 < loss_cap <= 100
            or spacing is None or spacing <= 0 or initial_cap is None
            or not 0 <= initial_cap <= ceiling
            or len(weights) != 3 or len(trim_weights) != 3
            or any(v is None or v <= 0 for v in weights+trim_weights)
            or abs(sum(weights)-1) > 1e-9 or abs(sum(trim_weights)-1) > 1e-9):
        return unavailable("资金/模型损失限额须大于0且不超过账户100%，"
                           "宏观未知首批须在本轮资金上限内，分批比例须合计100%。")
    row = next((r for r in watchlist_table if r.get("ticker") == ticker), {})
    tech = technical.get(ticker, {})
    try:
        as_of = date.fromisoformat(report_date)
    except ValueError:
        return unavailable("报告日期无效。")
    if ticker.startswith("^") or ticker in {"SPX", "SOXL", "TQQQ", "UPRO", "SQQQ", "SOXS", "SPXU"}:
        return unavailable("本方案仅用于非杠杆可交易标的，不直接购买指数点位或杠杆ETF。")
    if not _fresh(row, tech, as_of):
        return unavailable("需要来源明确、未过期且价格一致的Yahoo日线及20/50日线。")
    close = float(row["close"])
    plan["price_date"] = row.get("data_date")
    plan["model_inputs"] = {
        "close": close, "source": row.get("data_source"),
        "close_source": row.get("close_source", "daily_history"),
        "session_complete": row.get("session_complete"),
        "sma20": tech["indicators"]["sma20"], "sma50": tech["indicators"]["sma50"],
        "level_source": "本次technical.levels中的实际支撑与阻力；不生成新的整数字位。",
    }
    supports, resistance = _levels(tech, close, "support"), _levels(tech, close, "resistance")
    entries = []
    for support in supports:
        if support > close*0.995:
            continue
        if not entries or (entries[-1]-support)/entries[-1]*100 >= spacing:
            entries.append(support)
        if len(entries) == 3:
            break
    stop = next((p for p in supports if entries and p < entries[-1]), None)
    if len(entries) != 3 or stop is None:
        return unavailable("实际支撑不足以构成三档间隔入场及更低的止损参考位，暂不拼凑价格。")
    risk_unit = entries[0]-stop
    selected_exits = []
    previous = entries[0]
    for multiple in (1, 2, 3):
        target = next((p for p in resistance
                       if p >= entries[0]+multiple*risk_unit and p > previous), None)
        selected_exits.append(target)
        if target is not None:
            previous = target
    if selected_exits[0] is None:
        return unavailable("实际阻力没有达到首批1R，缺少足够风险回报结构，暂缓新方案。")
    cash = [ceiling*w for w in weights]
    modeled_loss = sum(pct*(1-stop/entry) for pct, entry in zip(cash, entries))
    scale = min(1.0, loss_cap/modeled_loss) if modeled_loss else 1.0
    cash = [math.floor(pct*scale*10_000)/10_000 for pct in cash]
    while sum(cash) > ceiling:  # Float addition must not exceed the configured account ceiling.
        largest = max(range(len(cash)), key=cash.__getitem__)
        cash[largest] = max(0., math.nextafter(cash[largest]-(sum(cash)-ceiling), 0.))
    if not any(cash):
        return unavailable("风险限额缩减后没有可分配预算，暂不启用本轮方案。")
    if scale < 1:
        plan["limitations"].append(
            f"按完整三批成交模型缩减全部批次，使止损参考价下账户损失不超过{loss_cap:g}%。"
        )
    state = macro_gate.get("state", "unknown")
    enabled = list(cash)
    if state == "risk_off":
        enabled = [0.0, 0.0, 0.0]
    elif state == "caution":
        enabled[0] *= 0.5
    elif state != "risk_on":
        enabled = [min(cash[0], initial_cap), 0.0, 0.0]
    for i, (price, pct, allowed) in enumerate(zip(entries, cash, enabled), start=1):
        status = "宏观偏防守，暂停" if state == "risk_off" else (
            "等待宏观门槛明确" if not allowed else "仅在回踩确认后启用"
        )
        sources = next((level.get("sources", []) for level in tech.get("levels", [])
                        if level.get("price") == price and level.get("kind") == "support"), [])
        plan["entries"].append({
            "batch": i, "price": price, "account_pct": pct,
            "currently_enabled_pct": allowed, "status": status, "sources": sources,
            "trigger": f"未来触及{price:.2f}且当日日收盘守住或收复；更低档亦需独立确认。",
        })
    for i, (price, weight) in enumerate(zip(selected_exits, trim_weights), start=1):
        plan["exits"].append({
            "stage": i, "price": price, "trim_original_dip_pct": round(weight*100, 6),
            "reference_return_pct": (price/entries[0]-1)*100 if price is not None else None,
            "status": ("本轮实际持仓建立后执行" if price is not None
                       else "缺少更高实际阻力，暂不设目标"),
        })
    funded = sum(cash)
    plan["budget"].update(planned_capital_pct=funded, currently_enabled_pct=sum(enabled),
                          uncommitted_cash_pct=ceiling-sum(enabled),
                          estimated_weighted_entry=funded/sum(
                              pct/p for pct, p in zip(cash, entries)))
    plan["stop"] = {
        "price": stop, "modeled_loss_pct": sum(pct*(1-stop/p) for pct, p in zip(cash, entries)),
        "currently_enabled_loss_pct": sum(pct*(1-stop/p) for pct, p in zip(enabled, entries)),
        "trigger": (f"日收盘跌破{stop:.2f}：取消所有未成交加仓，复核退出本轮抄底仓；"
                    "不指代核心持仓。"),
    }
    plan.update(status="blocked" if state == "risk_off" else "conditional_plan",
                headline=f"{ticker} 仅做一轮分批抄底；现在买入0%")
    if state == "risk_off":
        plan["current_action"] = "宏观门槛偏防守，本轮所有新增买入暂停。"
    else:
        plan["current_action"] = (f"先等{entries[0]:.2f}回踩及日收盘确认，"
                                  f"再讨论首批{enabled[0]:g}%账户资金。")
        if state not in {"risk_on", "caution"}:
            plan["current_action"] += "宏观未知，第二、三批仅为预算预案，暂不启用。"
    plan["limitations"].extend([
        "没有接入账户或成交记录；买入比例均为条件预算，不表示已有持仓或当前下单。",
        "首次止盈时停止并取消尚未成交的加仓，后续各档按锁定的原始抄底股数执行。",
        "损失模型假设在参考止损价成交；跳空、滑点及收盘后执行可使实际损失超过模型。",
    ])
    return plan
