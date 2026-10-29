"""Current-data structure hypotheses, not reconstructed author targets or trained odds."""
import math
from datetime import date, timedelta
from statistics import mean
from typing import Any, Dict, List, Optional

from .sessions import expected_complete_session, is_us_equity_session

PARAMETERS = {"range_sessions": 63, "pivot_left": 3, "pivot_right": 3,
              "confirmation_sessions": 5, "atr_period": 14, "cluster_atr": 1.0,
              "zone_buffer_atr": 0.25, "old_base_sessions": 126,
              "retest_buffer_atr": 0.25, "support_repeat_span": 5,
              "support_rebound_atr": 2.0, "corroboration_span": 20,
              "working_span": 10, "recovery_spacing_atr": 0.25}


def validate_structure_parameters(parameters=None):
    """Return an explicit parameter snapshot; never silently ignore research knobs."""
    supplied = {} if parameters is None else parameters
    if not isinstance(supplied, dict) or set(supplied)-set(PARAMETERS):
        raise ValueError("Unknown structure parameters or invalid parameter mapping.")
    result = {**PARAMETERS, **supplied}
    integer_keys = {"range_sessions", "pivot_left", "pivot_right", "confirmation_sessions",
                    "atr_period", "old_base_sessions", "support_repeat_span",
                    "corroboration_span", "working_span"}
    for key, value in result.items():
        number = _num(value)
        if (number is None or number <= 0 or number > 2520
                or key in integer_keys and type(value) is not int):
            raise ValueError(f"Invalid structure parameter: {key}.")
    return result


def _num(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _clean(rows, cutoff):
    valid, duplicates = {}, set()
    for row in rows:
        try:
            day = date.fromisoformat(str(row.get("date"))[:10]).isoformat()
        except ValueError:
            continue
        values = {key: _num(row.get(key)) for key in ("open", "high", "low", "close", "volume")}
        source = row.get("source")
        chart_trace = (source != "Yahoo chart" or (
            all(isinstance(row.get(key), str) and row[key] for key in (
                "raw_body_file", "raw_json_pointer", "raw_timestamp_pointer", "fetched_at"))
            and type(row.get("raw_array_index")) is int and row["raw_array_index"] >= 0))
        if (day > cutoff or not is_us_equity_session(date.fromisoformat(day))
                or source not in {"yfinance", "Yahoo chart"} or not chart_trace
                or row.get("session_complete") is False
                or any(v is None for v in values.values()) or values["volume"] < 0
                or min(values[k] for k in ("open", "high", "low", "close")) <= 0
                or not values["low"] <= min(values["open"], values["close"])
                <= max(values["open"], values["close"]) <= values["high"]):
            continue
        if day in valid:
            duplicates.add(day)
        valid[day] = {**row, **values, "date": day}
    return [valid[day] for day in sorted(valid) if day not in duplicates]


def _sma(rows, window):
    return mean(r["close"] for r in rows[-window:]) if len(rows) >= window else None


def _weekly(rows, cutoff):
    weeks = {}
    for row in rows:
        day = date.fromisoformat(row["date"])
        end = day + timedelta(days=4-day.weekday())
        while not is_us_equity_session(end):
            end -= timedelta(days=1)
        if end.isoformat() <= cutoff and day == end:
            weeks[end.isoformat()] = {"date": row["date"], "close": row["close"]}
    return [weeks[key] for key in sorted(weeks)]


def _atr(rows, period=14):
    if len(rows) < period+1:
        return None
    ranges = [max(cur["high"]-cur["low"], abs(cur["high"]-prev["close"]),
                  abs(cur["low"]-prev["close"]))
              for prev, cur in zip(rows[-period-1:-1], rows[-period:])]
    return mean(ranges)


def _pivots(rows, left=3, right=3):
    pivots = []
    for i in range(left, len(rows)-right):
        window = rows[i-left:i+right+1]
        for kind, field, compare in (("low", "low", min), ("high", "high", max)):
            value = rows[i][field]
            if (value == compare(r[field] for r in window)
                    and sum(r[field] == value for r in window) == 1):
                pivots.append({"family": "price_structure", "label": f"confirmed_pivot_{kind}",
                               "price": value, "date": rows[i]["date"], "kind": kind,
                               "confirmed_at": rows[i+right]["date"], "position": i})
    return pivots


def _clusters(origins, atr, anchor_size, parameters=None):
    params = validate_structure_parameters(parameters)
    groups = []
    for item in sorted(origins, key=lambda r: r["price"]):
        if groups and item["price"]-min(r["price"] for r in groups[-1]) <= (
                atr*params["cluster_atr"]):
            groups[-1].append(item)
        else:
            groups.append([item])
    zones = []
    for members in groups:
        families = sorted({r["family"] for r in members})
        price_origins = [r for r in members if r["family"] == "price_structure"]
        pivots = [r for r in price_origins if r["label"] == "confirmed_pivot_low"]
        distinct = {r["date"] for r in pivots}
        span = max((r["position"] for r in pivots), default=0)-min(
            (r["position"] for r in pivots), default=0)
        corroborated = bool(price_origins and "moving_average" in families
                            or len(distinct) >= 3 and span >= params["corroboration_span"])
        working = bool(price_origins and "projection" in families
                       or len(distinct) >= 2 and span >= params["working_span"])
        family_means = [mean(r["price"] for r in members if r["family"] == family)
                        for family in families]
        strength = ("corroborated" if corroborated else
                    "working_hypothesis" if working else "single_origin")
        zones.append({"lower": min(r["price"] for r in members)-atr*params["zone_buffer_atr"],
                      "upper": max(r["price"] for r in members)+atr*params["zone_buffer_atr"],
                      "center": mean(family_means), "origins": members, "families": families,
                      "independent_families": [f for f in families if f != "projection"],
                      "credible": corroborated or working,
                      "strength": strength,
                      "old_base": any(
                          r.get("position", anchor_size) < anchor_size-params["old_base_sessions"]
                          for r in pivots),
                      "basis": "已确认价格结构/长期均线聚类；ATR边界为本系统工程规则"})
    return zones


def _zone_signal(zone, recent, atr, tier, retest_buffer_atr=0.25):
    signal = {"state": "wait", "zone": tier, "probability": None, "reclaim_date": None,
              "retest_date": None, "swing_low": None, "buy_signal": False}
    reclaimed = None
    for row in recent:
        if row["close"] < zone["lower"]:
            reclaimed = None
            signal.update(state="failed", reclaim_date=None, retest_date=None, buy_signal=False)
        if reclaimed and row["date"] > reclaimed["date"]:
            if (zone["lower"] <= row["low"] <= zone["upper"]+atr*retest_buffer_atr
                    and row["close"] >= zone["upper"] and row["close"] > reclaimed["close"]):
                signal.update(state="confirmed_retest", retest_date=row["date"],
                              swing_low=min(reclaimed["low"], row["low"]), buy_signal=True)
        elif row["low"] <= zone["upper"] and row["close"] >= zone["upper"]:
            state = "sweep_reclaim" if row["low"] < zone["lower"] else "zone_hold"
            reclaimed = row
            signal.update(state=state, reclaim_date=row["date"], swing_low=row["low"])
    return signal


def _one_structure(ticker, rows, report_date, parameters=None, pivot_lookback_sessions=None):
    params = validate_structure_parameters(parameters)
    if (pivot_lookback_sessions is not None and (
            type(pivot_lookback_sessions) is not int or not 20 <= pivot_lookback_sessions <= 2520)):
        raise ValueError("Pivot lookback must be an integer between 20 and 2520 sessions.")
    rows = _clean(rows, report_date)
    result = {"status": "unavailable", "price_date": None, "close": None, "features": {},
              "zones": {"support": None, "deep": None, "extreme": None}, "structural_floor": None,
              "recovery_levels": [], "projection": {"status": "unavailable", "price": None},
              "signal": {"state": "wait", "probability": None, "buy_signal": False},
              "signals": {},
              "evidence": [], "limitations": [
                  "20MV及九转公式未在样本中定义，不把它们伪装成已复刻指标。",
                  "价格结构是流动性观察代理，不能证明机构止损单、吸筹或订单区块。",
                  "参数是透明工程选择，未据历史收益优化；极限区不是保证底部。"]}
    if not rows or rows[-1]["date"] != expected_complete_session(report_date):
        result["limitations"].append("缺少最新完整、来源明确且有效的OHLCV交易日。")
        return result
    bases = {r.get("price_basis") for r in rows if r.get("price_basis")}
    if len(bases) > 1:
        result["limitations"].append("历史价格口径混合，暂停结构与资金方案。")
        return result
    basis = next(iter(bases), "Yahoo缓存未注明复权口径")
    latest = rows[-1]
    weekly = _weekly(rows, latest["date"])
    features = {f"sma{n}": _sma(rows, n) for n in (20, 50, 60, 200)}
    features.update({f"weekly_sma{n}": _sma(weekly, n) for n in (20, 30, 50)})
    features.update(atr14=_atr(rows), decision_atr=_atr(rows, params["atr_period"]),
                    completed_weeks=len(weekly), history_sessions=len(rows),
                    price_basis=basis, parameters=dict(params))
    result.update(price_date=latest["date"], close=latest["close"], features=features,
                  data_source=latest["source"], session_complete=True)
    features["atr_method"] = (f"{params['atr_period']}日True Range简单均值；"
                              "工程定义，不声称复刻作者指标")
    count = params["confirmation_sessions"]
    anchor, recent = rows[:-count], rows[-count:]
    if (len(anchor) < 200 or features["weekly_sma50"] is None
            or not _atr(anchor, params["atr_period"])):
        result.update(status="insufficient_history")
        result["limitations"].append("需要足够200日及50个已完成周线；深远区域资料不足时不补造。")
        return result
    atr = _atr(anchor, params["atr_period"])
    pivots = _pivots(anchor, params["pivot_left"], params["pivot_right"])
    if pivot_lookback_sessions is not None:
        pivots = [pivot for pivot in pivots
                  if pivot["position"] >= len(anchor)-pivot_lookback_sessions]
    features["pivot_lookback_sessions"] = pivot_lookback_sessions
    window = anchor[-params["range_sessions"]:]
    low, high = min(window, key=lambda r: r["low"]), max(window, key=lambda r: r["high"])
    recent_pivots = [p for p in pivots if p["kind"] == "low" and p["price"] < latest["close"]
                     and p["position"] >= len(anchor)-params["old_base_sessions"]]
    recent_groups = _clusters(recent_pivots, atr, len(anchor), params)
    repeated = []
    for group in recent_groups:
        points = group["origins"]
        left, right = min(p["position"] for p in points), max(p["position"] for p in points)
        if (len({p["date"] for p in points}) >= 2
                and right-left >= params["support_repeat_span"]
                and right >= len(anchor)-params["range_sessions"]
                and max(r["high"] for r in anchor[left:right+1]) >= (
                    group["center"]+params["support_rebound_atr"]*atr)):
            repeated.append(group)
    chosen = max(repeated, key=lambda z: max(p["date"] for p in z["origins"]), default=None)
    fallback = max(recent_pivots, key=lambda p: p["date"], default=None)
    floor = chosen["center"] if chosen else fallback["price"] if fallback else low["low"]
    floor_date = (max(p["date"] for p in chosen["origins"]) if chosen
                  else fallback["date"] if fallback else low["date"])
    origins = [p for p in pivots if p["kind"] == "low"]
    origins.append({"family": "price_structure", "label": "derived_recent_support",
                    "price": floor, "date": floor_date, "dependent_on": "confirmed price pivots"})
    anchor_weekly = _weekly(anchor, anchor[-1]["date"])
    origins += [{"family": "price_structure", "label": "prior_traded_weekly_close",
                 "price": r["close"], "date": r["date"]}
                for r in anchor_weekly if r["close"] < floor]
    for label, price in [("sma200", _sma(anchor, 200))] + [
        (f"weekly_sma{n}", _sma(anchor_weekly, n)) for n in (20, 30, 50)
    ]:
        if price and price < floor:
            origins.append({"family": "moving_average", "label": label, "price": price,
                            "date": anchor[-1]["date"]})
    projected = 2*floor-high["high"]
    projection = {"status": "conditional" if 0 < projected < floor else "unavailable",
                  "price": projected if 0 < projected < floor else None,
                  "formula": "2*derived_floor-prior_range_high", "floor": floor,
                  "range_high": high["high"], "range_high_date": high["date"],
                  "anchor_date": anchor[-1]["date"], "probability": None,
                  "basis": "破位后的区间高度压力情景，不是独立验证或必达目标"}
    if projection["price"]:
        origins.append({"family": "projection", "label": "conditional_range_height",
                        "price": projected, "date": anchor[-1]["date"],
                        "dependent_on": "same prior range high/floor"})
    clusters = _clusters(origins, atr, len(anchor), params)
    support = next(z for z in clusters
                   if any(r["label"] == "derived_recent_support" for r in z["origins"]))
    lower = sorted([z for z in clusters
                    if z["credible"] and 0 < z["lower"] < z["upper"] < support["lower"]],
                   key=lambda z: -z["upper"])
    deep = lower[0] if lower else None
    beyond_deep = [z for z in lower if deep is None or z["upper"] < deep["lower"]]
    extreme = next((z for z in beyond_deep if "projection" in z["families"]), None) or next(
        (z for z in beyond_deep if any(r["date"] < floor_date for r in z["origins"]
                                     if r["family"] == "price_structure")), None)
    annual_low = min(anchor[-252:], key=lambda r: r["low"])["low"]
    tail = next((z for z in lower if z["lower"] <= annual_low <= z["upper"]
                 and extreme and z["upper"] < extreme["lower"]), None)
    zones = {"support": support, "deep": deep, "extreme": extreme}
    recovery = [{"price": p["price"], "kind": "pivot_high", "source": "confirmed_price_pivot",
                 "date": p["date"], "confirmed_at": p["confirmed_at"]}
                for p in pivots if p["kind"] == "high"
                and p["position"] >= len(anchor)-params["old_base_sessions"]]
    recovery += [{"price": price, "kind": key, "source": "current_completed_daily_SMA",
                  "date": latest["date"]} for key, price in features.items()
                 if key in {"sma20", "sma50", "sma60", "sma200"} and price]
    recovery = sorted(recovery, key=lambda r: r["price"])
    unique = []
    for item in recovery:
        if not unique or item["price"]-unique[-1]["price"] > atr*params["recovery_spacing_atr"]:
            unique.append(item)
    signals = [_zone_signal(z, recent, atr, tier, params["retest_buffer_atr"])
               for tier, z in zones.items() if z]
    selected = next((s for s in signals if s["zone"] == "extreme"), None) or next(
        (s for s in signals if s["zone"] == "deep"),
        {"state": "wait", "probability": None, "buy_signal": False})
    features.update(zone_anchor_date=anchor[-1]["date"], recent_range={"low": low["low"],
                    "low_date": low["date"], "high": high["high"], "high_date": high["date"]},
                    range_sessions=params["range_sessions"],
                    confirmed_pivots=pivots,
                    structural_floor_method="最近有反弹间隔的重复支撑族；无重复则单枢轴观察",
                    structural_floor_source_date=floor_date)
    if params["range_sessions"] == 63:
        features["recent_range63"] = dict(features["recent_range"])
    features["confirmation_bars"] = [
        {key: row[key] for key in ("date", "open", "high", "low", "close")} for row in recent]
    result.update(status="ready", zones=zones, structural_floor=floor, projection=projection,
                  signals={signal["zone"]: signal for signal in signals},
                  recovery_levels=unique, signal=selected, tail_stress=tail,
                  intermediate_zones=[z for z in lower
                                      if z is not deep and z is not extreme and z is not tail],
                  evidence=[f"{ticker} {latest['date']}完整日线；"
                            f"区域依据冻结至{anchor[-1]['date']}，观察其后{count}日确认。",
                            "均线族只算一族；不同回看窗口不当作独立证据；投影与原区间有关联。"])
    if extreme is None:
        result["limitations"].append("没有足够可信的长期极限区域；不把投影单独包装成底部。")
    return result


def build_market_structure(report_date: str, market_data: Dict[str, List[Dict[str, Any]]],
                           macro_context: Optional[Dict[str, Any]] = None,
                           parameters: Optional[Dict[str, Any]] = None,
                           pivot_lookback_sessions: Optional[int] = None) -> Dict[str, Any]:
    date.fromisoformat(report_date)
    params = validate_structure_parameters(parameters)
    by_ticker = {ticker: _one_structure(ticker, market_data.get(ticker, []), report_date, params,
                                      pivot_lookback_sessions)
                 for ticker in ("QQQ", "SPY")}
    state = (macro_context or {}).get("state", "unknown")
    status = "ready" if any(r["status"] == "ready" for r in by_ticker.values()) else "unavailable"
    return {"date": report_date, "status": status, "by_ticker": by_ticker,
            "risk_context": {"macro_state": state, "probability": None,
                             "conditions": ["宏观与事件仅改变谨慎程度，不合成QQQ上涨概率。"]}}


def build_dynamic_action_plan(
    structure: Dict[str, Any], macro_gate: Dict[str, Any], policy: Dict[str, Any],
    events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """One shared campaign; an observed signal authorizes a budget, never claims an execution."""
    ticker = str(policy.get("primary_ticker", "QQQ")).upper()
    item = structure.get("by_ticker", {}).get(ticker, {})
    cap = _num(policy.get("max_new_capital_pct", 10))
    loss_cap = _num(policy.get("max_campaign_loss_pct", 0.5))
    plan = {"ticker": ticker, "date": structure.get("date"), "status": "unavailable",
            "headline": "当前等待，不追价；依据动态结构分阶段行动",
            "current_action": "买入0%，先等待有效结构。",
            "buy_now_pct": 0.0, "price_date": item.get("price_date"),
            "data_source": item.get("data_source"),
            "model_inputs": {"close": _num(item.get("close")), "source": item.get("data_source"),
                             "price_basis": item.get("features", {}).get("price_basis")},
            "selected_zone": None, "selected_tier": None, "reference_zones": item.get("zones", {}),
            "budget": {"ticker": ticker, "max_new_capital_pct": cap, "planned_capital_pct": 0.0,
                       "currently_enabled_pct": 0.0, "uncommitted_cash_pct": cap,
                       "estimated_weighted_entry": None}, "entries": [], "exits": [],
            "stop": {"price": None, "modeled_loss_pct": None}, "leverage": False,
            "other_assets": [{"ticker": other, "new_allocation_pct": 0} for other in
                             structure.get("by_ticker", {}) if other != ticker],
            "budget_scope": "仅本轮新增资金，共享账户上限；未启用预算不是账户实际现金。",
            "entry_reference": "入场价是确认后的估价参考，未接入账户或成交记录。",
            "trim_basis": "按首次止盈前锁定的本轮原始抄底股数分批，不按剩余股数或核心仓。",
            "limitations": ["首次止盈后取消未成交加仓；跳空与滑点可使损失超过模型。",
                            "深跌或极限区域不是保证底部；概率不由技术标签或事件概率直接相加。"]}

    def wait(reason):
        plan["current_action"] = "当前买入0%，不追价。" + reason
        plan["limitations"].append(reason)
        return plan

    if cap is None or not 0 < cap <= 100 or loss_cap is None or not 0 < loss_cap <= 100:
        return wait("本轮资金及模型损失限额须大于0且不超过账户100%。")
    if item.get("status") != "ready" or not item.get("features", {}).get(
            "decision_atr", item.get("features", {}).get("atr14")):
        return wait("缺少完整且口径一致的当前结构数据；宏观方向另行保留，价格操作暂缓。")
    if (item.get("data_source") not in {"yfinance", "Yahoo chart"}
            or item.get("session_complete") is not True
            or item.get("price_date") != expected_complete_session(plan["date"])
            or _num(item.get("close")) is None or item["close"] <= 0):
        return wait("价格来源、完整交易日或当前收盘值不符合要求，保留宏观观察但暂停价格操作。")
    features, zones = item["features"], item["zones"]
    preferred = policy.get("preferred_entry_tier")
    if preferred is not None and preferred not in {"support", "deep", "extreme"}:
        return wait("情景指定的入场区域无效，不自动切换为更积极的资金方案。")
    tier = preferred or (
        "extreme" if zones.get("extreme") else "deep" if zones.get("deep") else None)
    zone = zones.get(tier) if tier else None
    if not zone or not zone.get("credible"):
        plan["status"] = "waiting_structure"
        return wait("支撑区可继续观察；长期深跌/极限区尚无足够多源依据，等待收复及后续回踩。")
    plan.update(selected_zone=zone, selected_tier=tier)
    atr, close = features.get("decision_atr", features.get("atr14")), item["close"]
    numbers = {key: _num(policy.get(key, default)) for key, default in (
        ("stop_buffer_atr", 0.25), ("target_buffer_atr", 0.5),
        ("target_spacing_atr", 0.5), ("max_chase_atr", 0.5),
        ("provisional_cap_pct", 2.0))}
    if any(value is None or value <= 0 for value in numbers.values()):
        return wait("止损、止盈、追价或试仓参数须为正有限数。")
    if numbers["provisional_cap_pct"] > cap:
        return wait("结构尚待验证的试仓预算不得超过本轮配置的资金上限。")
    sequences = {key: policy.get(key, default) for key, default in (
        ("entry_offsets_atr", [1.0, 0.5, 0.0]),
        ("target_r_multiples", [1.0, 2.0, 3.0]),
        ("trim_fractions", [0.4, 0.3, 0.3]))}
    for key, raw in sequences.items():
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            return wait("入场、目标及止盈分批参数必须各有三档。")
        sequences[key] = [_num(value) for value in raw]
        if any(value is None for value in sequences[key]):
            return wait("入场、目标及止盈分批参数必须为有限数。")
    offsets, multiples, trims = (sequences[key] for key in (
        "entry_offsets_atr", "target_r_multiples", "trim_fractions"))
    if (not offsets[0] > offsets[1] > offsets[2] >= 0
            or not 0 < multiples[0] < multiples[1] < multiples[2]
            or any(not 0 < value <= 1 for value in trims) or abs(sum(trims)-1) > 1e-9):
        return wait("入场观察价须逐档降低、目标R须递增，止盈比例须合计100%。")
    plan["action_parameters"] = {**numbers, **sequences}
    plan["trim_basis"] = ("/".join(f"{value*100:g}" for value in trims)
                          + "按首次止盈前锁定的本轮原始抄底股数，不按剩余股数或核心仓。")
    fallback_signal = item.get("signal", {})
    signal = item.get("signals", {}).get(tier) or (
        fallback_signal if fallback_signal.get("zone") == tier else
        {"state": "wait", "zone": tier, "probability": None, "buy_signal": False})
    plan["model_inputs"]["preferred_entry_tier"] = tier
    sequence_confirmed = bool(signal.get("state") == "confirmed_retest"
                              and signal.get("zone") == tier
                              and signal.get("retest_date") == item["price_date"]
                              and signal.get("reclaim_date")
                              and signal["reclaim_date"] < signal["retest_date"])
    observed_lows = [r["price"] for r in zone["origins"] if r["family"] == "price_structure"]
    if signal.get("swing_low") and signal.get("zone") == tier:
        observed_lows.append(signal["swing_low"])
    if not observed_lows:
        return wait("区域缺少可观察的摆动低点，暂不生成止损和资金安排。")
    stop = min(min(observed_lows), zone["lower"])-atr*numbers["stop_buffer_atr"]
    # Dip observations run from nearest/highest to deeper; these are not accepted orders.
    prices = [zone["upper"]+atr*offset for offset in offsets]
    first = prices[0]
    if not 0 < stop < first:
        return wait("无法建立有效且为正的结构失效参考价。")
    if not prices[0] > prices[1] > prices[2] > stop:
        return wait("区域宽度与风险间距不一致，等待结构重新形成。")
    latest_bar = next((r for r in features.get("confirmation_bars", [])
                       if r.get("date") == item["price_date"]), {})
    held = tuple(_num(latest_bar.get(key)) for key in ("low", "high", "close"))
    confirmed = bool(sequence_confirmed and all(value is not None for value in held)
                     and held[0] <= first <= held[1] and held[2] >= first
                     and abs(held[2]-close) <= 0.011
                     and close <= first+atr*numbers["max_chase_atr"])
    costs = [max(close, first) if confirmed else first, prices[1], prices[2]]
    reference_cost = max(costs)
    r_unit, targets, required_targets, previous = reference_cost-stop, [], [], reference_cost
    cost_bps = _num(policy.get("target_round_trip_cost_bps", 10))
    if cost_bps is None or cost_bps < 0:
        return wait("止盈成本假设无效，不分配资金。")
    buffer = max(atr*numbers["target_buffer_atr"], max(close, reference_cost)*cost_bps/10_000)
    spacing = atr*numbers["target_spacing_atr"]
    target_floor = max(close, reference_cost)+buffer
    plan["target_constraints"] = {"current_close": close, "highest_entry_cost": reference_cost,
                                  "minimum_price_exclusive": target_floor, "min_spacing": spacing,
                                  "round_trip_cost_bps": cost_bps,
                                  "rr_basis": "最高首档成本到止损，保守口径"}
    recovery = [r for r in item["recovery_levels"] if _num(r.get("price")) is not None
                and r["price"] > 0 and r.get("date", "9999") <= item["price_date"]
                and r.get("confirmed_at", r.get("date", "9999")) <= item["price_date"]]
    for multiple in multiples:
        required_targets.append(max(target_floor, reference_cost+multiple*r_unit, previous+spacing))
        target = next((r for r in recovery if r["price"] > target_floor
                       and r["price"] >= reference_cost+multiple*r_unit
                       and r["price"]-previous >= spacing), None)
        targets.append(target)
        if target:
            previous = target["price"]
    if not targets[0]:
        return wait(f"没有同时高于现价/最高成本及波动缓冲、并达到{multiples[0]:g}R的来源目标；"
                    "暂不分配资金，不造新价。")
    strong_rr = all(targets) and tier == "extreme" and zone.get("strength") == "corroborated"
    campaign_cap = cap if strong_rr else min(cap, numbers["provisional_cap_pct"])
    raw_weights = policy.get("entry_fractions", [0.4, 0.35, 0.25])
    weights = [_num(w) for w in raw_weights] if isinstance(raw_weights, list) else []
    if (len(weights) != 3 or any(w is None or not 0 < w <= 1 for w in weights)
            or abs(sum(weights)-1) > 1e-9):
        return wait("分批预算参数无效。")
    cash = [campaign_cap*w for w in weights]
    modeled_loss = sum(pct*(1-stop/p) for pct, p in zip(cash, costs))
    scale = min(1, loss_cap/modeled_loss) if modeled_loss else 1
    cash = [math.floor(pct*scale*10_000)/10_000 for pct in cash]
    while sum(cash) > campaign_cap:
        largest = max(range(len(cash)), key=cash.__getitem__)
        cash[largest] = max(0., math.nextafter(cash[largest]-(sum(cash)-campaign_cap), 0.))
    if not sum(cash):
        return wait("模型风险预算不足以建立本轮试仓。")
    state = macro_gate.get("state", "unknown")
    unknown_cap = _num(policy.get("unknown_macro_initial_cap_pct", 2))
    if unknown_cap is None or not 0 <= unknown_cap <= cap:
        return wait("谨慎或宏观未知首批须在本轮资金上限内。")
    initial = cash[0] if state == "risk_on" else min(cash[0], unknown_cap)
    scenario_cap = _num(policy.get("initial_cap_pct"))
    if "initial_cap_pct" in policy and scenario_cap is None:
        return wait("情景首批预算参数无效，不自动扩大授权。")
    if scenario_cap is not None:
        initial = min(initial, max(0, scenario_cap))
    imminent = []
    for event in events or []:
        try:
            when = date.fromisoformat(str(event.get("date"))[:10])
            days = (when-date.fromisoformat(plan["date"])).days
        except (TypeError, ValueError):
            continue
        if (0 <= days <= 3 and (_num(event.get("importance")) or 0) >= 4
                and event.get("precision") not in {"approximate", "rule_based"}
                and (not event.get("tickers") or ticker in event["tickers"])):
            imminent.append(event.get("name", "关键事件"))
    probe_cap = _num(policy.get("before_event_cap_pct", 2))
    if probe_cap is None or not 0 <= probe_cap <= cap:
        return wait("事件前首批预算须在本轮配置的资金上限内。")
    event_probe = bool(imminent and policy.get("event_main_case_confirmed") is True
                       and probe_cap > 0)
    if event_probe:
        initial = min(initial, probe_cap)
    failed = close < stop or signal.get("state") == "failed"
    blocked = state == "risk_off" or failed or bool(imminent and not event_probe)
    initial = 0 if blocked else initial
    plan["event_context"] = {"imminent": imminent,
                             "main_case_confirmed": policy.get("event_main_case_confirmed") is True,
                             "pre_event_probe_enabled": event_probe and initial > 0,
                             "before_event_cap_pct": probe_cap if event_probe else 0,
                             "price_confirmation_required": True, "later_adds_pending": True}
    for i, (price, pct) in enumerate(zip(prices, cash), 1):
        price_type = "atr_zone_approach_watch" if i < 3 else "zone_upper_watch"
        origin = (f"所选区域上沿+{offsets[i-1]:g}ATR工程观察价，非独立历史枢轴"
                  if offsets[i-1] else "所选结构区域上沿（来源聚类及ATR缓冲）")
        trigger = (f"触及{price:.2f}后须日收盘守住或收复本档，再独立确认；"
                   "结构失效取消所有未成交批次。")
        if i > 1:
            trigger += "后续批次须新报告，不是预先下单。"
        plan["entries"].append({"batch": i, "price": price, "account_pct": pct,
                                "currently_enabled_pct": initial if i == 1 else 0,
                                "price_type": price_type, "price_origin": origin,
                                "modeled_entry_cost": costs[i-1],
                                "status": ("确认信号预算，不表示成交"
                                           if i == 1 and confirmed else "等待独立确认"
                                           if i == 1 else "更低一档观察，须新报告独立确认"),
                                "trigger": trigger})
    for i, (target, trim) in enumerate(zip(targets, trims), 1):
        plan["exits"].append({"stage": i, "price": target["price"] if target else None,
                              "trim_original_dip_pct": trim*100,
                              "reference_return_pct": ((target["price"]/reference_cost-1)*100
                                                       if target else None),
                              "reward_risk_multiple": ((target["price"]-reference_cost)/r_unit
                                                       if target else None),
                              "reference_kind": target["source"] if target else "unavailable",
                              "minimum_required_price": required_targets[i-1],
                              "source_date": target.get("date") if target else None,
                              "status": ("本轮实际持仓建立后分批兑现" if target else
                                         f"没有符合{multiples[i-1]:g}R及现价/ATR间隔的来源目标，留空重评")})
    funded = sum(cash)
    plan["budget"].update(planned_capital_pct=funded, currently_enabled_pct=initial,
                          uncommitted_cash_pct=cap-initial,
                          estimated_weighted_entry=funded/sum(
                              pct/p for pct, p in zip(cash, costs)))
    plan["stop"] = {"price": stop,
                    "modeled_loss_pct": sum(pct*(1-stop/p) for pct, p in zip(cash, costs)),
                    "currently_enabled_loss_pct": initial*(1-stop/costs[0]),
                    "trigger": (f"本系统以日收盘低于{stop:.2f}取消未成交加仓并重评本轮交易仓；"
                                "非核心仓。"),
                    "basis": (f"已观察区域/回踩低点减{numbers['stop_buffer_atr']:g}ATR；"
                              "不是作者给出的止损公式")}
    stance = "已有回踩确认" if confirmed else "等待，不追价"
    tier_name = {"support": "近端结构支撑", "deep": "深回调", "extreme": "极限"}[tier]
    prefix = "已观察收复及后续回踩，可评估" if confirmed else "等待收复及后续回踩，届时最多评估"
    plan.update(status="blocked" if blocked else "conditional_plan", signal=signal,
                buy_now_pct=initial if confirmed and not blocked else 0.0,
                headline=f"{ticker} 当前{stance}，按{tier_name}区域分阶段",
                current_action=(f"{prefix}首批{initial:g}%账户信号预算；"
                                f"现在买入信号{initial if confirmed else 0:g}%，不代表已成交。"))
    if imminent and not event_probe:
        plan["current_action"] = ("关键事件临近，暂缓新增买入；事件落地后重新确认："
                                  + "、".join(imminent))
    elif state == "risk_off":
        plan["current_action"] = "宏观风险门槛阻止新增买入；保留区域观察，不因触价自动入场。"
    elif failed:
        plan["current_action"] = "已跌破本轮区域失效参考，暂停旧方案；等待收复、回踩及重评。"
    elif event_probe:
        plan["current_action"] += "事件主情景只放开有限首批试仓，后续加仓仍待事件落地后重评。"
    if not strong_rr:
        plan["limitations"].append("支撑/深回调试仓、工作假设或更高回升位不足时，"
                                    f"本轮沿用配置的{numbers['provisional_cap_pct']:g}%试仓限额，"
                                    "不预设用满资金上限。")
    if policy.get("reason"):
        plan["limitations"].append(str(policy["reason"]))
    plan["post_event_cap_pct"] = _num(policy.get("post_event_cap_pct"))
    plan["limitations"].append("事件后额度只是后续预案，须新报告确认；多个情景共享本轮预算，不重复相加。")
    plan["entry_reference"] = ("由高到低分批观察；各档为区域上沿+"
                               + "/".join(f"{offset:g}" for offset in offsets) + "ATR工程参考。"
                               "各档须独立守住/收复；风险以最高首档成本计算，不假定已成交。")
    plan["spacing_assessment"] = {
        "atr14": features.get("atr14", atr), "decision_atr": atr,
        "first_entry_distance_atr": (close-first)/atr,
        "entry_step_atr": offsets[0]-offsets[1],
        "entry_steps_atr": [offsets[0]-offsets[1], offsets[1]-offsets[2]],
        "first_target_distance_atr": (targets[0]["price"]-close)/atr,
        "verdict": ("首档仅距现价不足0.25ATR，属于贴近现价的小回撤观察，不能称为深度抄底。"
                    if 0 <= close-first < atr*0.25 else
                    "入场档距为" + "/".join(f"{offsets[i]-offsets[i+1]:g}"
                                             for i in range(2)) + "ATR；仍须各档独立确认。"),
        "target_note": ("止盈须高于当前现价和最高成本缓冲、逐档至少间隔"
                        f"{numbers['target_spacing_atr']:g}ATR，并满足"
                        + "/".join(f"{multiple:g}" for multiple in multiples)
                        + "R；不足档位留空。")}
    return plan
