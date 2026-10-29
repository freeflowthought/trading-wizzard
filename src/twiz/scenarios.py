"""Join event main cases, observed transmission and explicit portfolio actions."""

import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from .config import load_prediction_markets_config, validate_event_threshold
from .ingest.prediction_discovery import DISCOVERY_VERSION, discovery_contract
from .sessions import expected_complete_session, is_us_equity_session

SCENARIO_BUDGET_FRACTIONS = {
    "defensive": {"initial": .1, "post_event": .35},
    "stress": {"initial": 0., "post_event": .2},
    "repair_strong": {"initial": .2, "post_event": .5},
    "repair_weak": {"initial": .1, "post_event": .35},
    "unknown": {"initial": 0., "post_event": .2},
    "event_support_with_rates": {"initial": .2, "post_event": .35},
    "regional": {"initial": .1, "post_event": .2},
}


def _allocation_budget(policy):
    policy = {} if policy is None else policy
    if not isinstance(policy, dict):
        raise ValueError("Scenario allocation policy must be a mapping.")
    cap = _num(policy.get("max_new_capital_pct", 10))
    supplied = policy.get("scenario_budget_fractions", {})
    if (cap is None or not 0 < cap <= 100 or not isinstance(supplied, dict)
            or set(supplied)-set(SCENARIO_BUDGET_FRACTIONS)-{"before_event"}):
        raise ValueError("Invalid scenario capital ceiling or fraction mapping.")
    fractions = {}
    for name, default in SCENARIO_BUDGET_FRACTIONS.items():
        raw = supplied.get(name, default)
        if not isinstance(raw, dict) or set(raw) != {"initial", "post_event"}:
            raise ValueError("Scenario fractions require initial and post_event: " + name)
        initial, after = (_num(raw[key]) for key in ("initial", "post_event"))
        if initial is None or after is None or not 0 <= initial <= after <= 1:
            raise ValueError("Scenario fractions must satisfy 0 <= initial <= post_event <= 1.")
        fractions[name] = {"initial": initial, "post_event": after}
    before = _num(supplied.get("before_event", .2))
    if before is None or not 0 <= before <= 1:
        raise ValueError("Before-event fraction must be finite in [0, 1].")
    fractions["before_event"] = before
    return cap, fractions


def _num(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _time(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp.astimezone(timezone.utc) if stamp.tzinfo else None
    except (ValueError, TypeError):
        return None


def _event(row, report_date, now, contracts, threshold):
    contract = (discovery_contract(row) if row.get("selection_mode") == DISCOVERY_VERSION else
                contracts.get(str(row.get("id")))) or {}
    if (row.get("source") != "Polymarket Gamma" or not contract.get("driver")
            or row.get("driver") != contract["driver"]
            or row.get("question") != contract.get("expected_question")):
        return None
    yes, no = (_num(row.get(key)) for key in ("yes_probability", "no_probability"))
    updated, fetched, end = (_time(row.get(key)) for key in (
        "market_updated_at", "fetched_at", "end_date"))
    liquidity, spread = (_num(row.get(key)) for key in ("liquidity_usd", "spread"))
    if (row.get("status") not in {"yes_leaning", "no_leaning", "uncertain"}
            or any(v is None for v in (yes, no, updated, fetched, end, liquidity, spread))
            or not 0 <= yes <= 1 or not 0 <= no <= 1 or abs(yes+no-1) > 0.02+1e-12
            or updated > fetched or fetched > now or (now-updated).total_seconds() > 86400
            or end <= now or liquidity < 1000 or not 0 <= spread <= 0.10+1e-12
            or fetched.astimezone(ZoneInfo("America/Toronto")).date().isoformat() != report_date
            or not row.get("question")):
        return None
    metadata = row.get("contract_metadata") or {}
    if not isinstance(metadata, dict):
        return None
    created = _time(metadata.get("created_at"))
    if created and created > fetched:
        return None
    yes_main, no_main = yes >= threshold, no >= threshold
    main = "Yes" if yes_main and not no_main else "No" if no_main and not yes_main else "分支"
    return {"id": str(row["id"]), "driver": row.get("driver"),
            "label": contract.get("label") or row.get("label"),
            "question": row["question"], "yes_probability": yes, "no_probability": no,
            "main_case": main, "main_probability": max(yes, no),
            "end_date": row["end_date"], "url": row.get("url"),
            "market_updated_at": row["market_updated_at"],
            "selection_mode": row.get("selection_mode", "manual_watchlist"),
            "impact_channel": contract.get("impact_channel"),
            "importance_rank": contract.get("importance_rank"),
            "selection_reason": contract.get("selection_reason"),
            "settlement_scope_note": contract.get("settlement_scope_note"),
            "contract_metadata": row.get("contract_metadata", {})}


def _series(rows, ids, report_date):
    as_of = date.fromisoformat(report_date)
    source_by_id = {"DGS10": {"FRED", "FRED public CSV"},
                    "UST10Y": {"US Treasury"},
                    "BAMLH0A0HYM2": {"FRED", "FRED public CSV"},
                    "DCOILBRENTEU": {"FRED", "FRED public CSV"},
                    "USDJPY": {"ECB reference FX"}, "USDKRW": {"ECB reference FX"}}
    for key in ids:
        for row in rows:
            if (row.get("series_id") != key
                    or row.get("source") not in source_by_id.get(key, set())):
                continue
            values = {}
            for obs in [*row.get("history", []), row]:
                try:
                    day = date.fromisoformat(str(obs.get("date"))[:10])
                except ValueError:
                    continue
                value = _num(obs.get("value"))
                if day <= as_of and value is not None:
                    values[day.isoformat()] = value
            if not values:
                continue
            dates = sorted(values)
            last = dates[-1]
            if (as_of-date.fromisoformat(last)).days > 7:
                continue
            return {"series_id": key, "value": values[last], "date": last,
                    "source": row.get("source"), "url": row.get("url"),
                    "start_date_5": dates[-6] if len(dates) >= 6 else None,
                    "change_5": values[last]-values[dates[-6]] if len(dates) >= 6 else None,
                    "change_20": values[last]-values[dates[-21]] if len(dates) >= 21 else None}
    return None


def _rotation(market, report_date):
    expected = expected_complete_session(report_date)
    benchmark = {r.get("date"): r for r in market.get("SPY", [])
                 if r.get("source") == "yfinance" and r.get("session_complete") is not False
                 and str(r.get("date", "")) <= expected and _num(r.get("close"))
                 and float(r["close"]) > 0}
    dates = sorted(benchmark)
    if len(dates) < 21 or dates[-1] != expected:
        return []
    start, end = dates[-21], dates[-1]
    base = float(benchmark[end]["close"])/float(benchmark[start]["close"])-1
    labels = {"QQQ": "科技", "SMH": "半导体", "IGV": "软件", "DIA": "道指传统股",
              "XLF": "金融", "XLU": "公用事业", "XLV": "医疗", "XLE": "能源",
              "XLP": "必需消费", "ITA": "航空军工", "TLT": "长期美债", "GLD": "黄金"}
    output = []
    for ticker, name in labels.items():
        rows = {r.get("date"): r for r in market.get(ticker, [])
                if r.get("source") == "yfinance" and r.get("session_complete") is not False}
        if start not in rows or end not in rows:
            continue
        first, last = (_num(rows[d].get("close")) for d in (start, end))
        if first and last and first > 0 and last > 0:
            ret = last/first-1
            output.append({"ticker": ticker, "name": name, "start_date": start,
                           "end_date": end, "return_20d_pct": ret*100,
                           "vs_spy_pp": (ret-base)*100,
                           "basis": "同区间价格相对强弱，不冒充实际资金流"})
    return sorted(output, key=lambda r: -r["vs_spy_pp"])


def _semiconductor_transmission(market, report_date, now=None):
    """Six exact completed-session closes define a five-session relative return."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Semiconductor analysis timestamps require an explicit timezone.")
    day = date.fromisoformat(expected_complete_session(report_date, now=now))
    dates = []
    while len(dates) < 6:
        if is_us_equity_session(day):
            dates.append(day.isoformat())
        day -= timedelta(days=1)
    start, end = dates[-1], dates[0]
    returns, bases = {}, set()
    for ticker in ("SPY", "SMH"):
        rows, duplicates = {}, set()
        for row in market.get(ticker, []):
            stamp, close = row.get("date"), _num(row.get("close"))
            if (stamp not in dates or row.get("source") != "yfinance"
                    or close is None or close <= 0):
                continue
            completed = row.get("session_complete") is True
            if "session_complete" not in row:
                fetched = _time(row.get("fetched_at"))
                completed = bool(fetched and fetched <= now
                                 and expected_complete_session(stamp, now=fetched) == stamp)
            if not completed:
                continue
            if stamp in rows:
                duplicates.add(stamp)
            rows[stamp] = row
        bases.update(row.get("price_basis") for row in rows.values())
        if (duplicates or any(stamp not in rows for stamp in dates)
                or len(bases) != 1 or not next(iter(bases), None)):
            return {"ticker": "SMH", "status": "unavailable", "start_date": start,
                    "end_date": end, "return_5d_pct": None, "vs_spy_5d_pp": None,
                    "weak": None, "reason": "缺少同区间完整、来源及复权口径一致的SPY/SMH日收盘。"}
        returns[ticker] = float(rows[end]["close"])/float(rows[start]["close"])-1
    relative = (returns["SMH"]-returns["SPY"])*100
    return {"ticker": "SMH", "status": "observed", "start_date": start, "end_date": end,
            "return_5d_pct": returns["SMH"]*100, "spy_return_5d_pct": returns["SPY"]*100,
            "vs_spy_5d_pp": relative, "weak": relative <= -2+1e-12,
            "reason": "同区间5个已完成美股交易日；落后SPY至少2个百分点为风控观察阈值。"}


def _regional_context(by_driver, japan_fx, korea_fx, semiconductor, regional_caps=(1., 2.)):
    rows = []
    for region, driver, priority, fx in (("Japan", "boj_hold", "secondary", japan_fx),
                                        ("Korea", "bok_hike_25", "lower", korea_fx)):
        event = by_driver.get(driver)
        main = event["main_case"] if event else "unknown"
        change, strength, pressure = None, None, None
        alignment, alignment_reason = "unavailable", "缺少六个带日期的汇率观察值。"
        if fx and fx["change_5"] is not None:
            old = fx["value"]-fx["change_5"]
            if old > 0 and fx["value"] > 0:
                change = (fx["value"]/old-1)*100
                strength = (old/fx["value"]-1)*100
                fx["change_5_pct"] = change
                try:
                    first, last, end = (date.fromisoformat(stamp) for stamp in (
                        fx.get("start_date_5"), fx.get("date"), semiconductor.get("end_date")))
                    aligned = 0 <= (last-first).days <= 10 and 0 <= (end-last).days <= 3
                except (TypeError, ValueError):
                    aligned = False
                alignment = "aligned" if aligned else "misaligned"
                alignment_reason = ("汇率六次观察跨不超过10日，末值与半导体末日相距不超过3日。"
                                    if aligned else "汇率窗口稀疏、滞后或晚于半导体末日；"
                                    "不授予区域风控权限。")
                if aligned:
                    pressure = strength >= 2-1e-12 if region == "Japan" else change >= 3-1e-12
        transmitted = pressure is True and semiconductor["weak"] is True
        if region == "Japan":
            semantics = {"Yes": "该合约定义的10月利率不变为主情景。",
                         "No": "该利率不变结果不是主情景；No不自动等于加息。",
                         "分支": "日本利率不变与非不变结果仍保留分支。"}
            reason = ("日元近5观察间隔升值至少2%，且SMH近5美股交易日落后SPY至少2pp；"
                      "出现同向压力信号，不能据此证明因果。" if transmitted else
                      "日元与半导体未同时满足传导门槛；日本事件概率不单独改变美国主判断。")
            action = (f"区域传导上限为首批{regional_caps[0]:g}%、事件后合计{regional_caps[1]:g}%；"
                      "实际额度仍服从更低的美国门槛及价格确认。"
                      if transmitted else "保留融资/汇率观察，不单凭日本合约削减或放宽美股预算。")
        else:
            semantics = {"Yes": "按该合约规则判为25bp加息为主情景，包含其舍入定义。",
                         "No": "该25bp加息结果不是主情景；No不自动等于维持或降息。",
                         "分支": "该25bp加息结果与其补集仍保留分支。"}
            reason = ("美元兑韩元近5观察间隔上升至少3%，且SMH近5美股交易日落后SPY至少2pp；"
                      "记录存储/半导体链风险，不能推广为美股整体转空。" if transmitted else
                      "韩国事件以存储/半导体链为主；尚无同时满足门槛的汇率与板块价格传导。")
            action = ("提高半导体及QQQ相关敞口的复核优先级；不改变美股整体方向或预算上限。"
                      if transmitted else "维持美国主判断与预算；仅保留韩国及存储链观察。")
        if alignment != "aligned":
            reason += alignment_reason
        rows.append({"region": region, "driver": driver, "priority": priority,
                     "event_id": event.get("id") if event else None,
                     "event_main_case": main, "event_quality": "usable" if event else "unavailable",
                     "event_semantics": semantics.get(
                         main, "合约未通过身份、日期或报价质量校验；主情景未知。"),
                     "event_probability": event["main_probability"] if event else None,
                     "event_yes_probability": event["yes_probability"] if event else None,
                     "event_no_probability": event["no_probability"] if event else None,
                     "fx": {"series_id": "USDJPY" if region == "Japan" else "USDKRW",
                            "date": fx["date"] if fx else None,
                            "start_date_5": fx.get("start_date_5") if fx else None,
                            "alignment": alignment, "alignment_reason": alignment_reason,
                            "change_5_pct": change,
                            "local_currency_strength_5_pct": strength, "pressure": pressure},
                     "semiconductor": semiconductor, "transmission_confirmed": transmitted,
                     "qqq_priority_risk": transmitted, "reason": reason, "action": action})
    return rows


def build_scenario_context(
    report_date: str, macro_rows: List[Dict[str, Any]], prediction_rows: List[Dict[str, Any]],
    market_data: Dict[str, List[Dict[str, Any]]], global_context: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
    contracts: Optional[List[Dict[str, Any]]] = None,
    event_threshold: Optional[float] = None,
    allocation_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    capital_cap, budget_fractions = _allocation_budget(allocation_policy)

    def budget(name):
        weights = budget_fractions[name]
        return capital_cap * weights["initial"], capital_cap * weights["post_event"]

    now = now or datetime.now(timezone.utc)
    threshold = validate_event_threshold(event_threshold if event_threshold is not None else
                                         load_prediction_markets_config()["main_case_threshold"])
    if now.tzinfo is None:
        raise ValueError("Scenario timestamps require an explicit timezone.")
    registered = {str(row.get("id")): row for row in (
        contracts if contracts is not None else load_prediction_markets_config().get("markets", [])
    )}
    events = [event for row in prediction_rows
              if (event := _event(row, report_date, now, registered, threshold))]
    by_driver = {}
    for row in sorted(events, key=lambda event: event["end_date"]):
        by_driver.setdefault(row["driver"], row)
    automatic = any(event.get("selection_mode") == DISCOVERY_VERSION for event in events)
    fed, peace = by_driver.get("fed_hold"), by_driver.get("near_term_ceasefire")
    yield10 = _series(macro_rows, ("DGS10", "UST10Y"), report_date)
    credit = _series(macro_rows, ("BAMLH0A0HYM2",), report_date)
    oil = _series(macro_rows, ("DCOILBRENTEU",), report_date)
    if oil and (date.fromisoformat(report_date)-date.fromisoformat(oil["date"])).days > 3:
        oil = None  # A lagged spot series cannot override today's geopolitical transmission.
    japan_fx = _series(macro_rows, ("USDJPY",), report_date)
    korea_fx = _series(macro_rows, ("USDKRW",), report_date)
    rotation = _rotation(market_data, report_date)
    rate_relief = bool(yield10 and yield10["change_5"] is not None and yield10["change_5"] < 0)
    # Five basis points is a disclosed stability tolerance, not a fitted success threshold.
    credit_ok = bool(credit and credit["change_5"] is not None and credit["change_5"] <= 0.05)
    oil_stress = bool(oil and oil["change_5"] is not None
                      and oil["change_5"] > max(3, oil["value"]*0.05))
    event_stress = bool(peace and peace["main_case"] == "No")
    policy_uncertain = not fed or fed["main_case"] != "Yes"
    event_support = bool(fed and fed["main_case"] == "Yes" and not event_stress)
    if automatic:
        near = [event for event in events
                if 0 < (_time(event["end_date"])-now).total_seconds() <= 21*86400]
        event_stress = any(event["driver"] in {"geopolitical_risk", "fed_hike"}
                           and event["main_case"] == "Yes" for event in near)
        event_support = any(event["driver"] in {"fed_hold", "fed_cut"}
                            and event["main_case"] == "Yes" for event in near) and not event_stress
        policy_uncertain = not event_support
    state = "defensive"
    initial, after = budget("defensive")
    observed_stress = bool(yield10 and credit
                           and yield10["change_5"] is not None
                           and credit["change_5"] is not None
                           and yield10["change_5"] >= 0 and credit["change_5"] > 0.05)
    if event_stress or oil_stress or observed_stress:
        state = "stress_bias"
        initial, after = budget("stress")
        headline = "短线偏空：风险传导尚未解除，取消近端抄底，等待更深区域止跌。"
    elif event_support and rate_relief and credit_ok:
        state = "repair_bias"
        initial, after = budget("repair_strong")
        headline = "短线偏多看修复：事件主情景与利率/信用缓和同向，回踩确认可试仓，当前不追涨。"
    elif event_support:
        headline = "短线以反弹修复为主，但宏观传导未完全确认；轻仓等回踩，资金留给深跌。"
        initial, after = budget("event_support_with_rates" if rate_relief else "defensive")
    elif rate_relief and credit_ok:
        state = "repair_bias"
        initial, after = budget("repair_weak")
        headline = ("短线偏修复：利率与信用暂支持修复，但事件定价未确认近端核心政策利好；"
                    "回踩确认后仅小仓评估，不追涨。")
    elif policy_uncertain:
        state = "unknown"
        initial, after = budget("unknown")
        headline = "短线先防守：关键事件未形成可核验主情景，暂不新增，保留深跌后的确认方案。"
    else:
        headline = "短线偏防守：事件有分歧，优先等待价格收复，暂不把反弹升级为反转。"
    semiconductor = _semiconductor_transmission(market_data, report_date, now=now)
    regional_caps = budget("regional")
    regional = _regional_context(by_driver, japan_fx, korea_fx, semiconductor, regional_caps)
    if regional[0]["transmission_confirmed"]:
        initial = min(initial, regional_caps[0])
        after = min(after, regional_caps[1])
        headline += f"日元与半导体同时触及压力门槛，首批再压至{initial:g}%以内。"
    if regional[1]["transmission_confirmed"]:
        headline += "韩国相关压力先按半导体链风险观察，不直接转空美股整体。"
    chain = []
    for event in (events if automatic else (fed, peace)):
        if event:
            chain.append(f"{event['label']}：{event['main_case']} "
                         f"{event['main_probability']*100:.1f}%，按{threshold*100:g}%门槛选事件主情景")
            if event.get("settlement_scope_note"):
                chain.append(event["settlement_scope_note"])
    for row, label in ((yield10, "10年美债"), (credit, "高收益债利差"), (oil, "Brent")):
        if row:
            drift = (f"，5个观察间隔变化{row['change_5']:+.2f}"
                     if row["change_5"] is not None else "")
            unit = "%" if row is not oil else "美元/桶"
            chain.append(f"{label} {row['value']:.2f}{unit}（{row['date']}）{drift}")
    if fed and fed["main_case"] == "Yes":
        chain.append("利率不变已是主预期；单纯兑现不构成追涨利好，要看收益率与信用继续响应")
    if rotation:
        chain.append(f"板块选择（近20个交易日）：{rotation[0]['name']}相对领先；"
                     f"{rotation[-1]['name']}相对落后，反弹仓先看强者，弱者须额外确认")
    for row, label in ((japan_fx, "美元兑日元"), (korea_fx, "美元兑韩元")):
        if row:
            move = row.get("change_5_pct")
            chain.append(f"{label} {row['value']:.2f}（{row['date']}，ECB日参考）"
                         + (f"，近5个观察间隔{move:+.2f}%" if move is not None else ""))
    branches = [
        {"name": "主情景",
         "trigger": f"有效事件概率≥{threshold*100:g}%的结果为基础，结合已观察传导",
         "market_implication": headline, "action": f"价格确认后首批上限{initial:g}%，事件后再评估",
         "probability": None},
        {"name": "转强后加仓",
         "trigger": "决议/关键财报落地，收益率回落、信用不扩张，结构收复后回踩守住",
         "market_implication": "从反弹修复升级到结构改善",
         "action": f"先提高到合计{after:g}%，再验证趋势；总额不超过{capital_cap:g}%",
         "probability": None},
        {"name": "反向风险",
         "trigger": "关键事件主情景翻转，或油价/收益率上冲且信用恶化，支撑确认失效",
         "market_implication": "修复预期撤回，切换深跌压力情景",
         "action": "取消未成交加仓，退出失败交易仓；极限区另等确认",
         "probability": None},
    ]
    return {"date": report_date, "state": state, "headline": headline,
            "event_threshold": threshold, "causal_chain": chain,
            "event_selection": {"mode": "automatic" if automatic else "manual_watchlist",
                                "selected_market_ids": [event["id"] for event in events],
                                "legacy_required_iran_contract": False,
                                "rationales": [event.get("selection_reason") for event in events
                                               if event.get("selection_reason")]},
            "events": events, "branches": branches, "rotation": rotation,
            "event_priority": [
                {"driver": "fed_hold", "priority": "high", "authority": "US_primary",
                 "channel": "美联储及已观察美国收益率/信用主导美股整体方向与预算。"},
                {"driver": "boj_hold", "priority": "secondary",
                 "authority": "conditional_funding_fx",
                 "channel": "日本融资/汇率须与半导体相对走弱共同传导，才收紧广义预算。"},
                {"driver": "bok_hike_25", "priority": "lower", "authority": "sector_memory_chain",
                 "channel": "韩国主要影响存储/半导体观察；不单独改变美国整体方向及预算。"}],
            "regional_transmission": regional,
            "macro_features": {"yield10": yield10, "credit": credit, "oil": oil,
                               "japan_fx": japan_fx, "korea_fx": korea_fx},
            "policy": {"initial_cap_pct": initial, "post_event_cap_pct": after,
                       "max_new_capital_pct": capital_cap, "reason": headline,
                       "budget_fractions": budget_fractions,
                       "event_main_case_confirmed": bool(event_support and state != "stress_bias"),
                       "before_event_cap_pct": min(capital_cap * budget_fractions["before_event"],
                                                   initial),
                       "preferred_entry_tier": ("support" if state == "repair_bias" else
                                                "deep" if state == "defensive" else "extreme")},
            "global_context": (global_context if global_context
                               and global_context.get("as_of") == report_date else {}),
            "limitations": [f"本次事件主情景门槛为{threshold*100:g}%，可参与历史优化；"
                             "不是QQQ上涨概率。",
                            "不同合约期限和结算条件分别处理，不平均或相乘成股价概率。",
                            "信用5日变动5bp以内按稳定处理；区域传导阈值为未拟合的风控启发式。",
                            "日本门槛：日元5观察间隔升值≥2%且SMH近5交易日落后SPY≥2pp；"
                            "韩国门槛：USD/KRW上升≥3%且半导体同样走弱，仅提示板块风险。",
                            "区域共振门槛不证明因果关系或实际资金流，也不单独触发买入。",
                            "区域权限另要求汇率六次观察跨≤10日，末值不晚于且距半导体末日≤3日。",
                            "仓位映射为明确风险政策，尚无验证后的预测胜率；历史宏观资料可能修订。"]}
