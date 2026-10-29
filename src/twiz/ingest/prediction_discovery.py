"""Bounded discovery and transparent equity-relevance screening, without fixed IDs.

Importance here is a documented screening heuristic, not an estimated stock beta.
Raw candidate responses and rejection reasons are retained for later review.
"""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DISCOVERY_VERSION = "equity-event-screen-v1"
SCREENING_PRIORITY_VERSION = "national-joint-congress-14d-v1"
DEFAULT_QUERIES = ("Fed decision", "US inflation", "US government", "tariff",
                   "ceasefire", "Bank of Japan", "Bank of Korea", "Nvidia earnings")


def classify_market(question: str, event_title: str = "") -> Optional[dict]:
    if not isinstance(question, str) or not question.strip():
        return None
    text = question + " " + str(event_title)
    if re.search(r"\b(?:say|mention|tweet)\b|and.{0,60}win.{0,15}(?:senate|house|election)",
                 question, re.I):
        return None
    negated = bool(re.search(
        r"\b(?:no|not|never|stop|halt|fewer|less|zero|avoid|prevent|lift|cancel|pause|cease|"
        r"undo|delay|postpone|reverse|withdraw|reject)\b|rule out",
        re.sub(r"no change", "", question, flags=re.I), re.I))
    def has(pattern):
        return bool(re.search(pattern, text, re.I))

    role, topic, channel, priority, reason = "observe_only", "", "", 9, ""
    if has(r"\b(fed|fomc|federal reserve|federal funds)\b"):
        topic, channel, priority = "rates", "us_rates", 0
        reason = "美国政策利率通过美债收益率、折现率及融资条件影响大盘。"
        rate_object = re.search(
            r"interest rates?|federal funds|policy rates?|target range|\brates\b", question, re.I)
        plain = re.sub(r"\bor (?:more|less)\b", "", question, flags=re.I)
        auxiliary = re.search(
            r"dissent|member|governor|chair|vote|voting|favor of|\bor\b|\bif\b|\bunless\b|\band\b",
            plain, re.I)
        if (not negated and not auxiliary and rate_object and re.search(
            r"no change|unchanged|hold.{0,20}rates|rates.{0,20}unchanged", question, re.I
        )):
            role = "fed_hold"
        elif not negated and not auxiliary and re.search(
            r"\b(?:increase|hike|raise)s?\s+(?:the\s+)?(?:interest rates?|rates|federal funds|"
            r"policy rates?|target range)\b", question, re.I
        ):
            role = "fed_hike"
        elif not negated and not auxiliary and re.search(
            r"\b(?:decrease|cut|lower)s?\s+(?:the\s+)?(?:interest rates?|rates|federal funds|"
            r"policy rates?|target range)\b", question, re.I
        ):
            role = "fed_cut"
        if role == "observe_only":
            priority, channel = 4, "us_rates_background"
    elif has(r"bank of japan|\bboj\b|japan.{0,25}interest rate"):
        topic, channel, priority = "regional_rates", "yen_funding", 5
        role = "boj_hold" if not negated and re.search(
            r"no change|unchanged|hold", question, re.I) else role
        reason = "日本为辅助因素，重点核验日元融资与半导体价格传导。"
    elif has(r"bank of korea|\bbok\b|korea.{0,25}interest rate"):
        topic, channel, priority = "regional_rates", "korea_memory", 6
        role = "bok_hike_25" if not negated and re.search(
            r"hike by 25|increase.{0,12}25", question, re.I) else role
        reason = "韩国对美股大盘权重较低，重点核验韩元及存储产业链。"
    elif has(r"\b(us|u\.s\.|united states|america)\b") and has(
        r"inflation|\bcpi\b|\bpce\b|unemployment|recession|\bgdp\b|treasury|debt ceiling"
    ):
        topic, channel, priority = "macro", "us_growth_inflation", 1
        reason = "美国增长或通胀预期可能改变利率、盈利与风险偏好。"
    elif has(r"tariff|government shutdown|debt ceiling") and has(
        r"\b(us|u\.s\.|united states|trump|congress|federal)\b"
    ):
        topic, channel, priority = "policy", "us_fiscal_trade", 1
        reason = "美国财政或贸易政策可能改变企业成本与风险溢价。"
    elif has(r"iran|israel|hormuz|middle east|russia|ukraine") and has(
        r"war|strike|attack|ceasefire|blockade|sanction"
    ):
        topic, channel = "geopolitics", "oil_geopolitics"
        priority = 2 if has(r"\b(us|u\.s\.|united states)\b") else 3
        reason = "冲突经油价、通胀及供应链传导；须逐条核对结算规则与期限。"
        if not negated and re.search(r"ceasefire", question, re.I) and not re.search(
            r"\bno\b|\bnot\b|\bend\b|break|fail|violate|cancel|expire|collapse|terminate",
            question, re.I
        ):
            role = "geopolitical_relief"
        elif (not negated and re.search(r"strike|attack|invad|blockade", question, re.I)
              and not re.search(
            r"\bno\b|\bnot\b|avoid|ceasefire|stop|prevent|\bend\b|withdraw|lift", question, re.I
        )):
            role = "geopolitical_risk"
    elif has(r"\boil\b|\bbrent\b|\bwti\b"):
        topic, channel, priority = "energy", "oil_inflation", 2
        reason = "能源价格影响通胀、利润率与板块轮动；价格合约不直接等于股指方向。"
    elif has(r"taiwan|semiconductor|nvidia|tsmc|samsung|sk hynix|asml|\bchip") and has(
        r"war|invad|blockade|export|earnings|revenue|ban|sanction"
    ):
        topic, channel, priority = "semiconductors", "chip_supply_earnings", 3
        reason = "芯片供应或关键公司业绩可通过半导体权重影响QQQ和SPY。"
    elif has(r"microsoft|apple|amazon|alphabet|google|meta|tesla|openai") and has(
        r"earnings|revenue|antitrust|export|ban|acquisition"
    ):
        topic, channel, priority = "technology", "mega_cap_earnings_policy", 3
        reason = "科技权重股的业绩或监管变化可能传导到指数盈利预期。"
    elif has(r"\b(house|senate)\b") and has(r"democrat|republican|midterm|congress"):
        topic, channel, priority = "election", "us_policy_background", 4
        reason = "国会格局属于政策背景；不把党派胜率直接转换成股票涨跌。"
    if not topic:
        return None
    return {"topic": topic, "driver": role, "impact_channel": channel,
            "importance_rank": priority, "selection_reason": reason}


def screening_priority(classified: dict, question: str, event_title: str,
                       end_date: Any, as_of: Any) -> dict:
    """Rank an imminent national joint-control event without assigning a stock direction."""
    from .prediction_markets import _timestamp

    base = classified["importance_rank"]
    text = str(question) + " " + str(event_title)
    local = bool(re.search(
        r"state (?:house|senate)|legislature|assembly|district|house seat|senate seat|"
        r"\b[A-Z]{2}-\d{1,2}\b|county|municipal", text, re.I))
    both = bool(re.search(r"\bhouse\b", question, re.I)
                and re.search(r"\bsenate\b", question, re.I))
    both = both or bool(re.search(r"both (?:chambers|houses)", question, re.I))
    alternative = bool(re.search(r"\bhouse\b.{0,25}\bor\b.{0,25}\bsenate\b|"
                                 r"\bsenate\b.{0,25}\bor\b.{0,25}\bhouse\b", question, re.I))
    federal = bool(re.search(
        r"\b(?:U\.?S\.?|United States)\s+(?:Congress|Senate|House)\b|"
        r"\b(?:House|Senate|Congress)\s+of\s+(?:the\s+)?United States\b", question, re.I))
    control = bool(re.search(r"\bcontrol\b|\bmajorit\w*\b|balance of power", question, re.I))
    label = r"(?:D|R|I|Democratic|Republican|Independent|Other)"
    composition = rf"(?:{label}\s+Senate,\s*{label}\s+House|{label}\s+House,\s*{label}\s+Senate)"
    canonical = bool(re.fullmatch(r"Balance of Power:\s*\d{4}\s+Midterms(?: Elections)?",
                                 str(event_title).strip(), re.I) and re.fullmatch(
        rf"\d{{4}}\s+Balance of Power:\s*{composition}\s*[?.]?", question.strip(), re.I))
    joint = (classified["topic"] == "election" and both and control
             and (federal or canonical) and not local and not alternative)
    end, clock = _timestamp(end_date), _timestamp(as_of)
    days = (end-clock).total_seconds()/86400 if end is not None and clock is not None else None
    boost = joint and days is not None and 0 < days <= 14
    effective = min(base, 3) if boost else base
    reason = ("美国全国两院联合控制权合约进入未来14日观察窗口，优先于一般政策背景；"
              "不按党派或股票涨跌假设加权。" if boost else
              "未满足近14日全国两院联合控制权条件，保留主题基础优先级。")
    return {"base_importance_rank": base, "effective_importance_rank": effective,
            "importance_rank": effective, "urgency_window_days": 14, "urgency_applied": boost,
            "urgency_reason": reason, "ranking_version": SCREENING_PRIORITY_VERSION,
            "election_scope": "us_congress_joint_control" if joint else None,
            "selection_reason": classified["selection_reason"] + (reason if boost else "")}


def discovery_contract(row: Dict[str, Any]) -> Optional[dict]:
    """Re-derive meaning from the saved question, never trust a cached driver alone."""
    selection = row.get("discovery") or {}
    if not isinstance(selection, dict) or not isinstance(row.get("contract_metadata"), dict):
        return None
    if (row.get("selection_mode") != DISCOVERY_VERSION
            or selection.get("version") != DISCOVERY_VERSION):
        return None
    from .prediction_markets import _contract_metadata, _market_row, _timestamp

    original, parent = selection.get("provider_market"), selection.get("provider_event")
    if (not isinstance(original, dict) or not isinstance(parent, dict)
            or not isinstance(parent.get("market_ids"), list)):
        return None
    identity = str(original.get("id", ""))
    if (not identity.isdigit() or row.get("id") != identity
            or row.get("question") != original.get("question")
            or selection.get("event_id") != str(parent.get("id"))
            or selection.get("event_title") != parent.get("title")
            or identity not in parent.get("market_ids", [])
            or row["contract_metadata"] != _contract_metadata(original)):
        return None
    classified = classify_market(original.get("question"), parent.get("title", ""))
    if not classified or any(row.get(key) != classified[key] for key in (
        "topic", "driver", "impact_channel"
    )):
        return None
    fetched = _timestamp(row.get("fetched_at"))
    if fetched is None:
        return None
    checked = _market_row({**classified, "id": identity,
                           "expected_question": original["question"]}, original, fetched)
    if checked["status"] == "unavailable" or any(row.get(key) != checked.get(key) for key in (
        "yes_probability", "no_probability", "end_date", "market_updated_at",
        "liquidity_usd", "spread",
    )):
        return None
    ranking = selection.get("ranking_version")
    if ranking is not None:
        if ranking != SCREENING_PRIORITY_VERSION:
            return None
        priority = screening_priority(classified, row["question"], parent.get("title", ""),
                                      original.get("endDate"), fetched)
        if any(row.get(key) != value or selection.get(key) != value
               for key, value in priority.items()):
            return None
        classified = {**classified, **priority}
    return {**classified, "id": str(row.get("id")), "expected_question": row["question"],
            "label": row.get("label") or row["question"],
            "settlement_scope_note": "仅按保存的原始合约规则结算，不扩大为全面和平或股指方向。"}


def _fetch_json(path: str, params: dict, timeout: float) -> Any:
    url = "https://gamma-api.polymarket.com" + path + "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "trading-wizzard/0.1 personal research"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def discover_prediction_markets(config: dict, report_date: str, now: Optional[datetime] = None,
                                audit: Optional[dict] = None):
    from .prediction_markets import _market_row, _number, _timestamp

    injected = now
    now = now or datetime.now(timezone.utc)
    settings = config.get("discovery") or {}
    max_events = min(8, max(1, int(settings.get("max_events", 6))))
    max_days = min(180, max(1, int(settings.get("max_days_to_end", 90))))
    horizon = min(max_days, max(1, int(settings.get("horizon_days", 7))))
    pages = min(2, max(1, int(settings.get("active_pages", 2))))
    queries = settings.get("search_queries", DEFAULT_QUERIES)
    queries = [str(q)[:120] for q in queries][:8]
    budget = time.monotonic()+20
    requests = [("/events", {"active": "true", "closed": "false", "limit": 100,
                             "offset": page*100, "order": "volume24hr", "ascending": "false"})
                for page in range(pages)]
    requests += [("/public-search", {"q": query, "limit_per_type": 20,
                                     "events_status": "active", "search_profiles": "false"})
                 for query in queries]
    details = {"version": DISCOVERY_VERSION, "report_date": report_date,
               "started_at": now.isoformat(), "requests": [], "candidates": [],
               "limitations": ["有界扫描与主题搜索，不声称覆盖全市场或所有突发新闻。",
                               "重要程度为透明筛选规则，尚非校准的股票影响权重。",
                               "每个事件选一份代表合约，不把不同结果的No或概率混成总分。"]}
    events, warnings, market_receipts = {}, [], {}
    for path, params in requests[:10]:
        remaining = budget-time.monotonic()
        if remaining <= 0:
            warnings.append("Prediction discovery collection budget exhausted; coverage partial.")
            break
        try:
            payload = _fetch_json(path, params, min(5, remaining))
            received = injected or datetime.now(timezone.utc)
            found = payload if isinstance(payload, list) else (
                payload.get("events", []) if isinstance(payload, dict) else None)
            if not isinstance(found, list):
                raise ValueError("Discovery endpoint returned no event list.")
            details["requests"].append({"path": path, "params": params,
                                        "fetched_at": received.isoformat(), "response": payload})
            for event in found[:100]:
                if isinstance(event, dict) and str(event.get("id", "")).isdigit():
                    identity = str(event["id"])
                    previous = events.get(identity, ({}, received))[0]
                    merged = {str(m["id"]): m for m in previous.get("markets", [])
                              if isinstance(m, dict) and "id" in m}
                    for market in event.get("markets") or []:
                        if not isinstance(market, dict) or "id" not in market:
                            continue
                        key = str(market["id"])
                        old_time = _timestamp(merged.get(key, {}).get("updatedAt"))
                        new_time = _timestamp(market.get("updatedAt"))
                        if key not in merged or old_time is None or (
                            new_time is not None and new_time >= old_time
                        ):
                            merged[key] = market
                            market_receipts[(identity, key)] = received
                    combined = {**previous, **{key: value for key, value in event.items()
                                               if value is not None},
                                "markets": list(merged.values())}
                    events[identity] = (combined, received)
        except (OSError, ValueError, TypeError) as error:
            warnings.append(f"Prediction discovery request failed: {error}")
            details["requests"].append({"path": path, "params": params, "error": str(error)})
    representatives = []
    eligible_audit = {}
    for event_id, (event, received) in events.items():
        members = []
        markets = event.get("markets") or []
        if not isinstance(markets, list):
            continue
        for payload in markets[:100]:
            if not isinstance(payload, dict):
                continue
            question = payload.get("question")
            received = market_receipts.get((event_id, str(payload.get("id"))), received)
            classified = classify_market(question, event.get("title", ""))
            review = {"event_id": event_id, "market_id": str(payload.get("id")),
                      "question": question, "event_title": event.get("title")}
            if not str(payload.get("id", "")).isdigit():
                details["candidates"].append({**review, "decision": "excluded",
                                             "reason": "缺少有效的数字合约标识。"})
                continue
            if not classified:
                details["candidates"].append({**review, "decision": "excluded",
                                             "reason": "未识别到股票宏观或半导体传导渠道。"})
                continue
            priority = screening_priority(classified, question, event.get("title", ""),
                                          payload.get("endDate"), received)
            classified = {**classified, **priority}
            item = {**classified, "id": str(payload.get("id")), "expected_question": question,
                    "label": event.get("title") or question,
                    "yes_scenario": "该原始结果为真，沿所列渠道检查实际市场反应。",
                    "no_scenario": "仅表示该结果未满足；不自动推断相反政策或全面和平。"}
            raw = {**payload, "events": [{"slug": event.get("slug")} ]}
            row = _market_row(item, raw, received, config.get("main_case_threshold", .65))
            row.update({key: classified[key] for key in classified})
            end = _timestamp(row.get("end_date"))
            days = (end-received).total_seconds()/86400 if end else None
            reason = row["reason"] if row["status"] == "unavailable" else None
            if days is None or not 0 < days <= max_days:
                reason = reason or f"不在未来{max_days}日的观察窗口。"
            description = payload.get("description")
            if not isinstance(description, str) or not description.strip():
                reason = reason or "缺少可追溯的结算规则。"
            if reason:
                details["candidates"].append({**review, **classified,
                                             "decision": "excluded", "reason": reason})
                continue
            row["selection_mode"] = DISCOVERY_VERSION
            row["discovery"] = {"version": DISCOVERY_VERSION, "event_id": event_id,
                                "event_title": event.get("title"), "days_to_end": days,
                                "provider_market": raw,
                                "provider_event": {
                                    "id": event_id, "title": event.get("title"),
                                    "market_ids": [str(m.get("id")) for m in markets]},
                                "selection_reason": classified["selection_reason"],
                                "importance_rank": classified["importance_rank"], **priority}
            row["yes_scenario"] = item["yes_scenario"]
            row["no_scenario"] = item["no_scenario"]
            row["settlement_scope_note"] = "仅按保存的原始规则结算，不扩大为全面和平或股指方向。"
            members.append(row)
            candidate = {**review, **classified, "decision": "eligible",
                         "yes_probability": row["yes_probability"],
                         "final_disposition": "not_representative",
                         "final_reason": "同事件以观察期限及同期限的主流Yes结果选择代表合约。"}
            details["candidates"].append(candidate)
            eligible_audit[(event_id, row["id"])] = candidate
        if members:
            # Compare outcomes only at the same deadline; deadlines are not disjoint outcomes.
            deadlines = sorted({row["end_date"] for row in members})
            chosen = min(deadlines, key=lambda end: (
                abs(next(r["discovery"]["days_to_end"] for r in members
                         if r["end_date"] == end)-horizon), end))
            same_deadline = [row for row in members if row["end_date"] == chosen]
            highest_priority = min(row["importance_rank"] for row in same_deadline)
            same_deadline = [row for row in same_deadline
                             if row["importance_rank"] == highest_priority]
            representative = max(same_deadline, key=lambda r: (
                r["yes_probability"], _number(r.get("liquidity_usd")) or 0))
            representative["discovery"]["representative_method"] = (
                "nearest_analysis_horizon_then_dominant_yes_same_deadline")
            representative["discovery"]["eligible_sibling_market_ids"] = [r["id"] for r in members]
            representatives.append(representative)
    representatives.sort(key=lambda r: (r["importance_rank"], r["discovery"]["days_to_end"],
                                        -float(r["liquidity_usd"]), r["id"]))
    selected, channels = [], {}
    for row in representatives:
        channel = row["impact_channel"]
        candidate = eligible_audit[(row["discovery"]["event_id"], row["id"])]
        constraints = (["channel_limit"] if channels.get(channel, 0) >= 1 else []) + (
            ["max_events"] if len(selected) >= max_events else [])
        if constraints:
            candidate.update(final_disposition=constraints[0], exclusion_constraints=constraints,
                             final_reason="代表合约已通过质量校验，但受频道去重或事件数量上限限制。")
            continue
        selected.append(row)
        channels[channel] = channels.get(channel, 0)+1
        candidate.update(final_disposition="selected",
                         final_reason="通过质量、主题、期限及配额筛选。")
    for candidate in details["candidates"]:
        if candidate["decision"] == "excluded":
            candidate.update(final_disposition="ineligible", final_reason=candidate.get("reason"))
    successful = sum("response" in request for request in details["requests"])
    details.update(completed_at=(injected or datetime.now(timezone.utc)).isoformat(),
                   scanned_events=len(events), selected_market_ids=[r["id"] for r in selected],
                   successful_requests=successful,
                   coverage=("partial_bounded_live_scan" if warnings else "bounded_live_scan")
                   if successful else "unavailable")
    digest = hashlib.sha256(json.dumps(details, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()
    for rank, row in enumerate(selected, 1):
        row["discovery"].update(rank=rank, audit_sha256=digest,
                                scanned_events=len(events), candidates=len(details["candidates"]))
    if audit is not None:
        audit.update(details)
    if not selected:
        warnings.append("No discovered equity event passed relevance, "
                        "date and quote-quality gates.")
    return selected, warnings
