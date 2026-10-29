"""Directional commentary must follow measured structure and fixed daily gates."""
import json
import math
from copy import deepcopy

import pytest

from twiz.decision import build_daily_brief

AS_OF = "2026-10-27"


def fixture(ticker="QQQ", close=100, sma20=95, sma50=90):
    row = {"ticker": ticker, "data_source": "yfinance", "data_date": AS_OF,
           "stale": False, "close": close, "return_20d_pct": 2.0,
           "return_20d_start": "2026-09-29"}
    tech = {"indicators": {"close": close, "sma20": sma20, "sma50": sma50}, "levels": [
        {"kind": "support", "price": close-2}, {"kind": "support", "price": close-7},
        {"kind": "resistance", "price": close+1}, {"kind": "resistance", "price": close+20},
    ]}
    return row, tech


def brief(pairs=None, **overrides):
    pairs = pairs or [fixture("QQQ"), fixture("SPY")]
    args = {"report_date": AS_OF, "watchlist_table": [row for row, _ in pairs],
            "technical": {row["ticker"]: tech for row, tech in pairs}, "playbook": {},
            "macro_gate": {"state": "unknown"}, "leadership": {}, "events": [],
            "fundamentals": {}}
    args.update(overrides)
    return build_daily_brief(**args)


@pytest.mark.parametrize("close,sma20,sma50,view", [
    (100, 95, 90, "偏强延续"), (100, 110, 105, "偏弱防守"),
    (100, 95, 110, "反弹修复未确认反转"), (100, 105, 95, "偏强整理"),
    (100, 100, 100, "震荡等待"),
])
def test_direction_tracks_observed_sma_structure(close, sma20, sma50, view):
    result = brief([fixture(t, close, sma20, sma50) for t in ("QQQ", "SPY")])
    assert result["stance"] == view
    assert all(row["view"] == view for row in result["market_scenarios"])
    assert "Yahoo" in result["evidence"][0]
    assert AS_OF in result["evidence"][0]
    assert "日收盘" in result["market_scenarios"][0]["bullish"]


def test_nearby_resistance_break_is_not_downtrend_invalidation():
    result = brief([fixture(t, 100, 110, 105) for t in ("QQQ", "SPY")])
    scenario = result["market_scenarios"][0]
    assert "101.00" in scenario["bullish"] and "反弹第一关" in scenario["bullish"]
    assert "110.00" in scenario["invalidation"]
    assert "101.00" not in scenario["invalidation"]
    assert "120.00" in scenario["bullish"]
    assert "98.00" in scenario["bearish"] and "93.00" in scenario["bearish"]


@pytest.mark.parametrize("change", [
    {"data_source": "sample"}, {"data_source": "unverified"}, {"data_source": "unavailable"},
    {"data_date": "2026-10-15"}, {"data_date": "2026-10-28"}, {"data_date": "bad"},
    {"stale": True}, {"close": math.nan}, {"close": -1}, {"close": None},
])
def test_untrusted_or_invalid_market_inputs_suspend_direction(change):
    pairs = [fixture(t) for t in ("QQQ", "SPY")]
    for row, _ in pairs:
        row.update(change)
    result = brief(pairs)
    assert result["stance"] == "数据不足等待"
    assert result["market_scenarios"] == []
    assert not any(row["usable"] for row in result["ticker_decisions"])
    assert len(result["missing"]) <= 3  # Group missing names rather than flood the main text.


@pytest.mark.parametrize("field,value", [("sma20", None), ("sma50", math.inf),
                                         ("sma50", -100), ("close", 103)])
def test_incomplete_or_mismatched_technical_data_is_not_a_direction(field, value):
    pairs = [fixture(t) for t in ("QQQ", "SPY")]
    for _, tech in pairs:
        tech["indicators"][field] = value
    assert brief(pairs)["stance"] == "数据不足等待"


def test_divergent_anchors_do_not_generalize_qqq_to_whole_market():
    result = brief([fixture("QQQ", 100, 110, 105), fixture("SPY")])
    assert result["stance"] == "指数分化"
    assert "QQQ 偏弱防守" in result["headline"]
    assert "SPY 偏强延续" in result["headline"]
    assert "单一指数" in result["actions"][0]["action"]
    one = brief([fixture("SPY")])
    assert any("仅一个" in text for text in one["missing"])


def test_hardware_weakness_does_not_reverse_broad_index_view():
    result = brief(leadership={"state": "broken"})
    assert result["stance"] == "偏强延续"
    assert result["sectors"][-1]["ticker"] == "硬件领导股"
    assert "不据此改写SPY" in result["sectors"][-1]["condition"]


def test_macro_unknown_keeps_direction_but_does_not_expand_user_budget():
    result = brief()
    assert result["stance"] == "偏强延续"
    assert any("预算" in line and "不自动提高额度" in line for line in result["missing"])
    text = json.dumps(result, ensure_ascii=False)
    assert "概率" not in text and "胜率" not in text and "满仓" not in text


def test_fundamental_labels_alone_do_not_generate_sell_commands():
    result = brief(fundamentals={"QQQ": {"label": "burner", "score": 47, "flags": []}})
    assert "现金流" not in result["ticker_decisions"][0]["action"]
    flagged = brief(fundamentals={"QQQ": {"flags": ["share_issuance_dilution"]}})
    action = flagged["ticker_decisions"][0]["action"]
    assert "核对报表" in action and "不构成卖出指令" in action


def test_sector_priority_uses_aligned_measured_returns_only():
    pairs = [fixture(t) for t in ("QQQ", "SPY", "SMH", "IGV")]
    for row, _ in pairs:
        row["return_20d_pct"] = {"QQQ": -2, "SPY": -0.03, "SMH": -4, "IGV": 5.65}[row["ticker"]]
    result = brief(pairs)
    ranked = {row["ticker"]: row for row in result["sectors"]}
    assert result["sectors"][0]["ticker"] == "IGV"
    assert ranked["IGV"]["relative_return_20d_pct"] == 5.68
    assert ranked["SMH"]["relative_return_20d_pct"] == -3.97
    priority = next(text for text in result["evidence"] if "板块观察顺序" in text)
    assert priority.index("IGV") < priority.index("SMH")
    misaligned = deepcopy(pairs)
    misaligned[-1][0]["return_20d_start"] = "2026-09-28"
    unmatched = next(row for row in brief(misaligned)["sectors"] if row["ticker"] == "IGV")
    assert unmatched["relative_return_20d_pct"] is None
    assert "待验证" in unmatched["condition"]


def test_events_are_imminent_limited_and_estimates_explicit():
    events = [{"date": "2026-10-28", "name": f"Earnings{i}", "importance": 5,
               "tickers": [f"T{i}"], "precision": "scheduled"} for i in range(7)]
    events += [{"date": "2026-10-28", "name": "FOMC", "importance": 5},
               {"date": "2026-10-29", "name": "CPI", "importance": 4,
                "precision": "approximate"},
               {"date": "2026-11-10", "name": "FarFuture", "importance": 5}]
    result = brief(events=events)
    imminent = [text for text in result["evidence"] if text.startswith("未来")]
    assert len(imminent) == 3 and "FOMC" in imminent[0] and "估计日期" in imminent[1]
    assert not any("FarFuture" in text for text in result["evidence"])
    assert any("另有6项" in text for text in result["evidence"])


def test_holdings_actions_change_with_view_and_never_assume_actual_holdings():
    strong = brief()
    weak = brief([fixture(t, 100, 110, 105) for t in ("QQQ", "SPY")])
    unknown = brief([(dict(fixture(t)[0], data_source="sample"), fixture(t)[1])
                     for t in ("QQQ", "SPY")])
    assert "回踩" in strong["actions"][0]["action"]
    assert "暂不追反弹" in weak["actions"][0]["action"]
    assert "暂停" in unknown["actions"][0]["action"]
    assert strong["actions"] != weak["actions"] != unknown["actions"]
    assert all("成本" not in row["action"] for row in weak["actions"])


def test_invalid_levels_and_missing_next_levels_do_not_invent_targets():
    pairs = [fixture(t) for t in ("QQQ", "SPY")]
    for _, tech in pairs:
        tech["levels"] = [{"kind": "support", "price": -1},
                          {"kind": "support", "price": math.nan},
                          {"kind": "resistance", "price": 99},
                          {"kind": "resistance", "price": math.inf}]
    result = brief(pairs)
    scenario = result["market_scenarios"][0]
    assert "不外推" in scenario["bullish"] and "尚无有效支撑" in scenario["bearish"]
    json.dumps(result, allow_nan=False)
    assert "收盘完成" in result["missing"][-1]


def test_previous_session_cannot_look_current_even_when_caller_stale_flag_is_false(monkeypatch):
    monkeypatch.setattr("twiz.decision.expected_complete_session", lambda _: AS_OF)
    pairs = [fixture(t) for t in ("QQQ", "SPY")]
    for row, _ in pairs:
        row.update(data_date="2026-10-26", stale=False)
    assert brief(pairs)["stance"] == "数据不足等待"


def test_confirmed_session_is_not_described_as_unverified_completion():
    pairs = [fixture(t) for t in ("QQQ", "SPY")]
    for row, _ in pairs:
        row["session_complete"] = True
    result = brief(pairs)
    assert "已完成常规交易日收盘" in result["evidence"][0]
    assert not any("未记录收盘完成" in text for text in result["missing"])
