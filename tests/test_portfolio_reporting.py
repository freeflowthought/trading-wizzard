import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml

from twiz.config import DEFAULT_WATCHLIST
from twiz.daily_context import BASIS
from twiz.macro_rotation import DEFAULT_PARAMETERS, build_macro_rotation_policy
from twiz.provenance import begin_analysis_snapshot, finalize_analysis_snapshot, verify_snapshot
from twiz.reporting import build_report_bundle, render_daily_markdown, render_markdown
from twiz.sessions import is_us_equity_session
from twiz.storage import ProjectPaths, write_json
from twiz.swing_optimizer import _hash

DAY = "2026-06-06"


def model():
    return {"date": DAY, "status": "available", "available_at": "2026-06-06T13:00:00Z",
            "price_date": "2026-06-05",
            "price_dates": dict.fromkeys(("QQQ", "SPY", "TQQQ"), "2026-06-05"),
            "price_closes": {"QQQ": 100, "SPY": 100, "TQQQ": 200},
            "next_open_at": "2026-06-08T09:30:00-04:00",
            "target_weights": {"QQQ": .8, "TQQQ": .2, "cash": 0},
            "paper_weights": {"QQQ": 1, "TQQQ": 0, "cash": 0},
            "new_tqqq_increment": .2, "cause": "cycle_confirmed_bottom",
            "qqq_features": {"cycle_bottom_ready": True, "slow_sma": 98.719}, "structural_stop": 90,
            "profit_target_gain": .2, "profit_target_price": None, "time_limit_sessions": 40,
            "news_only": True, "prediction_markets_trigger": False,
            "paper_holdings_only": True, "orders_executed": False,
            "source_fingerprints": {"history": "a"*64}, "policy_sha256": "b"*64,
            "reference_source": {"recent_max_drawdown_pct": 23.641,
                                 "QQQ_recent_max_drawdown_pct": 11.2178}}


def project(tmp_path, plan=None):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    row = {"date": "2026-06-05", "source": "yfinance", "close": 100,
           "session_complete": True, "fetched_at": "2026-06-06T12:00:00Z"}
    write_json(paths.raw / f"market-{DAY}.json", {"QQQ": [row], "SPY": [row]})
    write_json(paths.raw / f"market-context-{DAY}.json", {"TQQQ": [{**row, "close": 200}]})
    plan = dict(plan or model())
    config = {"enabled": True, "automatic_orders": False,
              "activated_at": "2026-06-06T12:00:00Z",
              "parameters": {**DEFAULT_PARAMETERS, "use_events": True, "recovery_weight": .5},
              "cycle_parameters": {"bottom_min_age": 10, "bottom_trend_sessions": 50,
                                   "max_holding_sessions": 40, "migrate_gain": .2,
                                   "top_retained": .7},
              "reference_source": plan["reference_source"]}
    config_path = paths.config / "portfolio-strategy.yml"
    config_path.write_text(yaml.safe_dump(config))
    frozen = build_macro_rotation_policy(
        config["parameters"], cycle_parameters=config["cycle_parameters"])
    plan.update(policy_sha256=frozen["policy_sha256"],
                model_source_sha256=frozen["policy"]["source_code_sha256"],
                source_fingerprints={"portfolio-strategy.yml": {
                    "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}},
                decision_at=plan["available_at"], price_at="2026-06-05T16:15:00-04:00",
                tqqq_entry_wac=None, tqqq_gain=None, holding_sessions=None,
                pending_order={"paper": True})
    replay = {"policy": frozen["policy"], "policy_sha256": frozen["policy_sha256"],
              "source_code_sha256": frozen["policy"]["source_code_sha256"],
              "decisions": [{"at": plan["decision_at"], "target_weights": plan["target_weights"],
                             "risk_armed": True,
                             "cause": plan["cause"], "features": {"QQQ": plan["qqq_features"]},
                             "recovery_stop": plan["structural_stop"], "tqqq_entry_wac": None,
                             "tqqq_gain": None, "holding_sessions": None}],
              "equity_curve": [{"at": plan["price_at"], "weights": plan["paper_weights"]}],
              "pending_order": plan["pending_order"]}
    prices = {t: [{"date": plan["price_date"], "close": v}]
              for t, v in plan["price_closes"].items()}
    trace = {"date": DAY, "status": plan["status"], "available_at": plan["available_at"],
             "policy": frozen["policy"], "policy_sha256": frozen["policy_sha256"],
             "source_fingerprints": plan["source_fingerprints"], "replay": replay,
             "used_price_history": prices, "used_price_history_sha256": _hash(prices),
             "used_news": [], "used_news_sha256": _hash([])}
    write_json(paths.processed / f"portfolio-plan-{DAY}.json", plan)
    write_json(paths.processed / f"portfolio-training-{DAY}.json", trace)
    return paths


def test_model_targets_replace_dip_ladder_without_claiming_personal_holdings_or_orders(tmp_path):
    bundle = build_report_bundle(project(tmp_path), DAY, DEFAULT_WATCHLIST)
    bundle.action_plan = {"status": "conditional_plan", "buy_now_pct": 0,
                          "entries": [{"price": 777, "account_pct": 2}]}
    primary, details = render_daily_markdown(bundle), render_markdown(bundle)
    for text in (primary, details):
        assert "| QQQ | 80.0% |" in text and "| TQQQ | 20.0% |" in text
        assert "| 现金 | 0.0% |" in text and "总资金100%" in text
        assert "纸面仓位不代表你的持仓，也没有执行订单" in text
        assert "模型成本浮盈达20%后转回QQQ" in text and "最长40个交易日" in text
        assert "连续2日收盘低于各自200日线（当前98.72）" in text
        assert "独立恢复结构位90.00：日收盘跌破即触发" in text
        assert "低点形成满10交易日" in text and "站上50日线及前3日高点" in text
        assert "最高收盘回撤≥8%" in text and "TQQQ模型目标50%" in text
        assert "本轮参考价至最高收盘涨幅达3ATR" in text
        assert "日收盘跌破此前5日最低价" in text
        assert "QQQ最多70%（不超过当前总风险敞口）" in text
        assert "今日行动卡：抄底与止盈" not in text and "777.00" not in text
        assert "预测市场赔率仅作旁证，不额外触发周期交易" in text
    assert primary.count("23.64%") == 1 and "11.22%" in primary
    assert "调整价代理回测口径" in primary
    assert "稳健" not in primary
    assert "未来20个交易日方向概率" in primary and "BTC / 加密永续" in primary
    assert all(f"| {asset.ticker} |" in primary for asset in DEFAULT_WATCHLIST)


@pytest.mark.parametrize("change", [
    {"available_at": "2027-01-01T00:00:00Z"}, {"price_date": "2026-06-04"},
    {"price_closes": {"QQQ": 101, "SPY": 100, "TQQQ": 200}},
    {"target_weights": {"QQQ": 1, "TQQQ": .2, "cash": 0}},
    {"news_only": False}, {"prediction_markets_trigger": True}, {"orders_executed": True},
    {"next_open_at": "2026-06-06T09:00:00-04:00"},
])
def test_invalid_model_is_unknown_and_never_silently_falls_back_to_old_ladder(tmp_path, change):
    bundle = build_report_bundle(
        project(tmp_path, {**model(), **change}), DAY, DEFAULT_WATCHLIST[:2])
    bundle.action_plan = {"status": "conditional_plan", "entries": [{"price": 777}],
                          "buy_now_pct": 2}
    primary = render_daily_markdown(bundle)
    assert "模型计划不可用" in primary
    assert "80.0%" not in primary and "777.00" not in primary


def test_current_fomc_reminder_pauses_new_tqqq_without_changing_reference_or_liquidating(tmp_path):
    bundle = build_report_bundle(project(tmp_path), DAY, DEFAULT_WATCHLIST[:2])
    bundle.events = [{"kind": "fomc", "date": DAY, "precision": "verified"}]
    primary = render_daily_markdown(bundle)
    assert "当前暂停新开或新增TQQQ" in primary
    assert "| TQQQ | 20.0% |" in primary
    assert "新TQQQ增量参考：总资金20.0%" not in primary
    assert "不虚构已有仓位清仓" in primary


def test_cached_model_cannot_self_certify_changed_market_price(tmp_path):
    plan = model()
    plan["report_price_inputs"] = {"QQQ": {"close": 100}}
    paths = project(tmp_path, plan)
    write_json(paths.raw / f"market-context-{DAY}.json", {"TQQQ": [{
        "date": "2026-06-05", "close": 201, "source": "yfinance", "session_complete": True}]})
    bundle = build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2])
    assert bundle.portfolio_plan["report_price_inputs"]["TQQQ"]["close"] == 201
    assert "模型计划不可用" in render_daily_markdown(bundle)


def test_dated_plan_and_training_are_in_analysis_archive(tmp_path):
    paths = project(tmp_path)
    draft = begin_analysis_snapshot(paths, DAY, source_root=tmp_path)
    receipt = finalize_analysis_snapshot(
        draft, build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2]))
    run = Path(receipt["path"])
    manifest = verify_snapshot(run)["manifest"]
    originals = {row.get("original_path") for row in manifest["files"]}
    assert f"data/processed/portfolio-plan-{DAY}.json" in originals
    assert f"data/processed/portfolio-training-{DAY}.json" in originals
    assert (run / "decision-input-refs.json").read_text().count(
        "news_cycle_portfolio_reference") == 1


@pytest.mark.parametrize("field,value", [("policy_sha256", "d"*64),
                                        ("target_weights", {"QQQ": .7, "TQQQ": .3, "cash": 0}),
                                        ("cause", "cycle_hold"), ("structural_stop", 99),
                                        ("next_open_at", "2099-01-01T09:30:00-05:00")])
def test_plan_cannot_change_policy_or_tail_fields_at_same_current_price(tmp_path, field, value):
    paths = project(tmp_path)
    path = paths.processed / f"portfolio-plan-{DAY}.json"
    plan = json.loads(path.read_text())
    plan[field] = value
    write_json(path, plan)
    text = render_daily_markdown(build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2]))
    assert "模型计划不可用" in text


def test_missing_training_or_changed_active_policy_cannot_self_certify(tmp_path):
    paths = project(tmp_path)
    (paths.processed / f"portfolio-training-{DAY}.json").unlink()
    bundle = build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2])
    assert "模型计划不可用" in render_daily_markdown(bundle)


def test_below_long_ma_recovery_does_not_claim_unarmed_trend_exit(tmp_path):
    paths = project(tmp_path)
    plan_path = paths.processed / f"portfolio-plan-{DAY}.json"
    trace_path = paths.processed / f"portfolio-training-{DAY}.json"
    plan, trace = json.loads(plan_path.read_text()), json.loads(trace_path.read_text())
    plan["qqq_features"]["slow_sma"] = 110
    trace["replay"]["decisions"][-1]["features"]["QQQ"]["slow_sma"] = 110
    trace["replay"]["decisions"][-1]["risk_armed"] = False
    write_json(plan_path, plan)
    write_json(trace_path, trace)
    text = render_daily_markdown(build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2]))
    assert "当前未启用，须先日收盘达到/高于长期线后启用" in text
    assert "独立恢复结构位90.00：日收盘跌破即触发" in text


def level_project(tmp_path):
    paths = project(tmp_path)
    plan_path, trace_path = (paths.processed / f"{name}-{DAY}.json"
                            for name in ("portfolio-plan", "portfolio-training"))
    plan, trace = json.loads(plan_path.read_text()), json.loads(trace_path.read_text())
    features = {"close": 730, "cycle_bottom_ready": False, "anchor_low": 701.15997,
                "anchor_date": "2026-06-02", "anchor_age": 3, "drawdown": .077008,
                "cycle_bottom_sma": 725.4244, "prior3_high": 722.869995,
                "prior_low": 701.15997, "slow_sma": 678.7191}
    plan.update(qqq_features=features, price_closes={"QQQ": 730, "SPY": 770, "TQQQ": 200},
                target_weights={"QQQ": 1, "TQQQ": 0, "cash": 0}, new_tqqq_increment=0,
                pending_order=None, cause="cycle_hold")
    last = trace["replay"]["decisions"][-1]
    last.update(features={"QQQ": features}, target_weights=plan["target_weights"],
                cause="cycle_hold")
    trace["replay"]["pending_order"] = None
    trace["used_price_history"]["QQQ"][-1]["close"] = 730
    days, cursor = [], date(2026, 6, 5)
    while len(days) < 20:
        if is_us_equity_session(cursor):
            days.append(cursor.isoformat())
        cursor -= timedelta(days=1)
    trace["used_price_history"]["SPY"] = [{
        "date": day, "source": "yfinance", "session_complete": True, "price_basis": BASIS,
        "open": 770, "close": 770, "low": 758.67999, "high": 781.619995}
        for day in reversed(days)]
    trace["used_price_history_sha256"] = _hash(trace["used_price_history"])
    write_json(plan_path, plan)
    write_json(trace_path, trace)
    write_json(paths.raw / f"market-{DAY}.json", {ticker: [{
        "date": "2026-06-05", "source": "yfinance", "session_complete": True,
        "fetched_at": "2026-06-06T12:00:00Z", "open": close, "close": close,
        "high": close+1, "low": close-1}] for ticker, close in (("QQQ", 730), ("SPY", 770))})
    write_json(paths.processed / f"technical-{DAY}.json", {"QQQ": {
        "date": "2026-06-04", "indicators": {"close": 730, "sma50": 725.0977},
        "levels": [{"kind": "support", "price": 888888.88, "distance_pct": -1}]}})
    return paths


def test_dynamic_levels_use_bound_raw_prices_and_explain_reclaimed_breakout_is_not_buy(tmp_path):
    bundle = build_report_bundle(level_project(tmp_path), DAY, DEFAULT_WATCHLIST[:2])
    text = render_daily_markdown(bundle)
    section = text.split("### 动态关键价位", 1)[1].split("| 条件 | 动作 |", 1)[0]
    for number in ("701.16", "725.42", "722.87", "678.72", "758.68", "781.62"):
        assert number in section
    assert "725.10" not in section and "888888.88" not in text
    assert "前3日高点 | 722.87 | 日收盘已高于；不是未来买入挂单价" in section
    assert "低点年龄3日不足10日" in section and "低点回撤7.70%不足8%" in section
    assert "仅宏观观察，无新增SPY规则" in section
    assert "触价不是买入；须完整底部确认" in section


@pytest.mark.parametrize("change", ["feature", "future_clock"])
def test_unverified_or_expired_plan_cannot_show_bound_trigger_levels(tmp_path, change):
    paths = level_project(tmp_path)
    path = paths.processed / f"portfolio-plan-{DAY}.json"
    plan = json.loads(path.read_text())
    if change == "feature":
        plan["qqq_features"]["anchor_low"] = 123456.78
    else:
        plan["available_at"] = "2027-01-01T00:00:00Z"
    write_json(path, plan)
    text = render_daily_markdown(build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2]))
    assert "模型计划不可用" in text and "动态关键价位" not in text and "123456.78" not in text


def test_missing_spy_ohlc_is_unknown_and_history_tampering_is_rejected(tmp_path):
    paths = level_project(tmp_path)
    path = paths.processed / f"portfolio-training-{DAY}.json"
    trace = json.loads(path.read_text())
    del trace["used_price_history"]["SPY"][0]["high"]
    trace["used_price_history_sha256"] = _hash(trace["used_price_history"])
    write_json(path, trace)
    text = render_daily_markdown(build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2]))
    assert "SPY | 近20交易日高点 | 未知" in text
    trace["used_price_history"]["SPY"][0]["low"] = 1  # Deliberately do not renew the bound hash.
    write_json(path, trace)
    text = render_daily_markdown(build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2]))
    assert "模型计划不可用" in text and "动态关键价位" not in text
