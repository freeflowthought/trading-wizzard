import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twiz.config import DEFAULT_WATCHLIST
from twiz.reporting import (
    build_report_bundle,
    render_markdown,
    render_research_markdown,
    write_report_files,
)
from twiz.storage import ProjectPaths, write_json


class ReportingTests(unittest.TestCase):
    def test_report_bundle_and_files_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = ProjectPaths(Path(tmp))
            paths.ensure()
            report_date = "2026-06-06"
            write_json(
                paths.raw / f"market-{report_date}.json",
                {"SPY": [{"date": report_date, "close": 500}]},
            )
            write_json(
                paths.processed / f"technical-{report_date}.json",
                {
                    "SPY": {
                        "regime": "uptrend",
                        "levels": [
                            {"kind": "support", "price": 495, "distance_pct": -1.0},
                            {"kind": "support", "price": 490, "distance_pct": -2.0},
                            {"kind": "support", "price": 475, "distance_pct": -5.0},
                            {"kind": "support", "price": 440, "distance_pct": -12.0},
                            {"kind": "resistance", "price": 505, "distance_pct": 1.0},
                            {"kind": "resistance", "price": 510, "distance_pct": 2.0},
                            {"kind": "resistance", "price": 525, "distance_pct": 5.0},
                        ],
                    }
                },
            )
            write_json(
                paths.processed / f"sentiment-summary-{report_date}.json",
                {"SPY": {"weighted_score": 0.2, "total_weight": 3.0}},
            )
            bundle = build_report_bundle(paths, report_date, DEFAULT_WATCHLIST[:1])
            markdown = render_markdown(bundle, detailed=True)
            md_path, html_path = write_report_files(paths, bundle)
            self.assertIn("SPY", markdown)
            self.assertIn("Immediate support levels", markdown)
            self.assertIn("Intermediate downside checkpoints", markdown)
            self.assertIn("Deeper downside checkpoints, not immediate support", markdown)
            self.assertIn("Immediate resistance / possible ceiling", markdown)
            self.assertIn("Intermediate upside checkpoints", markdown)
            self.assertIn("495.00", markdown)
            self.assertIn("505.00", markdown)
            self.assertTrue(md_path.exists())
            self.assertTrue(html_path.exists())


if __name__ == "__main__":
    unittest.main()


def test_chinese_decision_summary_precedes_full_data_table(tmp_path):
    from datetime import date, timedelta

    paths = ProjectPaths(tmp_path)
    paths.ensure()
    report_date = "2026-06-06"
    rows = [
        {"date": (date(2026, 6, 5) - timedelta(days=60 - index)).isoformat(),
         "close": 100.0, "source": "yfinance", "session_complete": True}
        for index in range(61)
    ]
    technical = {
        ticker: {
            "regime": "mixed",
            "indicators": {"close": 100.0, "sma20": 110.0, "sma50": 105.0},
            "levels": [
                {"kind": "support", "price": 95, "distance_pct": -5},
                {"kind": "support", "price": 90, "distance_pct": -10},
                {"kind": "resistance", "price": 108, "distance_pct": 8},
                {"kind": "resistance", "price": 115, "distance_pct": 15},
            ],
        } for ticker in ("SPY", "QQQ")
    }
    write_json(paths.raw / f"market-{report_date}.json", {"SPY": rows, "QQQ": rows})
    write_json(paths.processed / f"technical-{report_date}.json", technical)
    bundle = build_report_bundle(paths, report_date, DEFAULT_WATCHLIST[:2])
    markdown = render_markdown(bundle, detailed=True)
    assert bundle.daily_brief["stance"] == "偏弱防守"
    assert markdown.index("## 今日结论") < markdown.index("## Watchlist")
    assert markdown.index("## 今日结论") < markdown.index("## Data Coverage")
    assert "行情快照（日线）：" in markdown
    assert "2026-06-06" in markdown
    assert "大盘条件与失效位" in markdown
    assert "已有杠杆持仓" in markdown
    assert "个股方向与现在怎么做" in markdown
    ticker_table = markdown.split("### 个股方向与现在怎么做", 1)[1].split("## 数据与技术明细", 1)[0]
    assert "日收盘跌破 95" in ticker_table
    assert "下一支撑 90" in ticker_table
    assert "日收盘收复 110" in ticker_table
    assert "Source / Last Bar" in markdown
    assert "Intermediate downside checkpoints" in markdown
    assert bundle.watchlist_table[0]["return_20d_pct"] == 0
    assert bundle.watchlist_table[0]["return_20d_start"] == "2026-05-16"


def test_sample_prices_cannot_create_chinese_direction(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    report_date = "2026-06-06"
    write_json(paths.raw / f"market-{report_date}.json", {
        "SPY": [{"date": report_date, "source": "sample", "close": 100}],
    })
    write_json(paths.processed / f"technical-{report_date}.json", {
        "SPY": {"indicators": {"close": 100, "sma20": 90, "sma50": 80}},
    })
    bundle = build_report_bundle(paths, report_date, DEFAULT_WATCHLIST[:1])
    assert bundle.daily_brief["stance"] == "数据不足等待"
    assert not bundle.daily_brief["ticker_decisions"][0]["usable"]
    assert "偏强延续" not in render_markdown(bundle, detailed=True)


def test_report_html_renders_chinese_tables_and_escapes_provider_markup():
    from twiz.reporting import render_html

    result = render_html(
        '# 日报\n\n## 今日结论\n\n**偏弱防守**\n\n'
        '| 标的 | 判断 |\n| --- | --- |\n| SPY | <script>bad()</script> |\n',
        "日报",
    )
    assert '<html lang="zh-CN">' in result
    assert "<h2>今日结论</h2>" in result
    assert "<strong>偏弱防守</strong>" in result
    assert "<table>" in result and "<th>标的</th>" in result
    assert "<script>" not in result
    assert "&lt;script&gt;bad()&lt;/script&gt;" in result


def prediction_quote(**overrides):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    return {
        "id": "fed-test", "label": "利率事件", "topic": "fed",
        "question": "Will the Fed decrease interest rates at its next meeting?",
        "status": "yes_leaning", "yes_probability": 0.72, "no_probability": 0.28,
        "reason": "", "market_updated_at": (now - timedelta(minutes=15)).isoformat(),
        "fetched_at": now.isoformat(), "end_date": (now + timedelta(days=7)).isoformat(),
        "liquidity_usd": 25000.0, "spread": 0.02,
        "url": "https://polymarket.com/event/fed-test",
        "yes_scenario": "若降息条件满足，再观察利率与价格的共同反应。",
        "no_scenario": "未满足本合约的降息条件，按实际决议继续观察。",
        **overrides,
    }


def test_prediction_market_table_preserves_exact_question_and_event_semantics(tmp_path):
    from datetime import date

    from twiz.reporting import render_html

    paths = ProjectPaths(tmp_path)
    paths.ensure()
    report_date = date.today().isoformat()
    quote = prediction_quote()
    write_json(paths.processed / f"prediction-markets-{report_date}.json", [quote])
    bundle = build_report_bundle(paths, report_date, DEFAULT_WATCHLIST[:1])
    markdown = render_markdown(bundle, detailed=True)
    assert bundle.prediction_markets == [quote]
    assert markdown.index("事件温度计（Polymarket）") < markdown.index("个股方向与现在怎么做")
    assert markdown.index("事件温度计（Polymarket）") < markdown.index("## Watchlist")
    assert quote["question"] in markdown and quote["url"] in markdown
    assert "72.0%" in markdown and "流动性$25,000" in markdown
    assert "不是QQQ上涨概率" in markdown
    assert "不同问题与期限不能直接横比" in markdown
    assert "不自动改变交易方向" in markdown
    assert bundle.daily_brief["stance"] == "数据不足等待"
    assert [row["step"] for row in bundle.daily_brief["framework"]] == [
        "大盘方向", "利率环境", "事件风险", "板块选择",
    ]
    result_html = render_html(markdown, "日报")
    assert '<a href="https://polymarket.com/event/fed-test"' in result_html


def test_unavailable_prediction_probability_is_unknown_not_no(tmp_path):
    from datetime import date

    paths = ProjectPaths(tmp_path)
    paths.ensure()
    report_date = date.today().isoformat()
    quote = prediction_quote(status="unavailable", yes_probability=None, no_probability=None,
                             reason="Liquidity too low.")
    write_json(paths.processed / f"prediction-markets-{report_date}.json", [quote])
    bundle = build_report_bundle(paths, report_date, DEFAULT_WATCHLIST[:1])
    thermometer = render_markdown(bundle, detailed=True).split(
        "### 事件温度计（Polymarket）", 1
    )[1].split("###", 1)[0]
    assert "| 未知 |" in thermometer
    assert "0.0%" not in thermometer
    assert "Liquidity too low." in thermometer
    assert any("未知不等于No" in row for row in bundle.daily_brief["missing"])


def test_stale_cached_prediction_quote_is_withheld(tmp_path):
    from datetime import date, datetime, timedelta, timezone

    paths = ProjectPaths(tmp_path)
    paths.ensure()
    report_date = date.today().isoformat()
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    write_json(paths.processed / f"prediction-markets-{report_date}.json", [
        prediction_quote(market_updated_at=old),
    ])
    bundle = build_report_bundle(paths, report_date, DEFAULT_WATCHLIST[:1])
    markdown = render_markdown(bundle, detailed=True)
    assert "超过24小时" in markdown
    assert "72.0%" not in markdown
    assert bundle.daily_brief["event_evidence"] == []


def conditional_action_plan():
    return {
        "status": "conditional_plan", "headline": "QQQ仅作一轮抄底计划。",
        "current_action": "现在不买；等待支撑确认，当前新增买入0%。", "buy_now_pct": 0,
        "budget": {"ticker": "QQQ", "max_new_capital_pct": 10,
                   "planned_capital_pct": 10, "currently_enabled_pct": 2,
                   "uncommitted_cash_pct": 8},
        "entries": [
            {"batch": 1, "price": 700, "account_pct": 3, "currently_enabled_pct": 2,
             "status": "等待确认", "trigger": "触及700后日收盘重新站回700"},
            {"batch": 2, "price": 690, "account_pct": 3, "currently_enabled_pct": 0,
             "status": "宏观待确认", "trigger": "第一批有确认且宏观门槛解除后再复核"},
            {"batch": 3, "price": 680, "account_pct": 4, "currently_enabled_pct": 0,
             "status": "宏观待确认", "trigger": "前两批依次确认后再复核"},
        ],
        "exits": [
            {"stage": 1, "price": 720, "trim_original_dip_pct": 40, "status": "仅已建仓后"},
            {"stage": 2, "price": 730, "trim_original_dip_pct": 30, "status": "仅已建仓后"},
            {"stage": 3, "price": 740, "trim_original_dip_pct": 30, "status": "仅已建仓后"},
        ],
        "stop": {"price": 675, "modeled_loss_pct": 0.31}, "limitations": [],
    }


def test_action_card_is_first_and_distinguishes_account_budget_from_campaign_trims(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    from datetime import date, timedelta

    rows = [
        {"date": (date(2026, 6, 5) - timedelta(days=60-index)).isoformat(),
         "close": 712.0, "source": "yfinance"} for index in range(61)
    ]
    write_json(paths.raw / "market-2026-06-06.json", {"SPY": rows, "QQQ": rows})
    write_json(paths.processed / "technical-2026-06-06.json", {
        ticker: {"indicators": {"close": 712, "sma20": 730, "sma50": 725}}
        for ticker in ("SPY", "QQQ")
    })
    bundle = build_report_bundle(paths, "2026-06-06", DEFAULT_WATCHLIST[:2])
    bundle.action_plan = conditional_action_plan()
    markdown = render_markdown(bundle, detailed=True)
    assert markdown.index("今日行动卡：抄底与止盈") < markdown.index("未来20个交易日方向概率")
    assert "| 框架 | 当前判断 |" not in markdown
    card = markdown.split("### 今日行动卡：抄底与止盈", 1)[1].split("### 未来20个交易日", 1)[0]
    assert "总账户的10%" in card
    assert "不是每只标的各用这个比例" in card
    assert "现在可执行买入 **0%**" in card
    assert "满足触发条件后最多可启用 2%" in card
    assert "本轮暂不启用预算 8%" in card
    assert "700.00 | 3% | 2%" in card
    assert "690.00 | 3% | 0%" in card
    assert "680.00 | 4% | 0%" in card
    assert "720.00 | 40%" in card and "730.00 | 30%" in card and "740.00 | 30%" in card
    assert "本轮抄底仓原始份额" in card
    assert "不是按剩余仓位，也不包含已有核心仓" in card
    assert "日收盘跌破 QQQ 675.00" in card
    assert "取消剩余加仓" in card
    assert "0.31%" in card and "实际损失超过这个数" in card
    assert "其他标的新增分配为0%，不新增杠杆" in card


def test_unavailable_action_card_does_not_invent_budget_or_prices():
    from twiz.reporting import _render_action_plan

    card = "\n".join(_render_action_plan({
        "status": "unavailable", "limitations": ["没有可用的真实行情。"],
    }))
    assert "当前新增买入0%" in card
    assert "没有可用的真实行情" in card
    assert "总账户的10%" not in card and "| 批次 |" not in card


def test_legacy_playbook_does_not_add_competing_allocation_instructions():
    from twiz.reporting import _playbook_notes

    notes = "\n".join(_playbook_notes({
        "staged_entries": [{"batch": 1, "price": 700, "fraction": 0.4}],
        "sell_put_strike": 690, "targets": [720, 730],
    }))
    assert "40%" not in notes and "回踩观察梯度" not in notes
    assert "卖PUT" not in notes and "Sell-put" not in notes
    assert "技术阻力参考（不另设仓位或止盈指令）" in notes


def test_missing_latest_completed_session_blocks_allocation_even_one_day_old(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "twiz.reporting.expected_complete_session", lambda report_date: "2026-10-27"
    )
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    write_json(paths.raw / "market-2026-10-27.json", {
        ticker: [{"date": "2026-10-26", "source": "yfinance", "close": 712.2}]
        for ticker in ("SPY", "QQQ")
    })
    bundle = build_report_bundle(paths, "2026-10-27", DEFAULT_WATCHLIST[:2])
    assert all(row["stale"] for row in bundle.watchlist_table)
    assert bundle.action_plan["status"] == "unavailable"
    assert bundle.action_plan["entries"] == [] and bundle.action_plan["exits"] == []
    markdown = render_markdown(bundle, detailed=True)
    assert "最新交易日行情未齐，暂停抄底/加仓/止盈价建议" in markdown
    assert "应有完整交易日：2026-10-27" in markdown
    assert "| 批次 | 观察价 |" not in markdown
    bundle.action_plan = conditional_action_plan()
    bundle.action_plan["buy_now_pct"] = 2
    assert "| 批次 | 观察价 |" not in render_markdown(bundle, detailed=True)
    assert "今天怎么做：当前新增买入信号 0%" in render_markdown(bundle)
    assert "今天怎么做：当前新增买入信号 2%" not in render_markdown(bundle)


def test_action_plan_uses_project_policy_and_preserves_close_provenance(tmp_path, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(
        "twiz.reporting.expected_complete_session", lambda report_date: "2026-10-27"
    )
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    write_json(paths.raw / "market-2026-10-27.json", {
        ticker: [{"date": "2026-10-27", "source": "yfinance", "close": 721.25,
                  "close_source": "yahoo_regular_market_metadata_after_close"}]
        for ticker in ("SPY", "QQQ")
    })
    loader = Mock(return_value={"primary_ticker": "QQQ", "max_new_capital_pct": 10})
    builder = Mock(return_value=conditional_action_plan())
    monkeypatch.setattr("twiz.reporting.load_allocation_config", loader)
    monkeypatch.setattr("twiz.reporting.build_dynamic_action_plan", builder)
    bundle = build_report_bundle(paths, "2026-10-27", DEFAULT_WATCHLIST[:2])
    loader.assert_called_once_with(paths.config / "allocation.yml")
    assert builder.call_args.kwargs["policy"]["max_new_capital_pct"] == 10
    assert all(row["close_source"] == "yahoo_regular_market_metadata_after_close"
               for row in bundle.watchlist_table)
    assert not bundle.daily_brief["price_data_pending"]
    assert bundle.action_plan["status"] == "conditional_plan"
    assert "Close provenance: yahoo_regular_market_metadata_after_close" in render_markdown(
        bundle, detailed=True
    )


def test_full_report_context_uses_configured_100_ceiling_without_hidden_scenario_10(
        tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from test_structure import action_structure

    paths = ProjectPaths(tmp_path)
    paths.ensure()
    paths.config.mkdir(parents=True, exist_ok=True)
    (paths.config / "allocation.yml").write_text(
        "max_new_capital_pct: 100\nmax_campaign_loss_pct: 100\n"
        "unknown_macro_initial_cap_pct: 2\nprovisional_cap_pct: 2\n")
    report_date = "2026-10-27"
    monkeypatch.setattr("twiz.reporting.expected_complete_session", lambda _: report_date)
    monkeypatch.setattr("twiz.structure.expected_complete_session", lambda _: report_date)
    write_json(paths.raw / f"market-{report_date}.json", {
        ticker: [{"date": report_date, "source": "yfinance", "session_complete": True,
                  "open": 104, "high": 105, "low": 99, "close": 104.5, "volume": 1000}]
        for ticker in ("SPY", "QQQ")})
    write_json(paths.processed / f"macro-gate-{report_date}.json", {"state": "risk_on"})
    structure = action_structure(True)
    structure["by_ticker"]["QQQ"]["recovery_levels"].append({
        "price": 200, "source": "confirmed_price_pivot", "date": "2026-10-01"})
    # Only the price-artifact fixture is supplied; loader, scenario and action join all run.
    monkeypatch.setattr("twiz.reporting.build_market_structure", lambda *_: structure)
    bundle = build_report_bundle(paths, report_date, DEFAULT_WATCHLIST[:2],
                                 analysis_clock=datetime(2026, 10, 28, 2, tzinfo=timezone.utc))
    assert bundle.scenario_context["policy"]["max_new_capital_pct"] == 100
    assert bundle.action_plan["budget"]["max_new_capital_pct"] == 100
    assert bundle.action_plan["budget"]["planned_capital_pct"] == 100
    assert sum(row["account_pct"] for row in bundle.action_plan["entries"]) == 100
    assert bundle.action_plan["buy_now_pct"] == 0  # Unknown event state does not authorize a buy.
    assert "总账户的100%" in render_markdown(bundle, detailed=True)


def dated_report_inputs(paths, report_date="2026-06-06"):
    write_json(paths.raw / f"market-{report_date}.json", {
        ticker: [{"date": "2026-06-05", "source": "yfinance", "close": 100,
                  "close_source": "regular_market_metadata", "session_complete": True}]
        for ticker in ("SPY", "QQQ")
    })
    write_json(paths.processed / f"technical-{report_date}.json", {
        ticker: {"indicators": {"close": 100, "sma20": 101, "sma50": 102}, "levels": [
            {"kind": "support", "price": 99, "distance_pct": -1},
            {"kind": "resistance", "price": 103, "distance_pct": 3},
        ]} for ticker in ("SPY", "QQQ")
    })


def test_event_policy_reaches_action_engine_and_context_cannot_replace_anchor(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    dated_report_inputs(paths)
    write_json(paths.raw / "market-context-2026-06-06.json", {
        "SPY": [{"date": "2026-06-05", "close": 999, "source": "yfinance"}],
        "XLE": [{"date": "2026-06-05", "close": 80, "source": "yfinance"}],
    })
    scenario = {"policy": {"initial_cap_pct": 2, "before_event_cap_pct": 2,
                           "event_main_case_confirmed": True, "preferred_entry_tier": "support"}}
    with patch("twiz.reporting.build_scenario_context", return_value=scenario) as context, patch(
        "twiz.reporting.build_dynamic_action_plan", return_value={"status": "conditional_plan"}
    ) as action:
        bundle = build_report_bundle(paths, "2026-06-06", DEFAULT_WATCHLIST[:2])
    passed = action.call_args.kwargs["policy"]
    assert passed["event_main_case_confirmed"] is True
    assert passed["before_event_cap_pct"] == 2
    assert passed["preferred_entry_tier"] == "support"
    assert context.call_args.args[3]["SPY"][-1]["close"] == 100
    assert "XLE" in context.call_args.args[3]
    assert bundle.shunge_context["scenario"] == {"active": False, "study_only": True}


def test_equity_proxy_reference_stays_separate_from_stock_direction_and_allocation(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    dated_report_inputs(paths)
    baseline = build_report_bundle(paths, "2026-06-06", DEFAULT_WATCHLIST[:2])
    references = [{
        "symbol": symbol, "reference_for": symbol, "asset_type": "equity_perpetual",
        "instrument_name": symbol, "source": "Lighter", "mapping_verified": True,
        "equity_mapping_verified": True, "market_id": {"SPY": 128, "QQQ": 129}[symbol],
        "mapping_basis": "Reviewed exact ETF mapping.", "source_url": "https://example.test/spec",
        "status": "reference_only", "fetched_at": "2026-06-06T12:00:00-04:00",
        "exchange_updated_at": None, "quote_age_verified": False,
        "mark_price": price, "day_change_pct": 1.25, "basis_pct": 0.02,
        "funding_rate_raw": 0.0000125, "funding_unit": "provider interval",
        "open_interest_notional_usd": 5000000,
    } for symbol, price in (("SPY", 500), ("QQQ", 400))]
    write_json(paths.processed / "cross-market-2026-06-06.json", references)
    bundle = build_report_bundle(paths, "2026-06-06", DEFAULT_WATCHLIST[:2])
    assert bundle.cross_market == references
    assert bundle.daily_brief["stance"] == baseline.daily_brief["stance"]
    assert bundle.action_plan == baseline.action_plan
    assert bundle.watchlist_table == baseline.watchlist_table
    assert all(row["session_complete"] is True for row in bundle.watchlist_table)
    markdown = render_markdown(bundle, detailed=True)
    assert markdown.index("事件温度计（Polymarket）") < markdown.index("盘后／周末风险参考")
    assert markdown.index("盘后／周末风险参考") < markdown.index("板块观察（")
    reference_table = markdown.split("### 盘后／周末风险参考", 1)[1].split("###", 1)[0]
    assert "SPY / ETF永续代理" in reference_table and "QQQ / ETF永续代理" in reference_table
    assert "$500.00" in reference_table and "+1.25%" in reference_table
    assert "provider interval" in reference_table and "交易所报价时间/年龄未知" in reference_table
    assert "不是ETF正式收盘价" in reference_table
    assert "不自动改变股票方向或仓位预算" in reference_table


def test_options_require_aligned_underlying_date_spot_and_fetch_metadata(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    dated_report_inputs(paths)
    aligned = {
        "contracts": [{"symbol": "test-call"}], "greeks": {"delta": 0.5},
        "underlying_date": "2026-06-05", "underlying_spot": 100,
        "fetched_at": "2026-06-06T16:00:00Z", "warnings": [],
    }
    option_path = paths.processed / "options-2026-06-06.json"
    write_json(option_path, {"QQQ": aligned})
    bundle = build_report_bundle(paths, "2026-06-06", DEFAULT_WATCHLIST[:2])
    assert bundle.watchlist_table[1]["option_status"] == "available"
    assert "QQQ" in bundle.options_context
    for change in ({"underlying_date": "2026-06-04"}, {"underlying_spot": 101},
                   {"fetched_at": None}):
        write_json(option_path, {"QQQ": {**aligned, **change}})
        bundle = build_report_bundle(paths, "2026-06-06", DEFAULT_WATCHLIST[:2])
        assert bundle.watchlist_table[1]["option_status"] == "unavailable"
        assert "QQQ" not in bundle.options_context
        assert any("option context withheld" in warning for warning in bundle.risk_flags)


def test_default_comprehensive_report_preserves_all_sections_and_crypto(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    dated_report_inputs(paths)
    nvda = next(asset for asset in DEFAULT_WATCHLIST if asset.ticker == "NVDA")
    bundle = build_report_bundle(paths, "2026-06-06", [*DEFAULT_WATCHLIST[:2], nvda])
    bundle.cross_market = [{
        "symbol": symbol, "reference_for": symbol, "asset_class": "crypto",
        "asset_type": "crypto_perpetual", "status": "reference_only", "source": "Hyperliquid",
        "fetched_at": "2026-06-06T12:00:00-04:00", "mark_price": price,
    } for symbol, price in (("BTC", 70000), ("ETH", 2500))]
    bundle.shunge_context = shunge_context_fixture()
    bundle.analyst_study = {"coverage": {"unique_posts": 10}}
    main = render_markdown(bundle)
    assert main == render_markdown(bundle, detailed=True)
    assert main.startswith("# Trading Wizzard 日报")
    for section in ("今日结论", "NVDA", "BTC / 加密永续", "ETH / 加密永续", "## Watchlist",
                    "## Macro Radar", "## Cash-Flow Data Coverage", "## Per-Ticker Notes",
                    "## Setup And Data Quality Notes", "### 板块观察"):
        assert section in main
    assert "$70,000.00" in main and "$2,500.00" in main
    assert "顺哥" not in main and "鹰哥" not in main
    md_path, html_path = write_report_files(paths, bundle)
    from twiz.reporting import render_daily_markdown

    assert md_path.read_text() == render_daily_markdown(bundle)
    assert "BTC / 加密永续" in html_path.read_text()
    assert "NVDA" in html_path.read_text()
    assert (paths.reports / "2026-06-06-details.md").read_text() == main
    assert (paths.reports / "2026-06-06-details.html").is_file()


def test_optional_compact_events_do_not_filter_comprehensive_report(tmp_path):
    from twiz.reporting import _main_event_rows

    paths = ProjectPaths(tmp_path)
    paths.ensure()
    bundle = build_report_bundle(paths, "2026-06-06", DEFAULT_WATCHLIST[:2])
    bundle.prediction_markets = [
        prediction_quote(id="fed", label="美联储", question="Will the Fed hold rates?"),
        prediction_quote(id="iran-short", label="短期停火", question="Iran ceasefire through June?",
                         end_date="2026-06-30T20:00:00Z"),
        prediction_quote(id="iran-long", label="长期停火",
                         question="Iran ceasefire through December?",
                         end_date="2026-12-31T20:00:00Z"),
        prediction_quote(id="politics", label="民主党控制众议院",
                         question="Will Democrats control the House?"),
    ]
    assert [row["id"] for row in _main_event_rows(bundle)] == ["fed", "iran-short"]
    compact = render_markdown(bundle, detailed=False)
    assert "民主党" not in compact and "长期停火" not in compact
    assert "美联储" in compact and "短期停火" in compact
    comprehensive = render_markdown(bundle)
    assert "民主党" in comprehensive and "长期停火" in comprehensive


def test_main_chain_requires_exact_verified_equity_mapping(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    dated_report_inputs(paths)
    bundle = build_report_bundle(paths, "2026-06-06", DEFAULT_WATCHLIST[:2])
    baseline_stance = bundle.daily_brief["stance"]
    baseline_plan = dict(bundle.action_plan)
    row = {"reference_for": "QQQ", "symbol": "QQQ", "source": "Lighter", "market_id": 129,
           "equity_mapping_verified": True, "instrument_name": "QQQ",
           "underlying_label": "QQQ perpetual",
           "asset_type": "equity_perpetual", "mapping_verified": True,
           "mapping_basis": "Exact active market ID verified.", "source_url": "https://example.test/spec",
           "status": "reference_only", "fetched_at": "2026-06-06T12:00:00-04:00",
           "mark_price": 100.5, "day_change_pct": 0.5}
    bundle.cross_market = [row]
    assert "100.50" in render_markdown(bundle)
    bundle.cross_market = [{**row, "mapping_verified": False}]
    assert "100.50" not in render_markdown(bundle)
    bundle.cross_market = [{**row, "asset_type": "index_perpetual"}]
    assert "100.50" not in render_markdown(bundle)
    assert bundle.daily_brief["stance"] == baseline_stance and bundle.action_plan == baseline_plan


def shunge_context_fixture():
    return {
        "scenario": {"active": True, "ticker": "QQQ", "source_date": "2026-10-21",
                     "data_date": "2026-10-27", "close": 721.25, "support_floor": 704,
                     "data_source": "yfinance", "session_complete": True,
                     "source": "data/raw/timelines/october-2026.txt",
                     "support_ceiling": 705, "confirmation": 725, "reversal": 740,
                     "extreme_zone": [645, 655], "review_after": "2026-10-30",
                     "state": "defensive_watch", "floor_breach": None},
        "case": {"available": True, "as_of": "2026-10-21", "source_support": [704, 705],
                 "source_target": 650, "price_basis": "事后下载、截断至发言日的Yahoo历史日线",
                 "range_high": 758.23, "range_high_date": "2026-10-05",
                 "measured_target": 650.77, "sma200": 676.49, "weekly_sma30": 699.84,
                 "weekly_sma50": 664.24, "prior_friday": {"date": "2026-10-16", "low": 707.23},
                 "prior_zone_hits": [{"date": "2026-04-10", "low": 650, "high": 654, "close": 652}],
                 "method_source_url": "https://example.test/range-method.pdf", "limitations": []},
        "patterns": [{"name": f"模式{i}", "interpretation": "条件判断与仓位分开。",
                      "evidence": [{"date": "2026-10-21", "message_time": "21:00",
                                    "source": "data/raw/timelines/private-long-path.txt",
                                    "line": 85, "summary": "先守支撑，再谈压力情景。"}]}
                     for i in range(1, 8)],
        "source_notes": ["这是来源归纳与解释假设，不是验证过的胜率模型。"],
    }


def test_historical_shunge_case_is_study_only_after_action(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    bundle = build_report_bundle(paths, "2026-06-06", DEFAULT_WATCHLIST[:2])
    bundle.shunge_context = shunge_context_fixture()
    bundle.daily_brief["expected_data_date"] = "2026-10-27"
    daily = render_markdown(bundle)
    assert "顺哥" not in daily and "鹰哥" not in daily
    assert "704.50" not in daily and "650.77" not in daily
    rendered = render_research_markdown(bundle)
    assert "不参与当前区域、买入、止盈或仓位计算" in rendered
    section = rendered.split("## 顺哥模式与历史案例复核", 1)[1].split("## 数据与技术明细", 1)[0]
    assert "707.23" in section and "不能写成已核验的精确前周五最低价" in section
    assert "2 × 704.50 − 758.23" in section and "650.77" in section
    assert "不能证明作者用了该公式" in section and "不能证明650必达或必守" in section
    assert "区间高度投影的方法参考" in section
    assert section.index("30周线 699.84") < section.index("200日线 676.49")
    assert section.index("200日线 676.49") < section.index("50周线 664.24")
    assert "2026-10-21 21:00" in section and "模式7" in section
    assert "data/raw/" not in section

def test_current_plan_joins_policy_and_ignores_author_levels(tmp_path, monkeypatch):
    from copy import deepcopy
    from unittest.mock import Mock

    paths = ProjectPaths(tmp_path)
    paths.ensure()
    report_date = "2026-10-27"
    write_json(paths.raw / f"market-{report_date}.json", {
        ticker: [{"date": report_date, "source": "yfinance", "session_complete": True,
                  "close": close, "open": close-0.5, "high": close+1, "low": close-1,
                  "volume": 1000}]
        for ticker, close in (("SPY", 769.66), ("QQQ", 721.25))
    })
    wrapper = {"source": "yfinance", "ticker": "QQQ", "as_of": "2026-10-21",
               "price_basis": "unadjusted", "rows": []}
    write_json(paths.processed / "shunge-case-QQQ-2026-10-21.json", wrapper)
    structure = {"status": "ready", "by_ticker": {"QQQ": {"status": "ready"}}}
    scenario = {"state": "defensive", "policy": {"initial_cap_pct": 1,
                 "post_event_cap_pct": 3.5, "max_new_capital_pct": 10, "reason": "事件仍待确认"}}
    old_case = shunge_context_fixture()
    author_builder = Mock(return_value=old_case)
    plan_builder = Mock(side_effect=lambda **kwargs: deepcopy(conditional_action_plan()))
    monkeypatch.setattr("twiz.reporting.build_shunge_context", author_builder)
    monkeypatch.setattr("twiz.reporting.build_market_structure", Mock(return_value=structure))
    monkeypatch.setattr("twiz.reporting.build_scenario_context", Mock(return_value=scenario))
    monkeypatch.setattr("twiz.reporting.build_dynamic_action_plan", plan_builder)
    bundle = build_report_bundle(paths, report_date, DEFAULT_WATCHLIST[:2])
    policy = plan_builder.call_args.kwargs["policy"]
    assert policy["initial_cap_pct"] == 1 and policy["post_event_cap_pct"] == 3.5
    assert policy["max_new_capital_pct"] == 10
    assert plan_builder.call_args.kwargs["structure"] == structure
    assert "shunge_context" not in plan_builder.call_args.kwargs
    assert author_builder.call_args.kwargs["case_history"] == wrapper
    assert bundle.market_structure == structure and bundle.scenario_context == scenario
    assert [row["price"] for row in bundle.action_plan["entries"]] == [700, 690, 680]
    assert bundle.action_plan["stop"]["price"] == 675
    old_case["scenario"]["support_floor"] = 9999
    old_case["case"]["source_target"] = 1
    second = build_report_bundle(paths, report_date, DEFAULT_WATCHLIST[:2])
    assert second.action_plan == bundle.action_plan


def test_strategy_leads_with_main_case_and_current_data_zones_before_action(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    dated_report_inputs(paths)
    bundle = build_report_bundle(paths, "2026-06-06", DEFAULT_WATCHLIST[:2])
    bundle.scenario_context = {
        "headline": "短线偏多看修复，当前不追价。", "state": "repair_bias",
        "causal_chain": ["Fed维持利率Yes 96%，按65%门槛选主情景", "10年美债回落、信用未扩大"],
        "events": [{"driver": "fed_hold"}],
        "macro_features": {"yield10": {"value": 4.99}, "credit": {"value": 3.3}},
        "policy": {"initial_cap_pct": 2, "post_event_cap_pct": 5},
        "branches": [{"name": "反向风险", "trigger": "收益率与信用同时恶化",
                       "market_implication": "撤回修复预期", "action": "取消未成交加仓"}],
    }
    zone = {"lower": 81, "upper": 84, "center": 82.5, "credible": True,
            "origins": [{"label": "sma200"}, {"label": "confirmed_pivot_low"}]}
    bundle.market_structure = {"by_ticker": {"QQQ": {
        "status": "ready", "price_date": "2026-06-05", "close": 100,
        "zones": {"support": None, "deep": zone, "extreme": None},
    }}}
    bundle.action_plan = conditional_action_plan()
    rendered = render_markdown(bundle)
    top = rendered.split("### 今日行动卡", 1)[0]
    assert "短线偏多看修复，当前不追价" in top
    assert "Fed维持利率Yes 96%" in rendered and "不是QQQ上涨概率" in rendered
    assert "深回调区 81.00–84.00" in rendered and "200日均线、已确认摆动低点" in rendered
    assert "当前事件阶段的首批政策上限为总账户2%" in rendered
    assert "复核合计 5%" in rendered and "实际可启用额以行动卡为准" in rendered
    assert "取消未成交加仓" in rendered
    assert "已有利率与信用观察" in rendered
    assert "未知，等待有效利率与信用数据" not in rendered
    assert rendered.index("今日行动卡") < rendered.index("主情景如何转向行动")
    bundle.market_structure["by_ticker"]["QQQ"]["zones"]["deep"]["lower"] = 78
    assert "深回调区 78.00–84.00" in render_markdown(bundle)


def test_analyst_learning_reports_real_coverage_and_filters_future_rules(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    write_json(paths.processed / "eagle-corpus.json", {
        "coverage": {"first_date": "2026-05-04", "last_date": "2026-10-24",
                     "unique_posts": 163, "duplicates_removed": 31, "unresolved_dates": 4},
        "messages": [{"date": "2026-05-04"}, {"date": "2026-10-24"}, {"date": None}],
        "limitations": ["选摘不是完整一年"],
    })
    (paths.config / "analyst-patterns.yml").write_text("""comparisons:
  - name: 可用规则
    available_from: '2026-06-01'
    common: 等待反应
    eagle: 有条件
    shunge: 有期限
    difference: 不照抄数值
    failure_conditions: 信用恶化
    evidence:
      - {author: Eagle, date: '2026-05-04', summary: 原文证据, source: private-path, line: 5}
  - name: 未来规则
    available_from: '2026-10-01'
    evidence: [{date: '2026-10-01'}]
resources:
  - name: SMC来源
    url: https://www.youtube.com/watch?v=caTouoQys7o
    access_level: description_only
    finding: 只核验说明
notes: ['20MV定义未解']
""")
    bundle = build_report_bundle(paths, "2026-06-06", DEFAULT_WATCHLIST[:2])
    assert len(bundle.analyst_study["comparisons"]) == 1
    assert bundle.analyst_study["comparisons"][0]["evidence"][0]["line"] == 5
    assert "鹰哥与顺哥" not in render_markdown(bundle)
    section = render_research_markdown(bundle).split("## 鹰哥与顺哥", 1)[1]
    assert "163篇唯一帖子" in section and "31篇重复" in section and "4篇日期未解" in section
    assert "可定日的帖子为1篇" in section
    assert "不是完整一年" in section and "发布时区未知" in section
    assert "可用规则" in section and "未来规则" not in section
    assert "未取得字幕正文" in section and "20MV定义未解" in section
    assert "private-path" not in section


def test_pending_spy_clears_serialized_qqq_trade_budget(tmp_path, monkeypatch):
    from unittest.mock import Mock

    paths = ProjectPaths(tmp_path)
    paths.ensure()
    write_json(paths.raw / "market-2026-10-27.json", {
        "QQQ": [{"date": "2026-10-27", "source": "yfinance", "close": 721.25}],
        "SPY": [{"date": "2026-10-26", "source": "yfinance", "close": 765}],
    })
    funded = conditional_action_plan()
    funded.update(buy_now_pct=2, selected_zone={"lower": 680}, selected_tier="deep",
                  signal={"buy_signal": True})
    funded["budget"]["estimated_weighted_entry"] = 690
    monkeypatch.setattr("twiz.reporting.build_dynamic_action_plan", Mock(return_value=funded))
    bundle = build_report_bundle(paths, "2026-10-27", DEFAULT_WATCHLIST[:2])
    assert bundle.action_plan["status"] == "unavailable"
    assert bundle.action_plan["budget"]["planned_capital_pct"] == 0
    assert bundle.action_plan["budget"]["currently_enabled_pct"] == 0
    assert bundle.action_plan["budget"]["uncommitted_cash_pct"] == 10
    assert bundle.action_plan["budget"]["estimated_weighted_entry"] is None
    assert bundle.action_plan["selected_zone"] is None
    assert bundle.action_plan["selected_tier"] is None
    assert bundle.action_plan["signal"]["buy_signal"] is False
    assert not bundle.action_plan["entries"] and not bundle.action_plan["exits"]
    assert not bundle.action_plan["stop"] and bundle.action_plan["buy_now_pct"] == 0


def test_global_policy_snapshot_dates_and_unknown_disclosures():
    from twiz.reporting import _render_global_context

    context = {
        "as_of": "2026-10-27", "policy_rates": [
            {"country": "日本", "value": 1.25, "rate_type": "操作目标",
             "announced_on": "2026-09-18", "effective_from": "2026-09-24",
             "verified_as_of": "2026-10-27", "source_url": "https://example.test/boj"},
            {"country": "未来政策", "value": 5, "announced_on": "2026-10-28",
             "verified_as_of": "2026-10-27", "source_url": "https://example.test/future"},
        ], "policy_events": [
            {"person": "Trump", "document": "文书", "signed_on": "2026-10-16",
             "published_on": "2026-10-21", "fact": "原文附条件豁免。",
             "source_url": "https://example.test/policy"},
        ], "disclosures": [
            {"person": "Pelosi", "status": "unavailable",
             "source_url": "https://example.test/disclosure"},
        ],
    }
    lines = "\n".join(_render_global_context(context, "2026-10-27"))
    assert "1.25%" in lines and "2026-09-18公布、2026-09-24生效" in lines
    assert "未来政策" not in lines
    assert "2026-10-16签署、2026-10-21公开" in lines
    assert "不单凭文件标题生成股市方向" in lines
    assert "当前具体申报未核验" in lines and "不作为实时持仓或买入信号" in lines
    historical = "\n".join(_render_global_context(context, "2026-06-06"))
    assert "本日背景待复核" in historical
    assert "1.25%" not in historical and "Trump" not in historical


def test_learning_replay_uses_prior_session_and_never_claims_verified_author_indicator(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    replay = {
        "published_date": "2026-06-08", "ticker": "^GSPC", "price_cutoff": "2026-06-05",
        "status": "reconstructed", "author_rule": "站回20天线并守三天",
        "features": {"close": 100, "sma20_proxy": 101, "sma60_proxy": 99,
                     "prior40_low": 90, "prior40_high": 110,
                     "public_9bar_proxy": {"buy_count": 7, "sell_count": 0}},
        "limits": "代理不是作者公式，也不是预测结果。",
    }
    write_json(paths.processed / "analyst-replays-2026-10-27.json", [
        replay, {**replay, "published_date": "2026-10-28", "ticker": "FUTURE"},
        {**replay, "price_cutoff": "2026-06-08", "ticker": "SAME_DAY"},
    ])
    bundle = build_report_bundle(paths, "2026-10-27", DEFAULT_WATCHLIST[:2])
    assert len(bundle.analyst_study["replays"]) == 1
    learning = render_research_markdown(bundle).split("## 鹰哥与顺哥", 1)[1]
    assert "截至2026-06-05" in learning and "SMA20代理101.00" in learning
    assert "事后下载的历史价格" in learning and "不据此宣称预测命中" in learning
    assert "作者版本未核实" in learning and "买序7" in learning
    assert "FUTURE" not in learning and "SAME_DAY" not in learning


def test_action_card_discloses_engineering_rungs_spacing_and_missing_real_targets():
    from twiz.reporting import _render_action_plan

    plan = conditional_action_plan()
    plan["entry_reference"] = "由高到低观察；后续各档须新报告独立确认。"
    plan["entries"][0]["price_origin"] = "区域上沿+1ATR工程观察价，非独立历史枢轴"
    plan["spacing_assessment"] = {
        "first_entry_distance_atr": .158, "first_target_distance_atr": 2.055,
        "verdict": "首档贴近现价，不能称为深度抄底。", "target_note": "不足目标档位留空。",
    }
    plan["exits"][0].update(reward_risk_multiple=1.057, source_date="2026-06-01",
                            reference_kind="confirmed_price_pivot")
    plan["exits"][1].update(price=None, status="没有符合2R的来源目标，留空重评")
    text = "\n".join(_render_action_plan(plan))
    assert "由高到低观察" in text and "各档须新报告独立确认" in text
    assert "工程观察价，非独立历史枢轴" in text
    assert "0.16倍日均真实波幅" in text and "不能称为深度抄底" in text
    assert "上方2.06ATR" in text and "约1.06R" in text
    assert "2026-06-01 已确认价格枢轴" in text
    assert "待确认 | 30%" in text and "没有符合2R的来源目标" in text
