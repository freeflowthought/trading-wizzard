from datetime import datetime, timezone
from pathlib import Path

from twiz.config import DEFAULT_WATCHLIST
from twiz.provenance import begin_analysis_snapshot, finalize_analysis_snapshot, verify_snapshot
from twiz.reporting import (
    build_report_bundle,
    render_daily_markdown,
    render_markdown,
    write_report_files,
)
from twiz.storage import ProjectPaths, write_json

DAY = "2026-06-06"


def report(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    write_json(paths.raw / f"market-{DAY}.json", {
        ticker: [{"date": "2026-06-05", "close": 100, "source": "yfinance",
                  "session_complete": True}] for ticker in ("SPY", "QQQ")})
    bundle = build_report_bundle(paths, DAY, DEFAULT_WATCHLIST)
    bundle.action_plan = {
        "status": "conditional_plan", "ticker": "QQQ", "buy_now_pct": 0,
        "current_action": "当前不追涨，等待价格收复及后续回踩。",
        "budget": {"max_new_capital_pct": 100, "currently_enabled_pct": 2},
        "entries": [{"batch": i, "price": price, "account_pct": 10,
                     "currently_enabled_pct": 2 if i == 1 else 0,
                     "price_origin": "不在主文显示的工程价梯ATR细节"*20}
                    for i, price in enumerate((95, 90, 85), 1)],
        "exits": [{"stage": i, "price": price, "trim_original_dip_pct": trim}
                  for i, (price, trim) in enumerate(((110, 40), (120, 30), (130, 30)), 1)],
        "stop": {"price": 80},
    }
    bundle.per_ticker_notes = {asset.ticker: ["长篇技术与来源明细"*100]
                              for asset in DEFAULT_WATCHLIST}
    bundle.cross_market = [{"symbol": ticker, "reference_for": ticker, "source": "Hyperliquid",
                            "asset_type": "crypto_perpetual", "status": "reference_only",
                            "fetched_at": "2026-06-06T12:00:00-04:00", "mark_price": price}
                           for ticker, price in (("BTC", 70000), ("ETH", 2500))]
    bundle.scenario_context["rotation"] = [
        {"ticker": "XLU", "name": "公用事业", "end_date": "2026-06-05",
         "return_20d_pct": 2, "vs_spy_pp": 1.2}]
    return paths, bundle


def test_primary_is_compact_but_keeps_all_assets_four_references_and_defensive_rotation(tmp_path):
    _, bundle = report(tmp_path)
    primary = render_daily_markdown(bundle)
    assert len(primary.splitlines()) <= 220
    for asset in DEFAULT_WATCHLIST:
        assert f"| {asset.ticker} |" in primary
    for ticker in ("BTC", "ETH", "SPY", "QQQ"):
        assert f"| {ticker} /" in primary
    assert "$70,000.00" in primary and "公用事业（XLU）" in primary
    assert "95.00 | 10% | 2%" in primary and "110.00 | 40%" in primary
    assert "日收盘跌破 80.00" in primary
    assert "未来20个交易日方向概率" in primary and "下跌／持平" in primary
    for absent in ("Brier", "工程价梯", "长篇技术", "顺哥", "鹰哥", "| 框架 |"):
        assert absent not in primary
    assert f"[完整数据与来源]({DAY}-details.html)" in primary


def test_writing_primary_keeps_full_content_in_details_and_legacy_render_api(tmp_path):
    paths, bundle = report(tmp_path)
    full = render_markdown(bundle)
    markdown, html = write_report_files(paths, bundle)
    assert markdown.read_text() == render_daily_markdown(bundle)
    details = paths.reports / f"{DAY}-details.md"
    assert details.read_text() == full
    assert "长篇技术与来源明细" in details.read_text()
    assert "BTC / 加密永续" in details.read_text() and "## Watchlist" in full
    assert (paths.reports / f"{DAY}-details.html").is_file()
    assert f'href="{DAY}-details.html"' in html.read_text()
    assert len(markdown.read_bytes()) < len(details.read_bytes()) / 3


def test_known_past_event_expiry_is_waiting_for_results_not_upcoming(tmp_path):
    _, bundle = report(tmp_path)
    bundle.scenario_context["headline"] = "关键事件临近，暂缓新增。"
    bundle.action_plan["current_action"] = "关键事件临近，等待结果。"
    bundle.prediction_markets = [{"id": "fed", "label": "美联储决议", "status": "unavailable",
                                  "end_date": "2026-06-06T14:00:00-04:00"}]
    primary = render_daily_markdown(bundle)
    assert "关键事件临近" not in primary
    assert "关键事件结果待核验" in primary
    assert "已到合约期限，结果待核验" in primary
    assert "Yes 0.0%" not in primary
    assert "关键事件临近" not in render_markdown(bundle)


def test_after_close_today_fomc_calendar_does_not_remain_upcoming_without_market_contract(
        tmp_path, monkeypatch):
    _, bundle = report(tmp_path)
    bundle.date = "2026-10-28"
    bundle.scenario_context["headline"] = "关键事件临近，暂缓新增。"
    bundle.action_plan["current_action"] = "关键事件临近，等待结果。"
    bundle.events = [{"date": bundle.date, "kind": "fomc", "precision": "scheduled"}]
    bundle.prediction_markets = []
    monkeypatch.setattr("twiz.reporting._report_clock", lambda _: datetime(
        2026, 10, 28, 22, 0, tzinfo=timezone.utc))
    primary = render_daily_markdown(bundle)
    assert "关键事件临近" not in primary
    assert "关键事件结果待核验" in primary


def test_material_events_are_ranked_bounded_not_limited_to_iran_or_fed(tmp_path, monkeypatch):
    _, bundle = report(tmp_path)
    monkeypatch.setattr("twiz.reporting._event_quote", lambda *args, **kwargs: (.7, ""))
    bundle.scenario_context["macro_features"] = {}
    bundle.prediction_markets = [{"id": str(i), "label": "重点事件"+str(i),
                                  "no_probability": .29,
                                  "question": f"Will event {i} happen by July 2026?",
                                  "importance_rank": i, "end_date": "2030-01-01T00:00:00Z",
                                  "url": f"https://polymarket.com/event/test-{i}"}
                                 for i in range(6, 0, -1)]
    primary = render_daily_markdown(bundle)
    for index in range(1, 6):
        assert f"Will event {index} happen by July 2026?" in primary
    assert "Will event 6" not in primary
    assert "事件概率不直接等于股指上涨概率" in primary
    assert "No 29.0%" in primary and "No 30.0%" not in primary


def test_ambiguous_group_label_never_hides_actual_rate_threshold_question(tmp_path, monkeypatch):
    _, bundle = report(tmp_path)
    monkeypatch.setattr("twiz.reporting._event_quote", lambda *args, **kwargs: (.785, ""))
    question = "Will the Fed increase interest rates by 25 bps at its December 2026 meeting?"
    bundle.prediction_markets = [{"id": "rates", "label": "Fed Decision in December?",
                                  "question": question, "no_probability": .215}]
    primary = render_daily_markdown(bundle)
    assert question in primary and "Yes 78.5%" in primary
    assert "Fed Decision in December?：Yes" not in primary
    bundle.prediction_markets[0]["question"] = "Long conditional question "*100
    primary = render_daily_markdown(bundle)
    assert "Yes 78.5%" not in primary and "主文不简写该合约赔率" in primary


def test_zero_priority_core_event_survives_five_factor_cap(tmp_path, monkeypatch):
    _, bundle = report(tmp_path)
    monkeypatch.setattr("twiz.reporting._event_quote", lambda *args, **kwargs: (.7, ""))
    bundle.scenario_context["macro_features"] = {}
    bundle.prediction_markets = [{"id": str(i), "importance_rank": i,
                                  "question": f"Will priority event {i} occur?",
                                  "no_probability": .3}
                                 for i in range(5, -1, -1)]
    primary = render_daily_markdown(bundle)
    assert "Will priority event 0 occur?" in primary
    assert "Will priority event 4 occur?" in primary
    assert "Will priority event 5 occur?" not in primary


def test_unavailable_reward_risk_and_stale_ticker_are_not_called_missing_prices(tmp_path):
    _, bundle = report(tmp_path)
    bundle.action_plan.update(status="unavailable", current_action="当前盈亏空间不足，不追价。")
    bundle.watchlist_table[0].update(stale=True, support_levels="STALE_SUPPORT")
    primary = render_daily_markdown(bundle)
    assert "当前盈亏空间不足，不追价" in primary and "STALE_SUPPORT" not in primary


def test_pending_prices_remove_primary_entry_exit_and_stop_prices(tmp_path):
    _, bundle = report(tmp_path)
    bundle.daily_brief["price_data_pending"] = ["QQQ待更新"]
    primary = render_daily_markdown(bundle)
    assert "95.00 | 10%" not in primary and "110.00 | 40%" not in primary
    assert "日收盘跌破 80.00" not in primary
    assert "当前操作价位暂停使用" in primary


def test_both_primary_and_details_are_sealed_and_tamper_detected(tmp_path):
    paths, bundle = report(tmp_path)
    draft = begin_analysis_snapshot(paths, DAY, source_root=tmp_path)
    md, html = write_report_files(paths, bundle)
    receipt = finalize_analysis_snapshot(draft, bundle, outputs={
        "report.md": md.read_bytes(), "report.html": html.read_bytes(),
        "details.md": (paths.reports / f"{DAY}-details.md").read_bytes(),
        "details.html": (paths.reports / f"{DAY}-details.html").read_bytes()})
    run = Path(receipt["path"])
    assert verify_snapshot(run)["valid"]
    (run / "details.md").write_text("altered")
    assert not verify_snapshot(run)["valid"]


def test_all_asset_observation_prices_are_complete_and_use_current_source_not_old_technical(
        tmp_path, monkeypatch):
    calls = []

    def clusters(rows, **kwargs):
        calls.append((rows, kwargs))
        return [{"kind": kind, "price": price} for kind, price in (
            ("support", 12345.67), ("support", 12340.89),
            ("resistance", 12360.12), ("resistance", 12370.34))]

    monkeypatch.setattr("twiz.reporting.cluster_support_resistance", clusters)
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    rows = [{"date": day, "source": "yfinance", "session_complete": True,
             "open": 12350, "close": 12350, "high": 12400, "low": 12300}
            for day in ("2026-06-04", "2026-06-05")]
    write_json(paths.raw / f"market-{DAY}.json", {"QQQ": rows})
    write_json(paths.processed / f"technical-{DAY}.json", {"QQQ": {
        "date": "2026-06-04", "levels": [{"kind": "support", "price": 999, "distance_pct": -1}]}})
    bundle = build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2])
    primary = render_daily_markdown(bundle)
    section = primary.split("## 全部观察标的", 1)[1].split("## 24小时", 1)[0]
    assert "12345.67 / 12340.89" in section and "12360.12 / 12370.34" in section
    assert "999.00" not in section and "candidates" not in section and "…" not in section
    assert len(calls) == 1 and calls[0][1] == {}  # Same default algorithm as cmd_analyze.
    assert section.index("通用聚类") < section.index("| 标的 |")
    lines = section.splitlines()
    divider = lines.index("| --- | ---: | --- | --- | --- | --- |")
    assert lines[divider+1].startswith("| ")  # No prose splitting the Markdown table.


def test_incomplete_ohlc_or_older_bars_do_not_generate_levels(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("twiz.reporting.cluster_support_resistance",
                        lambda *args: calls.append(args))
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    for rows in ([{"date": "2026-06-05", "source": "yfinance", "session_complete": True,
                   "close": 100}],
                 [{"date": day, "source": "yfinance", "session_complete": complete,
                   "open": 100, "high": 101, "low": 99, "close": 100}
                  for day, complete in (("2026-06-04", False), ("2026-06-05", True))]):
        write_json(paths.raw / f"market-{DAY}.json", {"QQQ": rows})
        bundle = build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2])
        row = next(r for r in bundle.watchlist_table if r["ticker"] == "QQQ")
        assert row["observation_support_prices"] == row["observation_resistance_prices"] == []
    assert not calls
