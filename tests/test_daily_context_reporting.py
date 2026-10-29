import copy
import json
from pathlib import Path

import pytest

from twiz.cloud_capex import CAPEX, REVENUE, build_cloud_capex_timeline
from twiz.config import DEFAULT_WATCHLIST
from twiz.provenance import begin_analysis_snapshot, finalize_analysis_snapshot, verify_snapshot
from twiz.reporting import build_report_bundle, render_html, render_markdown
from twiz.storage import ProjectPaths, write_json

DAY = "2026-06-06"


def project(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    write_json(paths.raw / f"market-{DAY}.json", {
        ticker: [{"date": "2026-06-05", "source": "yfinance", "close": 100,
                  "session_complete": True}] for ticker in ("SPY", "QQQ")})
    return paths


def direction():
    return {"date": DAY, "horizon_sessions": 20, "by_ticker": {
        ticker: {"status": "available", "price_date": "2026-06-05",
                 "available_at": "2026-06-05T16:15:00-04:00",
                 "probability_up": .62, "probability_down": .38,
                 "model_kind": "dated_price_prior", "validation": {
                     "brier": .24, "prior_brier": .26, "samples": 80},
                 "contributions": [{"factor": "信用条件", "log_odds_contribution": -.1}]}
        for ticker in ("QQQ", "SPY")}}


def cloud():
    data, hashes = {}, {}
    identities = (("MSFT", 789019), ("GOOGL", 1652044), ("META", 1326801))
    for index, (ticker, cik) in enumerate(identities):
        facts = {CAPEX: {"units": {"USD": []}}, REVENUE[0]: {"units": {"USD": []}}}
        for year, cash, revenue in ((2023, 100, 500), (2024, 150, 600), (2025, 300, 750)):
            for tag, value in ((CAPEX, cash), (REVENUE[0], revenue)):
                start = f"{year-1}-07-01" if ticker == "MSFT" else f"{year}-01-01"
                facts[tag]["units"]["USD"].append({
                    "start": start, "end": f"{year}-06-30", "filed": "2026-06-04",
                    "val": value, "form": "10-K", "accn": "0000000000-26-000001"})
        data[ticker] = {"cik": cik, "facts": {"us-gaap": facts}}
        hashes[ticker] = "abc"[index] * 64
    return build_cloud_capex_timeline(data, ["2026-06-05"], hashes)["rows"][0]


def test_dated_contexts_load_and_render_stock_probabilities_separate_from_event_odds(tmp_path):
    paths = project(tmp_path)
    write_json(paths.processed / f"direction-probabilities-{DAY}.json", direction())
    write_json(paths.processed / f"cloud-capex-{DAY}.json", cloud())
    bundle = build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2])
    bundle.prediction_markets = [{"yes_probability": .9, "status": "unavailable"}]
    text = render_markdown(bundle)
    assert bundle.directional_context == direction()
    assert "62.0%" in text and "38.0%" in text and "验证Brier 0.240" in text
    probability_table = text.split("### 未来20个交易日方向概率", 1)[1].split("###", 1)[0]
    assert "90.0%" not in probability_table and "事件市场Yes/No另列" in probability_table
    assert text.index("今日行动卡") < text.index("未来20个交易日方向概率")
    assert "MSFT | 2024-07-01至2025-06-30" in text
    assert "GOOGL | 2025-01-01至2025-06-30" in text
    assert "+100.0% | +25.0% | +50.0个百分点" in text
    assert "非环比或物理加速度" in text and "Amazon口径缺口保持缺失" in text
    assert "https://www.sec.gov/Archives/edgar/data/789019/000000000026000001" in text


@pytest.mark.parametrize("change", [
    {"price_date": "2026-06-04"}, {"available_at": "2027-01-01T16:15:00-05:00"},
    {"available_at": None}, {"expires_at": "2026-06-05T17:00:00-04:00"},
    {"expires_at": "not-a-clock"}, {"probability_up": float("nan")},
    {"probability_up": True}, {"probability_down": .9}, {"model_kind": ""},
    {"status": "unavailable", "reason": "验证未改善原价量基准"},
    {"origin_date": "2026-06-04"}, {"refit_at": "2026-06-06T16:15:00-04:00"},
    {"price_close": 90},
])
def test_unusable_direction_never_fills_half_or_copies_event_probability(tmp_path, change):
    paths = project(tmp_path)
    bundle = build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2])
    bundle.directional_context = direction()
    bundle.directional_context["by_ticker"] = {
        "QQQ": {**direction()["by_ticker"]["QQQ"], **change}}
    bundle.prediction_markets = [{"yes_probability": .95}]
    text = render_markdown(bundle).split("### 未来20个交易日方向概率", 1)[1].split("###", 1)[0]
    assert "| QQQ | 未知 | 未知 |" in text
    assert "50.0%" not in text and "95.0%" not in text


def test_wrong_context_date_horizon_and_sample_price_are_not_usable(tmp_path):
    paths = project(tmp_path)
    bundle = build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2])
    for key, value in (("date", "2026-06-05"), ("horizon_sessions", 5)):
        bundle.directional_context = {**direction(), key: value}
        assert "62.0%" not in render_markdown(bundle)
    bundle.directional_context = direction()
    for row in bundle.watchlist_table:
        row["data_source"] = "sample"
    assert "62.0%" not in render_markdown(bundle)


def test_cloud_future_snapshot_or_filed_fact_is_rejected_before_numeric_metrics(tmp_path):
    class PoisonMetrics(dict):
        def get(self, key, default=None):
            if key == "cash_capex_yoy_pct":
                raise AssertionError("unavailable financial numeric read")
            return super().get(key, default)

    paths = project(tmp_path)
    bundle = build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2])
    context = cloud()
    bundle.cloud_capex_context = {**context, "available_at": "2027-01-01T16:15:00-05:00"}
    assert "+100.0%" not in render_markdown(bundle)
    context = copy.deepcopy(context)
    for ref in context["source_refs"].values():
        ref["filed"] = "2027-01-01"
    context["issuers"] = {k: PoisonMetrics(v) for k, v in context["issuers"].items()}
    bundle.cloud_capex_context = context
    assert "+100.0%" not in render_markdown(bundle)


def test_cloud_issuer_identity_and_older_acceleration_period_cannot_be_relabelled(tmp_path):
    paths = project(tmp_path)
    bundle = build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2])
    context = cloud()
    context["issuers"] = {"MSFT": context["issuers"]["GOOGL"]}
    bundle.cloud_capex_context = context
    assert "+100.0%" not in render_markdown(bundle)
    context = cloud()
    for issuer in context["issuers"].values():
        issuer["previous_same_duration"]["period"]["end"] = "2020-06-30"
    bundle.cloud_capex_context = context
    assert "+50.0个百分点" not in render_markdown(bundle)


def test_inactive_contributions_and_frequency_fallback_are_not_told_as_active_causes(tmp_path):
    bundle = build_report_bundle(project(tmp_path), DAY, DEFAULT_WATCHLIST[:2])
    bundle.directional_context = direction()
    for row in bundle.directional_context["by_ticker"].values():
        row.update(model_kind="historical_base_rate", contributions=[{
            "factor": "不应显示", "active": False, "log_odds_contribution": .8}])
    text = render_markdown(bundle)
    assert "历史频率基准" in text and "当前仅为历史频率参考" in text
    assert "不应显示" not in text and "原价量基准" not in text


def test_options_execution_evidence_requires_fresh_aligned_unexpired_snapshot(tmp_path):
    paths = project(tmp_path)
    bundle = build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2])
    snapshot = {"underlying_date": "2026-06-05", "underlying_spot": 100,
                "fetched_at": "2026-06-06T12:00:00-04:00", "expiration": "2026-06-12",
                "contracts": [{"strike": 100}],
                "skew_summary": {"expected_move_pct": 3.2, "put_call_oi_ratio": 1.4}}
    bundle.options_context = {"QQQ": snapshot}
    assert "±3.20%" in render_markdown(bundle)
    assert "P/C持仓比1.40" in render_markdown(bundle)
    for change in ({"expiration": "2026-06-05"}, {"expiration": "bad"},
                   {"underlying_spot": 110}, {"fetched_at": "2027-01-01T00:00:00Z"}):
        bundle.options_context = {"QQQ": {**snapshot, **change}}
        assert "±3.20%" not in render_markdown(bundle)


def test_new_inputs_and_source_rows_are_archived_and_linked(tmp_path):
    paths = project(tmp_path)
    write_json(paths.processed / f"direction-probabilities-{DAY}.json", direction())
    write_json(paths.processed / f"cloud-capex-{DAY}.json", cloud())
    draft = begin_analysis_snapshot(paths, DAY, source_root=tmp_path)
    bundle = build_report_bundle(paths, DAY, DEFAULT_WATCHLIST[:2])
    receipt = finalize_analysis_snapshot(draft, bundle)
    manifest = verify_snapshot(Path(receipt["path"]))["manifest"]
    originals = {row.get("original_path") for row in manifest["files"]}
    assert f"data/processed/direction-probabilities-{DAY}.json" in originals
    assert f"data/processed/cloud-capex-{DAY}.json" in originals
    links = json.loads((Path(receipt["path"]) / "decision-input-refs.json").read_text())
    assert any(r["kind"] == "equity_direction_model" and r["status"] == "matched" for r in links)
    assert any(r["kind"] == "cloud_cash_capex" and r["status"] == "matched" for r in links)


def test_audit_is_a_short_safe_link_not_a_manifest_dump(tmp_path):
    bundle = build_report_bundle(project(tmp_path), DAY, DEFAULT_WATCHLIST[:2])
    bundle.analysis_run = {"run_id": "20260606T200000Z-test",
                           "input_manifest_sha256": "SECRET_HASH"}
    text = render_markdown(bundle)
    assert "SECRET_HASH" not in text
    assert "../data/runs/2026-06-06/20260606T200000Z-test/manifest.json" in text
    assert 'href="../data/runs/2026-06-06/20260606T200000Z-test/manifest.json"' in render_html(
        text, "日报")
