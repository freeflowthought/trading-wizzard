import copy
import gzip
import hashlib
import json
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from twiz import daily_context
from twiz.daily_context import BASIS, refresh_daily_context
from twiz.storage import ProjectPaths, read_json, write_json

DAY = "2026-06-08"
NOW = datetime(2026, 6, 8, 21, tzinfo=timezone.utc)


def bar(day="2026-06-08", price=100, raw=False, **extra):
    values = {"open": price, "high": price*1.02, "low": price*.98, "close": price}
    return {"date": day, "source": "yfinance", "session_complete": True, "volume": 100,
            "price_basis": BASIS if not raw else "legacy_dividend_adjusted",
            **values, **({"raw_"+key: value for key, value in values.items()} if raw else {}),
            **extra}


def project(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    return paths


def predict(rows, origin, **kwargs):
    return {"status": "available", "origin_date": origin, "model_kind": "price_only",
            "probability_up": .6, "probability_down": .4,
            "refit_at": origin+"T16:15:00-04:00", "training_samples": 100,
            "training_label_last": "2026-05-01", "model": {"beta": [0, .1]}}


def current():
    return {ticker: [bar(raw=True)] for ticker in ("QQQ", "SPY")}


def test_missing_local_history_is_unknown_and_does_not_break_report(tmp_path):
    paths = project(tmp_path)
    result = refresh_daily_context(paths, DAY, current(), analysis_clock=NOW)
    for row in result["directional_context"]["by_ticker"].values():
        assert row["status"] == "unavailable" and row["probability_up"] is None
        assert row["available_at"] == NOW.isoformat()
    assert result["cloud_capex_context"]["aggregate"]["status"] == "insufficient_issuers"
    assert (paths.processed / f"direction-training-{DAY}.json").is_file()


def test_raw_prices_and_actual_generation_clock(tmp_path, monkeypatch):
    paths = project(tmp_path)
    spy = Mock(side_effect=predict)
    monkeypatch.setattr(daily_context, "estimate_direction", spy)
    market = current()
    for rows in market.values():
        rows[0].update(close=80, provider_adjusted_close=80)
    result = refresh_daily_context(paths, DAY, market, analysis_clock=NOW)
    passed = spy.call_args_list[0].args[0]
    assert passed[-1]["close"] == 100 and passed[-1]["provider_adjusted_close"] == 80
    assert passed[-1]["price_basis"] == BASIS
    row = result["directional_context"]["by_ticker"]["QQQ"]
    assert row["price_close"] == 100 and row["price_date"] == DAY
    assert row["available_at"] == "2026-06-08T21:00:00+00:00"
    assert row["available_at"] != row["refit_at"]


def test_uniform_overlap_units_rescale_only_older_ohlc_and_replace_overlap(tmp_path, monkeypatch):
    paths = project(tmp_path)
    old = [bar(day, 200) for day in ("2026-06-03", "2026-06-04", "2026-06-05")]
    write_json(paths.processed / "direction-history.json", {"QQQ": old, "SPY": old, "OTHER": [1]})
    monkeypatch.setattr(daily_context, "estimate_direction", predict)
    incoming = {ticker: [bar("2026-06-05", raw=True), bar(raw=True)] for ticker in ("QQQ", "SPY")}
    refresh_daily_context(paths, DAY, incoming, analysis_clock=NOW)
    saved = read_json(paths.processed / "direction-history.json")
    assert saved["OTHER"] == [1]
    assert len(saved["QQQ"]) == 4 and all(row["close"] == 100 for row in saved["QQQ"])
    assert saved["QQQ"][0]["ohlc_reconciliation_factor"] == .5
    assert saved["QQQ"][0]["volume"] == 100
    trace = read_json(paths.processed / f"direction-training-{DAY}.json")
    assert trace["provider_reconciliation"]["QQQ"]["older_OHLC_factor"] == .5
    assert trace["used_price_history"]["QQQ"] == saved["QQQ"]


@pytest.mark.parametrize("change", [
    {"source": "sample"}, {"session_complete": False}, {"raw_close": None},
    {"fetched_at": "2027-01-01T00:00:00Z"}, {"fetched_at": "2026-06-08T20:00:00"},
])
def test_invalid_current_source_is_not_used(tmp_path, monkeypatch, change):
    spy = Mock(side_effect=predict)
    monkeypatch.setattr(daily_context, "estimate_direction", spy)
    market = {"QQQ": [bar(raw=True, **change)]}
    result = refresh_daily_context(project(tmp_path), DAY, market, analysis_clock=NOW)
    assert result["directional_context"]["by_ticker"]["QQQ"]["status"] == "unavailable"
    spy.assert_not_called()


def test_legacy_adjusted_without_raw_and_inconsistent_overlap_are_rejected(tmp_path, monkeypatch):
    paths = project(tmp_path)
    spy = Mock(side_effect=predict)
    monkeypatch.setattr(daily_context, "estimate_direction", spy)
    bad = bar(price_basis="dividend adjusted")
    result = refresh_daily_context(paths, DAY, {"QQQ": [bad]}, analysis_clock=NOW)
    assert "without_raw" in result["directional_context"]["by_ticker"]["QQQ"]["reason"]
    write_json(paths.processed / "direction-history.json", {"QQQ": [bar("2026-06-05", 200)]})
    incoming = [bar("2026-06-05", raw=True, raw_high=103), bar(raw=True)]
    result = refresh_daily_context(paths, DAY, {"QQQ": incoming}, analysis_clock=NOW)
    assert "inconsistent_provider_overlap" in (
        result["directional_context"]["by_ticker"]["QQQ"]["reason"])
    spy.assert_not_called()


def test_future_prices_unread_and_later_cache_preserved(tmp_path, monkeypatch):
    class Future(dict):
        def get(self, key, default=None):
            if key.startswith("raw_") or key in {"close", "open", "low", "high", "volume"}:
                raise AssertionError("future numeric price read")
            return super().get(key, default)

    paths = project(tmp_path)
    monkeypatch.setattr(daily_context, "estimate_direction", predict)
    history = {"QQQ": [bar("2026-06-05"), bar(), bar("2026-06-09", 110)]}
    write_json(paths.processed / "direction-history.json", history)
    result = refresh_daily_context(paths, DAY, {"QQQ": [bar(raw=True), Future(date="2027-01-01")]},
                                   analysis_clock=NOW)
    assert result["directional_context"]["by_ticker"]["QQQ"]["status"] == "available"
    assert read_json(paths.processed / "direction-history.json") == history


def capex_cache(clock="2026-06-05T16:15:00-04:00"):
    issuer = {"status": "available", "reason": None,
              "current": {kind: {"available_at_proxy": clock} for kind in ("capex", "revenue")},
              **dict.fromkeys(daily_context.METRICS, 20)}
    return {"source_capture_date": "2026-06-06", "rows": [{
        "date": "2026-06-05", "available_at": "2026-06-05T16:15:00-04:00",
        "issuers": {name: copy.deepcopy(issuer) for name in ("MSFT", "META")},
        "source_refs": {"original": {"filed": "2026-06-04", "source_sha256": "a"*64}}}]}


def test_capex_current_snapshot_never_repaints_fact_or_capture_clocks(tmp_path):
    paths = project(tmp_path)
    bundle = capex_cache()
    (paths.processed / "cloud-capex-history.json.gz").write_bytes(
        gzip.compress(json.dumps(bundle).encode()))
    result = refresh_daily_context(paths, DAY, {}, analysis_clock=NOW)["cloud_capex_context"]
    assert result["date"] == DAY and result["source_snapshot_date"] == "2026-06-05"
    assert result["source_capture_date"] == "2026-06-06"
    assert result["available_at"] == "2026-06-05T16:15:00-04:00"
    assert result["issuers"]["MSFT"]["current"]["capex"]["available_at_proxy"] == (
        "2026-06-05T16:15:00-04:00")
    assert result["aggregate"]["median_cash_capex_yoy_pct"] == 20
    assert result["source_refs"] == bundle["rows"][0]["source_refs"]


def test_stale_capex_missing_neutral_and_future_auxiliary_records_not_passed(tmp_path, monkeypatch):
    paths = project(tmp_path)
    bundle = capex_cache("2025-01-01T16:15:00-05:00")
    (paths.processed / "cloud-capex-history.json.gz").write_bytes(
        gzip.compress(json.dumps(bundle).encode()))
    write_json(paths.processed / "direction-event-history.json", [
        {"available_at": "2026-06-05T20:00:00Z", "direction": .5},
        {"available_at": "2027-01-01T20:00:00Z", "direction": 999}])
    spy = Mock(side_effect=predict)
    monkeypatch.setattr(daily_context, "estimate_direction", spy)
    result = refresh_daily_context(paths, DAY, current(), analysis_clock=NOW)
    assert result["cloud_capex_context"]["aggregate"]["status"] == "insufficient_issuers"
    assert result["cloud_capex_context"]["aggregate"]["median_cash_capex_yoy_pct"] is None
    assert len(spy.call_args.kwargs["event_rows"]) == 1


def test_live_macro_availability_and_unchanged_field_clocks(tmp_path, monkeypatch):
    paths = project(tmp_path)
    write_json(paths.processed / "direction-macro-history.json", [{
        "date": "2026-06-05", "available_at": "2026-06-05T20:15:00Z", "values": {"vix": 18},
        "source_observations": {"vix": {"available_at": "2026-06-05T20:15:00Z"}}}])
    write_json(paths.processed / f"macro-{DAY}.json", [
        {"source": "official", "series_id": "DGS10", "history": [
            {"date": DAY, "value": 3.8, "available_at": "2026-06-08T20:10:00Z"},
            {"date": "2026-06-09", "value": 99, "available_at": "2026-06-08T20:10:00Z"}]},
        {"source": "official", "series_id": "VIXCLS", "date": DAY, "value": 99}])
    spy = Mock(side_effect=predict)
    monkeypatch.setattr(daily_context, "estimate_direction", spy)
    refresh_daily_context(paths, DAY, current(), analysis_clock=NOW)
    macro = spy.call_args.kwargs["macro_rows"][-1]
    assert macro["values"] == {"vix": 18, "yield10_pct": 3.8}
    assert macro["source_observations"]["vix"]["available_at"] == "2026-06-05T20:15:00Z"


def test_training_trace_binds_source_bytes_and_model_failures_are_isolated(tmp_path, monkeypatch):
    paths = project(tmp_path)
    history = {"QQQ": [bar()], "SPY": [bar(price=200)]}
    write_json(paths.processed / "direction-history.json", history)
    raw = (paths.processed / "direction-history.json").read_bytes()

    def sometimes(rows, *args, **kwargs):
        if rows[-1]["close"] == 100:
            raise ArithmeticError("fit_failed")
        return predict(rows, *args, **kwargs)

    monkeypatch.setattr(daily_context, "estimate_direction", sometimes)
    result = refresh_daily_context(paths, DAY, history, analysis_clock=NOW)
    assert result["directional_context"]["by_ticker"]["QQQ"]["reason"] == "fit_failed"
    assert result["directional_context"]["by_ticker"]["SPY"]["status"] == "available"
    trace = read_json(paths.processed / f"direction-training-{DAY}.json")
    assert trace["source_fingerprints"]["direction-history.json"]["sha256"] == (
        hashlib.sha256(raw).hexdigest())
    assert trace["training_caps"]["SPY"]["training_label_last"] == "2026-05-01"


def test_invalid_analysis_clock_rejected(tmp_path):
    with pytest.raises(ValueError, match="timezone"):
        refresh_daily_context(project(tmp_path), DAY, {}, analysis_clock=datetime(2026, 6, 8))


@pytest.mark.parametrize("cache_name,payload", [
    ("cloud-capex-history.json.gz", {"rows": [{
        "date": "2026-06-05", "available_at": "2026-06-05T16:15:00-04:00",
        "issuers": {"META": None}}]}),
    ("direction-macro-history.json", [{"available_at": "2026-06-05T20:15:00Z", "values": 4}]),
])
def test_malformed_cache_replaces_old_forecast_with_explicit_unknown(
    tmp_path, monkeypatch, cache_name, payload
):
    paths = project(tmp_path)
    monkeypatch.setattr(daily_context, "estimate_direction", predict)
    write_json(paths.processed / f"direction-probabilities-{DAY}.json", {"probability_up": .62})
    raw = json.dumps(payload).encode()
    (paths.processed / cache_name).write_bytes(
        gzip.compress(raw) if cache_name.endswith(".gz") else raw)
    result = refresh_daily_context(paths, DAY, current(), analysis_clock=NOW)
    assert result["directional_context"]["by_ticker"]["QQQ"]["status"] == "unavailable"
    saved = read_json(paths.processed / f"direction-probabilities-{DAY}.json")
    assert saved["by_ticker"]["QQQ"]["probability_up"] is None
    trace = read_json(paths.processed / f"direction-training-{DAY}.json")
    assert trace["old_probabilities_reused"] is False


def test_report_refreshes_context_before_snapshot_capture(tmp_path, monkeypatch):
    from twiz import cli

    paths = project(tmp_path)
    monkeypatch.setattr(cli, "ProjectPaths", lambda: paths)
    for name in ("cmd_ingest", "cmd_analyze", "cmd_backtest", "_build_events_and_playbooks"):
        monkeypatch.setattr(cli, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_load_or_create_market", lambda *args, **kwargs: current())
    monkeypatch.setattr(cli, "build_learning_replays", lambda *args, **kwargs: [])

    def captured(*args, **kwargs):
        for prefix in ("direction-probabilities", "cloud-capex", "direction-training",
                       "portfolio-plan", "portfolio-training"):
            assert (paths.processed / f"{prefix}-{DAY}.json").is_file()
        raise RuntimeError("snapshot_order_verified")

    monkeypatch.setattr(cli, "begin_analysis_snapshot", captured)
    with pytest.raises(RuntimeError, match="snapshot_order_verified"):
        cli.cmd_report(DAY)
