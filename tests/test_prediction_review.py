"""Cached event beliefs must pass the same temporal and quality gates as live data."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from twiz import cli
from twiz.config import DEFAULT_WATCHLIST
from twiz.reporting import _event_quote, build_report_bundle, render_markdown
from twiz.storage import ProjectPaths, write_json


def snapshot(now):
    return {
        "id": "123", "label": "测试事件", "question": "Will the event happen?",
        "source": "Polymarket Gamma", "status": "yes_leaning",
        "yes_probability": 0.8, "no_probability": 0.2,
        "market_updated_at": (now - timedelta(minutes=5)).isoformat(),
        "fetched_at": (now - timedelta(minutes=1)).isoformat(),
        "end_date": (now + timedelta(days=2)).isoformat(),
        "liquidity_usd": 5000, "spread": 0.02,
    }


@pytest.mark.parametrize("override", [
    {"yes_probability": True, "no_probability": False},
    {"yes_probability": 0.8, "no_probability": 0.8},
    {"liquidity_usd": 0}, {"spread": 0.7},
])
def test_cached_quote_does_not_bypass_quality_validation(override):
    now = datetime.now(timezone.utc)
    probability, _ = _event_quote({**snapshot(now), **override})
    assert probability is None


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_expired_market_endpoint_prices_are_unknown(probability):
    now = datetime.now(timezone.utc)
    row = {**snapshot(now), "yes_probability": probability,
           "no_probability": 1 - probability,
           "end_date": (now - timedelta(minutes=1)).isoformat()}
    assert _event_quote(row)[0] is None


def test_current_cache_cannot_backfill_historical_report(tmp_path):
    now = datetime.now(timezone.utc)
    report_date = (now.astimezone(ZoneInfo("America/Toronto")).date()
                   - timedelta(days=2)).isoformat()
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    write_json(paths.processed / f"prediction-markets-{report_date}.json", [snapshot(now)])
    bundle = build_report_bundle(paths, report_date, DEFAULT_WATCHLIST[:1])
    assert bundle.daily_brief["event_evidence"] == []
    rendered = render_markdown(bundle)
    assert "80.0%" not in rendered and "未知" in rendered


def test_prediction_ingest_writes_rows_and_provider_warnings(tmp_path):
    from twiz.storage import read_json

    paths = ProjectPaths(tmp_path)
    report_date = "2026-06-07"
    rows = [{"id": "123", "status": "unavailable", "yes_probability": None}]
    warnings = ["Polymarket 123: provider unavailable"]
    with patch("twiz.cli.ProjectPaths", return_value=paths), patch(
        "twiz.cli.load_env_file"
    ), patch("twiz.cli.load_prediction_markets_config", return_value={"markets": []}), patch(
        "twiz.cli.fetch_prediction_markets", return_value=(rows, warnings)
    ) as provider:
        cli.cmd_ingest(source="predictions", report_date=report_date, live=True)
    provider.assert_called_once_with({"markets": []}, report_date, live=True)
    assert read_json(paths.processed / f"prediction-markets-{report_date}.json") == rows
    assert read_json(paths.processed / f"warnings-{report_date}.json") == warnings
    assert not (paths.raw / f"market-{report_date}.json").exists()


def test_offline_prediction_ingest_preserves_existing_snapshot_without_fetch(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    report_date = "2026-06-07"
    artifact = paths.processed / f"prediction-markets-{report_date}.json"
    write_json(artifact, [snapshot(datetime.now(timezone.utc))])
    before = artifact.read_bytes()
    with patch("twiz.cli.ProjectPaths", return_value=paths), patch(
        "twiz.cli.load_env_file"
    ), patch("twiz.cli.fetch_prediction_markets") as provider:
        cli.cmd_ingest(source="predictions", report_date=report_date, live=False)
    provider.assert_not_called()
    assert artifact.read_bytes() == before


def test_historical_prediction_ingest_never_calls_current_market_endpoint(tmp_path):
    from twiz.ingest import prediction_markets as pm
    from twiz.storage import read_json

    paths = ProjectPaths(tmp_path)
    report_date = "2026-06-07"
    with patch("twiz.cli.ProjectPaths", return_value=paths), patch(
        "twiz.cli.load_env_file"
    ), patch("twiz.cli.load_prediction_markets_config", return_value={
        "markets": [{"id": "123"}], "timezone": "America/Toronto",
    }), patch.object(pm, "_fetch_market") as endpoint:
        cli.cmd_ingest(source="predictions", report_date=report_date, live=True)
    endpoint.assert_not_called()
    assert read_json(paths.processed / f"prediction-markets-{report_date}.json") == []
    assert any("historical" in warning for warning in read_json(
        paths.processed / f"warnings-{report_date}.json"
    ))
