from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from twiz import cli
from twiz.config import load_prediction_markets_config, validate_event_threshold
from twiz.reporting import _render_strategy_context
from twiz.scenarios import build_scenario_context
from twiz.storage import ProjectPaths

NOW = datetime(2026, 10, 28, 2, tzinfo=timezone.utc)
DAY = "2026-10-27"


def event(driver, yes):
    contract = next(row for row in load_prediction_markets_config()["markets"]
                    if row.get("driver") == driver)
    return {"id": contract["id"], "question": contract["expected_question"], "driver": driver,
            "source": "Polymarket Gamma", "status": "uncertain", "yes_probability": yes,
            "no_probability": 1-yes, "market_updated_at": NOW.isoformat(),
            "fetched_at": NOW.isoformat(), "end_date": (NOW+timedelta(days=5)).isoformat(),
            "liquidity_usd": 10000, "spread": .01}


@pytest.mark.parametrize("value", [True, False, .5, .4, 0, 65, 1.01, float("nan"),
                                       float("inf"), None, "invalid"])
def test_invalid_threshold_never_silently_reverts_to_default(value):
    with pytest.raises(ValueError, match="threshold"):
        validate_event_threshold(value)


@pytest.mark.parametrize("value", [.500001, .65, .8, 1])
def test_valid_ratio_threshold(value):
    assert validate_event_threshold(value) == value


def test_custom_threshold_changes_main_cases_and_risk_policy_with_same_inputs(tmp_path):
    config = tmp_path / "prediction-markets.yml"
    config.write_text("main_case_threshold: 0.8\n")
    assert load_prediction_markets_config(config)["main_case_threshold"] == .8
    rows = [event("fed_hold", .75), event("near_term_ceasefire", .75)]
    normal = build_scenario_context(DAY, [], rows, {}, now=NOW, event_threshold=.65)
    strict = build_scenario_context(DAY, [], rows, {}, now=NOW, event_threshold=.8)
    assert [row["main_case"] for row in normal["events"]] == ["Yes", "Yes"]
    assert [row["main_case"] for row in strict["events"]] == ["分支", "分支"]
    assert normal["policy"]["initial_cap_pct"] > strict["policy"]["initial_cap_pct"]
    assert strict["policy"]["initial_cap_pct"] == 0
    text = "\n".join(_render_strategy_context(strict, {}, {}))
    assert "80%门槛" in text and "达到80%" in text
    assert "65%" not in text
    assert "80%" in strict["limitations"][0]
    assert all(row["probability"] is None for row in strict["branches"])


def test_complement_tolerance_cannot_force_yes_when_both_outcomes_meet_low_threshold():
    row = event("fed_hold", .51)
    row["no_probability"] = .51
    result = build_scenario_context(DAY, [], [row], {}, now=NOW, event_threshold=.51)
    assert result["events"][0]["main_case"] == "分支"


def test_created_after_retrieval_contract_is_not_available_historically():
    row = event("fed_hold", .9)
    row["contract_metadata"] = {"created_at": (NOW+timedelta(days=1)).isoformat()}
    assert not build_scenario_context(DAY, [], [row], {}, now=NOW)["events"]


def test_cli_threshold_override_is_forwarded_and_invalid_value_fails_before_ingest(monkeypatch,
                                                                                 tmp_path):
    report = Mock(return_value=Path("/tmp/report.md"))
    monkeypatch.setattr(cli, "cmd_report", report)
    assert cli._argparse_main(["report", "--date", DAY, "--event-threshold", "0.8"]) == 0
    assert report.call_args.kwargs["event_threshold"] == .8
    monkeypatch.undo()
    paths = ProjectPaths(tmp_path)
    monkeypatch.setattr(cli, "ProjectPaths", lambda: paths)
    ingest = Mock()
    monkeypatch.setattr(cli, "cmd_ingest", ingest)
    with pytest.raises(ValueError, match="threshold"):
        cli.cmd_report(DAY, event_threshold=.5)
    ingest.assert_not_called()


def test_typer_report_threshold_and_audit_options_are_available(monkeypatch):
    if cli.typer is None:
        pytest.skip("Typer is optional")
    from typer.testing import CliRunner

    report = Mock(return_value=Path("/tmp/report.md"))
    monkeypatch.setattr(cli, "cmd_report", report)
    result = CliRunner().invoke(cli.app, ["report", "--date", DAY, "--event-threshold", "0.8"])
    assert result.exit_code == 0, result.output
    assert report.call_args.kwargs["event_threshold"] == .8
    audit = Mock(return_value=0)
    monkeypatch.setattr(cli, "cmd_audit", audit)
    result = CliRunner().invoke(cli.app, ["audit", "--run", "/tmp/saved-run", "--list-inputs"])
    assert result.exit_code == 0, result.output
    audit.assert_called_once_with(Path("/tmp/saved-run"), None, True)
