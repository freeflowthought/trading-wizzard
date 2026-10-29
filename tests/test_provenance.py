import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from twiz.provenance import (
    begin_analysis_snapshot,
    finalize_analysis_snapshot,
    load_snapshot_inputs,
    verify_snapshot,
)
from twiz.storage import ProjectPaths, read_json, write_json

DAY = "2026-06-08"
NOW = datetime(2026, 10, 28, 5, tzinfo=timezone.utc)


def project(tmp_path):
    paths = ProjectPaths(tmp_path)
    paths.ensure()
    (paths.config / "prediction-markets.yml").write_text("main_case_threshold: 0.75\n")
    write_json(paths.raw / f"market-{DAY}.json", {"QQQ": [
        {"date": "2026-06-05", "close": 100, "source": "yfinance",
         "fetched_at": NOW.isoformat()},
    ]})
    code = tmp_path / "src" / "twiz" / "model.py"
    code.parent.mkdir(parents=True)
    code.write_text("MODEL_VERSION = 1\n")
    return paths


def test_begin_binds_immutable_inputs_and_two_runs_do_not_overwrite(tmp_path):
    paths = project(tmp_path)
    first = begin_analysis_snapshot(paths, DAY, captured_at=NOW, source_root=tmp_path)
    second = begin_analysis_snapshot(paths, DAY, captured_at=NOW, source_root=tmp_path)
    assert first["run_id"] != second["run_id"]
    write_json(paths.raw / f"market-{DAY}.json", {"QQQ": [{"close": 999}]})
    archived = ProjectPaths(Path(first["input_root"]))
    decision = {"close": read_json(archived.raw / f"market-{DAY}.json")["QQQ"][0]["close"]}
    receipt = finalize_analysis_snapshot(
        first, decision, outputs={"report.md": b"original display"}
    )
    verified = verify_snapshot(Path(first["path"]), receipt["manifest_sha256"])
    assert verified["valid"] and decision["close"] == 100
    assert verified["manifest"]["analysis_clock_utc"] == NOW.isoformat()
    assert verified["manifest"]["input_binding"] == "archived_inputs"
    assert (Path(first["path"]) / "report.md").read_bytes() == b"original display"
    assert load_snapshot_inputs(Path(first["path"])).root == Path(first["input_root"])
    with pytest.raises(FileExistsError):
        finalize_analysis_snapshot(first, {"close": 888})


def test_later_download_never_becomes_historical_availability(tmp_path):
    paths = project(tmp_path)
    write_json(paths.processed / f"options-{DAY}.json", {"QQQ": {
        "fetched_at": NOW.isoformat(), "contracts": [{"strike": 100, "last_price": 5}],
    }})
    write_json(paths.processed / f"prediction-markets-{DAY}.json", [{
        "id": "new-contract", "yes_probability": .9, "fetched_at": NOW.isoformat(),
        "market_updated_at": NOW.isoformat(), "date": DAY,
    }])
    draft = begin_analysis_snapshot(paths, DAY, captured_at=NOW, source_root=tmp_path)
    receipt = finalize_analysis_snapshot(draft, {})
    manifest = verify_snapshot(Path(draft["path"]))["manifest"]
    for category in ("options", "prediction-markets"):
        assert manifest["coverage"][category]["status"] == "historical_unavailable"
        assert manifest["coverage"][category]["historical_point_in_time_verified"] is False
    refs = [json.loads(line) for line in (Path(draft["path"]) / "row-index.jsonl")
            .read_text().splitlines()]
    price = next(row for row in refs if row["source_file"].startswith("data/raw/market-"))
    assert price["observed_at"] == "2026-06-05"
    assert price["retrieved_at"] == NOW.isoformat()
    assert price["declared_available_at"] is None
    assert price["retrieved_by_report_cutoff"] is False
    assert price["historical_point_in_time_verified"] is False
    assert receipt["manifest_sha256"]


def test_missing_or_empty_quotes_are_missing_not_zero_probability(tmp_path):
    paths = project(tmp_path)
    write_json(paths.processed / f"options-{DAY}.json", {})
    draft = begin_analysis_snapshot(paths, DAY, captured_at=NOW, source_root=tmp_path)
    receipt = finalize_analysis_snapshot(draft, {})
    assert all(row["status"] == "missing" for row in receipt["coverage"].values())


def test_hashes_detect_input_code_config_output_and_manifest_tampering(tmp_path):
    paths = project(tmp_path)
    draft = begin_analysis_snapshot(paths, DAY, captured_at=NOW, source_root=tmp_path)
    receipt = finalize_analysis_snapshot(draft, {"buy_now": 0}, outputs={"report.html": b"saved"})
    run = Path(draft["path"])
    for name in (f"inputs/data/raw/market-{DAY}.json", "inputs/config/prediction-markets.yml",
                 "inputs/src/twiz/model.py", "decision.json", "report.html"):
        target, original = run / name, (run / name).read_bytes()
        target.write_bytes(original + b"tamper")
        assert not verify_snapshot(run, receipt["manifest_sha256"])["valid"]
        target.write_bytes(original)
    assert verify_snapshot(run)["valid"]
    manifest = run / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    assert not verify_snapshot(run, receipt["manifest_sha256"])["valid"]
    with pytest.raises(ValueError, match="integrity failed"):
        load_snapshot_inputs(run)


def test_capture_excludes_secrets_and_symlinks_and_validates_clock(tmp_path):
    paths = project(tmp_path)
    (tmp_path / ".env").write_text("SECRET=not-archived")
    (paths.config / "secret.json").symlink_to(tmp_path / ".env")
    with pytest.raises(ValueError, match="timezone"):
        begin_analysis_snapshot(paths, DAY, captured_at=datetime(2026, 10, 28))
    draft = begin_analysis_snapshot(paths, DAY, captured_at=NOW, source_root=tmp_path)
    finalize_analysis_snapshot(draft, {}, outcome_refs=[{"status": "not_yet_observed"}])
    run = Path(draft["path"])
    assert not (run / "inputs/.env").exists()
    assert not (run / "inputs/config/secret.json").exists()
    assert "not-archived" not in (run / "manifest.json").read_text()
    assert verify_snapshot(run)["manifest"]["outcome_refs"] == [{"status": "not_yet_observed"}]


def test_inputs_cannot_change_between_capture_and_seal(tmp_path):
    paths = project(tmp_path)
    draft = begin_analysis_snapshot(paths, DAY, captured_at=NOW, source_root=tmp_path)
    archived_market = Path(draft["input_root"]) / f"data/raw/market-{DAY}.json"
    archived_market.write_text('{"QQQ": []}')
    with pytest.raises(ValueError, match="changed before sealing"):
        finalize_analysis_snapshot(draft, {})
    assert not (Path(draft["path"]) / "manifest.json").exists()


def test_full_chain_scope_is_single_expiry_and_quote_age_stays_unknown(tmp_path):
    paths = project(tmp_path)
    write_json(paths.processed / f"options-{DAY}.json", {"QQQ": {
        "fetched_at": NOW.isoformat(), "underlying_date": DAY,
        "contracts": [{"strike": 100}], "full_chain_contracts": [{"strike": 95}, {"strike": 105}],
        "chain_metadata": {"scope": "single_expiry", "quote_age_verified": False},
    }})
    draft = begin_analysis_snapshot(paths, DAY, captured_at=NOW, source_root=tmp_path)
    receipt = finalize_analysis_snapshot(draft, {})
    coverage = receipt["coverage"]["options"]
    assert coverage["contract_scope"] == "single_expiry_full_chain_plus_selected"
    assert coverage["full_chain_contract_references"] == 2
    refs = [json.loads(line) for line in (Path(draft["path"]) / "row-index.jsonl")
            .read_text().splitlines()]
    contract = next(row for row in refs if row["record_kind"] == "full_chain_contract")
    assert contract["underlying_date"] == DAY
    assert contract["observed_at"] is None
    assert contract["retrieved_at"] == NOW.isoformat()


def test_report_analyzes_archived_copies_and_keeps_exact_display_per_run(tmp_path, monkeypatch):
    from twiz import cli

    paths = project(tmp_path)
    (paths.config / "watchlist.yml").write_text("watchlist:\n  - ticker: QQQ\n  - ticker: SPY\n")
    market = {ticker: [{"date": DAY, "source": "yfinance", "session_complete": True,
                        "close": 100, "open": 100, "high": 101, "low": 99, "volume": 1000}]
              for ticker in ("SPY", "QQQ")}
    write_json(paths.raw / f"market-{DAY}.json", market)
    write_json(paths.processed / f"options-{DAY}.json", {})
    write_json(paths.processed / f"fundamentals-{DAY}.json", {})
    monkeypatch.setattr(cli, "ProjectPaths", lambda root=None: paths if root is None else
                        ProjectPaths(root))
    for name in ("cmd_ingest", "cmd_analyze", "cmd_backtest", "_build_events_and_playbooks"):
        monkeypatch.setattr(cli, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_load_or_create_market", lambda *args, **kwargs: market)
    begin = cli.begin_analysis_snapshot

    def capture_then_change_original(*args, **kwargs):
        draft = begin(*args, **kwargs)
        write_json(paths.raw / f"market-{DAY}.json", {
            ticker: [{**market[ticker][0], "close": 999}] for ticker in market
        })
        return draft

    monkeypatch.setattr(cli, "begin_analysis_snapshot", capture_then_change_original)
    report_path = cli.cmd_report(DAY, event_threshold=.8)
    receipt = read_json(paths.reports / f"{DAY}-run.json")
    run = Path(receipt["path"])
    original_markdown = (run / "report.md").read_bytes()
    assert original_markdown == report_path.read_bytes()
    assert (run / "report.html").read_bytes() == report_path.with_suffix(".html").read_bytes()
    decision = read_json(run / "decision.json")
    assert {row["close"] for row in decision["watchlist_table"]} == {100}
    assert decision["scenario_context"]["event_threshold"] == .8
    assert decision["analysis_run"]["run_id"] == receipt["run_id"]
    manifest = verify_snapshot(run, receipt["manifest_sha256"])["manifest"]
    assert manifest["metadata"]["event_threshold"] == .8
    assert manifest["metadata"]["threshold_source"] == "cli"
    assert "本次分析追溯" in original_markdown.decode()
    cli.cmd_report(DAY, event_threshold=.75)
    assert read_json(paths.reports / f"{DAY}-run.json")["run_id"] != receipt["run_id"]
    assert (run / "report.md").read_bytes() == original_markdown
    assert verify_snapshot(run, receipt["manifest_sha256"])["valid"]
    assert cli.cmd_audit(run, receipt["manifest_sha256"]) == 0


def test_jsonl_row_references_keep_line_and_standard_pointer_separate(tmp_path):
    paths = project(tmp_path)
    (paths.raw / f"items-{DAY}.jsonl").write_text(
        '{"source":"news","published_at":"2026-06-08T10:00:00Z","title":"first"}\n'
        '{"source":"news","published_at":"2026-06-08T11:00:00Z","title":"second"}\n'
    )
    draft = begin_analysis_snapshot(paths, DAY, source_root=tmp_path)
    finalize_analysis_snapshot(draft, {})
    refs = [json.loads(line) for line in (Path(draft["path"]) / "row-index.jsonl")
            .read_text().splitlines()]
    news = [row for row in refs if "/items-" in row["source_file"]]
    assert [row["source_line"] for row in news] == [1, 2]
    assert all(row["json_pointer"] == "" for row in news)


def test_directory_symlink_cannot_copy_external_text_as_source_study(tmp_path):
    paths = project(tmp_path)
    external = tmp_path.parent / (tmp_path.name + "-external")
    external.mkdir()
    (external / "private.txt").write_text("not an input")
    (paths.raw / "timelines").symlink_to(external, target_is_directory=True)
    draft = begin_analysis_snapshot(paths, DAY, source_root=tmp_path)
    finalize_analysis_snapshot(draft, {})
    assert not (Path(draft["input_root"]) / "data/raw/timelines/private.txt").exists()


def test_decision_links_require_matching_observed_value_and_preserve_missing_refs(tmp_path):
    paths = project(tmp_path)
    draft = begin_analysis_snapshot(paths, DAY, source_root=tmp_path)
    finalize_analysis_snapshot(draft, {"watchlist_table": [
        {"ticker": "QQQ", "data_date": "2026-06-05", "close": 100.0},
        {"ticker": "QQQ", "data_date": "2026-06-05", "close": 999},
        {"ticker": "SPY", "data_date": "2026-06-05", "close": 100},
    ]})
    refs = read_json(Path(draft["path"]) / "decision-input-refs.json")
    assert [row["status"] for row in refs] == ["matched", "missing", "missing"]
    assert refs[0]["references"][0]["json_pointer"] == "/QQQ/0"
    assert refs[0]["references"][0]["observed_value"] == 100
    assert refs[0]["references"][0]["historical_point_in_time_verified"] is False


def test_standalone_technical_analysis_archives_before_reading_source(tmp_path, monkeypatch):
    from twiz import cli

    paths = project(tmp_path)
    monkeypatch.setattr(cli, "ProjectPaths", lambda root=None: paths if root is None else
                        ProjectPaths(root))
    monkeypatch.setattr(cli, "_load_or_create_market", lambda *args, **kwargs: {})
    begin = cli.begin_analysis_snapshot

    def capture_then_modify(*args, **kwargs):
        draft = begin(*args, **kwargs)
        write_json(paths.raw / f"market-{DAY}.json", {
            "QQQ": [{"date": "2026-06-05", "close": 999}]
        })
        return draft

    monkeypatch.setattr(cli, "begin_analysis_snapshot", capture_then_modify)
    monkeypatch.setattr(cli, "analyze_technical_snapshots", lambda market, **kwargs: {
        "QQQ": {"indicators": {"close": market["QQQ"][-1]["close"]}}
    })
    cli.cmd_analyze(DAY, build_playbooks=False)
    receipt = read_json(paths.processed / f"analysis-run-{DAY}.json")
    run = Path(receipt["path"])
    assert verify_snapshot(run, receipt["manifest_sha256"])["valid"]
    decision = read_json(run / "decision.json")
    assert decision["kind"] == "technical_sentiment_analysis"
    assert decision["technical"]["QQQ"]["indicators"]["close"] == 100
    assert decision["macro_allocation_included"] is False
    reference = read_json(run / "decision-input-refs.json")[0]
    assert reference["kind"] == "technical_close" and reference["status"] == "matched"


def test_historical_live_ingest_keeps_saved_event_and_cross_market_snapshots(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from twiz import cli

    paths = project(tmp_path)
    monkeypatch.setattr(cli, "ProjectPaths", lambda: paths)
    stored = []
    for prefix in ("prediction-markets", "cross-market"):
        path = paths.processed / f"{prefix}-{DAY}.json"
        write_json(path, [
            {"source": "original saved source", "fetched_at": "2026-06-08T12:00:00Z"}
        ])
        stored.append((path, path.read_bytes()))
    predictions, cross = Mock(side_effect=AssertionError("no current backfill")), Mock(
        side_effect=AssertionError("no current backfill"))
    monkeypatch.setattr(cli, "fetch_prediction_markets", predictions)
    monkeypatch.setattr(cli, "fetch_cross_market", cross)
    cli.cmd_ingest(source="predictions", report_date=DAY, live=True)
    cli.cmd_ingest(source="cross-market", report_date=DAY, live=True)
    assert all(path.read_bytes() == content for path, content in stored)
    predictions.assert_not_called()
    cross.assert_not_called()
    warnings = read_json(paths.processed / f"warnings-{DAY}.json")
    assert any("historical Polymarket cache preserved" in warning for warning in warnings)
    assert any("historical continuous-market cache preserved" in warning for warning in warnings)


def test_historical_live_report_does_not_erase_saved_option_snapshot(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from twiz import cli

    paths = project(tmp_path)
    monkeypatch.setattr(cli, "ProjectPaths", lambda root=None: paths if root is None else
                        ProjectPaths(root))
    for name in ("cmd_ingest", "cmd_analyze", "cmd_backtest", "_build_events_and_playbooks"):
        monkeypatch.setattr(cli, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_load_or_create_market", lambda *args, **kwargs: {})
    options = {"QQQ": {"source": "original_saved_options", "contracts": [{"strike": 100}]}}
    fundamentals = {"QQQ": {"source": "original_saved_statements"}}
    option_path = paths.processed / f"options-{DAY}.json"
    statement_path = paths.processed / f"fundamentals-{DAY}.json"
    write_json(option_path, options)
    write_json(statement_path, fundamentals)
    option_bytes, statement_bytes = option_path.read_bytes(), statement_path.read_bytes()
    option_fetch = Mock(side_effect=AssertionError("historical options must not refresh"))
    statement_fetch = Mock(side_effect=AssertionError("historical statements must not refresh"))
    monkeypatch.setattr(cli, "build_option_snapshots", option_fetch)
    monkeypatch.setattr(cli, "build_fundamental_snapshots", statement_fetch)
    cli.cmd_report(DAY, live=True)
    assert option_path.read_bytes() == option_bytes
    assert statement_path.read_bytes() == statement_bytes
    option_fetch.assert_not_called()
    statement_fetch.assert_not_called()
