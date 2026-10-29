"""Append-only analysis archives with content integrity and honest availability metadata.

SHA256 detects changes against a retained receipt; it is not a signature or proof that
historical data was public at its observation date. Audit reads saved outputs rather
than re-evaluating their freshness against a later wall clock.
"""
import hashlib
import json
import marshal
import math
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from uuid import uuid4
from zoneinfo import ZoneInfo

from .storage import ProjectPaths, to_jsonable

PROCESSED_INPUTS = (
    "sentiment-summary", "technical", "backtests", "options", "macro", "warnings", "events",
    "playbook", "macro-gate", "leadership", "ipos", "fundamentals", "cross-market",
    "prediction-markets", "prediction-discovery", "global-context", "analyst-replays",
    "options-full-chain", "direction-probabilities", "cloud-capex", "direction-training",
    "portfolio-plan", "portfolio-training",
)


def _json(value: Any) -> bytes:
    content = json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, indent=2)
    return (content + "\n").encode()


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def _stamp(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (TypeError, ValueError):
        return None


def _clock(value: Optional[datetime]) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("analysis clock must include a timezone")
    return value.astimezone(timezone.utc)


def _records(value: Any, pointer: str = "", inherited: Optional[dict] = None) -> Iterator[tuple]:
    inherited = inherited or {}
    if isinstance(value, dict):
        context = {**inherited, **{k: value[k] for k in (
            "source", "fetched_at", "retrieved_at", "available_at", "published_at",
            "underlying_date", "series_id"
        ) if value.get(k) is not None}}
        if value and set(value) & {
            "source", "date", "close", "value", "published_at", "observed_at", "fetched_at",
            "contracts", "strike", "yes_probability", "market_updated_at", "mark_price",
            "probability_up", "median_cash_capex_yoy_pct",
        }:
            yield pointer, value, context
        for key, item in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from _records(item, pointer + "/" + escaped, context)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _records(item, pointer + "/" + str(index), inherited)


def _row_refs(data: bytes, original: str, report_date: str, captured: datetime) -> list:
    try:
        if original.endswith(".jsonl"):
            documents = [(i, json.loads(line)) for i, line in
                         enumerate(data.decode().splitlines(), 1) if line.strip()]
        else:
            documents = [(None, json.loads(data))]
    except (ValueError, UnicodeError):
        return []
    cutoff = datetime.combine(date.fromisoformat(report_date), time.max,
                              tzinfo=ZoneInfo("America/Toronto"))
    references = []
    for source_line, document in documents:
        for pointer, row, context in _records(document):
            observed = next((row.get(k) for k in (
                "observed_at", "date", "price_date", "published_at", "market_updated_at"
            ) if row.get(k) is not None), None)
            retrieved = context.get("retrieved_at") or context.get("fetched_at")
            available = context.get("available_at")
            retrieved_clock, available_clock = _stamp(retrieved), _stamp(available)
            references.append({
                "source_file": original, "json_pointer": pointer, "row_sha256": _hash(_json(row)),
                "source_line": source_line,
                "record_id": row.get("id") or row.get("contractSymbol") or (
                    context.get("series_id") or row.get("url")),
                "source": context.get("source"), "observed_at": observed,
                "observed_value": next((row[k] for k in (
                    "close", "yes_probability", "value", "mark_price", "last_price",
                    "probability_up", "median_cash_capex_yoy_pct",
                ) if row.get(k) is not None), None),
                "underlying_date": row.get("underlying_date", context.get("underlying_date")),
                "retrieved_at": retrieved, "declared_available_at": available,
                "archive_captured_at": captured.isoformat(),
                "known_available_no_later_than": captured.isoformat(),
                "retrieved_by_report_cutoff": bool(retrieved_clock and retrieved_clock <= cutoff),
                "declared_available_by_report_cutoff": bool(
                    available_clock and available_clock <= cutoff),
                "availability_basis": "explicit_provider_metadata" if available_clock else (
                    "retrieval_metadata_only" if retrieved_clock else "archive_capture_only"),
                "historical_point_in_time_verified": False,
                "record_kind": "full_chain_contract" if "/full_chain_contracts/" in pointer else (
                    "option_contract" if "strike" in row else (
                        "event_quote" if "yes_probability" in row else "observation"
                    )
                ),
                "reported_status": row.get("status"),
                "value_present": row.get("yes_probability") is not None if (
                    "yes_probability" in row) else True,
                "note": "Observation date is not publication availability; "
                        "later downloads may be revised.",
            })
    return references


def _input_files(paths: ProjectPaths) -> list:
    return sorted({*paths.raw.glob("timelines/*.txt"), paths.processed / "eagle-corpus.json",
                   *paths.processed.glob("shunge-case-*.json")})


def _coverage(references: list, filename: str, exists: bool, captured: datetime,
              report_date: str) -> dict:
    kinds = {"option_contract", "full_chain_contract"} if (
        "/options-" in filename) else {"event_quote"}
    rows = [r for r in references if r["source_file"] == filename
            and r["record_kind"] in kinds and r["value_present"]
            and r["reported_status"] != "unavailable"]
    timely = sum(bool(r["retrieved_by_report_cutoff"]) for r in rows)
    historical = captured.astimezone(ZoneInfo("America/Toronto")).date().isoformat() > report_date
    return {
        "status": "missing" if not exists or not rows else (
            "historical_unavailable" if historical and not timely else "captured_metadata_only"),
        "row_references": len(rows), "retrieved_by_report_cutoff": timely,
        "historical_point_in_time_verified": False,
        "reason": "No historical reconstruction from current quotes. "
                  "Observation dates alone do not prove availability.",
    }


def begin_analysis_snapshot(
    paths: ProjectPaths, report_date: str, *, captured_at: Optional[datetime] = None,
    source_root: Optional[Path] = None, metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Copy inputs before analysis; build from returned input_root to bind the decision."""
    date.fromisoformat(report_date)
    captured = _clock(captured_at)
    archive_clock = datetime.now(timezone.utc)
    run_id = archive_clock.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:12]
    run = paths.root / "data" / "runs" / report_date / run_id
    run.mkdir(parents=True, exist_ok=False)
    source_root = (source_root or Path(__file__).resolve().parents[2]).resolve()
    files, references, missing = [], [], []

    def copy_input(path: Path, original: str, role: str) -> None:
        owner = source_root if role == "source_code" else paths.root.resolve()
        if (not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(owner)):
            missing.append({"path": original, "reason": "missing_or_symlink"})
            return
        data = path.read_bytes()
        target = "inputs/" + original
        _write(run / target, data)
        files.append({"path": target, "original_path": original, "role": role,
                      "sha256": _hash(data), "size": len(data)})
        if role == "raw_input" or role == "processed_input" and path.stem.startswith((
            "options-", "macro-", "prediction-markets-", "cross-market-", "fundamentals-",
            "direction-probabilities-", "cloud-capex-",
            "portfolio-plan-",
        )):
            references.extend(_row_refs(data, original, report_date, archive_clock))

    for prefix in ("market", "market-context"):
        path = paths.raw / f"{prefix}-{report_date}.json"
        copy_input(path, path.relative_to(paths.root).as_posix(), "raw_input")
    items = paths.raw / f"items-{report_date}.jsonl"
    copy_input(items, items.relative_to(paths.root).as_posix(), "raw_input")
    for prefix in PROCESSED_INPUTS:
        path = paths.processed / f"{prefix}-{report_date}.json"
        copy_input(path, path.relative_to(paths.root).as_posix(), "processed_input")
    for path in _input_files(paths):
        copy_input(path, path.relative_to(paths.root).as_posix(), "source_study")
    for pattern in ("*.yml", "*.yaml", "*.json"):
        for path in sorted(paths.config.glob(pattern)):
            copy_input(path, path.relative_to(paths.root).as_posix(), "config")
    for path in sorted((source_root / "src" / "twiz").rglob("*.py")):
        copy_input(path, path.relative_to(source_root).as_posix(), "source_code")
    index = b"".join(_json(row).replace(b"\n", b"") + b"\n" for row in references)
    _write(run / "row-index.jsonl", index)
    files.append({"path": "row-index.jsonl", "role": "row_index",
                  "sha256": _hash(index), "size": len(index)})
    coverage = {}
    originals = {f.get("original_path") for f in files}
    for name in ("options", "prediction-markets"):
        filename = f"data/processed/{name}-{report_date}.json"
        coverage[name] = _coverage(
            references, filename, filename in originals, archive_clock, report_date
        )
        if name == "options":
            full = sum(row["record_kind"] == "full_chain_contract" for row in references
                       if row["source_file"] == filename)
            coverage[name]["contract_scope"] = (
                "single_expiry_full_chain_plus_selected" if full else "selected_contracts"
            )
            coverage[name]["full_chain_contract_references"] = full
            coverage[name]["full_chain_sidecar_captured"] = (
                f"data/processed/options-full-chain-{report_date}.json" in originals
            )
    draft = {
        "schema_version": 1, "run_id": run_id, "report_date": report_date,
        "analysis_clock_utc": captured.isoformat(), "metadata": metadata or {},
        "archive_written_at_utc": archive_clock.isoformat(),
        "analysis_clock_basis": "runtime_clock" if captured_at is None else "caller_supplied_clock",
        "files": files, "missing_inputs": missing, "coverage": coverage,
        "source_code_fingerprint": _hash(_json([
            (f["original_path"], f["sha256"]) for f in files if f["role"] == "source_code"])),
        "config_fingerprint": _hash(_json([
            (f["original_path"], f["sha256"]) for f in files if f["role"] == "config"])),
        "runtime_python": sys.version,
        "runtime_function_hashes": {
            name + "." + key: _hash(marshal.dumps(value.__code__))
            for name, module in sorted(list(sys.modules.items()))
            if module is not None and name.startswith("twiz")
            for key, value in list(vars(module).items())
            if hasattr(value, "__code__") and getattr(value, "__module__", "") == name
        },
        "immutability": "application_append_only_with_sha256_receipt",
        "availability_note": "Archive proves these bytes at capture, "
                             "not historical publication or unrevised data.",
    }
    content = _json(draft)
    _write(run / "draft.json", content)
    return {"run_id": run_id, "path": str(run), "input_root": str(run / "inputs"),
            "analysis_clock_utc": captured.isoformat(), "input_manifest_sha256": _hash(content),
            "coverage": coverage}


def finalize_analysis_snapshot(
    draft: Dict[str, Any], decision: Any, *, outputs: Optional[Dict[str, bytes]] = None,
    outcome_refs: Optional[list] = None, input_binding: str = "archived_inputs",
) -> Dict[str, Any]:
    """Seal once. Saved Markdown/HTML preserve the original display and freshness clock."""
    run = Path(draft["path"])
    if (run / "manifest.json").exists():
        raise FileExistsError("analysis snapshot is already sealed")
    if "decision.json" in (outputs or {}):
        raise ValueError("decision.json must come from the structured decision argument")
    for name, data in (outputs or {}).items():
        if name not in {"report.md", "report.html", "details.md", "details.html"}:
            raise ValueError("only report/details Markdown and HTML are valid display outputs")
        if not isinstance(data, bytes):
            raise TypeError("archive output must be bytes")
    draft_bytes = (run / "draft.json").read_bytes()
    if _hash(draft_bytes) != draft["input_manifest_sha256"]:
        raise ValueError("input manifest changed after capture")
    manifest = json.loads(draft_bytes)
    files = manifest["files"]
    errors = _verify_files(run.resolve(), files)
    if errors:
        raise ValueError("captured inputs changed before sealing: " + "; ".join(errors))
    files.append({"path": "draft.json", "role": "input_manifest",
                  "sha256": _hash(draft_bytes), "size": len(draft_bytes)})
    decision_value = to_jsonable(decision)
    refs = [json.loads(line) for line in (run / "row-index.jsonl").read_text().splitlines()]
    links = _decision_references(decision_value, refs, manifest["report_date"])
    reference_bytes = _json(links)
    _write(run / "decision-input-refs.json", reference_bytes)
    files.append({"path": "decision-input-refs.json", "role": "decision_input_references",
                  "sha256": _hash(reference_bytes), "size": len(reference_bytes)})
    payloads = {"decision.json": _json(decision_value), **(outputs or {})}
    for name, data in payloads.items():
        _write(run / name, data)
        files.append({"path": name, "role": "decision" if name == "decision.json" else "display",
                      "sha256": _hash(data), "size": len(data)})
    manifest.update(input_binding=input_binding, outcome_refs=outcome_refs or [],
                    row_reference_file="row-index.jsonl",
                    decision_reference_file="decision-input-refs.json")
    content = _json(manifest)
    _write(run / "manifest.json", content)
    _write(run / "manifest.sha256", (_hash(content) + "\n").encode())
    return {"run_id": manifest["run_id"], "path": str(run), "manifest_sha256": _hash(content),
            "input_manifest_sha256": _hash(draft_bytes),
            "analysis_clock_utc": manifest["analysis_clock_utc"], "coverage": manifest["coverage"]}


def _decision_references(decision: Any, references: list, report_date: str) -> list:
    if not isinstance(decision, dict):
        return []
    links = []

    def link(kind: str, path: str, candidates: list) -> None:
        links.append({"kind": kind, "decision_pointer": path, "references": candidates,
                      "status": "matched" if len(candidates) == 1 else (
                          "missing" if not candidates else "multiple_source_references"),
                      "availability_note": "A matching observation is not proof "
                                           "of historical publication."})

    def same_value(left: Any, right: Any) -> bool:
        if isinstance(left, bool) or isinstance(right, bool):
            return False
        try:
            return math.isfinite(float(left)) and float(left) == float(right)
        except (TypeError, ValueError, OverflowError):
            return False

    for index, row in enumerate(decision.get("watchlist_table", [])):
        ticker = str(row.get("ticker", "")).replace("~", "~0").replace("/", "~1")
        link("equity_close", f"/watchlist_table/{index}/close", [r for r in references
             if r["source_file"] == f"data/raw/market-{report_date}.json"
             and r["json_pointer"].startswith(f"/{ticker}/")
             and r["observed_at"] == row.get("data_date")
             and same_value(r.get("observed_value"), row.get("close"))])
    for ticker in decision.get("options_context", {}):
        escaped = str(ticker).replace("~", "~0").replace("/", "~1")
        link("option_snapshot", f"/options_context/{escaped}", [r for r in references
             if r["source_file"] == f"data/processed/options-{report_date}.json"
             and r["json_pointer"] == f"/{escaped}"])
    for ticker, snapshot in decision.get("technical", {}).items():
        escaped = str(ticker).replace("~", "~0").replace("/", "~1")
        link("technical_close", f"/technical/{escaped}/indicators/close", [r for r in references
             if r["source_file"] == f"data/raw/market-{report_date}.json"
             and r["json_pointer"].startswith(f"/{escaped}/")
             and r["observed_at"] == decision.get("source_observation_dates", {}).get(ticker)
             and same_value(r.get("observed_value"), snapshot.get("indicators", {}).get("close"))])
    for index, signal in enumerate(decision.get("sentiment_signals", [])):
        for source_index, url in enumerate(signal.get("source_refs", [])):
            link("sentiment_source", f"/sentiment_signals/{index}/source_refs/{source_index}", [
                r for r in references if r["source_file"] == f"data/raw/items-{report_date}.jsonl"
                and r["record_id"] == url])
    for index, row in enumerate(decision.get("prediction_markets", [])):
        link("event_quote", f"/prediction_markets/{index}", [r for r in references
             if r["source_file"] == f"data/processed/prediction-markets-{report_date}.json"
             and str(r["record_id"]) == str(row.get("id")) and r["record_kind"] == "event_quote"])
    for ticker in decision.get("directional_context", {}).get("by_ticker", {}):
        escaped = str(ticker).replace("~", "~0").replace("/", "~1")
        link("equity_direction_model", f"/directional_context/by_ticker/{escaped}", [
            r for r in references
            if r["source_file"] == f"data/processed/direction-probabilities-{report_date}.json"
            and r["json_pointer"] == f"/by_ticker/{escaped}"])
    if decision.get("cloud_capex_context"):
        link("cloud_cash_capex", "/cloud_capex_context/aggregate", [r for r in references
             if r["source_file"] == f"data/processed/cloud-capex-{report_date}.json"
             and r["json_pointer"] == "/aggregate"])
    if decision.get("portfolio_plan"):
        link("news_cycle_portfolio_reference", "/portfolio_plan", [r for r in references
             if r["source_file"] == f"data/processed/portfolio-plan-{report_date}.json"
             and r["json_pointer"] == ""])
    for key, row in decision.get("scenario_context", {}).get("macro_features", {}).items():
        if not isinstance(row, dict):
            continue
        link("macro_observation", f"/scenario_context/macro_features/{key}", [r for r in references
             if r["source_file"] == f"data/processed/macro-{report_date}.json"
             and r["record_id"] == row.get("series_id") and r["observed_at"] == row.get("date")])
    return links


def _verify_files(run: Path, files: list) -> list:
    errors = []
    for entry in files:
        target = run / entry["path"]
        if target.is_symlink() or not target.resolve().is_relative_to(run):
            errors.append(f"unsafe archive path: {entry['path']}")
        elif not target.is_file():
            errors.append(f"missing file: {entry['path']}")
        elif _hash(target.read_bytes()) != entry["sha256"]:
            errors.append(f"file hash mismatch: {entry['path']}")
    return errors


def verify_snapshot(path: Path, expected_manifest_sha256: Optional[str] = None) -> Dict[str, Any]:
    run, errors = Path(path).resolve(), []
    try:
        content = (run / "manifest.json").read_bytes()
        expected = expected_manifest_sha256 or (run / "manifest.sha256").read_text().strip()
        if _hash(content) != expected:
            errors.append("manifest hash mismatch")
        manifest = json.loads(content)
        errors.extend(_verify_files(run, manifest["files"]))
        return {"valid": not errors, "errors": errors, "manifest": manifest}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"valid": False, "errors": errors + [str(error)], "manifest": None}


def load_snapshot_inputs(
    path: Path, expected_manifest_sha256: Optional[str] = None
) -> ProjectPaths:
    result = verify_snapshot(path, expected_manifest_sha256)
    if not result["valid"]:
        raise ValueError("snapshot integrity failed: " + "; ".join(result["errors"]))
    return ProjectPaths(Path(path) / "inputs")
