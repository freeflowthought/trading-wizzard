"""Frozen, two-phase local evaluation runs. Completion is not proof of the research goal."""
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

DATA_FILES = ("pretest-market.json", "pretest-macro.json", "market.json", "macro.json")
EXTRA_DATA_FILES = ("pretest-seal.json", "dataset-seal.json", "market-manifest.json",
                    "macro-manifest.json")
WORKER = '''import sys
from pathlib import Path
run = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(run / "frozen" / "src"))
from twiz.evaluation_runs import _worker
_worker(sys.argv[1], run, sys.argv[3] if len(sys.argv) > 3 else None)
'''


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def _entry(run: Path, path: Path, role: str) -> dict:
    data = path.read_bytes()
    return {"path": path.relative_to(run).as_posix(), "role": role,
            "sha256": _hash(data), "size": len(data)}


def _fingerprint_distribution(name: str) -> dict:
    distribution = importlib.metadata.distribution(name)
    records = []
    for item in sorted(distribution.files or [], key=str):
        path = Path(distribution.locate_file(item))
        if path.is_file() and path.suffix != ".pyc":
            records.append({"path": str(item), "sha256": _hash(path.read_bytes())})
    if not records:
        raise RuntimeError(f"No auditable installed files for dependency {name}.")
    return {"version": distribution.version, "files": records, "sha256": _hash(_json(records))}


def runtime_fingerprint() -> dict:
    executable = Path(sys.executable).resolve()
    return {"python": {"version": sys.version, "executable": str(executable),
                       "sha256": _hash(executable.read_bytes())},
            "numpy": _fingerprint_distribution("numpy"),
            "PyYAML": _fingerprint_distribution("PyYAML")}


def _preregistration(protocol: Path, project: Path, receipt: Optional[Path]) -> dict:
    is_default = protocol == (project / "config/evaluation-v2.yml").resolve()
    if receipt is None and not is_default:
        return {"status": "not_provided_custom_protocol"}
    path = Path(receipt or project / "data/processed/evaluation/protocol-v2-freeze.json")
    if not path.is_file() or path.is_symlink():
        raise ValueError("Matching preregistration receipt is required for the default v2 protocol")
    payload = path.read_bytes()
    record = json.loads(payload)
    digest = _hash(protocol.read_bytes())
    if record.get("sha256") != digest:
        raise ValueError("Protocol differs from the preregistration receipt")
    return {"status": "matched", "original_receipt_path": str(path.resolve()),
            "receipt_sha256": _hash(payload), "protocol_sha256": digest,
            "frozen_at": record.get("frozen_at"), "receipt_file": "protocol-preregistration.json"}


def _checked_file(run: Path, item: dict) -> Path:
    path = run / item["path"]
    if (path.is_symlink() or not path.resolve().is_relative_to(run.resolve())
            or not path.is_file() or _hash(path.read_bytes()) != item["sha256"]):
        raise ValueError(f"Evaluation file missing, unsafe or changed: {item['path']}")
    return path


def _verify_input_archive(run: Path, expected_sha256: Optional[str] = None) -> dict:
    content = (run / "input-manifest.json").read_bytes()
    if _hash(content) != (run / "input-manifest.sha256").read_text().strip():
        raise ValueError("Evaluation input-manifest hash mismatch")
    if expected_sha256 is not None and _hash(content) != expected_sha256:
        raise ValueError("Captured input manifest changed from the controller receipt")
    manifest = json.loads(content)
    for item in manifest["files"]:
        _checked_file(run, item)
    registration = manifest.get("protocol_preregistration", {})
    if registration.get("status") == "matched":
        receipt_bytes = (run / "protocol-preregistration.json").read_bytes()
        receipt = json.loads(receipt_bytes)
        protocol_sha = _hash((run / "frozen/config/evaluation-v2.yml").read_bytes())
        if (receipt.get("sha256") != protocol_sha
                or registration.get("protocol_sha256") != protocol_sha
                or registration.get("receipt_sha256") != _hash(receipt_bytes)):
            raise ValueError("Captured protocol preregistration proof does not match")
    dataset = run / "frozen/datasets"
    for name in ("pretest-seal.json", "dataset-seal.json"):
        seal_path = dataset / name
        if not seal_path.is_file():
            continue
        seal = json.loads(seal_path.read_bytes())
        records = [{"path": key, **value} for key, value in seal.get("files", {}).items()]
        records.extend(seal.get("datasets", []))
        for record in records:
            filename = Path(record["path"]).name
            if filename not in {*DATA_FILES, *EXTRA_DATA_FILES}:
                continue
            target = dataset / filename
            if not target.is_file() or _hash(target.read_bytes()) != record["sha256"]:
                raise ValueError(f"Dataset differs from supplied upstream seal: {filename}")
    return manifest


def _verify_selection(run: Path, expected_sha256: Optional[str] = None) -> dict:
    seal = json.loads((run / "pretest-selection-seal.json").read_bytes())
    if seal["input_manifest_sha256"] != _hash((run / "input-manifest.json").read_bytes()):
        raise ValueError("Pretest selection is bound to a different input archive")
    data = (run / "pretest-selection.json").read_bytes()
    if _hash(data) != seal["selection_sha256"] or (
        expected_sha256 is not None and _hash(data) != expected_sha256
    ):
        raise ValueError("Sealed pretest selection changed")
    return json.loads(data)


def _worker(phase: str, run: Path, expected_selection_sha256: Optional[str] = None) -> None:
    """Executed only in a fresh isolated interpreter with the captured src first."""
    manifest = _verify_input_archive(run)
    if runtime_fingerprint() != manifest["runtime"]:
        raise ValueError("Runtime dependency fingerprint changed after capture")
    import yaml

    from twiz.evaluation_v2 import (
        EvaluationV2Protocol,
        run_walk_forward_evaluation_v2,
        select_pretest_models_v2,
    )

    frozen_src = (run / "frozen" / "src").resolve()
    module = sys.modules[EvaluationV2Protocol.__module__]
    if not Path(module.__file__).resolve().is_relative_to(frozen_src):
        raise ValueError("Evaluation engine was not imported from the captured source tree")
    for name, imported in list(sys.modules.items()):
        if (name == "twiz" or name.startswith("twiz.")) and getattr(imported, "__file__", None):
            if not Path(imported.__file__).resolve().is_relative_to(frozen_src):
                raise ValueError(f"Project helper imported outside captured source: {name}")
    print(f"phase={phase}; captured_engine={module.__file__}", flush=True)
    protocol = EvaluationV2Protocol.from_config(yaml.safe_load(
        (run / "frozen" / "config" / "evaluation-v2.yml").read_text()))
    dataset = run / "frozen" / "datasets"
    if phase == "selection":
        # This phase does not deserialize the evaluation-period files.
        market = json.loads((dataset / "pretest-market.json").read_bytes())
        macro = json.loads((dataset / "pretest-macro.json").read_bytes())
        result = select_pretest_models_v2(market, macro, protocol=protocol)
        _write(run / "pretest-selection.json", _json(result))
    elif phase == "evaluation":
        if not expected_selection_sha256:
            raise ValueError("Evaluation phase requires the controller's sealed selection hash")
        selection = _verify_selection(run, expected_selection_sha256)
        market = json.loads((dataset / "market.json").read_bytes())
        macro = json.loads((dataset / "macro.json").read_bytes())
        result = run_walk_forward_evaluation_v2(
            market, macro, protocol=protocol, pretest_selection=selection)
        _write(run / "result.json", _json(result))
    else:
        raise ValueError("Unknown evaluation phase")
    print(f"phase={phase}; completed", flush=True)


def _launch_phase(run: Path, phase: str, selection_sha256: Optional[str] = None) -> None:
    _verify_input_archive(run)
    if phase == "evaluation":
        if not selection_sha256:
            raise ValueError("Evaluation phase requires a sealed selection hash")
        _verify_selection(run, selection_sha256)
    log = run / f"{phase}.log"
    with log.open("xb") as output:
        command = [sys.executable, "-I", "-B", str(run / "worker.py"), phase, str(run)]
        if selection_sha256:
            command.append(selection_sha256)
        result = subprocess.run(command, cwd=run, stdout=output,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        _write(run / f"failed-{phase}.json", _json({
            "phase": phase, "returncode": result.returncode,
            "note": "Incomplete experiment retained; "
                    "do not automatically restart or claim running.",
        }))
        raise RuntimeError(f"Evaluation {phase} failed; preserved archive and log: {run}")


def run_evaluation_experiment(
    protocol_path: Optional[Path] = None, dataset_dir: Optional[Path] = None, *,
    experiment_root: Optional[Path] = None, source_root: Optional[Path] = None,
    preregistration_receipt: Optional[Path] = None,
) -> Dict[str, Any]:
    """Freeze dependencies, select on pretest only, seal, then evaluate in a new process."""
    project = Path(__file__).resolve().parents[2]
    protocol_path = Path(protocol_path or project / "config" / "evaluation-v2.yml").resolve()
    dataset_dir = Path(dataset_dir or project / "data/processed/evaluation/datasets").resolve()
    source_root = Path(source_root or project).resolve()
    root = Path(experiment_root or project / "data/processed/evaluation/experiments").resolve()
    required = [protocol_path, *[dataset_dir / name for name in DATA_FILES],
                source_root / "src/twiz/evaluation_v2.py"]
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise ValueError("Protocol, v2 engine and all four canonical dataset files are required")
    import yaml

    requested_protocol = yaml.safe_load(protocol_path.read_text()) or {}
    if not isinstance(requested_protocol, dict):
        raise ValueError("Evaluation protocol must be a mapping.")
    if (str(requested_protocol.get("protocol_version", "")).startswith("author-swing")
            or requested_protocol.get("target_task") == (
                "bottom_top_levels_and_profitable_long_swings")):
        raise ValueError("The direction evaluator cannot run a swing protocol; "
                         "use the swing research APIs with verified campaign records.")
    registration = _preregistration(protocol_path, project, preregistration_receipt)
    runtime = runtime_fingerprint()
    started = datetime.now(timezone.utc).isoformat()
    run_id = "v2-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ-") + uuid4().hex[:10]
    run = root / run_id
    run.mkdir(parents=True, exist_ok=False)
    files = []

    def capture(path: Path, relative: str, role: str) -> None:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Cannot capture missing or symlink input {path}")
        target = run / relative
        _write(target, path.read_bytes())
        files.append(_entry(run, target, role))

    capture(protocol_path, "frozen/config/evaluation-v2.yml", "protocol")
    if registration["status"] == "matched":
        capture(Path(registration["original_receipt_path"]), "protocol-preregistration.json",
                "protocol_preregistration")
    for path in sorted((source_root / "src/twiz").rglob("*.py")):
        if not path.resolve().is_relative_to(source_root):
            raise ValueError("Source tree contains an external symlink")
        capture(path, "frozen/" + path.relative_to(source_root).as_posix(), "source_code")
    for name in (*DATA_FILES, *EXTRA_DATA_FILES):
        path = dataset_dir / name
        if path.is_file():
            capture(path, "frozen/datasets/" + name, "dataset")
    _write(run / "worker.py", WORKER.encode())
    files.append(_entry(run, run / "worker.py", "worker_launcher"))
    inputs = {"schema_version": 1, "run_id": run_id, "started_at_utc": started,
              "files": files, "runtime": runtime,
              "protocol_preregistration": registration,
              "source_tree_sha256": _hash(_json([r for r in files if r["role"] == "source_code"])),
              "phase_order": "pretest_selection_then_seal_then_full_evaluation",
              "dataset_note": "Frozen supplied history; "
                              "capture alone does not establish historical PIT."}
    content = _json(inputs)
    _write(run / "input-manifest.json", content)
    _write(run / "input-manifest.sha256", (_hash(content)+"\n").encode())
    _write(run / "started.json", _json({"run_id": run_id, "started_at_utc": started,
                                       "input_manifest_sha256": _hash(content),
                                       "status_note": "Start record is not "
                                                      "proof of current activity."}))
    _verify_input_archive(run, _hash(content))
    _launch_phase(run, "selection")
    _verify_input_archive(run, _hash(content))
    selection = (run / "pretest-selection.json").read_bytes()
    _write(run / "pretest-selection-seal.json", _json({
        "selection_sha256": _hash(selection), "input_manifest_sha256": _hash(content),
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_scope": ["pretest-market.json", "pretest-macro.json"],
    }))
    _launch_phase(run, "evaluation", _hash(selection))
    _verify_input_archive(run, _hash(content))
    result = json.loads((run / "result.json").read_bytes())
    summary = {"status": result.get("status"), "model_version": result.get("model_version"),
               "goal_completion_proven": False,
               "research_note": "A completed research run does not establish the user's goal.",
               "protocol_preregistration": registration,
               "input_availability": result.get("input_availability"),
               "primary": result.get("primary", {}).get("primary_horizon"),
               "individual": result.get("individual", {}).get("primary_horizon")}
    _write(run / "summary.json", _json(summary))
    _verify_input_archive(run)
    _verify_selection(run)
    included = {item["path"] for item in files}
    outputs = [_entry(run, path, "run_artifact") for path in sorted(run.iterdir())
               if path.is_file() and path.name not in included]
    completion = {"schema_version": 1, "run_id": run_id, "status": "complete",
                  "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "goal_completion_proven": False, "files": [*files, *outputs]}
    payload = _json(completion)
    _write(run / "manifest.json", payload)
    _write(run / "manifest.sha256", (_hash(payload)+"\n").encode())
    verification = verify_evaluation_experiment(run, _hash(payload))
    if not verification["valid"]:
        raise ValueError("Completion verification failed: " + "; ".join(verification["errors"]))
    receipt = {"run_id": run_id, "path": str(run), "manifest_sha256": _hash(payload),
               "status": "complete", "goal_completion_proven": False}
    latest = root.parent / "latest-experiment-v2.json"
    temporary = latest.with_name(latest.name + "." + uuid4().hex + ".tmp")
    _write(temporary, _json(receipt))
    os.replace(temporary, latest)
    return receipt


def verify_evaluation_experiment(
    path: Path, expected_manifest_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Read-only verification. Incomplete archives never imply a running process."""
    run = Path(path).resolve()
    if not (run / "manifest.json").is_file():
        try:
            if (run / "input-manifest.json").is_file():
                _verify_input_archive(run)
            if (run / "pretest-selection-seal.json").is_file():
                _verify_selection(run)
        except (OSError, ValueError, TypeError, KeyError) as error:
            return {"valid": False, "status": "integrity_failed", "errors": [str(error)]}
        if any(run.glob("failed-*.json")):
            return {"valid": False, "status": "failed", "errors": [
                "A worker failed. Preserved logs and inputs require review; no automatic restart."
            ]}
        return {"valid": False, "status": "incomplete", "errors": [
            "No completion seal. Interrupted/started files are not proof of current activity."
        ]}
    try:
        data = (run / "manifest.json").read_bytes()
        expected = expected_manifest_sha256 or (run / "manifest.sha256").read_text().strip()
        if _hash(data) != expected:
            raise ValueError("Evaluation completion manifest hash mismatch")
        manifest = json.loads(data)
        if manifest.get("status") != "complete":
            raise ValueError("Evaluation completion status is not complete")
        for item in manifest["files"]:
            _checked_file(run, item)
        _verify_input_archive(run)
        _verify_selection(run)
        return {"valid": True, "status": "complete", "manifest": manifest, "errors": []}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {"valid": False, "status": "integrity_failed", "errors": [str(error)]}
