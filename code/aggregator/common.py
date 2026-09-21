"""Configuration, immutable sources, paths, and manifests for the aggregator."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.contracts import file_snapshot  # noqa: E402
from common.io_utils import (  # noqa: E402
    atomic_write_json,
    load_json,
    read_jsonl,
    resolve_project_path,
    sha256_file,
    sha256_json,
    utc_now,
    validate_run_id,
)


PIPELINE_VERSION = "aggregator-v2"
INPUT_SCHEMA = "aggregator-input-v1"
MODES = {"evidence_only", "with_expert_results"}
BRANCHES = {
    "evidence_only": "aggregator_evidence_only",
    "with_expert_results": "aggregator_with_expert_results",
}
STAGES = ("prepare", "predict", "diagnostics")


class AggregatorError(RuntimeError):
    """Raised when the aggregator must fail closed."""


@dataclass(frozen=True)
class AggregatorRunPaths:
    root: Path
    manifest: Path
    source_snapshot: Path
    inputs: Path
    prompts: Path
    raw: Path
    prediction: Path
    failures: Path
    results: Path
    audit: Path
    summary: Path
    manual_review: Path
    validation_issues: Path


def normalize_mode(value: str) -> str:
    aliases = {
        "EvidenceOnly": "evidence_only",
        "WithExpertResults": "with_expert_results",
        "evidence_only": "evidence_only",
        "with_expert_results": "with_expert_results",
    }
    try:
        return aliases[value]
    except KeyError as error:
        raise AggregatorError(f"Unsupported aggregator mode: {value}") from error


def load_aggregator_config(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema_version") != 1 or value.get("module") != "aggregator":
        raise AggregatorError("Aggregator config must be a schema_version 1 module")
    if value.get("pipeline_version") != PIPELINE_VERSION:
        raise AggregatorError("Unsupported aggregator pipeline_version")
    chat = value.get("chat", {})
    expected_chat = {
        "model": "qwen3.7-plus-2026-05-26",
        "enable_thinking": False,
        "structured_output": True,
        "temperature": 0.0,
        "max_tokens": 512,
        "repair_attempts": 2,
        "timeout_seconds": 120,
        "max_retries": 5,
        "max_prompt_characters": 60000,
    }
    if chat != expected_chat:
        raise AggregatorError("Aggregator chat settings must remain fixed and deterministic")
    if value.get("inputs") != {"max_examples": 8, "minimum_examples_for_complete": 2}:
        raise AggregatorError("Aggregator input limits have changed")
    if value.get("completion") != {"max_failure_rate": 0.03}:
        raise AggregatorError("Aggregator failure tolerance must remain 3%")
    review_size = value.get("diagnostics", {}).get("review_sample_size")
    if not isinstance(review_size, int) or isinstance(review_size, bool) or review_size < 0:
        raise AggregatorError("diagnostics.review_sample_size must be non-negative")
    if not isinstance(value.get("output", {}).get("runs_root"), str):
        raise AggregatorError("output.runs_root must be a path")
    return value


def aggregator_run_paths(config: dict[str, Any], run_id: str) -> AggregatorRunPaths:
    root = (
        resolve_project_path(PROJECT_ROOT, config["output"]["runs_root"])
        / validate_run_id(run_id)
    ).resolve()
    return AggregatorRunPaths(
        root=root,
        manifest=root / "manifest.json",
        source_snapshot=root / "source_snapshot.json",
        inputs=root / "inputs" / "records.jsonl",
        prompts=root / "prompts" / "records.jsonl",
        raw=root / "raw" / "responses.jsonl",
        prediction=root / "prediction" / "records.jsonl",
        failures=root / "prediction" / "failures.jsonl",
        results=root / "prediction" / "results.jsonl",
        audit=root / "audit",
        summary=root / "audit" / "summary.json",
        manual_review=root / "audit" / "manual_review.jsonl",
        validation_issues=root / "audit" / "validation_issues.jsonl",
    )


def _resolved_file(value: str | Path, label: str) -> Path:
    path = resolve_project_path(PROJECT_ROOT, value).resolve()
    if not path.is_file():
        raise AggregatorError(f"{label} does not exist: {path}")
    return path


def _assert_context_manifest(manifest_path: Path, records_path: Path) -> None:
    manifest = load_json(manifest_path)
    if manifest.get("status") != "completed":
        raise AggregatorError("Context manifest must describe a completed run")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise AggregatorError("Context manifest has no output SHA256 ledger")
    candidates = [
        details
        for name, details in outputs.items()
        if name.replace("\\", "/").endswith("context/records.jsonl")
    ]
    if len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise AggregatorError("Context manifest must ledger context/records.jsonl exactly once")
    actual = file_snapshot(records_path)
    if candidates[0].get("sha256") != actual["sha256"]:
        raise AggregatorError("Context records SHA256 does not match its manifest")
    if candidates[0].get("bytes") != actual["bytes"]:
        raise AggregatorError("Context records byte count does not match its manifest")


def snapshot_sources(
    mode: str,
    context_manifest: str | Path,
    context_records: str | Path,
    trf_predictions: str | Path | None,
    exemplar_predictions: str | Path | None,
) -> dict[str, Any]:
    mode = normalize_mode(mode)
    manifest_path = _resolved_file(context_manifest, "Context manifest")
    records_path = _resolved_file(context_records, "Context records")
    _assert_context_manifest(manifest_path, records_path)
    files: dict[str, Any] = {
        "context_manifest": file_snapshot(manifest_path),
        "context_records": file_snapshot(records_path),
    }
    if mode == "evidence_only":
        if trf_predictions is not None or exemplar_predictions is not None:
            raise AggregatorError(
                "evidence_only forbids expert prediction parameters to prevent leakage"
            )
    else:
        if trf_predictions is None or exemplar_predictions is None:
            raise AggregatorError(
                "with_expert_results requires both expert prediction file paths"
            )
        files["trf_predictions"] = file_snapshot(
            _resolved_file(trf_predictions, "TRF expert predictions")
        )
        files["exemplar_predictions"] = file_snapshot(
            _resolved_file(exemplar_predictions, "Exemplar expert predictions")
        )
    return {"schema_version": "aggregator-source-snapshot-v1", "mode": mode, "files": files}


def assert_sources_unchanged(snapshot: dict[str, Any]) -> None:
    for name, expected in snapshot.get("files", {}).items():
        actual = file_snapshot(Path(expected["path"]))
        if actual != expected:
            raise AggregatorError(f"Locked source artifact changed: {name}")


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    value = load_json(path)
    if isinstance(value, dict):
        value = value.get("records")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise AggregatorError(f"Expected JSON or JSONL records: {path}")
    return value


def implementation_hashes() -> dict[str, dict[str, Any]]:
    paths = sorted((PROJECT_ROOT / "code" / "aggregator").glob("*.py"))
    paths.extend(
        [
            PROJECT_ROOT / "code" / "common" / "qwen_client.py",
            PROJECT_ROOT / "code" / "common" / "skill_prediction.py",
            PROJECT_ROOT / "scripts" / "aggregator" / "run.ps1",
            PROJECT_ROOT / "scripts" / "aggregator" / "validate.ps1",
            PROJECT_ROOT / "scripts" / "aggregator" / "evaluate.ps1",
            PROJECT_ROOT / "config" / "aggregator.json",
            PROJECT_ROOT / "environment.yml",
            PROJECT_ROOT / "pyproject.toml",
        ]
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise AggregatorError(f"Implementation file is missing: {missing[0]}")
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    }


def dependency_versions() -> dict[str, str | None]:
    try:
        openai_version = version("openai")
    except PackageNotFoundError:
        openai_version = None
    return {"openai": openai_version}


def target_descriptor(records: list[dict[str, Any]], limit: int | None) -> dict[str, Any]:
    return {
        "record_count": len(records),
        "indexes": [item["idx"] for item in records],
        "identities": [[item["dataset_id"], item["record_id"]] for item in records],
        "records_sha256": sha256_json(records),
        "limit": limit,
    }


def compatibility_payload(
    config_path: Path,
    config: dict[str, Any],
    source: dict[str, Any],
    mode: str,
    descriptor: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "mode": mode,
        "branch": BRANCHES[mode],
        "config": file_snapshot(config_path),
        "source": source,
        "targets": descriptor,
        "chat": config["chat"],
        "inputs": config["inputs"],
        "completion": config["completion"],
    }


def initialize_or_resume_run(
    config_path: Path,
    config: dict[str, Any],
    source: dict[str, Any],
    mode: str,
    run_id: str,
    descriptor: dict[str, Any],
    command: list[str],
    *,
    resume: bool,
) -> AggregatorRunPaths:
    paths = aggregator_run_paths(config, run_id)
    compatibility = compatibility_payload(config_path, config, source, mode, descriptor)
    implementation = implementation_hashes()
    if paths.root.exists():
        if not resume:
            raise AggregatorError(f"Run-id already exists: {paths.root}")
        if not paths.manifest.is_file():
            raise AggregatorError("Existing run has no manifest")
        manifest = load_json(paths.manifest)
        if manifest.get("status") in {"completed", "failed"}:
            raise AggregatorError("Completed or failed aggregator runs are immutable")
        if manifest.get("compatibility") != compatibility:
            raise AggregatorError("Resume compatibility mismatch")
        if manifest.get("implementation") != implementation:
            raise AggregatorError("Implementation changed; use a new run-id")
        manifest["commands"].append({"at": utc_now(), "argv": command})
        manifest.update({"status": "running", "completed_at": None, "error": None})
        atomic_write_json(paths.manifest, manifest)
        return paths
    if resume:
        raise AggregatorError("Cannot resume a run that does not exist")
    for directory in (
        paths.inputs.parent,
        paths.prompts.parent,
        paths.raw.parent,
        paths.prediction.parent,
        paths.audit,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "run_id": run_id,
        "mode": mode,
        "status": "running",
        "created_at": utc_now(),
        "completed_at": None,
        "network_called": False,
        "compatibility": compatibility,
        "implementation": implementation,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        },
        "commands": [{"at": utc_now(), "argv": command}],
        "stages": {name: "pending" for name in STAGES},
        "source_unchanged": None,
        "outputs": {},
        "summary": None,
        "error": None,
    }
    atomic_write_json(paths.manifest, manifest)
    atomic_write_json(paths.source_snapshot, source)
    return paths


def update_manifest(paths: AggregatorRunPaths, **updates: Any) -> dict[str, Any]:
    manifest = load_json(paths.manifest)
    manifest.update(updates)
    atomic_write_json(paths.manifest, manifest)
    return manifest


def update_stage(paths: AggregatorRunPaths, stage: str, status: str) -> None:
    manifest = load_json(paths.manifest)
    manifest["stages"][stage] = status
    atomic_write_json(paths.manifest, manifest)


def collect_output_hashes(paths: AggregatorRunPaths) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(paths.root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path == paths.manifest:
            continue
        result[path.relative_to(paths.root).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return result
