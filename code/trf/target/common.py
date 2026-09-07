"""Contracts, immutable source checks, and manifests for target TRF extraction."""

from __future__ import annotations

import json
import copy
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

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
from common.contracts import (  # noqa: E402
    file_snapshot,
    record_source_sha256,
    validate_unique_identities,
)


class TargetTRFError(RuntimeError):
    """Raised when a target-TRF contract must fail closed."""


@dataclass(frozen=True)
class TargetRunPaths:
    root: Path
    manifest: Path
    targets: Path
    embeddings: Path
    retrieval: Path
    prompts: Path
    raw: Path
    parsed: Path
    audit: Path


def load_target_config(path: Path) -> dict[str, Any]:
    root_config = load_json(path)
    if root_config.get("module") == "trf":
        config = copy.deepcopy(root_config.get("target"))
        if not isinstance(config, dict):
            raise TargetTRFError("TRF module config must define target settings")
        config["output"] = {"runs_root": root_config["output"]["runs_root"]}
    else:
        config = root_config
    if config.get("schema_version") != 1:
        raise TargetTRFError("TRF target schema_version must equal 1")
    if config.get("pipeline_version") != "trf-target-extractor-v1":
        raise TargetTRFError("Unsupported target TRF pipeline_version")
    if config.get("configuration_role") not in {
        "module-template",
        "generated-run-config",
    }:
        raise TargetTRFError("Unsupported target TRF configuration_role")
    if not isinstance(config.get("output", {}).get("runs_root"), str):
        raise TargetTRFError("TRF target output.runs_root must be a path")
    is_template = config["configuration_role"] == "module-template"
    source_trf = config.get("source_trf", {})
    source_run_id = source_trf.get("run_id")
    source_run_root = source_trf.get("run_root")
    if not is_template and (
        not isinstance(source_run_id, str) or not isinstance(source_run_root, str)
    ):
        raise TargetTRFError("source_trf must define run_id and run_root")
    if not is_template:
        try:
            validate_run_id(source_run_id)
        except ValueError as error:
            raise TargetTRFError(str(error)) from error
        if Path(source_run_root).name != "offline":
            raise TargetTRFError("source_trf.run_root must point to an offline stage")
        if set(source_trf.get("files", {})) != {
            "manifest",
            "corpus",
            "candidates",
            "pseudo_trfs",
        }:
            raise TargetTRFError(
                "source_trf.files does not match the fixed artifact contract"
            )
        demonstrations = config.get("demonstrations", {})
        if demonstrations.get("expected_count") != 326:
            raise TargetTRFError(
                "The demonstration contract must remain fixed at 326 records"
            )
        if set(demonstrations.get("files", {})) != {"manifest", "records"}:
            raise TargetTRFError(
                "demonstrations.files must contain manifest and records"
            )
    embedding = config.get("embedding", {})
    if embedding != {
        "model": "qwen3.7-text-embedding",
        "dimensions": 1024,
        "batch_size": 20,
    }:
        raise TargetTRFError("The fixed qwen3.7 embedding contract has changed")
    retrieval = config.get("retrieval", {})
    if retrieval.get("nearest_neighbors") != 50 or retrieval.get("demonstrations") != 16:
        raise TargetTRFError("Retrieval must remain K=50 and k=16")
    candidates = config.get("candidates", {})
    expected_main_trfs = candidates.get("expected_main_trfs")
    if not isinstance(expected_main_trfs, int) or expected_main_trfs <= 0:
        raise TargetTRFError(
            "candidates.expected_main_trfs must be a positive integer"
        )
    chat = config.get("chat", {})
    if (
        chat.get("model") != "qwen3.7-plus-2026-05-26"
        or chat.get("enable_thinking") is not False
        or chat.get("temperature") != 0.0
        or chat.get("structured_output") is not True
        or chat.get("max_retries") != 5
    ):
        raise TargetTRFError("Target TRF chat must use temperature=0 and structured JSON")
    if config.get("diagnostics", {}).get("review_sample_size") != 20:
        raise TargetTRFError("The manual review sample must remain fixed at 20")
    return config


def target_run_paths(config: dict[str, Any], run_id: str) -> TargetRunPaths:
    root = (
        resolve_project_path(PROJECT_ROOT, config["output"]["runs_root"])
        / validate_run_id(run_id)
        / "target"
    ).resolve()
    return TargetRunPaths(
        root=root,
        manifest=root / "manifest.json",
        targets=root / "targets",
        embeddings=root / "embeddings",
        retrieval=root / "retrieval",
        prompts=root / "prompts",
        raw=root / "raw",
        parsed=root / "parsed",
        audit=root / "audit",
    )


def source_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = resolve_project_path(PROJECT_ROOT, config["source_trf"]["run_root"]).resolve()
    paths = {
        name: (root / details["path"]).resolve()
        for name, details in config["source_trf"]["files"].items()
    }
    paths["demonstration_manifest"] = resolve_project_path(
        PROJECT_ROOT, config["demonstrations"]["files"]["manifest"]["path"]
    ).resolve()
    paths["demonstration_records"] = resolve_project_path(
        PROJECT_ROOT, config["demonstrations"]["files"]["records"]["path"]
    ).resolve()
    return paths


def _load_feedback_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = read_jsonl(path)
    else:
        value = load_json(path)
        records = value.get("records") if isinstance(value, dict) else value
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise TargetTRFError("Feedback must be a JSON/JSONL record collection")
    validate_unique_identities(records, "feedback")
    for position, record in enumerate(records):
        if not isinstance(record.get("schema_version"), str):
            raise TargetTRFError(
                f"feedback[{position}].schema_version must be a string"
            )
        record_source_sha256(record, f"feedback[{position}]")
    return records


def snapshot_sources(
    config: dict[str, Any], feedback_path: str | Path | None = None
) -> dict[str, Any]:
    paths = source_paths(config)
    expected = {
        name: details["sha256"]
        for name, details in config["source_trf"]["files"].items()
    }
    expected.update(
        {
            f"demonstration_{name}": details["sha256"]
            for name, details in config["demonstrations"]["files"].items()
        }
    )
    snapshot: dict[str, Any] = {
        "trf_run_id": config["source_trf"]["run_id"],
        "demonstration_dataset_id": config["demonstrations"]["dataset_id"],
        "feedback": {"context": "absent"},
        "files": {},
    }
    for name, path in paths.items():
        if not path.is_file():
            raise TargetTRFError(f"Required source file does not exist: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected[name]:
            raise TargetTRFError(
                f"Source hash mismatch for {name}: expected {expected[name]}, got {actual_hash}"
            )
        snapshot["files"][name] = {
            "path": str(path),
            "sha256": actual_hash,
            "bytes": path.stat().st_size,
        }
    manifest = load_json(paths["manifest"])
    if manifest.get("run_id") != config["source_trf"]["run_id"]:
        raise TargetTRFError("Source TRF manifest run_id mismatch")
    if manifest.get("status") != "completed" or manifest.get("source_unchanged") is not True:
        raise TargetTRFError("Source TRF run is not safely completed")
    if feedback_path is not None:
        feedback = resolve_project_path(PROJECT_ROOT, feedback_path).resolve()
        if not feedback.is_file():
            raise TargetTRFError(f"Feedback file does not exist: {feedback}")
        feedback_records = _load_feedback_records(feedback)
        snapshot["feedback"] = {
            "context": "present",
            "file": file_snapshot(feedback),
            "record_count": len(feedback_records),
        }
    return snapshot


def assert_sources_unchanged(config: dict[str, Any], expected: dict[str, Any]) -> None:
    feedback = expected.get("feedback", {})
    feedback_path = (
        feedback.get("file", {}).get("path")
        if feedback.get("context") == "present"
        else None
    )
    if snapshot_sources(config, feedback_path) != expected:
        raise TargetTRFError("A locked TRF source or feedback file changed during the run")


def implementation_hashes() -> dict[str, dict[str, Any]]:
    paths = sorted((PROJECT_ROOT / "code" / "trf" / "target").glob("*.py"))
    paths.extend(
        [
            PROJECT_ROOT / "code" / "common" / "qwen_client.py",
            PROJECT_ROOT / "scripts" / "trf" / "run.ps1",
            PROJECT_ROOT / "environment.yml",
            PROJECT_ROOT / "pyproject.toml",
        ]
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise TargetTRFError(f"Implementation file is missing: {missing[0]}")
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    }


def dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("numpy", "openai"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def stable_target_descriptor(
    mode: str, records: list[dict[str, Any]], input_path: Path | None, limit: int | None
) -> dict[str, Any]:
    return {
        "mode": mode,
        "input_path": str(input_path.resolve()) if input_path else None,
        "input_sha256": sha256_file(input_path) if input_path else None,
        "records_sha256": sha256_json(records),
        "record_count": len(records),
        "limit": limit,
        "indexes": [record["idx"] for record in records],
        "identities": [
            [record["dataset_id"], record["record_id"]] for record in records
        ],
    }


def initialize_or_resume_run(
    config_path: Path,
    config: dict[str, Any],
    run_id: str,
    target_descriptor: dict[str, Any],
    command: list[str],
    *,
    resume: bool,
    feedback_path: str | Path | None = None,
) -> TargetRunPaths:
    paths = target_run_paths(config, run_id)
    current_source = snapshot_sources(config, feedback_path)
    compatibility = {
        "pipeline_version": config["pipeline_version"],
        "config_sha256": sha256_file(config_path),
        "source": current_source,
        "targets": target_descriptor,
        "embedding": config["embedding"],
        "retrieval": config["retrieval"],
        "chat": config["chat"],
    }
    if paths.root.exists():
        if not resume:
            raise TargetTRFError(f"Target TRF run-id already exists: {paths.root}")
        if not paths.manifest.is_file():
            raise TargetTRFError("Existing run has no manifest")
        manifest = load_json(paths.manifest)
        if manifest.get("status") == "completed":
            raise TargetTRFError("Completed target TRF runs are immutable")
        if manifest.get("compatibility") != compatibility:
            raise TargetTRFError("Resume compatibility mismatch")
        if manifest.get("implementation") != implementation_hashes():
            raise TargetTRFError("Implementation changed; resume with a new run-id")
        manifest["commands"].append({"at": utc_now(), "argv": command})
        manifest["status"] = "running"
        manifest["error"] = None
        atomic_write_json(paths.manifest, manifest)
        return paths

    if resume:
        raise TargetTRFError("Cannot resume a target TRF run that does not exist")
    paths.root.mkdir(parents=True, exist_ok=False)
    for directory in (
        paths.targets,
        paths.embeddings,
        paths.retrieval,
        paths.prompts,
        paths.raw,
        paths.parsed,
        paths.audit,
    ):
        directory.mkdir()
    manifest = {
        "schema_version": 1,
        "pipeline_version": config["pipeline_version"],
        "run_id": run_id,
        "status": "running",
        "created_at": utc_now(),
        "completed_at": None,
        "network_called": False,
        "compatibility": compatibility,
        "implementation": implementation_hashes(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        },
        "commands": [{"at": utc_now(), "argv": command}],
        "stages": {
            "prepare_targets": "pending",
            "embed_and_retrieve": "pending",
            "extract_target_trfs": "pending",
            "diagnostics": "pending",
        },
        "source_unchanged": None,
        "outputs": {},
        "summary": None,
        "error": None,
    }
    atomic_write_json(paths.manifest, manifest)
    atomic_write_json(paths.root / "source_snapshot.json", current_source)
    return paths


def update_manifest(paths: TargetRunPaths, **updates: Any) -> dict[str, Any]:
    manifest = load_json(paths.manifest)
    manifest.update(updates)
    atomic_write_json(paths.manifest, manifest)
    return manifest


def update_stage(paths: TargetRunPaths, stage: str, status: str) -> None:
    manifest = load_json(paths.manifest)
    manifest["stages"][stage] = status
    atomic_write_json(paths.manifest, manifest)


def collect_output_hashes(paths: TargetRunPaths) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for path in sorted(paths.root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path == paths.manifest:
            continue
        relative = path.relative_to(paths.root).as_posix()
        values[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return values


def latest_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for line_number, record in enumerate(read_jsonl(path), start=1):
        idx = record.get("idx")
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise TargetTRFError(f"Invalid idx in {path.name} record {line_number}")
        records[idx] = record
    return records


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
