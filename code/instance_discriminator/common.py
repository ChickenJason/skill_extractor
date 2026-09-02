"""Immutable sources, configuration, run paths, and manifests."""

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


class InstanceDiscriminatorError(RuntimeError):
    """Raised whenever the discriminator must fail closed."""


@dataclass(frozen=True)
class DiscriminatorRunPaths:
    root: Path
    manifest: Path
    candidates: Path
    prompts: Path
    raw: Path
    parsed: Path
    selected: Path
    audit: Path


def load_discriminator_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    if config.get("schema_version") != 1:
        raise InstanceDiscriminatorError("config schema_version must equal 1")
    if config.get("pipeline_version") != "instance-discriminator-v1":
        raise InstanceDiscriminatorError("Unsupported discriminator pipeline_version")
    source = config.get("source", {})
    if source.get("candidate_count") != 16:
        raise InstanceDiscriminatorError("The candidate contract must remain fixed at 16")
    if not isinstance(source.get("target_runs_root"), str):
        raise InstanceDiscriminatorError("source.target_runs_root must be a path")
    chat = config.get("chat", {})
    if (
        chat.get("model") != "qwen3.7-plus-2026-05-26"
        or chat.get("enable_thinking") is not False
        or chat.get("structured_output") is not True
        or chat.get("temperature") != 0.0
        or chat.get("max_retries") != 5
        or chat.get("max_reason_characters") != 240
    ):
        raise InstanceDiscriminatorError(
            "Discriminator chat must use fixed Qwen structured JSON at temperature 0"
        )
    gate = config.get("gate", {})
    if gate != {
        "minimum_helpfulness": 4,
        "max_selected": 8,
        "max_supporting": 6,
        "max_contrastive": 3,
        "minimum_for_complete": 2,
    }:
        raise InstanceDiscriminatorError("The v1 hard-gate contract has changed")
    review_size = config.get("diagnostics", {}).get("review_sample_size")
    if not isinstance(review_size, int) or review_size < 0:
        raise InstanceDiscriminatorError("review_sample_size must be non-negative")
    if not isinstance(config.get("output", {}).get("runs_root"), str):
        raise InstanceDiscriminatorError("output.runs_root must be a path")
    return config


def discriminator_run_paths(
    config: dict[str, Any], run_id: str
) -> DiscriminatorRunPaths:
    root = (
        resolve_project_path(PROJECT_ROOT, config["output"]["runs_root"])
        / validate_run_id(run_id)
    ).resolve()
    return DiscriminatorRunPaths(
        root=root,
        manifest=root / "manifest.json",
        candidates=root / "candidates",
        prompts=root / "prompts",
        raw=root / "raw",
        parsed=root / "parsed",
        selected=root / "selected",
        audit=root / "audit",
    )


def target_run_root(config: dict[str, Any], target_run_id: str) -> Path:
    return (
        resolve_project_path(PROJECT_ROOT, config["source"]["target_runs_root"])
        / validate_run_id(target_run_id)
    ).resolve()


def _ledger_hash(manifest: dict[str, Any], relative: str) -> str:
    item = manifest.get("outputs", {}).get(relative)
    if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
        raise InstanceDiscriminatorError(
            f"Target TRF manifest does not lock output {relative}"
        )
    return item["sha256"]


def snapshot_sources(config: dict[str, Any], target_run_id: str) -> dict[str, Any]:
    root = target_run_root(config, target_run_id)
    paths = {
        "manifest": root / "manifest.json",
        "source_snapshot": root / "source_snapshot.json",
        "retrieval": root / "retrieval" / "records.jsonl",
        "parsed": root / "parsed" / "records.jsonl",
    }
    for path in paths.values():
        if not path.is_file():
            raise InstanceDiscriminatorError(f"Required target TRF artifact is missing: {path}")
    manifest = load_json(paths["manifest"])
    if manifest.get("run_id") != target_run_id:
        raise InstanceDiscriminatorError("Target TRF manifest run_id mismatch")
    if manifest.get("status") != "completed" or manifest.get("source_unchanged") is not True:
        raise InstanceDiscriminatorError("Target TRF run is not safely completed")
    for name, relative in {
        "source_snapshot": "source_snapshot.json",
        "retrieval": "retrieval/records.jsonl",
        "parsed": "parsed/records.jsonl",
    }.items():
        expected = _ledger_hash(manifest, relative)
        actual = sha256_file(paths[name])
        if actual != expected:
            raise InstanceDiscriminatorError(
                f"Target TRF output hash mismatch for {relative}: expected {expected}, got {actual}"
            )
    target_source = load_json(paths["source_snapshot"])
    decision_item = target_source.get("files", {}).get("decisions")
    if not isinstance(decision_item, dict):
        raise InstanceDiscriminatorError("Target source snapshot has no decisions artifact")
    decision_path = Path(decision_item.get("path", "")).resolve()
    if not decision_path.is_file():
        raise InstanceDiscriminatorError(f"Source decisions are missing: {decision_path}")
    decision_hash = sha256_file(decision_path)
    if decision_hash != decision_item.get("sha256"):
        raise InstanceDiscriminatorError("Source decisions hash mismatch")
    compatibility_decision = (
        manifest.get("compatibility", {})
        .get("source", {})
        .get("files", {})
        .get("decisions")
    )
    if compatibility_decision != decision_item:
        raise InstanceDiscriminatorError(
            "Target manifest and target source snapshot disagree on decisions"
        )
    all_paths = {**paths, "decisions": decision_path}
    return {
        "target_trf_run_id": target_run_id,
        "files": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in all_paths.items()
        },
    }


def assert_sources_unchanged(
    config: dict[str, Any], target_run_id: str, expected: dict[str, Any]
) -> None:
    if snapshot_sources(config, target_run_id) != expected:
        raise InstanceDiscriminatorError("Locked source artifacts changed during the run")


def _unique_by_idx(records: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        idx = record.get("idx")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx in result:
            raise InstanceDiscriminatorError(f"{label} has an invalid or duplicate idx")
        result[idx] = record
    return result


def load_source_bundle(
    config: dict[str, Any], target_run_id: str, limit: int | None = None
) -> dict[str, Any]:
    if limit is not None and limit <= 0:
        raise InstanceDiscriminatorError("--limit must be positive")
    snapshot = snapshot_sources(config, target_run_id)
    files = snapshot["files"]
    parsed = read_jsonl(Path(files["parsed"]["path"]))
    retrieval = read_jsonl(Path(files["retrieval"]["path"]))
    decisions = read_jsonl(Path(files["decisions"]["path"]))
    parsed_by_idx = _unique_by_idx(parsed, "target parsed records")
    retrieval_by_idx = _unique_by_idx(retrieval, "target retrieval records")
    decisions_by_idx = _unique_by_idx(decisions, "source decisions")
    if set(parsed_by_idx) != set(retrieval_by_idx):
        raise InstanceDiscriminatorError("Target parsed and retrieval indexes do not match")
    manifest = load_json(Path(files["manifest"]["path"]))
    expected_indexes = manifest.get("compatibility", {}).get("targets", {}).get("indexes")
    if expected_indexes != [item["idx"] for item in parsed]:
        raise InstanceDiscriminatorError("Target records do not match manifest indexes")
    for item in parsed:
        idx = item["idx"]
        if retrieval_by_idx[idx].get("sentence") != item.get("sentence"):
            raise InstanceDiscriminatorError(
                f"Target parsed/retrieval sentence mismatch for idx={idx}"
            )
    targets = parsed[:limit] if limit is not None else parsed
    return {
        "snapshot": snapshot,
        "target_manifest": manifest,
        "targets": targets,
        "retrieval_by_idx": retrieval_by_idx,
        "decisions_by_idx": decisions_by_idx,
    }


def implementation_hashes() -> dict[str, dict[str, Any]]:
    paths = sorted((PROJECT_ROOT / "code" / "instance_discriminator").glob("*.py"))
    paths.extend(
        [
            PROJECT_ROOT / "code" / "common" / "qwen_client.py",
            PROJECT_ROOT / "scripts" / "run-instance-discriminator.ps1",
            PROJECT_ROOT / "environment.yml",
            PROJECT_ROOT / "pyproject.toml",
        ]
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise InstanceDiscriminatorError(f"Implementation file is missing: {missing[0]}")
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    }


def dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("openai",):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def target_descriptor(records: list[dict[str, Any]], limit: int | None) -> dict[str, Any]:
    return {
        "record_count": len(records),
        "indexes": [item["idx"] for item in records],
        "records_sha256": sha256_json(records),
        "limit": limit,
    }


def initialize_or_resume_run(
    config_path: Path,
    config: dict[str, Any],
    target_run_id: str,
    run_id: str,
    descriptor: dict[str, Any],
    command: list[str],
    *,
    resume: bool,
) -> DiscriminatorRunPaths:
    paths = discriminator_run_paths(config, run_id)
    source = snapshot_sources(config, target_run_id)
    compatibility = {
        "pipeline_version": config["pipeline_version"],
        "config_sha256": sha256_file(config_path),
        "source": source,
        "targets": descriptor,
        "chat": config["chat"],
        "gate": config["gate"],
    }
    current_implementation = implementation_hashes()
    if paths.root.exists():
        if not resume:
            raise InstanceDiscriminatorError(f"Run-id already exists: {paths.root}")
        if not paths.manifest.is_file():
            raise InstanceDiscriminatorError("Existing run has no manifest")
        manifest = load_json(paths.manifest)
        if manifest.get("status") == "completed":
            raise InstanceDiscriminatorError("Completed discriminator runs are immutable")
        if manifest.get("compatibility") != compatibility:
            raise InstanceDiscriminatorError("Resume compatibility mismatch")
        if manifest.get("implementation") != current_implementation:
            raise InstanceDiscriminatorError("Implementation changed; use a new run-id")
        manifest["commands"].append({"at": utc_now(), "argv": command})
        manifest.update({"status": "running", "completed_at": None, "error": None})
        atomic_write_json(paths.manifest, manifest)
        return paths
    if resume:
        raise InstanceDiscriminatorError("Cannot resume a run that does not exist")
    paths.root.mkdir(parents=True, exist_ok=False)
    for directory in (
        paths.candidates,
        paths.prompts,
        paths.raw,
        paths.parsed,
        paths.selected,
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
        "implementation": current_implementation,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        },
        "commands": [{"at": utc_now(), "argv": command}],
        "stages": {
            "prepare": "pending",
            "discriminate": "pending",
            "diagnostics": "pending",
        },
        "source_unchanged": None,
        "outputs": {},
        "summary": None,
        "error": None,
    }
    atomic_write_json(paths.manifest, manifest)
    atomic_write_json(paths.root / "source_snapshot.json", source)
    return paths


def update_manifest(paths: DiscriminatorRunPaths, **updates: Any) -> dict[str, Any]:
    manifest = load_json(paths.manifest)
    manifest.update(updates)
    atomic_write_json(paths.manifest, manifest)
    return manifest


def update_stage(paths: DiscriminatorRunPaths, stage: str, status: str) -> None:
    manifest = load_json(paths.manifest)
    manifest["stages"][stage] = status
    atomic_write_json(paths.manifest, manifest)


def collect_output_hashes(paths: DiscriminatorRunPaths) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(paths.root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path == paths.manifest:
            continue
        relative = path.relative_to(paths.root).as_posix()
        result[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def latest_records(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for line_number, record in enumerate(read_jsonl(path), start=1):
        idx = record.get("idx")
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise InstanceDiscriminatorError(
                f"Invalid idx in {path.name} record {line_number}"
            )
        result[idx] = record
    return result
