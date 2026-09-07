"""Immutable sources, configuration, run paths, and manifests."""

from __future__ import annotations

import platform
import json
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
from common.contracts import (  # noqa: E402
    file_snapshot,
    record_identity,
    record_source_sha256,
    validate_unique_identities,
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
    if config.get("module") != "instance_discriminator":
        raise InstanceDiscriminatorError("config module must equal instance_discriminator")
    if config.get("pipeline_version") != "instance-discriminator-v1":
        raise InstanceDiscriminatorError("Unsupported discriminator pipeline_version")
    source = config.get("inputs", {})
    if source.get("candidate_count") != 16:
        raise InstanceDiscriminatorError("The candidate contract must remain fixed at 16")
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


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    value = load_json(path)
    if isinstance(value, dict):
        value = value.get("records")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise InstanceDiscriminatorError(f"Expected a JSON/JSONL record collection: {path}")
    return value


def _resolved_input(value: str | Path) -> Path:
    return resolve_project_path(PROJECT_ROOT, value).resolve()


def snapshot_sources(
    targets_path: str | Path,
    candidates_path: str | Path,
    features_path: str | Path | None,
) -> dict[str, Any]:
    paths = {
        "targets": _resolved_input(targets_path),
        "candidates": _resolved_input(candidates_path),
    }
    if features_path is not None:
        paths["features"] = _resolved_input(features_path)
    return {
        "feature_context": "present" if features_path is not None else "absent",
        "files": {name: file_snapshot(path) for name, path in paths.items()},
    }


def assert_sources_unchanged(expected: dict[str, Any]) -> None:
    files = expected.get("files", {})
    current = {
        "feature_context": expected.get("feature_context"),
        "files": {
            name: file_snapshot(Path(details["path"]))
            for name, details in files.items()
        },
    }
    if current != expected:
        raise InstanceDiscriminatorError("Locked source artifacts changed during the run")


def _normalize_targets(records: list[dict[str, Any]], source_hash: str) -> list[dict[str, Any]]:
    dataset_id = f"sentence-dataset-{source_hash[:16]}"
    normalized: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    for position, record in enumerate(records):
        idx = record.get("idx")
        sentence = record.get("sentence")
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise InstanceDiscriminatorError(f"targets[{position}].idx must be an integer")
        if idx in seen_indexes:
            raise InstanceDiscriminatorError("Target idx values must be unique within a run")
        seen_indexes.add(idx)
        if not isinstance(sentence, str) or not sentence.strip():
            raise InstanceDiscriminatorError(f"targets[{position}].sentence is invalid")
        normalized_record = {
            "schema_version": record.get("schema_version", "sentence-record-v1"),
            "dataset_id": record.get("dataset_id", dataset_id),
            "record_id": record.get("record_id", str(idx)),
            "source_sha256": record.get("source_sha256", source_hash),
            **record,
        }
        if not isinstance(normalized_record["schema_version"], str):
            raise InstanceDiscriminatorError(
                f"targets[{position}].schema_version must be a string"
            )
        record_source_sha256(normalized_record, f"targets[{position}]")
        normalized.append(normalized_record)
    validate_unique_identities(normalized, "targets")
    return normalized


def _by_identity(
    records: list[dict[str, Any]], label: str
) -> dict[tuple[str, str], dict[str, Any]]:
    validate_unique_identities(records, label)
    for position, record in enumerate(records):
        if not isinstance(record.get("schema_version"), str):
            raise InstanceDiscriminatorError(
                f"{label}[{position}].schema_version must be a string"
            )
        record_source_sha256(record, f"{label}[{position}]")
    return {record_identity(record, label): record for record in records}


def load_source_bundle(
    targets_path: str | Path,
    candidates_path: str | Path,
    features_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if limit is not None and limit <= 0:
        raise InstanceDiscriminatorError("--limit must be positive")
    snapshot = snapshot_sources(targets_path, candidates_path, features_path)
    target_file = Path(snapshot["files"]["targets"]["path"])
    targets = _normalize_targets(
        _load_records(target_file), snapshot["files"]["targets"]["sha256"]
    )
    candidates = _load_records(Path(snapshot["files"]["candidates"]["path"]))
    candidates_by_identity = _by_identity(candidates, "candidates")
    features_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    if features_path is not None:
        features = _load_records(Path(snapshot["files"]["features"]["path"]))
        features_by_identity = _by_identity(features, "features")
    selected_targets = targets[:limit] if limit is not None else targets
    for target in selected_targets:
        identity = record_identity(target, "target")
        candidate = candidates_by_identity.get(identity)
        if candidate is None:
            raise InstanceDiscriminatorError(
                f"No candidate record for target identity {identity!r}"
            )
        if candidate.get("sentence") != target["sentence"]:
            raise InstanceDiscriminatorError(
                f"Target/candidate sentence mismatch for {identity!r}"
            )
        if features_path is not None:
            feature = features_by_identity.get(identity)
            if feature is None or feature.get("sentence") != target["sentence"]:
                raise InstanceDiscriminatorError(
                    f"Missing or mismatched feature record for {identity!r}"
                )
    return {
        "snapshot": snapshot,
        "targets": selected_targets,
        "candidates_by_identity": candidates_by_identity,
        "features_by_identity": features_by_identity,
        "feature_context": "present" if features_path is not None else "absent",
    }


def implementation_hashes() -> dict[str, dict[str, Any]]:
    paths = sorted((PROJECT_ROOT / "code" / "instance_discriminator").glob("*.py"))
    paths.extend(
        [
            PROJECT_ROOT / "code" / "common" / "qwen_client.py",
            PROJECT_ROOT / "scripts" / "instance_discriminator" / "run.ps1",
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
        "identities": [list(record_identity(item, "target")) for item in records],
        "records_sha256": sha256_json(records),
        "limit": limit,
    }


def initialize_or_resume_run(
    config_path: Path,
    config: dict[str, Any],
    source: dict[str, Any],
    run_id: str,
    descriptor: dict[str, Any],
    command: list[str],
    *,
    resume: bool,
) -> DiscriminatorRunPaths:
    paths = discriminator_run_paths(config, run_id)
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
