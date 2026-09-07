"""Versioned, module-neutral artifact contracts shared by all business modules."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from common.io_utils import load_json, read_jsonl, sha256_file


SENTENCE_DATASET_SCHEMA = "sentence-dataset-v1"
DEMONSTRATION_SET_SCHEMA = "demonstration-set-v1"
DEMONSTRATION_RECORD_SCHEMA = "demonstration-record-v1"
DEMONSTRATION_AUDIT_SCHEMA = "demonstration-audit-v1"
CANDIDATE_INSTANCES_SCHEMA = "candidate-instances-v1"
FEATURE_RECORDS_SCHEMA = "feature-records-v1"
INSTANCE_JUDGMENTS_SCHEMA = "instance-judgments-v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ArtifactContractError(ValueError):
    """Raised when a neutral artifact violates its declared contract."""


def record_identity(record: dict[str, Any], label: str = "record") -> tuple[str, str]:
    dataset_id = record.get("dataset_id")
    record_id = record.get("record_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ArtifactContractError(f"{label}.dataset_id must be a non-empty string")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ArtifactContractError(f"{label}.record_id must be a non-empty string")
    return dataset_id, record_id


def record_source_sha256(record: dict[str, Any], label: str = "record") -> str:
    value = record.get("source_sha256")
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ArtifactContractError(f"{label}.source_sha256 must be a lowercase SHA256")
    return value


def validate_unique_identities(
    records: Iterable[dict[str, Any]], label: str = "records"
) -> None:
    seen: set[tuple[str, str]] = set()
    for position, record in enumerate(records):
        identity = record_identity(record, f"{label}[{position}]")
        if identity in seen:
            raise ArtifactContractError(
                f"{label} contains duplicate identity {identity!r}"
            )
        seen.add(identity)


def file_snapshot(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ArtifactContractError(f"Artifact file does not exist: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def load_demonstration_set(manifest_path: Path) -> dict[str, Any]:
    """Load and verify a canonical demonstration set without knowing its producer."""

    resolved_manifest = manifest_path.resolve()
    manifest = load_json(resolved_manifest)
    if manifest.get("schema_version") != DEMONSTRATION_SET_SCHEMA:
        raise ArtifactContractError("Unsupported demonstration-set schema")
    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ArtifactContractError("Demonstration manifest needs a dataset_id")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"records", "audit"}:
        raise ArtifactContractError(
            "Demonstration manifest files must be exactly records and audit"
        )
    resolved: dict[str, Path] = {}
    for name, details in files.items():
        if not isinstance(details, dict) or not isinstance(details.get("path"), str):
            raise ArtifactContractError(f"Invalid demonstration file entry: {name}")
        path = (resolved_manifest.parent / details["path"]).resolve()
        snapshot = file_snapshot(path)
        if snapshot["sha256"] != details.get("sha256"):
            raise ArtifactContractError(f"Demonstration {name} SHA256 mismatch")
        if snapshot["bytes"] != details.get("bytes"):
            raise ArtifactContractError(f"Demonstration {name} byte count mismatch")
        resolved[name] = path
    records = read_jsonl(resolved["records"])
    audit = read_jsonl(resolved["audit"])
    for position, record in enumerate(records):
        if record.get("schema_version") != DEMONSTRATION_RECORD_SCHEMA:
            raise ArtifactContractError(
                f"records[{position}] has an unsupported schema_version"
            )
        if record.get("dataset_id") != dataset_id:
            raise ArtifactContractError(f"records[{position}] dataset_id mismatch")
        record_source_sha256(record, f"records[{position}]")
    validate_unique_identities(records, "demonstrations")
    if len(audit) != len(records):
        raise ArtifactContractError("Demonstration records and audit counts differ")
    for position, record in enumerate(audit):
        if record.get("schema_version") != DEMONSTRATION_AUDIT_SCHEMA:
            raise ArtifactContractError(
                f"audit[{position}] has an unsupported schema_version"
            )
        if record.get("dataset_id") != dataset_id:
            raise ArtifactContractError(f"audit[{position}] dataset_id mismatch")
        record_source_sha256(record, f"audit[{position}]")
    validate_unique_identities(audit, "demonstration audit")
    counts = manifest.get("counts", {})
    if counts.get("total") != len(records):
        raise ArtifactContractError("Demonstration manifest total is incorrect")
    actual_statuses = {
        status: sum(record.get("status") == status for record in records)
        for status in ("accepted", "negative", "unsolved", "abstained")
    }
    for status, count in actual_statuses.items():
        if counts.get(status) != count:
            raise ArtifactContractError(
                f"Demonstration manifest {status} count is incorrect"
            )
    return {
        "manifest": manifest,
        "manifest_path": resolved_manifest,
        "records_path": resolved["records"],
        "audit_path": resolved["audit"],
        "records": records,
        "audit": audit,
        "snapshot": {
            "dataset_id": dataset_id,
            "manifest": file_snapshot(resolved_manifest),
            "records": file_snapshot(resolved["records"]),
            "audit": file_snapshot(resolved["audit"]),
        },
    }
