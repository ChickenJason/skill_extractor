"""Contracts and immutable manifests for the stratified reflection gate."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import load_json, sha256_file, utc_now  # noqa: E402
from second_layer.collaborative_reflection.common import (  # noqa: E402
    BRANCHES,
    PIPELINE_VERSION,
    ReflectionError,
    ReflectionPaths,
    assert_snapshot,
    collect_output_hashes,
    file_snapshot,
    write_manifest,
)
from second_layer.collaborative_reflection_json_schema.common import (  # noqa: E402
    PROTOCOL_VERSION as UNDERLYING_PROTOCOL_VERSION,
    implementation_hashes as underlying_implementation_hashes,
    paths_for,
)


GATE_PROTOCOL_VERSION = "collaborative-reflection-stratified-gate-v1"
SCHEDULE_TYPE = "collaborative_reflection_stratified_gate"
GATE_STAGES = (
    "validate_base",
    "select_gate",
    "freeze_base",
    "prepare_reflection",
    "reflect_parallel",
    "validate_reflections",
    "assemble_context",
    "evaluate_gate",
    "diagnostics",
)
DEFAULT_GATE_SPEC_PATH = PROJECT_ROOT / "config" / "gates" / "collaborative_reflection_stratified12_v1.json"
DEFAULT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "second_layer" / "run_reflection_stratified_gate.ps1"


def implementation_hashes() -> dict[str, Any]:
    files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    files.extend([DEFAULT_GATE_SPEC_PATH, DEFAULT_SCRIPT_PATH])
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise ReflectionError(f"Stratified-gate implementation file is missing: {missing[0]}")
    return {
        "gate": {
            path.relative_to(PROJECT_ROOT).as_posix(): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
        "underlying_json_schema": underlying_implementation_hashes(),
    }


def compatibility_payload(
    *, config_path: Path, gate_spec_path: Path, run_id: str,
    base_run_id: str, base_manifest: dict[str, Any], base_manifest_path: Path,
    base_sources: dict[str, Path], identities: list[dict[str, Any]],
    coverage: dict[str, int], gate_spec: dict[str, Any], config: dict[str, Any],
    branch_run_ids: dict[str, str],
) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "gate_protocol_version": GATE_PROTOCOL_VERSION,
        "underlying_protocol": UNDERLYING_PROTOCOL_VERSION,
        "schedule_type": SCHEDULE_TYPE,
        "base_schedule_type": "concurrent",
        "config": file_snapshot(config_path),
        "gate_spec": file_snapshot(gate_spec_path),
        "gate_id": gate_spec["gate_id"],
        "base": {
            "run_id": base_run_id,
            "pipeline_version": base_manifest.get("pipeline_version"),
            "manifest": file_snapshot(base_manifest_path),
            "validator": "passed",
            "child_manifests": {
                name: file_snapshot(Path(base_manifest["children"][name]["manifest"]))
                for name in BRANCHES
            },
            "artifacts": {name: file_snapshot(path) for name, path in base_sources.items()},
        },
        "branch_run_ids": branch_run_ids,
        "target": {
            "count": len(identities),
            "identities": identities,
            "coverage": coverage,
        },
        "model": {
            "chat": config["chat"]["model"],
            "temperature": config["chat"]["temperature"],
            "response_format_mode": config["chat"]["response_format_mode"],
        },
        "acceptance": gate_spec["acceptance"],
    }


def initialize_run(
    paths: ReflectionPaths, *, run_id: str, base_run_id: str,
    compatibility: dict[str, Any], command: list[str], resume: bool,
) -> dict[str, Any]:
    implementation = implementation_hashes()
    if paths.root.exists():
        if not resume or not paths.manifest.is_file():
            raise ReflectionError("Existing stratified-gate run requires a valid Resume")
        manifest = load_json(paths.manifest)
        if manifest.get("status") == "completed":
            raise ReflectionError("Completed stratified-gate runs are immutable")
        if manifest.get("status") == "failed":
            raise ReflectionError("Failed stratified-gate runs are immutable; use a new run-id")
        if manifest.get("compatibility") != compatibility:
            raise ReflectionError("Stratified-gate resume compatibility mismatch")
        if manifest.get("implementation") != implementation:
            raise ReflectionError("Stratified-gate implementation changed; use a new run-id")
        manifest["commands"].append({"at": utc_now(), "argv": command})
        manifest.update({"status": "running", "completed_at": None, "error": None})
        write_manifest(paths.manifest, manifest)
        return manifest
    if resume:
        raise ReflectionError("Cannot resume a stratified-gate run that does not exist")
    paths.root.mkdir(parents=True, exist_ok=False)
    for directory in (paths.source, paths.interaction, paths.context, paths.audit, paths.logs):
        directory.mkdir()
    manifest = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "gate_protocol_version": GATE_PROTOCOL_VERSION,
        "underlying_protocol": UNDERLYING_PROTOCOL_VERSION,
        "schedule_type": SCHEDULE_TYPE,
        "base_schedule_type": "concurrent",
        "run_id": run_id,
        "base_run_id": base_run_id,
        "gate_id": compatibility["gate_id"],
        "status": "running",
        "gate_status": "not_evaluated",
        "created_at": utc_now(),
        "completed_at": None,
        "network_called": False,
        "network": {name: False for name in BRANCHES},
        "compatibility": compatibility,
        "implementation": implementation,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "commands": [{"at": utc_now(), "argv": command}],
        "stages": {stage: "pending" for stage in GATE_STAGES},
        "base_validation": None,
        "source_snapshot": None,
        "processes": {},
        "branches": {},
        "summary": None,
        "gate_evaluation": None,
        "outputs": {},
        "error": None,
    }
    write_manifest(paths.manifest, manifest)
    return manifest


def update_manifest(paths: ReflectionPaths, **updates: Any) -> dict[str, Any]:
    manifest = load_json(paths.manifest)
    manifest.update(updates)
    write_manifest(paths.manifest, manifest)
    return manifest


def update_stage(paths: ReflectionPaths, stage: str, status: str) -> None:
    if stage not in GATE_STAGES:
        raise ReflectionError(f"Unknown stratified-gate stage: {stage}")
    manifest = load_json(paths.manifest)
    manifest["stages"][stage] = status
    write_manifest(paths.manifest, manifest)


__all__ = [
    "BRANCHES", "DEFAULT_GATE_SPEC_PATH", "GATE_PROTOCOL_VERSION", "GATE_STAGES",
    "PIPELINE_VERSION", "PROJECT_ROOT", "ReflectionError", "SCHEDULE_TYPE",
    "UNDERLYING_PROTOCOL_VERSION", "assert_snapshot", "collect_output_hashes",
    "compatibility_payload", "file_snapshot", "implementation_hashes", "initialize_run",
    "paths_for", "update_manifest", "update_stage",
]
