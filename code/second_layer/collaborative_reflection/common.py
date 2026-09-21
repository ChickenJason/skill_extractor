"""Configuration, paths, immutable snapshots, and manifests for reflection runs."""

from __future__ import annotations

import copy
import platform
import sys
import time
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
    resolve_project_path,
    sha256_file,
    utc_now,
    validate_run_id,
)


PIPELINE_VERSION = "collaborative-reflection-context-v1"
INPUT_VERSION = "collaborative-reflection-input-v1"
BASE_PIPELINE_VERSION = "second-layer-parallel-v1"
TRF_SEMANTIC_CONTRACT_PATH = PROJECT_ROOT / "config" / "trf_reflection_semantics.json"
SCHEDULE_TYPE = "collaborative_reflection"
BASE_SCHEDULE_TYPE = "concurrent"
BRANCHES = ("trf", "exemplar")
STAGES = (
    "validate_base",
    "freeze_base",
    "prepare_reflection",
    "reflect_parallel",
    "validate_reflections",
    "assemble_context",
    "diagnostics",
)


class ReflectionError(RuntimeError):
    """Raised when a collaborative-reflection contract must fail closed."""


def write_manifest(path: Path, value: dict[str, Any]) -> None:
    """Atomically write a reflection manifest, tolerating brief Windows read locks."""

    for attempt in range(5):
        try:
            atomic_write_json(path, value)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


@dataclass(frozen=True)
class ReflectionPaths:
    root: Path
    manifest: Path
    source: Path
    inputs: Path
    trf: Path
    exemplar: Path
    interaction: Path
    context: Path
    audit: Path
    logs: Path

    def branch(self, name: str) -> Path:
        if name == "trf":
            return self.trf
        if name == "exemplar":
            return self.exemplar
        raise ReflectionError(f"Unknown reflection branch: {name}")


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ReflectionError(f"{label} must be a positive integer")
    return value


def load_reflection_config(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema_version") != 1 or value.get("module") != "second_layer_reflection":
        raise ReflectionError("Reflection config must be a schema_version 1 module")
    if value.get("pipeline_version") != PIPELINE_VERSION:
        raise ReflectionError("Unsupported reflection pipeline_version")
    if not TRF_SEMANTIC_CONTRACT_PATH.is_file():
        raise ReflectionError(
            f"TRF semantic contract does not exist: {TRF_SEMANTIC_CONTRACT_PATH}"
        )
    base = value.get("base")
    if not isinstance(base, dict) or set(base) != {
        "concurrent_config", "accepted_pipeline_versions", "expected_target_count"
    }:
        raise ReflectionError("Reflection base config contract is invalid")
    if base.get("accepted_pipeline_versions") != [BASE_PIPELINE_VERSION]:
        raise ReflectionError("Reflection only accepts the concurrent v1 baseline")
    _require_positive_int(base.get("expected_target_count"), "base.expected_target_count")
    concurrent_path = resolve_project_path(PROJECT_ROOT, base["concurrent_config"]).resolve()
    if not concurrent_path.is_file():
        raise ReflectionError(f"Concurrent config does not exist: {concurrent_path}")

    provider = value.get("provider")
    chat = value.get("chat")
    gate = value.get("gate")
    reflection = value.get("reflection")
    if not isinstance(provider, dict) or set(provider) != {"name", "api_key", "base_url"}:
        raise ReflectionError("Reflection provider contract is invalid")
    if not isinstance(chat, dict):
        raise ReflectionError("Reflection chat config is missing")
    required_chat = {
        "model", "enable_thinking", "structured_output", "temperature", "max_tokens",
        "timeout_seconds", "max_retries", "max_prompt_characters", "max_reason_characters",
    }
    if set(chat) != required_chat:
        raise ReflectionError("Reflection chat config has missing or unknown keys")
    if chat["temperature"] != 0.0 or chat["structured_output"] is not True:
        raise ReflectionError("Reflection v1 requires temperature 0 and structured output")
    for key in ("max_tokens", "max_retries", "max_prompt_characters", "max_reason_characters"):
        _require_positive_int(chat[key], f"chat.{key}")
    if not isinstance(reflection, dict) or set(reflection) != {
        "candidate_count", "max_repair_attempts"
    }:
        raise ReflectionError("Reflection protocol config is invalid")
    if _require_positive_int(reflection["candidate_count"], "reflection.candidate_count") != 16:
        raise ReflectionError("Reflection v1 requires exactly 16 candidates")
    if reflection["max_repair_attempts"] != 2:
        raise ReflectionError("Reflection v1 requires exactly two structured repairs")
    expected_gate = {
        "minimum_helpfulness", "max_selected", "max_supporting",
        "max_contrastive", "minimum_for_complete",
    }
    if not isinstance(gate, dict) or set(gate) != expected_gate:
        raise ReflectionError("Reflection hard-gate config is invalid")
    for key in expected_gate:
        _require_positive_int(gate[key], f"gate.{key}")
    output_root = value.get("output", {}).get("runs_root")
    if not isinstance(output_root, str) or not output_root.strip():
        raise ReflectionError("Reflection output.runs_root must be a path")
    diagnostics = value.get("diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != {"review_sample_size"}:
        raise ReflectionError("Reflection diagnostics config is invalid")
    _require_positive_int(diagnostics["review_sample_size"], "diagnostics.review_sample_size")

    config = copy.deepcopy(value)
    config["base"]["concurrent_config"] = str(concurrent_path)
    return config


def reflection_paths(config: dict[str, Any], run_id: str) -> ReflectionPaths:
    root = (resolve_project_path(PROJECT_ROOT, config["output"]["runs_root"]) / validate_run_id(run_id)).resolve()
    return ReflectionPaths(
        root=root,
        manifest=root / "manifest.json",
        source=root / "source",
        inputs=root / "source" / "reflection-inputs.jsonl",
        trf=root / "trf-reflection",
        exemplar=root / "exemplar-reflection",
        interaction=root / "interaction",
        context=root / "context",
        audit=root / "audit",
        logs=root / "logs",
    )


def branch_run_ids(run_id: str) -> dict[str, str]:
    validate_run_id(run_id)
    return {
        "trf": validate_run_id(f"{run_id}-trf-reflection"),
        "exemplar": validate_run_id(f"{run_id}-exemplar-reflection"),
    }


def file_snapshot(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ReflectionError(f"Required artifact does not exist: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def implementation_hashes() -> dict[str, dict[str, Any]]:
    files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    files.extend(
        [
            PROJECT_ROOT / "config" / "second_layer_reflection.json",
            TRF_SEMANTIC_CONTRACT_PATH,
            PROJECT_ROOT / "scripts" / "second_layer" / "run_reflection.ps1",
            PROJECT_ROOT / "code" / "common" / "qwen_client.py",
            PROJECT_ROOT / "code" / "instance_discriminator" / "pipeline.py",
            PROJECT_ROOT / "environment.yml",
            PROJECT_ROOT / "pyproject.toml",
        ]
    )
    missing = [item for item in files if not item.is_file()]
    if missing:
        raise ReflectionError(f"Reflection implementation file is missing: {missing[0]}")
    return {
        item.relative_to(PROJECT_ROOT).as_posix(): {
            "sha256": sha256_file(item),
            "bytes": item.stat().st_size,
        }
        for item in files
    }


def compatibility_payload(
    *,
    config_path: Path,
    run_id: str,
    base_run_id: str,
    base_manifest: dict[str, Any],
    base_manifest_path: Path,
    base_sources: dict[str, Path],
    identities: list[dict[str, Any]],
    limit: int | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "schedule_type": SCHEDULE_TYPE,
        "base_schedule_type": BASE_SCHEDULE_TYPE,
        "config": file_snapshot(config_path),
        "base": {
            "run_id": base_run_id,
            "pipeline_version": base_manifest.get("pipeline_version"),
            "manifest": file_snapshot(base_manifest_path),
            "validator": "passed",
            "child_manifests": {
                name: file_snapshot(Path(base_manifest["children"][name]["manifest"]))
                for name in BRANCHES
            },
            "artifacts": {
                name: file_snapshot(path) for name, path in base_sources.items()
            },
        },
        "branch_run_ids": branch_run_ids(run_id),
        "target": {"limit": limit, "count": len(identities), "identities": identities},
        "model": {
            "chat": config["chat"]["model"],
            "temperature": config["chat"]["temperature"],
            "structured_output": config["chat"]["structured_output"],
        },
        "trf_semantic_contract": file_snapshot(TRF_SEMANTIC_CONTRACT_PATH),
        "gate": copy.deepcopy(config["gate"]),
    }


def initialize_run(
    paths: ReflectionPaths,
    *,
    run_id: str,
    base_run_id: str,
    compatibility: dict[str, Any],
    command: list[str],
    resume: bool,
) -> dict[str, Any]:
    current_implementation = implementation_hashes()
    if paths.root.exists():
        if not resume:
            raise ReflectionError(f"Reflection run-id already exists: {paths.root}")
        if not paths.manifest.is_file():
            raise ReflectionError("Existing reflection run has no manifest")
        manifest = load_json(paths.manifest)
        if manifest.get("status") == "completed":
            raise ReflectionError("Completed reflection runs are immutable")
        if manifest.get("status") == "failed":
            raise ReflectionError("Failed reflection runs are immutable; use a new run-id")
        if manifest.get("compatibility") != compatibility:
            raise ReflectionError("Reflection resume compatibility mismatch")
        if manifest.get("implementation") != current_implementation:
            raise ReflectionError("Reflection implementation changed; use a new run-id")
        manifest["commands"].append({"at": utc_now(), "argv": command})
        manifest.update({"status": "running", "completed_at": None, "error": None})
        write_manifest(paths.manifest, manifest)
        return manifest
    if resume:
        raise ReflectionError("Cannot resume a reflection run that does not exist")
    paths.root.mkdir(parents=True, exist_ok=False)
    for directory in (
        paths.source, paths.interaction, paths.context, paths.audit, paths.logs
    ):
        directory.mkdir()
    manifest = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "schedule_type": SCHEDULE_TYPE,
        "base_schedule_type": BASE_SCHEDULE_TYPE,
        "run_id": run_id,
        "base_run_id": base_run_id,
        "status": "running",
        "created_at": utc_now(),
        "completed_at": None,
        "network_called": False,
        "network": {name: False for name in BRANCHES},
        "compatibility": compatibility,
        "implementation": current_implementation,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "commands": [{"at": utc_now(), "argv": command}],
        "stages": {stage: "pending" for stage in STAGES},
        "base_validation": None,
        "source_snapshot": None,
        "processes": {},
        "branches": {},
        "summary": None,
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
    if stage not in STAGES:
        raise ReflectionError(f"Unknown reflection stage: {stage}")
    manifest = load_json(paths.manifest)
    manifest["stages"][stage] = status
    write_manifest(paths.manifest, manifest)


def assert_snapshot(snapshot: dict[str, Any]) -> None:
    if not isinstance(snapshot, dict) or not snapshot:
        raise ReflectionError("Reflection source snapshot is missing")
    for label, expected in snapshot.items():
        if label == "schema_version":
            continue
        if not isinstance(expected, dict) or file_snapshot(Path(expected["path"])) != expected:
            raise ReflectionError(f"Frozen reflection source changed: {label}")


def collect_output_hashes(paths: ReflectionPaths) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for path in sorted(paths.root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path != paths.manifest:
            outputs[path.relative_to(paths.root).as_posix()] = {
                "sha256": sha256_file(path), "bytes": path.stat().st_size
            }
    return outputs
