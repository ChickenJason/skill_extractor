"""Versioned configuration, paths, compatibility, and manifests for JSON-Schema reflection."""

from __future__ import annotations

import copy
import platform
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import load_json, resolve_project_path, sha256_file, utc_now, validate_run_id  # noqa: E402
from second_layer.collaborative_reflection.common import (  # noqa: E402
    BASE_PIPELINE_VERSION,
    BASE_SCHEDULE_TYPE,
    BRANCHES,
    PIPELINE_VERSION,
    SCHEDULE_TYPE,
    STAGES,
    TRF_SEMANTIC_CONTRACT_PATH,
    ReflectionError,
    ReflectionPaths,
    assert_snapshot,
    collect_output_hashes,
    file_snapshot,
    write_manifest,
)


PROTOCOL_VERSION = "collaborative-reflection-json-schema-v2"
CONFIG_MODULE = "second_layer_reflection_json_schema"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "second_layer_reflection_json_schema.json"
DEFAULT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "second_layer" / "run_reflection_json_schema.ps1"


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ReflectionError(f"{label} must be a positive integer")
    return value


def load_config(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema_version") != 1 or value.get("module") != CONFIG_MODULE:
        raise ReflectionError("JSON-Schema reflection config has an invalid module contract")
    if value.get("pipeline_version") != PIPELINE_VERSION:
        raise ReflectionError("JSON-Schema reflection must preserve context v1")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise ReflectionError("Unsupported JSON-Schema reflection protocol")
    base = value.get("base")
    if not isinstance(base, dict) or set(base) != {
        "concurrent_config", "accepted_pipeline_versions", "expected_target_count"
    }:
        raise ReflectionError("JSON-Schema reflection base config is invalid")
    if base["accepted_pipeline_versions"] != [BASE_PIPELINE_VERSION]:
        raise ReflectionError("JSON-Schema reflection only accepts concurrent v1")
    _positive_int(base["expected_target_count"], "base.expected_target_count")
    concurrent = resolve_project_path(PROJECT_ROOT, base["concurrent_config"]).resolve()
    if not concurrent.is_file():
        raise ReflectionError(f"Concurrent config does not exist: {concurrent}")

    provider = value.get("provider")
    if not isinstance(provider, dict) or set(provider) != {"name", "api_key", "base_url"}:
        raise ReflectionError("JSON-Schema reflection provider config is invalid")
    chat = value.get("chat")
    expected_chat = {
        "model", "enable_thinking", "structured_output", "response_format_mode",
        "temperature", "max_tokens", "timeout_seconds", "max_retries",
        "max_prompt_characters", "max_reason_characters",
    }
    if not isinstance(chat, dict) or set(chat) != expected_chat:
        raise ReflectionError("JSON-Schema reflection chat config is invalid")
    if chat["structured_output"] is not True or chat["response_format_mode"] != "json_schema":
        raise ReflectionError("JSON-Schema reflection cannot downgrade its response format")
    if chat["enable_thinking"] is not False or chat["temperature"] != 0.0:
        raise ReflectionError("JSON-Schema reflection requires non-thinking temperature-zero inference")
    for key in ("max_tokens", "max_retries", "max_prompt_characters", "max_reason_characters"):
        _positive_int(chat[key], f"chat.{key}")

    reflection = value.get("reflection")
    if not isinstance(reflection, dict) or set(reflection) != {"candidate_count", "max_repair_attempts"}:
        raise ReflectionError("JSON-Schema reflection protocol config is invalid")
    if reflection["candidate_count"] != 16 or reflection["max_repair_attempts"] != 2:
        raise ReflectionError("JSON-Schema reflection requires 16 candidates and two repairs")
    gate = value.get("gate")
    expected_gate = {
        "minimum_helpfulness", "max_selected", "max_supporting",
        "max_contrastive", "minimum_for_complete",
    }
    if not isinstance(gate, dict) or set(gate) != expected_gate:
        raise ReflectionError("JSON-Schema reflection gate config is invalid")
    for key in expected_gate:
        _positive_int(gate[key], f"gate.{key}")
    diagnostics = value.get("diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != {"review_sample_size"}:
        raise ReflectionError("JSON-Schema reflection diagnostics config is invalid")
    _positive_int(diagnostics["review_sample_size"], "diagnostics.review_sample_size")
    output_root = value.get("output", {}).get("runs_root")
    if not isinstance(output_root, str) or not output_root.strip():
        raise ReflectionError("JSON-Schema reflection output root is invalid")
    if not TRF_SEMANTIC_CONTRACT_PATH.is_file():
        raise ReflectionError("TRF semantic contract is missing")
    config = copy.deepcopy(value)
    config["base"]["concurrent_config"] = str(concurrent)
    return config


def paths_for(config: dict[str, Any], run_id: str) -> ReflectionPaths:
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
        "trf": validate_run_id(f"{run_id}-trf-reflection-jsonschema"),
        "exemplar": validate_run_id(f"{run_id}-exemplar-reflection-jsonschema"),
    }


def implementation_hashes() -> dict[str, dict[str, Any]]:
    files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    files.extend(sorted((PROJECT_ROOT / "code" / "second_layer" / "collaborative_reflection").glob("*.py")))
    files.extend(
        [
            DEFAULT_CONFIG_PATH,
            DEFAULT_SCRIPT_PATH,
            TRF_SEMANTIC_CONTRACT_PATH,
            PROJECT_ROOT / "code" / "common" / "qwen_client.py",
            PROJECT_ROOT / "code" / "instance_discriminator" / "pipeline.py",
            PROJECT_ROOT / "environment.yml",
            PROJECT_ROOT / "pyproject.toml",
        ]
    )
    missing = [item for item in files if not item.is_file()]
    if missing:
        raise ReflectionError(f"JSON-Schema reflection implementation file is missing: {missing[0]}")
    return {
        item.relative_to(PROJECT_ROOT).as_posix(): {
            "sha256": sha256_file(item),
            "bytes": item.stat().st_size,
        }
        for item in files
    }


def compatibility_payload(
    *, config_path: Path, run_id: str, base_run_id: str,
    base_manifest: dict[str, Any], base_manifest_path: Path,
    base_sources: dict[str, Path], identities: list[dict[str, Any]],
    limit: int | None, config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
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
            "artifacts": {name: file_snapshot(path) for name, path in base_sources.items()},
        },
        "branch_run_ids": branch_run_ids(run_id),
        "target": {"limit": limit, "count": len(identities), "identities": identities},
        "model": {
            "chat": config["chat"]["model"],
            "temperature": config["chat"]["temperature"],
            "structured_output": config["chat"]["structured_output"],
            "response_format_mode": config["chat"]["response_format_mode"],
        },
        "trf_semantic_contract": file_snapshot(TRF_SEMANTIC_CONTRACT_PATH),
        "gate": copy.deepcopy(config["gate"]),
    }


def initialize_run(
    paths: ReflectionPaths, *, run_id: str, base_run_id: str,
    compatibility: dict[str, Any], command: list[str], resume: bool,
) -> dict[str, Any]:
    implementation = implementation_hashes()
    if paths.root.exists():
        if not resume or not paths.manifest.is_file():
            raise ReflectionError("Existing JSON-Schema reflection run requires a valid Resume")
        manifest = load_json(paths.manifest)
        if manifest.get("status") == "completed":
            raise ReflectionError("Completed JSON-Schema reflection runs are immutable")
        if manifest.get("status") == "failed":
            raise ReflectionError("Failed JSON-Schema reflection runs are immutable; use a new run-id")
        if manifest.get("compatibility") != compatibility:
            raise ReflectionError("JSON-Schema reflection resume compatibility mismatch")
        if manifest.get("implementation") != implementation:
            raise ReflectionError("JSON-Schema reflection implementation changed; use a new run-id")
        manifest["commands"].append({"at": utc_now(), "argv": command})
        manifest.update({"status": "running", "completed_at": None, "error": None})
        write_manifest(paths.manifest, manifest)
        return manifest
    if resume:
        raise ReflectionError("Cannot resume a JSON-Schema reflection run that does not exist")
    paths.root.mkdir(parents=True, exist_ok=False)
    for directory in (paths.source, paths.interaction, paths.context, paths.audit, paths.logs):
        directory.mkdir()
    manifest = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
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
        "implementation": implementation,
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


__all__ = [
    "BASE_SCHEDULE_TYPE", "BRANCHES", "PIPELINE_VERSION", "PROTOCOL_VERSION",
    "SCHEDULE_TYPE", "STAGES", "ReflectionError", "assert_snapshot",
    "branch_run_ids", "collect_output_hashes", "compatibility_payload",
    "file_snapshot", "implementation_hashes", "initialize_run", "load_config",
    "paths_for", "update_manifest", "update_stage", "write_manifest",
]
