"""Configuration, immutable paths, manifests, and hashing for interaction runs."""

from __future__ import annotations

import copy
import platform
import sys
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


PIPELINE_VERSION = "bidirectional-interaction-context-v1"
PROTOCOL_VERSION = "trf-exemplar-synchronous-v1"
BASE_PIPELINE_VERSION = "second-layer-parallel-v1"
SCHEDULE_TYPE = "bidirectional_interaction"
BASE_SCHEDULE_TYPE = "concurrent"
CONFIG_MODULE = "second_layer_bidirectional_interaction"
MODES = frozenset({"none", "trf_to_exemplar", "exemplar_to_trf", "bidirectional"})
BRANCHES = ("trf", "exemplar")
STAGES = (
    "validate_base",
    "freeze_base",
    "round_1_prepare",
    "round_1_run",
    "round_1_validate",
    "round_1_apply",
    "round_2_prepare",
    "round_2_run",
    "round_2_validate",
    "round_2_apply",
    "assemble_context",
    "diagnostics",
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "second_layer_bidirectional_interaction.json"
DEFAULT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "second_layer" / "run_bidirectional_interaction.ps1"
DEFAULT_GATE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "second_layer" / "run_bidirectional_interaction_gate.ps1"
TRF_SEMANTIC_CONTRACT_PATH = PROJECT_ROOT / "config" / "trf_reflection_semantics.json"
INSTANCE_CONFIG_PATH = PROJECT_ROOT / "config" / "instance_discriminator.json"


class InteractionError(RuntimeError):
    """A source/config/hash contract violation requiring a fresh run ID."""


class InteractionPartialError(RuntimeError):
    """A recoverable target, transport, parse, or branch failure."""


@dataclass(frozen=True)
class InteractionPaths:
    root: Path
    manifest: Path
    source: Path
    inputs: Path
    rounds: Path
    context: Path
    audit: Path
    logs: Path

    def round(self, number: int) -> Path:
        return self.rounds / f"round-{number:02d}"

    def branch(self, number: int, branch: str) -> Path:
        if branch not in BRANCHES:
            raise InteractionError(f"Unknown interaction branch: {branch}")
        return self.round(number) / "branches" / branch


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise InteractionError(f"{label} must be a positive integer")
    return value


def load_config(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema_version") != 1 or value.get("module") != CONFIG_MODULE:
        raise InteractionError("Interaction config has an invalid module contract")
    if value.get("pipeline_version") != PIPELINE_VERSION:
        raise InteractionError("Unsupported interaction pipeline version")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise InteractionError("Unsupported interaction protocol version")
    base = value.get("base")
    if not isinstance(base, dict) or set(base) != {
        "concurrent_config", "accepted_pipeline_versions", "expected_target_count"
    }:
        raise InteractionError("Interaction base config is invalid")
    if base["accepted_pipeline_versions"] != [BASE_PIPELINE_VERSION]:
        raise InteractionError("Only completed concurrent-v1 baselines are accepted")
    _positive_int(base["expected_target_count"], "base.expected_target_count")
    resources = value.get("resources")
    if not isinstance(resources, dict) or set(resources) != {
        "instance_discriminator_config", "trf_semantic_contract"
    }:
        raise InteractionError("Interaction resources config is invalid")
    provider = value.get("provider")
    if not isinstance(provider, dict) or set(provider) != {"name", "api_key", "base_url"}:
        raise InteractionError("Interaction provider config is invalid")
    chat = value.get("chat")
    expected_chat = {
        "model", "enable_thinking", "structured_output", "response_format_mode",
        "temperature", "max_tokens", "timeout_seconds", "max_retries",
        "max_prompt_characters", "max_reason_characters",
    }
    if not isinstance(chat, dict) or set(chat) != expected_chat:
        raise InteractionError("Interaction chat config is invalid")
    if chat["structured_output"] is not True or chat["response_format_mode"] != "json_schema":
        raise InteractionError("Interaction responses must use strict JSON Schema")
    if chat["enable_thinking"] is not False or chat["temperature"] != 0.0:
        raise InteractionError("Interaction inference must be deterministic")
    for key in ("max_tokens", "max_retries", "max_prompt_characters", "max_reason_characters"):
        _positive_int(chat[key], f"chat.{key}")
    interaction = value.get("interaction")
    if not isinstance(interaction, dict) or set(interaction) != {
        "default_mode", "max_rounds", "candidate_count", "max_repair_attempts"
    }:
        raise InteractionError("Interaction protocol config is invalid")
    if interaction["default_mode"] not in MODES:
        raise InteractionError("Invalid default interaction mode")
    if interaction["max_rounds"] not in {1, 2}:
        raise InteractionError("Interaction max_rounds must be 1 or 2")
    if interaction["candidate_count"] != 16 or interaction["max_repair_attempts"] != 2:
        raise InteractionError("Interaction requires 16 candidates and two repairs")
    output_root = value.get("output", {}).get("runs_root")
    if not isinstance(output_root, str) or not output_root.strip():
        raise InteractionError("Interaction output.runs_root is invalid")

    result = copy.deepcopy(value)
    result["base"]["concurrent_config"] = str(
        resolve_project_path(PROJECT_ROOT, base["concurrent_config"]).resolve()
    )
    for key, raw in resources.items():
        if not isinstance(raw, str) or not raw.strip():
            raise InteractionError(f"resources.{key} must be a path")
        resolved = resolve_project_path(PROJECT_ROOT, raw).resolve()
        if not resolved.is_file():
            raise InteractionError(f"Interaction resource does not exist: {resolved}")
        result["resources"][key] = str(resolved)
    if not Path(result["base"]["concurrent_config"]).is_file():
        raise InteractionError("Concurrent config does not exist")
    return result


def paths_for(config: dict[str, Any], run_id: str) -> InteractionPaths:
    root = (
        resolve_project_path(PROJECT_ROOT, config["output"]["runs_root"])
        / validate_run_id(run_id)
    ).resolve()
    return InteractionPaths(
        root=root,
        manifest=root / "manifest.json",
        source=root / "source",
        inputs=root / "source" / "interaction-inputs.jsonl",
        rounds=root / "rounds",
        context=root / "context",
        audit=root / "audit",
        logs=root / "logs",
    )


def branch_run_id(run_id: str, round_number: int, branch: str) -> str:
    if branch not in BRANCHES or round_number not in {1, 2}:
        raise InteractionError("Invalid branch run identity")
    return validate_run_id(f"{run_id}-r{round_number:02d}-{branch}")


def file_snapshot(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise InteractionError(f"Required artifact does not exist: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def assert_snapshot(snapshot: dict[str, Any]) -> None:
    path = Path(snapshot["path"])
    if file_snapshot(path) != snapshot:
        raise InteractionError(f"Immutable source changed: {path}")


def implementation_hashes() -> dict[str, dict[str, Any]]:
    files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    files.extend(
        [
            DEFAULT_CONFIG_PATH,
            DEFAULT_SCRIPT_PATH,
            DEFAULT_GATE_SCRIPT_PATH,
            PROJECT_ROOT / "config" / "gates" / "bidirectional_interaction_stratified_v1.json",
            PROJECT_ROOT / "config" / "gates" / "bidirectional_interaction_smoke3_v1.json",
            PROJECT_ROOT / "config" / "gates" / "bidirectional_interaction_smoke_lite_v1.json",
            Path(load_json(DEFAULT_CONFIG_PATH)["resources"]["instance_discriminator_config"]),
            Path(load_json(DEFAULT_CONFIG_PATH)["resources"]["trf_semantic_contract"]),
            PROJECT_ROOT / "code" / "common" / "contracts.py",
            PROJECT_ROOT / "code" / "common" / "io_utils.py",
            PROJECT_ROOT / "code" / "common" / "qwen_client.py",
            PROJECT_ROOT / "code" / "instance_discriminator" / "pipeline.py",
            PROJECT_ROOT / "code" / "second_layer" / "validator.py",
        ]
    )
    resolved: list[Path] = []
    for item in files:
        path = item if item.is_absolute() else PROJECT_ROOT / item
        if not path.is_file():
            raise InteractionError(f"Interaction implementation file is missing: {path}")
        if path not in resolved:
            resolved.append(path)
    return {
        path.resolve().relative_to(PROJECT_ROOT).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in resolved
    }


def collect_output_hashes(paths: InteractionPaths) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not paths.root.exists():
        return result
    for path in sorted(item for item in paths.root.rglob("*") if item.is_file()):
        if path == paths.manifest:
            continue
        result[path.relative_to(paths.root).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return result


def initialize_run(
    paths: InteractionPaths,
    *,
    run_id: str,
    base_run_id: str,
    compatibility: dict[str, Any],
    command: list[str],
    resume: bool,
) -> dict[str, Any]:
    implementation = implementation_hashes()
    if paths.root.exists():
        if not resume or not paths.manifest.is_file():
            raise InteractionError("Existing interaction run requires --resume")
        manifest = load_json(paths.manifest)
        if manifest.get("status") in {"completed", "failed"}:
            raise InteractionError("Completed/failed runs are immutable; use a new run-id")
        if manifest.get("compatibility") != compatibility:
            raise InteractionError("Interaction resume compatibility mismatch")
        if manifest.get("implementation") != implementation:
            raise InteractionError("Implementation changed; use a new run-id")
        if manifest.get("outputs") and manifest["outputs"] != collect_output_hashes(paths):
            raise InteractionError("Interaction output ledger changed before resume")
        manifest["commands"].append({"at": utc_now(), "argv": command})
        manifest.update({"status": "running", "completed_at": None, "error": None})
        atomic_write_json(paths.manifest, manifest)
        return manifest
    if resume:
        raise InteractionError("Cannot resume a run that does not exist")
    paths.root.mkdir(parents=True, exist_ok=False)
    for directory in (paths.source, paths.rounds, paths.context, paths.audit, paths.logs):
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
        "network": {branch: False for branch in BRANCHES},
        "compatibility": compatibility,
        "implementation": implementation,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "commands": [{"at": utc_now(), "argv": command}],
        "stages": {stage: "pending" for stage in STAGES},
        "base_validation": None,
        "source_snapshot": None,
        "rounds": [],
        "processes": {},
        "summary": None,
        "outputs": {},
        "error": None,
    }
    atomic_write_json(paths.manifest, manifest)
    return manifest


def update_manifest(paths: InteractionPaths, **updates: Any) -> dict[str, Any]:
    manifest = load_json(paths.manifest)
    manifest.update(updates)
    atomic_write_json(paths.manifest, manifest)
    return manifest


def update_stage(paths: InteractionPaths, stage: str, status: str) -> None:
    if stage not in STAGES:
        raise InteractionError(f"Unknown interaction stage: {stage}")
    manifest = load_json(paths.manifest)
    manifest["stages"][stage] = status
    atomic_write_json(paths.manifest, manifest)
