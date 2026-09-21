"""Configuration, paths, manifests, and immutable snapshots for the second layer."""

from __future__ import annotations

import copy
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


PIPELINE_VERSION = "second-layer-parallel-v1"
STAGES = ("prepare_shared", "parallel_inference", "validate_children", "assemble_context")


class SecondLayerError(RuntimeError):
    """Raised when the parallel second-layer contract must fail closed."""


@dataclass(frozen=True)
class SecondLayerRunPaths:
    root: Path
    manifest: Path
    shared: Path
    context: Path
    logs: Path
    validation: Path


def load_second_layer_config(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if value.get("schema_version") != 1 or value.get("module") != "second_layer":
        raise SecondLayerError("Second-layer config must be a schema_version 1 module")
    if value.get("pipeline_version") != PIPELINE_VERSION:
        raise SecondLayerError("Unsupported second-layer pipeline_version")
    children = value.get("children")
    if not isinstance(children, dict) or set(children) != {
        "trf_config",
        "instance_discriminator_config",
    }:
        raise SecondLayerError("Second-layer children must define the two module configs")
    output_root = value.get("output", {}).get("runs_root")
    if not isinstance(output_root, str) or not output_root.strip():
        raise SecondLayerError("Second-layer output.runs_root must be a path")

    config = copy.deepcopy(value)
    for name, raw in children.items():
        if not isinstance(raw, str) or not raw.strip():
            raise SecondLayerError(f"children.{name} must be a path")
        resolved = resolve_project_path(PROJECT_ROOT, raw).resolve()
        if not resolved.is_file():
            raise SecondLayerError(f"Child config does not exist: {resolved}")
        config["children"][name] = str(resolved)
    return config


def second_layer_run_paths(config: dict[str, Any], run_id: str) -> SecondLayerRunPaths:
    root = (
        resolve_project_path(PROJECT_ROOT, config["output"]["runs_root"])
        / validate_run_id(run_id)
    ).resolve()
    return SecondLayerRunPaths(
        root=root,
        manifest=root / "manifest.json",
        shared=root / "shared",
        context=root / "context",
        logs=root / "logs",
        validation=root / "child-validation.json",
    )


def child_run_ids(run_id: str) -> dict[str, str]:
    validate_run_id(run_id)
    return {
        "trf": validate_run_id(f"{run_id}-trf"),
        "exemplar": validate_run_id(f"{run_id}-examples"),
    }


def child_run_roots(config: dict[str, Any], run_id: str) -> dict[str, Path]:
    ids = child_run_ids(run_id)
    trf_config = load_json(Path(config["children"]["trf_config"]))
    exemplar_config = load_json(
        Path(config["children"]["instance_discriminator_config"])
    )
    try:
        trf_runs_root = trf_config["output"]["runs_root"]
        exemplar_runs_root = exemplar_config["output"]["runs_root"]
    except (KeyError, TypeError) as error:
        raise SecondLayerError("A child config has no output.runs_root") from error
    return {
        "trf": (
            resolve_project_path(PROJECT_ROOT, trf_runs_root) / ids["trf"]
        ).resolve(),
        "exemplar": (
            resolve_project_path(PROJECT_ROOT, exemplar_runs_root) / ids["exemplar"]
        ).resolve(),
    }


def file_snapshot(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SecondLayerError(f"Required artifact does not exist: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def implementation_hashes() -> dict[str, dict[str, Any]]:
    files = sorted((PROJECT_ROOT / "code" / "second_layer").glob("*.py"))
    files.extend(
        [
            PROJECT_ROOT / "scripts" / "second_layer" / "run.ps1",
            PROJECT_ROOT / "config" / "second_layer.json",
            PROJECT_ROOT / "environment.yml",
            PROJECT_ROOT / "pyproject.toml",
        ]
    )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise SecondLayerError(f"Second-layer implementation file is missing: {missing[0]}")
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    }


def target_descriptor(
    mode: str,
    input_path: Path | None,
    limit: int | None,
) -> dict[str, Any]:
    if mode not in {"independent", "leave-one-out"}:
        raise SecondLayerError(f"Unsupported target mode: {mode}")
    if mode == "leave-one-out" and input_path is not None:
        raise SecondLayerError("An input file is not allowed in leave-one-out mode")
    if mode == "independent" and (input_path is None or not input_path.is_file()):
        raise SecondLayerError("Independent mode requires an existing input file")
    if limit is not None and limit < 1:
        raise SecondLayerError("--limit must be a positive integer")
    return {
        "mode": mode,
        "input_path": str(input_path.resolve()) if input_path else None,
        "input_sha256": sha256_file(input_path) if input_path else None,
        "limit": limit,
    }


def compatibility_payload(
    config_path: Path,
    config: dict[str, Any],
    run_id: str,
    target: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "config": file_snapshot(config_path),
        "child_configs": {
            "trf": file_snapshot(Path(config["children"]["trf_config"])),
            "exemplar": file_snapshot(
                Path(config["children"]["instance_discriminator_config"])
            ),
        },
        "child_run_ids": child_run_ids(run_id),
        "target": target,
        "exemplar_feature_context": "absent",
    }


def initialize_or_resume_run(
    paths: SecondLayerRunPaths,
    *,
    run_id: str,
    compatibility: dict[str, Any],
    command: list[str],
    resume: bool,
    child_roots: dict[str, Path],
) -> dict[str, Any]:
    current_implementation = implementation_hashes()
    if paths.root.exists():
        if not resume:
            raise SecondLayerError(f"Second-layer run-id already exists: {paths.root}")
        if not paths.manifest.is_file():
            raise SecondLayerError("Existing second-layer run has no manifest")
        manifest = load_json(paths.manifest)
        if manifest.get("status") == "completed":
            raise SecondLayerError("Completed second-layer runs are immutable")
        if manifest.get("status") == "failed":
            raise SecondLayerError("Failed second-layer runs are immutable; use a new run-id")
        if manifest.get("compatibility") != compatibility:
            raise SecondLayerError("Second-layer resume compatibility mismatch")
        if manifest.get("implementation") != current_implementation:
            raise SecondLayerError("Second-layer implementation changed; use a new run-id")
        manifest["commands"].append({"at": utc_now(), "argv": command})
        manifest.update({"status": "running", "completed_at": None, "error": None})
        atomic_write_json(paths.manifest, manifest)
        return manifest

    if resume:
        raise SecondLayerError("Cannot resume a second-layer run that does not exist")
    collisions = [root for root in child_roots.values() if root.exists()]
    if collisions:
        raise SecondLayerError(f"Derived child run already exists: {collisions[0]}")
    paths.root.mkdir(parents=True, exist_ok=False)
    paths.shared.mkdir()
    paths.context.mkdir()
    paths.logs.mkdir()
    manifest = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "run_id": run_id,
        "status": "running",
        "created_at": utc_now(),
        "completed_at": None,
        "network_called": False,
        "network": {"trf": False, "exemplar": False},
        "compatibility": compatibility,
        "implementation": current_implementation,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "commands": [{"at": utc_now(), "argv": command}],
        "stages": {stage: "pending" for stage in STAGES},
        "shared_inputs": None,
        "processes": {},
        "children": {},
        "source_unchanged": None,
        "outputs": {},
        "summary": None,
        "error": None,
    }
    atomic_write_json(paths.manifest, manifest)
    return manifest


def update_manifest(paths: SecondLayerRunPaths, **updates: Any) -> dict[str, Any]:
    manifest = load_json(paths.manifest)
    manifest.update(updates)
    atomic_write_json(paths.manifest, manifest)
    return manifest


def update_stage(paths: SecondLayerRunPaths, stage: str, status: str) -> None:
    if stage not in STAGES:
        raise SecondLayerError(f"Unknown second-layer stage: {stage}")
    manifest = load_json(paths.manifest)
    manifest["stages"][stage] = status
    atomic_write_json(paths.manifest, manifest)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(name)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            while block := reader.read(1024 * 1024):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def freeze_shared_inputs(
    paths: SecondLayerRunPaths,
    target_source: Path,
    candidate_source: Path,
) -> dict[str, Any]:
    targets = paths.shared / "targets.jsonl"
    candidates = paths.shared / "candidates.jsonl"
    if targets.exists() or candidates.exists():
        raise SecondLayerError("Shared input snapshots already exist")
    atomic_copy(target_source, targets)
    atomic_copy(candidate_source, candidates)
    return {"targets": file_snapshot(targets), "candidates": file_snapshot(candidates)}


def assert_snapshots_unchanged(snapshots: dict[str, Any]) -> None:
    if not isinstance(snapshots, dict) or set(snapshots) != {"targets", "candidates"}:
        raise SecondLayerError("Shared input snapshot contract is incomplete")
    current = {
        name: file_snapshot(Path(details["path"])) for name, details in snapshots.items()
    }
    if current != snapshots:
        raise SecondLayerError("A frozen second-layer input changed")


def assert_compatibility_unchanged(compatibility: dict[str, Any]) -> None:
    if file_snapshot(Path(compatibility["config"]["path"])) != compatibility["config"]:
        raise SecondLayerError("The second-layer config changed during the run")
    for name, expected in compatibility["child_configs"].items():
        if file_snapshot(Path(expected["path"])) != expected:
            raise SecondLayerError(f"The {name} child config changed during the run")
    target = compatibility["target"]
    if target.get("input_path"):
        current_hash = sha256_file(Path(target["input_path"]))
        if current_hash != target.get("input_sha256"):
            raise SecondLayerError("The independent target input changed during the run")


def collect_output_hashes(paths: SecondLayerRunPaths) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for path in sorted(paths.root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path != paths.manifest:
            outputs[path.relative_to(paths.root).as_posix()] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return outputs
