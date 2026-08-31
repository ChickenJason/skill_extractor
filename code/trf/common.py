"""Shared contracts, hashing, and manifest handling for the TRF pipeline."""

from __future__ import annotations

import importlib.metadata
import copy
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar


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


class TRFError(RuntimeError):
    """Raised when a fail-closed TRF contract is violated."""


@dataclass(frozen=True)
class SourceFiles:
    root: Path
    manifest: Path
    decisions: Path
    aggregation_audit: Path


@dataclass(frozen=True)
class TRFRunPaths:
    root: Path
    manifest: Path
    source: Path
    corpus: Path
    candidates: Path
    demonstrations: Path
    audit: Path


T = TypeVar("T")
STAGE_NAMES = ("build_corpus", "extract_candidates", "assign_pseudo_trfs")


def bind_self_annotation_run(
    config: dict[str, Any], self_annotator_run_id: str
) -> dict[str, Any]:
    """Bind a TRF config template to one immutable self-annotation run."""

    safe_id = validate_run_id(self_annotator_run_id)
    resolved = copy.deepcopy(config)
    source = resolved.get("source")
    if not isinstance(source, dict):
        raise TRFError("TRF config must define a source object")
    runs_root = source.get("runs_root")
    if not isinstance(runs_root, str) or not runs_root.strip():
        raise TRFError("TRF source must define a non-empty runs_root")
    root = (resolve_project_path(PROJECT_ROOT, runs_root) / safe_id).resolve()
    source["run_id"] = safe_id
    source["run_root"] = str(root)
    return resolved


def load_trf_config(
    path: Path, self_annotator_run_id: str | None = None
) -> dict[str, Any]:
    config = load_json(path)
    if config.get("schema_version") != 1:
        raise TRFError("config/trf.json schema_version must equal 1")
    if config.get("pipeline_version") != "trf-offline-v2":
        raise TRFError("Unsupported TRF pipeline_version")
    if self_annotator_run_id is None:
        source = config.get("source", {})
        if not isinstance(source.get("run_id"), str) or not isinstance(
            source.get("run_root"), str
        ):
            raise TRFError(
                "TRF config is an input template; provide self_annotator_run_id"
            )
        return config
    return bind_self_annotation_run(config, self_annotator_run_id)


def source_files(config: dict[str, Any]) -> SourceFiles:
    source = config["source"]
    root = resolve_project_path(PROJECT_ROOT, source["run_root"]).resolve()
    return SourceFiles(
        root=root,
        manifest=(root / source["manifest"]).resolve(),
        decisions=(root / source["decisions"]).resolve(),
        aggregation_audit=(root / source["aggregation_audit"]).resolve(),
    )


def run_paths(config: dict[str, Any], run_id: str) -> TRFRunPaths:
    safe_id = validate_run_id(run_id)
    root = (resolve_project_path(PROJECT_ROOT, config["output"]["runs_root"]) / safe_id).resolve()
    return TRFRunPaths(
        root=root,
        manifest=root / "manifest.json",
        source=root / "source",
        corpus=root / "corpus",
        candidates=root / "candidates",
        demonstrations=root / "demonstrations",
        audit=root / "audit",
    )


def snapshot_source(config: dict[str, Any]) -> dict[str, Any]:
    files = source_files(config)
    for path in (files.manifest, files.decisions, files.aggregation_audit):
        if not path.is_file():
            raise TRFError(f"Required source file does not exist: {path}")

    source_manifest = load_json(files.manifest)
    expected_id = config["source"]["run_id"]
    if source_manifest.get("run_id") != expected_id:
        raise TRFError(
            f"Source run id mismatch: expected {expected_id!r}, "
            f"got {source_manifest.get('run_id')!r}"
        )
    if source_manifest.get("status") != "completed":
        raise TRFError("Source self-annotation run is not completed")

    return {
        "run_id": expected_id,
        "root": str(files.root),
        "files": {
            "manifest": {
                "path": str(files.manifest),
                "sha256": sha256_file(files.manifest),
                "bytes": files.manifest.stat().st_size,
            },
            "decisions": {
                "path": str(files.decisions),
                "sha256": sha256_file(files.decisions),
                "bytes": files.decisions.stat().st_size,
            },
            "aggregation_audit": {
                "path": str(files.aggregation_audit),
                "sha256": sha256_file(files.aggregation_audit),
                "bytes": files.aggregation_audit.stat().st_size,
            },
        },
    }


def assert_source_unchanged(config: dict[str, Any], expected: dict[str, Any]) -> None:
    current = snapshot_source(config)
    if current != expected:
        raise TRFError("Source self-annotation files changed during the TRF run")


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    for distribution in ("numpy", "scikit-learn", "torch", "transformers"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def implementation_hashes() -> dict[str, dict[str, Any]]:
    files = sorted((PROJECT_ROOT / "code" / "trf").glob("*.py"), key=lambda path: path.name)
    files.extend(
        [
            PROJECT_ROOT / "scripts" / "run-trf-offline.ps1",
            PROJECT_ROOT / "environment.yml",
            PROJECT_ROOT / "pyproject.toml",
        ]
    )
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    }


def initialize_run(
    config_path: Path,
    config: dict[str, Any],
    run_id: str,
    command: list[str],
) -> TRFRunPaths:
    paths = run_paths(config, run_id)
    if paths.root.exists():
        raise TRFError(f"TRF run-id already exists; overwrite is forbidden: {paths.root}")

    initial_source = snapshot_source(config)
    paths.root.mkdir(parents=True, exist_ok=False)
    paths.source.mkdir()
    manifest = {
        "schema_version": 1,
        "pipeline_version": config["pipeline_version"],
        "run_id": run_id,
        "status": "running",
        "created_at": utc_now(),
        "completed_at": None,
        "network_policy": "model-download-explicit-only",
        "network_called": False,
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "source": initial_source,
        "source_unchanged": None,
        "dependencies": dependency_versions(),
        "implementation": implementation_hashes(),
        "model": {
            "name": config["model"]["name"],
            "revision": config["model"]["revision"],
            "weight_files": [],
        },
        "commands": [{"at": utc_now(), "argv": command}],
        "stages": {
            name: {"status": "pending", "started_at": None, "completed_at": None}
            for name in STAGE_NAMES
        },
        "outputs": {},
        "error": None,
    }
    atomic_write_json(paths.manifest, manifest)
    atomic_write_json(paths.source / "snapshot.json", initial_source)
    return paths


def load_run(config: dict[str, Any], run_id: str, command: list[str]) -> TRFRunPaths:
    paths = run_paths(config, run_id)
    if not paths.manifest.is_file():
        raise TRFError(f"TRF run does not exist or has no manifest: {paths.root}")
    manifest = load_json(paths.manifest)
    if manifest.get("run_id") != run_id:
        raise TRFError("TRF manifest run_id mismatch")
    if manifest.get("pipeline_version") != config["pipeline_version"]:
        raise TRFError("TRF manifest pipeline_version mismatch")
    if manifest.get("config", {}).get("sha256") != sha256_file(
        resolve_project_path(PROJECT_ROOT, manifest["config"]["path"])
    ):
        raise TRFError("The configuration used to create this run has changed")
    manifest["commands"].append({"at": utc_now(), "argv": command})
    atomic_write_json(paths.manifest, manifest)
    return paths


def read_manifest(paths: TRFRunPaths) -> dict[str, Any]:
    return load_json(paths.manifest)


def write_manifest(paths: TRFRunPaths, manifest: dict[str, Any]) -> None:
    atomic_write_json(paths.manifest, manifest)


def verify_run_source(config: dict[str, Any], paths: TRFRunPaths) -> None:
    expected = read_manifest(paths)["source"]
    assert_source_unchanged(config, expected)


def stage_completed(paths: TRFRunPaths, stage: str) -> bool:
    return read_manifest(paths)["stages"][stage]["status"] == "completed"


def execute_stage(
    config: dict[str, Any],
    paths: TRFRunPaths,
    stage: str,
    prerequisite: str | None,
    operation: Callable[[], T],
) -> T:
    if stage not in STAGE_NAMES:
        raise TRFError(f"Unknown TRF stage: {stage}")
    manifest = read_manifest(paths)
    if manifest["status"] == "failed":
        raise TRFError("Cannot continue a failed TRF run; choose a new run-id")
    if manifest["stages"][stage]["status"] != "pending":
        raise TRFError(f"Stage {stage} is not pending; overwrite is forbidden")
    if prerequisite and manifest["stages"][prerequisite]["status"] != "completed":
        raise TRFError(f"Stage {stage} requires completed stage {prerequisite}")

    verify_run_source(config, paths)
    manifest["stages"][stage]["status"] = "running"
    manifest["stages"][stage]["started_at"] = utc_now()
    write_manifest(paths, manifest)
    try:
        result = operation()
        verify_run_source(config, paths)
    except Exception as error:
        mark_failed(paths, stage, error)
        raise

    manifest = read_manifest(paths)
    manifest["stages"][stage]["status"] = "completed"
    manifest["stages"][stage]["completed_at"] = utc_now()
    if isinstance(result, dict):
        manifest["stages"][stage]["summary"] = result
    manifest["source_unchanged"] = True
    write_manifest(paths, manifest)
    return result


def mark_failed(paths: TRFRunPaths, stage: str | None, error: Exception) -> None:
    if not paths.manifest.is_file():
        return
    manifest = read_manifest(paths)
    manifest["status"] = "failed"
    manifest["completed_at"] = utc_now()
    manifest["source_unchanged"] = False if "Source" in str(error) else None
    manifest["error"] = {"type": type(error).__name__, "message": str(error)}
    if stage and stage in manifest["stages"]:
        manifest["stages"][stage]["status"] = "failed"
        manifest["stages"][stage]["completed_at"] = utc_now()
    write_manifest(paths, manifest)


def collect_output_hashes(paths: TRFRunPaths) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for path in sorted(paths.root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path == paths.manifest:
            continue
        relative = path.relative_to(paths.root).as_posix()
        outputs[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return outputs


def finalize_run(config: dict[str, Any], paths: TRFRunPaths) -> None:
    verify_run_source(config, paths)
    manifest = read_manifest(paths)
    incomplete = [
        name for name in STAGE_NAMES if manifest["stages"][name]["status"] != "completed"
    ]
    if incomplete:
        raise TRFError(f"Cannot finalize; incomplete stages: {', '.join(incomplete)}")
    manifest["outputs"] = collect_output_hashes(paths)
    manifest["source_unchanged"] = True
    manifest["status"] = "completed"
    manifest["completed_at"] = utc_now()
    manifest["error"] = None
    write_manifest(paths, manifest)


def parse_config_argument(value: str) -> Path:
    return resolve_project_path(PROJECT_ROOT, value).resolve()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
