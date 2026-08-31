"""Contracts and manifests for a complete offline-plus-target TRF rerun."""

from __future__ import annotations

import platform
import sys
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


class FullTRFError(RuntimeError):
    """Raised when a complete TRF rerun violates its fail-closed contract."""


@dataclass(frozen=True)
class FullTRFPaths:
    root: Path
    manifest: Path
    target_config: Path
    validation: Path


STAGES = ("offline_trf", "target_trf", "joint_validation")


def full_paths(runs_root: str | Path, run_id: str) -> FullTRFPaths:
    root = (
        resolve_project_path(PROJECT_ROOT, runs_root) / validate_run_id(run_id)
    ).resolve()
    return FullTRFPaths(
        root=root,
        manifest=root / "manifest.json",
        target_config=root / "target-config.generated.json",
        validation=root / "validation.json",
    )


def implementation_hashes() -> dict[str, dict[str, Any]]:
    files = sorted((PROJECT_ROOT / "code" / "trf_full").glob("*.py"))
    files.extend(
        [
            PROJECT_ROOT / "scripts" / "rerun-all-trf.ps1",
            PROJECT_ROOT / "environment.yml",
            PROJECT_ROOT / "pyproject.toml",
        ]
    )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FullTRFError(f"Complete TRF implementation file is missing: {missing[0]}")
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    }


def collect_output_hashes(paths: FullTRFPaths) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for path in sorted(paths.root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path != paths.manifest:
            outputs[path.relative_to(paths.root).as_posix()] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return outputs


def initialize_or_resume(
    paths: FullTRFPaths,
    run_id: str,
    compatibility: dict[str, Any],
    command: list[str],
    *,
    resume: bool,
) -> dict[str, Any]:
    if paths.root.exists():
        if not resume:
            raise FullTRFError(f"Complete TRF run-id already exists: {paths.root}")
        if not paths.manifest.is_file():
            raise FullTRFError("Existing complete TRF run has no manifest")
        manifest = load_json(paths.manifest)
        if manifest.get("status") == "completed":
            raise FullTRFError("Completed complete-TRF runs are immutable")
        if manifest.get("compatibility") != compatibility:
            raise FullTRFError("Complete TRF resume compatibility mismatch")
        if manifest.get("implementation") != implementation_hashes():
            raise FullTRFError("Complete TRF implementation changed; use a new run-id")
        manifest["status"] = "running"
        manifest["error"] = None
        manifest["commands"].append({"at": utc_now(), "argv": command})
        atomic_write_json(paths.manifest, manifest)
        return manifest

    if resume:
        raise FullTRFError("Cannot resume a complete TRF run that does not exist")
    paths.root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "pipeline_version": "trf-complete-v2",
        "run_id": run_id,
        "status": "running",
        "created_at": utc_now(),
        "completed_at": None,
        "network_called": False,
        "network": {"offline_model": False, "target_qwen": False},
        "compatibility": compatibility,
        "implementation": implementation_hashes(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "commands": [{"at": utc_now(), "argv": command}],
        "stages": {stage: "pending" for stage in STAGES},
        "children": {},
        "outputs": {},
        "summary": None,
        "error": None,
    }
    atomic_write_json(paths.manifest, manifest)
    return manifest


def update_manifest(paths: FullTRFPaths, **updates: Any) -> dict[str, Any]:
    manifest = load_json(paths.manifest)
    manifest.update(updates)
    atomic_write_json(paths.manifest, manifest)
    return manifest


def update_stage(paths: FullTRFPaths, stage: str, status: str) -> None:
    if stage not in STAGES:
        raise FullTRFError(f"Unknown complete TRF stage: {stage}")
    manifest = load_json(paths.manifest)
    manifest["stages"][stage] = status
    atomic_write_json(paths.manifest, manifest)
