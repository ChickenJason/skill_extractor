"""Read-only joint validation for a completed full TRF rerun."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import load_json, resolve_project_path, sha256_file  # noqa: E402
from trf.ValidateTRFRun import validate_completed_run  # noqa: E402
from trf.common import load_trf_config, run_paths as offline_run_paths  # noqa: E402
from trf_full.common import (  # noqa: E402
    FullTRFError,
    STAGES,
    collect_output_hashes,
    full_paths,
    implementation_hashes,
)
from trf_target.ValidateTargetTRFRun import validate as validate_target  # noqa: E402
from trf_target.common import load_target_config, target_run_paths  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--full-runs-root", default="output/trf_full_runs")
    return parser.parse_args(argv)


def validate(run_id: str, runs_root: str | Path) -> dict[str, Any]:
    paths = full_paths(runs_root, run_id)
    if not paths.manifest.is_file():
        raise FullTRFError(f"Complete TRF manifest does not exist: {paths.manifest}")
    manifest = load_json(paths.manifest)
    if manifest.get("status") != "completed":
        raise FullTRFError("Only completed complete-TRF runs can pass validation")
    stages = manifest.get("stages", {})
    if set(stages) != set(STAGES) or any(stages[stage] != "completed" for stage in STAGES):
        raise FullTRFError("Every complete TRF stage must be completed")
    if manifest.get("implementation") != implementation_hashes():
        raise FullTRFError("Complete TRF implementation hashes changed")
    if collect_output_hashes(paths) != manifest.get("outputs"):
        raise FullTRFError("Complete TRF output hash ledger mismatch")

    compatibility = manifest["compatibility"]
    offline_config_path = resolve_project_path(
        PROJECT_ROOT, compatibility["offline_config"]["path"]
    ).resolve()
    template_path = resolve_project_path(
        PROJECT_ROOT, compatibility["target_config_template"]["path"]
    ).resolve()
    if sha256_file(offline_config_path) != compatibility["offline_config"]["sha256"]:
        raise FullTRFError("Offline config changed after the complete rerun")
    if sha256_file(template_path) != compatibility["target_config_template"]["sha256"]:
        raise FullTRFError("Target config template changed after the complete rerun")
    input_details = compatibility["input"]
    if input_details["path"]:
        input_path = Path(input_details["path"])
        if not input_path.is_file() or sha256_file(input_path) != input_details["sha256"]:
            raise FullTRFError("Independent target input changed after the complete rerun")

    self_annotation = compatibility.get("self_annotation", {})
    self_annotator_run_id = self_annotation.get("run_id")
    if not isinstance(self_annotator_run_id, str):
        raise FullTRFError("Complete TRF manifest has no self-annotation run id")
    offline_config = load_trf_config(
        offline_config_path, self_annotator_run_id
    )
    if offline_config["source"]["run_root"] != self_annotation.get("run_root"):
        raise FullTRFError("Self-annotation source path differs from the run id binding")
    target_config = load_target_config(paths.target_config)
    offline_run_id = compatibility["offline_run_id"]
    target_run_id = compatibility["target_run_id"]
    if target_config["source_trf"]["run_id"] != offline_run_id:
        raise FullTRFError("Generated target config does not reference the offline child")
    offline_paths = offline_run_paths(offline_config, offline_run_id)
    target_paths = target_run_paths(target_config, target_run_id)

    expected_children = {
        "offline": {
            "root": str(offline_paths.root),
            "manifest": str(offline_paths.manifest),
            "manifest_sha256": sha256_file(offline_paths.manifest),
            "status": load_json(offline_paths.manifest)["status"],
        },
        "target": {
            "root": str(target_paths.root),
            "manifest": str(target_paths.manifest),
            "manifest_sha256": sha256_file(target_paths.manifest),
            "status": load_json(target_paths.manifest)["status"],
        },
    }
    if manifest.get("children") != expected_children:
        raise FullTRFError("Child manifest hashes or statuses changed")
    expected_network = {
        "offline_model": bool(load_json(offline_paths.manifest).get("network_called")),
        "target_qwen": bool(load_json(target_paths.manifest).get("network_called")),
    }
    if manifest.get("network") != expected_network:
        raise FullTRFError("Complete TRF network audit mismatch")
    if manifest.get("network_called") != any(expected_network.values()):
        raise FullTRFError("Complete TRF aggregate network flag mismatch")

    offline_summary = validate_completed_run(offline_config, offline_run_id)
    target_summary = validate_target(paths.target_config, target_run_id)
    expected_summary = {
        "schema_version": "trf-complete-summary-v1",
        "status": "validated",
        "offline": offline_summary,
        "target": target_summary,
    }
    if load_json(paths.validation) != expected_summary:
        raise FullTRFError("Joint validation summary file mismatch")
    if manifest.get("summary") != expected_summary:
        raise FullTRFError("Joint validation manifest summary mismatch")
    return {
        "run_id": run_id,
        "status": "valid",
        "offline_run_id": offline_run_id,
        "target_run_id": target_run_id,
        "mode": compatibility["mode"],
        "targets": target_summary["targets"],
        "semantic_acceptance": "pending_manual_review",
    }


def main() -> int:
    args = parse_args()
    try:
        result = validate(args.run_id, args.full_runs_root)
    except Exception as error:
        print(f"ValidateFullTRFRun failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
