"""Read-only reconstruction and validation for completed second-layer runs."""

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

from common.io_utils import load_json, read_jsonl, resolve_project_path  # noqa: E402
from instance_discriminator.validator import validate as validate_exemplar  # noqa: E402
from second_layer.assembler import assemble_context_records, context_summary  # noqa: E402
from second_layer.common import (  # noqa: E402
    STAGES,
    SecondLayerError,
    assert_compatibility_unchanged,
    assert_snapshots_unchanged,
    child_run_ids,
    child_run_roots,
    collect_output_hashes,
    compatibility_payload,
    file_snapshot,
    implementation_hashes,
    load_second_layer_config,
    second_layer_run_paths,
    target_descriptor,
)
from trf.orchestration.validator import validate as validate_trf  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/second_layer.json")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise SecondLayerError(f"Validation mismatch: {label}")


def _child_states(roots: dict[str, Path]) -> dict[str, Any]:
    states: dict[str, Any] = {}
    for name, root in roots.items():
        manifest_path = root / "manifest.json"
        manifest = load_json(manifest_path)
        states[name] = {
            "run_root": str(root),
            "manifest": str(manifest_path),
            "manifest_sha256": file_snapshot(manifest_path)["sha256"],
            "status": manifest.get("status"),
            "network_called": bool(manifest.get("network_called")),
        }
    return states


def _provenance(
    paths: Any,
    roots: dict[str, Path],
    ids: dict[str, str],
) -> dict[str, Any]:
    return {
        "trf_run_id": ids["trf"],
        "exemplar_run_id": ids["exemplar"],
        "artifacts": {
            "shared_targets": file_snapshot(paths.shared / "targets.jsonl"),
            "shared_candidates": file_snapshot(paths.shared / "candidates.jsonl"),
            "trf_features": file_snapshot(
                roots["trf"] / "target" / "parsed" / "records.jsonl"
            ),
            "exemplar_selection": file_snapshot(
                roots["exemplar"] / "selected" / "records.jsonl"
            ),
        },
    }


def validate(config_path: Path, run_id: str) -> dict[str, Any]:
    config = load_second_layer_config(config_path)
    paths = second_layer_run_paths(config, run_id)
    manifest = load_json(paths.manifest)
    if manifest.get("status") != "completed":
        raise SecondLayerError("Only completed second-layer runs can pass validation")
    _equal(manifest.get("run_id"), run_id, "parent run ID")
    if set(manifest.get("stages", {})) != set(STAGES) or any(
        manifest["stages"][stage] != "completed" for stage in STAGES
    ):
        raise SecondLayerError("Every second-layer stage must be completed")
    _equal(manifest.get("implementation"), implementation_hashes(), "implementation")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, dict):
        raise SecondLayerError("Second-layer manifest has no compatibility contract")
    assert_compatibility_unchanged(compatibility)
    if Path(compatibility["config"]["path"]).resolve() != config_path.resolve():
        raise SecondLayerError("Validator was given a different second-layer config")
    if compatibility.get("exemplar_feature_context") != "absent":
        raise SecondLayerError("The v1 exemplar branch must have absent feature context")
    target = compatibility.get("target")
    if not isinstance(target, dict):
        raise SecondLayerError("Second-layer manifest has no target descriptor")
    rebuilt_target = target_descriptor(
        target.get("mode"),
        Path(target["input_path"]) if target.get("input_path") else None,
        target.get("limit"),
    )
    _equal(target, rebuilt_target, "target descriptor")
    _equal(
        compatibility,
        compatibility_payload(config_path, config, run_id, rebuilt_target),
        "compatibility contract",
    )
    if manifest.get("source_unchanged") is not True:
        raise SecondLayerError("Manifest does not attest immutable sources")

    shared_inputs = manifest.get("shared_inputs")
    assert_snapshots_unchanged(shared_inputs)
    roots = child_run_roots(config, run_id)
    ids = child_run_ids(run_id)
    _equal(compatibility.get("child_run_ids"), ids, "child run IDs")
    states = _child_states(roots)
    _equal(manifest.get("children"), states, "child manifests")
    if any(state["status"] != "completed" for state in states.values()):
        raise SecondLayerError("Both child runs must be completed")
    network = {
        "trf": states["trf"]["network_called"],
        "exemplar": states["exemplar"]["network_called"],
    }
    _equal(manifest.get("network"), network, "network audit")
    _equal(manifest.get("network_called"), any(network.values()), "network aggregate")

    trf_config_path = Path(config["children"]["trf_config"])
    exemplar_config_path = Path(config["children"]["instance_discriminator_config"])
    child_validation = {
        "schema_version": "second-layer-child-validation-v1",
        "trf": validate_trf(ids["trf"], trf_config_path),
        "exemplar": validate_exemplar(exemplar_config_path, ids["exemplar"]),
    }
    _equal(load_json(paths.validation), child_validation, "child validation")

    targets = read_jsonl(Path(shared_inputs["targets"]["path"]))
    trf_records = read_jsonl(
        roots["trf"] / "target" / "parsed" / "records.jsonl"
    )
    exemplar_records = read_jsonl(
        roots["exemplar"] / "selected" / "records.jsonl"
    )
    expected_context = assemble_context_records(
        targets,
        trf_records,
        exemplar_records,
        _provenance(paths, roots, ids),
    )
    actual_context = read_jsonl(paths.context / "records.jsonl")
    _equal(actual_context, expected_context, "assembled context")
    expected_summary = context_summary(expected_context)
    _equal(load_json(paths.context / "summary.json"), expected_summary, "context summary")
    _equal(manifest.get("summary"), expected_summary, "manifest summary")
    _equal(collect_output_hashes(paths), manifest.get("outputs"), "output SHA256 ledger")
    return {
        "run_id": run_id,
        "status": "valid",
        "targets": len(expected_context),
        "ready": sum(item["assembly_status"] == "ready" for item in expected_context),
        "trf_run_id": ids["trf"],
        "exemplar_run_id": ids["exemplar"],
        "exemplar_feature_context": "absent",
    }


def main() -> int:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    try:
        result = validate(config_path, args.run_id)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"ValidateSecondLayer failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
