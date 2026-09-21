"""Read-only deterministic validator for JSON-Schema collaborative reflection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import load_json, read_jsonl, resolve_project_path  # noqa: E402
from second_layer.collaborative_reflection.runner import (  # noqa: E402
    _branch_states,
    _identity_contract,
    validate_base,
)
from second_layer.collaborative_reflection_json_schema.branch_validator import (  # noqa: E402
    validate_branch,
)
from second_layer.collaborative_reflection_json_schema.common import (  # noqa: E402
    BRANCHES,
    PROTOCOL_VERSION,
    STAGES,
    ReflectionError,
    assert_snapshot,
    branch_run_ids,
    collect_output_hashes,
    compatibility_payload,
    implementation_hashes,
    load_config,
    paths_for,
)
from second_layer.collaborative_reflection_json_schema.pipeline import (  # noqa: E402
    assemble_context_records,
    build_prompt,
    build_reflection_inputs,
    context_summary,
)
from second_layer.collaborative_reflection_json_schema.runner import _provenance  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/second_layer_reflection_json_schema.json")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ReflectionError(f"Validation mismatch: {label}")


def validate(config_path: Path, run_id: str) -> dict[str, Any]:
    config = load_config(config_path)
    paths = paths_for(config, run_id)
    manifest = load_json(paths.manifest)
    if manifest.get("status") != "completed":
        raise ReflectionError("Only completed JSON-Schema reflection runs can pass validation")
    _equal(manifest.get("protocol_version"), PROTOCOL_VERSION, "protocol version")
    _equal(manifest.get("run_id"), run_id, "reflection run ID")
    _equal(manifest.get("schedule_type"), "collaborative_reflection", "schedule type")
    _equal(manifest.get("base_schedule_type"), "concurrent", "base schedule type")
    if set(manifest.get("stages", {})) != set(STAGES) or any(
        manifest["stages"].get(stage) != "completed" for stage in STAGES
    ):
        raise ReflectionError("Every JSON-Schema reflection stage must be completed")
    _equal(manifest.get("implementation"), implementation_hashes(), "reflection implementation")

    base_manifest, base_manifest_path, base_validation, sources = validate_base(
        config, manifest.get("base_run_id")
    )
    _equal(manifest.get("base_validation"), base_validation, "base validator result")
    snapshot = load_json(paths.source / "snapshot.json")
    assert_snapshot(snapshot)
    _equal(manifest.get("source_snapshot"), snapshot, "source snapshot")
    inputs = build_reflection_inputs(
        read_jsonl(paths.source / "base-context.jsonl"),
        read_jsonl(paths.source / "candidate-records.jsonl"),
        read_jsonl(paths.source / "judgments.jsonl"),
        candidate_count=config["reflection"]["candidate_count"],
    )
    _equal(read_jsonl(paths.inputs), inputs, "reflection input reconstruction")
    compatibility = compatibility_payload(
        config_path=config_path,
        run_id=run_id,
        base_run_id=manifest["base_run_id"],
        base_manifest=base_manifest,
        base_manifest_path=base_manifest_path,
        base_sources=sources,
        identities=_identity_contract(inputs),
        limit=manifest["compatibility"]["target"]["limit"],
        config=config,
    )
    _equal(manifest.get("compatibility"), compatibility, "reflection compatibility")
    if Path(manifest["compatibility"]["config"]["path"]).resolve() != config_path.resolve():
        raise ReflectionError("Validator was given a different JSON-Schema reflection config")
    for branch in BRANCHES:
        expected_prompts = [
            build_prompt(
                item,
                branch,
                config["chat"]["max_prompt_characters"],
                config["chat"]["max_reason_characters"],
            )
            for item in inputs
        ]
        _equal(
            read_jsonl(paths.branch(branch) / "prompts" / "records.jsonl"),
            expected_prompts,
            f"{branch} prompts",
        )

    branches = _branch_states(paths)
    _equal(manifest.get("branches"), branches, "branch manifest states")
    if any(branches[name]["status"] != "completed" for name in BRANCHES):
        raise ReflectionError("Both JSON-Schema reflection branches must be completed")
    validation_results = {
        branch: validate_branch(
            config_path=config_path,
            parent_root=paths.root,
            run_id=branch_run_ids(run_id)[branch],
            branch=branch,
        )
        for branch in BRANCHES
    }
    _equal(load_json(paths.audit / "branch-validation.json"), validation_results, "branch validation ledger")

    processes = manifest.get("processes")
    if not isinstance(processes, dict) or set(processes) != set(BRANCHES):
        raise ReflectionError("JSON-Schema reflection process audit is incomplete")
    if any(processes[name].get("exit_code") != 0 for name in BRANCHES):
        raise ReflectionError("A JSON-Schema reflection process did not exit successfully")
    if not (
        processes["trf"]["started_at"] <= processes["exemplar"]["ended_at"]
        and processes["exemplar"]["started_at"] <= processes["trf"]["ended_at"]
    ):
        raise ReflectionError("JSON-Schema reflection branch process intervals do not overlap")
    network = {name: branches[name]["network_called"] for name in BRANCHES}
    _equal(manifest.get("network"), network, "branch network audit")
    _equal(manifest.get("network_called"), any(network.values()), "aggregate network audit")

    provenance = _provenance(paths, run_id, branches)
    expected_context, expected_interactions = assemble_context_records(
        inputs,
        read_jsonl(paths.trf / "parsed" / "records.jsonl"),
        read_jsonl(paths.exemplar / "parsed" / "judgments.jsonl"),
        read_jsonl(paths.exemplar / "selected" / "records.jsonl"),
        provenance,
    )
    actual_context = read_jsonl(paths.context / "records.jsonl")
    _equal(actual_context, expected_context, "assembled JSON-Schema reflection context")
    _equal(read_jsonl(paths.interaction / "records.jsonl"), expected_interactions, "interaction records")
    branch_metrics = {name: branches[name]["metrics"] for name in BRANCHES}
    expected_summary = context_summary(expected_context, branch_metrics)
    _equal(load_json(paths.context / "summary.json"), expected_summary, "reflection summary")
    _equal(manifest.get("summary"), expected_summary, "manifest summary")
    expected_manual = [
        {
            "dataset_id": item["dataset_id"],
            "record_id": item["record_id"],
            "idx": item["idx"],
            "interaction_status": item["interaction"]["status"],
            "conflicts": item["interaction"]["conflicts"],
            "review_reasons": item["interaction"]["review_reasons"],
            "unreviewed_added_trfs": item["interaction"]["unreviewed_added_trfs"],
        }
        for item in expected_context
        if item["interaction"]["status"] != "aligned"
    ]
    _equal(read_jsonl(paths.audit / "manual_review.jsonl"), expected_manual, "manual review audit")
    _equal(manifest.get("outputs"), collect_output_hashes(paths), "parent SHA256 ledger")
    return {
        "run_id": run_id,
        "status": "valid",
        "protocol_version": PROTOCOL_VERSION,
        "base_run_id": manifest["base_run_id"],
        "targets": len(expected_context),
        "ready": sum(item["assembly_status"] == "ready" for item in expected_context),
        "revised_judgments_per_target": config["reflection"]["candidate_count"],
        "final_skill_spans_generated": False,
        "interaction_statuses": expected_summary["interaction_statuses"],
    }


def main() -> int:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    try:
        print(json.dumps(validate(config_path, args.run_id), ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"ValidateJsonSchemaCollaborativeReflection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
