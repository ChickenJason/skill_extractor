"""Read-only deterministic validator for a completed stratified reflection gate."""

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
from second_layer.collaborative_reflection.runner import _branch_states, validate_base  # noqa: E402
from second_layer.collaborative_reflection_json_schema.branch_validator import validate_branch  # noqa: E402
from second_layer.collaborative_reflection_json_schema.common import branch_run_ids, load_config  # noqa: E402
from second_layer.collaborative_reflection_json_schema.pipeline import (  # noqa: E402
    assemble_context_records,
    build_prompt,
    context_summary,
)
from second_layer.collaborative_reflection_stratified_gate.common import (  # noqa: E402
    BRANCHES,
    GATE_PROTOCOL_VERSION,
    GATE_STAGES,
    ReflectionError,
    SCHEDULE_TYPE,
    UNDERLYING_PROTOCOL_VERSION,
    assert_snapshot,
    collect_output_hashes,
    compatibility_payload,
    implementation_hashes,
    paths_for,
)
from second_layer.collaborative_reflection_stratified_gate.evaluator import evaluate_gate  # noqa: E402
from second_layer.collaborative_reflection_stratified_gate.runner import (  # noqa: E402
    _gate_source_records,
    _provenance,
)
from second_layer.collaborative_reflection_stratified_gate.selection import (  # noqa: E402
    identity_contract,
    load_gate_spec,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/second_layer_reflection_json_schema.json")
    parser.add_argument("--gate-spec", default="config/gates/collaborative_reflection_stratified12_v1.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--require-gate-pass", action="store_true")
    return parser.parse_args(argv)


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ReflectionError(f"Validation mismatch: {label}")


def validate(config_path: Path, gate_spec_path: Path, run_id: str) -> dict[str, Any]:
    config = load_config(config_path)
    gate_spec = load_gate_spec(gate_spec_path)
    paths = paths_for(config, run_id)
    manifest = load_json(paths.manifest)
    if manifest.get("status") != "completed":
        raise ReflectionError("Only completed stratified-gate runs can pass validation")
    _equal(manifest.get("run_id"), run_id, "run ID")
    _equal(manifest.get("gate_protocol_version"), GATE_PROTOCOL_VERSION, "gate protocol")
    _equal(manifest.get("underlying_protocol"), UNDERLYING_PROTOCOL_VERSION, "underlying protocol")
    _equal(manifest.get("schedule_type"), SCHEDULE_TYPE, "schedule type")
    _equal(manifest.get("gate_id"), gate_spec["gate_id"], "gate ID")
    if set(manifest.get("stages", {})) != set(GATE_STAGES) or any(
        manifest["stages"].get(stage) != "completed" for stage in GATE_STAGES
    ):
        raise ReflectionError("Every stratified-gate stage must be completed")
    _equal(manifest.get("implementation"), implementation_hashes(), "implementation")

    base_run_id = manifest.get("base_run_id")
    if base_run_id != gate_spec["base_concurrent_run_id"]:
        raise ReflectionError("Manifest base run does not match gate spec")
    base_manifest, base_manifest_path, base_validation, sources = validate_base(config, base_run_id)
    _equal(manifest.get("base_validation"), base_validation, "base validation")
    contexts, candidates, judgments, inputs, coverage = _gate_source_records(
        sources, gate_spec, config["reflection"]["candidate_count"]
    )

    snapshot = load_json(paths.source / "snapshot.json")
    assert_snapshot(snapshot)
    _equal(manifest.get("source_snapshot"), snapshot, "source snapshot")
    _equal(load_json(paths.source / "gate-spec.json"), gate_spec, "frozen gate spec")
    _equal(read_jsonl(paths.source / "base-context.jsonl"), contexts, "frozen contexts")
    _equal(read_jsonl(paths.source / "candidate-records.jsonl"), candidates, "frozen candidates")
    _equal(read_jsonl(paths.source / "judgments.jsonl"), judgments, "frozen judgments")
    _equal(read_jsonl(paths.inputs), inputs, "reflection inputs")

    ids = branch_run_ids(run_id)
    compatibility = compatibility_payload(
        config_path=config_path,
        gate_spec_path=gate_spec_path,
        run_id=run_id,
        base_run_id=base_run_id,
        base_manifest=base_manifest,
        base_manifest_path=base_manifest_path,
        base_sources=sources,
        identities=identity_contract(inputs),
        coverage=coverage,
        gate_spec=gate_spec,
        config=config,
        branch_run_ids=ids,
    )
    _equal(manifest.get("compatibility"), compatibility, "compatibility")
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
    _equal(manifest.get("branches"), branches, "branch states")
    if any(branches[name]["status"] != "completed" for name in BRANCHES):
        raise ReflectionError("Both stratified-gate reflection branches must be completed")
    branch_validation = {
        branch: validate_branch(
            config_path=config_path,
            parent_root=paths.root,
            run_id=ids[branch],
            branch=branch,
        )
        for branch in BRANCHES
    }
    _equal(load_json(paths.audit / "branch-validation.json"), branch_validation, "branch validation")

    processes = manifest.get("processes")
    if not isinstance(processes, dict) or set(processes) != set(BRANCHES):
        raise ReflectionError("Stratified-gate process audit is incomplete")
    if any(processes[name].get("exit_code") != 0 for name in BRANCHES):
        raise ReflectionError("A stratified-gate branch process exited unsuccessfully")
    if not (
        processes["trf"]["started_at"] <= processes["exemplar"]["ended_at"]
        and processes["exemplar"]["started_at"] <= processes["trf"]["ended_at"]
    ):
        raise ReflectionError("Stratified-gate branch process intervals do not overlap")
    network = {name: branches[name]["network_called"] for name in BRANCHES}
    _equal(manifest.get("network"), network, "network audit")
    _equal(manifest.get("network_called"), any(network.values()), "aggregate network audit")

    provenance = _provenance(paths, run_id, branches, gate_spec)
    expected_context, expected_interactions = assemble_context_records(
        inputs,
        read_jsonl(paths.trf / "parsed" / "records.jsonl"),
        read_jsonl(paths.exemplar / "parsed" / "judgments.jsonl"),
        read_jsonl(paths.exemplar / "selected" / "records.jsonl"),
        provenance,
    )
    actual_context = read_jsonl(paths.context / "records.jsonl")
    _equal(actual_context, expected_context, "assembled context")
    _equal(read_jsonl(paths.interaction / "records.jsonl"), expected_interactions, "interactions")
    metrics = {name: branches[name]["metrics"] for name in BRANCHES}
    expected_summary = context_summary(expected_context, metrics)
    _equal(load_json(paths.context / "summary.json"), expected_summary, "summary")
    _equal(manifest.get("summary"), expected_summary, "manifest summary")

    expected_regular_manual = [
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
    _equal(read_jsonl(paths.audit / "manual_review.jsonl"), expected_regular_manual, "manual review")
    evaluation, gate_manual = evaluate_gate(
        spec=gate_spec,
        inputs=inputs,
        contexts=expected_context,
        trf_raw=read_jsonl(paths.trf / "raw" / "responses.jsonl"),
        exemplar_raw=read_jsonl(paths.exemplar / "raw" / "responses.jsonl"),
        branch_metrics=metrics,
        processes=processes,
        coverage=coverage,
    )
    _equal(load_json(paths.audit / "gate-evaluation.json"), evaluation, "gate evaluation")
    _equal(read_jsonl(paths.audit / "gate-manual-review.jsonl"), gate_manual, "gate manual review")
    _equal(manifest.get("gate_evaluation"), evaluation, "manifest gate evaluation")
    _equal(manifest.get("gate_status"), evaluation["gate_status"], "manifest gate status")
    _equal(manifest.get("outputs"), collect_output_hashes(paths), "output SHA256 ledger")
    return {
        "run_id": run_id,
        "status": "valid",
        "gate_status": evaluation["gate_status"],
        "gate_id": gate_spec["gate_id"],
        "targets": len(expected_context),
        "ready": sum(item["assembly_status"] == "ready" for item in expected_context),
        "keep_anchors_passed": evaluation["anchors"]["passed_keep"],
        "drop_anchors_passed": evaluation["anchors"]["passed_drop"],
        "wrapper_repairs": evaluation["request_audit"]["wrapper_repairs"],
        "total_repairs": evaluation["request_audit"]["total_repairs"],
        "model_calls": evaluation["request_audit"]["model_calls"],
        "final_skill_spans_generated": False,
    }


def main() -> int:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    gate_spec_path = resolve_project_path(PROJECT_ROOT, args.gate_spec).resolve()
    try:
        result = validate(config_path, gate_spec_path, args.run_id)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        if args.require_gate_pass and result["gate_status"] != "passed":
            print("Stratified reflection run is valid but the experimental gate failed", file=sys.stderr)
            return 2
        return 0
    except Exception as error:
        print(f"ValidateStratifiedReflectionGate failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
