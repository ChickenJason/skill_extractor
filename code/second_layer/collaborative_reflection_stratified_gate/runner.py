"""Run a hash-locked stratified gate over JSON-Schema collaborative reflection."""

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

from common.io_utils import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    read_jsonl,
    resolve_project_path,
    utc_now,
)
from second_layer.collaborative_reflection.runner import (  # noqa: E402
    _branch_states,
    _freeze as freeze_base,
    _remove_unpublished_context,
    validate_base,
)
from second_layer.collaborative_reflection_json_schema.branch_validator import validate_branch  # noqa: E402
from second_layer.collaborative_reflection_json_schema.common import (  # noqa: E402
    branch_run_ids,
    load_config,
)
from second_layer.collaborative_reflection_json_schema.pipeline import (  # noqa: E402
    assemble_context_records,
    context_summary,
)
from second_layer.collaborative_reflection_json_schema.runner import (  # noqa: E402
    _prepare,
    _provenance as underlying_provenance,
    _run_parallel,
)
from second_layer.collaborative_reflection_stratified_gate.common import (  # noqa: E402
    BRANCHES,
    GATE_PROTOCOL_VERSION,
    ReflectionError,
    UNDERLYING_PROTOCOL_VERSION,
    assert_snapshot,
    collect_output_hashes,
    compatibility_payload,
    file_snapshot,
    initialize_run,
    paths_for,
    update_manifest,
    update_stage,
)
from second_layer.collaborative_reflection_stratified_gate.evaluator import evaluate_gate  # noqa: E402
from second_layer.collaborative_reflection_stratified_gate.selection import (  # noqa: E402
    identity_contract,
    load_gate_spec,
    select_gate_sources,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/second_layer_reflection_json_schema.json")
    parser.add_argument("--gate-spec", default="config/gates/collaborative_reflection_stratified12_v1.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-concurrent-run-id", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.retry_failed and not args.resume:
        raise ReflectionError("--retry-failed requires --resume")
    if not args.prepare_only and not args.allow_network:
        raise ReflectionError("Stratified-gate inference requires --allow-network")


def _freeze_with_gate(
    paths: Any, contexts: list[dict[str, Any]], candidates: list[dict[str, Any]],
    judgments: list[dict[str, Any]], gate_spec: dict[str, Any],
) -> dict[str, Any]:
    snapshot = freeze_base(paths, contexts, candidates, judgments)
    gate_copy = paths.source / "gate-spec.json"
    atomic_write_json(gate_copy, gate_spec)
    snapshot["gate_spec"] = file_snapshot(gate_copy)
    atomic_write_json(paths.source / "snapshot.json", snapshot)
    return snapshot


def _provenance(
    paths: Any, run_id: str, branches: dict[str, Any], gate_spec: dict[str, Any],
) -> dict[str, Any]:
    value = underlying_provenance(paths, run_id, branches)
    value.update(
        {
            "gate_protocol_version": GATE_PROTOCOL_VERSION,
            "underlying_protocol": UNDERLYING_PROTOCOL_VERSION,
            "gate_id": gate_spec["gate_id"],
            "schedule_type": "collaborative_reflection_stratified_gate",
        }
    )
    value["artifacts"]["gate_spec"] = file_snapshot(paths.source / "gate-spec.json")
    return value


def _gate_source_records(
    sources: dict[str, Path], gate_spec: dict[str, Any], candidate_count: int,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], dict[str, int],
]:
    return select_gate_sources(
        read_jsonl(sources["base_context"]),
        read_jsonl(sources["candidates"]),
        read_jsonl(sources["judgments"]),
        gate_spec,
        candidate_count=candidate_count,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    gate_spec_path = resolve_project_path(PROJECT_ROOT, args.gate_spec).resolve()
    config = load_config(config_path)
    gate_spec = load_gate_spec(gate_spec_path)
    if gate_spec["base_concurrent_run_id"] != args.base_concurrent_run_id:
        raise ReflectionError("BaseConcurrentRunId does not match the gate spec")

    # Fail closed before creating a run directory.
    base_manifest, base_manifest_path, base_validation, sources = validate_base(
        config, args.base_concurrent_run_id
    )
    contexts, candidates, judgments, preview_inputs, coverage = _gate_source_records(
        sources, gate_spec, config["reflection"]["candidate_count"]
    )
    paths = paths_for(config, args.run_id)
    ids = branch_run_ids(args.run_id)
    compatibility = compatibility_payload(
        config_path=config_path,
        gate_spec_path=gate_spec_path,
        run_id=args.run_id,
        base_run_id=args.base_concurrent_run_id,
        base_manifest=base_manifest,
        base_manifest_path=base_manifest_path,
        base_sources=sources,
        identities=identity_contract(preview_inputs),
        coverage=coverage,
        gate_spec=gate_spec,
        config=config,
        branch_run_ids=ids,
    )
    manifest = initialize_run(
        paths,
        run_id=args.run_id,
        base_run_id=args.base_concurrent_run_id,
        compatibility=compatibility,
        command=sys.argv,
        resume=args.resume,
    )
    try:
        update_stage(paths, "validate_base", "completed")
        update_manifest(paths, base_validation=base_validation)
        update_stage(paths, "select_gate", "completed")
        if manifest.get("source_snapshot") is None:
            snapshot = _freeze_with_gate(paths, contexts, candidates, judgments, gate_spec)
            update_manifest(paths, source_snapshot=snapshot)
        else:
            snapshot = load_json(paths.source / "snapshot.json")
            assert_snapshot(snapshot)
            if snapshot != manifest["source_snapshot"]:
                raise ReflectionError("Manifest source snapshot mismatch")
            if load_json(paths.source / "gate-spec.json") != gate_spec:
                raise ReflectionError("Frozen gate spec changed")
        update_stage(paths, "freeze_base", "completed")

        inputs = _prepare(paths, config)
        if identity_contract(inputs) != compatibility["target"]["identities"]:
            raise ReflectionError("Prepared gate identities changed")
        update_stage(paths, "prepare_reflection", "completed")
        if args.prepare_only:
            update_manifest(
                paths,
                status="partial",
                gate_status="not_evaluated",
                completed_at=utc_now(),
                error=None,
            )
            update_manifest(paths, outputs=collect_output_hashes(paths))
            return load_json(paths.manifest)

        states = _run_parallel(
            paths=paths,
            config_path=config_path,
            run_id=args.run_id,
            python=sys.executable,
            allow_network=args.allow_network,
            resume=args.resume,
            retry_failed=args.retry_failed,
        )
        update_manifest(paths, processes=states)
        if any(state["exit_code"] != 0 for state in states.values()):
            update_stage(paths, "reflect_parallel", "failed")
            _remove_unpublished_context(paths)
            raise ReflectionError("At least one stratified-gate reflector process failed")
        branches = _branch_states(paths)
        network = {name: bool(branches[name].get("network_called")) for name in BRANCHES}
        update_manifest(paths, branches=branches, network=network, network_called=any(network.values()))
        if any(branches[name]["status"] != "completed" for name in BRANCHES):
            update_stage(paths, "reflect_parallel", "partial")
            _remove_unpublished_context(paths)
            update_manifest(
                paths,
                status="partial",
                gate_status="not_evaluated",
                completed_at=utc_now(),
                error="At least one reflection branch is partial",
                outputs=collect_output_hashes(paths),
            )
            return load_json(paths.manifest)
        update_stage(paths, "reflect_parallel", "completed")

        validations = {
            branch: validate_branch(
                config_path=config_path,
                parent_root=paths.root,
                run_id=ids[branch],
                branch=branch,
            )
            for branch in BRANCHES
        }
        atomic_write_json(paths.audit / "branch-validation.json", validations)
        update_stage(paths, "validate_reflections", "completed")

        branches = _branch_states(paths)
        provenance = _provenance(paths, args.run_id, branches, gate_spec)
        contexts_out, interactions = assemble_context_records(
            inputs,
            read_jsonl(paths.trf / "parsed" / "records.jsonl"),
            read_jsonl(paths.exemplar / "parsed" / "judgments.jsonl"),
            read_jsonl(paths.exemplar / "selected" / "records.jsonl"),
            provenance,
        )
        atomic_write_jsonl(paths.interaction / "records.jsonl", interactions)
        atomic_write_jsonl(paths.context / "records.jsonl", contexts_out)
        update_stage(paths, "assemble_context", "completed")

        branch_metrics = {name: branches[name]["metrics"] for name in BRANCHES}
        gate_evaluation, gate_manual = evaluate_gate(
            spec=gate_spec,
            inputs=inputs,
            contexts=contexts_out,
            trf_raw=read_jsonl(paths.trf / "raw" / "responses.jsonl"),
            exemplar_raw=read_jsonl(paths.exemplar / "raw" / "responses.jsonl"),
            branch_metrics=branch_metrics,
            processes=states,
            coverage=coverage,
        )
        atomic_write_json(paths.audit / "gate-evaluation.json", gate_evaluation)
        atomic_write_jsonl(paths.audit / "gate-manual-review.jsonl", gate_manual)
        update_stage(paths, "evaluate_gate", "completed")

        summary = context_summary(contexts_out, branch_metrics)
        atomic_write_json(paths.context / "summary.json", summary)
        regular_manual = [
            {
                "dataset_id": item["dataset_id"],
                "record_id": item["record_id"],
                "idx": item["idx"],
                "interaction_status": item["interaction"]["status"],
                "conflicts": item["interaction"]["conflicts"],
                "review_reasons": item["interaction"]["review_reasons"],
                "unreviewed_added_trfs": item["interaction"]["unreviewed_added_trfs"],
            }
            for item in contexts_out
            if item["interaction"]["status"] != "aligned"
        ]
        atomic_write_jsonl(paths.audit / "manual_review.jsonl", regular_manual)
        update_stage(paths, "diagnostics", "completed")
        update_manifest(
            paths,
            status="completed",
            gate_status=gate_evaluation["gate_status"],
            completed_at=utc_now(),
            branches=branches,
            summary=summary,
            gate_evaluation=gate_evaluation,
            error=None,
        )
        update_manifest(paths, outputs=collect_output_hashes(paths))
        return load_json(paths.manifest)
    except Exception as error:
        _remove_unpublished_context(paths)
        update_manifest(
            paths,
            status="failed",
            gate_status="not_evaluated",
            completed_at=utc_now(),
            error=str(error),
            outputs=collect_output_hashes(paths),
        )
        raise


def main() -> int:
    args = parse_args()
    try:
        manifest = run(args)
        print(
            json.dumps(
                {
                    "run_id": manifest["run_id"],
                    "status": manifest["status"],
                    "gate_status": manifest["gate_status"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as error:
        print(f"RunStratifiedReflectionGate failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
