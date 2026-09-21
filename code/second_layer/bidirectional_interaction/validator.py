"""Read-only reconstruction and validation for completed interaction runs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.contracts import record_identity  # noqa: E402
from common.io_utils import load_json, read_jsonl, resolve_project_path  # noqa: E402
from second_layer.bidirectional_interaction.branch_validator import validate_branch  # noqa: E402
from second_layer.bidirectional_interaction.common import (  # noqa: E402
    InteractionError,
    assert_snapshot,
    collect_output_hashes,
    file_snapshot,
    implementation_hashes,
    load_config,
    paths_for,
)
from second_layer.bidirectional_interaction.contracts import rebuild_inputs  # noqa: E402
from second_layer.bidirectional_interaction.gate import evaluate_gate, load_gate_spec  # noqa: E402
from second_layer.bidirectional_interaction.pipeline import (  # noqa: E402
    active_branches,
    apply_round,
    assemble_context,
    context_summary,
    detect_conflicts,
    effective_signature,
)
from second_layer.bidirectional_interaction.prompts import build_prompt  # noqa: E402
from second_layer.validator import validate as validate_second_layer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/second_layer_bidirectional_interaction.json")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise InteractionError(f"Validation mismatch: {label}")


def _round_input(record: dict[str, Any], state: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    value = copy.deepcopy(record)
    value["previous_r"] = copy.deepcopy(state[0])
    value["previous_h"] = copy.deepcopy(state[1])
    return value


def _metrics(manifests: list[dict[str, Any]]) -> dict[str, int]:
    names = ("model_responses", "repair_responses", "failed_records", "prompt_tokens", "completion_tokens")
    return {name: sum(int(item.get("metrics", {}).get(name, 0)) for item in manifests) for name in names}


def validate(config_path: Path, run_id: str) -> dict[str, Any]:
    config = load_config(config_path)
    paths = paths_for(config, run_id)
    manifest = load_json(paths.manifest)
    if manifest.get("status") != "completed":
        raise InteractionError("Only completed interaction runs can validate")
    _equal(manifest.get("run_id"), run_id, "run ID")
    _equal(manifest.get("implementation"), implementation_hashes(), "implementation ledger")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, dict):
        raise InteractionError("Interaction manifest has no compatibility contract")
    _equal(Path(compatibility["config"]["path"]).resolve(), config_path.resolve(), "config path")
    for item in (
        compatibility["config"],
        compatibility["base"]["manifest"],
        compatibility["hard_gate_config"],
        *compatibility["base"]["artifacts"].values(),
    ):
        assert_snapshot(item)
    if "semantic_gate" in compatibility:
        assert_snapshot(compatibility["semantic_gate"])
    base_result = validate_second_layer(
        Path(config["base"]["concurrent_config"]), manifest["base_run_id"]
    )
    _equal(manifest.get("base_validation"), base_result, "base validation")
    snapshot = load_json(paths.source / "snapshot.json")
    _equal(snapshot, manifest.get("source_snapshot"), "source snapshot")
    for group in ("originals", "frozen"):
        for item in snapshot[group].values():
            assert_snapshot(item)

    instance_config = load_json(Path(config["resources"]["instance_discriminator_config"]))
    gate = instance_config["gate"]
    rebuilt = rebuild_inputs(
        read_jsonl(paths.source / "base-context.jsonl"),
        read_jsonl(paths.source / "candidate-records.jsonl"),
        read_jsonl(paths.source / "judgments.jsonl"),
        gate,
    )
    _equal(rebuilt, read_jsonl(paths.inputs), "R0/H0 reconstruction")
    _equal(
        [
            {key: item[key] for key in ("dataset_id", "record_id", "source_sha256", "idx", "sentence")}
            for item in rebuilt
        ],
        compatibility["target"]["identities"],
        "target identities",
    )
    mode = compatibility["interaction"]["mode"]
    max_rounds = compatibility["interaction"]["max_rounds"]
    states = {
        record_identity(item, "interaction input"): (copy.deepcopy(item["r0"]), copy.deepcopy(item["h0"]))
        for item in rebuilt
    }
    histories: dict[tuple[str, str], list[dict[str, Any]]] = {
        record_identity(item, "interaction input"): [] for item in rebuilt
    }
    active = list(rebuilt)
    branch_manifests: list[dict[str, Any]] = []
    if mode == "none":
        for number in (1, 2):
            for suffix in ("prepare", "run", "validate", "apply"):
                _equal(manifest["stages"][f"round_{number}_{suffix}"], "skipped_mode_none", f"none round {number} stage")
        if paths.rounds.exists() and any(paths.rounds.iterdir()):
            raise InteractionError("Mode none must not create round artifacts")
    else:
        for number in range(1, max_rounds + 1):
            states_path = paths.round(number) / "states" / "records.jsonl"
            if not states_path.is_file():
                if not active:
                    break
                raise InteractionError("Completed run is missing an active round state")
            expected_inputs = [
                _round_input(item, states[record_identity(item, "interaction input")])
                for item in active
            ]
            actual_inputs = read_jsonl(paths.round(number) / "inputs" / "records.jsonl")
            _equal(actual_inputs, expected_inputs, f"round {number} frozen inputs")
            validations: dict[str, Any] = {}
            proposal_lists: dict[str, list[dict[str, Any]]] = {}
            for branch in active_branches(mode):
                expected_prompts = [
                    build_prompt(
                        item,
                        branch,
                        item["previous_r"],
                        item["previous_h"],
                        number,
                        max_characters=config["chat"]["max_prompt_characters"],
                        max_reason_characters=config["chat"]["max_reason_characters"],
                    )
                    for item in expected_inputs
                ]
                _equal(
                    read_jsonl(paths.branch(number, branch) / "prompts" / "records.jsonl"),
                    expected_prompts,
                    f"round {number} {branch} prompts",
                )
                validations[branch] = validate_branch(
                    config_path=config_path,
                    parent_root=paths.root,
                    round_number=number,
                    branch=branch,
                )
                proposal_lists[branch] = read_jsonl(
                    paths.branch(number, branch) / "parsed" / "proposals.jsonl"
                )
                branch_manifests.append(load_json(paths.branch(number, branch) / "manifest.json"))
            validation_audit = load_json(paths.round(number) / "branch-validation.json")
            _equal(validation_audit["branches"], validations, f"round {number} branch validation")
            if mode == "bidirectional":
                starts = [item["started_unix_ns"] for item in validations.values()]
                ends = [item["completed_unix_ns"] for item in validations.values()]
                if max(starts) > min(ends) or validation_audit["process_overlap"].get("overlapped") is not True:
                    raise InteractionError("Bidirectional process intervals do not overlap")
            actual_states = read_jsonl(states_path)
            if len(actual_states) != len(active):
                raise InteractionError("Round state count mismatch")
            expected_states: list[dict[str, Any]] = []
            expected_interactions: list[dict[str, Any]] = []
            for position, record in enumerate(active):
                identity = record_identity(record, "interaction input")
                proposals = {branch: values[position] for branch, values in proposal_lists.items()}
                state = apply_round(record, *states[identity], proposals, mode, number, gate)
                expected_states.append(state)
                histories[identity].append(state)
                states[identity] = (state["trf_state"], state["exemplar_state"])
                initial = effective_signature(record["r0"], record["h0"])
                history = histories[identity]
                oscillation = (
                    len(history) == 2
                    and history[0]["effective_signature"] != initial
                    and history[1]["effective_signature"] == initial
                )
                expected_interactions.append(
                    {
                        "dataset_id": record["dataset_id"],
                        "record_id": record["record_id"],
                        "source_sha256": record["source_sha256"],
                        "idx": record["idx"],
                        "round": number,
                        "effective_changed": state["effective_changed"],
                        "trf_transitions": state["trf_transitions"],
                        "conflicts": detect_conflicts(
                            state["trf_state"], state["exemplar_state"], oscillation=oscillation
                        ),
                    }
                )
            _equal(actual_states, expected_states, f"round {number} applied states")
            _equal(
                read_jsonl(paths.round(number) / "interactions" / "records.jsonl"),
                expected_interactions,
                f"round {number} interactions",
            )
            _equal(
                load_json(paths.round(number) / "summary.json"),
                {
                    "schema_version": "bidirectional-interaction-round-summary-v1",
                    "round": number,
                    "target_count": len(expected_states),
                    "changed_count": sum(item["effective_changed"] for item in expected_states),
                    "conflict_count": sum(len(item["conflicts"]) for item in expected_interactions),
                },
                f"round {number} summary",
            )
            active = [item for item, state in zip(active, expected_states) if state["effective_changed"]]
            if not active:
                break

    round_artifacts: list[dict[str, Any]] = []
    for number in (1, 2):
        round_root = paths.round(number)
        if not round_root.exists():
            continue
        round_artifacts.append(
            {
                "round": number,
                "files": {
                    item.relative_to(round_root).as_posix(): file_snapshot(item)
                    for item in sorted(path for path in round_root.rglob("*") if path.is_file())
                },
            }
        )
    provenance = {
        "interaction_run_id": run_id,
        "pipeline_version": manifest["pipeline_version"],
        "protocol_version": manifest["protocol_version"],
        "base_concurrent_run_id": manifest["base_run_id"],
        "artifacts": {
            "source_snapshot": file_snapshot(paths.source / "snapshot.json"),
            "interaction_inputs": file_snapshot(paths.inputs),
        },
        "round_artifacts": round_artifacts,
    }
    contexts: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    for record in rebuilt:
        identity = record_identity(record, "interaction input")
        r_state, h_state = states[identity]
        history = histories[identity]
        initial_signature = effective_signature(record["r0"], record["h0"])
        oscillation = (
            len(history) == 2
            and history[0]["effective_signature"] != initial_signature
            and history[1]["effective_signature"] == initial_signature
        )
        conflicts = detect_conflicts(r_state, h_state, oscillation=oscillation)
        contexts.append(
            assemble_context(
                record,
                r_state,
                h_state,
                mode=mode,
                rounds=len(history),
                converged=mode == "none" or not history or not history[-1]["effective_changed"],
                conflicts=conflicts,
                provenance=provenance,
            )
        )
        interactions.append(
            {
                "dataset_id": record["dataset_id"],
                "record_id": record["record_id"],
                "idx": record["idx"],
                "rounds": history,
                "final_conflicts": conflicts,
            }
        )
    actual_contexts = read_jsonl(paths.context / "records.jsonl")
    _equal(actual_contexts, contexts, "final contexts")
    _equal(read_jsonl(paths.audit / "interactions.jsonl"), interactions, "interaction audit")
    summary = context_summary(contexts, _metrics(branch_manifests))
    _equal(load_json(paths.context / "summary.json"), summary, "context summary")
    _equal(manifest.get("summary"), summary, "manifest summary")

    gate_status = manifest.get("gate_status")
    gate_path = paths.audit / "gate-evaluation.json"
    if "semantic_gate" in compatibility:
        spec_path = Path(compatibility["semantic_gate"]["path"])
        spec = load_gate_spec(spec_path)
        if not gate_path.is_file():
            raise InteractionError("Gate run is missing gate-evaluation.json")
        evaluation = evaluate_gate(spec, contexts, summary)
        _equal(load_json(gate_path), evaluation, "gate evaluation")
        _equal(gate_status, evaluation["status"], "manifest gate status")
    elif gate_path.exists() or gate_status is not None:
        raise InteractionError("Normal interaction run must not contain gate evaluation")
    _equal(collect_output_hashes(paths), manifest.get("outputs"), "output SHA256 ledger")
    return {
        "run_id": run_id,
        "status": "valid",
        "targets": len(contexts),
        "ready": sum(item["assembly_status"] == "ready" for item in contexts),
        "mode": mode,
        "gate_status": gate_status,
    }


def main() -> int:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    try:
        result = validate(config_path, args.run_id)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"ValidateBidirectionalInteraction failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
