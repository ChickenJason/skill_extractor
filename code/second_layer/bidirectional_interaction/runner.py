"""Orchestrate synchronous TRF--exemplar interaction rounds."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.contracts import record_identity  # noqa: E402
from common.io_utils import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    read_jsonl,
    resolve_project_path,
    utc_now,
)
from second_layer.bidirectional_interaction.branch_validator import validate_branch  # noqa: E402
from second_layer.bidirectional_interaction.common import (  # noqa: E402
    BASE_PIPELINE_VERSION,
    BRANCHES,
    MODES,
    PIPELINE_VERSION,
    PROTOCOL_VERSION,
    InteractionError,
    InteractionPartialError,
    InteractionPaths,
    assert_snapshot,
    branch_run_id,
    collect_output_hashes,
    file_snapshot,
    initialize_run,
    load_config,
    paths_for,
    update_manifest,
    update_stage,
)
from second_layer.bidirectional_interaction.contracts import rebuild_inputs  # noqa: E402
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


def _now_microseconds() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/second_layer_bidirectional_interaction.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-concurrent-run-id", required=True)
    parser.add_argument("--mode", choices=sorted(MODES))
    parser.add_argument("--max-rounds", type=int, choices=[1, 2])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace, mode: str, max_rounds: int) -> None:
    if args.limit is not None and args.limit < 1:
        raise InteractionError("--limit must be a positive integer")
    if args.retry_failed and not args.resume:
        raise InteractionError("--retry-failed requires --resume")
    if max_rounds not in {1, 2}:
        raise InteractionError("--max-rounds must be 1 or 2")
    if mode not in MODES:
        raise InteractionError("Unsupported interaction mode")
    if mode != "none" and not args.prepare_only and not args.allow_network:
        raise InteractionError("Online interaction requires --allow-network")
    if mode != "none" and args.limit is None and not args.prepare_only and not args.confirm_full_run:
        raise InteractionError("An unlimited online interaction requires --confirm-full-run")


def _base_sources(
    config: dict[str, Any], base_run_id: str
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Path]]:
    concurrent_config = Path(config["base"]["concurrent_config"])
    validation = validate_second_layer(concurrent_config, base_run_id)
    concurrent = load_json(concurrent_config)
    base_root = (
        resolve_project_path(PROJECT_ROOT, concurrent["output"]["runs_root"]) / base_run_id
    ).resolve()
    manifest_path = base_root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("pipeline_version") not in config["base"]["accepted_pipeline_versions"]:
        raise InteractionError("Base pipeline version is not accepted")
    exemplar_child = manifest.get("children", {}).get("exemplar")
    if not isinstance(exemplar_child, dict) or not exemplar_child.get("run_root"):
        raise InteractionError("Base manifest has no exemplar child root")
    exemplar_root = Path(exemplar_child["run_root"]).resolve()
    if exemplar_root.name != Path(exemplar_child["manifest"]).resolve().parent.name:
        raise InteractionError("Base exemplar child root/manifest disagree")
    sources = {
        "base_context": base_root / "context" / "records.jsonl",
        "candidates": exemplar_root / "candidates" / "records.jsonl",
        "judgments": exemplar_root / "parsed" / "judgments.jsonl",
    }
    for path in sources.values():
        if not path.is_file():
            raise InteractionError(f"Required base source does not exist: {path}")
    return manifest, manifest_path, validation, sources


def _gate(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    path = Path(config["resources"]["instance_discriminator_config"])
    value = load_json(path)
    gate = value.get("gate")
    required = {
        "minimum_helpfulness", "max_selected", "max_supporting",
        "max_contrastive", "minimum_for_complete",
    }
    if not isinstance(gate, dict) or set(gate) != required:
        raise InteractionError("Instance discriminator hard gate contract is invalid")
    return copy.deepcopy(gate), path


def _select_inputs(
    sources: dict[str, Path],
    gate: dict[str, Any],
    *,
    limit: int | None,
    selected_identities: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    all_inputs = rebuild_inputs(
        read_jsonl(sources["base_context"]),
        read_jsonl(sources["candidates"]),
        read_jsonl(sources["judgments"]),
        gate,
    )
    if selected_identities is not None:
        requested = {
            (item["dataset_id"], item["record_id"]): item for item in selected_identities
        }
        if len(requested) != len(selected_identities):
            raise InteractionError("Gate target identities are duplicated")
        by_id = {record_identity(item, "interaction input"): item for item in all_inputs}
        if not set(requested) <= set(by_id):
            raise InteractionError("Gate target identity is absent from the base run")
        selected = []
        for item in selected_identities:
            identity = (item["dataset_id"], item["record_id"])
            record = by_id[identity]
            expected = {
                key: record[key]
                for key in ("dataset_id", "record_id", "idx", "sentence", "source_sha256")
            }
            actual = {key: item.get(key) for key in expected}
            if actual != expected:
                raise InteractionError(f"Gate target full identity mismatch for {identity!r}")
            selected.append(record)
        return sorted(selected, key=lambda item: item["idx"])
    return all_inputs[:limit] if limit is not None else all_inputs


def _identities(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset_id": item["dataset_id"],
            "record_id": item["record_id"],
            "source_sha256": item["source_sha256"],
            "idx": item["idx"],
            "sentence": item["sentence"],
        }
        for item in records
    ]


def _compatibility(
    *,
    config_path: Path,
    base_run_id: str,
    base_manifest_path: Path,
    sources: dict[str, Path],
    records: list[dict[str, Any]],
    mode: str,
    max_rounds: int,
    gate_path: Path,
) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "schedule_type": "bidirectional_interaction",
        "base_schedule_type": "concurrent",
        "config": file_snapshot(config_path),
        "base": {
            "run_id": base_run_id,
            "pipeline_version": BASE_PIPELINE_VERSION,
            "manifest": file_snapshot(base_manifest_path),
            "artifacts": {name: file_snapshot(path) for name, path in sources.items()},
        },
        "target": {"count": len(records), "identities": _identities(records)},
        "interaction": {"mode": mode, "max_rounds": max_rounds},
        "hard_gate_config": file_snapshot(gate_path),
    }


def _freeze(
    paths: InteractionPaths,
    records: list[dict[str, Any]],
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    base_contexts = [item["baseline_context"] for item in records]
    candidates = [
        {
            "schema_version": "candidate-instances-v1",
            "dataset_id": item["dataset_id"],
            "record_id": item["record_id"],
            "source_sha256": item["source_sha256"],
            "idx": item["idx"],
            "sentence": item["sentence"],
            "candidate_count": 16,
            "candidates": item["candidates"],
        }
        for item in records
    ]
    judgments = [
        {
            "schema_version": "instance-judgments-v1",
            "dataset_id": item["dataset_id"],
            "record_id": item["record_id"],
            "source_sha256": item["source_sha256"],
            "idx": item["idx"],
            "sentence": item["sentence"],
            "candidate_count": 16,
            "judgments": item["baseline_judgments"],
        }
        for item in records
    ]
    atomic_write_jsonl(paths.source / "base-context.jsonl", base_contexts)
    atomic_write_jsonl(paths.source / "candidate-records.jsonl", candidates)
    atomic_write_jsonl(paths.source / "judgments.jsonl", judgments)
    atomic_write_jsonl(paths.inputs, records)
    snapshot = {
        "schema_version": "bidirectional-interaction-source-snapshot-v1",
        "originals": copy.deepcopy(compatibility["base"]["artifacts"]),
        "frozen": {
            "base_context": file_snapshot(paths.source / "base-context.jsonl"),
            "candidates": file_snapshot(paths.source / "candidate-records.jsonl"),
            "judgments": file_snapshot(paths.source / "judgments.jsonl"),
            "interaction_inputs": file_snapshot(paths.inputs),
        },
    }
    atomic_write_json(paths.source / "snapshot.json", snapshot)
    return snapshot


def _assert_frozen(snapshot: dict[str, Any]) -> None:
    for group in ("originals", "frozen"):
        for item in snapshot[group].values():
            assert_snapshot(item)


def _round_input(record: dict[str, Any], r_state: dict[str, Any], h_state: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(record)
    value["previous_r"] = copy.deepcopy(r_state)
    value["previous_h"] = copy.deepcopy(h_state)
    return value


def _prepare_round(
    paths: InteractionPaths,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    states: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]],
    mode: str,
    round_number: int,
) -> list[dict[str, Any]]:
    round_root = paths.round(round_number)
    (round_root / "inputs").mkdir(parents=True, exist_ok=True)
    round_inputs = [
        _round_input(item, *states[record_identity(item, "interaction input")])
        for item in records
    ]
    input_path = round_root / "inputs" / "records.jsonl"
    if input_path.exists():
        if read_jsonl(input_path) != round_inputs:
            raise InteractionError("Frozen round inputs changed before resume")
    else:
        atomic_write_jsonl(input_path, round_inputs)
    for branch in active_branches(mode):
        prompt_dir = paths.branch(round_number, branch) / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompts = [
            build_prompt(
                item,
                branch,
                item["previous_r"],
                item["previous_h"],
                round_number,
                max_characters=config["chat"]["max_prompt_characters"],
                max_reason_characters=config["chat"]["max_reason_characters"],
            )
            for item in round_inputs
        ]
        prompt_path = prompt_dir / "records.jsonl"
        if prompt_path.exists():
            if read_jsonl(prompt_path) != prompts:
                raise InteractionError("Frozen round prompts changed before resume")
        else:
            atomic_write_jsonl(prompt_path, prompts)
    prepared_path = round_root / "prepared.json"
    static = {
        "schema_version": "bidirectional-interaction-round-prepared-v1",
        "round": round_number,
        "mode": mode,
        "branches": list(active_branches(mode)),
        "target_count": len(round_inputs),
    }
    if prepared_path.exists():
        existing = load_json(prepared_path)
        if {key: existing.get(key) for key in static} != static:
            raise InteractionError("Frozen round preparation contract changed")
    else:
        atomic_write_json(prepared_path, {**static, "prepared_at": utc_now()})
    return round_inputs


def _branch_command(
    *,
    config_path: Path,
    paths: InteractionPaths,
    run_id: str,
    round_number: int,
    branch: str,
    allow_network: bool,
    resume: bool,
    retry_failed: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "branch_runner.py"),
        "--config", str(config_path),
        "--parent-root", str(paths.root),
        "--run-id", branch_run_id(run_id, round_number, branch),
        "--branch", branch,
        "--round", str(round_number),
    ]
    if allow_network:
        command.append("--allow-network")
    if resume:
        command.append("--resume")
    if retry_failed:
        command.append("--retry-failed")
    return command


def _run_branches(
    *,
    config_path: Path,
    paths: InteractionPaths,
    run_id: str,
    mode: str,
    round_number: int,
    allow_network: bool,
    resume: bool,
    retry_failed: bool,
) -> dict[str, Any]:
    processes: dict[str, tuple[subprocess.Popen[str], Any, list[str], int, str, Path]] = {}
    for branch in active_branches(mode):
        command = _branch_command(
            config_path=config_path,
            paths=paths,
            run_id=run_id,
            round_number=round_number,
            branch=branch,
            allow_network=allow_network,
            resume=resume,
            retry_failed=retry_failed,
        )
        log_path = paths.logs / f"round-{round_number:02d}-{branch}.log"
        handle = log_path.open("a", encoding="utf-8")
        started_ns = time.time_ns()
        started_at = _now_microseconds()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes[branch] = (process, handle, command, started_ns, started_at, log_path)
    states: dict[str, Any] = {}
    for branch, (process, handle, command, started_ns, started_at, log_path) in processes.items():
        returncode = process.wait()
        completed_ns = time.time_ns()
        completed_at = _now_microseconds()
        handle.close()
        states[branch] = {
            "branch": branch,
            "command": command,
            "pid": process.pid,
            "started_at": started_at,
            "started_unix_ns": started_ns,
            "completed_at": completed_at,
            "completed_unix_ns": completed_ns,
            "returncode": returncode,
            "log": str(log_path),
        }
    return states


def _overlap(validations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(validations) != set(BRANCHES):
        return {"required": False, "overlapped": None}
    starts = [item["started_unix_ns"] for item in validations.values()]
    ends = [item["completed_unix_ns"] for item in validations.values()]
    overlapped = max(starts) <= min(ends)
    if not overlapped:
        raise InteractionError("Bidirectional branch process intervals did not overlap")
    return {
        "required": True,
        "overlapped": True,
        "intersection_start_unix_ns": max(starts),
        "intersection_end_unix_ns": min(ends),
    }


def _remove_context(paths: InteractionPaths) -> None:
    for name in ("records.jsonl", "summary.json"):
        path = paths.context / name
        if path.exists():
            path.unlink()


def _finish_partial(paths: InteractionPaths, error: str | None) -> dict[str, Any]:
    _remove_context(paths)
    update_manifest(
        paths,
        status="partial",
        completed_at=utc_now(),
        error=error,
        outputs=collect_output_hashes(paths),
    )
    return load_json(paths.manifest)


def _aggregate_metrics(round_manifests: list[dict[str, Any]]) -> dict[str, int]:
    names = ("model_responses", "repair_responses", "failed_records", "prompt_tokens", "completion_tokens")
    return {name: sum(int(item.get("metrics", {}).get(name, 0)) for item in round_manifests) for name in names}


def run(
    args: argparse.Namespace,
    *,
    selected_identities: list[dict[str, Any]] | None = None,
    gate_spec_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    config = load_config(config_path)
    mode = args.mode or config["interaction"]["default_mode"]
    max_rounds = args.max_rounds or config["interaction"]["max_rounds"]
    _validate_args(args, mode, max_rounds)
    if selected_identities is not None and args.limit is not None:
        raise InteractionError("Gate-selected runs cannot also use --limit")

    base_manifest, base_manifest_path, base_validation, sources = _base_sources(
        config, args.base_concurrent_run_id
    )
    if len(read_jsonl(sources["base_context"])) != config["base"]["expected_target_count"]:
        raise InteractionError("Base run target count differs from the frozen contract")
    gate, gate_path = _gate(config)
    records = _select_inputs(
        sources,
        gate,
        limit=args.limit,
        selected_identities=selected_identities,
    )
    if not records:
        raise InteractionError("Interaction target selection is empty")
    compatibility = _compatibility(
        config_path=config_path,
        base_run_id=args.base_concurrent_run_id,
        base_manifest_path=base_manifest_path,
        sources=sources,
        records=records,
        mode=mode,
        max_rounds=max_rounds,
        gate_path=gate_path,
    )
    if gate_spec_snapshot is not None:
        compatibility["semantic_gate"] = copy.deepcopy(gate_spec_snapshot)
    paths = paths_for(config, args.run_id)
    manifest = initialize_run(
        paths,
        run_id=args.run_id,
        base_run_id=args.base_concurrent_run_id,
        compatibility=compatibility,
        command=list(sys.argv),
        resume=args.resume,
    )
    try:
        update_stage(paths, "validate_base", "completed")
        update_manifest(paths, base_validation=base_validation)
        if manifest.get("source_snapshot") is None:
            snapshot = _freeze(paths, records, compatibility)
            update_manifest(paths, source_snapshot=snapshot)
        else:
            snapshot = load_json(paths.source / "snapshot.json")
            if snapshot != manifest["source_snapshot"]:
                raise InteractionError("Manifest/source snapshot mismatch")
            _assert_frozen(snapshot)
            records = read_jsonl(paths.inputs)
        update_stage(paths, "freeze_base", "completed")

        states = {
            record_identity(item, "interaction input"): (copy.deepcopy(item["r0"]), copy.deepcopy(item["h0"]))
            for item in records
        }
        histories: dict[tuple[str, str], list[dict[str, Any]]] = {
            record_identity(item, "interaction input"): [] for item in records
        }
        active = list(records)
        round_manifests: list[dict[str, Any]] = []
        process_audit: dict[str, Any] = copy.deepcopy(load_json(paths.manifest).get("processes", {}))

        if mode == "none":
            for round_number in (1, 2):
                for suffix in ("prepare", "run", "validate", "apply"):
                    update_stage(paths, f"round_{round_number}_{suffix}", "skipped_mode_none")
        else:
            for round_number in range(1, max_rounds + 1):
                stage_prefix = f"round_{round_number}"
                states_path = paths.round(round_number) / "states" / "records.jsonl"
                if states_path.is_file():
                    existing = read_jsonl(states_path)
                    if {record_identity(item, "round state") for item in existing} != {
                        record_identity(item, "interaction input") for item in active
                    }:
                        raise InteractionError("Applied round state identity mismatch on resume")
                    by_id = {record_identity(item, "round state"): item for item in existing}
                    for item in active:
                        identity = record_identity(item, "interaction input")
                        state = by_id[identity]
                        histories[identity].append(state)
                        states[identity] = (state["trf_state"], state["exemplar_state"])
                    for branch in active_branches(mode):
                        branch_manifest_path = paths.branch(round_number, branch) / "manifest.json"
                        if not branch_manifest_path.is_file():
                            raise InteractionError("Applied round is missing its branch manifest")
                        round_manifests.append(load_json(branch_manifest_path))
                    active = [
                        item for item in active
                        if by_id[record_identity(item, "interaction input")]["effective_changed"]
                    ]
                    if not active and round_number == 1 and max_rounds == 2:
                        for suffix in ("prepare", "run", "validate", "apply"):
                            update_stage(paths, f"round_2_{suffix}", "skipped_converged")
                        break
                    continue

                _prepare_round(paths, config, active, states, mode, round_number)
                update_stage(paths, f"{stage_prefix}_prepare", "completed")
                if args.prepare_only:
                    update_stage(paths, f"{stage_prefix}_run", "pending")
                    return _finish_partial(paths, None)

                branches = _run_branches(
                    config_path=config_path,
                    paths=paths,
                    run_id=args.run_id,
                    mode=mode,
                    round_number=round_number,
                    allow_network=args.allow_network,
                    resume=args.resume,
                    retry_failed=args.retry_failed,
                )
                process_audit[f"round_{round_number}"] = branches
                update_manifest(paths, processes=process_audit)
                if any(item["returncode"] != 0 for item in branches.values()):
                    if any(item["returncode"] == 2 for item in branches.values()):
                        update_stage(paths, f"{stage_prefix}_run", "failed")
                        raise InteractionError("A branch detected a source/config/hash contract violation")
                    update_stage(paths, f"{stage_prefix}_run", "partial")
                    return _finish_partial(paths, "At least one branch process failed")
                branch_manifests = {
                    branch: load_json(paths.branch(round_number, branch) / "manifest.json")
                    for branch in active_branches(mode)
                }
                round_manifests.extend(branch_manifests.values())
                network = copy.deepcopy(load_json(paths.manifest)["network"])
                for branch, item in branch_manifests.items():
                    network[branch] = network[branch] or bool(item.get("network_called"))
                update_manifest(paths, network=network, network_called=any(network.values()))
                if any(item.get("status") != "completed" for item in branch_manifests.values()):
                    update_stage(paths, f"{stage_prefix}_run", "partial")
                    return _finish_partial(paths, "At least one interaction branch is partial")
                update_stage(paths, f"{stage_prefix}_run", "completed")

                validations = {
                    branch: validate_branch(
                        config_path=config_path,
                        parent_root=paths.root,
                        round_number=round_number,
                        branch=branch,
                    )
                    for branch in active_branches(mode)
                }
                overlap = _overlap(validations) if mode == "bidirectional" else {"required": False, "overlapped": None}
                atomic_write_json(
                    paths.round(round_number) / "branch-validation.json",
                    {"branches": validations, "process_overlap": overlap},
                )
                update_stage(paths, f"{stage_prefix}_validate", "completed")

                proposal_lists = {
                    branch: read_jsonl(paths.branch(round_number, branch) / "parsed" / "proposals.jsonl")
                    for branch in active_branches(mode)
                }
                if any(len(values) != len(active) for values in proposal_lists.values()):
                    raise InteractionError("Validated proposal count differs from active targets")
                applied: list[dict[str, Any]] = []
                for position, record in enumerate(active):
                    identity = record_identity(record, "interaction input")
                    proposals = {branch: values[position] for branch, values in proposal_lists.items()}
                    state = apply_round(
                        record,
                        *states[identity],
                        proposals,
                        mode,
                        round_number,
                        gate,
                    )
                    histories[identity].append(state)
                    states[identity] = (state["trf_state"], state["exemplar_state"])
                    applied.append(state)
                states_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_jsonl(states_path, applied)
                round_interactions: list[dict[str, Any]] = []
                for record, state in zip(active, applied):
                    identity = record_identity(record, "interaction input")
                    initial = effective_signature(record["r0"], record["h0"])
                    history = histories[identity]
                    oscillation = (
                        len(history) == 2
                        and history[0]["effective_signature"] != initial
                        and history[1]["effective_signature"] == initial
                    )
                    round_interactions.append(
                        {
                            "dataset_id": record["dataset_id"],
                            "record_id": record["record_id"],
                            "source_sha256": record["source_sha256"],
                            "idx": record["idx"],
                            "round": round_number,
                            "effective_changed": state["effective_changed"],
                            "trf_transitions": state["trf_transitions"],
                            "conflicts": detect_conflicts(
                                state["trf_state"],
                                state["exemplar_state"],
                                oscillation=oscillation,
                            ),
                        }
                    )
                interaction_path = paths.round(round_number) / "interactions" / "records.jsonl"
                interaction_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_jsonl(interaction_path, round_interactions)
                atomic_write_json(
                    paths.round(round_number) / "summary.json",
                    {
                        "schema_version": "bidirectional-interaction-round-summary-v1",
                        "round": round_number,
                        "target_count": len(applied),
                        "changed_count": sum(item["effective_changed"] for item in applied),
                        "conflict_count": sum(len(item["conflicts"]) for item in round_interactions),
                    },
                )
                update_stage(paths, f"{stage_prefix}_apply", "completed")
                active = [
                    record for record, state in zip(active, applied) if state["effective_changed"]
                ]
                if not active:
                    if round_number == 1:
                        for suffix in ("prepare", "run", "validate", "apply"):
                            update_stage(paths, f"round_2_{suffix}", "skipped_converged")
                    break
            if max_rounds == 1:
                for suffix in ("prepare", "run", "validate", "apply"):
                    update_stage(paths, f"round_2_{suffix}", "skipped_max_rounds")

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
            "interaction_run_id": args.run_id,
            "pipeline_version": PIPELINE_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "base_concurrent_run_id": args.base_concurrent_run_id,
            "artifacts": {
                "source_snapshot": file_snapshot(paths.source / "snapshot.json"),
                "interaction_inputs": file_snapshot(paths.inputs),
            },
            "round_artifacts": round_artifacts,
        }
        contexts: list[dict[str, Any]] = []
        interactions: list[dict[str, Any]] = []
        for record in records:
            identity = record_identity(record, "interaction input")
            r_state, h_state = states[identity]
            history = histories[identity]
            oscillation = (
                len(history) == 2
                and history[0]["effective_signature"] != effective_signature(record["r0"], record["h0"])
                and history[1]["effective_signature"] == effective_signature(record["r0"], record["h0"])
            )
            conflicts = detect_conflicts(r_state, h_state, oscillation=oscillation)
            rounds = len(history)
            converged = mode == "none" or not history or not history[-1]["effective_changed"]
            context = assemble_context(
                record,
                r_state,
                h_state,
                mode=mode,
                rounds=rounds,
                converged=converged,
                conflicts=conflicts,
                provenance=provenance,
            )
            contexts.append(context)
            interactions.append(
                {
                    "dataset_id": record["dataset_id"],
                    "record_id": record["record_id"],
                    "idx": record["idx"],
                    "rounds": history,
                    "final_conflicts": conflicts,
                }
            )
        atomic_write_jsonl(paths.context / "records.jsonl", contexts)
        update_stage(paths, "assemble_context", "completed")
        atomic_write_jsonl(paths.audit / "interactions.jsonl", interactions)
        manual = [
            {
                "dataset_id": item["dataset_id"],
                "record_id": item["record_id"],
                "idx": item["idx"],
                "status": item["interaction"]["status"],
                "conflicts": item["interaction"]["conflicts"],
                "review_reasons": item["interaction"]["review_reasons"],
            }
            for item in contexts
            if item["interaction"]["status"] != "aligned"
        ]
        atomic_write_jsonl(paths.audit / "manual-review.jsonl", manual)
        metrics = _aggregate_metrics(round_manifests)
        summary = context_summary(contexts, metrics)
        atomic_write_json(paths.context / "summary.json", summary)
        update_stage(paths, "diagnostics", "completed")
        update_manifest(
            paths,
            status="completed",
            completed_at=utc_now(),
            rounds=[
                {
                    "round": number,
                    "status": load_json(paths.manifest)["stages"][f"round_{number}_apply"],
                }
                for number in (1, 2)
            ],
            summary=summary,
            error=None,
        )
        update_manifest(paths, outputs=collect_output_hashes(paths))
        return load_json(paths.manifest)
    except InteractionPartialError as error:
        return _finish_partial(paths, str(error))
    except Exception as error:
        _remove_context(paths)
        update_manifest(
            paths,
            status="failed",
            completed_at=utc_now(),
            error=str(error),
            outputs=collect_output_hashes(paths),
        )
        raise


def main() -> int:
    args = parse_args()
    try:
        manifest = run(args)
        print(json.dumps({"run_id": manifest["run_id"], "status": manifest["status"]}, ensure_ascii=False))
        return 0
    except Exception as error:
        print(f"RunBidirectionalInteraction failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
