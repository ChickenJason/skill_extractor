"""Schedule two single-round reflectors concurrently over a validated frozen R0 baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
from second_layer.collaborative_reflection.branch_validator import validate_branch  # noqa: E402
from second_layer.collaborative_reflection.common import (  # noqa: E402
    BRANCHES,
    PIPELINE_VERSION,
    ReflectionError,
    assert_snapshot,
    branch_run_ids,
    collect_output_hashes,
    compatibility_payload,
    file_snapshot,
    initialize_run,
    load_reflection_config,
    reflection_paths,
    update_manifest,
    update_stage,
)
from second_layer.collaborative_reflection.pipeline import (  # noqa: E402
    assemble_context_records,
    build_prompt,
    build_reflection_inputs,
    context_summary,
)
from second_layer.common import load_second_layer_config, second_layer_run_paths  # noqa: E402
from second_layer.validator import validate as validate_concurrent  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/second_layer_reflection.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-concurrent-run-id", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise ReflectionError("--limit must be a positive integer")
    if args.retry_failed and not args.resume:
        raise ReflectionError("--retry-failed requires --resume")
    if not args.prepare_only and not args.allow_network:
        raise ReflectionError("Reflection inference requires --allow-network")
    if args.limit is None and not args.prepare_only and not args.confirm_full_run:
        raise ReflectionError("An unlimited online reflection run requires --confirm-full-run")


def validate_base(
    config: dict[str, Any], base_run_id: str
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Path]]:
    concurrent_config_path = Path(config["base"]["concurrent_config"])
    concurrent_config = load_second_layer_config(concurrent_config_path)
    base_paths = second_layer_run_paths(concurrent_config, base_run_id)
    base_manifest = load_json(base_paths.manifest)
    if base_manifest.get("status") != "completed":
        raise ReflectionError("Base concurrent run must be completed")
    validation = validate_concurrent(concurrent_config_path, base_run_id)
    expected = config["base"]["expected_target_count"]
    if validation.get("targets") != expected or validation.get("ready") != expected:
        raise ReflectionError(
            f"Base concurrent run must contain exactly {expected} ready targets"
        )
    exemplar_root = Path(base_manifest["children"]["exemplar"]["run_root"]).resolve()
    sources = {
        "base_context": base_paths.context / "records.jsonl",
        "candidates": exemplar_root / "candidates" / "records.jsonl",
        "judgments": exemplar_root / "parsed" / "judgments.jsonl",
    }
    for name, path in sources.items():
        if not path.is_file():
            raise ReflectionError(f"Base {name} artifact is missing: {path}")
    return base_manifest, base_paths.manifest, validation, sources


def _selected_sources(
    sources: dict[str, Path], *, limit: int | None, candidate_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    contexts = sorted(read_jsonl(sources["base_context"]), key=lambda item: item.get("idx", -1))
    if limit is not None:
        contexts = contexts[:limit]
    identities = {(item["dataset_id"], item["record_id"]) for item in contexts}
    candidates = [
        item for item in read_jsonl(sources["candidates"])
        if (item.get("dataset_id"), item.get("record_id")) in identities
    ]
    judgments = [
        item for item in read_jsonl(sources["judgments"])
        if (item.get("dataset_id"), item.get("record_id")) in identities
    ]
    inputs = build_reflection_inputs(
        contexts, candidates, judgments, candidate_count=candidate_count
    )
    return contexts, candidates, judgments, inputs


def _identity_contract(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset_id": item["dataset_id"],
            "record_id": item["record_id"],
            "source_sha256": item["source_sha256"],
            "idx": item["idx"],
        }
        for item in inputs
    ]


def _freeze(
    paths: Any,
    contexts: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    destinations = {
        "base_context": paths.source / "base-context.jsonl",
        "candidates": paths.source / "candidate-records.jsonl",
        "judgments": paths.source / "judgments.jsonl",
    }
    atomic_write_jsonl(destinations["base_context"], contexts)
    atomic_write_jsonl(destinations["candidates"], candidates)
    atomic_write_jsonl(destinations["judgments"], judgments)
    snapshot: dict[str, Any] = {"schema_version": "collaborative-reflection-source-snapshot-v1"}
    snapshot.update({name: file_snapshot(path) for name, path in destinations.items()})
    atomic_write_json(paths.source / "snapshot.json", snapshot)
    return snapshot


def _prepare(paths: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot = load_json(paths.source / "snapshot.json")
    assert_snapshot(snapshot)
    inputs = build_reflection_inputs(
        read_jsonl(Path(snapshot["base_context"]["path"])),
        read_jsonl(Path(snapshot["candidates"]["path"])),
        read_jsonl(Path(snapshot["judgments"]["path"])),
        candidate_count=config["reflection"]["candidate_count"],
    )
    atomic_write_jsonl(paths.inputs, inputs)
    for branch in BRANCHES:
        branch_root = paths.branch(branch)
        prompt_path = branch_root / "prompts" / "records.jsonl"
        prompts = [
            build_prompt(
                item,
                branch,
                config["chat"]["max_prompt_characters"],
                config["chat"]["max_reason_characters"],
            )
            for item in inputs
        ]
        atomic_write_jsonl(prompt_path, prompts)
    return inputs


def _creation_flags() -> tuple[int, Any | None]:
    if os.name != "nt":
        return 0, None
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    return flags, startup


def _branch_command(
    *, python: str, config_path: Path, paths: Any, run_id: str, branch: str,
    allow_network: bool, resume: bool, retry_failed: bool
) -> list[str]:
    command = [
        python,
        str(Path(__file__).resolve().parent / "branch_runner.py"),
        "--config", str(config_path),
        "--parent-root", str(paths.root),
        "--run-id", branch_run_ids(run_id)[branch],
        "--branch", branch,
    ]
    if allow_network:
        command.append("--allow-network")
    if resume:
        command.append("--resume")
    if retry_failed:
        command.append("--retry-failed")
    return command


def _run_parallel(
    *, paths: Any, config_path: Path, run_id: str, python: str,
    allow_network: bool, resume: bool, retry_failed: bool
) -> dict[str, Any]:
    commands: dict[str, tuple[list[str], Path]] = {}
    for branch in BRANCHES:
        commands[branch] = (
            _branch_command(
                python=python,
                config_path=config_path,
                paths=paths,
                run_id=run_id,
                branch=branch,
                allow_network=allow_network,
                resume=resume,
                retry_failed=retry_failed,
            ),
            paths.logs / f"{branch}.log",
        )
    return execute_parallel_processes(commands)


def execute_parallel_processes(
    commands: dict[str, tuple[list[str], Path]],
    *,
    cwd: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Start every process before waiting, then wait for every process safely."""

    flags, startup = _creation_flags()
    processes: dict[str, subprocess.Popen[Any]] = {}
    handles: dict[str, Any] = {}
    states: dict[str, dict[str, Any]] = {}
    try:
        for branch, (command, log_path) in commands.items():
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8", newline="\n")
            handles[branch] = handle
            started = utc_now()
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                startupinfo=startup,
            )
            processes[branch] = process
            states[branch] = {
                "command": command,
                "pid": process.pid,
                "started_at": started,
                "ended_at": None,
                "exit_code": None,
                "log": str(log_path.resolve()),
            }
        for branch in commands:
            exit_code = processes[branch].wait()
            states[branch]["ended_at"] = utc_now()
            states[branch]["exit_code"] = exit_code
    finally:
        for handle in handles.values():
            handle.close()
    return states


def _branch_states(paths: Any) -> dict[str, Any]:
    states: dict[str, Any] = {}
    for branch in BRANCHES:
        root = paths.branch(branch)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            states[branch] = {"run_root": str(root), "status": "failed", "manifest": None}
            continue
        manifest = load_json(manifest_path)
        states[branch] = {
            "run_root": str(root),
            "manifest": file_snapshot(manifest_path),
            "status": manifest.get("status"),
            "network_called": bool(manifest.get("network_called")),
            "counts": manifest.get("counts"),
            "metrics": manifest.get("metrics"),
        }
    return states


def _provenance(paths: Any, run_id: str, branches: dict[str, Any]) -> dict[str, Any]:
    return {
        "reflection_run_id": run_id,
        "base_concurrent_run_id": load_json(paths.manifest)["base_run_id"],
        "trf_reflection_run_id": branch_run_ids(run_id)["trf"],
        "exemplar_reflection_run_id": branch_run_ids(run_id)["exemplar"],
        "schedule_type": "collaborative_reflection",
        "base_schedule_type": "concurrent",
        "artifacts": {
            "base_context": file_snapshot(paths.source / "base-context.jsonl"),
            "candidates": file_snapshot(paths.source / "candidate-records.jsonl"),
            "baseline_judgments": file_snapshot(paths.source / "judgments.jsonl"),
            "reflection_inputs": file_snapshot(paths.inputs),
            "trf_reflection": file_snapshot(paths.trf / "parsed" / "records.jsonl"),
            "exemplar_reflection": file_snapshot(paths.exemplar / "parsed" / "judgments.jsonl"),
            "exemplar_selection": file_snapshot(paths.exemplar / "selected" / "records.jsonl"),
            "trf_manifest": branches["trf"]["manifest"],
            "exemplar_manifest": branches["exemplar"]["manifest"],
        },
    }


def _remove_unpublished_context(paths: Any) -> None:
    for path in (paths.context / "records.jsonl", paths.context / "summary.json", paths.interaction / "records.jsonl"):
        if path.exists():
            path.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    config = load_reflection_config(config_path)

    # This gate deliberately occurs before a reflection directory can be created.
    base_manifest, base_manifest_path, base_validation, sources = validate_base(
        config, args.base_concurrent_run_id
    )
    contexts, candidates, judgments, preview_inputs = _selected_sources(
        sources,
        limit=args.limit,
        candidate_count=config["reflection"]["candidate_count"],
    )
    if not preview_inputs:
        raise ReflectionError("Reflection target selection is empty")
    paths = reflection_paths(config, args.run_id)
    compatibility = compatibility_payload(
        config_path=config_path,
        run_id=args.run_id,
        base_run_id=args.base_concurrent_run_id,
        base_manifest=base_manifest,
        base_manifest_path=base_manifest_path,
        base_sources=sources,
        identities=_identity_contract(preview_inputs),
        limit=args.limit,
        config=config,
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
        if manifest.get("source_snapshot") is None:
            snapshot = _freeze(paths, contexts, candidates, judgments)
            update_manifest(paths, source_snapshot=snapshot)
        else:
            snapshot = load_json(paths.source / "snapshot.json")
            assert_snapshot(snapshot)
            if snapshot != manifest["source_snapshot"]:
                raise ReflectionError("Manifest source snapshot mismatch")
        update_stage(paths, "freeze_base", "completed")
        inputs = _prepare(paths, config)
        if _identity_contract(inputs) != compatibility["target"]["identities"]:
            raise ReflectionError("Prepared reflection identities changed")
        update_stage(paths, "prepare_reflection", "completed")
        if args.prepare_only:
            update_manifest(paths, status="partial", completed_at=utc_now(), error=None)
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
            raise ReflectionError("At least one reflector process failed; inspect branch logs")
        branches = _branch_states(paths)
        network = {name: bool(branches[name].get("network_called")) for name in BRANCHES}
        update_manifest(
            paths,
            branches=branches,
            network=network,
            network_called=any(network.values()),
        )
        if any(branches[name]["status"] != "completed" for name in BRANCHES):
            update_stage(paths, "reflect_parallel", "partial")
            _remove_unpublished_context(paths)
            update_manifest(
                paths,
                status="partial",
                completed_at=utc_now(),
                error="At least one reflection branch is partial",
            )
            update_manifest(paths, outputs=collect_output_hashes(paths))
            return load_json(paths.manifest)
        update_stage(paths, "reflect_parallel", "completed")
        validations = {
            branch: validate_branch(
                config_path=config_path,
                parent_root=paths.root,
                run_id=branch_run_ids(args.run_id)[branch],
                branch=branch,
            )
            for branch in BRANCHES
        }
        atomic_write_json(paths.audit / "branch-validation.json", validations)
        update_stage(paths, "validate_reflections", "completed")

        branches = _branch_states(paths)
        provenance = _provenance(paths, args.run_id, branches)
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
        metrics = {name: branches[name]["metrics"] for name in BRANCHES}
        summary = context_summary(contexts_out, metrics)
        atomic_write_json(paths.context / "summary.json", summary)
        manual = [
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
        atomic_write_jsonl(paths.audit / "manual_review.jsonl", manual)
        update_stage(paths, "diagnostics", "completed")
        update_manifest(
            paths,
            status="completed",
            completed_at=utc_now(),
            branches=branches,
            summary=summary,
            error=None,
        )
        update_manifest(paths, outputs=collect_output_hashes(paths))
        return load_json(paths.manifest)
    except Exception as error:
        _remove_unpublished_context(paths)
        current = load_json(paths.manifest)
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
        print(f"RunCollaborativeReflection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
