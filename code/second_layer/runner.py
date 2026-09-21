"""Prepare shared retrieval, run TRF and exemplar branches, and assemble their context."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
from instance_discriminator.validator import validate as validate_exemplar  # noqa: E402
from second_layer.assembler import assemble_context_records, context_summary  # noqa: E402
from second_layer.common import (  # noqa: E402
    SecondLayerError,
    SecondLayerRunPaths,
    assert_compatibility_unchanged,
    assert_snapshots_unchanged,
    child_run_ids,
    child_run_roots,
    collect_output_hashes,
    compatibility_payload,
    file_snapshot,
    freeze_shared_inputs,
    initialize_or_resume_run,
    load_second_layer_config,
    second_layer_run_paths,
    target_descriptor,
    update_manifest,
    update_stage,
)
from trf.orchestration.validator import validate as validate_trf  # noqa: E402


CommandExecutor = Callable[[list[str], Path, str], dict[str, Any]]
ParallelExecutor = Callable[
    [dict[str, tuple[list[str], Path]]], dict[str, dict[str, Any]]
]
ChildValidator = Callable[
    [Path, str, Path, str], dict[str, Any]
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/second_layer.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--target-mode",
        choices=("independent", "leave-one-out"),
        default="independent",
    )
    parser.add_argument("--input")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--reuse-embeddings-from")
    return parser.parse_args(argv)


def _target_input_path(
    trf_config_path: Path,
    mode: str,
    supplied: str | None,
) -> Path | None:
    if mode == "leave-one-out":
        if supplied:
            raise SecondLayerError("--input is only valid in independent mode")
        return None
    trf_config = load_json(trf_config_path)
    try:
        raw = supplied or trf_config["target"]["targets"]["independent_input"]
    except (KeyError, TypeError) as error:
        raise SecondLayerError("TRF config has no default independent target input") from error
    path = resolve_project_path(PROJECT_ROOT, raw).resolve()
    if not path.is_file():
        raise SecondLayerError(f"Independent target input does not exist: {path}")
    return path


def _append_option(command: list[str], name: str, value: Any | None) -> None:
    if value is not None:
        command.extend([name, str(value)])


def build_trf_command(
    args: argparse.Namespace,
    trf_config_path: Path,
    trf_run_id: str,
    input_path: Path | None,
    *,
    prepare_only: bool,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "code" / "trf" / "orchestration" / "runner.py"),
        "--config",
        str(trf_config_path),
        "--run-id",
        trf_run_id,
        "--mode",
        args.target_mode,
    ]
    _append_option(command, "--input", input_path)
    _append_option(command, "--limit", args.limit)
    _append_option(command, "--reuse-embeddings-from", args.reuse_embeddings_from)
    if prepare_only:
        command.append("--prepare-only")
    if args.allow_model_download:
        command.append("--allow-model-download")
    if args.allow_network:
        command.append("--allow-network")
    if args.confirm_full_run:
        command.append("--confirm-full-run")
    if resume:
        command.append("--resume")
    if args.retry_failed and not prepare_only:
        command.append("--retry-failed")
    return command


def build_exemplar_command(
    args: argparse.Namespace,
    exemplar_config_path: Path,
    exemplar_run_id: str,
    targets_path: Path,
    candidates_path: Path,
    *,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "code" / "instance_discriminator" / "runner.py"),
        "--config",
        str(exemplar_config_path),
        "--run-id",
        exemplar_run_id,
        "--targets",
        str(targets_path),
        "--candidates",
        str(candidates_path),
    ]
    _append_option(command, "--limit", args.limit)
    if args.allow_network:
        command.append("--allow-network")
    if args.confirm_full_run:
        command.append("--confirm-full-run")
    if resume:
        command.append("--resume")
    if args.retry_failed:
        command.append("--retry-failed")
    return command


def execute_command(command: list[str], log_path: Path, name: str) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.perf_counter()
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"\n[{started_at}] {name} command: {json.dumps(command, ensure_ascii=False)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            creationflags=creationflags,
        )
    return {
        "name": name,
        "command": command,
        "started_at": started_at,
        "completed_at": utc_now(),
        "duration_seconds": round(time.perf_counter() - started, 6),
        "returncode": completed.returncode,
        "log": str(log_path.resolve()),
    }


def execute_parallel_processes(
    specifications: dict[str, tuple[list[str], Path]],
) -> dict[str, dict[str, Any]]:
    """Start every process before waiting for any process to finish."""

    if not specifications:
        return {}
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    running: dict[str, tuple[subprocess.Popen[Any], Any, float, dict[str, Any]]] = {}
    try:
        for name, (command, log_path) in specifications.items():
            log_path.parent.mkdir(parents=True, exist_ok=True)
            started_at = utc_now()
            started = time.perf_counter()
            log = log_path.open("a", encoding="utf-8", newline="\n")
            log.write(
                f"\n[{started_at}] {name} command: "
                f"{json.dumps(command, ensure_ascii=False)}\n"
            )
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            running[name] = (
                process,
                log,
                started,
                {
                    "name": name,
                    "command": command,
                    "started_at": started_at,
                    "pid": process.pid,
                    "log": str(log_path.resolve()),
                },
            )

        results: dict[str, dict[str, Any]] = {}
        remaining = set(running)
        while remaining:
            finished = []
            for name in sorted(remaining):
                process, log, started, metadata = running[name]
                returncode = process.poll()
                if returncode is None:
                    continue
                log.flush()
                results[name] = {
                    **metadata,
                    "completed_at": utc_now(),
                    "duration_seconds": round(time.perf_counter() - started, 6),
                    "returncode": returncode,
                }
                finished.append(name)
            remaining.difference_update(finished)
            if remaining:
                time.sleep(0.02)
        return results
    finally:
        for process, log, _, _ in running.values():
            if process.poll() is None:
                process.wait()
            log.close()


def _prepared_artifacts(root: Path) -> tuple[Path, Path] | None:
    target_manifest = root / "target" / "manifest.json"
    targets = root / "target" / "targets" / "records.jsonl"
    candidates = root / "target" / "retrieval" / "records.jsonl"
    if not (target_manifest.is_file() and targets.is_file() and candidates.is_file()):
        return None
    manifest = load_json(target_manifest)
    if manifest.get("stages", {}).get("embed_and_retrieve") != "completed":
        return None
    return targets, candidates


def _child_states(roots: dict[str, Path]) -> dict[str, Any]:
    states: dict[str, Any] = {}
    for name, root in roots.items():
        manifest_path = root / "manifest.json"
        manifest = load_json(manifest_path) if manifest_path.is_file() else None
        states[name] = {
            "run_root": str(root),
            "manifest": str(manifest_path),
            "manifest_sha256": file_snapshot(manifest_path)["sha256"] if manifest else None,
            "status": manifest.get("status") if manifest else None,
            "network_called": bool(manifest.get("network_called")) if manifest else False,
        }
    return states


def _update_terminal_manifest(
    paths: SecondLayerRunPaths,
    roots: dict[str, Path],
    *,
    status: str,
    summary: dict[str, Any] | None,
    error: dict[str, Any] | None,
) -> None:
    manifest = load_json(paths.manifest)
    assert_compatibility_unchanged(manifest["compatibility"])
    if manifest.get("shared_inputs") is not None:
        assert_snapshots_unchanged(manifest["shared_inputs"])
    children = _child_states(roots)
    network = {
        "trf": children["trf"]["network_called"],
        "exemplar": children["exemplar"]["network_called"],
    }
    update_manifest(
        paths,
        status=status,
        completed_at=utc_now() if status in {"completed", "failed"} else None,
        network_called=any(network.values()),
        network=network,
        children=children,
        source_unchanged=True,
        outputs=collect_output_hashes(paths),
        summary=summary,
        error=error,
    )


def validate_children(
    trf_config_path: Path,
    trf_run_id: str,
    exemplar_config_path: Path,
    exemplar_run_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": "second-layer-child-validation-v1",
        "trf": validate_trf(trf_run_id, trf_config_path),
        "exemplar": validate_exemplar(exemplar_config_path, exemplar_run_id),
    }


def _provenance(
    paths: SecondLayerRunPaths,
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


def _partial(
    paths: SecondLayerRunPaths,
    roots: dict[str, Path],
    reason: str,
) -> str:
    summary = {
        "schema_version": "second-layer-summary-v1",
        "status": "partial",
        "reason": reason,
    }
    _update_terminal_manifest(paths, roots, status="partial", summary=summary, error=None)
    return "partial"


def _withdraw_context(paths: SecondLayerRunPaths) -> None:
    """Ensure a failed parent never exposes a context package as publishable."""

    for path in (paths.context / "records.jsonl", paths.context / "summary.json"):
        if path.is_file():
            path.unlink()


def run(
    args: argparse.Namespace,
    *,
    command_executor: CommandExecutor = execute_command,
    parallel_executor: ParallelExecutor = execute_parallel_processes,
    child_validator: ChildValidator = validate_children,
) -> str:
    if args.retry_failed and not args.resume:
        raise SecondLayerError("--retry-failed requires --resume")
    if (
        args.allow_network
        and not args.prepare_only
        and args.limit is None
        and not args.confirm_full_run
    ):
        raise SecondLayerError("An unrestricted online run requires --confirm-full-run")

    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    config = load_second_layer_config(config_path)
    trf_config_path = Path(config["children"]["trf_config"])
    exemplar_config_path = Path(config["children"]["instance_discriminator_config"])
    input_path = _target_input_path(trf_config_path, args.target_mode, args.input)
    descriptor = target_descriptor(args.target_mode, input_path, args.limit)
    paths = second_layer_run_paths(config, args.run_id)
    ids = child_run_ids(args.run_id)
    roots = child_run_roots(config, args.run_id)
    compatibility = compatibility_payload(
        config_path, config, args.run_id, descriptor
    )
    initialize_or_resume_run(
        paths,
        run_id=args.run_id,
        compatibility=compatibility,
        command=list(sys.argv),
        resume=args.resume,
        child_roots=roots,
    )

    try:
        manifest = load_json(paths.manifest)
        shared_inputs = manifest.get("shared_inputs")
        if manifest["stages"]["prepare_shared"] == "completed":
            assert_snapshots_unchanged(shared_inputs)
        else:
            prepared = _prepared_artifacts(roots["trf"])
            if prepared is None:
                update_stage(paths, "prepare_shared", "running")
                prepare_command = build_trf_command(
                    args,
                    trf_config_path,
                    ids["trf"],
                    input_path,
                    prepare_only=True,
                    resume=(roots["trf"] / "manifest.json").is_file(),
                )
                result = command_executor(
                    prepare_command, paths.logs / "prepare-trf.log", "prepare_trf"
                )
                processes = load_json(paths.manifest).get("processes", {})
                processes["prepare_trf"] = result
                update_manifest(paths, processes=processes)
                if result.get("returncode") != 0:
                    raise SecondLayerError("TRF shared preparation process failed")
                prepared = _prepared_artifacts(roots["trf"])
            if prepared is None:
                update_stage(paths, "prepare_shared", "partial")
                return _partial(paths, roots, "shared_retrieval_not_ready")

            targets_source, candidates_source = prepared
            targets_snapshot = paths.shared / "targets.jsonl"
            candidates_snapshot = paths.shared / "candidates.jsonl"
            if targets_snapshot.is_file() and candidates_snapshot.is_file():
                if (
                    file_snapshot(targets_snapshot)["sha256"]
                    != file_snapshot(targets_source)["sha256"]
                    or file_snapshot(candidates_snapshot)["sha256"]
                    != file_snapshot(candidates_source)["sha256"]
                ):
                    raise SecondLayerError("Recovered shared snapshots differ from TRF preparation")
                shared_inputs = {
                    "targets": file_snapshot(targets_snapshot),
                    "candidates": file_snapshot(candidates_snapshot),
                }
            elif targets_snapshot.exists() or candidates_snapshot.exists():
                raise SecondLayerError("Only one recovered shared snapshot exists")
            else:
                shared_inputs = freeze_shared_inputs(
                    paths, targets_source, candidates_source
                )
            update_manifest(paths, shared_inputs=shared_inputs)
            update_stage(paths, "prepare_shared", "completed")

        if args.prepare_only:
            return _partial(paths, roots, "prepare_only")

        assert_snapshots_unchanged(shared_inputs)
        states = _child_states(roots)
        specifications: dict[str, tuple[list[str], Path]] = {}
        if states["trf"]["status"] != "completed":
            specifications["trf"] = (
                build_trf_command(
                    args,
                    trf_config_path,
                    ids["trf"],
                    input_path,
                    prepare_only=False,
                    resume=True,
                ),
                paths.logs / "trf.log",
            )
        if states["exemplar"]["status"] != "completed":
            specifications["exemplar"] = (
                build_exemplar_command(
                    args,
                    exemplar_config_path,
                    ids["exemplar"],
                    Path(shared_inputs["targets"]["path"]),
                    Path(shared_inputs["candidates"]["path"]),
                    resume=(roots["exemplar"] / "manifest.json").is_file(),
                ),
                paths.logs / "exemplar.log",
            )
        update_stage(paths, "parallel_inference", "running")
        results = parallel_executor(specifications)
        processes = load_json(paths.manifest).get("processes", {})
        processes.update(results)
        update_manifest(paths, processes=processes)
        failed_processes = [
            name for name, result in results.items() if result.get("returncode") != 0
        ]
        if failed_processes:
            raise SecondLayerError(
                f"Parallel child process failed: {', '.join(sorted(failed_processes))}"
            )

        states = _child_states(roots)
        if any(states[name]["status"] != "completed" for name in states):
            update_stage(paths, "parallel_inference", "partial")
            return _partial(paths, roots, "one_or_more_children_partial")
        update_stage(paths, "parallel_inference", "completed")

        update_stage(paths, "validate_children", "running")
        validation = child_validator(
            trf_config_path,
            ids["trf"],
            exemplar_config_path,
            ids["exemplar"],
        )
        atomic_write_json(paths.validation, validation)
        update_stage(paths, "validate_children", "completed")

        update_stage(paths, "assemble_context", "running")
        targets = read_jsonl(Path(shared_inputs["targets"]["path"]))
        trf_records = read_jsonl(
            roots["trf"] / "target" / "parsed" / "records.jsonl"
        )
        exemplar_records = read_jsonl(
            roots["exemplar"] / "selected" / "records.jsonl"
        )
        provenance = _provenance(paths, roots, ids)
        context = assemble_context_records(
            targets, trf_records, exemplar_records, provenance
        )
        atomic_write_jsonl(paths.context / "records.jsonl", context)
        summary = context_summary(context)
        atomic_write_json(paths.context / "summary.json", summary)
        update_stage(paths, "assemble_context", "completed")
        _update_terminal_manifest(
            paths, roots, status="completed", summary=summary, error=None
        )
        return "completed"
    except Exception as error:
        try:
            _withdraw_context(paths)
            current = load_json(paths.manifest)
            active = next(
                (
                    name
                    for name, status in current["stages"].items()
                    if status == "running"
                ),
                None,
            )
            if active:
                update_stage(paths, active, "failed")
            _update_terminal_manifest(
                paths,
                roots,
                status="failed",
                summary=None,
                error={"type": error.__class__.__name__, "message": str(error)},
            )
        except Exception as terminal_error:
            update_manifest(
                paths,
                status="failed",
                completed_at=utc_now(),
                source_unchanged=False,
                outputs=collect_output_hashes(paths),
                summary=None,
                error={
                    "type": terminal_error.__class__.__name__,
                    "message": str(terminal_error),
                    "prior_error": {
                        "type": error.__class__.__name__,
                        "message": str(error),
                    },
                },
            )
            raise terminal_error from error
        raise


def main() -> int:
    args = parse_args()
    try:
        status = run(args)
        config = load_second_layer_config(
            resolve_project_path(PROJECT_ROOT, args.config).resolve()
        )
        root = second_layer_run_paths(config, args.run_id).root
        print(f"Second-layer parallel pipeline status={status}: {root}")
        return 0
    except Exception as error:
        print(f"RunSecondLayer failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
