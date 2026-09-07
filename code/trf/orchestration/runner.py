"""Rerun every implemented TRF stage under one auditable parent manifest."""

from __future__ import annotations

import argparse
import copy
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import (  # noqa: E402
    atomic_write_json,
    load_json,
    resolve_project_path,
    sha256_file,
    utc_now,
)
from common.contracts import load_demonstration_set  # noqa: E402
from trf.offline.assign_pseudo_trfs import assign_pseudo_trfs  # noqa: E402
from trf.offline.build_corpus import write_corpus  # noqa: E402
from trf.offline.extract_candidates import write_candidate_bank  # noqa: E402
from trf.offline.validator import validate_completed_run  # noqa: E402
from trf.offline.common import (  # noqa: E402
    execute_stage,
    finalize_run,
    initialize_run,
    load_trf_config,
    mark_failed,
    read_manifest,
    run_paths as offline_run_paths,
)
from trf.orchestration.common import (  # noqa: E402
    FullTRFError,
    FullTRFPaths,
    collect_output_hashes,
    full_paths,
    initialize_or_resume,
    update_manifest,
    update_stage,
)
from trf.target.runner import run as run_target  # noqa: E402
from trf.target.validator import validate as validate_target  # noqa: E402
from trf.target.common import (  # noqa: E402
    load_target_config,
    target_run_paths,
)


OfflineExecutor = Callable[[Path, str | Path | None, str, bool, list[str]], None]
OfflineValidator = Callable[[dict[str, Any], str], dict[str, Any]]
TargetValidator = Callable[[Path, str], dict[str, Any]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--demonstrations")
    parser.add_argument("--mode", choices=("independent", "leave-one-out"), required=True)
    parser.add_argument("--input")
    parser.add_argument(
        "--feedback",
        help="Optional neutral JSON/JSONL feedback passed to the target mode",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--reuse-embeddings-from")
    parser.add_argument("--config", default="config/trf.json")
    return parser.parse_args(argv)


def execute_offline_pipeline(
    config_path: Path,
    demonstrations: str | Path | None,
    run_id: str,
    allow_model_download: bool,
    command: list[str],
) -> None:
    config = load_trf_config(config_path, demonstrations)
    paths = None
    try:
        paths = initialize_run(config_path, config, run_id, command)
        execute_stage(
            config,
            paths,
            "build_corpus",
            None,
            lambda: write_corpus(config, paths),
        )
        execute_stage(
            config,
            paths,
            "extract_candidates",
            "build_corpus",
            lambda: write_candidate_bank(config, paths),
        )
        execute_stage(
            config,
            paths,
            "assign_pseudo_trfs",
            "extract_candidates",
            lambda: assign_pseudo_trfs(config, paths, allow_model_download),
        )
        finalize_run(config, paths)
    except Exception as error:
        if paths is not None and paths.manifest.is_file():
            manifest = read_manifest(paths)
            if manifest.get("status") != "failed":
                active = next(
                    (
                        name
                        for name, details in manifest["stages"].items()
                        if details["status"] == "running"
                    ),
                    None,
                )
                mark_failed(paths, active, error)
        raise


def build_generated_target_config(
    template: dict[str, Any],
    offline_config: dict[str, Any],
    offline_run_id: str,
) -> dict[str, Any]:
    paths = offline_run_paths(offline_config, offline_run_id)
    required = {
        "manifest": paths.manifest,
        "corpus": paths.corpus / "records.jsonl",
        "candidates": paths.candidates / "selected_trfs.json",
        "pseudo_trfs": paths.demonstrations / "pseudo_trfs.jsonl",
    }
    for path in required.values():
        if not path.is_file():
            raise FullTRFError(f"Completed offline TRF artifact is missing: {path}")
    manifest = load_json(paths.manifest)
    if manifest.get("status") != "completed" or manifest.get("run_id") != offline_run_id:
        raise FullTRFError("Offline TRF child is not an immutable completed run")

    demonstration_bundle = load_demonstration_set(
        Path(offline_config["source"]["demonstrations"])
    )
    generated = copy.deepcopy(template)
    generated["configuration_role"] = "generated-run-config"
    generated.setdefault("candidates", {})["expected_main_trfs"] = int(
        offline_config["candidates"]["max_trfs"]
    )
    generated["source_trf"] = {
        "run_id": offline_run_id,
        "run_root": str(paths.root),
        "files": {
            name: {
                "path": str(path.relative_to(paths.root).as_posix()),
                "sha256": sha256_file(path),
            }
            for name, path in required.items()
        },
    }
    generated["demonstrations"] = {
        "dataset_id": demonstration_bundle["manifest"]["dataset_id"],
        "expected_count": int(offline_config["source"]["expected_counts"]["total"]),
        "files": {
            "manifest": {
                "path": str(demonstration_bundle["manifest_path"]),
                "sha256": sha256_file(demonstration_bundle["manifest_path"]),
            },
            "records": {
                "path": str(demonstration_bundle["records_path"]),
                "sha256": sha256_file(demonstration_bundle["records_path"]),
            },
        },
    }
    return generated


def _input_descriptor(
    mode: str,
    supplied: str | None,
    target_template: dict[str, Any],
) -> dict[str, Any]:
    if mode == "leave-one-out":
        if supplied:
            raise FullTRFError("--input is only valid in independent mode")
        return {"path": None, "sha256": None}
    raw = supplied or target_template["targets"]["independent_input"]
    path = resolve_project_path(PROJECT_ROOT, raw).resolve()
    if not path.is_file():
        raise FullTRFError(f"Independent target input does not exist: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def _resolve_reuse_id(value: str | None, runs_root: str | Path) -> str | None:
    if value is None:
        return None
    candidate = full_paths(runs_root, value)
    if candidate.manifest.is_file():
        manifest = load_json(candidate.manifest)
        target_id = manifest.get("compatibility", {}).get("target_run_id")
        if not isinstance(target_id, str):
            raise FullTRFError("Referenced complete run has no target child id")
        return target_id
    return value


def _children(paths: FullTRFPaths, offline_root: Path, target_root: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, root in (("offline", offline_root), ("target", target_root)):
        manifest_path = root / "manifest.json"
        values[name] = {
            "root": str(root),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
            "status": load_json(manifest_path).get("status") if manifest_path.is_file() else None,
        }
    return values


def _network(offline_root: Path, target_root: Path) -> dict[str, bool]:
    def called(root: Path) -> bool:
        manifest_path = root / "manifest.json"
        return (
            bool(load_json(manifest_path).get("network_called"))
            if manifest_path.is_file()
            else False
        )

    return {
        "offline_model": called(offline_root),
        "target_qwen": called(target_root),
    }


def _finish(
    paths: FullTRFPaths,
    *,
    status: str,
    offline_root: Path,
    target_root: Path,
    summary: dict[str, Any] | None,
    error: dict[str, str] | None,
) -> None:
    network = _network(offline_root, target_root)
    update_manifest(
        paths,
        status=status,
        completed_at=utc_now() if status in {"completed", "failed"} else None,
        network_called=any(network.values()),
        network=network,
        children=_children(paths, offline_root, target_root),
        outputs=collect_output_hashes(paths),
        summary=summary,
        error=error,
    )


def run(
    args: argparse.Namespace,
    *,
    target_client_factory: Callable[[], Any] | None = None,
    offline_executor: OfflineExecutor = execute_offline_pipeline,
    offline_validator: OfflineValidator = validate_completed_run,
    target_validator: TargetValidator = validate_target,
) -> str:
    if args.retry_failed and not args.resume:
        raise FullTRFError("--retry-failed requires --resume")
    if (
        args.allow_network
        and not args.prepare_only
        and args.limit is None
        and not args.confirm_full_run
    ):
        raise FullTRFError("An unrestricted online rerun requires --confirm-full-run")

    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    offline_config = load_trf_config(config_path, args.demonstrations)
    target_template = load_target_config(config_path)
    input_descriptor = _input_descriptor(args.mode, args.input, target_template)
    feedback_value = getattr(args, "feedback", None)
    feedback_descriptor = (
        _input_descriptor("independent", feedback_value, {"targets": {"independent_input": feedback_value}})
        if feedback_value
        else {"path": None, "sha256": None}
    )
    offline_run_id = args.run_id
    target_run_id = args.run_id
    module_config = load_json(config_path)
    runs_root = module_config["output"]["runs_root"]
    paths = full_paths(runs_root, args.run_id)
    offline_paths = offline_run_paths(offline_config, offline_run_id)
    target_paths = target_run_paths(target_template, target_run_id)
    compatibility = {
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "demonstrations": {
            "manifest": offline_config["source"]["demonstrations"],
            "dataset_id": load_demonstration_set(
                Path(offline_config["source"]["demonstrations"])
            )["manifest"]["dataset_id"],
        },
        "offline_run_id": offline_run_id,
        "target_run_id": target_run_id,
        "mode": args.mode,
        "input": input_descriptor,
        "feedback": feedback_descriptor,
        "limit": args.limit,
    }
    if not paths.root.exists() and (
        offline_paths.root.exists() or target_paths.root.exists()
    ):
        raise FullTRFError("Derived child run-id already exists; choose a new parent run-id")
    initialize_or_resume(
        paths,
        args.run_id,
        compatibility,
        list(sys.argv),
        resume=args.resume,
    )

    try:
        offline_manifest = (
            load_json(offline_paths.manifest) if offline_paths.manifest.is_file() else None
        )
        if offline_manifest is None:
            update_stage(paths, "offline_trf", "running")
            offline_executor(
                config_path,
                args.demonstrations,
                offline_run_id,
                args.allow_model_download,
                list(sys.argv),
            )
        elif offline_manifest.get("status") != "completed":
            raise FullTRFError(
                "Offline TRF child is not completed and cannot be resumed; use a new run-id"
            )
        offline_summary = offline_validator(offline_config, offline_run_id)
        update_stage(paths, "offline_trf", "completed")

        generated = build_generated_target_config(
            target_template, offline_config, offline_run_id
        )
        if paths.target_config.is_file():
            if load_json(paths.target_config) != generated:
                raise FullTRFError("Generated target config changed during resume")
        else:
            atomic_write_json(paths.target_config, generated)
        load_target_config(paths.target_config)
        target_paths = target_run_paths(generated, target_run_id)
        target_manifest = (
            load_json(target_paths.manifest) if target_paths.manifest.is_file() else None
        )
        update_stage(paths, "target_trf", "running")
        if target_manifest is None or target_manifest.get("status") != "completed":
            target_args = Namespace(
                config=str(paths.target_config),
                run_id=target_run_id,
                mode=args.mode,
                input=input_descriptor["path"],
                feedback=feedback_descriptor["path"],
                limit=args.limit,
                prepare_only=args.prepare_only,
                allow_network=args.allow_network,
                resume=target_manifest is not None,
                retry_failed=args.retry_failed,
                reuse_embeddings_from=_resolve_reuse_id(
                    args.reuse_embeddings_from, runs_root
                ),
                confirm_full_run=args.confirm_full_run,
            )
            target_status = run_target(
                target_args, client_factory=target_client_factory
            )
        else:
            target_status = "completed"

        if target_status != "completed":
            update_stage(paths, "target_trf", "partial")
            update_stage(paths, "joint_validation", "pending")
            partial_summary = {
                "status": "partial",
                "offline": offline_summary,
                "target_status": target_status,
            }
            _finish(
                paths,
                status="partial",
                offline_root=offline_paths.root,
                target_root=target_paths.root,
                summary=partial_summary,
                error=None,
            )
            return "partial"

        update_stage(paths, "target_trf", "completed")
        update_stage(paths, "joint_validation", "running")
        target_summary = target_validator(paths.target_config, target_run_id)
        joint_summary = {
            "schema_version": "trf-complete-summary-v1",
            "status": "validated",
            "offline": offline_summary,
            "target": target_summary,
        }
        atomic_write_json(paths.validation, joint_summary)
        update_stage(paths, "joint_validation", "completed")
        _finish(
            paths,
            status="completed",
            offline_root=offline_paths.root,
            target_root=target_paths.root,
            summary=joint_summary,
            error=None,
        )
        return "completed"
    except Exception as error:
        manifest = load_json(paths.manifest)
        active = next(
            (name for name, status in manifest["stages"].items() if status != "completed"),
            None,
        )
        if active:
            update_stage(paths, active, "failed")
        _finish(
            paths,
            status="failed",
            offline_root=offline_paths.root,
            target_root=target_paths.root,
            summary=None,
            error={"type": error.__class__.__name__, "message": str(error)},
        )
        raise


def main() -> int:
    args = parse_args()
    try:
        status = run(args)
    except Exception as error:
        print(f"RunFullTRF failed: {error}", file=sys.stderr)
        return 1
    config = load_json(resolve_project_path(PROJECT_ROOT, args.config))
    print(
        f"Complete TRF rerun status={status}: "
        f"{full_paths(config['output']['runs_root'], args.run_id).root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
