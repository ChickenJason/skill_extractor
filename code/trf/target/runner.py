"""Run target retrieval, open TRF extraction, and TRF-guided Skill prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import (  # noqa: E402
    MissingEnvironmentVariable,
    atomic_write_json,
    atomic_write_jsonl,
    expand_environment_references,
    load_json,
    redact_secrets,
    resolve_project_path,
    utc_now,
)
from common.qwen_client import QwenClient, QwenRequestError, QwenSettings  # noqa: E402
from common.skill_prediction import (  # noqa: E402
    SkillPredictionNetworkRequired,
    build_prediction_failure_records,
    build_prediction_result_records,
    evaluate_result_completion,
    merge_validation_issue_records,
    parse_successful_prediction_records,
    prediction_validation_issue_events,
    run_skill_predictions,
    unavailable_prediction_result,
)
from trf.target.common import (  # noqa: E402
    TargetRunPaths,
    TargetTRFError,
    assert_sources_unchanged,
    collect_output_hashes,
    initialize_or_resume_run,
    latest_records,
    load_target_config,
    stable_target_descriptor,
    target_run_paths,
    update_manifest,
    update_stage,
)
from trf.target.diagnostics import build_diagnostics  # noqa: E402
from trf.target.online import (  # noqa: E402
    NetworkRequiredError,
    ensure_embeddings,
    load_reuse_embeddings,
    parse_terminal_raw_records,
    run_two_turn_extraction,
)
from trf.target.pipeline import (  # noqa: E402
    build_prompts,
    build_skill_prediction_prompt,
    build_targets,
    load_source_bundle,
    retrieve_demonstrations,
    sentence_hash,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/trf.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("independent", "leave-one-out"), required=True)
    parser.add_argument("--input")
    parser.add_argument(
        "--feedback",
        help="Optional neutral JSON/JSONL feedback; locked for future interaction but not consumed yet",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--reuse-embeddings-from")
    parser.add_argument("--confirm-full-run", action="store_true")
    return parser.parse_args(argv)


def _target_input_path(
    config: dict[str, Any], mode: str, supplied: str | None
) -> Path | None:
    if mode != "independent":
        if supplied:
            raise TargetTRFError("--input is only valid in independent mode")
        return None
    raw = supplied or config["targets"]["independent_input"]
    path = resolve_project_path(PROJECT_ROOT, raw).resolve()
    if not path.is_file():
        raise TargetTRFError(f"Independent target input does not exist: {path}")
    return path


def _client_factory(
    config: dict[str, Any], secret_values: list[str]
) -> Callable[[], QwenClient]:
    def create() -> QwenClient:
        provider = expand_environment_references(config["provider"])
        api_key = provider["api_key"]
        if api_key not in secret_values:
            secret_values.append(api_key)
        settings = QwenSettings(
            chat_model=config["chat"]["model"],
            embedding_model=config["embedding"]["model"],
            base_url=provider["base_url"],
            enable_thinking=config["chat"]["enable_thinking"],
            structured_output=config["chat"]["structured_output"],
            timeout_seconds=config["chat"]["timeout_seconds"],
            max_retries=config["chat"]["max_retries"],
        )
        return QwenClient(api_key=api_key, settings=settings)

    return create


def _set_network_called(paths: TargetRunPaths) -> None:
    manifest = load_json(paths.manifest)
    if manifest.get("network_called") is not True:
        update_manifest(paths, network_called=True)


def _reuse_root(config: dict[str, Any], run_id: str | None, current: str) -> Path | None:
    if run_id is None:
        return None
    if run_id == current:
        raise TargetTRFError("--reuse-embeddings-from must name a different run")
    return target_run_paths(config, run_id).root


def _write_terminal_manifest(
    config: dict[str, Any],
    paths: TargetRunPaths,
    source_snapshot: dict[str, Any],
    *,
    status: str,
    summary: dict[str, Any] | None,
    error: dict[str, str] | None,
) -> None:
    assert_sources_unchanged(config, source_snapshot)
    update_manifest(
        paths,
        status=status,
        completed_at=utc_now() if status in {"completed", "failed"} else None,
        source_unchanged=True,
        outputs=collect_output_hashes(paths),
        summary=summary,
        error=error,
    )


def _validate_reuse_run(
    paths: TargetRunPaths,
    reuse_root: Path | None,
) -> dict[str, Any] | None:
    if reuse_root is None:
        return None
    reuse_manifest_path = reuse_root / "manifest.json"
    if not reuse_manifest_path.is_file():
        raise TargetTRFError(f"Embedding reuse run has no manifest: {reuse_root}")
    current = load_json(paths.manifest)["compatibility"]
    reused = load_json(reuse_manifest_path).get("compatibility", {})
    if reused.get("embedding") != current["embedding"]:
        raise TargetTRFError("Embedding reuse run uses a different embedding contract")
    reused_corpus_hash = (
        reused.get("source", {}).get("files", {}).get("corpus", {}).get("sha256")
    )
    current_corpus_hash = current["source"]["files"]["corpus"]["sha256"]
    if reused_corpus_hash != current_corpus_hash:
        raise TargetTRFError("Embedding reuse run uses a different source corpus hash")
    return reused


def _recoverable_error(error: Exception) -> bool:
    return isinstance(
        error,
        (
            NetworkRequiredError,
            SkillPredictionNetworkRequired,
            MissingEnvironmentVariable,
            QwenRequestError,
        ),
    )


def run(args: argparse.Namespace, *, client_factory: Callable[[], QwenClient] | None = None) -> str:
    if args.retry_failed and not args.resume:
        raise TargetTRFError("--retry-failed requires --resume")
    if (
        args.allow_network
        and not args.prepare_only
        and args.limit is None
        and not args.confirm_full_run
    ):
        raise TargetTRFError(
            "An unrestricted online extraction requires --confirm-full-run"
        )

    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    config = load_target_config(config_path)
    if config["configuration_role"] != "generated-run-config":
        raise TargetTRFError(
            "config/trf.json is a module template; use scripts/trf/run.ps1 "
            "or pass a generated target config"
        )
    bundle = load_source_bundle(config)
    input_path = _target_input_path(config, args.mode, args.input)
    independent_value = load_json(input_path) if input_path else None
    targets = build_targets(args.mode, bundle, independent_value, args.limit)
    descriptor = stable_target_descriptor(args.mode, targets, input_path, args.limit)
    paths = initialize_or_resume_run(
        config_path,
        config,
        args.run_id,
        descriptor,
        list(sys.argv),
        resume=args.resume,
        feedback_path=getattr(args, "feedback", None),
    )
    source_snapshot = load_json(paths.root / "source_snapshot.json")
    secret_values: list[str] = []
    provider_factory = client_factory or _client_factory(config, secret_values)
    target_path = paths.targets / "records.jsonl"
    demo_embedding_path = paths.embeddings / "demonstrations.jsonl"
    target_embedding_path = paths.embeddings / "targets.jsonl"
    retrieval_path = paths.retrieval / "records.jsonl"
    prompt_path = paths.prompts / "records.jsonl"
    raw_path = paths.raw / "responses.jsonl"
    parsed_path = paths.parsed / "records.jsonl"
    prediction_prompt_path = paths.prediction / "prompts.jsonl"
    prediction_raw_path = paths.prediction / "raw.jsonl"
    prediction_path = paths.prediction / "records.jsonl"
    prediction_failure_path = paths.prediction / "failures.jsonl"
    prediction_results_path = paths.prediction / "results.jsonl"
    validation_issues_path = paths.audit / "validation_issues.jsonl"
    summary_path = paths.audit / "summary.json"
    review_path = paths.audit / "manual_review.jsonl"

    try:
        reuse_root = _reuse_root(config, args.reuse_embeddings_from, args.run_id)
        reuse_compatibility = _validate_reuse_run(paths, reuse_root)
        atomic_write_jsonl(target_path, targets)
        update_stage(paths, "prepare_targets", "completed")

        embedding = config["embedding"]
        reused_demos = load_reuse_embeddings(
            reuse_root,
            "demonstrations.jsonl",
            bundle["demonstrations"],
            embedding["model"],
            embedding["dimensions"],
        )
        demo_vectors = ensure_embeddings(
            bundle["demonstrations"],
            demo_embedding_path,
            model=embedding["model"],
            dimensions=embedding["dimensions"],
            batch_size=embedding["batch_size"],
            allow_network=args.allow_network,
            client_factory=provider_factory,
            reused=reused_demos,
            reuse_label=args.reuse_embeddings_from,
            on_network_call=lambda: _set_network_called(paths),
        )
        demo_cache = latest_records(demo_embedding_path)
        demos_by_sentence = {
            sentence_hash(item["sentence"]): demo_cache[item["idx"]]
            for item in bundle["demonstrations"]
        }
        same_targets = (
            reuse_compatibility is not None
            and reuse_compatibility.get("targets", {}).get("records_sha256")
            == descriptor["records_sha256"]
        )
        reused_targets = (
            load_reuse_embeddings(
                reuse_root,
                "targets.jsonl",
                targets,
                embedding["model"],
                embedding["dimensions"],
            )
            if same_targets
            else {}
        )
        target_vectors = ensure_embeddings(
            targets,
            target_embedding_path,
            model=embedding["model"],
            dimensions=embedding["dimensions"],
            batch_size=embedding["batch_size"],
            allow_network=args.allow_network,
            client_factory=provider_factory,
            reused=reused_targets,
            same_run_by_sentence=demos_by_sentence,
            reuse_label=args.reuse_embeddings_from,
            on_network_call=lambda: _set_network_called(paths),
        )

        retrieval_records = [
            retrieve_demonstrations(
                target,
                bundle["demonstrations"],
                target_vectors[target["idx"]],
                demo_vectors,
                leave_one_out=args.mode == "leave-one-out",
                nearest_neighbors=config["retrieval"]["nearest_neighbors"],
                selected_count=config["retrieval"]["demonstrations"],
                similarity_decimals=config["retrieval"]["similarity_decimals"],
            )
            for target in targets
        ]
        retrieval_by_idx = {record["idx"]: record for record in retrieval_records}
        prompt_records = [
            build_prompts(
                target,
                retrieval_by_idx[target["idx"]],
                config["chat"]["max_prompt_characters"],
            )
            for target in targets
        ]
        prompts_by_idx = {record["idx"]: record for record in prompt_records}
        atomic_write_jsonl(retrieval_path, retrieval_records)
        atomic_write_jsonl(prompt_path, prompt_records)
        update_stage(paths, "embed_and_retrieve", "completed")

        if args.prepare_only:
            update_stage(paths, "extract_target_trfs", "pending")
            update_stage(paths, "predict_target_skills", "pending")
            update_stage(paths, "diagnostics", "pending")
            prepared_summary = {
                "status": "prepared",
                "target_count": len(targets),
                "retrieval_count": len(retrieval_records),
                "online_extraction_performed": False,
                "target_skill_prediction_performed": False,
            }
            atomic_write_json(summary_path, prepared_summary)
            _write_terminal_manifest(
                config,
                paths,
                source_snapshot,
                status="partial",
                summary=prepared_summary,
                error=None,
            )
            return "partial"

        latest_raw = run_two_turn_extraction(
            targets,
            prompts_by_idx,
            raw_path,
            main_bank=bundle["main_trfs"],
            embedding_model=embedding["model"],
            chat_model=config["chat"]["model"],
            temperature=config["chat"]["temperature"],
            stage1_max_tokens=config["chat"]["stage1_max_tokens"],
            stage2_max_tokens=config["chat"]["stage2_max_tokens"],
            allow_network=args.allow_network,
            retry_failed=args.retry_failed,
            client_factory=provider_factory,
            secrets=secret_values,
            on_network_call=lambda: _set_network_called(paths),
        )
        parsed, extraction_validation_events, extraction_blockers = parse_terminal_raw_records(
            targets,
            latest_raw,
            bundle["main_trfs"],
            embedding["model"],
            config["chat"]["model"],
        )
        atomic_write_jsonl(parsed_path, parsed)
        extraction_ready = len(parsed) == len(targets) and not extraction_blockers
        parsed_by_idx = {item["idx"]: item for item in parsed}
        prediction_prompts = [
            build_skill_prediction_prompt(
                target,
                parsed_by_idx[target["idx"]],
                config["chat"]["max_prompt_characters"],
            )
            for target in targets
            if target["idx"] in parsed_by_idx
        ]
        atomic_write_jsonl(prediction_prompt_path, prediction_prompts)
        latest_prediction_raw = run_skill_predictions(
            prediction_prompts,
            prediction_raw_path,
            chat_model=config["chat"]["model"],
            temperature=config["chat"]["temperature"],
            max_tokens=config["chat"]["stage3_max_tokens"],
            maximum_repairs=config["chat"]["prediction_repair_attempts"],
            allow_network=args.allow_network,
            retry_failed=args.retry_failed,
            client_factory=provider_factory,
            secrets=secret_values,
            on_network_call=lambda: _set_network_called(paths),
        )
        predictions = parse_successful_prediction_records(
            prediction_prompts,
            latest_prediction_raw,
            chat_model=config["chat"]["model"],
            maximum_repairs=config["chat"]["prediction_repair_attempts"],
        )
        atomic_write_jsonl(prediction_path, predictions)
        prediction_failures = build_prediction_failure_records(
            prediction_prompts,
            latest_prediction_raw,
            maximum_repairs=config["chat"]["prediction_repair_attempts"],
        )
        atomic_write_jsonl(prediction_failure_path, prediction_failures)
        prompt_results = build_prediction_result_records(
            prediction_prompts,
            predictions,
            prediction_failures,
            latest_prediction_raw,
        )
        prompt_results_by_idx = {item["idx"]: item for item in prompt_results}
        results = []
        for target in targets:
            if target["idx"] in prompt_results_by_idx:
                results.append(prompt_results_by_idx[target["idx"]])
                continue
            blocker = extraction_blockers.get(target["idx"])
            results.append(
                unavailable_prediction_result(
                    target,
                    branch="trf",
                    outcome="runtime_failed" if blocker else "missing",
                    failure_kind=(blocker or {}).get("failure_kind", "missing_prediction_prompt"),
                    review_reasons=["trf_extraction_failed_or_missing"],
                    failure=blocker,
                )
            )
        validation_issues = merge_validation_issue_records(
            targets,
            [
                *extraction_validation_events,
                *prediction_validation_issue_events(
                    prompt_results, stage="trf_skill_prediction"
                ),
            ],
        )
        atomic_write_jsonl(prediction_results_path, results)
        atomic_write_jsonl(validation_issues_path, validation_issues)
        prediction_policy = evaluate_result_completion(
            expected_count=len(targets),
            results=results,
            validation_issues=validation_issues,
            maximum_failure_rate=config["chat"]["prediction_max_failure_rate"],
        )
        prediction_complete = prediction_policy["within_tolerance"]
        update_stage(
            paths,
            "extract_target_trfs",
            "completed" if extraction_ready and prediction_complete else "partial",
        )
        update_stage(
            paths,
            "predict_target_skills",
            "completed" if prediction_complete else "partial",
        )
        summary, review = build_diagnostics(
            args.mode,
            targets,
            parsed,
            retrieval_by_idx,
            bundle,
            config["diagnostics"]["review_sample_size"],
            predictions,
            latest_prediction_raw,
            prediction_failures,
            results,
            validation_issues,
            prediction_policy,
        )
        atomic_write_json(summary_path, summary)
        atomic_write_jsonl(review_path, review)
        complete = prediction_complete
        update_stage(paths, "diagnostics", "completed" if complete else "partial")
        final_status = "completed" if complete else "partial"
        _write_terminal_manifest(
            config,
            paths,
            source_snapshot,
            status=final_status,
            summary=summary,
            error=None,
        )
        return final_status
    except Exception as error:
        recoverable = _recoverable_error(error)
        current = load_json(paths.manifest)
        active_stage = next(
            (name for name, value in current["stages"].items() if value != "completed"),
            None,
        )
        if active_stage:
            update_stage(paths, active_stage, "partial" if recoverable else "failed")
        error_value = redact_secrets(
            {"type": error.__class__.__name__, "message": str(error)},
            secret_values,
        )
        try:
            _write_terminal_manifest(
                config,
                paths,
                source_snapshot,
                status="partial" if recoverable else "failed",
                summary=None,
                error=error_value,
            )
        except Exception as source_error:
            update_manifest(
                paths,
                status="failed",
                completed_at=utc_now(),
                source_unchanged=False,
                outputs=collect_output_hashes(paths),
                summary=None,
                error={
                    "type": source_error.__class__.__name__,
                    "message": str(source_error),
                    "prior_error": error_value,
                },
            )
            raise source_error from error
        if recoverable:
            return "partial"
        raise


def main() -> int:
    args = parse_args()
    try:
        status = run(args)
    except Exception as error:
        print(f"RunTargetTRF failed: {error}", file=sys.stderr)
        return 1
    root = target_run_paths(
        load_target_config(resolve_project_path(PROJECT_ROOT, args.config).resolve()),
        args.run_id,
    ).root
    print(f"Target TRF pipeline status={status}: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
