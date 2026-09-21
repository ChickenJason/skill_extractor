"""Run the instance discriminator with explicit targets, candidates, and optional features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
from common.contracts import record_identity  # noqa: E402
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
from instance_discriminator.common import (  # noqa: E402
    DiscriminatorRunPaths,
    InstanceDiscriminatorError,
    assert_sources_unchanged,
    collect_output_hashes,
    discriminator_run_paths,
    initialize_or_resume_run,
    load_discriminator_config,
    load_source_bundle,
    target_descriptor,
    update_manifest,
    update_stage,
)
from instance_discriminator.diagnostics import (  # noqa: E402
    build_diagnostics,
    prepared_summary,
)
from instance_discriminator.online import (  # noqa: E402
    NetworkRequiredError,
    parse_terminal_raw_records,
    run_judgments,
)
from instance_discriminator.pipeline import (  # noqa: E402
    build_candidate_record,
    build_discriminator_prompt,
    build_empty_selected_record,
    build_selected_record,
    build_skill_prediction_prompt,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/instance_discriminator.json")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--features")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    return parser.parse_args(argv)


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
            base_url=provider["base_url"],
            enable_thinking=config["chat"]["enable_thinking"],
            structured_output=config["chat"]["structured_output"],
            timeout_seconds=config["chat"]["timeout_seconds"],
            max_retries=config["chat"]["max_retries"],
        )
        return QwenClient(api_key=api_key, settings=settings)

    return create


def _set_network_called(paths: DiscriminatorRunPaths) -> None:
    if load_json(paths.manifest).get("network_called") is not True:
        update_manifest(paths, network_called=True)


def _write_terminal_manifest(
    paths: DiscriminatorRunPaths,
    source_snapshot: dict[str, Any],
    *,
    status: str,
    summary: dict[str, Any] | None,
    error: dict[str, Any] | None,
) -> None:
    assert_sources_unchanged(source_snapshot)
    update_manifest(
        paths,
        status=status,
        completed_at=utc_now() if status in {"completed", "failed"} else None,
        source_unchanged=True,
        outputs=collect_output_hashes(paths),
        summary=summary,
        error=error,
    )


def _recoverable(error: Exception) -> bool:
    return isinstance(
        error,
        (
            NetworkRequiredError,
            SkillPredictionNetworkRequired,
            MissingEnvironmentVariable,
            QwenRequestError,
        ),
    )


def run(
    args: argparse.Namespace,
    *,
    client_factory: Callable[[], QwenClient] | None = None,
) -> str:
    if args.retry_failed and not args.resume:
        raise InstanceDiscriminatorError("--retry-failed requires --resume")
    if (
        args.allow_network
        and not args.prepare_only
        and args.limit is None
        and not args.confirm_full_run
    ):
        raise InstanceDiscriminatorError(
            "An unrestricted online run requires --confirm-full-run"
        )
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    config = load_discriminator_config(config_path)
    bundle = load_source_bundle(
        args.targets, args.candidates, args.features, args.limit
    )
    descriptor = target_descriptor(bundle["targets"], args.limit)
    paths = initialize_or_resume_run(
        config_path,
        config,
        bundle["snapshot"],
        args.run_id,
        descriptor,
        list(sys.argv),
        resume=args.resume,
        allow_repair_upgrade=args.retry_failed,
    )
    source_snapshot = load_json(paths.root / "source_snapshot.json")
    secret_values: list[str] = []
    provider_factory = client_factory or _client_factory(config, secret_values)
    candidate_path = paths.candidates / "records.jsonl"
    prompt_path = paths.prompts / "records.jsonl"
    raw_path = paths.raw / "responses.jsonl"
    parsed_path = paths.parsed / "judgments.jsonl"
    selected_path = paths.selected / "records.jsonl"
    prediction_prompt_path = paths.prediction / "prompts.jsonl"
    prediction_raw_path = paths.prediction / "raw.jsonl"
    prediction_path = paths.prediction / "records.jsonl"
    prediction_failure_path = paths.prediction / "failures.jsonl"
    prediction_results_path = paths.prediction / "results.jsonl"
    validation_issues_path = paths.audit / "validation_issues.jsonl"
    summary_path = paths.audit / "summary.json"
    review_path = paths.audit / "manual_review.jsonl"

    try:
        candidate_records = [
            build_candidate_record(
                target,
                bundle["candidates_by_identity"][record_identity(target, "target")],
                bundle["features_by_identity"].get(record_identity(target, "target")),
                config["inputs"]["candidate_count"],
            )
            for target in bundle["targets"]
        ]
        prompt_records = [
            build_discriminator_prompt(item, config["chat"]["max_prompt_characters"])
            for item in candidate_records
        ]
        prompts_by_idx = {item["idx"]: item for item in prompt_records}
        atomic_write_jsonl(candidate_path, candidate_records)
        atomic_write_jsonl(prompt_path, prompt_records)
        update_stage(paths, "prepare", "completed")

        if args.prepare_only:
            summary = prepared_summary(len(candidate_records))
            atomic_write_json(summary_path, summary)
            update_stage(paths, "discriminate", "pending")
            update_stage(paths, "predict_target_skills", "pending")
            update_stage(paths, "diagnostics", "pending")
            _write_terminal_manifest(
                paths,
                source_snapshot,
                status="partial",
                summary=summary,
                error=None,
            )
            return "partial"

        latest_raw = run_judgments(
            candidate_records,
            prompts_by_idx,
            raw_path,
            chat_model=config["chat"]["model"],
            temperature=config["chat"]["temperature"],
            max_tokens=config["chat"]["max_tokens"],
            allow_network=args.allow_network,
            retry_failed=args.retry_failed,
            client_factory=provider_factory,
            secrets=secret_values,
            on_network_call=lambda: _set_network_called(paths),
        )
        parsed_records, judgment_validation_events, judgment_blockers = parse_terminal_raw_records(
            candidate_records,
            prompts_by_idx,
            latest_raw,
            chat_model=config["chat"]["model"],
        )
        parsed_by_idx = {item["idx"]: item for item in parsed_records}
        selected_records = []
        for item in candidate_records:
            parsed_item = parsed_by_idx.get(item["idx"])
            if parsed_item is None:
                continue
            if "exemplar_judgment_validation_failed" in parsed_item.get(
                "review_reasons", []
            ):
                selected_records.append(
                    build_empty_selected_record(item, config["chat"]["model"])
                )
            else:
                selected_records.append(
                    build_selected_record(
                        item,
                        parsed_item["judgments"],
                        config["gate"],
                        config["chat"]["model"],
                    )
                )
        atomic_write_jsonl(parsed_path, parsed_records)
        atomic_write_jsonl(selected_path, selected_records)
        prediction_prompts = [
            build_skill_prediction_prompt(
                item, config["chat"]["max_prompt_characters"]
            )
            for item in selected_records
        ]
        atomic_write_jsonl(prediction_prompt_path, prediction_prompts)
        latest_prediction_raw = run_skill_predictions(
            prediction_prompts,
            prediction_raw_path,
            chat_model=config["chat"]["model"],
            temperature=config["chat"]["temperature"],
            max_tokens=config["chat"]["prediction_max_tokens"],
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
        for item in candidate_records:
            if item["idx"] in prompt_results_by_idx:
                results.append(prompt_results_by_idx[item["idx"]])
                continue
            blocker = judgment_blockers.get(item["idx"])
            results.append(
                unavailable_prediction_result(
                    item,
                    branch="exemplar",
                    outcome="runtime_failed" if blocker else "missing",
                    failure_kind=(blocker or {}).get(
                        "failure_kind", "missing_prediction_prompt"
                    ),
                    review_reasons=["exemplar_judgment_failed_or_missing"],
                    failure=blocker,
                )
            )
        validation_issues = merge_validation_issue_records(
            candidate_records,
            [
                *judgment_validation_events,
                *prediction_validation_issue_events(
                    prompt_results, stage="exemplar_skill_prediction"
                ),
            ],
        )
        atomic_write_jsonl(prediction_results_path, results)
        atomic_write_jsonl(validation_issues_path, validation_issues)
        completion = evaluate_result_completion(
            expected_count=len(candidate_records),
            results=results,
            validation_issues=validation_issues,
            maximum_failure_rate=config["completion"]["max_failure_rate"],
        )
        discriminate_complete = not judgment_blockers and completion["within_tolerance"]
        update_stage(
            paths,
            "discriminate",
            "completed" if discriminate_complete else "partial",
        )
        prediction_complete = completion["within_tolerance"]
        update_stage(
            paths,
            "predict_target_skills",
            "completed" if prediction_complete else "partial",
        )
        summary, review = build_diagnostics(
            candidate_records,
            parsed_records,
            selected_records,
            latest_raw,
            config["diagnostics"]["review_sample_size"],
            predictions,
            latest_prediction_raw,
            prediction_failures,
            results,
            validation_issues,
            completion,
        )
        atomic_write_json(summary_path, summary)
        atomic_write_jsonl(review_path, review)
        complete = discriminate_complete and prediction_complete
        update_stage(paths, "diagnostics", "completed" if complete else "partial")
        final_status = "completed" if complete else "partial"
        _write_terminal_manifest(
            paths,
            source_snapshot,
            status=final_status,
            summary=summary,
            error=None,
        )
        return final_status
    except Exception as error:
        recoverable = _recoverable(error)
        current = load_json(paths.manifest)
        active = next(
            (name for name, status in current["stages"].items() if status != "completed"),
            None,
        )
        if active:
            update_stage(paths, active, "partial" if recoverable else "failed")
        error_value = redact_secrets(
            {"type": error.__class__.__name__, "message": str(error)}, secret_values
        )
        try:
            _write_terminal_manifest(
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
        config = load_discriminator_config(
            resolve_project_path(PROJECT_ROOT, args.config).resolve()
        )
        root = discriminator_run_paths(config, args.run_id).root
        print(f"Instance discriminator status={status}: {root}")
        return 0
    except Exception as error:
        print(f"RunInstanceDiscriminator failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
