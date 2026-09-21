"""Run the shared final aggregator in evidence-only or expert-aware mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from aggregator.common import (  # noqa: E402
    AggregatorError,
    AggregatorRunPaths,
    aggregator_run_paths,
    assert_sources_unchanged,
    collect_output_hashes,
    initialize_or_resume_run,
    load_aggregator_config,
    load_records,
    normalize_mode,
    snapshot_sources,
    target_descriptor,
    update_manifest,
    update_stage,
)
from aggregator.diagnostics import completed_diagnostics, prepared_summary  # noqa: E402
from aggregator.pipeline import build_aggregator_inputs, build_prompt, expert_indexes  # noqa: E402
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
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/aggregator.json")
    parser.add_argument("--mode", required=True, choices=["EvidenceOnly", "WithExpertResults", "evidence_only", "with_expert_results"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--context-manifest", required=True)
    parser.add_argument("--context-records", required=True)
    parser.add_argument("--trf-predictions")
    parser.add_argument("--exemplar-predictions")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args(argv)


def _client_factory(config: dict[str, Any], secrets: list[str]) -> Callable[[], QwenClient]:
    def create() -> QwenClient:
        provider = expand_environment_references(config["provider"])
        api_key = provider["api_key"]
        if api_key not in secrets:
            secrets.append(api_key)
        return QwenClient(
            api_key=api_key,
            settings=QwenSettings(
                chat_model=config["chat"]["model"],
                base_url=provider["base_url"],
                enable_thinking=config["chat"]["enable_thinking"],
                structured_output=config["chat"]["structured_output"],
                timeout_seconds=config["chat"]["timeout_seconds"],
                max_retries=config["chat"]["max_retries"],
            ),
        )

    return create


def _set_network_called(paths: AggregatorRunPaths) -> None:
    if load_json(paths.manifest).get("network_called") is not True:
        update_manifest(paths, network_called=True)


def _terminal_manifest(
    paths: AggregatorRunPaths,
    source: dict[str, Any],
    *,
    status: str,
    summary: dict[str, Any] | None,
    error: dict[str, Any] | None,
) -> None:
    assert_sources_unchanged(source)
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
        (SkillPredictionNetworkRequired, MissingEnvironmentVariable, QwenRequestError),
    )


def _source_records(source: dict[str, Any], limit: int | None) -> tuple[list[dict[str, Any]], Any]:
    if limit is not None and limit < 1:
        raise AggregatorError("--limit must be a positive integer")
    contexts = load_records(Path(source["files"]["context_records"]["path"]))
    contexts = contexts[:limit] if limit is not None else contexts
    if not contexts:
        raise AggregatorError("Aggregator input contains no target contexts")
    mode = source["mode"]
    experts = None
    if mode == "with_expert_results":
        experts = expert_indexes(
            load_records(Path(source["files"]["trf_predictions"]["path"])),
            load_records(Path(source["files"]["exemplar_predictions"]["path"])),
        )
    return contexts, experts


def run(
    args: argparse.Namespace,
    *,
    client_factory: Callable[[], QwenClient] | None = None,
) -> str:
    if args.retry_failed and not args.resume:
        raise AggregatorError("--retry-failed requires --resume")
    if args.allow_network and not args.prepare_only and args.limit is None and not args.confirm_full_run:
        raise AggregatorError("An unrestricted online run requires --confirm-full-run")
    mode = normalize_mode(args.mode)
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    config = load_aggregator_config(config_path)
    source = snapshot_sources(
        mode,
        args.context_manifest,
        args.context_records,
        args.trf_predictions,
        args.exemplar_predictions,
    )
    contexts, experts = _source_records(source, args.limit)
    inputs = build_aggregator_inputs(
        contexts,
        mode=mode,
        maximum_examples=config["inputs"]["max_examples"],
        experts=experts,
    )
    prompts = [
        build_prompt(item, context, maximum_characters=config["chat"]["max_prompt_characters"])
        for item, context in zip(inputs, contexts)
    ]
    paths = initialize_or_resume_run(
        config_path,
        config,
        source,
        mode,
        args.run_id,
        target_descriptor(inputs, args.limit),
        list(sys.argv),
        resume=args.resume,
    )
    locked_source = load_json(paths.source_snapshot)
    secrets: list[str] = []
    provider_factory = client_factory or _client_factory(config, secrets)
    summary: dict[str, Any] | None = None
    try:
        atomic_write_jsonl(paths.inputs, inputs)
        atomic_write_jsonl(paths.prompts, prompts)
        update_stage(paths, "prepare", "completed")
        if args.prepare_only:
            summary = prepared_summary(inputs)
            atomic_write_json(paths.summary, summary)
            _terminal_manifest(paths, locked_source, status="partial", summary=summary, error=None)
            return "partial"

        latest_raw = run_skill_predictions(
            prompts,
            paths.raw,
            chat_model=config["chat"]["model"],
            temperature=config["chat"]["temperature"],
            max_tokens=config["chat"]["max_tokens"],
            maximum_repairs=config["chat"]["repair_attempts"],
            allow_network=args.allow_network,
            retry_failed=args.retry_failed,
            client_factory=provider_factory,
            secrets=secrets,
            on_network_call=lambda: _set_network_called(paths),
        )
        predictions = parse_successful_prediction_records(
            prompts,
            latest_raw,
            chat_model=config["chat"]["model"],
            maximum_repairs=config["chat"]["repair_attempts"],
        )
        failures = build_prediction_failure_records(
            prompts,
            latest_raw,
            maximum_repairs=config["chat"]["repair_attempts"],
        )
        atomic_write_jsonl(paths.prediction, predictions)
        atomic_write_jsonl(paths.failures, failures)
        results = build_prediction_result_records(
            prompts, predictions, failures, latest_raw
        )
        validation_issues = merge_validation_issue_records(
            [
                {
                    "dataset_id": item["dataset_id"],
                    "record_id": item["record_id"],
                    "source_sha256": item["source_sha256"],
                    "idx": item["idx"],
                    "sentence": item["target_sentence"],
                }
                for item in inputs
            ],
            prediction_validation_issue_events(results, stage="aggregator_prediction"),
        )
        atomic_write_jsonl(paths.results, results)
        atomic_write_jsonl(paths.validation_issues, validation_issues)
        completion = evaluate_result_completion(
            expected_count=len(prompts),
            results=results,
            validation_issues=validation_issues,
            maximum_failure_rate=config["completion"]["max_failure_rate"],
        )
        update_stage(paths, "predict", "completed" if completion["within_tolerance"] else "partial")
        summary, review = completed_diagnostics(
            inputs,
            prompts,
            predictions,
            failures,
            results,
            validation_issues,
            latest_raw,
            completion,
            config["diagnostics"]["review_sample_size"],
        )
        atomic_write_json(paths.summary, summary)
        atomic_write_jsonl(paths.manual_review, review)
        complete = completion["within_tolerance"]
        update_stage(paths, "diagnostics", "completed" if complete else "partial")
        status = "completed" if complete else "partial"
        _terminal_manifest(paths, locked_source, status=status, summary=summary, error=None)
        return status
    except Exception as error:
        recoverable = _recoverable(error)
        manifest = load_json(paths.manifest)
        active = next((name for name, state in manifest["stages"].items() if state != "completed"), None)
        if active:
            update_stage(paths, active, "partial" if recoverable else "failed")
        if recoverable and not paths.summary.exists():
            summary = prepared_summary(inputs)
            atomic_write_json(paths.summary, summary)
        error_value = redact_secrets(
            {"type": error.__class__.__name__, "message": str(error)}, secrets
        )
        try:
            _terminal_manifest(
                paths,
                locked_source,
                status="partial" if recoverable else "failed",
                summary=summary,
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
        config = load_aggregator_config(resolve_project_path(PROJECT_ROOT, args.config).resolve())
        print(f"Aggregator status={status}: {aggregator_run_paths(config, args.run_id).root}")
        return 0
    except Exception as error:
        print(f"RunAggregator failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
