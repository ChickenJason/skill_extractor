"""Read-only reconstruction validator for prepared and completed aggregator runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from aggregator.common import (  # noqa: E402
    AggregatorError,
    aggregator_run_paths,
    assert_sources_unchanged,
    collect_output_hashes,
    compatibility_payload,
    implementation_hashes,
    load_aggregator_config,
    load_records,
    target_descriptor,
)
from aggregator.diagnostics import completed_diagnostics, prepared_summary  # noqa: E402
from aggregator.pipeline import build_aggregator_inputs, build_prompt, expert_indexes  # noqa: E402
from common.io_utils import load_json, read_jsonl, resolve_project_path  # noqa: E402
from common.skill_prediction import (  # noqa: E402
    build_prediction_failure_records,
    build_prediction_result_records,
    evaluate_result_completion,
    latest_prediction_raw_records,
    merge_validation_issue_records,
    parse_successful_prediction_records,
    prediction_validation_issue_events,
)


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AggregatorError(f"Validation mismatch: {label}")


def validate(config_path: Path, run_id: str) -> dict[str, Any]:
    config = load_aggregator_config(config_path)
    paths = aggregator_run_paths(config, run_id)
    manifest = load_json(paths.manifest)
    if manifest.get("status") not in {"partial", "completed"}:
        raise AggregatorError("Only prepared partial or completed runs can pass validation")
    _equal(manifest.get("run_id"), run_id, "run ID")
    _equal(manifest.get("implementation"), implementation_hashes(), "implementation")
    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, dict):
        raise AggregatorError("Manifest has no compatibility contract")
    _equal(compatibility.get("config")["path"], str(config_path.resolve()), "config path")
    from common.contracts import file_snapshot
    _equal(compatibility.get("config"), file_snapshot(config_path), "config snapshot")
    source = load_json(paths.source_snapshot)
    _equal(compatibility.get("source"), source, "source snapshot")
    assert_sources_unchanged(source)
    if manifest.get("source_unchanged") is not True:
        raise AggregatorError("Manifest does not attest immutable sources")
    contexts = load_records(Path(source["files"]["context_records"]["path"]))
    limit = compatibility["targets"].get("limit")
    contexts = contexts[:limit] if limit is not None else contexts
    experts = None
    if source["mode"] == "with_expert_results":
        experts = expert_indexes(
            load_records(Path(source["files"]["trf_predictions"]["path"])),
            load_records(Path(source["files"]["exemplar_predictions"]["path"])),
        )
    inputs = build_aggregator_inputs(
        contexts,
        mode=source["mode"],
        maximum_examples=config["inputs"]["max_examples"],
        experts=experts,
    )
    prompts = [
        build_prompt(item, context, maximum_characters=config["chat"]["max_prompt_characters"])
        for item, context in zip(inputs, contexts)
    ]
    descriptor = target_descriptor(inputs, limit)
    _equal(compatibility.get("targets"), descriptor, "targets")
    _equal(
        compatibility,
        compatibility_payload(config_path, config, source, source["mode"], descriptor),
        "compatibility contract",
    )
    _equal(read_jsonl(paths.inputs), inputs, "normalized inputs")
    _equal(read_jsonl(paths.prompts), prompts, "prompts")
    stages = manifest.get("stages", {})
    if set(stages) != {"prepare", "predict", "diagnostics"} or stages["prepare"] != "completed":
        raise AggregatorError("Invalid stage ledger")

    has_terminal_artifacts = (
        paths.prediction.exists() or paths.failures.exists() or paths.results.exists()
    )
    if manifest["status"] == "partial" and not has_terminal_artifacts:
        expected_summary = prepared_summary(inputs)
        _equal(load_json(paths.summary), expected_summary, "prepared summary")
        _equal(manifest.get("summary"), expected_summary, "manifest prepared summary")
    else:
        if manifest["status"] == "completed" and any(
            state != "completed" for state in stages.values()
        ):
            raise AggregatorError("Completed validation requires every stage to be completed")
        latest_raw = latest_prediction_raw_records(paths.raw)
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
        _equal(read_jsonl(paths.prediction), predictions, "predictions")
        _equal(read_jsonl(paths.failures), failures, "prediction failures")
        results = build_prediction_result_records(prompts, predictions, failures, latest_raw)
        expected_records = [
            {
                "dataset_id": item["dataset_id"],
                "record_id": item["record_id"],
                "source_sha256": item["source_sha256"],
                "idx": item["idx"],
                "sentence": item["target_sentence"],
            }
            for item in inputs
        ]
        validation_issues = merge_validation_issue_records(
            expected_records,
            prediction_validation_issue_events(results, stage="aggregator_prediction"),
        )
        _equal(read_jsonl(paths.results), results, "prediction results")
        _equal(
            read_jsonl(paths.validation_issues),
            validation_issues,
            "validation issue ledger",
        )
        completion = evaluate_result_completion(
            expected_count=len(prompts),
            results=results,
            validation_issues=validation_issues,
            maximum_failure_rate=config["completion"]["max_failure_rate"],
        )
        if manifest["status"] == "completed" and not completion["within_tolerance"]:
            raise AggregatorError("Completed run exceeds strict failure tolerance")
        if manifest["status"] == "partial" and completion["within_tolerance"]:
            raise AggregatorError("A terminal run within tolerance must be completed")
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
        _equal(load_json(paths.summary), summary, "audit summary")
        _equal(read_jsonl(paths.manual_review), review, "manual review")
        _equal(manifest.get("summary"), summary, "manifest summary")
    _equal(collect_output_hashes(paths), manifest.get("outputs"), "SHA256 output ledger")
    return {
        "run_id": run_id,
        "mode": source["mode"],
        "status": "valid",
        "run_status": manifest["status"],
        "targets": len(inputs),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/aggregator.json")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        result = validate(resolve_project_path(PROJECT_ROOT, args.config).resolve(), args.run_id)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"ValidateAggregator failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
