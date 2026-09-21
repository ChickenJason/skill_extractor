"""Read-only reconstruction of prepared or completed discriminator runs."""

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

from common.io_utils import load_json, read_jsonl, resolve_project_path  # noqa: E402
from common.contracts import record_identity  # noqa: E402
from common.skill_prediction import (  # noqa: E402
    build_prediction_failure_records,
    build_prediction_result_records,
    evaluate_result_completion,
    latest_prediction_raw_records,
    merge_validation_issue_records,
    parse_successful_prediction_records,
    prediction_validation_issue_events,
    unavailable_prediction_result,
)
from instance_discriminator.common import (  # noqa: E402
    InstanceDiscriminatorError,
    assert_sources_unchanged,
    collect_output_hashes,
    discriminator_run_paths,
    implementation_hashes,
    latest_records,
    load_discriminator_config,
    load_source_bundle,
    target_descriptor,
)
from instance_discriminator.diagnostics import (  # noqa: E402
    build_diagnostics,
    prepared_summary,
)
from instance_discriminator.online import parse_terminal_raw_records  # noqa: E402
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
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise InstanceDiscriminatorError(f"Validation mismatch: {label}")


def validate(config_path: Path, run_id: str) -> dict[str, Any]:
    config = load_discriminator_config(config_path)
    paths = discriminator_run_paths(config, run_id)
    manifest = load_json(paths.manifest)
    if manifest.get("status") not in {"partial", "completed"}:
        raise InstanceDiscriminatorError("Only prepared or completed runs can pass validation")
    source = manifest.get("compatibility", {}).get("source", {})
    source_snapshot = load_json(paths.root / "source_snapshot.json")
    _equal(source_snapshot, source, "source snapshot")
    assert_sources_unchanged(source_snapshot)
    if manifest.get("source_unchanged") is not True:
        raise InstanceDiscriminatorError("Manifest does not attest source immutability")
    _equal(manifest.get("implementation"), implementation_hashes(), "implementation")
    _equal(collect_output_hashes(paths), manifest.get("outputs"), "output SHA256 ledger")

    descriptor = manifest["compatibility"]["targets"]
    source_files = source_snapshot["files"]
    bundle = load_source_bundle(
        source_files["targets"]["path"],
        source_files["candidates"]["path"],
        source_files.get("features", {}).get("path"),
        descriptor.get("limit"),
    )
    _equal(target_descriptor(bundle["targets"], descriptor.get("limit")), descriptor, "targets")
    expected_candidates = [
        build_candidate_record(
            target,
            bundle["candidates_by_identity"][record_identity(target, "target")],
            bundle["features_by_identity"].get(record_identity(target, "target")),
            config["inputs"]["candidate_count"],
        )
        for target in bundle["targets"]
    ]
    actual_candidates = read_jsonl(paths.candidates / "records.jsonl")
    _equal(actual_candidates, expected_candidates, "candidate records")
    expected_prompts = [
        build_discriminator_prompt(item, config["chat"]["max_prompt_characters"])
        for item in expected_candidates
    ]
    actual_prompts = read_jsonl(paths.prompts / "records.jsonl")
    _equal(actual_prompts, expected_prompts, "prompt records")

    summary = load_json(paths.audit / "summary.json")
    if summary.get("status") == "prepared":
        _equal(manifest.get("status"), "partial", "prepared manifest status")
        _equal(
            manifest.get("stages"),
            {
                "prepare": "completed",
                "discriminate": "pending",
                "predict_target_skills": "pending",
                "diagnostics": "pending",
            },
            "prepared stages",
        )
        _equal(summary, prepared_summary(len(expected_candidates)), "prepared summary")
        _equal(manifest.get("summary"), summary, "manifest prepared summary")
        if manifest.get("network_called") is not False:
            raise InstanceDiscriminatorError("Prepare-only run must not call the network")
        for path in (
            paths.raw / "responses.jsonl",
            paths.parsed / "judgments.jsonl",
            paths.selected / "records.jsonl",
            paths.prediction / "prompts.jsonl",
            paths.prediction / "raw.jsonl",
            paths.prediction / "records.jsonl",
            paths.prediction / "failures.jsonl",
            paths.prediction / "results.jsonl",
            paths.audit / "validation_issues.jsonl",
            paths.audit / "manual_review.jsonl",
        ):
            if path.exists() and path.stat().st_size:
                raise InstanceDiscriminatorError(
                    f"Prepare-only run has an unexpected downstream artifact: {path}"
                )
        return {
            "run_id": run_id,
            "status": "valid_prepared",
            "targets": len(expected_candidates),
            "candidate_count_per_target": config["inputs"]["candidate_count"],
            "network_called": False,
        }

    if manifest.get("status") not in {"completed", "partial"}:
        raise InstanceDiscriminatorError("Online run must be completed or partial")
    latest_raw = latest_records(paths.raw / "responses.jsonl")
    expected_ids = {item["idx"] for item in expected_candidates}
    if set(latest_raw) != expected_ids or any(
        item.get("status") not in {"complete", "failed"}
        for item in latest_raw.values()
    ):
        raise InstanceDiscriminatorError(
            "Completed run needs one terminal latest raw response per target"
        )
    prompts_by_idx = {item["idx"]: item for item in expected_prompts}
    expected_parsed, judgment_validation_events, judgment_blockers = parse_terminal_raw_records(
        expected_candidates,
        prompts_by_idx,
        latest_raw,
        chat_model=config["chat"]["model"],
    )
    actual_parsed = read_jsonl(paths.parsed / "judgments.jsonl")
    _equal(actual_parsed, expected_parsed, "parsed judgments")
    parsed_by_idx = {item["idx"]: item for item in expected_parsed}
    expected_selected = []
    for item in expected_candidates:
        parsed_item = parsed_by_idx.get(item["idx"])
        if parsed_item is None:
            continue
        if "exemplar_judgment_validation_failed" in parsed_item.get(
            "review_reasons", []
        ):
            expected_selected.append(
                build_empty_selected_record(item, config["chat"]["model"])
            )
        else:
            expected_selected.append(
                build_selected_record(
                    item,
                    parsed_item["judgments"],
                    config["gate"],
                    config["chat"]["model"],
                )
            )
    actual_selected = read_jsonl(paths.selected / "records.jsonl")
    _equal(actual_selected, expected_selected, "selected records")
    expected_prediction_prompts = [
        build_skill_prediction_prompt(
            item, config["chat"]["max_prompt_characters"]
        )
        for item in expected_selected
    ]
    _equal(
        read_jsonl(paths.prediction / "prompts.jsonl"),
        expected_prediction_prompts,
        "skill prediction prompts",
    )
    latest_prediction_raw = latest_prediction_raw_records(
        paths.prediction / "raw.jsonl"
    )
    expected_prediction_ids = {item["idx"] for item in expected_prediction_prompts}
    if set(latest_prediction_raw) != expected_prediction_ids or any(
        item.get("status") not in {"complete", "failed"}
        for item in latest_prediction_raw.values()
    ):
        raise InstanceDiscriminatorError(
            "Completed run needs one terminal prediction per successful judgment"
        )
    expected_predictions = parse_successful_prediction_records(
        expected_prediction_prompts,
        latest_prediction_raw,
        chat_model=config["chat"]["model"],
        maximum_repairs=config["chat"]["prediction_repair_attempts"],
    )
    actual_predictions = read_jsonl(paths.prediction / "records.jsonl")
    _equal(actual_predictions, expected_predictions, "skill predictions")
    expected_prediction_failures = build_prediction_failure_records(
        expected_prediction_prompts,
        latest_prediction_raw,
        maximum_repairs=config["chat"]["prediction_repair_attempts"],
    )
    _equal(
        read_jsonl(paths.prediction / "failures.jsonl"),
        expected_prediction_failures,
        "skill prediction failures",
    )
    prompt_results = build_prediction_result_records(
        expected_prediction_prompts,
        actual_predictions,
        expected_prediction_failures,
        latest_prediction_raw,
    )
    prompt_results_by_idx = {item["idx"]: item for item in prompt_results}
    results = []
    for item in expected_candidates:
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
        expected_candidates,
        [
            *judgment_validation_events,
            *prediction_validation_issue_events(
                prompt_results, stage="exemplar_skill_prediction"
            ),
        ],
    )
    _equal(read_jsonl(paths.prediction / "results.jsonl"), results, "prediction results")
    _equal(
        read_jsonl(paths.audit / "validation_issues.jsonl"),
        validation_issues,
        "validation issue ledger",
    )
    completion = evaluate_result_completion(
        expected_count=len(expected_candidates),
        results=results,
        validation_issues=validation_issues,
        maximum_failure_rate=config["completion"]["max_failure_rate"],
    )
    expected_status = "completed" if completion["within_tolerance"] else "partial"
    _equal(manifest.get("status"), expected_status, "terminal run status")
    expected_stage_status = "completed" if completion["within_tolerance"] else "partial"
    _equal(
        manifest.get("stages"),
        {
            "prepare": "completed",
            "discriminate": expected_stage_status,
            "predict_target_skills": expected_stage_status,
            "diagnostics": expected_stage_status,
        },
        "terminal stages",
    )
    expected_summary, expected_review = build_diagnostics(
        expected_candidates,
        expected_parsed,
        expected_selected,
        latest_raw,
        config["diagnostics"]["review_sample_size"],
        actual_predictions,
        latest_prediction_raw,
        expected_prediction_failures,
        results,
        validation_issues,
        completion,
    )
    _equal(expected_summary.get("status"), "complete" if completion["within_tolerance"] else "partial", "summary status")
    _equal(summary, expected_summary, "summary")
    _equal(read_jsonl(paths.audit / "manual_review.jsonl"), expected_review, "review sample")
    _equal(manifest.get("summary"), expected_summary, "manifest summary")
    return {
        "run_id": run_id,
        "status": f"valid_{expected_status}",
        "targets": len(expected_candidates),
        "selected_total": expected_summary["selection"]["selected_total"],
        "needs_review": expected_summary["needs_review_count"],
        "predictions": len(actual_predictions),
        "allowed_failures": completion["maximum_failure_count"],
        "failed_predictions": sum(item["prediction"] is None for item in results),
        "predicted_spans": sum(len(item["spans"]) for item in actual_predictions),
        "accuracy_improvement": "not_evaluated",
    }


def main() -> int:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    try:
        result = validate(config_path, args.run_id)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"ValidateInstanceDiscriminatorRun failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
