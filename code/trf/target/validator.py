"""Fail-closed validation for a completed target TRF extraction run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import load_json, read_jsonl, resolve_project_path  # noqa: E402
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
from trf.target.common import (  # noqa: E402
    TargetTRFError,
    assert_sources_unchanged,
    collect_output_hashes,
    latest_records,
    load_target_config,
    target_run_paths,
)
from trf.target.diagnostics import build_diagnostics  # noqa: E402
from trf.target.online import (  # noqa: E402
    load_embedding_records,
    parse_terminal_raw_records,
)
from trf.target.pipeline import (  # noqa: E402
    build_prompts,
    build_skill_prediction_prompt,
    load_source_bundle,
    retrieve_demonstrations,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/trf.json")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise TargetTRFError(f"Validation mismatch: {label}")


def validate(config_path: Path, run_id: str) -> dict[str, Any]:
    config = load_target_config(config_path)
    paths = target_run_paths(config, run_id)
    manifest = load_json(paths.manifest)
    if manifest.get("status") not in {"partial", "completed"}:
        raise TargetTRFError("Only partial or completed target TRF runs can pass validation")
    source_snapshot = load_json(paths.root / "source_snapshot.json")
    assert_sources_unchanged(config, source_snapshot)
    if manifest.get("source_unchanged") is not True:
        raise TargetTRFError("Manifest does not attest source immutability")
    _equal(collect_output_hashes(paths), manifest.get("outputs"), "output SHA256 ledger")

    bundle = load_source_bundle(config)
    targets = read_jsonl(paths.targets / "records.jsonl")
    descriptor = manifest["compatibility"]["targets"]
    _equal(len(targets), descriptor["record_count"], "target count")
    _equal([item["idx"] for item in targets], descriptor["indexes"], "target indexes")

    embedding = config["embedding"]
    demo_cache = load_embedding_records(
        paths.embeddings / "demonstrations.jsonl",
        bundle["demonstrations"],
        embedding["model"],
        embedding["dimensions"],
    )
    target_cache = load_embedding_records(
        paths.embeddings / "targets.jsonl",
        targets,
        embedding["model"],
        embedding["dimensions"],
    )
    demo_vectors = {idx: item["vector"] for idx, item in demo_cache.items()}
    target_vectors = {idx: item["vector"] for idx, item in target_cache.items()}
    _equal(
        len(demo_vectors),
        len(bundle["demonstrations"]),
        "demonstration embedding count",
    )
    _equal(len(target_vectors), len(targets), "target embedding count")

    mode = descriptor["mode"]
    expected_retrieval = [
        retrieve_demonstrations(
            target,
            bundle["demonstrations"],
            target_vectors[target["idx"]],
            demo_vectors,
            leave_one_out=mode == "leave-one-out",
            nearest_neighbors=config["retrieval"]["nearest_neighbors"],
            selected_count=config["retrieval"]["demonstrations"],
            similarity_decimals=config["retrieval"]["similarity_decimals"],
        )
        for target in targets
    ]
    actual_retrieval = read_jsonl(paths.retrieval / "records.jsonl")
    _equal(actual_retrieval, expected_retrieval, "retrieval records")
    retrieval_by_idx = {item["idx"]: item for item in actual_retrieval}

    expected_prompts = [
        build_prompts(
            target,
            retrieval_by_idx[target["idx"]],
            config["chat"]["max_prompt_characters"],
        )
        for target in targets
    ]
    _equal(read_jsonl(paths.prompts / "records.jsonl"), expected_prompts, "prompts")

    summary_path = paths.audit / "summary.json"
    current_summary = load_json(summary_path)
    if current_summary.get("status") == "prepared":
        expected_prepared = {
            "status": "prepared",
            "target_count": len(targets),
            "retrieval_count": len(expected_retrieval),
            "online_extraction_performed": False,
            "target_skill_prediction_performed": False,
        }
        _equal(manifest.get("status"), "partial", "prepared manifest status")
        _equal(current_summary, expected_prepared, "prepared summary")
        _equal(manifest.get("summary"), expected_prepared, "manifest prepared summary")
        return {
            "run_id": run_id,
            "status": "valid_prepared",
            "mode": mode,
            "targets": len(targets),
        }

    latest_raw = latest_records(paths.raw / "responses.jsonl")
    if set(latest_raw) != {item["idx"] for item in targets}:
        raise TargetTRFError("A terminal run needs one latest raw result per target")
    if any(item.get("status") not in {"complete", "needs_review", "failed"} for item in latest_raw.values()):
        raise TargetTRFError("A terminal run contains a non-terminal raw result")
    expected_parsed, extraction_validation_events, extraction_blockers = parse_terminal_raw_records(
        targets,
        latest_raw,
        bundle["main_trfs"],
        embedding["model"],
        config["chat"]["model"],
    )
    actual_parsed = read_jsonl(paths.parsed / "records.jsonl")
    _equal(actual_parsed, expected_parsed, "parsed TRFs")
    parsed_by_idx = {item["idx"]: item for item in actual_parsed}
    expected_prediction_prompts = [
        build_skill_prediction_prompt(
            target,
            parsed_by_idx[target["idx"]],
            config["chat"]["max_prompt_characters"],
        )
        for target in targets
        if target["idx"] in parsed_by_idx
    ]
    _equal(
        read_jsonl(paths.prediction / "prompts.jsonl"),
        expected_prediction_prompts,
        "skill prediction prompts",
    )
    latest_prediction_raw = latest_prediction_raw_records(
        paths.prediction / "raw.jsonl"
    )
    if set(latest_prediction_raw) != {item["idx"] for item in expected_prediction_prompts}:
        raise TargetTRFError(
            "A terminal run needs one latest skill-prediction result per prompt"
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
    actual_prediction_failures = read_jsonl(
        paths.prediction / "failures.jsonl"
    )
    _equal(
        actual_prediction_failures,
        expected_prediction_failures,
        "skill prediction failures",
    )
    prompt_results = build_prediction_result_records(
        expected_prediction_prompts,
        actual_predictions,
        actual_prediction_failures,
        latest_prediction_raw,
    )
    prompt_results_by_idx = {item["idx"]: item for item in prompt_results}
    results = [
        prompt_results_by_idx.get(target["idx"])
        or unavailable_prediction_result(
            target,
            branch="trf",
            outcome="runtime_failed" if target["idx"] in extraction_blockers else "missing",
            failure_kind=(extraction_blockers.get(target["idx"]) or {}).get(
                "failure_kind", "missing_prediction_prompt"
            ),
            review_reasons=["trf_extraction_failed_or_missing"],
            failure=extraction_blockers.get(target["idx"]),
        )
        for target in targets
    ]
    validation_issues = merge_validation_issue_records(
        targets,
        [
            *extraction_validation_events,
            *prediction_validation_issue_events(
                prompt_results, stage="trf_skill_prediction"
            ),
        ],
    )
    _equal(
        read_jsonl(paths.prediction / "results.jsonl"),
        results,
        "prediction results",
    )
    _equal(
        read_jsonl(paths.audit / "validation_issues.jsonl"),
        validation_issues,
        "validation issue ledger",
    )
    prediction_policy = evaluate_result_completion(
        expected_count=len(targets),
        results=results,
        validation_issues=validation_issues,
        maximum_failure_rate=config["chat"]["prediction_max_failure_rate"],
    )
    expected_status = "completed" if prediction_policy["within_tolerance"] else "partial"
    _equal(manifest.get("status"), expected_status, "terminal run status")
    expected_stage = "completed" if prediction_policy["within_tolerance"] else "partial"
    _equal(
        manifest.get("stages"),
        {
            "prepare_targets": "completed",
            "embed_and_retrieve": "completed",
            "extract_target_trfs": expected_stage,
            "predict_target_skills": expected_stage,
            "diagnostics": expected_stage,
        },
        "terminal stages",
    )
    expected_summary, expected_review = build_diagnostics(
        mode,
        targets,
        actual_parsed,
        retrieval_by_idx,
        bundle,
        config["diagnostics"]["review_sample_size"],
        actual_predictions,
        latest_prediction_raw,
        actual_prediction_failures,
        results,
        validation_issues,
        prediction_policy,
    )
    _equal(current_summary, expected_summary, "summary")
    _equal(read_jsonl(paths.audit / "manual_review.jsonl"), expected_review, "review sample")
    _equal(manifest.get("summary"), expected_summary, "manifest summary")
    return {
        "run_id": run_id,
        "status": "valid",
        "run_status": expected_status,
        "mode": mode,
        "targets": len(targets),
        "retrieval_k": config["retrieval"]["nearest_neighbors"],
        "prompt_demonstrations": config["retrieval"]["demonstrations"],
        "predictions": len(actual_predictions),
        "prediction_failures": len(actual_prediction_failures),
        "prediction_failure_rate": prediction_policy["validation_issue_rate"],
        "predicted_spans": sum(len(item["spans"]) for item in actual_predictions),
        "semantic_acceptance": "pending_manual_review",
    }


def main() -> int:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    try:
        result = validate(config_path, args.run_id)
    except Exception as error:
        print(f"ValidateTargetTRFRun failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
