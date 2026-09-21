"""Deterministic aggregate audit and manual-review sampling."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any


def prepared_summary(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "aggregator-summary-v2",
        "status": "prepared",
        "mode": inputs[0]["mode"] if inputs else None,
        "target_count": len(inputs),
        "input_record_count": len(inputs),
        "prompt_record_count": len(inputs),
        "prediction_performed": False,
    }


def completed_diagnostics(
    inputs: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    results: list[dict[str, Any]],
    validation_issues: list[dict[str, Any]],
    latest_raw: dict[int, dict[str, Any]],
    completion: dict[str, Any],
    review_sample_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions_by_idx = {item["idx"]: item for item in predictions}
    failures_by_idx = {item["idx"]: item for item in failures}
    results_by_idx = {item["idx"]: item for item in results}
    statuses = Counter(item["status"] for item in predictions)
    expert_inputs = [item for item in inputs if "expert_proposals" in item]
    both_available = [
        item for item in expert_inputs
        if all(value["available"] for value in item["expert_proposals"].values())
    ]
    summary = {
        "schema_version": "aggregator-summary-v2",
        "status": "complete" if completion["within_tolerance"] else "partial",
        "mode": inputs[0]["mode"] if inputs else None,
        "target_count": len(inputs),
        "prediction_count": len(predictions),
        "complete_count": statuses["complete"],
        "needs_review_count": statuses["needs_review"],
        "positive_count": sum(item["has_skill"] == 1 for item in predictions),
        "span_total": sum(len(item["spans"]) for item in predictions),
        "failed_indexes": [item["idx"] for item in failures],
        "result_count": len(results),
        "result_outcomes": dict(sorted(Counter(item["outcome"] for item in results).items())),
        "validation_issue_count": len(validation_issues),
        "validation_issue_indexes": [item["idx"] for item in validation_issues],
        "repair_attempted_count": sum(bool(item.get("repair")) for item in latest_raw.values()),
        "repaired_count": sum(
            item.get("status") == "complete" and bool(item.get("repair"))
            for item in latest_raw.values()
        ),
        "recovered_count": sum(
            item.get("status") == "complete" and bool(item.get("recovery"))
            for item in latest_raw.values()
        ),
        "evidence": {
            "empty_target_trf_count": sum(not item["target_trfs"] for item in inputs),
            "zero_example_count": sum(not item["examples"] for item in inputs),
            "one_example_count": sum(len(item["examples"]) == 1 for item in inputs),
            "example_total": sum(len(item["examples"]) for item in inputs),
        },
        "experts": {
            "applicable": bool(expert_inputs),
            "both_available_count": len(both_available),
            "missing_trf_count": sum(
                not item["expert_proposals"]["trf"]["available"] for item in expert_inputs
            ),
            "missing_exemplar_count": sum(
                not item["expert_proposals"]["exemplar"]["available"] for item in expert_inputs
            ),
            "presence_agreement_count": sum(
                item["expert_proposals"]["trf"]["has_skill"]
                == item["expert_proposals"]["exemplar"]["has_skill"]
                for item in both_available
            ),
            "exact_span_agreement_count": sum(
                item["expert_proposals"]["trf"]["coordinate_space"] == "target_sentence"
                and item["expert_proposals"]["exemplar"]["coordinate_space"] == "target_sentence"
                and item["expert_proposals"]["trf"]["spans"]
                == item["expert_proposals"]["exemplar"]["spans"]
                for item in both_available
            ),
            "provisional_trf_count": sum(
                item["expert_proposals"]["trf"].get("outcome") == "provisional"
                for item in expert_inputs
            ),
            "provisional_exemplar_count": sum(
                item["expert_proposals"]["exemplar"].get("outcome") == "provisional"
                for item in expert_inputs
            ),
        },
        "failure_tolerance": completion,
        "claims": {"accuracy_improvement": "not_evaluated", "gold_used_in_prompt": False},
    }
    pool = []
    prompts_by_idx = {item["idx"]: item for item in prompts}
    for item in inputs:
        idx = item["idx"]
        prediction = predictions_by_idx.get(idx)
        failure = failures_by_idx.get(idx)
        result = results_by_idx[idx]
        pool.append(
            {
                "idx": idx,
                "sentence": item["target_sentence"],
                "status": prediction["status"] if prediction else result["outcome"],
                "review_reasons": result["review_reasons"],
                "prediction": prediction,
                "result": result,
                "failure": failure,
                "expert_audit": prompts_by_idx[idx]["evidence"].get("expert_agreement"),
            }
        )
    priority = [item for item in pool if item["status"] != "complete"]
    ordinary = [item for item in pool if item["status"] == "complete"]
    rng = random.Random(0)
    rng.shuffle(priority)
    rng.shuffle(ordinary)
    review = sorted((priority + ordinary)[:review_sample_size], key=lambda item: item["idx"])
    return summary, review
