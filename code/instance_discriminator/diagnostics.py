"""Deterministic diagnostics and manual-review sampling."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return round(numerator / denominator, 6) if denominator else None


def prepared_summary(target_count: int) -> dict[str, Any]:
    return {
        "status": "prepared",
        "target_count": target_count,
        "candidate_record_count": target_count,
        "prompt_record_count": target_count,
        "online_discrimination_performed": False,
        "target_skill_prediction_performed": False,
    }


def build_diagnostics(
    targets: list[dict[str, Any]],
    parsed: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    latest_raw: dict[int, dict[str, Any]],
    review_sample_size: int,
    predictions: list[dict[str, Any]],
    latest_prediction_raw: dict[int, dict[str, Any]],
    prediction_failures: list[dict[str, Any]],
    prediction_results: list[dict[str, Any]],
    validation_issues: list[dict[str, Any]],
    completion: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parsed_by_idx = {item["idx"]: item for item in parsed}
    selected_by_idx = {item["idx"]: item for item in selected}
    predictions_by_idx = {item["idx"]: item for item in predictions}
    results_by_idx = {item["idx"]: item for item in prediction_results}
    status_counts = Counter(item["status"] for item in selected)
    judgment_validation_ids = {
        item["idx"]
        for item in validation_issues
        if "exemplar_judgment" in item["stages"]
    }
    failed_ids = [
        item["idx"]
        for item in targets
        if latest_raw.get(item["idx"], {}).get("status") == "failed"
        and item["idx"] not in judgment_validation_ids
    ]
    pending_ids = [item["idx"] for item in targets if item["idx"] not in latest_raw]
    judgments = [
        judgment for item in parsed for judgment in item.get("judgments", [])
    ]
    scores = Counter(str(item["helpfulness_score"]) for item in judgments)
    roles = Counter(item["role"] for item in judgments)
    source_status: dict[int, str] = {
        candidate["demo_idx"]: candidate["status"]
        for target in targets
        for candidate in target["candidates"]
    }
    selected_items = [item for record in selected for item in record["selected"]]
    selected_source = Counter(item["status"] for item in selected_items)
    source_totals = Counter(source_status[item["demo_idx"]] for item in judgments)
    similarity = [
        float(candidate["similarity"])
        for target in targets
        if target["idx"] in parsed_by_idx
        for candidate in target["candidates"]
    ]
    existence = [
        float(candidate["existence_score"])
        for target in targets
        if target["idx"] in parsed_by_idx
        for candidate in target["candidates"]
    ]
    score_values = [float(item["helpfulness_score"]) for item in judgments]
    trf_empty_scores: list[float] = []
    trf_nonempty_scores: list[float] = []
    for target in targets:
        parsed_item = parsed_by_idx.get(target["idx"])
        if parsed_item is None:
            continue
        destination = (
            trf_nonempty_scores if target["target_evidence"]["trfs"] else trf_empty_scores
        )
        destination.extend(float(item["helpfulness_score"]) for item in parsed_item["judgments"])
    selected_counts = [item["selected_count"] for item in selected]
    prediction_statuses = Counter(item["status"] for item in predictions)
    prediction_failed_ids = [
        item["idx"] for item in prediction_results if item["prediction"] is None
    ]
    prediction_pending_ids: list[int] = []
    within_tolerance = completion["within_tolerance"]
    summary = {
        "status": "complete" if within_tolerance else "partial",
        "target_count": len(targets),
        "parsed_count": len(parsed),
        "selected_record_count": len(selected),
        "complete_count": status_counts["complete"],
        "needs_review_count": status_counts["needs_review"],
        "failed_count": len(failed_ids),
        "pending_count": len(pending_ids),
        "failed_indexes": failed_ids,
        "pending_indexes": pending_ids,
        "validation_issue_count": len(validation_issues),
        "validation_issue_indexes": [item["idx"] for item in validation_issues],
        "failure_tolerance": completion,
        "prediction": {
            "record_count": len(predictions),
            "complete_count": prediction_statuses["complete"],
            "needs_review_count": prediction_statuses["needs_review"],
            "positive_count": sum(item["has_skill"] == 1 for item in predictions),
            "span_total": sum(len(item["spans"]) for item in predictions),
            "failed_indexes": prediction_failed_ids,
            "pending_indexes": prediction_pending_ids,
            "failure_record_count": len(prediction_failures),
            "result_count": len(prediction_results),
            "outcomes": dict(
                sorted(Counter(item["outcome"] for item in prediction_results).items())
            ),
            "repair_attempted_count": sum(
                bool(item.get("repair")) for item in latest_prediction_raw.values()
            ),
            "repaired_count": sum(
                item.get("status") == "complete" and bool(item.get("repair"))
                for item in latest_prediction_raw.values()
            ),
            "recovered_count": sum(
                item.get("status") == "complete" and bool(item.get("recovery"))
                for item in latest_prediction_raw.values()
            ),
        },
        "judgment_count": len(judgments),
        "score_distribution": {str(i): scores[str(i)] for i in range(1, 6)},
        "role_distribution": {
            role: roles[role] for role in ("supporting", "contrastive", "irrelevant")
        },
        "selection": {
            "selected_total": len(selected_items),
            "candidate_total": len(judgments),
            "retention_rate": _rate(len(selected_items), len(judgments)),
            "mean_per_target": _mean([float(value) for value in selected_counts]),
            "zero_selected_count": sum(value == 0 for value in selected_counts),
            "one_selected_count": sum(value == 1 for value in selected_counts),
            "accepted": {
                "selected": selected_source["accepted"],
                "total": source_totals["accepted"],
                "rate": _rate(selected_source["accepted"], source_totals["accepted"]),
            },
            "negative": {
                "selected": selected_source["negative"],
                "total": source_totals["negative"],
                "rate": _rate(selected_source["negative"], source_totals["negative"]),
            },
        },
        "correlations": {
            "helpfulness_similarity": _pearson(score_values, similarity),
            "helpfulness_existence_score": _pearson(score_values, existence),
        },
        "target_trf_groups": {
            "empty_mean_helpfulness": _mean(trf_empty_scores),
            "nonempty_mean_helpfulness": _mean(trf_nonempty_scores),
        },
        "claims": {
            "accuracy_improvement": "not_evaluated",
            "manual_gold_available": False,
        },
    }

    review_pool: list[dict[str, Any]] = []
    for target in targets:
        idx = target["idx"]
        chosen = selected_by_idx.get(idx)
        if chosen is not None:
            prediction = predictions_by_idx.get(idx)
            prediction_result = results_by_idx[idx]
            prediction_raw = latest_prediction_raw.get(idx)
            prediction_status = (
                prediction["status"]
                if prediction is not None
                else prediction_raw.get("status", "pending")
                if prediction_raw
                else "pending"
            )
            review_reasons = list(chosen["review_reasons"])
            if prediction is None:
                review_reasons.append("prediction_failed_or_pending")
            review_pool.append(
                {
                    "idx": idx,
                    "sentence": target["sentence"],
                    "status": prediction_status,
                    "review_reasons": review_reasons,
                    "target_evidence": chosen["target_evidence"],
                    "selected_count": chosen["selected_count"],
                    "selected_demo_ids": [
                        item["demo_idx"] for item in chosen["selected"]
                    ],
                    "prediction": prediction,
                    "prediction_result": prediction_result,
                    "prediction_error": (
                        prediction_raw.get("error") if prediction_raw else None
                    ),
                }
            )
        else:
            raw = latest_raw.get(idx)
            review_pool.append(
                {
                    "idx": idx,
                    "sentence": target["sentence"],
                    "status": raw.get("status", "pending") if raw else "pending",
                    "review_reasons": ["judgment_failed_or_pending"],
                    "target_evidence": target["target_evidence"],
                    "selected_count": 0,
                    "selected_demo_ids": [],
                    "error": raw.get("error") if raw else None,
                    "prediction": predictions_by_idx.get(idx),
                    "prediction_result": results_by_idx[idx],
                    "prediction_error": (
                        latest_prediction_raw.get(idx, {}).get("error")
                    ),
                }
            )
    priority = [item for item in review_pool if item["status"] != "complete"]
    ordinary = [item for item in review_pool if item["status"] == "complete"]
    rng = random.Random(0)
    rng.shuffle(priority)
    rng.shuffle(ordinary)
    review = (priority + ordinary)[:review_sample_size]
    review.sort(key=lambda item: item["idx"])
    return summary, review
