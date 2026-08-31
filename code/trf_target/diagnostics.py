"""Deterministic structural and pseudo-label diagnostics for target TRF runs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from trf_target.pipeline import jaccard_and_recall


def build_diagnostics(
    mode: str,
    targets: list[dict[str, Any]],
    parsed: list[dict[str, Any]],
    retrieval_by_idx: dict[int, dict[str, Any]],
    bundle: dict[str, Any],
    review_sample_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parsed_by_idx = {record["idx"]: record for record in parsed}
    trf_items = [item for record in parsed for item in record["trfs"]]
    selected_demos = [
        item
        for idx in parsed_by_idx
        for item in retrieval_by_idx[idx]["selected"]
    ]
    status_counts = Counter(record["status"] for record in parsed)
    selected_statuses = Counter(item["status"] for item in selected_demos)
    neighbor_items = [
        item
        for idx in parsed_by_idx
        for item in retrieval_by_idx[idx]["neighbors"]
    ]
    summary: dict[str, Any] = {
        "schema_version": "target-trf-summary-v1",
        "mode": mode,
        "expected_targets": len(targets),
        "parsed_targets": len(parsed),
        "missing_or_failed_targets": len(targets) - len(parsed),
        "statuses": dict(sorted(status_counts.items())),
        "entity_type_empty": sum(not record["entity_types"] for record in parsed),
        "entity_type_empty_rate": (
            sum(not record["entity_types"] for record in parsed) / len(parsed)
            if parsed
            else 0.0
        ),
        "trf_empty": sum(not record["trfs"] for record in parsed),
        "trf_empty_rate": (
            sum(not record["trfs"] for record in parsed) / len(parsed)
            if parsed
            else 0.0
        ),
        "trf_total": len(trf_items),
        "trf_mean_per_target": len(trf_items) / len(parsed) if parsed else 0.0,
        "main_bank_exact_rate": (
            sum(item["in_main_bank_exact"] for item in trf_items) / len(trf_items)
            if trf_items
            else 0.0
        ),
        "main_bank_casefold_rate": (
            sum(item["in_main_bank_casefold"] for item in trf_items) / len(trf_items)
            if trf_items
            else 0.0
        ),
        "appears_in_target_exact_rate": (
            sum(item["appears_in_target_exact"] for item in trf_items) / len(trf_items)
            if trf_items
            else 0.0
        ),
        "appears_in_target_casefold_rate": (
            sum(item["appears_in_target_casefold"] for item in trf_items)
            / len(trf_items)
            if trf_items
            else 0.0
        ),
        "review_conflicts": sum(bool(record["review_reasons"]) for record in parsed),
        "review_conflict_rate": (
            sum(bool(record["review_reasons"]) for record in parsed) / len(parsed)
            if parsed
            else 0.0
        ),
        "selected_demo_statuses": dict(
            sorted(selected_statuses.items())
        ),
        "selected_demo_status_rates": {
            status: count / len(selected_demos)
            for status, count in sorted(selected_statuses.items())
        }
        if selected_demos
        else {},
        "selected_similarity_mean": (
            sum(float(item["similarity"]) for item in selected_demos) / len(selected_demos)
            if selected_demos
            else 0.0
        ),
        "neighbor_similarity_mean": (
            sum(float(item["similarity"]) for item in neighbor_items)
            / len(neighbor_items)
            if neighbor_items
            else 0.0
        ),
        "neighbor_similarity_min": (
            min(float(item["similarity"]) for item in neighbor_items)
            if neighbor_items
            else None
        ),
        "pseudo_label_agreement": None,
        "semantic_acceptance": {
            "status": "pending_manual_review",
            "passed": False,
            "sample_size": min(review_sample_size, len(parsed)),
        },
    }
    if mode == "leave-one-out":
        demos_by_idx = {item["idx"]: item for item in bundle["demonstrations"]}
        formal = [record for record in parsed if record["idx"] in demos_by_idx]
        type_correct = 0
        jaccards: list[float] = []
        recalls: list[float] = []
        for record in formal:
            demo = demos_by_idx[record["idx"]]
            expected_types = ["Skill"] if demo["status"] == "accepted" else []
            type_correct += record["entity_types"] == expected_types
            predicted = {item["normalized_text"] for item in record["trfs"]}
            reference = set(demo["trfs"])
            jaccard, recall = jaccard_and_recall(predicted, reference)
            jaccards.append(jaccard)
            if recall is not None:
                recalls.append(recall)
        summary["pseudo_label_agreement"] = {
            "formal_targets": len(formal),
            "excluded_targets": sum(
                record["idx"] not in demos_by_idx for record in parsed
            ),
            "entity_type_accuracy": type_correct / len(formal) if formal else None,
            "trf_exact_jaccard_mean": sum(jaccards) / len(jaccards) if jaccards else None,
            "trf_exact_recall_mean_positive": (
                sum(recalls) / len(recalls) if recalls else None
            ),
            "is_gold_metric": False,
        }

    def similarity(record: dict[str, Any]) -> float:
        selected = retrieval_by_idx[record["idx"]]["selected"]
        return sum(item["similarity"] for item in selected) / len(selected)

    categories = [
        (
            "type_trf_conflict",
            [record for record in parsed if record["review_reasons"]],
        ),
        (
            "open_oov",
            [
                record
                for record in parsed
                if any(not item["in_main_bank_exact"] for item in record["trfs"])
            ],
        ),
        ("type_positive", [record for record in parsed if record["entity_types"]]),
        ("type_negative", [record for record in parsed if not record["entity_types"]]),
        ("low_similarity", sorted(parsed, key=lambda record: (similarity(record), record["idx"]))),
    ]
    review: list[dict[str, Any]] = []
    used: set[int] = set()
    quota = max(1, review_sample_size // len(categories))
    for category, records in categories:
        added = 0
        ordered_records = (
            records
            if category == "low_similarity"
            else sorted(records, key=lambda item: item["idx"])
        )
        for record in ordered_records:
            if record["idx"] in used:
                continue
            review.append(
                {
                    "review_category": category,
                    **record,
                    "retrieval": retrieval_by_idx[record["idx"]]["selected"],
                }
            )
            used.add(record["idx"])
            added += 1
            if added == quota or len(review) == review_sample_size:
                break
        if len(review) == review_sample_size:
            break
    if len(review) < min(review_sample_size, len(parsed)):
        for record in sorted(parsed, key=lambda item: item["idx"]):
            if record["idx"] in used:
                continue
            review.append(
                {
                    "review_category": "deterministic_fill",
                    **record,
                    "retrieval": retrieval_by_idx[record["idx"]]["selected"],
                }
            )
            used.add(record["idx"])
            if len(review) == min(review_sample_size, len(parsed)):
                break
    return summary, review
