"""Deterministic automatic and manual-review evaluation for the stratified gate."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from second_layer.collaborative_reflection.common import ReflectionError
from second_layer.collaborative_reflection_json_schema.branch_runner import _latest


ROOT_FIELDS = {
    "trf": {
        "revised_entity_types", "original_trf_decisions", "added_trfs",
        "overall_reason", "unresolved_issues",
    },
    "exemplar": {"revised_judgments", "overall_reason", "unresolved_issues"},
}


def _map_by_idx(records: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        idx = record.get("idx")
        if type(idx) is not int or idx in result:
            raise ReflectionError(f"{label} has invalid or duplicate idx={idx!r}")
        result[idx] = record
    return result


def _response_root(content: str) -> set[str] | None:
    try:
        value = json.loads(content)
    except Exception:
        return None
    return set(value) if isinstance(value, dict) else None


def evaluate_gate(
    *, spec: dict[str, Any], inputs: list[dict[str, Any]],
    contexts: list[dict[str, Any]], trf_raw: list[dict[str, Any]],
    exemplar_raw: list[dict[str, Any]], branch_metrics: dict[str, dict[str, Any]],
    processes: dict[str, dict[str, Any]], coverage: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[str] = []
    input_by_idx = _map_by_idx(inputs, "gate inputs")
    context_by_idx = _map_by_idx(contexts, "gate contexts")
    expected_indexes = [item["idx"] for item in spec["targets"]]
    if sorted(input_by_idx) != expected_indexes or sorted(context_by_idx) != expected_indexes:
        failures.append("target_identity_coverage_mismatch")

    action_counts: Counter[str] = Counter()
    anchor_results: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    unanchored_total = 0
    added_total = 0
    exemplar_changes = 0
    review_transitions: list[dict[str, Any]] = []
    for target in spec["targets"]:
        idx = target["idx"]
        if idx not in input_by_idx or idx not in context_by_idx:
            continue
        source = input_by_idx[idx]
        context = context_by_idx[idx]
        for key in ("dataset_id", "record_id", "source_sha256", "idx", "sentence"):
            if context.get(key) != source.get(key):
                failures.append(f"context_identity_mismatch:idx={idx}:{key}")
        if context.get("assembly_status") != "ready":
            failures.append(f"context_not_ready:idx={idx}")
        baseline = context["baseline_context"]
        reflected = context["reflected_context"]
        trf = reflected["trf_context"]
        exemplar = reflected["exemplar_context"]
        original = [item["normalized_text"] for item in baseline["trf_context"]["trfs"]]
        decisions = trf.get("original_trf_decisions", [])
        decision_map: dict[str, str] = {}
        for decision in decisions:
            text = decision.get("normalized_text")
            if text in decision_map:
                failures.append(f"duplicate_trf_decision:idx={idx}:trf={text}")
            decision_map[text] = decision.get("action")
            action_counts[decision.get("action")] += 1
        if list(decision_map) != original:
            failures.append(f"trf_decision_coverage_or_order_mismatch:idx={idx}")
        anchors = target["trf_anchors"]
        for expected_action, key in (("keep", "must_keep"), ("drop", "must_drop")):
            for text in anchors[key]:
                actual = decision_map.get(text)
                passed = actual == expected_action
                anchor_results.append(
                    {"idx": idx, "normalized_text": text, "expected": expected_action, "actual": actual, "passed": passed}
                )
                if not passed:
                    failures.append(
                        f"anchor_mismatch:idx={idx}:trf={text}:expected={expected_action}:actual={actual}"
                    )
        anchored = set(anchors["must_keep"]) | set(anchors["must_drop"])
        unanchored = [item for item in decisions if item["normalized_text"] not in anchored]
        unanchored_total += len(unanchored)
        added = trf.get("added_trfs", [])
        added_total += len(added)

        revised_judgments = exemplar.get("revised_judgments", [])
        if len(revised_judgments) != 16:
            failures.append(f"revised_judgment_count:idx={idx}:actual={len(revised_judgments)}")
        candidate_by_id = {item["demo_idx"]: item for item in source["candidates"]}
        judgment_ids: set[int] = set()
        for judgment in revised_judgments:
            demo_idx = judgment.get("demo_idx")
            if demo_idx in judgment_ids or demo_idx not in candidate_by_id:
                failures.append(f"revised_judgment_identity:idx={idx}:demo={demo_idx}")
                continue
            judgment_ids.add(demo_idx)
            status = candidate_by_id[demo_idx]["status"]
            role = judgment.get("role")
            allowed = {"supporting", "irrelevant"} if status == "accepted" else {"contrastive", "irrelevant"}
            if role not in allowed:
                failures.append(f"illegal_candidate_role:idx={idx}:demo={demo_idx}:status={status}:role={role}")
        if judgment_ids != set(candidate_by_id):
            failures.append(f"revised_judgment_coverage:idx={idx}")
        selected_ids = [item["demo_idx"] for item in exemplar.get("selected", [])]
        if len(selected_ids) != len(set(selected_ids)) or set(selected_ids) - set(candidate_by_id):
            failures.append(f"selected_candidate_provenance:idx={idx}")
        if baseline["exemplar_context"].get("feature_context") != "absent":
            failures.append(f"baseline_feature_context_present:idx={idx}")
        if exemplar.get("feature_context") != "baseline_trf_only":
            failures.append(f"reflected_feature_context_contract:idx={idx}")
        changed_ids = set(exemplar.get("promoted_demo_ids", [])) | set(exemplar.get("removed_demo_ids", []))
        exemplar_changes += len(changed_ids)
        review_transitions.append(
            {
                "idx": idx,
                "trf": {
                    "baseline_status": baseline["trf_context"]["status"],
                    "reflected_status": trf["status"],
                    "baseline_reasons": baseline["trf_context"].get("review_reasons", []),
                    "reflected_reasons": trf.get("review_reasons", []),
                },
                "exemplar": {
                    "baseline_status": baseline["exemplar_context"]["status"],
                    "reflected_status": exemplar["status"],
                    "baseline_reasons": baseline["exemplar_context"].get("review_reasons", []),
                    "reflected_reasons": exemplar.get("review_reasons", []),
                },
            }
        )
        for forbidden in ("final_skill_spans", "predicted_skill_spans", "skill_span_prediction"):
            if forbidden in context or forbidden in reflected:
                failures.append(f"final_skill_span_prediction_present:idx={idx}:{forbidden}")
        manual.append(
            {
                "schema_version": "collaborative-reflection-gate-manual-review-v1",
                "dataset_id": context["dataset_id"],
                "record_id": context["record_id"],
                "source_sha256": context["source_sha256"],
                "idx": idx,
                "sentence": context["sentence"],
                "unanchored_original_trf_decisions": unanchored,
                "added_trfs": added,
                "entity_type_change": {
                    "baseline": baseline["trf_context"]["entity_types"],
                    "reflected": trf["revised_entity_types"],
                },
                "exemplar_changes": {
                    "retained_demo_ids": exemplar.get("retained_demo_ids", []),
                    "promoted_demo_ids": exemplar.get("promoted_demo_ids", []),
                    "removed_demo_ids": exemplar.get("removed_demo_ids", []),
                    "score_changed_demo_ids": exemplar.get("score_changed_demo_ids", []),
                    "role_changed_demo_ids": exemplar.get("role_changed_demo_ids", []),
                },
                "interaction": context["interaction"],
                "requires_manual_review": bool(
                    unanchored
                    or added
                    or baseline["trf_context"]["entity_types"] != trf["revised_entity_types"]
                    or changed_ids
                    or context["interaction"]["status"] != "aligned"
                ),
            }
        )

    raw_by_branch = {"trf": trf_raw, "exemplar": exemplar_raw}
    initial_requests = 0
    valid_initial_roots = 0
    wrapper_initials = 0
    wrapper_repairs = 0
    for branch, records in raw_by_branch.items():
        latest = _latest(records, branch)
        if set(latest) != set(expected_indexes):
            failures.append(f"latest_raw_coverage:{branch}")
        for raw in records:
            previous_root: set[str] | None = None
            for attempt in raw.get("attempts", []):
                response = attempt.get("response")
                root = _response_root(response.get("content", "")) if response else None
                if attempt.get("kind") == "initial":
                    initial_requests += 1
                    if root == ROOT_FIELDS[branch]:
                        valid_initial_roots += 1
                    if root == {"required_output"}:
                        wrapper_initials += 1
                elif attempt.get("kind") == "repair" and previous_root == {"required_output"}:
                    wrapper_repairs += 1
                previous_root = root

    total_repairs = sum(metrics.get("repair_requests", 0) for metrics in branch_metrics.values())
    model_calls = sum(metrics.get("model_calls", 0) for metrics in branch_metrics.values())
    failed_records = sum(metrics.get("failed_records", 0) for metrics in branch_metrics.values())
    acceptance = spec["acceptance"]
    if len(contexts) != acceptance["target_count"]:
        failures.append(f"target_count:actual={len(contexts)}")
    for name, minimum in acceptance["minimum_strata_counts"].items():
        if coverage.get(name, 0) < minimum:
            failures.append(f"stratum_count:{name}:actual={coverage.get(name, 0)}:minimum={minimum}")
    if sum(item["passed"] and item["expected"] == "keep" for item in anchor_results) != acceptance["required_keep_anchors"]:
        failures.append("required_keep_anchor_count")
    if sum(item["passed"] and item["expected"] == "drop" for item in anchor_results) != acceptance["required_drop_anchors"]:
        failures.append("required_drop_anchor_count")
    if action_counts["keep"] < 1 or action_counts["drop"] < 1:
        failures.append("keep_and_drop_paths_not_both_observed")
    if initial_requests != acceptance["target_count"] * 2:
        failures.append(f"initial_request_count:actual={initial_requests}")
    if valid_initial_roots != initial_requests:
        failures.append(f"invalid_initial_response_roots:actual={initial_requests - valid_initial_roots}")
    if wrapper_repairs != acceptance["required_wrapper_repairs"] or wrapper_initials:
        failures.append(f"required_output_wrapper:initials={wrapper_initials}:repairs={wrapper_repairs}")
    if total_repairs > acceptance["maximum_total_repairs"]:
        failures.append(f"repair_budget:actual={total_repairs}:maximum={acceptance['maximum_total_repairs']}")
    if model_calls > acceptance["maximum_model_calls"]:
        failures.append(f"model_call_budget:actual={model_calls}:maximum={acceptance['maximum_model_calls']}")
    if failed_records != acceptance["required_failed_records"]:
        failures.append(f"failed_records:actual={failed_records}")
    if not isinstance(processes, dict) or set(processes) != {"trf", "exemplar"}:
        failures.append("process_audit_incomplete")
        overlap = False
    else:
        overlap = (
            processes["trf"].get("started_at") <= processes["exemplar"].get("ended_at")
            and processes["exemplar"].get("started_at") <= processes["trf"].get("ended_at")
        )
        if not overlap:
            failures.append("branch_processes_did_not_overlap")

    report = {
        "schema_version": "collaborative-reflection-stratified-gate-evaluation-v1",
        "gate_id": spec["gate_id"],
        "gate_status": "passed" if not failures else "failed",
        "target_count": len(contexts),
        "coverage": coverage,
        "action_counts": {"keep": action_counts["keep"], "drop": action_counts["drop"]},
        "anchors": {
            "required_keep": acceptance["required_keep_anchors"],
            "required_drop": acceptance["required_drop_anchors"],
            "passed_keep": sum(item["passed"] and item["expected"] == "keep" for item in anchor_results),
            "passed_drop": sum(item["passed"] and item["expected"] == "drop" for item in anchor_results),
            "results": anchor_results,
        },
        "request_audit": {
            "initial_requests": initial_requests,
            "valid_initial_roots": valid_initial_roots,
            "wrapper_initials": wrapper_initials,
            "wrapper_repairs": wrapper_repairs,
            "total_repairs": total_repairs,
            "model_calls": model_calls,
            "failed_records": failed_records,
        },
        "process_overlap": overlap,
        "review_transitions": review_transitions,
        "manual_review": {
            "record_count": len(manual),
            "required_record_count": sum(item["requires_manual_review"] for item in manual),
            "unanchored_original_trf_count": unanchored_total,
            "added_trf_count": added_total,
            "exemplar_selection_change_count": exemplar_changes,
        },
        "final_skill_spans_generated": False,
        "hard_failures": failures,
        "accuracy_claimed": False,
    }
    return report, manual
