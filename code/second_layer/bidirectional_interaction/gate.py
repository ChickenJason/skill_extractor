"""Frozen semantic gate contracts and deterministic evaluation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from common.io_utils import load_json
from second_layer.bidirectional_interaction.common import InteractionError, PROTOCOL_VERSION


GATE_SCHEMA = "bidirectional-interaction-gate-spec-v1"
TARGET_KEYS = {
    "dataset_id", "record_id", "idx", "sentence", "source_sha256",
    "strata", "trf_anchors", "exemplar_anchors",
}
ANCHOR_KEYS = {
    "demo_idx", "minimum_score", "maximum_score", "expected_role",
    "expected_selected", "expected_change_from_baseline", "rationale",
}
ACCEPTANCE_KEYS = {
    "target_count", "required_exemplar_anchors", "required_keep_anchors",
    "required_drop_anchors", "minimum_changed_exemplar_anchors",
    "maximum_model_responses", "maximum_repair_responses", "required_failed_records",
}
OPTIONAL_ACCEPTANCE_KEYS = {
    "required_ready_targets",
    "minimum_targets_with_active_trf",
    "minimum_targets_with_selected_exemplar",
}


def load_gate_spec(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "gate_id", "base_concurrent_run_id",
        "underlying_protocol", "targets", "acceptance",
    }:
        raise InteractionError("Interaction gate spec has missing or unknown keys")
    if value["schema_version"] != GATE_SCHEMA or value["underlying_protocol"] != PROTOCOL_VERSION:
        raise InteractionError("Interaction gate schema/protocol is unsupported")
    if not isinstance(value["gate_id"], str) or not value["gate_id"]:
        raise InteractionError("Interaction gate_id is invalid")
    targets = value["targets"]
    if not isinstance(targets, list) or not targets:
        raise InteractionError("Interaction gate targets must be non-empty")
    identities: set[tuple[str, str]] = set()
    keep_count = drop_count = exemplar_count = changed_constraints = 0
    for target in targets:
        if not isinstance(target, dict) or set(target) != TARGET_KEYS:
            raise InteractionError("Interaction gate target has missing or unknown keys")
        identity = (target["dataset_id"], target["record_id"])
        if identity in identities or not all(isinstance(item, str) and item for item in identity):
            raise InteractionError("Interaction gate target identity is invalid or duplicated")
        identities.add(identity)
        if type(target["idx"]) is not int or not isinstance(target["sentence"], str):
            raise InteractionError("Interaction gate target idx/sentence is invalid")
        if not isinstance(target["source_sha256"], str) or len(target["source_sha256"]) != 64:
            raise InteractionError("Interaction gate target source_sha256 is invalid")
        if not isinstance(target["strata"], list) or not all(isinstance(item, str) and item for item in target["strata"]):
            raise InteractionError("Interaction gate strata are invalid")
        trf = target["trf_anchors"]
        if not isinstance(trf, dict) or set(trf) != {"must_keep", "must_drop"}:
            raise InteractionError("Interaction TRF anchors are invalid")
        if any(not isinstance(item, str) or not item for values in trf.values() for item in values):
            raise InteractionError("Interaction TRF anchor text is invalid")
        if set(trf["must_keep"]) & set(trf["must_drop"]):
            raise InteractionError("A TRF cannot be both keep and drop anchored")
        keep_count += len(trf["must_keep"])
        drop_count += len(trf["must_drop"])
        anchors = target["exemplar_anchors"]
        if not isinstance(anchors, list):
            raise InteractionError("Interaction exemplar anchors must be a list")
        demo_ids: set[int] = set()
        for anchor in anchors:
            if not isinstance(anchor, dict) or set(anchor) != ANCHOR_KEYS:
                raise InteractionError("Exemplar anchor has missing or unknown keys")
            demo_idx = anchor["demo_idx"]
            if type(demo_idx) is not int or demo_idx in demo_ids:
                raise InteractionError("Exemplar anchor demo_idx is invalid or duplicated")
            demo_ids.add(demo_idx)
            for key in ("minimum_score", "maximum_score"):
                if anchor[key] is not None and (type(anchor[key]) is not int or not 1 <= anchor[key] <= 5):
                    raise InteractionError(f"Exemplar anchor {key} must be null or 1..5")
            if anchor["minimum_score"] is not None and anchor["maximum_score"] is not None and anchor["minimum_score"] > anchor["maximum_score"]:
                raise InteractionError("Exemplar anchor score interval is reversed")
            if anchor["expected_role"] not in {None, "supporting", "contrastive", "irrelevant"}:
                raise InteractionError("Exemplar anchor expected_role is invalid")
            for key in ("expected_selected", "expected_change_from_baseline"):
                if anchor[key] is not None and type(anchor[key]) is not bool:
                    raise InteractionError(f"Exemplar anchor {key} is invalid")
            if not isinstance(anchor["rationale"], str) or not anchor["rationale"].strip():
                raise InteractionError("Exemplar anchor rationale must be non-empty")
            if all(anchor[key] is None for key in ANCHOR_KEYS - {"demo_idx", "rationale"}):
                raise InteractionError("Every exemplar anchor needs at least one result constraint")
            if anchor["expected_change_from_baseline"] is True:
                changed_constraints += 1
        exemplar_count += len(anchors)
    acceptance = value["acceptance"]
    if (
        not isinstance(acceptance, dict)
        or not ACCEPTANCE_KEYS <= set(acceptance)
        or not set(acceptance) <= ACCEPTANCE_KEYS | OPTIONAL_ACCEPTANCE_KEYS
    ):
        raise InteractionError("Interaction gate acceptance contract is invalid")
    for key, item in acceptance.items():
        if type(item) is not int or item < 0:
            raise InteractionError(f"Interaction gate acceptance {key} must be non-negative")
    exact = {
        "target_count": len(targets),
        "required_exemplar_anchors": exemplar_count,
        "required_keep_anchors": keep_count,
        "required_drop_anchors": drop_count,
    }
    for key, actual in exact.items():
        if acceptance[key] != actual:
            raise InteractionError(f"Interaction gate {key} does not match its anchors")
    if acceptance["minimum_changed_exemplar_anchors"] > changed_constraints:
        raise InteractionError("Gate requires more changed anchors than are marked")
    return copy.deepcopy(value)


def target_identities(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: item[key] for key in ("dataset_id", "record_id", "idx", "sentence", "source_sha256")}
        for item in spec["targets"]
    ]


def _check(name: str, passed: bool, expected: Any, actual: Any) -> dict[str, Any]:
    return {"name": name, "passed": passed, "expected": expected, "actual": actual}


def evaluate_gate(
    spec: dict[str, Any],
    contexts: list[dict[str, Any]],
    run_summary: dict[str, Any],
) -> dict[str, Any]:
    by_id = {(item["dataset_id"], item["record_id"]): item for item in contexts}
    target_results: list[dict[str, Any]] = []
    changed_passed = 0
    all_anchor_checks: list[dict[str, Any]] = []
    for target in spec["targets"]:
        identity = (target["dataset_id"], target["record_id"])
        if identity not in by_id:
            raise InteractionError(f"Gate result is missing target {identity!r}")
        context = by_id[identity]
        full_actual = {key: context[key] for key in ("dataset_id", "record_id", "idx", "sentence", "source_sha256")}
        full_expected = {key: target[key] for key in full_actual}
        if full_actual != full_expected:
            raise InteractionError(f"Gate result target identity mismatch for {identity!r}")
        trfs = {item["normalized_text"]: bool(item["active"]) for item in context["final_state"]["trf"]["known_trfs"]}
        checks: list[dict[str, Any]] = []
        for text in target["trf_anchors"]["must_keep"]:
            checks.append(_check(f"trf_keep:{text}", trfs.get(text) is True, True, trfs.get(text)))
        for text in target["trf_anchors"]["must_drop"]:
            checks.append(_check(f"trf_drop:{text}", trfs.get(text) is False, False, trfs.get(text)))
        final_h = context["final_state"]["exemplar"]
        final_by_demo = {item["demo_idx"]: item for item in final_h["judgments"]}
        selected = {item["demo_idx"] for item in final_h["selection"]["selected"]}
        baseline_h = context["baseline_state"]["exemplar"]
        baseline_selected = {item["demo_idx"] for item in baseline_h["selection"]["selected"]}
        baseline_all = {item["demo_idx"]: item for item in baseline_h["judgments"]}
        for anchor in target["exemplar_anchors"]:
            demo_idx = anchor["demo_idx"]
            if demo_idx not in final_by_demo:
                raise InteractionError(f"Gate anchor demo_idx={demo_idx} is absent for {identity!r}")
            judgment = final_by_demo[demo_idx]
            if anchor["minimum_score"] is not None:
                checks.append(_check(f"demo:{demo_idx}:minimum_score", judgment["helpfulness_score"] >= anchor["minimum_score"], anchor["minimum_score"], judgment["helpfulness_score"]))
            if anchor["maximum_score"] is not None:
                checks.append(_check(f"demo:{demo_idx}:maximum_score", judgment["helpfulness_score"] <= anchor["maximum_score"], anchor["maximum_score"], judgment["helpfulness_score"]))
            if anchor["expected_role"] is not None:
                checks.append(_check(f"demo:{demo_idx}:role", judgment["role"] == anchor["expected_role"], anchor["expected_role"], judgment["role"]))
            if anchor["expected_selected"] is not None:
                checks.append(_check(f"demo:{demo_idx}:selected", (demo_idx in selected) == anchor["expected_selected"], anchor["expected_selected"], demo_idx in selected))
            if anchor["expected_change_from_baseline"] is not None:
                baseline = baseline_all.get(demo_idx)
                if baseline is None:
                    raise InteractionError("Full baseline judgment is unavailable for a change anchor")
                changed = (
                    judgment["helpfulness_score"], judgment["role"], demo_idx in selected
                ) != (
                    baseline["helpfulness_score"], baseline["role"], demo_idx in baseline_selected
                )
                passed = changed == anchor["expected_change_from_baseline"]
                checks.append(_check(f"demo:{demo_idx}:changed", passed, anchor["expected_change_from_baseline"], changed))
                if anchor["expected_change_from_baseline"] is True and passed:
                    changed_passed += 1
        all_anchor_checks.extend(checks)
        target_results.append(
            {
                "dataset_id": identity[0],
                "record_id": identity[1],
                "idx": target["idx"],
                "passed": all(item["passed"] for item in checks),
                "checks": checks,
            }
        )
    acceptance = spec["acceptance"]
    budget_checks = [
        _check("target_count", len(contexts) == acceptance["target_count"], acceptance["target_count"], len(contexts)),
        _check("changed_exemplar_anchors", changed_passed >= acceptance["minimum_changed_exemplar_anchors"], acceptance["minimum_changed_exemplar_anchors"], changed_passed),
        _check("model_responses", run_summary.get("model_responses", 0) <= acceptance["maximum_model_responses"], acceptance["maximum_model_responses"], run_summary.get("model_responses", 0)),
        _check("repair_responses", run_summary.get("repair_responses", 0) <= acceptance["maximum_repair_responses"], acceptance["maximum_repair_responses"], run_summary.get("repair_responses", 0)),
        _check("failed_records", run_summary.get("failed_records", 0) == acceptance["required_failed_records"], acceptance["required_failed_records"], run_summary.get("failed_records", 0)),
    ]
    optional_metrics = {
        "required_ready_targets": sum(
            item.get("assembly_status") == "ready" for item in contexts
        ),
        "minimum_targets_with_active_trf": sum(
            bool(item["final_state"]["trf"]["active_trfs"]) for item in contexts
        ),
        "minimum_targets_with_selected_exemplar": sum(
            bool(item["final_state"]["exemplar"]["selection"]["selected"])
            for item in contexts
        ),
    }
    for name, actual in optional_metrics.items():
        if name in acceptance:
            budget_checks.append(
                _check(name, actual >= acceptance[name], acceptance[name], actual)
            )
    passed = all(item["passed"] for item in all_anchor_checks + budget_checks)
    return {
        "schema_version": "bidirectional-interaction-gate-evaluation-v1",
        "gate_id": spec["gate_id"],
        "status": "passed" if passed else "failed",
        "target_results": target_results,
        "acceptance_checks": budget_checks,
    }
