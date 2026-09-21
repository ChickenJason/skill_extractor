"""Pure proposal application, signatures, conflicts, and context assembly."""

from __future__ import annotations

import copy
from typing import Any

from instance_discriminator.pipeline import apply_hard_gate
from second_layer.bidirectional_interaction.common import PIPELINE_VERSION, PROTOCOL_VERSION, InteractionError
from second_layer.bidirectional_interaction.contracts import normalize_trf


def active_branches(mode: str) -> tuple[str, ...]:
    mapping = {
        "none": (),
        "trf_to_exemplar": ("exemplar",),
        "exemplar_to_trf": ("trf",),
        "bidirectional": ("trf", "exemplar"),
    }
    try:
        return mapping[mode]
    except KeyError as error:
        raise InteractionError(f"Unknown interaction mode: {mode}") from error


def trf_signature(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_types": list(state["entity_types"]),
        "trfs": [
            [item["normalized_text"], bool(item["active"])]
            for item in state["known_trfs"]
        ],
    }


def exemplar_signature(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "judgments": [
            [item["demo_idx"], item["helpfulness_score"], item["role"]]
            for item in state["judgments"]
        ],
        "selected_demo_ids": [item["demo_idx"] for item in state["selection"]["selected"]],
    }


def effective_signature(r_state: dict[str, Any], h_state: dict[str, Any]) -> dict[str, Any]:
    return {"trf": trf_signature(r_state), "exemplar": exemplar_signature(h_state)}


def _trf_reviews(entity_types: list[str], active: list[str], unresolved: list[str]) -> list[str]:
    reasons = list(unresolved)
    if not entity_types and active:
        reasons.append("trfs_without_skill_type")
    if entity_types == ["Skill"] and not active:
        reasons.append("skill_type_without_trfs")
    return list(dict.fromkeys(reasons))


def apply_trf_proposal(
    previous: dict[str, Any],
    proposal: dict[str, Any],
    sentence: str,
    round_number: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = copy.deepcopy(previous)
    decisions = proposal["known_trf_decisions"]
    transitions: list[dict[str, Any]] = []
    for item, decision in zip(state["known_trfs"], decisions):
        before = bool(item["active"])
        after = decision["action"] == "keep"
        item["active"] = after
        item["last_decision"] = decision["action"]
        item["reflection"] = {
            "round": round_number,
            "confidence": decision["confidence"],
            "evidence_sources": decision["evidence_sources"],
            "evidence_demo_ids": decision["evidence_demo_ids"],
            "reason_codes": decision["reason_codes"],
            "reason": decision["reason"],
        }
        if before != after:
            transitions.append(
                {
                    "normalized_text": item["normalized_text"],
                    "from": "keep" if before else "drop",
                    "to": "keep" if after else "drop",
                    "round": round_number,
                }
            )
    for addition in proposal["added_trfs"]:
        display = normalize_trf(addition["text"])
        added = {
            "raw_text": display,
            "text": display,
            "normalized_text": display,
            "first_seen_order": len(state["known_trfs"]) + 1,
            "in_main_bank_exact": None,
            "in_main_bank_casefold": None,
            "appears_in_target_exact": display in sentence,
            "appears_in_target_casefold": display.casefold() in sentence.casefold(),
            "semantic_class": addition["semantic_class"],
            "semantic_validation": {
                "contract_id": "trf-addition-semantics-v1",
                "accepted": True,
            },
            "origin": "added",
            "first_seen_round": round_number,
            "active": True,
            "last_decision": "keep",
            "reflection": {
                "round": round_number,
                "confidence": addition["confidence"],
                "evidence_sources": addition["evidence_sources"],
                "evidence_demo_ids": addition["evidence_demo_ids"],
                "reason_codes": addition["reason_codes"],
                "reason": addition["reason"],
            },
        }
        state["known_trfs"].append(added)
        transitions.append(
            {"normalized_text": display, "from": None, "to": "keep", "round": round_number}
        )
    state["entity_types"] = copy.deepcopy(proposal["revised_entity_types"])
    state["active_trfs"] = [item["normalized_text"] for item in state["known_trfs"] if item["active"]]
    state["added_trfs"] = [copy.deepcopy(item) for item in state["known_trfs"] if item["origin"] == "added"]
    state["review_reasons"] = _trf_reviews(
        state["entity_types"], state["active_trfs"], proposal["unresolved_issues"]
    )
    state["status"] = "needs_review" if state["review_reasons"] else "complete"
    state["last_overall_reason"] = proposal["overall_reason"]
    return state, transitions


def apply_exemplar_proposal(
    previous: dict[str, Any],
    proposal: dict[str, Any],
    candidates: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    state = copy.deepcopy(previous)
    state["feature_context"] = "interaction_trf_previous_round"
    state["judgments"] = copy.deepcopy(proposal["revised_judgments"])
    state["selection"] = apply_hard_gate(candidates, state["judgments"], gate)
    reasons = list(proposal["unresolved_issues"])
    count = len(state["selection"]["selected"])
    if count == 0:
        reasons.append("no_helpful_examples")
    elif count < gate["minimum_for_complete"]:
        reasons.append("insufficient_helpful_examples")
    state["review_reasons"] = list(dict.fromkeys(reasons))
    state["unresolved_issues"] = copy.deepcopy(proposal["unresolved_issues"])
    state["status"] = "needs_review" if state["review_reasons"] else "complete"
    state["last_overall_reason"] = proposal["overall_reason"]
    return state


def detect_conflicts(
    r_state: dict[str, Any],
    h_state: dict[str, Any],
    *,
    oscillation: bool = False,
) -> list[dict[str, Any]]:
    selected = {item["demo_idx"] for item in h_state["selection"]["selected"]}
    active = {item["normalized_text"] for item in r_state["known_trfs"] if item["active"]}
    conflicts: list[dict[str, Any]] = []
    for trf in r_state["known_trfs"]:
        reflection = trf.get("reflection") or {}
        if trf["active"]:
            missing = [idx for idx in reflection.get("evidence_demo_ids", []) if idx not in selected]
            if missing:
                conflicts.append(
                    {
                        "type": "trf_evidence_demo_not_retained",
                        "normalized_text": trf["normalized_text"],
                        "demo_ids": missing,
                    }
                )
    for judgment in h_state["judgments"]:
        if judgment["demo_idx"] not in selected:
            continue
        dropped = [
            relation["normalized_text"]
            for relation in judgment.get("trf_relations", [])
            if relation["relation"] == "supports" and relation["normalized_text"] not in active
        ]
        if dropped:
            conflicts.append(
                {
                    "type": "selected_demo_supports_dropped_trf",
                    "demo_idx": judgment["demo_idx"],
                    "normalized_texts": dropped,
                }
            )
    if oscillation:
        conflicts.append({"type": "state_oscillation"})
    return conflicts


def apply_round(
    record: dict[str, Any],
    previous_r: dict[str, Any],
    previous_h: dict[str, Any],
    proposals: dict[str, dict[str, Any]],
    mode: str,
    round_number: int,
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Atomically apply independently produced proposals after both validate."""

    required = set(active_branches(mode))
    if set(proposals) != required:
        raise InteractionError("Round proposals do not exactly match active branches")
    r_state = copy.deepcopy(previous_r)
    h_state = copy.deepcopy(previous_h)
    transitions: list[dict[str, Any]] = []
    if "trf" in proposals:
        r_state, transitions = apply_trf_proposal(
            previous_r, proposals["trf"], record["sentence"], round_number
        )
    if "exemplar" in proposals:
        h_state = apply_exemplar_proposal(
            previous_h, proposals["exemplar"], record["candidates"], gate
        )
    before = effective_signature(previous_r, previous_h)
    after = effective_signature(r_state, h_state)
    return {
        "schema_version": "bidirectional-round-state-v1",
        "dataset_id": record["dataset_id"],
        "record_id": record["record_id"],
        "source_sha256": record["source_sha256"],
        "idx": record["idx"],
        "sentence": record["sentence"],
        "round": round_number,
        "previous_signature": before,
        "effective_signature": after,
        "effective_changed": before != after,
        "trf_state": r_state,
        "exemplar_state": h_state,
        "trf_transitions": transitions,
        "proposal_audit": copy.deepcopy(proposals),
    }


def assemble_context(
    record: dict[str, Any],
    r_state: dict[str, Any],
    h_state: dict[str, Any],
    *,
    mode: str,
    rounds: int,
    converged: bool,
    conflicts: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    reviews = list(dict.fromkeys(r_state["review_reasons"] + h_state["review_reasons"]))
    interaction_status = "conflicted" if conflicts else ("mixed" if reviews else "aligned")
    output = {
        "schema_version": "bidirectional-interaction-context-v1",
        "dataset_id": record["dataset_id"],
        "record_id": record["record_id"],
        "source_sha256": record["source_sha256"],
        "idx": record["idx"],
        "sentence": record["sentence"],
        "assembly_status": "ready",
        "baseline_context": copy.deepcopy(record["baseline_context"]),
        "baseline_state": {"trf": copy.deepcopy(record["r0"]), "exemplar": copy.deepcopy(record["h0"])},
        "final_state": {"trf": copy.deepcopy(r_state), "exemplar": copy.deepcopy(h_state)},
        "interaction": {
            "mode": mode,
            "rounds": rounds,
            "converged": converged,
            "status": interaction_status,
            "conflicts": copy.deepcopy(conflicts),
            "review_reasons": reviews,
        },
        "provenance": copy.deepcopy(provenance),
    }
    forbidden = {"prediction", "predictions", "final_spans", "skill_spans_generated"}
    if forbidden & set(output):
        raise InteractionError("Final context contains a forbidden target-span field")
    return output


def context_summary(records: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "bidirectional-interaction-summary-v1",
        "status": "completed",
        "target_count": len(records),
        "ready_count": sum(item["assembly_status"] == "ready" for item in records),
        "aligned": sum(item["interaction"]["status"] == "aligned" for item in records),
        "mixed": sum(item["interaction"]["status"] == "mixed" for item in records),
        "conflicted": sum(item["interaction"]["status"] == "conflicted" for item in records),
        "rounds_total": sum(item["interaction"]["rounds"] for item in records),
        "model_responses": metrics.get("model_responses", 0),
        "repair_responses": metrics.get("repair_responses", 0),
        "failed_records": metrics.get("failed_records", 0),
        "prompt_tokens": metrics.get("prompt_tokens", 0),
        "completion_tokens": metrics.get("completion_tokens", 0),
    }
