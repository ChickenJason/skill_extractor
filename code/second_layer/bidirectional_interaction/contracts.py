"""Rebuild R0/H0 and parse strict per-round interaction proposals."""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from typing import Any

from common.contracts import record_identity, record_source_sha256, validate_unique_identities
from instance_discriminator.pipeline import REASON_CODES, ROLES, apply_hard_gate
from second_layer.bidirectional_interaction.common import InteractionError


EVIDENCE_SOURCES = frozenset({"TARGET_TEXT", "EXEMPLARS"})
TRF_REASON_CODES = frozenset(
    {
        "TARGET_EXPLICIT_SUPPORT", "TARGET_IMPLICIT_SUPPORT", "EXEMPLAR_SUPPORT",
        "EXEMPLAR_CONTRADICTION", "TOO_GENERIC", "TYPE_MISMATCH",
        "MISSING_SKILL_CUE", "BOUNDARY_EVIDENCE", "OPEN_VOCABULARY_INFERENCE",
    }
)
TRF_UNRESOLVED = frozenset(
    {"TYPE_TRF_CONFLICT", "INSUFFICIENT_EVIDENCE", "CONFLICTING_EXEMPLARS", "POTENTIAL_SKILL_SPAN"}
)
EXEMPLAR_UNRESOLVED = frozenset({"INSUFFICIENT_TRF_EVIDENCE", "CONFLICTING_TRF_EVIDENCE"})
RELATIONS = frozenset({"supports", "contrasts", "unrelated"})
ADDITION_CLASSES = frozenset({"domain_feature", "skill_type_cue"})
TRF_EFFECTS = frozenset({"boost", "penalize", "none"})
BOUNDARY_SIMILARITY_FLOOR = 0.55
ACCEPTED_ALIGNMENT_SIMILARITY_FLOOR = 0.58
SKILL_TYPE_CUES = frozenset(
    {"skill", "skills", "ability", "abilities", "proficiency", "competence", "experience"}
)
POSITIVE_TRF_REASON_CODES = frozenset(
    {
        "TARGET_EXPLICIT_SUPPORT",
        "TARGET_IMPLICIT_SUPPORT",
        "EXEMPLAR_SUPPORT",
        "OPEN_VOCABULARY_INFERENCE",
    }
)
FORBIDDEN_TARGET_FIELDS = frozenset({"prediction", "predictions", "final_spans", "skill_spans_generated"})


def normalize_trf(value: str) -> str:
    if not isinstance(value, str):
        raise InteractionError("TRF text must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized:
        raise InteractionError("TRF text must be non-empty")
    return normalized


def trf_key(value: str) -> str:
    return normalize_trf(value).casefold()


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized = normalize_trf(phrase).casefold()
    pattern = r"(?<!\w)" + re.escape(normalized).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.search(pattern, text.casefold()) is not None


def _index(records: list[dict[str, Any]], label: str) -> dict[tuple[str, str], dict[str, Any]]:
    validate_unique_identities(records, label)
    return {record_identity(item, label): item for item in records}


def _aligned(base: dict[str, Any], other: dict[str, Any], label: str) -> None:
    identity = record_identity(base, "baseline context")
    if record_identity(other, label) != identity:
        raise InteractionError(f"{label} identity mismatch for {identity!r}")
    if other.get("idx") != base.get("idx") or other.get("sentence") != base.get("sentence"):
        raise InteractionError(f"{label} idx/sentence mismatch for {identity!r}")
    if record_source_sha256(other, label) != record_source_sha256(base, "baseline context"):
        raise InteractionError(f"{label} source SHA256 mismatch for {identity!r}")


def _same_selected(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    return actual == expected


def _trf_state(context: dict[str, Any]) -> dict[str, Any]:
    source = context["trf_context"]
    entity_types = source.get("entity_types")
    if entity_types not in ([], ["Skill"]):
        raise InteractionError("Baseline entity_types must be [] or ['Skill']")
    known: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, raw in enumerate(source.get("trfs", []), start=1):
        if not isinstance(raw, dict):
            raise InteractionError("Baseline TRFs must be objects")
        display = normalize_trf(raw.get("normalized_text"))
        key = trf_key(display)
        if key in seen:
            raise InteractionError("Baseline TRFs collide after NFKC/casefold normalization")
        seen.add(key)
        payload = copy.deepcopy(raw)
        payload.update(
            {
                "raw_text": raw.get("raw_text", display),
                "text": raw.get("text", display),
                "normalized_text": display,
                "first_seen_order": position,
                "semantic_class": raw.get("semantic_class"),
                "semantic_validation": raw.get("semantic_validation"),
                "origin": "baseline",
                "first_seen_round": 0,
                "active": True,
                "last_decision": "keep",
                "reflection": None,
            }
        )
        known.append(payload)
    return {
        "schema_version": "trf-interaction-state-v1",
        "entity_types": list(entity_types),
        "known_trfs": known,
        "active_trfs": [item["normalized_text"] for item in known],
        "added_trfs": [],
        "status": source.get("status"),
        "review_reasons": list(source.get("review_reasons", [])),
        "models": copy.deepcopy(source.get("models", {})),
    }


def _exemplar_state(
    candidate: dict[str, Any],
    judgment: dict[str, Any],
    context: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    candidates = candidate.get("candidates")
    judgments = judgment.get("judgments")
    if not isinstance(candidates, list) or len(candidates) != 16:
        raise InteractionError("Each target must have exactly 16 candidates")
    if not isinstance(judgments, list) or len(judgments) != 16:
        raise InteractionError("Each target must have exactly 16 judgments")
    candidate_ids = [item.get("demo_idx") for item in candidates]
    judgment_ids = [item.get("demo_idx") for item in judgments]
    if candidate_ids != judgment_ids or len(set(candidate_ids)) != 16:
        raise InteractionError("Candidate/judgment demo identities are incomplete or reordered")
    seen_demo: set[tuple[str, str]] = set()
    for item in candidates:
        demo_identity = (item.get("demo_dataset_id"), item.get("demo_record_id"))
        if not all(isinstance(value, str) and value for value in demo_identity):
            raise InteractionError("Candidate has no complete demo identity")
        if demo_identity in seen_demo:
            raise InteractionError("Duplicate candidate demo identity")
        seen_demo.add(demo_identity)
        if not isinstance(item.get("demo_source_sha256"), str) or len(item["demo_source_sha256"]) != 64:
            raise InteractionError("Candidate has no valid demo source SHA256")
        if not isinstance(item.get("sentence"), str) or not item["sentence"]:
            raise InteractionError("Candidate has no sentence")
    selection = apply_hard_gate(candidates, judgments, gate)
    baseline = context["exemplar_context"]
    if not _same_selected(selection["selected"], baseline.get("selected", [])):
        raise InteractionError("Baseline selected examples cannot be rebuilt by apply_hard_gate")
    if selection["rejected_demo_ids"] != baseline.get("rejected_demo_ids"):
        raise InteractionError("Baseline rejected examples cannot be rebuilt by apply_hard_gate")
    if selection["eligible_count"] != baseline.get("eligible_count"):
        raise InteractionError("Baseline eligible_count cannot be rebuilt")
    for item in judgments:
        if item.get("role") not in ROLES or item.get("helpfulness_score") not in range(1, 6):
            raise InteractionError("Baseline judgment role/score is invalid")
    return {
        "schema_version": "exemplar-interaction-state-v1",
        "feature_context": "absent",
        "judgments": [dict(copy.deepcopy(item), trf_relations=[]) for item in judgments],
        "selection": selection,
        "status": baseline.get("status"),
        "review_reasons": list(baseline.get("review_reasons", [])),
        "unresolved_issues": [],
        "models": copy.deepcopy(baseline.get("models", {})),
    }


def rebuild_inputs(
    contexts: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    gate: dict[str, Any],
    *,
    candidate_count: int = 16,
) -> list[dict[str, Any]]:
    """Join the three immutable sources and deterministically reconstruct R0/H0."""

    if candidate_count != 16:
        raise InteractionError("The interaction protocol fixes candidate_count at 16")
    context_by_id = _index(contexts, "baseline contexts")
    candidate_by_id = _index(candidates, "candidate records")
    judgment_by_id = _index(judgments, "judgment records")
    if set(context_by_id) != set(candidate_by_id) or set(context_by_id) != set(judgment_by_id):
        raise InteractionError("The three input sources do not cover identical target identities")
    output: list[dict[str, Any]] = []
    for identity, context in sorted(context_by_id.items(), key=lambda pair: pair[1]["idx"]):
        if context.get("schema_version") != "second-layer-context-v1":
            raise InteractionError(f"Unsupported baseline context schema for {identity!r}")
        if any(key in context for key in FORBIDDEN_TARGET_FIELDS):
            raise InteractionError("Baseline context contains a forbidden final-span field")
        candidate = candidate_by_id[identity]
        judgment = judgment_by_id[identity]
        if candidate.get("schema_version") != "candidate-instances-v1":
            raise InteractionError(f"Unsupported candidate schema for {identity!r}")
        if judgment.get("schema_version") not in {
            "instance-judgments-v1",
            "instance-judgments-v2",
        }:
            raise InteractionError(f"Unsupported judgment schema for {identity!r}")
        _aligned(context, candidate, "candidate record")
        _aligned(context, judgment, "judgment record")
        if candidate.get("candidate_count") != candidate_count or judgment.get("candidate_count") != candidate_count:
            raise InteractionError(f"Candidate/judgment count mismatch for {identity!r}")
        r0 = _trf_state(context)
        h0 = _exemplar_state(candidate, judgment, context, gate)
        output.append(
            {
                "schema_version": "bidirectional-interaction-input-v1",
                "dataset_id": identity[0],
                "record_id": identity[1],
                "source_sha256": context["source_sha256"],
                "idx": context["idx"],
                "sentence": context["sentence"],
                "baseline_context": copy.deepcopy(context),
                "candidates": copy.deepcopy(candidate["candidates"]),
                "baseline_judgments": copy.deepcopy(judgment["judgments"]),
                "r0": r0,
                "h0": h0,
            }
        )
    return output


def _strict_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise InteractionError(f"{label} has missing or unknown keys")
    return value


def _parse_json(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError) as error:
        raise InteractionError(f"Proposal is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise InteractionError("Proposal root must be an object")
    return value


def _parse_evidence(item: dict[str, Any], demo_ids: set[int], max_reason: int) -> dict[str, Any]:
    confidence = item["confidence"]
    if type(confidence) is not int or not 1 <= confidence <= 5:
        raise InteractionError("TRF confidence must be an integer from 1 to 5")
    sources = item["evidence_sources"]
    if not isinstance(sources, list) or not sources or len(sources) != len(set(sources)) or not set(sources) <= EVIDENCE_SOURCES:
        raise InteractionError("TRF evidence_sources are invalid")
    evidence_ids = item["evidence_demo_ids"]
    if not isinstance(evidence_ids, list) or len(evidence_ids) != len(set(evidence_ids)) or not set(evidence_ids) <= demo_ids:
        raise InteractionError("TRF evidence_demo_ids are invalid")
    codes = item["reason_codes"]
    if not isinstance(codes, list) or not codes or len(codes) != len(set(codes)) or not set(codes) <= TRF_REASON_CODES:
        raise InteractionError("TRF reason_codes are invalid")
    reason = item["reason"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > max_reason:
        raise InteractionError("TRF reason is invalid")
    code_set = set(codes)
    source_set = set(sources)
    if evidence_ids and "EXEMPLARS" not in source_set:
        raise InteractionError("TRF evidence_demo_ids require the EXEMPLARS source")
    if "EXEMPLARS" in source_set and not evidence_ids:
        raise InteractionError("The EXEMPLARS source requires at least one evidence_demo_id")
    if code_set & {"TARGET_EXPLICIT_SUPPORT", "TARGET_IMPLICIT_SUPPORT"} and "TARGET_TEXT" not in source_set:
        raise InteractionError("Target-support reason codes require the TARGET_TEXT source")
    if code_set & {"EXEMPLAR_SUPPORT", "EXEMPLAR_CONTRADICTION"} and "EXEMPLARS" not in source_set:
        raise InteractionError("Exemplar reason codes require the EXEMPLARS source")
    return copy.deepcopy(item)


def parse_trf_proposal(content: str, record: dict[str, Any], previous_r: dict[str, Any], *, max_reason: int) -> dict[str, Any]:
    root = _strict_keys(
        _parse_json(content),
        {"schema_version", "revised_entity_types", "known_trf_decisions", "added_trfs", "overall_reason", "unresolved_issues"},
        "TRF proposal",
    )
    if root["schema_version"] != "trf-interaction-proposal-v1":
        raise InteractionError("Unsupported TRF proposal schema")
    if root["revised_entity_types"] not in ([], ["Skill"]):
        raise InteractionError("revised_entity_types must be [] or ['Skill']")
    known = previous_r["known_trfs"]
    decisions = root["known_trf_decisions"]
    if not isinstance(decisions, list) or len(decisions) != len(known):
        raise InteractionError("TRF proposal must decide every known TRF")
    expected = [item["normalized_text"] for item in known]
    actual: list[str] = []
    demo_ids = {item["demo_idx"] for item in record["candidates"]}
    evidence_keys = {"confidence", "evidence_sources", "evidence_demo_ids", "reason_codes", "reason"}
    for item in decisions:
        value = _strict_keys(item, {"normalized_text", "action", *evidence_keys}, "known TRF decision")
        if value["action"] not in {"keep", "drop"}:
            raise InteractionError("Known TRF action must be keep/drop")
        actual.append(value["normalized_text"])
        _parse_evidence(value, demo_ids, max_reason)
        model_action = value["action"]
        text = value["normalized_text"]
        is_skill_type_cue = trf_key(text) in SKILL_TYPE_CUES
        cue_is_explicit = _contains_phrase(record["sentence"], text)
        explicit_skill_cue = (
            is_skill_type_cue
            and root["revised_entity_types"] == ["Skill"]
            and cue_is_explicit
        )
        missing_skill_cue = is_skill_type_cue and not cue_is_explicit
        obvious_modifier = (
            len(text.split()) == 1
            and text.isalpha()
            and text.casefold().endswith("ly")
            and _contains_phrase(record["sentence"], text)
        )
        if explicit_skill_cue:
            value["action"] = "keep"
            if "TARGET_TEXT" not in value["evidence_sources"]:
                value["evidence_sources"].append("TARGET_TEXT")
            if "TARGET_EXPLICIT_SUPPORT" not in value["reason_codes"]:
                value["reason_codes"].append("TARGET_EXPLICIT_SUPPORT")
        elif missing_skill_cue:
            value["action"] = "drop"
            value["reason_codes"] = ["MISSING_SKILL_CUE"]
        elif obvious_modifier:
            value["action"] = "drop"
            value["reason_codes"] = ["TYPE_MISMATCH"]
        value["model_action"] = model_action
        value["semantic_gate_action"] = (
            "forced_keep_explicit_skill_type_cue"
            if explicit_skill_cue
            else (
                "forced_drop_missing_skill_type_cue"
                if missing_skill_cue
                else ("forced_drop_descriptive_modifier" if obvious_modifier else "none")
            )
        )
        if value["action"] == "keep" and not set(value["reason_codes"]) & POSITIVE_TRF_REASON_CODES:
            raise InteractionError("A kept TRF requires positive type-related support")
    if actual != expected:
        raise InteractionError("Known TRF decisions must exactly preserve universe order")
    additions = root["added_trfs"]
    if not isinstance(additions, list):
        raise InteractionError("added_trfs must be a list")
    seen = {trf_key(item) for item in expected}
    for item in additions:
        value = _strict_keys(item, {"text", "semantic_class", *evidence_keys}, "added TRF")
        display = normalize_trf(value["text"])
        if len(display) > 160 or display == normalize_trf(record["sentence"]):
            raise InteractionError("Added TRF fails the lexical semantic gate")
        key = trf_key(display)
        if key in seen:
            raise InteractionError("Added TRF duplicates the known universe")
        seen.add(key)
        if value["semantic_class"] not in ADDITION_CLASSES:
            raise InteractionError("Added TRF semantic_class is not allowed")
        if value["semantic_class"] == "skill_type_cue" and not _contains_phrase(
            record["sentence"], display
        ):
            raise InteractionError("An added skill_type_cue must occur explicitly in target text")
        _parse_evidence(value, demo_ids, max_reason)
        if not set(value["reason_codes"]) & POSITIVE_TRF_REASON_CODES:
            raise InteractionError("An added TRF requires positive type-related support")
    active_after = sum(item["action"] == "keep" for item in decisions) + len(additions)
    if root["revised_entity_types"] == [] and active_after:
        raise InteractionError("No TRF may remain active when revised_entity_types is empty")
    if additions and root["revised_entity_types"] != ["Skill"]:
        raise InteractionError("TRF additions require revised_entity_types ['Skill']")
    if not isinstance(root["overall_reason"], str) or not root["overall_reason"].strip() or len(root["overall_reason"]) > max_reason:
        raise InteractionError("TRF overall_reason is invalid")
    unresolved = root["unresolved_issues"]
    if not isinstance(unresolved, list) or len(unresolved) != len(set(unresolved)) or not set(unresolved) <= TRF_UNRESOLVED:
        raise InteractionError("TRF unresolved_issues are invalid")
    return copy.deepcopy(root)


def parse_exemplar_proposal(
    content: str,
    record: dict[str, Any],
    previous_r: dict[str, Any],
    previous_h: dict[str, Any],
    *,
    max_reason: int,
) -> dict[str, Any]:
    root = _strict_keys(
        _parse_json(content),
        {"schema_version", "revised_judgments", "overall_reason", "unresolved_issues"},
        "exemplar proposal",
    )
    if root["schema_version"] != "exemplar-interaction-proposal-v1":
        raise InteractionError("Unsupported exemplar proposal schema")
    revised = root["revised_judgments"]
    if not isinstance(revised, list) or len(revised) != 16:
        raise InteractionError("Exemplar proposal must rejudge all 16 candidates")
    candidates = {item["demo_idx"]: item for item in record["candidates"]}
    previous = {item["demo_idx"]: item for item in previous_h["judgments"]}
    if set(previous) != set(candidates):
        raise InteractionError("Previous exemplar state does not cover the candidate set")
    active = set(previous_r["active_trfs"])
    seen: set[int] = set()
    keys = {
        "demo_idx",
        "helpfulness_score",
        "role",
        "trf_relations",
        "reason_codes",
        "reason",
    }
    for item in revised:
        value = _strict_keys(item, keys, "revised exemplar judgment")
        demo_idx = value["demo_idx"]
        if type(demo_idx) is not int or demo_idx not in candidates or demo_idx in seen:
            raise InteractionError("Exemplar proposal has an unknown/duplicate demo_idx")
        seen.add(demo_idx)
        prior = previous[demo_idx]
        previous_score = prior["helpfulness_score"]
        previous_role = prior["role"]
        score = value["helpfulness_score"]
        if type(score) is not int or not 1 <= score <= 5:
            raise InteractionError("Exemplar helpfulness_score must be 1..5")
        role = value["role"]
        status = candidates[demo_idx]["status"]
        allowed_roles = {"supporting", "irrelevant"} if status == "accepted" else {"contrastive", "irrelevant"}
        if role not in allowed_roles:
            raise InteractionError("Exemplar role conflicts with candidate formal status")
        relations = value["trf_relations"]
        if not isinstance(relations, list):
            raise InteractionError("trf_relations must be a list")
        related: set[str] = set()
        for relation in relations:
            relation = _strict_keys(relation, {"normalized_text", "relation"}, "TRF relation")
            text = relation["normalized_text"]
            if text not in active or text in related or relation["relation"] not in RELATIONS:
                raise InteractionError("TRF relation is unknown, duplicated, or invalid")
            related.add(text)
        codes = value["reason_codes"]
        if not isinstance(codes, list) or not codes or len(codes) != len(set(codes)) or not set(codes) <= REASON_CODES:
            raise InteractionError("Exemplar reason_codes are invalid")
        reason = value["reason"]
        if not isinstance(reason, str) or not reason.strip() or len(reason) > max_reason:
            raise InteractionError("Exemplar reason is invalid")
        submitted_supports = {
            relation["normalized_text"]
            for relation in relations
            if relation["relation"] == "supports"
        }
        span_matches = {
            text
            for text in active
            if any(
                text.casefold() in span.casefold()
                for span in candidates[demo_idx].get("skill_spans", [])
            )
        }
        calibration_actions: list[str] = []
        if status == "accepted":
            invalid_supports = submitted_supports - span_matches
            relations[:] = [
                relation
                for relation in relations
                if relation["relation"] != "supports"
                or relation["normalized_text"] in span_matches
            ]
            if invalid_supports:
                calibration_actions.append("unsupported_trf_support_removed")
            existing_supports = {
                relation["normalized_text"]
                for relation in relations
                if relation["relation"] == "supports"
            }
            for text in previous_r["active_trfs"]:
                if text in span_matches and text not in existing_supports:
                    relations.append({"normalized_text": text, "relation": "supports"})
                    calibration_actions.append("skill_span_trf_support_added")
            relation_pairs = {
                (relation["normalized_text"], relation["relation"])
                for relation in relations
            }
            if any(kind == "supports" for _, kind in relation_pairs):
                if score < 4:
                    calibration_actions.append("accepted_trf_support_floor_applied")
                score = max(score, 4)
                role = "supporting"
            elif invalid_supports:
                score = min(score, 3)
                role = "irrelevant"
                calibration_actions.append("unsupported_trf_promotion_rejected")
            has_valid_support = any(
                kind == "supports" for _, kind in relation_pairs
            )
            exact_target_span = any(
                _contains_phrase(record["sentence"], span)
                for span in candidates[demo_idx].get("skill_spans", [])
            )
            strong_retrieval_alignment = (
                float(candidates[demo_idx]["similarity"])
                >= ACCEPTED_ALIGNMENT_SIMILARITY_FLOOR
            )
            strong_alignment = (
                has_valid_support or exact_target_span or strong_retrieval_alignment
            )
            if strong_alignment:
                if score < 4:
                    calibration_actions.append("accepted_target_alignment_floor_applied")
                score = max(score, 4)
                role = "supporting"
                if "BOUNDARY_TRANSFERABLE" not in codes:
                    codes.append("BOUNDARY_TRANSFERABLE")
            elif previous_score == 4 and score >= 4:
                score = 3
                role = "irrelevant"
                codes = [
                    code
                    for code in codes
                    if code not in {"TRF_ALIGNED", "BOUNDARY_TRANSFERABLE"}
                ]
                if not codes:
                    codes = ["SEMANTIC_ONLY"]
                calibration_actions.append("unsubstantiated_hard_gate_score_rejected")
        else:
            relation_pairs = {
                (relation["normalized_text"], relation["relation"])
                for relation in relations
            }
        if status == "negative" and previous_score >= 4:
            has_boundary = "BOUNDARY_TRANSFERABLE" in codes
            has_specific_relation = any(kind in {"supports", "contrasts"} for _, kind in relation_pairs)
            if not has_boundary and not has_specific_relation:
                strong_prior_boundary = (
                    previous_role == "contrastive"
                    and "NEGATIVE_CONTRAST" in prior["reason_codes"]
                    and float(candidates[demo_idx]["similarity"]) >= BOUNDARY_SIMILARITY_FLOOR
                )
                if strong_prior_boundary:
                    score = max(score, previous_score, 4)
                    role = "contrastive"
                    if "NEGATIVE_CONTRAST" not in codes:
                        codes.append("NEGATIVE_CONTRAST")
                    if "BOUNDARY_TRANSFERABLE" not in codes:
                        codes.append("BOUNDARY_TRANSFERABLE")
                    calibration_actions.append("strong_negative_boundary_retained")
                else:
                    score = min(score, 3)
                    role = "irrelevant"
                    codes = [code for code in codes if code != "NEGATIVE_CONTRAST"]
                    if not codes:
                        codes = ["SEMANTIC_ONLY"]
                    calibration_actions.append("generic_high_negative_penalized")
        if (
            status == "negative"
            and previous_score <= 3
            and previous_role == "contrastive"
            and not set(codes) & {"TASK_MISMATCH", "REDUNDANT"}
        ):
            role = "contrastive"
            if "NEGATIVE_CONTRAST" not in codes:
                codes.append("NEGATIVE_CONTRAST")
            calibration_actions.append("low_negative_contrast_preserved")
        if status == "negative" and "NEGATIVE_CONTRAST" in codes:
            role = "contrastive"
        if "TRF_ALIGNED" in codes and not relations:
            codes = [code for code in codes if code != "TRF_ALIGNED"]
            if not codes:
                codes = ["TYPE_ALIGNED" if status == "accepted" else "SEMANTIC_ONLY"]
            calibration_actions.append("unsubstantiated_trf_aligned_removed")
        adjustment = score - previous_score
        value["previous_helpfulness_score"] = previous_score
        value["previous_role"] = previous_role
        value["trf_effect"] = "boost" if adjustment > 0 else ("penalize" if adjustment < 0 else "none")
        value["score_adjustment"] = adjustment
        value["helpfulness_score"] = score
        value["role"] = role
        value["reason_codes"] = codes
        value["calibration_actions"] = list(dict.fromkeys(calibration_actions))
    if seen != set(candidates):
        raise InteractionError("Exemplar proposal does not cover the exact candidate set")
    if not isinstance(root["overall_reason"], str) or not root["overall_reason"].strip() or len(root["overall_reason"]) > max_reason:
        raise InteractionError("Exemplar overall_reason is invalid")
    unresolved = root["unresolved_issues"]
    if not isinstance(unresolved, list) or len(unresolved) != len(set(unresolved)) or not set(unresolved) <= EXEMPLAR_UNRESOLVED:
        raise InteractionError("Exemplar unresolved_issues are invalid")
    ordered = {item["demo_idx"]: item for item in revised}
    result = copy.deepcopy(root)
    result["revised_judgments"] = [ordered[item["demo_idx"]] for item in record["candidates"]]
    return result
