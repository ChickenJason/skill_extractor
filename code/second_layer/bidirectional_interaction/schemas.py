"""Dynamic closed JSON Schemas for both synchronous interaction branches."""

from __future__ import annotations

from typing import Any

from instance_discriminator.pipeline import REASON_CODES
from second_layer.bidirectional_interaction.common import InteractionError
from second_layer.bidirectional_interaction.contracts import (
    ADDITION_CLASSES,
    EVIDENCE_SOURCES,
    EXEMPLAR_UNRESOLVED,
    RELATIONS,
    TRF_REASON_CODES,
    TRF_UNRESOLVED,
)


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _enum_array(
    values: set[str] | frozenset[str], *, allow_empty: bool = True
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "array",
        "items": {"type": "string", "enum": sorted(values)},
    }
    if not allow_empty:
        result["minItems"] = 1
    return result


def _reason(maximum: int) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def _demo_ids(record: dict[str, Any]) -> list[int]:
    return [item["demo_idx"] for item in record["candidates"]]


def _trf_evidence(record: dict[str, Any], maximum: int) -> dict[str, Any]:
    return {
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
        "evidence_sources": _enum_array(EVIDENCE_SOURCES, allow_empty=False),
        "evidence_demo_ids": {
            "type": "array",
            "items": {"type": "integer", "enum": _demo_ids(record)},
        },
        "reason_codes": _enum_array(TRF_REASON_CODES, allow_empty=False),
        "reason": _reason(maximum),
    }


def build_response_schema(
    record: dict[str, Any],
    branch: str,
    previous_r: dict[str, Any],
    *,
    max_reason_characters: int,
) -> dict[str, Any]:
    """Create a transport schema; cross-field semantics remain parser-enforced."""

    if branch == "trf":
        known = [item["normalized_text"] for item in previous_r["known_trfs"]]
        normalized: dict[str, Any] = {"type": "string", "minLength": 1}
        if known:
            normalized["enum"] = known
        decision = _object(
            {
                "normalized_text": normalized,
                "action": {"type": "string", "enum": ["drop", "keep"]},
                **_trf_evidence(record, max_reason_characters),
            }
        )
        addition = _object(
            {
                "text": {"type": "string", "minLength": 1, "maxLength": 160},
                "semantic_class": {"type": "string", "enum": sorted(ADDITION_CLASSES)},
                **_trf_evidence(record, max_reason_characters),
            }
        )
        root = _object(
            {
                "schema_version": {"type": "string", "const": "trf-interaction-proposal-v1"},
                "revised_entity_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["Skill"]},
                    "maxItems": 1,
                },
                "known_trf_decisions": {
                    "type": "array",
                    "items": decision,
                    "minItems": len(known),
                    "maxItems": len(known),
                },
                "added_trfs": {"type": "array", "items": addition},
                "overall_reason": _reason(max_reason_characters),
                "unresolved_issues": _enum_array(TRF_UNRESOLVED),
            }
        )
    elif branch == "exemplar":
        active = list(previous_r["active_trfs"])
        relation_text: dict[str, Any] = {"type": "string", "minLength": 1}
        if active:
            relation_text["enum"] = active
        relation = _object(
            {
                "normalized_text": relation_text,
                "relation": {"type": "string", "enum": sorted(RELATIONS)},
            }
        )
        relations: dict[str, Any] = {"type": "array", "items": relation}
        if not active:
            relations["maxItems"] = 0
        judgment = _object(
            {
                "demo_idx": {"type": "integer", "enum": _demo_ids(record)},
                "helpfulness_score": {"type": "integer", "minimum": 1, "maximum": 5},
                "role": {"type": "string", "enum": ["contrastive", "irrelevant", "supporting"]},
                "trf_relations": relations,
                "reason_codes": _enum_array(REASON_CODES, allow_empty=False),
                "reason": _reason(max_reason_characters),
            }
        )
        root = _object(
            {
                "schema_version": {"type": "string", "const": "exemplar-interaction-proposal-v1"},
                "revised_judgments": {
                    "type": "array",
                    "items": judgment,
                    "minItems": len(record["candidates"]),
                    "maxItems": len(record["candidates"]),
                },
                "overall_reason": _reason(max_reason_characters),
                "unresolved_issues": _enum_array(EXEMPLAR_UNRESOLVED),
            }
        )
    else:
        raise InteractionError(f"Unknown interaction branch: {branch}")
    return root


def response_format(
    record: dict[str, Any],
    branch: str,
    previous_r: dict[str, Any],
    *,
    max_reason_characters: int,
) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{branch}_interaction_proposal_v1",
            "strict": True,
            "schema": build_response_schema(
                record,
                branch,
                previous_r,
                max_reason_characters=max_reason_characters,
            ),
        },
    }
