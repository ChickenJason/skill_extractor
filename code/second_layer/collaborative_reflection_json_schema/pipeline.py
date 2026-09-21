"""Dynamic strict JSON Schemas and wrapper-resistant reflection prompts."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import sha256_json  # noqa: E402
from second_layer.collaborative_reflection import pipeline as legacy  # noqa: E402
from second_layer.collaborative_reflection.common import ReflectionError  # noqa: E402


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _enum_array(values: set[str], *, allow_empty: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "array",
        "items": {"type": "string", "enum": sorted(values)},
    }
    if not allow_empty:
        result["minItems"] = 1
    return result


def _reason(maximum: int) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def _demo_ids(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "integer",
            "enum": [item["demo_idx"] for item in record["candidates"]],
        },
    }


def _evidence_properties(record: dict[str, Any], maximum: int) -> dict[str, Any]:
    return {
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
        "evidence_sources": _enum_array(legacy.EVIDENCE_SOURCES, allow_empty=False),
        "evidence_demo_ids": _demo_ids(record),
        "reason_codes": _enum_array(legacy.TRF_REASON_CODES, allow_empty=False),
        "reason": _reason(maximum),
    }


def build_response_schema(
    record: dict[str, Any], branch: str, max_reason_characters: int
) -> dict[str, Any]:
    """Build the transport schema; semantic cross-field checks remain in the strict parser."""

    if branch not in {"trf", "exemplar"}:
        raise ReflectionError(f"Unknown JSON-Schema reflection branch: {branch}")
    if branch == "trf":
        original = [
            item["normalized_text"]
            for item in record["baseline_context"]["trf_context"]["trfs"]
        ]
        normalized: dict[str, Any] = {"type": "string", "minLength": 1}
        if original:
            normalized["enum"] = original
        decision = _object(
            {
                "normalized_text": normalized,
                "action": {"type": "string", "enum": sorted(legacy.TRF_ACTIONS)},
                **_evidence_properties(record, max_reason_characters),
            }
        )
        added = _object(
            {
                "text": {"type": "string", "minLength": 1},
                "semantic_class": {
                    "type": "string",
                    "enum": sorted(legacy.ADDITION_SEMANTIC_CLASSES),
                },
                **_evidence_properties(record, max_reason_characters),
            }
        )
        root = _object(
            {
                "revised_entity_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["Skill"]},
                    "maxItems": 1,
                },
                "original_trf_decisions": {
                    "type": "array",
                    "items": decision,
                    "minItems": len(original),
                    "maxItems": len(original),
                },
                "added_trfs": {"type": "array", "items": added},
                "overall_reason": _reason(max_reason_characters),
                "unresolved_issues": _enum_array(legacy.TRF_UNRESOLVED),
            }
        )
    else:
        baseline = [
            item["normalized_text"]
            for item in record["baseline_context"]["trf_context"]["trfs"]
        ]
        relation_text: dict[str, Any] = {"type": "string", "minLength": 1}
        if baseline:
            relation_text["enum"] = baseline
        relation = _object(
            {
                "normalized_text": relation_text,
                "relation": {"type": "string", "enum": sorted(legacy.RELATIONS)},
            }
        )
        relations: dict[str, Any] = {"type": "array", "items": relation}
        if not baseline:
            relations["maxItems"] = 0
        judgment = _object(
            {
                "demo_idx": {
                    "type": "integer",
                    "enum": [item["demo_idx"] for item in record["candidates"]],
                },
                "helpfulness_score": {"type": "integer", "minimum": 1, "maximum": 5},
                "role": {"type": "string", "enum": sorted(legacy.ROLES)},
                "trf_relations": relations,
                "reason_codes": _enum_array(set(legacy.REASON_CODES), allow_empty=False),
                "reason": _reason(max_reason_characters),
            }
        )
        root = _object(
            {
                "revised_judgments": {
                    "type": "array",
                    "items": judgment,
                    "minItems": len(record["candidates"]),
                    "maxItems": len(record["candidates"]),
                },
                "overall_reason": _reason(max_reason_characters),
                "unresolved_issues": _enum_array(legacy.EXEMPLAR_UNRESOLVED),
            }
        )
    return root


def build_prompt(
    record: dict[str, Any], branch: str, max_characters: int,
    max_reason_characters: int = 240,
) -> dict[str, Any]:
    base = legacy.build_prompt(record, branch, max_characters, max_reason_characters)
    body = json.loads(base["messages"][1]["content"])
    guidance = body.pop("required_output")
    body = {
        "output_instruction": (
            "Return the named fields directly at the JSON root. Do not wrap the result in "
            "required_output, output_field_guidance, result, data, or any other container."
        ),
        "output_field_guidance": guidance,
        "constraints": body["constraints"],
        "input": body["input"],
    }
    messages = copy.deepcopy(base["messages"])
    messages[0]["content"] += (
        " The response is constrained by a strict JSON Schema. Put the required fields directly "
        "at the root and never add a wrapper object."
    )
    messages[1]["content"] = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    character_count = sum(len(item["content"]) for item in messages)
    if character_count > max_characters:
        raise ReflectionError(
            f"{branch} JSON-Schema prompt for idx={record['idx']} has {character_count} characters"
        )
    schema = build_response_schema(record, branch, max_reason_characters)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": f"{branch}_reflection_v2",
            "strict": True,
            "schema": schema,
        },
    }
    prompt = copy.deepcopy(base)
    prompt.update(
        {
            "schema_version": f"{branch}-reflection-prompt-v2",
            "messages": messages,
            "prompt_sha256": sha256_json(messages),
            "character_count": character_count,
            "response_format": response_format,
            "response_format_sha256": sha256_json(response_format),
            "request_contract_sha256": request_sha256(messages, response_format),
        }
    )
    return prompt


def request_sha256(messages: list[dict[str, str]], response_format: dict[str, Any]) -> str:
    return sha256_json({"messages": messages, "response_format": response_format})


build_repair_messages = legacy.build_repair_messages
parse_trf_response = legacy.parse_trf_response
parse_exemplar_response = legacy.parse_exemplar_response
build_trf_record = legacy.build_trf_record
build_exemplar_records = legacy.build_exemplar_records
build_reflection_inputs = legacy.build_reflection_inputs
assemble_context_records = legacy.assemble_context_records
context_summary = legacy.context_summary
