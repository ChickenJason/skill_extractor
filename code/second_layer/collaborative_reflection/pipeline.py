"""Deterministic input, prompt, parse, selection, interaction, and context logic."""

from __future__ import annotations

import copy
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.contracts import (  # noqa: E402
    record_identity,
    record_source_sha256,
    validate_unique_identities,
)
from common.io_utils import sha256_json  # noqa: E402
from instance_discriminator.pipeline import REASON_CODES, apply_hard_gate  # noqa: E402
from second_layer.collaborative_reflection.common import (  # noqa: E402
    INPUT_VERSION,
    PIPELINE_VERSION,
    TRF_SEMANTIC_CONTRACT_PATH,
    ReflectionError,
)


TRF_DECISION_KEYS = {
    "normalized_text", "action", "confidence", "evidence_sources",
    "evidence_demo_ids", "reason_codes", "reason",
}
ADDED_TRF_KEYS = {
    "text", "semantic_class", "confidence", "evidence_sources", "evidence_demo_ids",
    "reason_codes", "reason",
}
TRF_RESPONSE_KEYS = {
    "revised_entity_types", "original_trf_decisions", "added_trfs",
    "overall_reason", "unresolved_issues",
}
EXEMPLAR_JUDGMENT_KEYS = {
    "demo_idx", "helpfulness_score", "role", "trf_relations",
    "reason_codes", "reason",
}
EXEMPLAR_RESPONSE_KEYS = {"revised_judgments", "overall_reason", "unresolved_issues"}
TRF_RELATION_KEYS = {"normalized_text", "relation"}
TRF_ACTIONS = {"keep", "drop"}
ROLES = {"supporting", "contrastive", "irrelevant"}
RELATIONS = {"supports", "contrasts", "unrelated"}
EVIDENCE_SOURCES = {"TARGET_TEXT", "EXEMPLARS"}
TRF_REASON_CODES = {
    "TARGET_EXPLICIT_SUPPORT",
    "TARGET_IMPLICIT_SUPPORT",
    "EXEMPLAR_SUPPORT",
    "EXEMPLAR_CONTRADICTION",
    "TOO_GENERIC",
    "TYPE_MISMATCH",
    "MISSING_SKILL_CUE",
    "BOUNDARY_EVIDENCE",
    "OPEN_VOCABULARY_INFERENCE",
}
TRF_UNRESOLVED = {
    "TYPE_TRF_CONFLICT",
    "INSUFFICIENT_EVIDENCE",
    "CONFLICTING_EXEMPLARS",
    "POTENTIAL_SKILL_SPAN",
}
EXEMPLAR_UNRESOLVED = {"INSUFFICIENT_TRF_EVIDENCE", "CONFLICTING_TRF_EVIDENCE"}
ADDITION_SEMANTIC_CLASSES = {"domain_feature", "skill_type_cue"}
EXCLUDED_SEMANTIC_CLASSES = {
    "quality_or_trait",
    "descriptive_modifier",
    "task_or_object",
    "skill_span_or_paraphrase",
}


def _strict_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ReflectionError(f"{label} must be an integer")
    return value


def _strict_list_of_enums(value: Any, allowed: set[str], label: str, *, empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not empty and not value):
        raise ReflectionError(f"{label} must be a {'possibly empty ' if empty else 'non-empty '}list")
    if not all(isinstance(item, str) and item in allowed for item in value):
        raise ReflectionError(f"{label} contains an unsupported value")
    if len(value) != len(set(value)):
        raise ReflectionError(f"{label} contains duplicates")
    return list(value)


def _reason(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ReflectionError(f"{label} must be non-empty and at most {maximum} characters")
    return value


def normalize_trf(value: Any) -> str:
    if not isinstance(value, str):
        raise ReflectionError("TRF text must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized:
        raise ReflectionError("TRF text is empty after normalization")
    return normalized


def load_trf_semantic_contract() -> dict[str, Any]:
    try:
        value = json.loads(TRF_SEMANTIC_CONTRACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReflectionError(f"Cannot load TRF semantic contract: {error}") from error
    required = {
        "schema_version", "contract_id", "definition", "allowed_addition_classes",
        "excluded_classes", "human_labeled_examples",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ReflectionError("TRF semantic contract has missing or unknown keys")
    if value["schema_version"] != 1 or value["contract_id"] != "trf-addition-semantics-v1":
        raise ReflectionError("Unsupported TRF semantic contract version")
    if set(value["allowed_addition_classes"]) != ADDITION_SEMANTIC_CLASSES:
        raise ReflectionError("TRF semantic contract allowed classes are invalid")
    if set(value["excluded_classes"]) != EXCLUDED_SEMANTIC_CLASSES:
        raise ReflectionError("TRF semantic contract excluded classes are invalid")
    if not all(
        isinstance(item, str) and item.strip()
        for group in (value["allowed_addition_classes"], value["excluded_classes"])
        for item in group.values()
    ):
        raise ReflectionError("TRF semantic class definitions must be non-empty strings")
    examples = value["human_labeled_examples"]
    if not isinstance(examples, list) or not examples:
        raise ReflectionError("TRF semantic contract requires human-labeled examples")
    seen: set[str] = set()
    for position, item in enumerate(examples, start=1):
        if not isinstance(item, dict) or set(item) != {
            "text", "semantic_class", "is_trf", "rationale"
        }:
            raise ReflectionError(f"TRF semantic example {position} has an invalid shape")
        text = normalize_trf(item["text"])
        key = text.casefold()
        if key in seen:
            raise ReflectionError(f"Duplicate TRF semantic example: {text!r}")
        seen.add(key)
        if type(item["is_trf"]) is not bool:
            raise ReflectionError(f"TRF semantic example {text!r} has a non-boolean label")
        expected_classes = ADDITION_SEMANTIC_CLASSES if item["is_trf"] else EXCLUDED_SEMANTIC_CLASSES
        if item["semantic_class"] not in expected_classes:
            raise ReflectionError(f"TRF semantic example {text!r} has an inconsistent class")
        _reason(item["rationale"], f"semantic rationale for {text}", 500)
    return value


def _semantic_example(text: str) -> dict[str, Any] | None:
    key = normalize_trf(text).casefold()
    return next(
        (
            item for item in load_trf_semantic_contract()["human_labeled_examples"]
            if normalize_trf(item["text"]).casefold() == key
        ),
        None,
    )


def _index(records: list[dict[str, Any]], label: str) -> dict[tuple[str, str], dict[str, Any]]:
    validate_unique_identities(records, label)
    return {record_identity(record, label): record for record in records}


def _align(base: dict[str, Any], record: dict[str, Any], label: str) -> None:
    identity = record_identity(base, "base context")
    if record_identity(record, label) != identity:
        raise ReflectionError(f"{label} identity mismatch for {identity!r}")
    if record.get("idx") != base.get("idx"):
        raise ReflectionError(f"{label} idx mismatch for {identity!r}")
    if record.get("sentence") != base.get("sentence"):
        raise ReflectionError(f"{label} sentence mismatch for {identity!r}")
    if record_source_sha256(record, label) != record_source_sha256(base, "base context"):
        raise ReflectionError(f"{label} source hash mismatch for {identity!r}")


def build_reflection_inputs(
    base_contexts: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    judgment_records: list[dict[str, Any]],
    *,
    candidate_count: int = 16,
) -> list[dict[str, Any]]:
    """Join the three frozen R0 artifacts and enforce their complete identity contract."""

    validate_unique_identities(base_contexts, "base contexts")
    candidate_by_id = _index(candidate_records, "candidate records")
    judgment_by_id = _index(judgment_records, "judgment records")
    base_ids = {record_identity(record, "base context") for record in base_contexts}
    if set(candidate_by_id) != base_ids or set(judgment_by_id) != base_ids:
        raise ReflectionError("Frozen candidate and judgment records must exactly cover base contexts")

    output: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    for base in sorted(base_contexts, key=lambda item: item.get("idx", -1)):
        identity = record_identity(base, "base context")
        if base.get("schema_version") != "second-layer-context-v1" or base.get("assembly_status") != "ready":
            raise ReflectionError(f"Unsupported or incomplete base context for {identity!r}")
        idx = _strict_int(base.get("idx"), "base idx")
        if idx in seen_indexes:
            raise ReflectionError(f"Duplicate base idx={idx}")
        seen_indexes.add(idx)
        candidate_record = candidate_by_id[identity]
        judgment_record = judgment_by_id[identity]
        _align(base, candidate_record, "candidate record")
        _align(base, judgment_record, "judgment record")
        if candidate_record.get("schema_version") != "candidate-instances-v1":
            raise ReflectionError(f"Unsupported candidate schema for {identity!r}")
        if judgment_record.get("schema_version") not in {
            "instance-judgments-v1",
            "instance-judgments-v2",
        }:
            raise ReflectionError(f"Unsupported judgment schema for {identity!r}")
        evidence = candidate_record.get("target_evidence")
        if not isinstance(evidence, dict) or evidence.get("feature_context") != "absent":
            raise ReflectionError(f"Baseline exemplar feature_context must be absent for {identity!r}")
        if evidence.get("entity_types") or evidence.get("trfs"):
            raise ReflectionError(f"Feature-absent baseline contains features for {identity!r}")
        candidates = candidate_record.get("candidates")
        judgments = judgment_record.get("judgments")
        if (
            not isinstance(candidates, list)
            or candidate_record.get("candidate_count") != candidate_count
            or len(candidates) != candidate_count
        ):
            raise ReflectionError(f"Exactly {candidate_count} candidates are required for {identity!r}")
        if (
            not isinstance(judgments, list)
            or judgment_record.get("candidate_count") != candidate_count
            or len(judgments) != candidate_count
        ):
            raise ReflectionError(f"Exactly {candidate_count} baseline judgments are required for {identity!r}")
        candidate_ids = [item.get("demo_idx") for item in candidates]
        judgment_ids = [item.get("demo_idx") for item in judgments]
        if any(type(item) is not int for item in candidate_ids) or len(set(candidate_ids)) != candidate_count:
            raise ReflectionError(f"Candidate IDs are invalid for {identity!r}")
        if set(judgment_ids) != set(candidate_ids) or len(set(judgment_ids)) != candidate_count:
            raise ReflectionError(f"Judgments must cover exactly the candidate IDs for {identity!r}")
        judgment_by_demo = {item["demo_idx"]: item for item in judgments}
        ordered_judgments = [copy.deepcopy(judgment_by_demo[item]) for item in candidate_ids]
        selected = base.get("exemplar_context", {}).get("selected")
        if not isinstance(selected, list):
            raise ReflectionError(f"Base selected exemplars are invalid for {identity!r}")
        selected_ids = [item.get("demo_idx") for item in selected]
        if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids).issubset(set(candidate_ids)):
            raise ReflectionError(f"Base selected exemplars are not a candidate subset for {identity!r}")
        if base["exemplar_context"].get("feature_context") != "absent":
            raise ReflectionError(f"Base context feature_context must remain absent for {identity!r}")
        trf_context = base.get("trf_context")
        if not isinstance(trf_context, dict) or not isinstance(trf_context.get("trfs"), list):
            raise ReflectionError(f"Base TRF context is invalid for {identity!r}")
        baseline_trfs = [normalize_trf(item.get("normalized_text")) for item in trf_context["trfs"]]
        if len(baseline_trfs) != len(set(baseline_trfs)):
            raise ReflectionError(f"Base TRFs contain duplicates for {identity!r}")
        output.append(
            {
                "schema_version": INPUT_VERSION,
                "dataset_id": identity[0],
                "record_id": identity[1],
                "source_sha256": base["source_sha256"],
                "idx": idx,
                "sentence": base["sentence"],
                "baseline_context": copy.deepcopy(base),
                "candidates": copy.deepcopy(candidates),
                "baseline_judgments": ordered_judgments,
                "baseline_selected_demo_ids": selected_ids,
            }
        )
    return output


def _compact_demos(record: dict[str, Any]) -> list[dict[str, Any]]:
    judgments = {item["demo_idx"]: item for item in record["baseline_judgments"]}
    selected = set(record["baseline_selected_demo_ids"])
    return [
        {
            "demo_idx": item["demo_idx"],
            "sentence": item["sentence"],
            "status": item["status"],
            "skill_spans": item["skill_spans"],
            "pseudo_trfs": item["pseudo_trfs"],
            "similarity": item["similarity"],
            "reliability": item["existence_score"],
            "baseline_judgment": judgments[item["demo_idx"]],
            "baseline_selected": item["demo_idx"] in selected,
        }
        for item in record["candidates"]
    ]


def build_prompt(
    record: dict[str, Any],
    branch: str,
    max_characters: int,
    max_reason_characters: int = 240,
) -> dict[str, Any]:
    if branch not in {"trf", "exemplar"}:
        raise ReflectionError(f"Unknown prompt branch: {branch}")
    baseline_trf = record["baseline_context"]["trf_context"]
    target = {
        "sentence": record["sentence"],
        "baseline_entity_types": baseline_trf["entity_types"],
        "baseline_trfs": baseline_trf["trfs"],
        "baseline_trf_status": baseline_trf["status"],
        "baseline_trf_review_reasons": baseline_trf["review_reasons"],
    }
    demonstrations = _compact_demos(record)
    if branch == "trf":
        semantic_contract = load_trf_semantic_contract()
        system = (
            "You are the TRF reflector in a single-round skill-extraction review. A TRF is a token or "
            "phrase that functions as type-related evidence associated with Skill entities; it is not "
            "the target Skill entity span and must not be a renamed or paraphrased target skill. Use "
            "the target text and all 16 frozen exemplar judgments to revise only entity types and "
            "open-vocabulary TRFs. Cover every original TRF once and cite evidence. If a proposed TRF "
            "could itself be the target Skill span, a quality/trait, a descriptive modifier, or a task/object, "
            "omit it and report POTENTIAL_SKILL_SPAN when applicable. Every addition must be classified as "
            "domain_feature or skill_type_cue under the supplied human semantic contract. An original TRF "
            "that matches a human-labeled excluded example must be dropped. You cannot "
            "see the exemplar reflector's new decisions. Keep every reason and overall_reason concise. "
            "Return only the exact JSON object."
        )
        required = {
            "revised_entity_types": "[] or [\"Skill\"]",
            "original_trf_decisions": [
                {
                    "normalized_text": "an original TRF, each exactly once in original order",
                    "action": "keep | drop",
                    "confidence": "integer 1..5",
                    "evidence_sources": sorted(EVIDENCE_SOURCES),
                    "evidence_demo_ids": "unique candidate demo_idx values; [] is allowed",
                    "reason_codes": sorted(TRF_REASON_CODES),
                    "reason": f"one short sentence, at most {max_reason_characters} characters",
                }
            ],
            "added_trfs": [
                {
                    "text": "non-empty type-related feature, never a target Skill span or paraphrase",
                    "semantic_class": "domain_feature | skill_type_cue",
                    "confidence": "integer 1..5",
                    "evidence_sources": sorted(EVIDENCE_SOURCES),
                    "evidence_demo_ids": "unique candidate demo_idx values; [] is allowed",
                    "reason_codes": sorted(TRF_REASON_CODES),
                    "reason": f"one short sentence, at most {max_reason_characters} characters",
                }
            ],
            "overall_reason": f"one short sentence, at most {max_reason_characters} characters",
            "unresolved_issues": sorted(TRF_UNRESOLVED),
        }
    else:
        system = (
            "You are the exemplar reflector in a single-round skill-extraction review. Rejudge all 16 "
            "frozen candidates using only the baseline target TRFs. Output exactly one judgment for each "
            "candidate in input order. Do not select the final subset and do not invent TRFs. You cannot "
            "see the TRF reflector's new decisions. When baseline TRFs are empty, every trf_relations "
            "list must be empty. Keep every reason and overall_reason concise. Return only the exact "
            "JSON object."
        )
        required = {
            "revised_judgments": [
                {
                    "demo_idx": "integer copied from input",
                    "helpfulness_score": "integer 1..5",
                    "role": "supporting | contrastive | irrelevant",
                    "trf_relations": [
                        {
                            "normalized_text": "one baseline TRF only",
                            "relation": "supports | contrasts | unrelated",
                        }
                    ],
                    "reason_codes": sorted(REASON_CODES),
                    "reason": f"one short sentence, at most {max_reason_characters} characters",
                }
            ],
            "overall_reason": f"one short sentence, at most {max_reason_characters} characters",
            "unresolved_issues": sorted(EXEMPLAR_UNRESOLVED),
        }
    constraints: dict[str, Any] = {
        "candidate_count": 16,
        "all_reason_fields_max_characters": max_reason_characters,
        "recommended_reason_target_characters": min(160, max_reason_characters),
        "single_round_reads_frozen_r0_only": True,
    }
    if branch == "trf":
        constraints["added_trfs_must_not_be_skill_spans_or_paraphrases"] = True
        constraints["added_trfs_must_not_be_qualities_modifiers_tasks_or_objects"] = True
        constraints["human_labeled_excluded_original_trfs_must_be_dropped"] = True
        constraints["trf_semantic_contract"] = semantic_contract
    else:
        constraints.update(
            {
                "accepted_roles": ["supporting", "irrelevant"],
                "negative_roles": ["contrastive", "irrelevant"],
                "trf_relations_contract": (
                    "every_list_must_be_empty"
                    if not baseline_trf["trfs"]
                    else "may_reference_only_the_supplied_baseline_trfs"
                ),
            }
        )
    body = {
        "required_output": required,
        "constraints": constraints,
        "input": {"target": target, "demonstrations": demonstrations},
    }
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(body, ensure_ascii=False, separators=(",", ":"))},
    ]
    character_count = sum(len(item["content"]) for item in messages)
    if character_count > max_characters:
        raise ReflectionError(
            f"{branch} prompt for idx={record['idx']} has {character_count} characters, above {max_characters}"
        )
    prompt = {
        "schema_version": f"{branch}-reflection-prompt-v1",
        "branch": branch,
        "dataset_id": record["dataset_id"],
        "record_id": record["record_id"],
        "source_sha256": record["source_sha256"],
        "idx": record["idx"],
        "sentence": record["sentence"],
        "required_demo_ids": [item["demo_idx"] for item in record["candidates"]],
        "required_original_trfs": [item["normalized_text"] for item in baseline_trf["trfs"]],
        "messages": messages,
        "prompt_sha256": sha256_json(messages),
        "character_count": character_count,
    }
    if branch == "trf":
        prompt["trf_semantic_contract_id"] = semantic_contract["contract_id"]
        prompt["trf_semantic_contract"] = semantic_contract
    return prompt


def build_repair_messages(
    prompt: dict[str, Any],
    branch: str,
    error: str,
    prior_content: str,
    max_reason_characters: int = 240,
) -> list[dict[str, str]]:
    contract: dict[str, Any]
    if branch == "trf":
        contract = {
            "top_level_keys": sorted(TRF_RESPONSE_KEYS),
            "required_original_trfs_in_order": prompt["required_original_trfs"],
            "allowed_actions": sorted(TRF_ACTIONS),
            "allowed_demo_ids": prompt["required_demo_ids"],
            "allowed_evidence_sources": sorted(EVIDENCE_SOURCES),
            "allowed_reason_codes": sorted(TRF_REASON_CODES),
            "allowed_added_trf_semantic_classes": sorted(ADDITION_SEMANTIC_CLASSES),
            "trf_semantic_contract": prompt["trf_semantic_contract"],
            "all_reason_and_overall_reason_max_characters": max_reason_characters,
            "repair_reason_target_characters": min(160, max_reason_characters),
            "do_not_output_skill_spans_as_trfs": True,
            "human_labeled_excluded_original_trfs_must_be_dropped": True,
        }
    else:
        contract = {
            "top_level_keys": sorted(EXEMPLAR_RESPONSE_KEYS),
            "required_demo_ids_in_order": prompt["required_demo_ids"],
            "allowed_roles": sorted(ROLES),
            "accepted_roles": ["supporting", "irrelevant"],
            "negative_roles": ["contrastive", "irrelevant"],
            "allowed_baseline_trfs": prompt["required_original_trfs"],
            "allowed_relations": sorted(RELATIONS),
            "allowed_reason_codes": sorted(REASON_CODES),
            "all_reason_and_overall_reason_max_characters": max_reason_characters,
            "repair_reason_target_characters": min(160, max_reason_characters),
            "trf_relations_when_allowed_baseline_trfs_empty": "every_list_must_be_empty",
        }
    repair = {
        "repair_request": {
            "validation_error": error,
            "contract": contract,
            "instruction": (
                "Rewrite the entire supplied JSON so every constraint is satisfied. Use short audit "
                "sentences, preferably within the repair target length. Return one valid JSON object "
                "and no commentary."
            ),
            "invalid_response": prior_content,
        }
    }
    return [
        *copy.deepcopy(prompt["messages"]),
        {"role": "assistant", "content": prior_content},
        {"role": "user", "content": json.dumps(repair, ensure_ascii=False, separators=(",", ":"))},
    ]


def _parse_json_object(content: str, expected_keys: set[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError) as error:
        raise ReflectionError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ReflectionError(f"{label} must contain exactly {sorted(expected_keys)}")
    return value


def _evidence(
    item: dict[str, Any], candidate_ids: set[int], max_reason_characters: int, label: str
) -> dict[str, Any]:
    sources = _strict_list_of_enums(item.get("evidence_sources"), EVIDENCE_SOURCES, f"{label}.evidence_sources")
    ids = item.get("evidence_demo_ids")
    if not isinstance(ids, list) or any(type(value) is not int for value in ids):
        raise ReflectionError(f"{label}.evidence_demo_ids must be an integer list")
    if len(ids) != len(set(ids)) or not set(ids).issubset(candidate_ids):
        raise ReflectionError(f"{label}.evidence_demo_ids are duplicated or unknown")
    if ids and "EXEMPLARS" not in sources:
        raise ReflectionError(f"{label} cites demos without the EXEMPLARS source")
    if "EXEMPLARS" in sources and not ids:
        raise ReflectionError(f"{label} cites EXEMPLARS without demo IDs")
    confidence = _strict_int(item.get("confidence"), f"{label}.confidence")
    if not 1 <= confidence <= 5:
        raise ReflectionError(f"{label}.confidence must be 1..5")
    return {
        "confidence": confidence,
        "evidence_sources": sources,
        "evidence_demo_ids": ids,
        "reason_codes": _strict_list_of_enums(item.get("reason_codes"), TRF_REASON_CODES, f"{label}.reason_codes"),
        "reason": _reason(item.get("reason"), f"{label}.reason", max_reason_characters),
    }


def parse_trf_response(
    content: str, record: dict[str, Any], *, max_reason_characters: int = 240
) -> dict[str, Any]:
    value = _parse_json_object(content, TRF_RESPONSE_KEYS, "TRF reflection response")
    entity_types = value["revised_entity_types"]
    if entity_types not in ([], ["Skill"]):
        raise ReflectionError("revised_entity_types must be [] or ['Skill']")
    baseline = record["baseline_context"]["trf_context"]["trfs"]
    expected = [normalize_trf(item.get("normalized_text")) for item in baseline]
    candidate_ids = {item["demo_idx"] for item in record["candidates"]}
    raw_decisions = value["original_trf_decisions"]
    if not isinstance(raw_decisions, list) or len(raw_decisions) != len(expected):
        raise ReflectionError("original_trf_decisions must exactly cover original TRFs")
    decisions: list[dict[str, Any]] = []
    for position, (item, expected_text) in enumerate(zip(raw_decisions, expected), start=1):
        if not isinstance(item, dict) or set(item) != TRF_DECISION_KEYS:
            raise ReflectionError(f"Original TRF decision {position} has missing or unknown keys")
        text = normalize_trf(item["normalized_text"])
        if text != expected_text:
            raise ReflectionError("Original TRF decisions must preserve original order and text")
        if item.get("action") not in TRF_ACTIONS:
            raise ReflectionError(f"Invalid original TRF action for {text!r}")
        labeled = _semantic_example(text)
        if labeled is not None and not labeled["is_trf"] and item["action"] != "drop":
            raise ReflectionError(
                f"Original TRF {text!r} is a human-labeled non-TRF and must be dropped"
            )
        decisions.append(
            {"normalized_text": text, "action": item["action"], **_evidence(item, candidate_ids, max_reason_characters, f"decision {text}")}
        )
    raw_added = value["added_trfs"]
    if not isinstance(raw_added, list):
        raise ReflectionError("added_trfs must be a list")
    added: list[dict[str, Any]] = []
    seen = set(expected)
    for position, item in enumerate(raw_added, start=1):
        if not isinstance(item, dict) or set(item) != ADDED_TRF_KEYS:
            raise ReflectionError(f"Added TRF {position} has missing or unknown keys")
        text = normalize_trf(item["text"])
        if text in seen:
            raise ReflectionError(f"Duplicate original or added TRF: {text!r}")
        seen.add(text)
        semantic_class = item.get("semantic_class")
        if semantic_class not in ADDITION_SEMANTIC_CLASSES:
            raise ReflectionError(
                f"Added TRF {text!r} must have one allowed semantic_class"
            )
        labeled = _semantic_example(text)
        if labeled is not None and not labeled["is_trf"]:
            raise ReflectionError(
                f"Added TRF {text!r} is a human-labeled non-TRF ({labeled['semantic_class']})"
            )
        if labeled is not None and labeled["semantic_class"] != semantic_class:
            raise ReflectionError(
                f"Added TRF {text!r} must use human-labeled class {labeled['semantic_class']!r}"
            )
        added.append(
            {
                "text": text,
                "semantic_class": semantic_class,
                **_evidence(item, candidate_ids, max_reason_characters, f"added TRF {text}"),
            }
        )
    unresolved = _strict_list_of_enums(value["unresolved_issues"], TRF_UNRESOLVED, "unresolved_issues", empty=True)
    return {
        "revised_entity_types": list(entity_types),
        "original_trf_decisions": decisions,
        "added_trfs": added,
        "overall_reason": _reason(value["overall_reason"], "overall_reason", max_reason_characters),
        "unresolved_issues": unresolved,
    }


def parse_exemplar_response(
    content: str, record: dict[str, Any], *, max_reason_characters: int = 240
) -> dict[str, Any]:
    value = _parse_json_object(content, EXEMPLAR_RESPONSE_KEYS, "Exemplar reflection response")
    candidates = record["candidates"]
    expected_ids = [item["demo_idx"] for item in candidates]
    candidate_by_id = {item["demo_idx"]: item for item in candidates}
    baseline_trfs = [
        normalize_trf(item["normalized_text"])
        for item in record["baseline_context"]["trf_context"]["trfs"]
    ]
    baseline_set = set(baseline_trfs)
    raw = value["revised_judgments"]
    if not isinstance(raw, list) or len(raw) != len(expected_ids):
        raise ReflectionError(f"revised_judgments must contain exactly {len(expected_ids)} items")
    parsed: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for position, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or set(item) != EXEMPLAR_JUDGMENT_KEYS:
            raise ReflectionError(f"Revised judgment {position} has missing or unknown keys")
        demo_idx = _strict_int(item.get("demo_idx"), "demo_idx")
        if demo_idx not in candidate_by_id or demo_idx in seen_ids:
            raise ReflectionError(f"Unknown or duplicate revised demo_idx={demo_idx}")
        if demo_idx != expected_ids[position - 1]:
            raise ReflectionError("Revised judgments must preserve candidate order")
        seen_ids.add(demo_idx)
        score = _strict_int(item.get("helpfulness_score"), "helpfulness_score")
        if not 1 <= score <= 5:
            raise ReflectionError(f"helpfulness_score for demo_idx={demo_idx} must be 1..5")
        role = item.get("role")
        if role not in ROLES:
            raise ReflectionError(f"Invalid role for demo_idx={demo_idx}")
        status = candidate_by_id[demo_idx]["status"]
        if status == "accepted" and role not in {"supporting", "irrelevant"}:
            raise ReflectionError(f"Accepted demo_idx={demo_idx} cannot be {role}")
        if status == "negative" and role not in {"contrastive", "irrelevant"}:
            raise ReflectionError(f"Negative demo_idx={demo_idx} cannot be {role}")
        relations = item.get("trf_relations")
        if not isinstance(relations, list):
            raise ReflectionError(f"trf_relations for demo_idx={demo_idx} must be a list")
        parsed_relations: list[dict[str, str]] = []
        seen_relations: set[str] = set()
        for relation in relations:
            if not isinstance(relation, dict) or set(relation) != TRF_RELATION_KEYS:
                raise ReflectionError(f"Invalid TRF relation for demo_idx={demo_idx}")
            text = normalize_trf(relation["normalized_text"])
            if text not in baseline_set or text in seen_relations or relation.get("relation") not in RELATIONS:
                raise ReflectionError(f"Unknown, duplicate, or invalid TRF relation for demo_idx={demo_idx}")
            seen_relations.add(text)
            parsed_relations.append({"normalized_text": text, "relation": relation["relation"]})
        parsed.append(
            {
                "demo_idx": demo_idx,
                "helpfulness_score": score,
                "role": role,
                "trf_relations": parsed_relations,
                "reason_codes": _strict_list_of_enums(item.get("reason_codes"), set(REASON_CODES), f"reason_codes for demo_idx={demo_idx}"),
                "reason": _reason(item.get("reason"), f"reason for demo_idx={demo_idx}", max_reason_characters),
            }
        )
    if seen_ids != set(expected_ids):
        raise ReflectionError("Revised judgment IDs are incomplete")
    return {
        "revised_judgments": parsed,
        "overall_reason": _reason(value["overall_reason"], "overall_reason", max_reason_characters),
        "unresolved_issues": _strict_list_of_enums(value["unresolved_issues"], EXEMPLAR_UNRESOLVED, "unresolved_issues", empty=True),
    }


def build_trf_record(record: dict[str, Any], parsed: dict[str, Any], chat_model: str) -> dict[str, Any]:
    baseline_trfs = record["baseline_context"]["trf_context"]["trfs"]
    kept = [
        {**copy.deepcopy(source), "origin": "baseline"}
        for source, decision in zip(baseline_trfs, parsed["original_trf_decisions"])
        if decision["action"] == "keep"
    ]
    for order, item in enumerate(kept, start=1):
        item["first_seen_order"] = order
    sentence = record["sentence"]
    next_order = len(kept) + 1
    additions: list[dict[str, Any]] = []
    for offset, addition in enumerate(parsed["added_trfs"]):
        text = addition["text"]
        additions.append(
            {
                "raw_text": text,
                "text": text,
                "normalized_text": text,
                "first_seen_order": next_order + offset,
                "in_main_bank_exact": None,
                "in_main_bank_casefold": None,
                "appears_in_target_exact": text in sentence,
                "appears_in_target_casefold": text.casefold() in sentence.casefold(),
                "semantic_class": addition["semantic_class"],
                "semantic_validation": (
                    "human_validated" if _semantic_example(text) is not None else "model_classified"
                ),
                "origin": "added",
            }
        )
    trfs = kept + additions
    reasons = list(parsed["unresolved_issues"])
    if any(item["semantic_validation"] == "model_classified" for item in additions):
        reasons.append("added_trf_semantics_unverified")
    if parsed["revised_entity_types"] == [] and trfs:
        reasons.append("trfs_without_skill_type")
    if parsed["revised_entity_types"] == ["Skill"] and not trfs:
        reasons.append("skill_type_without_trfs")
    return {
        "schema_version": "trf-reflection-v1",
        "dataset_id": record["dataset_id"],
        "record_id": record["record_id"],
        "source_sha256": record["source_sha256"],
        "idx": record["idx"],
        "sentence": sentence,
        "status": "needs_review" if reasons else "complete",
        "revised_entity_types": parsed["revised_entity_types"],
        "revised_trfs": trfs,
        "original_trf_decisions": parsed["original_trf_decisions"],
        "added_trfs": parsed["added_trfs"],
        "overall_reason": parsed["overall_reason"],
        "unresolved_issues": parsed["unresolved_issues"],
        "review_reasons": reasons,
        "models": {"chat": chat_model},
    }


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return 1.0 if not union else round(len(left & right) / len(union), 10)


def build_exemplar_records(
    record: dict[str, Any], parsed: dict[str, Any], gate: dict[str, Any], chat_model: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    judgments = parsed["revised_judgments"]
    result = apply_hard_gate(record["candidates"], judgments, gate)
    baseline_by_id = {item["demo_idx"]: item for item in record["baseline_judgments"]}
    revised_by_id = {item["demo_idx"]: item for item in judgments}
    baseline_selected = set(record["baseline_selected_demo_ids"])
    revised_selected = {item["demo_idx"] for item in result["selected"]}
    reasons = list(parsed["unresolved_issues"])
    count = len(result["selected"])
    if count == 0:
        reasons.append("no_helpful_examples")
    elif count < gate["minimum_for_complete"]:
        reasons.append("insufficient_helpful_examples")
    changes = {
        "retained_demo_ids": sorted(baseline_selected & revised_selected),
        "promoted_demo_ids": sorted(revised_selected - baseline_selected),
        "removed_demo_ids": sorted(baseline_selected - revised_selected),
        "score_changed_demo_ids": sorted(
            demo_idx for demo_idx in revised_by_id
            if revised_by_id[demo_idx]["helpfulness_score"] != baseline_by_id[demo_idx]["helpfulness_score"]
        ),
        "role_changed_demo_ids": sorted(
            demo_idx for demo_idx in revised_by_id
            if revised_by_id[demo_idx]["role"] != baseline_by_id[demo_idx]["role"]
        ),
        "selection_jaccard": _jaccard(baseline_selected, revised_selected),
    }
    parsed_record = {
        "schema_version": "exemplar-reflection-v1",
        "dataset_id": record["dataset_id"],
        "record_id": record["record_id"],
        "source_sha256": record["source_sha256"],
        "idx": record["idx"],
        "sentence": record["sentence"],
        "status": "complete",
        "candidate_count": len(record["candidates"]),
        "revised_judgments": judgments,
        "overall_reason": parsed["overall_reason"],
        "unresolved_issues": parsed["unresolved_issues"],
        "models": {"chat": chat_model},
    }
    selected_record = {
        "schema_version": "exemplar-reflection-selection-v1",
        "dataset_id": record["dataset_id"],
        "record_id": record["record_id"],
        "source_sha256": record["source_sha256"],
        "idx": record["idx"],
        "sentence": record["sentence"],
        "status": "needs_review" if reasons else "complete",
        "feature_context": "baseline_trf_only",
        "candidate_count": len(record["candidates"]),
        "eligible_count": result["eligible_count"],
        "selected_count": count,
        "selected": result["selected"],
        "rejected_demo_ids": result["rejected_demo_ids"],
        "role_counts": result["role_counts"],
        **changes,
        "review_reasons": reasons,
        "models": {"chat": chat_model},
    }
    return parsed_record, selected_record


def build_interaction(
    record: dict[str, Any], trf: dict[str, Any], exemplar: dict[str, Any]
) -> dict[str, Any]:
    revised_by_id = {
        item["demo_idx"]: item
        for item in exemplar["revised_judgments"]
    }
    conflicts: list[dict[str, Any]] = []
    evidence_items = [
        {"trf": item["normalized_text"], "action": item["action"], "demo_ids": item["evidence_demo_ids"]}
        for item in trf["original_trf_decisions"]
        if item["action"] == "keep"
    ] + [
        {"trf": item["text"], "action": "add", "demo_ids": item["evidence_demo_ids"]}
        for item in trf["added_trfs"]
    ]
    for evidence in evidence_items:
        for demo_idx in evidence["demo_ids"]:
            judgment = revised_by_id[demo_idx]
            if judgment["role"] == "irrelevant":
                conflicts.append(
                    {
                        "type": "trf_evidence_demo_not_retained",
                        "trf": evidence["trf"],
                        "trf_action": evidence["action"],
                        "demo_idx": demo_idx,
                        "revised_role": judgment["role"],
                        "revised_helpfulness_score": judgment["helpfulness_score"],
                    }
                )
    dropped = {
        item["normalized_text"]
        for item in trf["original_trf_decisions"]
        if item["action"] == "drop"
    }
    for judgment in exemplar["revised_judgments"]:
        for relation in judgment.get("trf_relations", []):
            if relation["relation"] == "supports" and relation["normalized_text"] in dropped:
                conflicts.append(
                    {
                        "type": "selected_demo_supports_dropped_trf",
                        "trf": relation["normalized_text"],
                        "demo_idx": judgment["demo_idx"],
                    }
                )
    additions = [item["text"] for item in trf["added_trfs"]]
    review_reasons: list[dict[str, Any]] = []
    if trf["status"] == "needs_review":
        review_reasons.append({"branch": "trf", "reasons": trf["review_reasons"]})
    if exemplar["status"] == "needs_review":
        review_reasons.append({"branch": "exemplar", "reasons": exemplar["review_reasons"]})
    if conflicts:
        status = "conflicted"
    elif additions or review_reasons:
        status = "mixed"
    else:
        status = "aligned"
    return {
        "schema_version": "collaborative-reflection-interaction-v1",
        "dataset_id": record["dataset_id"],
        "record_id": record["record_id"],
        "source_sha256": record["source_sha256"],
        "idx": record["idx"],
        "sentence": record["sentence"],
        "status": status,
        "conflicts": conflicts,
        "unreviewed_added_trfs": additions,
        "review_reasons": review_reasons,
    }


def assemble_context_records(
    inputs: list[dict[str, Any]],
    trf_records: list[dict[str, Any]],
    exemplar_parsed: list[dict[str, Any]],
    exemplar_selected: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_index = _index(inputs, "reflection inputs")
    trf_index = _index(trf_records, "TRF reflections")
    parsed_index = _index(exemplar_parsed, "exemplar reflections")
    selected_index = _index(exemplar_selected, "exemplar selections")
    identities = set(input_index)
    if any(set(index) != identities for index in (trf_index, parsed_index, selected_index)):
        raise ReflectionError("Completed reflection branches must exactly cover all targets")
    contexts: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    for record in sorted(inputs, key=lambda item: item["idx"]):
        identity = record_identity(record, "reflection input")
        trf = trf_index[identity]
        parsed = parsed_index[identity]
        selected = selected_index[identity]
        for candidate, label in ((trf, "TRF reflection"), (parsed, "exemplar reflection"), (selected, "exemplar selection")):
            _align(record, candidate, label)
        if trf["status"] not in {"complete", "needs_review"} or selected["status"] not in {"complete", "needs_review"}:
            raise ReflectionError(f"Incomplete reflected record for {identity!r}")
        merged_exemplar = {**copy.deepcopy(parsed), **copy.deepcopy(selected)}
        interaction = build_interaction(record, trf, merged_exemplar)
        interactions.append(interaction)
        contexts.append(
            {
                "schema_version": PIPELINE_VERSION,
                "dataset_id": record["dataset_id"],
                "record_id": record["record_id"],
                "source_sha256": record["source_sha256"],
                "idx": record["idx"],
                "sentence": record["sentence"],
                "baseline_context": copy.deepcopy(record["baseline_context"]),
                "reflected_context": {
                    "trf_context": copy.deepcopy(trf),
                    "exemplar_context": merged_exemplar,
                },
                "interaction": copy.deepcopy(interaction),
                "provenance": copy.deepcopy(provenance),
                "assembly_status": "ready",
            }
        )
    return contexts, interactions


def context_summary(records: list[dict[str, Any]], branch_metrics: dict[str, Any]) -> dict[str, Any]:
    trf_contexts = [item["reflected_context"]["trf_context"] for item in records]
    exemplar_contexts = [item["reflected_context"]["exemplar_context"] for item in records]
    baseline_empty = [item for item in records if not item["baseline_context"]["trf_context"]["trfs"]]
    keep = sum(
        decision["action"] == "keep"
        for context in trf_contexts for decision in context["original_trf_decisions"]
    )
    drop = sum(
        decision["action"] == "drop"
        for context in trf_contexts for decision in context["original_trf_decisions"]
    )
    add = sum(len(context["added_trfs"]) for context in trf_contexts)
    jaccards = [context["selection_jaccard"] for context in exemplar_contexts]
    statuses = {name: sum(item["interaction"]["status"] == name for item in records) for name in ("aligned", "mixed", "conflicted")}
    return {
        "schema_version": "collaborative-reflection-summary-v1",
        "status": "completed",
        "target_count": len(records),
        "ready_count": sum(item["assembly_status"] == "ready" for item in records),
        "baseline_empty_trf_count": len(baseline_empty),
        "baseline_empty_trf_supplemented": sum(
            bool(item["reflected_context"]["trf_context"]["added_trfs"]) for item in baseline_empty
        ),
        "trf_decisions": {"keep": keep, "drop": drop, "add": add},
        "entity_type_changed": sum(
            item["baseline_context"]["trf_context"]["entity_types"]
            != item["reflected_context"]["trf_context"]["revised_entity_types"]
            for item in records
        ),
        "exemplar_score_changed": sum(len(item["score_changed_demo_ids"]) for item in exemplar_contexts),
        "exemplar_role_changed": sum(len(item["role_changed_demo_ids"]) for item in exemplar_contexts),
        "exemplar_promoted": sum(len(item["promoted_demo_ids"]) for item in exemplar_contexts),
        "exemplar_removed": sum(len(item["removed_demo_ids"]) for item in exemplar_contexts),
        "mean_selection_jaccard": round(sum(jaccards) / len(jaccards), 10) if jaccards else None,
        "interaction_statuses": statuses,
        "trf_needs_review": sum(item["status"] == "needs_review" for item in trf_contexts),
        "exemplar_needs_review": sum(item["status"] == "needs_review" for item in exemplar_contexts),
        "branch_metrics": copy.deepcopy(branch_metrics),
        "accuracy_claimed": False,
    }
