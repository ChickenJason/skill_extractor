"""Leakage-controlled round prompts and stable request hashes."""

from __future__ import annotations

import copy
import json
from typing import Any

from common.io_utils import sha256_json
from second_layer.bidirectional_interaction.common import InteractionError
from second_layer.bidirectional_interaction.schemas import response_format


OVERALL_REASON_PROMPT_MAX_CHARACTERS = 120


def request_sha256(messages: list[dict[str, str]], response: dict[str, Any]) -> str:
    return sha256_json({"messages": messages, "response_format": response})


def _public_trfs(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "normalized_text": item["normalized_text"],
            "active": item["active"],
            "origin": item["origin"],
            "first_seen_round": item["first_seen_round"],
        }
        for item in state["known_trfs"]
    ]


def _public_judgments(state: dict[str, Any]) -> list[dict[str, Any]]:
    public = []
    for item in state["judgments"]:
        value = {
            "demo_idx": item["demo_idx"],
            "helpfulness_score": item["helpfulness_score"],
            "role": item["role"],
            "reason_codes": item["reason_codes"],
            "trf_relations": item.get("trf_relations", []),
        }
        if "reason" in item:
            value["reason"] = item["reason"]
        public.append(value)
    return public


def build_prompt(
    record: dict[str, Any],
    branch: str,
    previous_r: dict[str, Any],
    previous_h: dict[str, Any],
    round_number: int,
    *,
    max_characters: int,
    max_reason_characters: int,
) -> dict[str, Any]:
    """Build one branch prompt from the same frozen R(t-1), H(t-1)."""

    overall_reason_maximum = min(
        max_reason_characters, OVERALL_REASON_PROMPT_MAX_CHARACTERS
    )
    identity = {
        "dataset_id": record["dataset_id"],
        "record_id": record["record_id"],
        "source_sha256": record["source_sha256"],
        "idx": record["idx"],
        "sentence": record["sentence"],
    }
    demonstrations = [
        {
            "demo_dataset_id": item["demo_dataset_id"],
            "demo_record_id": item["demo_record_id"],
            "demo_source_sha256": item["demo_source_sha256"],
            "demo_idx": item["demo_idx"],
            "sentence": item["sentence"],
            "status": item["status"],
            "skill_spans": item["skill_spans"],
            "pseudo_trfs": item["pseudo_trfs"],
            "similarity": item["similarity"],
            "existence_score": item["existence_score"],
        }
        for item in record["candidates"]
    ]
    frozen = {
        "round": round_number,
        "target": identity,
        "previous_trf_state": {
            "entity_types": previous_r["entity_types"],
            "known_trfs": _public_trfs(previous_r),
        },
        "previous_exemplar_state": {
            "judgments": _public_judgments(previous_h),
            "selected_demo_ids": [item["demo_idx"] for item in previous_h["selection"]["selected"]],
        },
        "demonstrations": demonstrations,
    }
    if branch == "trf":
        system = (
            "You revise type-related features (TRFs) for a Skill target using the target text "
            "and the frozen previous-round exemplar judgments. Decide keep/drop for every known "
            "TRF in the given order. You may add only a domain/tool/technology feature or explicit "
            "skill-type cue, never a Skill span, paraphrase, work-style description, task, quality, "
            "or generic modifier. A keep or addition needs positive type evidence; mismatch, genericity, "
            "or boundary evidence alone is negative evidence. Evidence sources, demo IDs, and reason "
            "codes must describe the same evidence. Keep overall_reason to one short sentence and do not repeat individual TRF "
            "decisions there. Return only the strict JSON object required by the response schema."
        )
        instruction = {
            "task": "exemplar_to_trf",
            "constraints": {
                "decide_every_known_trf": True,
                "known_trf_order_is_fixed": True,
                "addition_semantic_classes": ["domain_feature", "skill_type_cue"],
                "confidence_range": [1, 5],
                "every_evidence_sources_array_must_be_non_empty": True,
                "every_reason_including_overall_reason_must_be_non_empty": True,
                "evidence_consistency": {
                    "target_support_codes_require_source": "TARGET_TEXT",
                    "exemplar_codes_require_source": "EXEMPLARS",
                    "exemplars_source_requires_demo_ids": True,
                    "demo_ids_require_exemplars_source": True,
                    "boundary_or_mismatch_codes_may_be_grounded_in_exemplars": True,
                    "keep_and_add_require_one_of": [
                        "TARGET_EXPLICIT_SUPPORT",
                        "TARGET_IMPLICIT_SUPPORT",
                        "EXEMPLAR_SUPPORT",
                        "OPEN_VOCABULARY_INFERENCE",
                    ],
                    "negative_codes_alone_cannot_keep_or_add": [
                        "EXEMPLAR_CONTRADICTION",
                        "TOO_GENERIC",
                        "TYPE_MISMATCH",
                        "MISSING_SKILL_CUE",
                        "BOUNDARY_EVIDENCE",
                    ],
                    "empty_entity_types_requires_zero_active_trfs": True,
                    "skill_type_cues_are_target_lexical_only": [
                        "skill", "skills", "ability", "abilities", "proficiency", "competence", "experience"
                    ],
                    "present_skill_type_cue_action": "keep",
                    "absent_skill_type_cue_action": "drop; exemplars cannot invent a lexical cue",
                    "single_word_target_modifiers_ending_ly_are_dropped": True,
                },
                "addition_definition": {
                    "domain_feature": "concrete domain, technology, system, tool, or method family",
                    "skill_type_cue": "explicit cue such as skill, proficiency, competence, or experience",
                    "excluded": ["skill span or paraphrase", "quality or trait", "descriptive modifier", "task or object", "work-style description"],
                },
                "reason_max_characters": max_reason_characters,
                "overall_reason_max_characters": overall_reason_maximum,
                "overall_reason_style": "one concise sentence; do not restate item-level reasons",
            },
            "input": frozen,
        }
        prompt_schema = "trf-interaction-prompt-v1"
    elif branch == "exemplar":
        system = (
            "You rejudge all 16 demonstrations for later exact Skill span extraction using only "
            "the frozen previous-round active TRFs. Previous judgments are provisional, not answers: "
            "compare against them and return the justified final score and role. The program records the "
            "previous values and signed score adjustment deterministically. Topic similarity alone is insufficient. "
            "Accepted demos may be supporting or irrelevant; negative demos may be contrastive "
            "or irrelevant. The program, not you, applies the final hard gate. Keep overall_reason "
            "to one short sentence and do not summarize all 16 judgments there. Return only the "
            "strict JSON object required by the response schema."
        )
        instruction = {
            "task": "trf_to_exemplar",
            "constraints": {
                "rejudge_all_candidates": True,
                "candidate_count": 16,
                "relation_targets": list(previous_r["active_trfs"]),
                "counterfactual_update": {
                    "instruction": "return a final score/role after comparing the demo with the frozen previous judgment and active TRFs",
                    "program_derived_audit_fields": ["previous_helpfulness_score", "previous_role", "score_adjustment", "trf_effect"],
                },
                "relation_semantics": {
                    "supports": "the named active TRF occurs in a labeled skill_span; a sentence mention or pseudo_trf alone is not enough",
                    "contrasts": "the demo specifically contradicts or clarifies the boundary of the named active TRF",
                    "unrelated": "the named active TRF has no meaningful bearing on this demo",
                    "accepted_exact_lexical_match": "for every active TRF appearing case-insensitively in an accepted demo skill_span, include a supports relation",
                    "accepted_support_score_floor": 4,
                    "generic_match_warning": "words such as process, tools, impact, or techniques in ordinary sentence text do not establish supports unless present in a labeled skill_span",
                },
                "accepted_demo_alignment": {
                    "strong_if_labeled_skill_span_appears_as_a_whole_phrase_in_target": True,
                    "strong_retrieval_similarity_floor": 0.58,
                    "strong_alignment_score_floor": 4,
                    "unsubstantiated_previous_score_4_is_below_hard_gate": True,
                },
                "negative_demo_rule": (
                    "If a negative demo previously scored 4 or 5 but has neither BOUNDARY_TRANSFERABLE "
                    "nor a specific supports/contrasts relation to an active TRF, penalize it to score 1..3 "
                    "and role irrelevant; do not use NEGATIVE_CONTRAST for that irrelevant result. Otherwise, "
                    "retain it as high contrastive boundary evidence when it is a strongly similar retrieved negative "
                    "that directly clarifies the target boundary. NEGATIVE_CONTRAST means role contrastive. Do not "
                    "invent a relation for generic topic similarity."
                ),
                "reason_code_rule": "TRF_ALIGNED requires at least one explicit trf_relation",
                "every_reason_codes_array_must_be_non_empty": True,
                "every_reason_including_overall_reason_must_be_non_empty": True,
                "reason_max_characters": max_reason_characters,
                "overall_reason_max_characters": overall_reason_maximum,
                "overall_reason_style": "one concise sentence; do not restate the 16 judgments",
            },
            "input": frozen,
        }
        prompt_schema = "exemplar-interaction-prompt-v1"
    else:
        raise InteractionError(f"Unknown interaction branch: {branch}")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(instruction, ensure_ascii=False, separators=(",", ":"))},
    ]
    character_count = sum(len(item["content"]) for item in messages)
    if character_count > max_characters:
        raise InteractionError(
            f"{branch} prompt for idx={record['idx']} has {character_count} characters, above {max_characters}"
        )
    response = response_format(
        record,
        branch,
        previous_r,
        max_reason_characters=max_reason_characters,
    )
    return {
        "schema_version": prompt_schema,
        **identity,
        "round": round_number,
        "branch": branch,
        "messages": messages,
        "prompt_sha256": sha256_json(messages),
        "character_count": character_count,
        "response_format": response,
        "response_format_sha256": sha256_json(response),
        "request_contract_sha256": request_sha256(messages, response),
        "max_reason_characters": max_reason_characters,
    }


def build_repair_messages(
    prompt: dict[str, Any],
    parse_error: str,
    prior_content: str,
) -> list[dict[str, str]]:
    maximum = prompt["max_reason_characters"]
    overall_maximum = min(maximum, OVERALL_REASON_PROMPT_MAX_CHARACTERS)
    messages = copy.deepcopy(prompt["messages"])
    messages.append({"role": "assistant", "content": prior_content})
    messages.append(
        {
            "role": "user",
            "content": (
                "The response violated the strict contract: "
                + parse_error[:800]
                + ". Return a complete corrected JSON object only; preserve every required "
                "identity and item. Every evidence_sources array, when present, MUST contain "
                "at least one allowed value. Every reason_codes array MUST contain at least one "
                "allowed value. Every item-level reason MUST be non-empty and no "
                f"longer than {maximum} characters. overall_reason MUST be exactly one concise "
                f"sentence of no more than {overall_maximum} characters; do not restate the "
                "individual decisions or judgments. For TRFs, make evidence sources, demo IDs, reason codes, and keep/add "
                "support logically consistent. Replace invalid text instead of repeating "
                "the prior response."
            ),
        }
    )
    return messages
