"""Pure input normalization and prompt construction for both aggregator modes."""

from __future__ import annotations

import json
from typing import Any

from aggregator.common import BRANCHES, INPUT_SCHEMA, AggregatorError
from common.contracts import record_identity, record_source_sha256, validate_unique_identities
from common.io_utils import sha256_json
from common.skill_prediction import PROMPT_SCHEMA, SKILL_SPAN_INSTRUCTION


EXPERT_BRANCHES = {"trf": "trf", "exemplar": "exemplar"}


def _validate_span(span: dict[str, Any], sentence: str, label: str) -> dict[str, Any]:
    if not isinstance(span, dict):
        raise AggregatorError(f"{label} must be an object")
    text, start, end = span.get("text"), span.get("start"), span.get("end")
    if (
        not isinstance(text, str)
        or not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not 0 <= start < end <= len(sentence)
        or sentence[start:end] != text
    ):
        raise AggregatorError(f"{label} is not an exact end-exclusive sentence span")
    return {"text": text, "start": start, "end": end}


def _validate_span_set(spans: Any, sentence: str, label: str) -> list[dict[str, Any]]:
    if not isinstance(spans, list):
        raise AggregatorError(f"{label} must be a list")
    normalized = [_validate_span(item, sentence, f"{label}[{i}]") for i, item in enumerate(spans)]
    if normalized != sorted(normalized, key=lambda item: (item["start"], item["end"])):
        raise AggregatorError(f"{label} must be sorted by source position")
    for left, right in zip(normalized, normalized[1:]):
        if left["end"] > right["start"]:
            raise AggregatorError(f"{label} contains overlapping spans")
    return normalized


def _target_trfs(record: dict[str, Any]) -> list[str]:
    context = record.get("trf_context")
    if not isinstance(context, dict) or context.get("status") not in {"complete", "needs_review"}:
        raise AggregatorError("Invalid target TRF context")
    trfs = context.get("trfs")
    if not isinstance(trfs, list):
        raise AggregatorError("trf_context.trfs must be a list")
    result: list[str] = []
    seen: set[str] = set()
    for position, item in enumerate(trfs):
        if not isinstance(item, dict):
            raise AggregatorError(f"trf_context.trfs[{position}] must be an object")
        text = item.get("normalized_text")
        if not isinstance(text, str) or not text.strip():
            raise AggregatorError(f"trf_context.trfs[{position}].normalized_text is invalid")
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _example(item: dict[str, Any], position: int) -> dict[str, Any]:
    label = f"selected[{position}]"
    status = item.get("status")
    sentence = item.get("sentence")
    score = item.get("helpfulness_score")
    if status not in {"accepted", "negative"}:
        raise AggregatorError(f"{label}.status is not hard-gate eligible")
    if not isinstance(sentence, str) or not sentence:
        raise AggregatorError(f"{label}.sentence is invalid")
    if not isinstance(score, int) or isinstance(score, bool) or not 4 <= score <= 5:
        raise AggregatorError(
            f"{label}.helpfulness_score must be a hard-gated integer from 4 to 5"
        )
    for key in ("demo_dataset_id", "demo_record_id"):
        if not isinstance(item.get(key), str) or not item[key]:
            raise AggregatorError(f"{label}.{key} is invalid")
    demo_idx = item.get("demo_idx")
    if not isinstance(demo_idx, int) or isinstance(demo_idx, bool):
        raise AggregatorError(f"{label}.demo_idx is invalid")
    details = item.get("accepted_span_details")
    if status == "negative":
        if details not in ([], None) or item.get("skill_spans") not in ([], None):
            raise AggregatorError(f"{label} negative example must have no spans")
        spans: list[dict[str, Any]] = []
        has_skill = 0
    else:
        spans = _validate_span_set(details, sentence, f"{label}.accepted_span_details")
        if not spans:
            raise AggregatorError(f"{label} accepted example must have spans")
        if item.get("skill_spans") != [span["text"] for span in spans]:
            raise AggregatorError(f"{label}.skill_spans disagrees with exact span details")
        has_skill = 1
    return {
        "demo_dataset_id": item["demo_dataset_id"],
        "demo_record_id": item["demo_record_id"],
        "demo_idx": demo_idx,
        "sentence": sentence,
        "has_skill": has_skill,
        "skill_spans": spans,
        "helpfulness_score": score,
    }


def _examples(record: dict[str, Any], maximum: int) -> list[dict[str, Any]]:
    context = record.get("exemplar_context")
    if (
        not isinstance(context, dict)
        or context.get("status") not in {"complete", "needs_review"}
        or context.get("feature_context") != "absent"
    ):
        raise AggregatorError("Invalid independent exemplar context")
    selected = context.get("selected")
    if not isinstance(selected, list) or len(selected) != context.get("selected_count"):
        raise AggregatorError("Exemplar selected_count does not match selected records")
    ranks = [item.get("gate_rank") for item in selected]
    if any(not isinstance(rank, int) or isinstance(rank, bool) for rank in ranks):
        raise AggregatorError("Every selected example needs an integer gate_rank")
    if ranks != sorted(ranks) or len(set(ranks)) != len(ranks):
        raise AggregatorError("Selected examples must be uniquely ordered by gate_rank")
    return [_example(item, position) for position, item in enumerate(selected[:maximum])]


def _expert_index(records: list[dict[str, Any]], branch: str) -> dict[tuple[str, str], dict[str, Any]]:
    validate_unique_identities(records, f"{branch} expert predictions")
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for position, record in enumerate(records):
        if record.get("schema_version") not in {
            "skill-prediction-v1",
            "skill-prediction-result-v1",
        }:
            raise AggregatorError(f"{branch} expert record {position} has an invalid schema")
        if record.get("branch") != EXPERT_BRANCHES[branch]:
            raise AggregatorError(f"{branch} expert record {position} has the wrong branch")
        idx, sentence = record.get("idx"), record.get("sentence")
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise AggregatorError(f"{branch} expert record {position} has an invalid idx")
        if not isinstance(sentence, str) or not sentence:
            raise AggregatorError(f"{branch} expert record {position} has an invalid sentence")
        record_source_sha256(record, f"{branch} expert[{position}]")
        if record.get("schema_version") == "skill-prediction-v1":
            if record.get("status") not in {"complete", "needs_review"}:
                raise AggregatorError(f"{branch} expert record {position} has an invalid status")
            spans = _validate_span_set(
                record.get("spans"), sentence, f"{branch} expert[{position}].spans"
            )
            if record.get("has_skill") != (1 if spans else 0):
                raise AggregatorError(
                    f"{branch} expert record {position} has_skill disagrees with spans"
                )
        else:
            outcome = record.get("outcome")
            if outcome not in {
                "exact",
                "recovered",
                "provisional",
                "validation_failed",
                "runtime_failed",
                "missing",
            }:
                raise AggregatorError(f"{branch} expert record {position} has an invalid outcome")
            if outcome in {"exact", "recovered"}:
                prediction = record.get("prediction")
                if not isinstance(prediction, dict) or prediction.get("schema_version") != "skill-prediction-v1":
                    raise AggregatorError(f"{branch} expert result {position} lacks a formal prediction")
                for key in ("branch", "dataset_id", "record_id", "source_sha256", "idx", "sentence"):
                    if prediction.get(key) != record.get(key):
                        raise AggregatorError(
                            f"{branch} expert result {position} prediction {key} mismatch"
                        )
                if record.get("coordinate_space") != "target_sentence":
                    raise AggregatorError(f"{branch} expert result {position} has the wrong coordinate space")
                spans = _validate_span_set(
                    prediction.get("spans"), sentence, f"{branch} expert[{position}].prediction.spans"
                )
                if prediction.get("status") not in {"complete", "needs_review"}:
                    raise AggregatorError(
                        f"{branch} expert result {position} prediction status is invalid"
                    )
                if prediction.get("has_skill") != (1 if spans else 0):
                    raise AggregatorError(
                        f"{branch} expert result {position} prediction has_skill mismatch"
                    )
                if record.get("validation_issue") is not (outcome == "recovered"):
                    raise AggregatorError(
                        f"{branch} expert result {position} validation issue mismatch"
                    )
            elif outcome == "provisional":
                provisional = record.get("provisional_extraction")
                if not isinstance(provisional, dict):
                    raise AggregatorError(f"{branch} provisional result {position} is missing")
                model_sentence = provisional.get("model_sentence")
                if not isinstance(model_sentence, str) or record.get("coordinate_space") != "model_sentence":
                    raise AggregatorError(f"{branch} provisional result {position} has invalid coordinates")
                spans = _validate_span_set(
                    provisional.get("spans"), model_sentence, f"{branch} expert[{position}].provisional.spans"
                )
                if provisional.get("has_skill") != (1 if spans else 0):
                    raise AggregatorError(
                        f"{branch} provisional result {position} has_skill mismatch"
                    )
                if record.get("validation_issue") is not True:
                    raise AggregatorError(
                        f"{branch} provisional result {position} must be a validation issue"
                    )
            else:
                if record.get("coordinate_space") is not None:
                    raise AggregatorError(f"{branch} unavailable result {position} cannot expose coordinates")
                expected_issue = outcome == "validation_failed"
                if record.get("validation_issue") is not expected_issue:
                    raise AggregatorError(
                        f"{branch} unavailable result {position} validation issue mismatch"
                    )
                if record.get("prediction") is not None:
                    raise AggregatorError(
                        f"{branch} unavailable result {position} cannot contain a formal prediction"
                    )
        reasons = record.get("review_reasons")
        if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
            raise AggregatorError(
                f"{branch} expert record {position} has invalid review_reasons"
            )
        output[record_identity(record, f"{branch} expert[{position}]")] = record
    return output


def expert_indexes(
    trf_records: list[dict[str, Any]], exemplar_records: list[dict[str, Any]]
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    return {
        "trf": _expert_index(trf_records, "trf"),
        "exemplar": _expert_index(exemplar_records, "exemplar"),
    }


def _expert_proposal(
    branch: str,
    expert: dict[str, Any] | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    if expert is None:
        return {
            "available": False,
            "status": "missing",
            "has_skill": None,
            "spans": [],
            "coordinate_space": None,
            "model_sentence": None,
            "review_reasons": [f"expert_prediction_missing:{branch}"],
        }
    identity = record_identity(context, "context")
    if record_identity(expert, f"{branch} expert") != identity:
        raise AggregatorError(f"{branch} expert identity mismatch for {identity!r}")
    if expert.get("idx") != context.get("idx"):
        raise AggregatorError(f"{branch} expert idx mismatch for {identity!r}")
    if expert.get("sentence") != context.get("sentence"):
        raise AggregatorError(f"{branch} expert sentence mismatch for {identity!r}")
    if record_source_sha256(expert, f"{branch} expert") != record_source_sha256(context, "context"):
        raise AggregatorError(f"{branch} expert source hash mismatch for {identity!r}")
    if expert.get("schema_version") == "skill-prediction-result-v1":
        outcome = expert["outcome"]
        if outcome in {"exact", "recovered"}:
            formal = expert["prediction"]
            status, has_skill = formal.get("status"), formal.get("has_skill")
            spans = _validate_span_set(
                formal.get("spans"), context["sentence"], f"{branch} expert spans"
            )
            reasons = list(expert.get("review_reasons", []))
            return {
                "available": True,
                "status": status,
                "outcome": outcome,
                "has_skill": has_skill,
                "spans": spans,
                "coordinate_space": "target_sentence",
                "model_sentence": None,
                "review_reasons": reasons,
            }
        if outcome == "provisional":
            provisional = expert["provisional_extraction"]
            model_sentence = provisional["model_sentence"]
            spans = _validate_span_set(
                provisional.get("spans"), model_sentence, f"{branch} provisional spans"
            )
            return {
                "available": True,
                "status": "needs_review",
                "outcome": outcome,
                "has_skill": provisional.get("has_skill"),
                "spans": spans,
                "coordinate_space": "model_sentence",
                "model_sentence": model_sentence,
                "review_reasons": list(expert.get("review_reasons", [])),
            }
        return {
            "available": False,
            "status": outcome,
            "outcome": outcome,
            "has_skill": None,
            "spans": [],
            "coordinate_space": None,
            "model_sentence": None,
            "review_reasons": list(expert.get("review_reasons", [])),
        }

    status, has_skill = expert.get("status"), expert.get("has_skill")
    if status not in {"complete", "needs_review"} or has_skill not in {0, 1}:
        raise AggregatorError(f"{branch} expert has invalid status or has_skill")
    spans = _validate_span_set(expert.get("spans"), context["sentence"], f"{branch} expert spans")
    if has_skill != (1 if spans else 0):
        raise AggregatorError(f"{branch} expert has_skill disagrees with spans")
    reasons = expert.get("review_reasons")
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        raise AggregatorError(f"{branch} expert review_reasons are invalid")
    return {
        "available": True,
        "status": status,
        "outcome": "exact",
        "has_skill": has_skill,
        "spans": spans,
        "coordinate_space": "target_sentence",
        "model_sentence": None,
        "review_reasons": list(reasons),
    }


def build_aggregator_inputs(
    contexts: list[dict[str, Any]],
    *,
    mode: str,
    maximum_examples: int,
    experts: dict[str, dict[tuple[str, str], dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    if mode not in BRANCHES:
        raise AggregatorError(f"Unsupported mode: {mode}")
    validate_unique_identities(contexts, "second-layer contexts")
    seen_indexes: set[int] = set()
    output: list[dict[str, Any]] = []
    for position, context in enumerate(contexts):
        if context.get("schema_version") != "second-layer-context-v1":
            raise AggregatorError(f"context[{position}] has an invalid schema")
        if context.get("assembly_status") != "ready":
            raise AggregatorError(f"context[{position}] is not ready")
        idx, sentence = context.get("idx"), context.get("sentence")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx in seen_indexes:
            raise AggregatorError(f"context[{position}].idx is invalid or duplicated")
        seen_indexes.add(idx)
        if not isinstance(sentence, str) or not sentence:
            raise AggregatorError(f"context[{position}].sentence is invalid")
        identity = record_identity(context, f"context[{position}]")
        source_sha256 = record_source_sha256(context, f"context[{position}]")
        result: dict[str, Any] = {
            "schema_version": INPUT_SCHEMA,
            "mode": mode,
            "dataset_id": identity[0],
            "record_id": identity[1],
            "source_sha256": source_sha256,
            "idx": idx,
            "target_sentence": sentence,
            "target_trfs": _target_trfs(context),
            "examples": _examples(context, maximum_examples),
        }
        if mode == "with_expert_results":
            if experts is None:
                raise AggregatorError("Expert indexes are required in expert mode")
            proposals = {}
            for branch in ("trf", "exemplar"):
                expert = experts[branch].get(identity)
                if expert is None and any(
                    candidate.get("idx") == idx for candidate in experts[branch].values()
                ):
                    raise AggregatorError(
                        f"{branch} expert identity mismatch for target idx={idx}"
                    )
                proposals[branch] = _expert_proposal(branch, expert, context)
            result["expert_proposals"] = proposals
        elif experts is not None:
            raise AggregatorError("evidence_only cannot receive expert indexes")
        output.append(result)
    return output


def review_reasons(input_record: dict[str, Any], context: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if context["trf_context"]["status"] == "needs_review":
        reasons.append("trf_context_needs_review")
    if context["exemplar_context"]["status"] == "needs_review":
        reasons.append("exemplar_context_needs_review")
    count = len(input_record["examples"])
    if count == 0:
        reasons.append("no_helpful_examples")
    elif count == 1:
        reasons.append("insufficient_helpful_examples")
    for branch, proposal in input_record.get("expert_proposals", {}).items():
        if not proposal["available"]:
            reasons.append(f"expert_prediction_missing:{branch}")
    return reasons


def _expert_audit(proposals: dict[str, Any]) -> dict[str, Any]:
    trf, exemplar = proposals["trf"], proposals["exemplar"]
    both = trf["available"] and exemplar["available"]
    return {
        "availability": {name: value["available"] for name, value in proposals.items()},
        "presence_agreement": both and trf["has_skill"] == exemplar["has_skill"],
        "exact_span_agreement": (
            both
            and trf["coordinate_space"] == "target_sentence"
            and exemplar["coordinate_space"] == "target_sentence"
            and trf["spans"] == exemplar["spans"]
        ),
    }


def build_prompt(
    input_record: dict[str, Any],
    context: dict[str, Any],
    *,
    maximum_characters: int,
) -> dict[str, Any]:
    mode = input_record["mode"]
    reasons = review_reasons(input_record, context)
    payload: dict[str, Any] = {
        "target_sentence": input_record["target_sentence"],
        "target_trfs": input_record["target_trfs"],
        "examples": input_record["examples"],
        "evidence_guidance": {
            "target_trfs": "Soft type evidence only; an empty list does not imply no skill.",
            "examples": (
                "Use examples as boundary/type references. helpfulness_score is reference "
                "strength, never a copying weight. Never copy a skill absent from the target."
            ),
        },
    }
    expert_instruction = ""
    evidence: dict[str, Any] = {
        "mode": mode,
        "target_trfs": input_record["target_trfs"],
        "examples": [
            {
                "demo_dataset_id": item["demo_dataset_id"],
                "demo_record_id": item["demo_record_id"],
                "demo_idx": item["demo_idx"],
                "has_skill": item["has_skill"],
                "helpfulness_score": item["helpfulness_score"],
            }
            for item in input_record["examples"]
        ],
    }
    if mode == "with_expert_results":
        proposals = input_record["expert_proposals"]
        payload["expert_proposals"] = proposals
        expert_instruction = """
The two expert proposals are fallible references, even when they agree, and may both be wrong.
Re-check the target sentence, target TRFs, and examples yourself. Agreement and disagreement are
never binding. You may accept, reject, shorten, expand, split, merge, completely replace, or ignore
their spans, and you may output a new target-sentence span proposed by neither expert.
"""
        evidence["expert_proposals"] = proposals
        evidence["expert_agreement"] = _expert_audit(proposals)
    system = (
        "You are the final independent skill-span aggregator.\n"
        + SKILL_SPAN_INSTRUCTION
        + expert_instruction
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]
    if sum(len(item["content"]) for item in messages) > maximum_characters:
        raise AggregatorError(f"Prompt exceeds maximum characters for idx={input_record['idx']}")
    prompt = {
        "schema_version": PROMPT_SCHEMA,
        "branch": BRANCHES[mode],
        "dataset_id": input_record["dataset_id"],
        "record_id": input_record["record_id"],
        "source_sha256": input_record["source_sha256"],
        "idx": input_record["idx"],
        "sentence": input_record["target_sentence"],
        "messages": messages,
        "review_reasons": reasons,
        "evidence": evidence,
    }
    prompt["prompt_sha256"] = sha256_json(messages)
    return prompt
