"""Pure contracts for candidate joins, prompts, parsing, and hard gating."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import sha256_json  # noqa: E402


REASON_CODES = frozenset(
    {
        "TYPE_ALIGNED",
        "TRF_ALIGNED",
        "BOUNDARY_TRANSFERABLE",
        "NEGATIVE_CONTRAST",
        "SEMANTIC_ONLY",
        "LABEL_UNCERTAIN",
        "TASK_MISMATCH",
        "REDUNDANT",
    }
)
ROLES = frozenset({"supporting", "contrastive", "irrelevant"})
FORMAL_STATUSES = frozenset({"accepted", "negative"})
JUDGMENT_KEYS = frozenset(
    {"demo_idx", "helpfulness_score", "role", "reason_codes", "reason"}
)


def _error(message: str) -> Exception:
    from instance_discriminator.common import InstanceDiscriminatorError

    return InstanceDiscriminatorError(message)


def _strict_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _error(f"{label} must be an integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _error(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise _error(f"{label} must be finite")
    return result


def normalized_target_trfs(target_record: dict[str, Any]) -> list[str]:
    raw = target_record.get("trfs")
    if not isinstance(raw, list):
        raise _error("Target trfs must be a list")
    result: list[str] = []
    for item in raw:
        value = item.get("normalized_text") if isinstance(item, dict) else item
        if not isinstance(value, str) or not value.strip():
            raise _error("Every target TRF must have non-empty normalized text")
        result.append(value.strip())
    if len(result) != len(set(result)):
        raise _error("Target TRFs must be unique")
    return result


def build_candidate_record(
    target: dict[str, Any],
    retrieval: dict[str, Any],
    decisions_by_idx: dict[int, dict[str, Any]],
    expected_count: int,
) -> dict[str, Any]:
    """Join one target's persisted retrieval choices to formal decisions."""

    idx = _strict_int(target.get("idx"), "target idx")
    sentence = target.get("sentence")
    if not isinstance(sentence, str) or not sentence.strip():
        raise _error(f"Target idx={idx} has an invalid sentence")
    if retrieval.get("idx") != idx or retrieval.get("sentence") != sentence:
        raise _error(f"Target/retrieval idx or sentence mismatch for idx={idx}")

    entity_types = target.get("entity_types")
    if entity_types not in ([], ["Skill"]):
        raise _error(f"Invalid entity_types for idx={idx}")
    upstream_status = target.get("status")
    if upstream_status not in {"complete", "needs_review"}:
        raise _error(f"Invalid target TRF status for idx={idx}")
    upstream_review = target.get("review_reasons")
    if not isinstance(upstream_review, list) or not all(
        isinstance(item, str) for item in upstream_review
    ):
        raise _error(f"Invalid target review reasons for idx={idx}")

    selected = retrieval.get("selected")
    if not isinstance(selected, list) or len(selected) != expected_count:
        actual = len(selected) if isinstance(selected, list) else "non-list"
        raise _error(
            f"Target idx={idx} must have exactly {expected_count} candidates, got {actual}"
        )

    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    for position, source in enumerate(selected, start=1):
        if not isinstance(source, dict):
            raise _error(f"Candidate {position} for idx={idx} must be an object")
        demo_idx = _strict_int(source.get("demo_idx"), "demo_idx")
        if demo_idx == idx:
            raise _error(f"Leave-one-out violation: target idx={idx} is its own candidate")
        if demo_idx in seen:
            raise _error(f"Duplicate candidate demo_idx={demo_idx} for target idx={idx}")
        seen.add(demo_idx)
        decision = decisions_by_idx.get(demo_idx)
        if decision is None:
            raise _error(f"Candidate demo_idx={demo_idx} has no source decision")
        status = source.get("status")
        if status not in FORMAL_STATUSES or decision.get("status") != status:
            raise _error(f"Candidate status mismatch for demo_idx={demo_idx}")
        demo_sentence = source.get("sentence")
        if demo_sentence != decision.get("sentence") or not isinstance(demo_sentence, str):
            raise _error(f"Candidate sentence mismatch for demo_idx={demo_idx}")
        spans = decision.get("spans")
        if not isinstance(spans, list) or not all(
            isinstance(item, str) and item for item in spans
        ):
            raise _error(f"Invalid spans for demo_idx={demo_idx}")
        if status == "accepted" and not spans:
            raise _error(f"Accepted demo_idx={demo_idx} must have skill spans")
        if status == "negative" and spans:
            raise _error(f"Negative demo_idx={demo_idx} must have empty skill spans")
        pseudo_trfs = source.get("trfs")
        if not isinstance(pseudo_trfs, list) or not all(
            isinstance(item, str) and item for item in pseudo_trfs
        ):
            raise _error(f"Invalid pseudo TRFs for demo_idx={demo_idx}")
        if status == "negative" and pseudo_trfs:
            raise _error(f"Negative demo_idx={demo_idx} must have empty pseudo TRFs")
        selected_rank = _strict_int(source.get("selected_rank"), "selected_rank")
        if selected_rank != position:
            raise _error(f"Candidate selected_rank mismatch for target idx={idx}")
        similarity = _finite_number(source.get("similarity"), "similarity")
        existence_score = _finite_number(
            source.get("existence_score"), "existence_score"
        )
        details = decision.get("accepted_span_details", [])
        if not isinstance(details, list):
            raise _error(f"Invalid accepted_span_details for demo_idx={demo_idx}")
        candidates.append(
            {
                "demo_idx": demo_idx,
                "retrieval_rank": selected_rank,
                "sentence": demo_sentence,
                "status": status,
                "skill_spans": list(spans),
                "accepted_span_details": details if status == "accepted" else [],
                "pseudo_trfs": list(pseudo_trfs),
                "similarity": similarity,
                "existence_score": existence_score,
            }
        )

    return {
        "idx": idx,
        "sentence": sentence,
        "target_evidence": {
            "entity_types": list(entity_types),
            "trfs": normalized_target_trfs(target),
        },
        "upstream_target_trf": {
            "status": upstream_status,
            "review_reasons": list(upstream_review),
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_discriminator_prompt(
    record: dict[str, Any], max_characters: int
) -> dict[str, Any]:
    """Build a leakage-controlled prompt using target TRF evidence only."""

    target = {
        "sentence": record["sentence"],
        "entity_types": record["target_evidence"]["entity_types"],
        "target_trfs": record["target_evidence"]["trfs"],
    }
    demonstrations = [
        {
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
    payload = {"target": target, "demonstrations": demonstrations}
    system = (
        "You are an instance discriminator for exact Skill character-span extraction. "
        "Judge whether each demonstration would help a later predictor decide whether the "
        "target contains a Skill and locate its exact source-text boundary. Topic similarity "
        "alone is not helpfulness. Accepted demonstrations may be supporting or irrelevant; "
        "negative demonstrations may be contrastive or irrelevant. Return one JSON object "
        "with exactly one judgment for every supplied demo_idx and no additional keys."
    )
    schema = {
        "required_output": {
            "judgments": [
                {
                    "demo_idx": "integer copied from input",
                    "helpfulness_score": "integer 1..5",
                    "role": "supporting | contrastive | irrelevant",
                    "reason_codes": sorted(REASON_CODES),
                    "reason": "non-empty audit reason, at most 240 characters",
                }
            ]
        },
        "constraints": {
            "accepted_roles": ["supporting", "irrelevant"],
            "negative_roles": ["contrastive", "irrelevant"],
            "judgment_count": record["candidate_count"],
        },
        "input": payload,
    }
    user = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    character_count = sum(len(item["content"]) for item in messages)
    if character_count > max_characters:
        raise _error(
            f"Prompt for idx={record['idx']} has {character_count} characters, "
            f"above the limit {max_characters}"
        )
    return {
        "idx": record["idx"],
        "sentence": record["sentence"],
        "demo_indexes": [item["demo_idx"] for item in record["candidates"]],
        "messages": messages,
        "prompt_sha256": sha256_json(messages),
        "character_count": character_count,
    }


def parse_judgments(
    content: str,
    candidates: list[dict[str, Any]],
    *,
    max_reason_characters: int = 240,
) -> list[dict[str, Any]]:
    """Parse the complete batch fail-closed; one malformed item fails the target."""

    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError) as error:
        raise _error(f"Judgment response is not valid JSON: {error}") from error
    if not isinstance(value, dict) or set(value) != {"judgments"}:
        raise _error("Judgment response must contain only the judgments key")
    raw = value["judgments"]
    if not isinstance(raw, list) or len(raw) != len(candidates):
        raise _error(f"Judgment response must contain exactly {len(candidates)} items")

    candidates_by_idx = {item["demo_idx"]: item for item in candidates}
    expected_ids = set(candidates_by_idx)
    parsed_by_idx: dict[int, dict[str, Any]] = {}
    for position, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or set(item) != JUDGMENT_KEYS:
            raise _error(f"Judgment {position} has missing or unknown keys")
        demo_idx = _strict_int(item.get("demo_idx"), "judgment demo_idx")
        if demo_idx not in expected_ids:
            raise _error(f"Unknown judgment demo_idx={demo_idx}")
        if demo_idx in parsed_by_idx:
            raise _error(f"Duplicate judgment demo_idx={demo_idx}")
        score = _strict_int(item.get("helpfulness_score"), "helpfulness_score")
        if not 1 <= score <= 5:
            raise _error(f"helpfulness_score for demo_idx={demo_idx} must be 1..5")
        role = item.get("role")
        if role not in ROLES:
            raise _error(f"Invalid role for demo_idx={demo_idx}")
        status = candidates_by_idx[demo_idx]["status"]
        if status == "accepted" and role not in {"supporting", "irrelevant"}:
            raise _error(f"Accepted demo_idx={demo_idx} cannot be {role}")
        if status == "negative" and role not in {"contrastive", "irrelevant"}:
            raise _error(f"Negative demo_idx={demo_idx} cannot be {role}")
        codes = item.get("reason_codes")
        if (
            not isinstance(codes, list)
            or not codes
            or not all(isinstance(code, str) and code in REASON_CODES for code in codes)
            or len(codes) != len(set(codes))
        ):
            raise _error(f"Invalid reason_codes for demo_idx={demo_idx}")
        reason = item.get("reason")
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > max_reason_characters
        ):
            raise _error(f"Invalid audit reason for demo_idx={demo_idx}")
        parsed_by_idx[demo_idx] = {
            "demo_idx": demo_idx,
            "helpfulness_score": score,
            "role": role,
            "reason_codes": list(codes),
            "reason": reason,
        }
    if set(parsed_by_idx) != expected_ids:
        raise _error("Judgment IDs are incomplete")
    return [parsed_by_idx[item["demo_idx"]] for item in candidates]


def apply_hard_gate(
    candidates: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    candidates_by_idx = {item["demo_idx"]: item for item in candidates}
    judgments_by_idx = {item["demo_idx"]: item for item in judgments}
    if set(candidates_by_idx) != set(judgments_by_idx):
        raise _error("Candidate and judgment IDs do not match")
    eligible = [
        {**candidates_by_idx[demo_idx], **judgments_by_idx[demo_idx]}
        for demo_idx in candidates_by_idx
        if judgments_by_idx[demo_idx]["helpfulness_score"]
        >= gate["minimum_helpfulness"]
        and judgments_by_idx[demo_idx]["role"] != "irrelevant"
    ]
    eligible.sort(
        key=lambda item: (
            -item["helpfulness_score"],
            -item["existence_score"],
            -item["similarity"],
            item["demo_idx"],
        )
    )
    selected: list[dict[str, Any]] = []
    role_counts = {"supporting": 0, "contrastive": 0}
    role_limits = {
        "supporting": gate["max_supporting"],
        "contrastive": gate["max_contrastive"],
    }
    for item in eligible:
        if len(selected) >= gate["max_selected"]:
            break
        role = item["role"]
        if role_counts[role] >= role_limits[role]:
            continue
        role_counts[role] += 1
        selected.append({**item, "gate_rank": len(selected) + 1})
    selected_ids = {item["demo_idx"] for item in selected}
    return {
        "selected": selected,
        "rejected_demo_ids": [
            item["demo_idx"] for item in candidates if item["demo_idx"] not in selected_ids
        ],
        "eligible_count": len(eligible),
        "role_counts": role_counts,
    }


def build_parsed_record(
    record: dict[str, Any], judgments: list[dict[str, Any]], chat_model: str
) -> dict[str, Any]:
    return {
        "idx": record["idx"],
        "sentence": record["sentence"],
        "status": "complete",
        "candidate_count": record["candidate_count"],
        "judgments": judgments,
        "models": {"chat": chat_model},
    }


def build_selected_record(
    record: dict[str, Any],
    judgments: list[dict[str, Any]],
    gate: dict[str, Any],
    chat_model: str,
) -> dict[str, Any]:
    result = apply_hard_gate(record["candidates"], judgments, gate)
    reasons: list[str] = []
    if record["upstream_target_trf"]["status"] == "needs_review":
        reasons.append("upstream_target_trf_needs_review")
    evidence = record["target_evidence"]
    if evidence["entity_types"] == ["Skill"] and not evidence["trfs"]:
        reasons.append("skill_type_without_target_trfs")
    count = len(result["selected"])
    if count == 0:
        reasons.append("no_helpful_examples")
    elif count < gate["minimum_for_complete"]:
        reasons.append("insufficient_helpful_examples")
    return {
        "idx": record["idx"],
        "sentence": record["sentence"],
        "status": "needs_review" if reasons else "complete",
        "target_evidence": evidence,
        "candidate_count": record["candidate_count"],
        "eligible_count": result["eligible_count"],
        "selected_count": count,
        "selected": result["selected"],
        "rejected_demo_ids": result["rejected_demo_ids"],
        "review_reasons": reasons,
        "upstream_review_reasons": record["upstream_target_trf"]["review_reasons"],
        "models": {"chat": chat_model},
    }
