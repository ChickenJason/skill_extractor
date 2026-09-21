"""Strict target-skill prediction with audited retention and append-only execution."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from common.io_utils import append_jsonl, read_jsonl, redact_secrets, sha256_json, utc_now


OPEN_TAG = "<skill>"
CLOSE_TAG = "</skill>"
PREDICTION_SCHEMA = "skill-prediction-v1"
PROMPT_SCHEMA = "skill-prediction-prompt-v1"
RAW_SCHEMA = "skill-prediction-raw-v1"
REPAIR_SCHEMA = "skill-prediction-repair-v1"
PREDICTION_FAILURE_SCHEMA = "skill-prediction-failure-v1"
RECOVERY_SCHEMA = "skill-prediction-recovery-v1"
PROVISIONAL_SCHEMA = "skill-prediction-provisional-v1"
RESULT_SCHEMA = "skill-prediction-result-v1"
VALIDATION_ISSUE_SCHEMA = "skill-prediction-validation-issue-v1"
RECOVERY_REVIEW_REASON = "prediction_recovered_after_validation_failure"

# One-codepoint repairs for common CP1252 punctuation bytes that were decoded as
# C1 control characters before reaching this pipeline. Keeping this list narrow
# prevents arbitrary model rewrites from being projected onto the source sentence.
SAFE_CHARACTER_EQUIVALENCES = {
    ("\u0085", "\u2026"),
    ("\u0091", "\u2018"),
    ("\u0092", "\u2019"),
    ("\u0093", "\u201c"),
    ("\u0094", "\u201d"),
    ("\u0096", "\u2013"),
    ("\u0097", "\u2014"),
}

SKILL_SPAN_INSTRUCTION = """Identify only skills or competencies explicitly expressed in the target sentence.
Treat the target sentence as immutable source text. Copy it character for character and only
insert <skill> immediately before each skill span and </skill> immediately after it.

Rules:
1. Never repair grammar, spelling, encoding, whitespace, or formatting.
2. Never add a shared word to another coordinated phrase.
3. Preserve unusual bullets, apostrophes, dashes, and control characters exactly.
4. Exclude duty or requirement framing by placing tags around only the skill substring.
5. Every tagged span must be one continuous exact source substring.
6. Mark the shortest continuous span that is semantically complete as a skill.
7. Split coordinated skills only when each exact source substring is independently meaningful.
8. Do not create empty, nested, overlapping, duplicate, or redundant tagged spans.
9. Do not infer a skill absent from the target sentence.
10. If no explicit skill exists, return the original target sentence unchanged.
11. Return one JSON object with exactly one field named annotated_sentence and no explanation."""


class SkillPredictionError(ValueError):
    """Raised when a target-skill prediction violates the strict contract."""


class SkillPredictionNetworkRequired(SkillPredictionError):
    """Raised when pending target predictions require explicit network permission."""


def _strict_json_object(content: Any) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise SkillPredictionError("Prediction response content is empty")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise SkillPredictionError(f"Prediction response is not valid JSON: {error}") from error
    if not isinstance(value, dict) or set(value) != {"annotated_sentence"}:
        raise SkillPredictionError(
            "Prediction response must contain only the annotated_sentence key"
        )
    return value


def _parse_tagged_prediction(content: Any) -> dict[str, Any]:
    """Parse JSON and tag structure without asserting source-text equality."""

    value = _strict_json_object(content)
    annotated = value["annotated_sentence"]
    if not isinstance(annotated, str):
        raise SkillPredictionError("annotated_sentence must be a string")

    restored: list[str] = []
    ranges: list[tuple[int, int]] = []
    active_start: int | None = None
    cursor = 0
    while cursor < len(annotated):
        if annotated.startswith(OPEN_TAG, cursor):
            if active_start is not None:
                raise SkillPredictionError("Skill tags cannot be nested or overlapping")
            active_start = len(restored)
            cursor += len(OPEN_TAG)
            continue
        if annotated.startswith(CLOSE_TAG, cursor):
            if active_start is None:
                raise SkillPredictionError(
                    "Closing skill tag has no matching opening tag"
                )
            end = len(restored)
            if end == active_start:
                raise SkillPredictionError("Skill tags cannot be empty")
            ranges.append((active_start, end))
            active_start = None
            cursor += len(CLOSE_TAG)
            continue
        restored.append(annotated[cursor])
        cursor += 1
    if active_start is not None:
        raise SkillPredictionError("Opening skill tag has no matching closing tag")

    restored_sentence = "".join(restored)
    spans = [
        {
            "text": restored_sentence[start:end],
            "start": start,
            "end": end,
        }
        for start, end in ranges
    ]
    return {
        "annotated_sentence": annotated,
        "model_sentence": restored_sentence,
        "has_skill": 1 if spans else 0,
        "spans": spans,
    }


def _validate_source_sentence(sentence: str) -> None:
    if not isinstance(sentence, str) or not sentence:
        raise SkillPredictionError("Prediction source sentence must be non-empty")
    if OPEN_TAG in sentence or CLOSE_TAG in sentence:
        raise SkillPredictionError("Prediction source sentence contains reserved skill tags")


def parse_skill_prediction(content: Any, sentence: str) -> dict[str, Any]:
    """Parse exact inline tags without recovery, projection, or silent correction."""

    _validate_source_sentence(sentence)
    parsed = _parse_tagged_prediction(content)
    restored_sentence = parsed["model_sentence"]
    if restored_sentence != sentence:
        raise SkillPredictionError(
            "Removing skill tags must reconstruct the target sentence exactly"
        )
    return {"has_skill": parsed["has_skill"], "spans": parsed["spans"]}


def provisional_skill_prediction(content: Any) -> dict[str, Any]:
    """Retain structurally valid tagged extraction on the model-returned sentence."""

    parsed = _parse_tagged_prediction(content)
    return {
        "schema_version": PROVISIONAL_SCHEMA,
        "model_sentence": parsed["model_sentence"],
        "has_skill": parsed["has_skill"],
        "spans": parsed["spans"],
    }


def recover_skill_prediction(
    content: Any,
    sentence: str,
    strict_validation_error: dict[str, str],
) -> dict[str, Any]:
    """Project tags only across audited, one-to-one punctuation normalizations."""

    _validate_source_sentence(sentence)
    provisional = provisional_skill_prediction(content)
    model_sentence = provisional["model_sentence"]
    if model_sentence == sentence:
        raise SkillPredictionError("Strictly valid predictions do not require recovery")
    if len(model_sentence) != len(sentence):
        raise SkillPredictionError(
            "Prediction text length changed and cannot be safely projected"
        )
    substitutions: list[dict[str, Any]] = []
    for position, (source_character, model_character) in enumerate(
        zip(sentence, model_sentence)
    ):
        if source_character == model_character:
            continue
        if (source_character, model_character) not in SAFE_CHARACTER_EQUIVALENCES:
            raise SkillPredictionError(
                "Prediction contains a non-whitelisted source character change"
            )
        substitutions.append(
            {
                "position": position,
                "source_character": source_character,
                "model_character": model_character,
                "source_codepoint": f"U+{ord(source_character):04X}",
                "model_codepoint": f"U+{ord(model_character):04X}",
            }
        )
    if not substitutions:
        raise SkillPredictionError("Prediction recovery found no character substitutions")
    source_spans = [
        {
            "text": sentence[item["start"] : item["end"]],
            "start": item["start"],
            "end": item["end"],
        }
        for item in provisional["spans"]
    ]
    return {
        "schema_version": RECOVERY_SCHEMA,
        "strategy": "whitelisted_character_projection",
        "strict_validation_error": strict_validation_error,
        "model_sentence": model_sentence,
        "character_substitutions": substitutions,
        "annotation": {
            "has_skill": 1 if source_spans else 0,
            "spans": source_spans,
        },
    }


def latest_prediction_raw_records(path: Path) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for line_number, record in enumerate(read_jsonl(path), start=1):
        idx = record.get("idx")
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise SkillPredictionError(
                f"Invalid idx in prediction raw record {line_number}"
            )
        latest[idx] = record
    return latest


def _error_value(error: Exception, secrets: list[str]) -> dict[str, str]:
    return redact_secrets(
        {"type": error.__class__.__name__, "message": str(error)}, secrets
    )


def _repair_messages(
    prompt: dict[str, Any],
    invalid_content: str,
    parse_error: dict[str, str],
    repair_round: int,
) -> list[dict[str, str]]:
    correction = {
        "repair_round": repair_round,
        "prior_validation_error": parse_error,
        "instruction": (
            "Replace the entire prior response with one corrected JSON object. "
            "Do not explain the correction and do not add markdown. Preserve the target "
            "sentence character for character and only insert valid <skill> tags."
        ),
        "required_output": {"annotated_sentence": "exact target text with optional tags"},
    }
    return [
        *prompt["messages"],
        {"role": "assistant", "content": invalid_content},
        {
            "role": "user",
            "content": json.dumps(correction, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _validate_repair_audit(
    raw: dict[str, Any],
    prompt: dict[str, Any],
    maximum_repairs: int,
    *,
    allow_terminal_failure: bool = False,
) -> None:
    repair = raw.get("repair")
    if repair is None:
        if raw.get("recovery") is not None:
            raise SkillPredictionError(
                f"Prediction recovery has no repair audit for idx={prompt['idx']}"
            )
        if allow_terminal_failure:
            raise SkillPredictionError(
                f"Terminal prediction failure has no complete repair audit for idx={prompt['idx']}"
            )
        return
    if not isinstance(repair, dict) or set(repair) != {"schema_version", "attempts"}:
        raise SkillPredictionError(f"Invalid prediction repair audit for idx={prompt['idx']}")
    if repair.get("schema_version") != REPAIR_SCHEMA:
        raise SkillPredictionError(
            f"Unsupported prediction repair schema for idx={prompt['idx']}"
        )
    attempts = repair.get("attempts")
    if not isinstance(attempts, list) or not 2 <= len(attempts) <= maximum_repairs + 1:
        raise SkillPredictionError(
            f"Invalid prediction repair attempt count for idx={prompt['idx']}"
        )

    expected_messages = prompt["messages"]
    validated_recovery = False
    for position, attempt in enumerate(attempts, start=1):
        expected_kind = "initial" if position == 1 else "repair"
        if (
            not isinstance(attempt, dict)
            or set(attempt)
            != {"attempt", "kind", "request_sha256", "response", "parse_error"}
            or attempt.get("attempt") != position
            or attempt.get("kind") != expected_kind
            or attempt.get("request_sha256") != sha256_json(expected_messages)
            or not isinstance(attempt.get("response"), dict)
        ):
            raise SkillPredictionError(
                f"Invalid prediction repair attempt for idx={prompt['idx']} attempt={position}"
            )
        response = attempt["response"]
        try:
            parse_skill_prediction(response.get("content", ""), prompt["sentence"])
        except SkillPredictionError as error:
            actual_error = {"type": error.__class__.__name__, "message": str(error)}
            if attempt.get("parse_error") != actual_error:
                raise SkillPredictionError(
                    f"Prediction repair error audit mismatch for idx={prompt['idx']}"
                ) from error
            if position == len(attempts):
                if raw.get("status") == "complete" and raw.get("recovery") is not None:
                    expected_recovery = recover_skill_prediction(
                        response.get("content", ""),
                        prompt["sentence"],
                        actual_error,
                    )
                    if (
                        len(attempts) != maximum_repairs + 1
                        or response != raw.get("response")
                        or raw.get("error") is not None
                        or raw.get("recovery") != expected_recovery
                    ):
                        raise SkillPredictionError(
                            f"Prediction recovery audit mismatch for idx={prompt['idx']}"
                        ) from error
                    validated_recovery = True
                    continue
                if (
                    not allow_terminal_failure
                    or len(attempts) != maximum_repairs + 1
                    or raw.get("status") != "failed"
                    or response != raw.get("response")
                    or raw.get("error") != actual_error
                ):
                    raise SkillPredictionError(
                        f"Prediction repair ended invalidly for idx={prompt['idx']}"
                    ) from error
                continue
            expected_messages = _repair_messages(
                prompt,
                response.get("content", ""),
                actual_error,
                position,
            )
        else:
            if position != len(attempts) or attempt.get("parse_error") is not None:
                raise SkillPredictionError(
                    f"Prediction repair success audit mismatch for idx={prompt['idx']}"
                )
            if response != raw.get("response"):
                raise SkillPredictionError(
                    f"Final repaired response mismatch for idx={prompt['idx']}"
                )
            if raw.get("recovery") is not None:
                raise SkillPredictionError(
                    f"Strict prediction success cannot carry recovery for idx={prompt['idx']}"
                )
            if raw.get("status") != "complete" or raw.get("error") is not None:
                raise SkillPredictionError(
                    f"Strict prediction success has invalid terminal state for idx={prompt['idx']}"
                )
    if (raw.get("recovery") is not None) != validated_recovery:
        raise SkillPredictionError(
            f"Prediction recovery state mismatch for idx={prompt['idx']}"
        )


def build_prediction_failure_records(
    prompts: list[dict[str, Any]],
    latest_raw: dict[int, dict[str, Any]],
    *,
    maximum_repairs: int,
) -> list[dict[str, Any]]:
    """Build auditable records for missing or unsuccessful target predictions."""

    failures: list[dict[str, Any]] = []
    for prompt in prompts:
        raw = latest_raw.get(prompt["idx"])
        if raw and raw.get("status") == "complete":
            continue

        failure_kind = "missing_raw"
        response: dict[str, Any] | None = None
        error: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        tolerance_eligible = False
        if raw is not None:
            for key in ("branch", "dataset_id", "record_id", "source_sha256", "idx"):
                if raw.get(key) != prompt.get(key):
                    raise SkillPredictionError(
                        f"Prediction failure raw {key} mismatch for idx={prompt['idx']}"
                    )
            if raw.get("schema_version") != RAW_SCHEMA:
                raise SkillPredictionError(
                    f"Unsupported prediction failure raw schema for idx={prompt['idx']}"
                )
            if raw.get("prompt_sha256") != prompt["prompt_sha256"]:
                raise SkillPredictionError(
                    f"Prediction failure prompt hash mismatch for idx={prompt['idx']}"
                )
            response = raw.get("response") if isinstance(raw.get("response"), dict) else None
            error = raw.get("error") if isinstance(raw.get("error"), dict) else None
            repair = raw.get("repair")
            if isinstance(repair, dict) and isinstance(repair.get("attempts"), list):
                attempts = repair["attempts"]
            if error and error.get("type") == "SkillPredictionError":
                try:
                    _validate_repair_audit(
                        raw,
                        prompt,
                        maximum_repairs,
                        allow_terminal_failure=True,
                    )
                except SkillPredictionError:
                    failure_kind = "invalid_failure_audit"
                else:
                    failure_kind = "validation_exhausted"
                    tolerance_eligible = True
            else:
                failure_kind = "provider_or_runtime_failure"

        failures.append(
            {
                "schema_version": PREDICTION_FAILURE_SCHEMA,
                "branch": prompt["branch"],
                "dataset_id": prompt["dataset_id"],
                "record_id": prompt["record_id"],
                "source_sha256": prompt["source_sha256"],
                "idx": prompt["idx"],
                "sentence": prompt["sentence"],
                "status": "failed",
                "failure_kind": failure_kind,
                "tolerance_eligible": tolerance_eligible,
                "model_output": response.get("content") if response else None,
                "validation_error": error,
                "provisional_extraction": _provisional_or_none(response),
                "attempts": [
                    {
                        "attempt": item.get("attempt"),
                        "kind": item.get("kind"),
                        "request_sha256": item.get("request_sha256"),
                        "model_output": (
                            item.get("response", {}).get("content")
                            if isinstance(item.get("response"), dict)
                            else None
                        ),
                        "validation_error": item.get("parse_error"),
                    }
                    for item in attempts
                ],
                "prompt_sha256": prompt["prompt_sha256"],
            }
        )
    return failures


def _provisional_or_none(response: dict[str, Any] | None) -> dict[str, Any] | None:
    if response is None:
        return None
    try:
        return provisional_skill_prediction(response.get("content", ""))
    except SkillPredictionError:
        return None


def build_prediction_result_records(
    prompts: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    latest_raw: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one public result envelope for every prediction prompt.

    ``skill-prediction-v1`` remains restricted to exact target coordinates.  The
    result envelope is the lossless downstream contract for recovered,
    provisional, validation-failed, and runtime-failed outcomes.
    """

    predictions_by_idx = {item["idx"]: item for item in predictions}
    failures_by_idx = {item["idx"]: item for item in failures}
    if len(predictions_by_idx) != len(predictions):
        raise SkillPredictionError("Prediction records contain duplicate idx values")
    if len(failures_by_idx) != len(failures):
        raise SkillPredictionError("Prediction failure records contain duplicate idx values")
    if set(predictions_by_idx) & set(failures_by_idx):
        raise SkillPredictionError("A target cannot have both a prediction and a failure")

    results: list[dict[str, Any]] = []
    for prompt in prompts:
        idx = prompt["idx"]
        prediction = predictions_by_idx.get(idx)
        failure = failures_by_idx.get(idx)
        raw = latest_raw.get(idx)
        if prediction is None and failure is None:
            raise SkillPredictionError(f"Prediction result is missing for idx={idx}")
        if prediction is not None:
            recovery = prediction.get("recovery")
            outcome = "recovered" if recovery is not None else "exact"
            response = raw.get("response") if isinstance(raw, dict) else None
            model_output = (
                response.get("content") if isinstance(response, dict) else None
            )
            provisional = _provisional_or_none(response) if recovery is not None else None
            result = {
                "schema_version": RESULT_SCHEMA,
                "branch": prompt["branch"],
                "dataset_id": prompt["dataset_id"],
                "record_id": prompt["record_id"],
                "source_sha256": prompt["source_sha256"],
                "idx": idx,
                "sentence": prompt["sentence"],
                "outcome": outcome,
                "validation_issue": recovery is not None,
                "coordinate_space": "target_sentence",
                "prediction": prediction,
                "model_output": model_output if recovery is not None else None,
                "provisional_extraction": provisional,
                "failure": None,
                "review_reasons": list(prediction["review_reasons"]),
            }
        else:
            assert failure is not None
            eligible = failure.get("tolerance_eligible") is True
            provisional = failure.get("provisional_extraction")
            if eligible and provisional is not None:
                outcome = "provisional"
                coordinate_space = "model_sentence"
            elif eligible:
                outcome = "validation_failed"
                coordinate_space = None
            elif failure.get("failure_kind") == "missing_raw":
                outcome = "missing"
                coordinate_space = None
            else:
                outcome = "runtime_failed"
                coordinate_space = None
            reasons = list(prompt.get("review_reasons", []))
            reason = (
                "prediction_validation_failed"
                if eligible
                else f"prediction_{failure.get('failure_kind', 'failed')}"
            )
            if reason not in reasons:
                reasons.append(reason)
            result = {
                "schema_version": RESULT_SCHEMA,
                "branch": prompt["branch"],
                "dataset_id": prompt["dataset_id"],
                "record_id": prompt["record_id"],
                "source_sha256": prompt["source_sha256"],
                "idx": idx,
                "sentence": prompt["sentence"],
                "outcome": outcome,
                "validation_issue": eligible,
                "coordinate_space": coordinate_space,
                "prediction": None,
                "model_output": failure.get("model_output"),
                "provisional_extraction": provisional,
                "failure": failure,
                "review_reasons": reasons,
            }
        results.append(result)
    return results


def unavailable_prediction_result(
    record: dict[str, Any],
    *,
    branch: str,
    outcome: str,
    failure_kind: str,
    review_reasons: list[str],
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Represent a target that could not reach the final prediction stage."""

    if outcome not in {"runtime_failed", "missing"}:
        raise SkillPredictionError("Unavailable prediction outcome must be blocking")
    sentence = record.get("sentence")
    if not isinstance(sentence, str) or not sentence:
        raise SkillPredictionError("Unavailable prediction result needs a sentence")
    return {
        "schema_version": RESULT_SCHEMA,
        "branch": branch,
        "dataset_id": record["dataset_id"],
        "record_id": record["record_id"],
        "source_sha256": record["source_sha256"],
        "idx": record["idx"],
        "sentence": sentence,
        "outcome": outcome,
        "validation_issue": False,
        "coordinate_space": None,
        "prediction": None,
        "model_output": None,
        "provisional_extraction": None,
        "failure": {
            "failure_kind": failure_kind,
            "details": failure,
        },
        "review_reasons": list(dict.fromkeys(review_reasons)),
    }


def prediction_validation_issue_events(
    results: list[dict[str, Any]], *, stage: str
) -> list[dict[str, Any]]:
    """Convert terminal prediction validation issues into mergeable events."""

    events: list[dict[str, Any]] = []
    for result in results:
        if result.get("validation_issue") is not True:
            continue
        failure = result.get("failure")
        events.append(
            {
                "dataset_id": result["dataset_id"],
                "record_id": result["record_id"],
                "source_sha256": result["source_sha256"],
                "idx": result["idx"],
                "sentence": result["sentence"],
                "stage": stage,
                "outcome": result["outcome"],
                "coordinate_space": result["coordinate_space"],
                "retained": True,
                "model_output": result.get("model_output"),
                "validation_error": (
                    failure.get("validation_error")
                    if isinstance(failure, dict)
                    else result.get("prediction", {}).get("recovery", {}).get(
                        "strict_validation_error"
                    )
                ),
            }
        )
    return events


def merge_validation_issue_records(
    expected_records: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate validation issues by target identity while retaining every stage event."""

    expected_by_idx = {item["idx"]: item for item in expected_records}
    if len(expected_by_idx) != len(expected_records):
        raise SkillPredictionError("Expected records contain duplicate idx values")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        idx = event.get("idx")
        expected = expected_by_idx.get(idx)
        if expected is None:
            raise SkillPredictionError(f"Validation issue has unexpected idx={idx}")
        for key in ("dataset_id", "record_id", "source_sha256", "sentence"):
            if event.get(key) != expected.get(key):
                raise SkillPredictionError(
                    f"Validation issue {key} mismatch for idx={idx}"
                )
        grouped.setdefault(idx, []).append(event)

    records: list[dict[str, Any]] = []
    for expected in expected_records:
        target_events = grouped.get(expected["idx"])
        if not target_events:
            continue
        records.append(
            {
                "schema_version": VALIDATION_ISSUE_SCHEMA,
                "dataset_id": expected["dataset_id"],
                "record_id": expected["record_id"],
                "source_sha256": expected["source_sha256"],
                "idx": expected["idx"],
                "sentence": expected["sentence"],
                "stages": list(dict.fromkeys(item["stage"] for item in target_events)),
                "event_count": len(target_events),
                "events": target_events,
            }
        )
    return records


def evaluate_result_completion(
    *,
    expected_count: int,
    results: list[dict[str, Any]],
    validation_issues: list[dict[str, Any]],
    maximum_failure_rate: float,
) -> dict[str, Any]:
    """Evaluate the 3% policy over unique affected targets, not attempts or stages."""

    if expected_count < 1:
        raise SkillPredictionError("Result completion needs at least one target")
    if not 0.0 <= maximum_failure_rate <= 1.0:
        raise SkillPredictionError("Result maximum failure rate must be within [0, 1]")
    indexes = [item.get("idx") for item in results]
    unique_result_count = len(set(indexes))
    issue_indexes = [item.get("idx") for item in validation_issues]
    if len(issue_indexes) != len(set(issue_indexes)):
        raise SkillPredictionError("Validation issue records must be unique by target")
    issue_count = len(validation_issues)
    maximum_failure_count = math.floor(expected_count * maximum_failure_rate + 1e-12)
    blocking = [
        item
        for item in results
        if item.get("outcome") in {"runtime_failed", "missing"}
    ]
    outcome_counts = Counter(item.get("outcome") for item in results)
    within_tolerance = (
        len(results) == expected_count
        and unique_result_count == expected_count
        and not blocking
        and issue_count <= maximum_failure_count
    )
    return {
        "expected_count": expected_count,
        "result_count": len(results),
        "unique_result_count": unique_result_count,
        "validation_issue_count": issue_count,
        "validation_issue_rate": issue_count / expected_count,
        "failure_count": issue_count,
        "failure_rate": issue_count / expected_count,
        "maximum_failure_rate": maximum_failure_rate,
        "maximum_failure_count": maximum_failure_count,
        "allowed_failure_count": maximum_failure_count,
        "blocking_failure_count": len(blocking),
        "blocking_indexes": [item["idx"] for item in blocking],
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "within_tolerance": within_tolerance,
    }


def run_skill_predictions(
    prompts: list[dict[str, Any]],
    raw_path: Path,
    *,
    chat_model: str,
    temperature: float,
    max_tokens: int,
    maximum_repairs: int,
    allow_network: bool,
    retry_failed: bool,
    client_factory: Callable[[], Any] | None,
    secrets: list[str],
    on_network_call: Callable[[], None] | None = None,
) -> dict[int, dict[str, Any]]:
    """Run one deterministic prediction per prompt with bounded strict repairs."""

    latest = latest_prediction_raw_records(raw_path)
    prompts_by_idx = {item["idx"]: item for item in prompts}
    successful = {
        idx
        for idx, record in latest.items()
        if record.get("status") == "complete"
        and idx in prompts_by_idx
        and record.get("prompt_sha256") == prompts_by_idx[idx]["prompt_sha256"]
    }
    failed = {
        idx
        for idx, record in latest.items()
        if record.get("status") == "failed"
        and idx in prompts_by_idx
        and record.get("prompt_sha256") == prompts_by_idx[idx]["prompt_sha256"]
    }
    pending = [
        prompt
        for prompt in prompts
        if prompt["idx"] not in successful
        and (prompt["idx"] not in failed or retry_failed)
    ]
    if pending and (not allow_network or client_factory is None):
        raise SkillPredictionNetworkRequired(
            f"{len(pending)} target skill predictions are pending; network permission is required"
        )

    client: Any | None = None
    for prompt in pending:
        idx = prompt["idx"]
        response: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        recovery: dict[str, Any] | None = None
        try:
            if client is None:
                client = client_factory()
                if on_network_call:
                    on_network_call()
            messages = prompt["messages"]
            for attempt_number in range(1, maximum_repairs + 2):
                response = client.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=chat_model,
                )
                safe_response = redact_secrets(response, secrets)
                try:
                    parse_skill_prediction(
                        response.get("content", ""), prompt["sentence"]
                    )
                except SkillPredictionError as parse_error:
                    error_value = _error_value(parse_error, secrets)
                    attempts.append(
                        {
                            "attempt": attempt_number,
                            "kind": "initial" if attempt_number == 1 else "repair",
                            "request_sha256": sha256_json(messages),
                            "response": safe_response,
                            "parse_error": error_value,
                        }
                    )
                    if attempt_number > maximum_repairs:
                        try:
                            recovery = recover_skill_prediction(
                                safe_response.get("content", ""),
                                prompt["sentence"],
                                error_value,
                            )
                        except SkillPredictionError:
                            raise parse_error
                        break
                    messages = _repair_messages(
                        prompt,
                        safe_response.get("content", ""),
                        error_value,
                        attempt_number,
                    )
                    continue
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "kind": "initial" if attempt_number == 1 else "repair",
                        "request_sha256": sha256_json(messages),
                        "response": safe_response,
                        "parse_error": None,
                    }
                )
                break
            raw_record: dict[str, Any] = {
                "schema_version": RAW_SCHEMA,
                "branch": prompt["branch"],
                "dataset_id": prompt["dataset_id"],
                "record_id": prompt["record_id"],
                "source_sha256": prompt["source_sha256"],
                "idx": idx,
                "status": "complete",
                "prompt_sha256": prompt["prompt_sha256"],
                "model": chat_model,
                "response": redact_secrets(response, secrets),
                "error": None,
                "created_at": utc_now(),
            }
            if len(attempts) > 1:
                raw_record["repair"] = {
                    "schema_version": REPAIR_SCHEMA,
                    "attempts": attempts,
                }
            if recovery is not None:
                raw_record["recovery"] = recovery
        except Exception as error:
            raw_record = {
                "schema_version": RAW_SCHEMA,
                "branch": prompt["branch"],
                "dataset_id": prompt["dataset_id"],
                "record_id": prompt["record_id"],
                "source_sha256": prompt["source_sha256"],
                "idx": idx,
                "status": "failed",
                "prompt_sha256": prompt["prompt_sha256"],
                "model": chat_model,
                "response": redact_secrets(response, secrets) if response else None,
                "error": _error_value(error, secrets),
                "created_at": utc_now(),
            }
            if attempts:
                raw_record["repair"] = {
                    "schema_version": REPAIR_SCHEMA,
                    "attempts": attempts,
                }
        append_jsonl(raw_path, raw_record)
        latest[idx] = raw_record
    return latest


def parse_successful_prediction_records(
    prompts: list[dict[str, Any]],
    latest_raw: dict[int, dict[str, Any]],
    *,
    chat_model: str,
    maximum_repairs: int,
) -> list[dict[str, Any]]:
    """Rebuild public prediction records from prompt and append-only raw evidence."""

    predictions: list[dict[str, Any]] = []
    for prompt in prompts:
        raw = latest_raw.get(prompt["idx"])
        if not raw or raw.get("status") != "complete":
            continue
        for key in ("branch", "dataset_id", "record_id", "source_sha256", "idx"):
            if raw.get(key) != prompt.get(key):
                raise SkillPredictionError(
                    f"Prediction raw {key} mismatch for idx={prompt['idx']}"
                )
        if raw.get("schema_version") != RAW_SCHEMA:
            raise SkillPredictionError(
                f"Unsupported prediction raw schema for idx={prompt['idx']}"
            )
        if raw.get("prompt_sha256") != prompt["prompt_sha256"]:
            raise SkillPredictionError(
                f"Prediction prompt hash mismatch for idx={prompt['idx']}"
            )
        response = raw.get("response")
        if not isinstance(response, dict):
            raise SkillPredictionError(
                f"Prediction raw response is missing for idx={prompt['idx']}"
            )
        _validate_repair_audit(raw, prompt, maximum_repairs)
        recovery = raw.get("recovery")
        if recovery is not None:
            annotation = recovery["annotation"]
        else:
            annotation = parse_skill_prediction(
                response.get("content", ""), prompt["sentence"]
            )
        reasons = list(prompt["review_reasons"])
        if recovery is not None and RECOVERY_REVIEW_REASON not in reasons:
            reasons.append(RECOVERY_REVIEW_REASON)
        prediction = {
            "schema_version": PREDICTION_SCHEMA,
            "branch": prompt["branch"],
            "dataset_id": prompt["dataset_id"],
            "record_id": prompt["record_id"],
            "source_sha256": prompt["source_sha256"],
            "idx": prompt["idx"],
            "sentence": prompt["sentence"],
            "status": "needs_review" if reasons else "complete",
            "has_skill": annotation["has_skill"],
            "spans": annotation["spans"],
            "review_reasons": reasons,
            "evidence": prompt["evidence"],
            "models": {"chat": chat_model},
            "prompt_sha256": prompt["prompt_sha256"],
        }
        if recovery is not None:
            prediction["recovery"] = recovery
        predictions.append(prediction)
    return predictions
