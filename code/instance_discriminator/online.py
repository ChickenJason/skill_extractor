"""Append-only Qwen judgments and deterministic raw-response reconstruction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import (  # noqa: E402
    append_jsonl,
    redact_secrets,
    sha256_json,
    utc_now,
)
from common.qwen_client import QwenClient  # noqa: E402
from instance_discriminator.common import (  # noqa: E402
    InstanceDiscriminatorError,
    latest_records,
)
from instance_discriminator.pipeline import (  # noqa: E402
    build_parsed_record,
    parse_judgments,
)


class NetworkRequiredError(InstanceDiscriminatorError):
    """Raised when pending targets require explicit network permission."""


MAX_STRUCTURED_REPAIR_ATTEMPTS = 2


def _error_value(error: Exception, secrets: list[str]) -> dict[str, str]:
    return redact_secrets(
        {"type": error.__class__.__name__, "message": str(error)}, secrets
    )


def _repair_messages(
    prompt: dict[str, Any],
    record: dict[str, Any],
    invalid_content: str,
    parse_error: dict[str, str],
    repair_round: int,
) -> list[dict[str, str]]:
    role_contract = [
        {
            "demo_idx": item["demo_idx"],
            "status": item["status"],
            "allowed_roles": (
                ["supporting", "irrelevant"]
                if item["status"] == "accepted"
                else ["contrastive", "irrelevant"]
            ),
        }
        for item in record["candidates"]
    ]
    correction = {
        "repair_round": repair_round,
        "prior_validation_error": parse_error,
        "instruction": (
            "Replace the entire prior response with one corrected JSON object. "
            "Do not explain the correction and do not add markdown."
        ),
        "required_demo_indexes_in_order": [
            item["demo_idx"] for item in record["candidates"]
        ],
        "role_contract": role_contract,
        "hard_requirements": [
            "Return exactly one judgment for every required demo_idx.",
            "Copy every demo_idx exactly once; never duplicate or omit an ID.",
            "Use only the allowed role listed for that demo_idx.",
            "Return an object with only the judgments key.",
            (
                "Each judgment must contain exactly demo_idx, helpfulness_score, role, "
                "and reason_codes; never return a reason field or prose explanation."
            ),
        ],
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
    record: dict[str, Any],
    *,
    allow_terminal_failure: bool = False,
) -> None:
    repair = raw.get("repair")
    if repair is None:
        if allow_terminal_failure:
            raise InstanceDiscriminatorError(
                f"Terminal judgment failure has no complete repair audit for idx={record['idx']}"
            )
        return
    if not isinstance(repair, dict) or set(repair) != {"schema_version", "attempts"}:
        raise InstanceDiscriminatorError(
            f"Invalid repair audit structure for idx={record['idx']}"
        )
    if repair.get("schema_version") != "instance-discriminator-repair-v2":
        raise InstanceDiscriminatorError(
            f"Unsupported repair audit schema for idx={record['idx']}"
        )
    attempts = repair.get("attempts")
    if (
        not isinstance(attempts, list)
        or not 2 <= len(attempts) <= MAX_STRUCTURED_REPAIR_ATTEMPTS + 1
    ):
        raise InstanceDiscriminatorError(
            f"Invalid repair attempt count for idx={record['idx']}"
        )

    expected_messages = prompt["messages"]
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
            raise InstanceDiscriminatorError(
                f"Invalid repair attempt audit for idx={record['idx']} attempt={position}"
            )
        response = attempt["response"]
        try:
            parse_judgments(
                response.get("content", ""),
                record["candidates"],
            )
        except InstanceDiscriminatorError as error:
            actual_error = {"type": error.__class__.__name__, "message": str(error)}
            if attempt.get("parse_error") != actual_error:
                raise InstanceDiscriminatorError(
                    f"Repair error audit mismatch for idx={record['idx']} attempt={position}"
                ) from error
            if position == len(attempts):
                if (
                    not allow_terminal_failure
                    or len(attempts) != MAX_STRUCTURED_REPAIR_ATTEMPTS + 1
                    or raw.get("status") != "failed"
                    or response != raw.get("response")
                    or raw.get("error") != actual_error
                ):
                    raise InstanceDiscriminatorError(
                        f"Repair ended invalidly for idx={record['idx']} attempt={position}"
                    ) from error
                continue
            expected_messages = _repair_messages(
                prompt,
                record,
                response.get("content", ""),
                actual_error,
                position,
            )
        else:
            if position != len(attempts) or attempt.get("parse_error") is not None:
                raise InstanceDiscriminatorError(
                    f"Repair success audit mismatch for idx={record['idx']} attempt={position}"
                )
            if response != raw.get("response"):
                raise InstanceDiscriminatorError(
                    f"Final repaired response mismatch for idx={record['idx']}"
                )
            if raw.get("status") != "complete" or raw.get("error") is not None:
                raise InstanceDiscriminatorError(
                    f"Repair success has invalid terminal state for idx={record['idx']}"
                )


def run_judgments(
    candidate_records: list[dict[str, Any]],
    prompts_by_idx: dict[int, dict[str, Any]],
    raw_path: Path,
    *,
    chat_model: str,
    temperature: float,
    max_tokens: int,
    allow_network: bool,
    retry_failed: bool,
    client_factory: Callable[[], QwenClient] | None,
    secrets: list[str],
    on_network_call: Callable[[], None] | None = None,
) -> dict[int, dict[str, Any]]:
    latest = latest_records(raw_path)
    expected_ids = {item["idx"] for item in candidate_records}
    unexpected = set(latest) - expected_ids
    if unexpected:
        raise InstanceDiscriminatorError(
            f"Raw response log contains unexpected target IDs: {sorted(unexpected)}"
        )
    successful = {
        idx for idx, record in latest.items() if record.get("status") == "complete"
    }
    failed = {idx for idx, record in latest.items() if record.get("status") == "failed"}
    pending = [
        item
        for item in candidate_records
        if item["idx"] not in successful
        and (item["idx"] not in failed or retry_failed)
    ]
    if pending and (not allow_network or client_factory is None):
        raise NetworkRequiredError(
            f"{len(pending)} target judgments are pending; --allow-network is required"
        )
    if not pending:
        return latest

    client = client_factory()
    if on_network_call:
        on_network_call()
    for record in pending:
        idx = record["idx"]
        prompt = prompts_by_idx[idx]
        response: dict[str, Any] | None = None
        repair_attempts: list[dict[str, Any]] = []
        try:
            messages = prompt["messages"]
            for attempt_number in range(1, MAX_STRUCTURED_REPAIR_ATTEMPTS + 2):
                response = client.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=chat_model,
                )
                safe_response = redact_secrets(response, secrets)
                try:
                    parse_judgments(
                        response.get("content", ""),
                        record["candidates"],
                    )
                except InstanceDiscriminatorError as parse_error:
                    error_value = _error_value(parse_error, secrets)
                    repair_attempts.append(
                        {
                            "attempt": attempt_number,
                            "kind": "initial" if attempt_number == 1 else "repair",
                            "request_sha256": sha256_json(messages),
                            "response": safe_response,
                            "parse_error": error_value,
                        }
                    )
                    if attempt_number > MAX_STRUCTURED_REPAIR_ATTEMPTS:
                        raise
                    messages = _repair_messages(
                        prompt,
                        record,
                        safe_response.get("content", ""),
                        error_value,
                        attempt_number,
                    )
                    continue
                repair_attempts.append(
                    {
                        "attempt": attempt_number,
                        "kind": "initial" if attempt_number == 1 else "repair",
                        "request_sha256": sha256_json(messages),
                        "response": safe_response,
                        "parse_error": None,
                    }
                )
                break
            raw_record = {
                "schema_version": "instance-discriminator-raw-v2",
                "dataset_id": record["dataset_id"],
                "record_id": record["record_id"],
                "source_sha256": record["source_sha256"],
                "idx": idx,
                "sentence": record["sentence"],
                "status": "complete",
                "prompt_sha256": prompt["prompt_sha256"],
                "model": chat_model,
                "response": redact_secrets(response, secrets),
                "error": None,
                "created_at": utc_now(),
            }
            if len(repair_attempts) > 1:
                raw_record["repair"] = {
                    "schema_version": "instance-discriminator-repair-v2",
                    "attempts": repair_attempts,
                }
        except Exception as error:
            raw_record = {
                "schema_version": "instance-discriminator-raw-v2",
                "dataset_id": record["dataset_id"],
                "record_id": record["record_id"],
                "source_sha256": record["source_sha256"],
                "idx": idx,
                "sentence": record["sentence"],
                "status": "failed",
                "prompt_sha256": prompt["prompt_sha256"],
                "model": chat_model,
                "response": redact_secrets(response, secrets) if response else None,
                "error": _error_value(error, secrets),
                "created_at": utc_now(),
            }
            if repair_attempts:
                raw_record["repair"] = {
                    "schema_version": "instance-discriminator-repair-v2",
                    "attempts": repair_attempts,
                }
        append_jsonl(raw_path, raw_record)
        latest[idx] = raw_record
    return latest


def parse_terminal_raw_records(
    candidate_records: list[dict[str, Any]],
    prompts_by_idx: dict[int, dict[str, Any]],
    latest_raw: dict[int, dict[str, Any]],
    *,
    chat_model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Retain terminal judgment validation failures as reviewable empty judgments."""

    parsed: list[dict[str, Any]] = []
    validation_events: list[dict[str, Any]] = []
    blockers: dict[int, dict[str, Any]] = {}
    for record in candidate_records:
        idx = record["idx"]
        raw = latest_raw.get(idx)
        if raw is None:
            blockers[idx] = {"failure_kind": "missing_raw", "error": None}
            continue
        if raw.get("schema_version") != "instance-discriminator-raw-v2":
            raise InstanceDiscriminatorError(f"Unsupported raw schema for idx={idx}")
        for key in ("dataset_id", "record_id", "source_sha256", "idx", "sentence"):
            if raw.get(key) != record.get(key):
                raise InstanceDiscriminatorError(f"Raw {key} mismatch for idx={idx}")
        if raw.get("prompt_sha256") != prompts_by_idx[idx]["prompt_sha256"]:
            raise InstanceDiscriminatorError(f"Raw prompt hash mismatch for idx={idx}")
        if raw.get("model") != chat_model:
            raise InstanceDiscriminatorError(f"Raw model mismatch for idx={idx}")
        status = raw.get("status")
        if status == "complete":
            response = raw.get("response")
            if not isinstance(response, dict):
                raise InstanceDiscriminatorError(f"Raw response is missing for idx={idx}")
            _validate_repair_audit(raw, prompts_by_idx[idx], record)
            judgments = parse_judgments(response.get("content", ""), record["candidates"])
            parsed.append(build_parsed_record(record, judgments, chat_model))
            continue
        if status != "failed":
            raise InstanceDiscriminatorError(f"Invalid raw status for idx={idx}")
        error = raw.get("error")
        if isinstance(error, dict) and error.get("type") == "InstanceDiscriminatorError":
            _validate_repair_audit(
                raw,
                prompts_by_idx[idx],
                record,
                allow_terminal_failure=True,
            )
            placeholder = build_parsed_record(record, [], chat_model)
            placeholder["status"] = "needs_review"
            placeholder["review_reasons"] = ["exemplar_judgment_validation_failed"]
            parsed.append(placeholder)
            validation_events.append(
                {
                    "dataset_id": record["dataset_id"],
                    "record_id": record["record_id"],
                    "source_sha256": record["source_sha256"],
                    "idx": idx,
                    "sentence": record["sentence"],
                    "stage": "exemplar_judgment",
                    "outcome": "validation_failed",
                    "coordinate_space": None,
                    "retained": True,
                    "model_output": (
                        raw.get("response", {}).get("content")
                        if isinstance(raw.get("response"), dict)
                        else None
                    ),
                    "validation_error": error,
                }
            )
            continue
        blockers[idx] = {
            "failure_kind": "provider_or_runtime_failure",
            "error": error,
        }
    return parsed, validation_events, blockers
