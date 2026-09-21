"""Strict proposal requests with an append-only initial/repair audit chain."""

from __future__ import annotations

from typing import Any, Callable

from common.io_utils import redact_secrets, utc_now
from second_layer.bidirectional_interaction.common import InteractionError
from second_layer.bidirectional_interaction.contracts import parse_exemplar_proposal, parse_trf_proposal
from second_layer.bidirectional_interaction.prompts import build_repair_messages, request_sha256


def _raw(
    *,
    record: dict[str, Any],
    prompt: dict[str, Any],
    branch: str,
    status: str,
    attempts: list[dict[str, Any]],
    proposal: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "bidirectional-interaction-raw-v1",
        "branch": branch,
        "round": prompt["round"],
        "dataset_id": record["dataset_id"],
        "record_id": record["record_id"],
        "source_sha256": record["source_sha256"],
        "idx": record["idx"],
        "sentence": record["sentence"],
        "prompt_sha256": prompt["prompt_sha256"],
        "response_format_sha256": prompt["response_format_sha256"],
        "request_contract_sha256": prompt["request_contract_sha256"],
        "status": status,
        "attempts": attempts,
        "proposal": proposal,
        "error": error,
        "created_at": utc_now(),
    }


def reflect_one(
    *,
    branch: str,
    record: dict[str, Any],
    previous_r: dict[str, Any],
    previous_h: dict[str, Any],
    prompt: dict[str, Any],
    client: Any,
    chat: dict[str, Any],
    max_repairs: int,
    secrets: list[str],
    before_call: Callable[[], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    parser = parse_trf_proposal if branch == "trf" else parse_exemplar_proposal
    if branch not in {"trf", "exemplar"}:
        raise InteractionError(f"Unknown interaction branch: {branch}")
    messages = prompt["messages"]
    response_format = prompt["response_format"]
    attempts: list[dict[str, Any]] = []
    prior_content = ""
    parse_error: str | None = None
    for attempt_number in range(1, max_repairs + 2):
        kind = "initial" if attempt_number == 1 else "repair"
        if attempt_number > 1:
            assert parse_error is not None
            messages = build_repair_messages(prompt, parse_error, prior_content)
        digest = request_sha256(messages, response_format)
        try:
            if before_call:
                before_call()
            response = client.chat(
                messages,
                temperature=chat["temperature"],
                max_tokens=chat["max_tokens"],
                model=chat["model"],
                response_format=response_format,
            )
            safe_response = redact_secrets(response, secrets)
        except Exception as error:
            safe_error = str(redact_secrets(str(error), secrets))
            attempts.append(
                {
                    "attempt": attempt_number,
                    "kind": kind,
                    "request_sha256": digest,
                    "response": None,
                    "parse_error": None,
                    "transport_error": safe_error,
                    "transport_attempts": getattr(error, "attempts", None),
                }
            )
            return (
                _raw(
                    record=record,
                    prompt=prompt,
                    branch=branch,
                    status="failed",
                    attempts=attempts,
                    proposal=None,
                    error=safe_error,
                ),
                None,
            )
        prior_content = safe_response.get("content", "")
        try:
            if branch == "trf":
                proposal = parser(
                    prior_content,
                    record,
                    previous_r,
                    max_reason=chat["max_reason_characters"],
                )
            else:
                proposal = parser(
                    prior_content,
                    record,
                    previous_r,
                    previous_h,
                    max_reason=chat["max_reason_characters"],
                )
            parse_error = None
        except Exception as error:
            proposal = None
            parse_error = str(error)
        attempts.append(
            {
                "attempt": attempt_number,
                "kind": kind,
                "request_sha256": digest,
                "response": safe_response,
                "parse_error": parse_error,
                "transport_error": None,
                "transport_attempts": safe_response.get("attempts"),
            }
        )
        if proposal is not None:
            return (
                _raw(
                    record=record,
                    prompt=prompt,
                    branch=branch,
                    status="complete",
                    attempts=attempts,
                    proposal=proposal,
                    error=None,
                ),
                proposal,
            )
    return (
        _raw(
            record=record,
            prompt=prompt,
            branch=branch,
            status="failed",
            attempts=attempts,
            proposal=None,
            error=parse_error,
        ),
        None,
    )
