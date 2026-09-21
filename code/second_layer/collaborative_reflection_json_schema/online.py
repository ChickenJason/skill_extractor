"""Strict JSON-Schema calls with append-only initial/repair audit chains."""

from __future__ import annotations

from typing import Any

from common.io_utils import utc_now
from second_layer.collaborative_reflection.common import ReflectionError
from second_layer.collaborative_reflection_json_schema.pipeline import (
    build_repair_messages,
    parse_exemplar_response,
    parse_trf_response,
    request_sha256,
)


def reflect_one(
    *, branch: str, record: dict[str, Any], prompt: dict[str, Any],
    client: Any, chat: dict[str, Any], max_repairs: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if branch == "trf":
        parser = parse_trf_response
    elif branch == "exemplar":
        parser = parse_exemplar_response
    else:
        raise ReflectionError(f"Unknown reflection branch: {branch}")
    attempts: list[dict[str, Any]] = []
    messages = prompt["messages"]
    response_format = prompt["response_format"]
    prior_content = ""
    parse_error: str | None = None
    for attempt in range(1, max_repairs + 2):
        kind = "initial" if attempt == 1 else "repair"
        if attempt > 1:
            assert parse_error is not None
            messages = build_repair_messages(
                prompt, branch, parse_error, prior_content, chat["max_reason_characters"]
            )
        digest = request_sha256(messages, response_format)
        try:
            response = client.chat(
                messages,
                temperature=chat["temperature"],
                max_tokens=chat["max_tokens"],
                model=chat["model"],
                response_format=response_format,
            )
        except Exception as error:
            attempts.append(
                {
                    "attempt": attempt,
                    "kind": kind,
                    "request_sha256": digest,
                    "response": None,
                    "parse_error": None,
                    "transport_error": str(error),
                    "transport_attempts": getattr(error, "attempts", None),
                }
            )
            return _raw(record, prompt, branch, "failed", attempts, None, str(error)), None
        prior_content = response.get("content", "")
        try:
            parsed = parser(
                prior_content,
                record,
                max_reason_characters=chat["max_reason_characters"],
            )
            parse_error = None
        except Exception as error:
            parsed = None
            parse_error = str(error)
        attempts.append(
            {
                "attempt": attempt,
                "kind": kind,
                "request_sha256": digest,
                "response": response,
                "parse_error": parse_error,
                "transport_error": None,
                "transport_attempts": response.get("attempts"),
            }
        )
        if parsed is not None:
            return _raw(record, prompt, branch, "complete", attempts, response, None), parsed
    return _raw(
        record, prompt, branch, "failed", attempts,
        attempts[-1].get("response"), parse_error,
    ), None


def _raw(
    record: dict[str, Any], prompt: dict[str, Any], branch: str, status: str,
    attempts: list[dict[str, Any]], final_response: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "collaborative-reflection-raw-v2",
        "branch": branch,
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
        "final_response": final_response,
        "error": error,
        "created_at": utc_now(),
    }
