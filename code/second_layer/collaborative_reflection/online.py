"""Strict structured reflection calls with a bounded, append-only repair audit chain."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import sha256_json, utc_now  # noqa: E402
from second_layer.collaborative_reflection.common import ReflectionError  # noqa: E402
from second_layer.collaborative_reflection.pipeline import (  # noqa: E402
    build_repair_messages,
    parse_exemplar_response,
    parse_trf_response,
)


Parser = Callable[..., dict[str, Any]]


def _parser(branch: str) -> Parser:
    if branch == "trf":
        return parse_trf_response
    if branch == "exemplar":
        return parse_exemplar_response
    raise ReflectionError(f"Unknown reflection branch: {branch}")


def reflect_one(
    *,
    branch: str,
    record: dict[str, Any],
    prompt: dict[str, Any],
    client: Any,
    chat: dict[str, Any],
    max_repairs: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Call one reflector and keep every initial/repair response in one raw record."""

    attempts: list[dict[str, Any]] = []
    messages = prompt["messages"]
    prior_content = ""
    parse_error: str | None = None
    parser = _parser(branch)
    for attempt in range(1, max_repairs + 2):
        kind = "initial" if attempt == 1 else "repair"
        if attempt > 1:
            assert parse_error is not None
            messages = build_repair_messages(
                prompt,
                branch,
                parse_error,
                prior_content,
                chat["max_reason_characters"],
            )
        request_sha256 = sha256_json(messages)
        try:
            response = client.chat(
                messages,
                temperature=chat["temperature"],
                max_tokens=chat["max_tokens"],
                model=chat["model"],
            )
        except Exception as error:
            attempts.append(
                {
                    "attempt": attempt,
                    "kind": kind,
                    "request_sha256": request_sha256,
                    "response": None,
                    "parse_error": None,
                    "transport_error": str(error),
                    "transport_attempts": getattr(error, "attempts", None),
                }
            )
            return (
                {
                    "schema_version": "collaborative-reflection-raw-v1",
                    "branch": branch,
                    "dataset_id": record["dataset_id"],
                    "record_id": record["record_id"],
                    "source_sha256": record["source_sha256"],
                    "idx": record["idx"],
                    "sentence": record["sentence"],
                    "prompt_sha256": prompt["prompt_sha256"],
                    "status": "failed",
                    "attempts": attempts,
                    "final_response": None,
                    "error": str(error),
                    "created_at": utc_now(),
                },
                None,
            )
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
                "request_sha256": request_sha256,
                "response": response,
                "parse_error": parse_error,
                "transport_error": None,
                "transport_attempts": response.get("attempts"),
            }
        )
        if parsed is not None:
            return (
                {
                    "schema_version": "collaborative-reflection-raw-v1",
                    "branch": branch,
                    "dataset_id": record["dataset_id"],
                    "record_id": record["record_id"],
                    "source_sha256": record["source_sha256"],
                    "idx": record["idx"],
                    "sentence": record["sentence"],
                    "prompt_sha256": prompt["prompt_sha256"],
                    "status": "complete",
                    "attempts": attempts,
                    "final_response": response,
                    "error": None,
                    "created_at": utc_now(),
                },
                parsed,
            )
    return (
        {
            "schema_version": "collaborative-reflection-raw-v1",
            "branch": branch,
            "dataset_id": record["dataset_id"],
            "record_id": record["record_id"],
            "source_sha256": record["source_sha256"],
            "idx": record["idx"],
            "sentence": record["sentence"],
            "prompt_sha256": prompt["prompt_sha256"],
            "status": "failed",
            "attempts": attempts,
            "final_response": attempts[-1]["response"],
            "error": parse_error,
            "created_at": utc_now(),
        },
        None,
    )
