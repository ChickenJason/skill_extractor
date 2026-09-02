"""Append-only Qwen judgments and deterministic raw-response reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import append_jsonl, redact_secrets, utc_now  # noqa: E402
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


def run_judgments(
    candidate_records: list[dict[str, Any]],
    prompts_by_idx: dict[int, dict[str, Any]],
    raw_path: Path,
    *,
    chat_model: str,
    temperature: float,
    max_tokens: int,
    max_reason_characters: int,
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
        try:
            response = client.chat(
                prompt["messages"],
                temperature=temperature,
                max_tokens=max_tokens,
                model=chat_model,
            )
            parse_judgments(
                response.get("content", ""),
                record["candidates"],
                max_reason_characters=max_reason_characters,
            )
            raw_record = {
                "idx": idx,
                "sentence": record["sentence"],
                "status": "complete",
                "prompt_sha256": prompt["prompt_sha256"],
                "model": chat_model,
                "response": redact_secrets(response, secrets),
                "error": None,
                "created_at": utc_now(),
            }
        except Exception as error:
            raw_record = {
                "idx": idx,
                "sentence": record["sentence"],
                "status": "failed",
                "prompt_sha256": prompt["prompt_sha256"],
                "model": chat_model,
                "response": None,
                "error": redact_secrets(
                    {"type": error.__class__.__name__, "message": str(error)}, secrets
                ),
                "created_at": utc_now(),
            }
        append_jsonl(raw_path, raw_record)
        latest[idx] = raw_record
    return latest


def parse_successful_raw_records(
    candidate_records: list[dict[str, Any]],
    prompts_by_idx: dict[int, dict[str, Any]],
    latest_raw: dict[int, dict[str, Any]],
    *,
    chat_model: str,
    max_reason_characters: int,
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for record in candidate_records:
        idx = record["idx"]
        raw = latest_raw.get(idx)
        if raw is None or raw.get("status") != "complete":
            continue
        if raw.get("sentence") != record["sentence"]:
            raise InstanceDiscriminatorError(f"Raw sentence mismatch for idx={idx}")
        if raw.get("prompt_sha256") != prompts_by_idx[idx]["prompt_sha256"]:
            raise InstanceDiscriminatorError(f"Raw prompt hash mismatch for idx={idx}")
        if raw.get("model") != chat_model:
            raise InstanceDiscriminatorError(f"Raw model mismatch for idx={idx}")
        response = raw.get("response")
        if not isinstance(response, dict):
            raise InstanceDiscriminatorError(f"Raw response is missing for idx={idx}")
        judgments = parse_judgments(
            response.get("content", ""),
            record["candidates"],
            max_reason_characters=max_reason_characters,
        )
        parsed.append(build_parsed_record(record, judgments, chat_model))
    return parsed
