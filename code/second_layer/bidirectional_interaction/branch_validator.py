"""Read-only deterministic validation of a completed round branch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common.io_utils import load_json, read_jsonl, sha256_file
from second_layer.bidirectional_interaction.common import InteractionError, load_config
from second_layer.bidirectional_interaction.contracts import parse_exemplar_proposal, parse_trf_proposal
from second_layer.bidirectional_interaction.prompts import (
    build_prompt,
    build_repair_messages,
    request_sha256,
)


def _identity(item: dict[str, Any]) -> tuple[str, str]:
    return item["dataset_id"], item["record_id"]


def _latest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for item in read_jsonl(path):
        output[_identity(item)] = item
    return output


def _cumulative_metrics(raw_records: list[dict[str, Any]]) -> dict[str, int]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    values = {
        "model_responses": 0,
        "repair_responses": 0,
        "failed_records": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    for raw in raw_records:
        latest[_identity(raw)] = raw
        for attempt in raw.get("attempts", []):
            response = attempt.get("response")
            if response is None:
                continue
            values["model_responses"] += 1
            if attempt.get("kind") == "repair":
                values["repair_responses"] += 1
            usage = response.get("usage", {})
            values["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            values["completion_tokens"] += int(usage.get("completion_tokens", 0))
    values["failed_records"] = sum(
        raw.get("status") == "failed" for raw in latest.values()
    )
    return values


def validate_branch(
    *,
    config_path: Path,
    parent_root: Path,
    round_number: int,
    branch: str,
) -> dict[str, Any]:
    config = load_config(config_path)
    root = parent_root / "rounds" / f"round-{round_number:02d}"
    branch_root = root / "branches" / branch
    input_path = root / "inputs" / "records.jsonl"
    prompt_path = branch_root / "prompts" / "records.jsonl"
    raw_path = branch_root / "raw" / "records.jsonl"
    parsed_path = branch_root / "parsed" / "proposals.jsonl"
    manifest = load_json(branch_root / "manifest.json")
    if manifest.get("status") != "completed":
        raise InteractionError("Only completed interaction branches can validate")
    if manifest.get("input_sha256") != sha256_file(input_path) or manifest.get("prompt_sha256") != sha256_file(prompt_path):
        raise InteractionError("Branch manifest frozen hashes mismatch")
    inputs = read_jsonl(input_path)
    prompts = read_jsonl(prompt_path)
    prompt_by_id = {_identity(item): item for item in prompts}
    raw_records = read_jsonl(raw_path)
    latest = _latest(raw_path)
    if set(latest) != {_identity(item) for item in inputs}:
        raise InteractionError("Branch raw audit does not exactly cover inputs")
    if manifest.get("metrics") != _cumulative_metrics(raw_records):
        raise InteractionError("Branch cumulative call/token metrics do not rebuild from raw audit")
    rebuilt_proposals: list[dict[str, Any]] = []
    for record in inputs:
        identity = _identity(record)
        expected_prompt = build_prompt(
            record,
            branch,
            record["previous_r"],
            record["previous_h"],
            round_number,
            max_characters=config["chat"]["max_prompt_characters"],
            max_reason_characters=config["chat"]["max_reason_characters"],
        )
        if prompt_by_id.get(identity) != expected_prompt:
            raise InteractionError(f"Branch prompt reconstruction mismatch for {identity!r}")
        raw = latest[identity]
        if raw.get("status") != "complete":
            raise InteractionError(f"Branch target is not complete for {identity!r}")
        for key in ("dataset_id", "record_id", "source_sha256", "idx", "sentence"):
            if raw.get(key) != record.get(key):
                raise InteractionError(f"Raw identity mismatch for {identity!r}")
        if raw.get("prompt_sha256") != expected_prompt["prompt_sha256"]:
            raise InteractionError(f"Raw prompt hash mismatch for {identity!r}")
        if raw.get("response_format_sha256") != expected_prompt["response_format_sha256"]:
            raise InteractionError(f"Raw schema hash mismatch for {identity!r}")
        if raw.get("request_contract_sha256") != expected_prompt["request_contract_sha256"]:
            raise InteractionError(f"Raw request contract hash mismatch for {identity!r}")
        attempts = raw.get("attempts")
        if not isinstance(attempts, list) or not 1 <= len(attempts) <= 3:
            raise InteractionError(f"Invalid repair audit length for {identity!r}")
        parser = parse_trf_proposal if branch == "trf" else parse_exemplar_proposal
        messages = expected_prompt["messages"]
        rebuilt: dict[str, Any] | None = None
        for position, attempt in enumerate(attempts, start=1):
            expected_kind = "initial" if position == 1 else "repair"
            if attempt.get("attempt") != position or attempt.get("kind") != expected_kind:
                raise InteractionError(f"Repair attempt order mismatch for {identity!r}")
            if attempt.get("request_sha256") != request_sha256(messages, expected_prompt["response_format"]):
                raise InteractionError(f"Repair request hash mismatch for {identity!r}")
            response = attempt.get("response")
            if not isinstance(response, dict) or attempt.get("transport_error") is not None:
                raise InteractionError(f"Completed audit contains a transport failure for {identity!r}")
            try:
                if branch == "trf":
                    parsed = parser(
                        response.get("content", ""),
                        record,
                        record["previous_r"],
                        max_reason=config["chat"]["max_reason_characters"],
                    )
                else:
                    parsed = parser(
                        response.get("content", ""),
                        record,
                        record["previous_r"],
                        record["previous_h"],
                        max_reason=config["chat"]["max_reason_characters"],
                    )
            except Exception as error:
                if position == len(attempts) or attempt.get("parse_error") != str(error):
                    raise InteractionError(f"Repair parse-error audit mismatch for {identity!r}") from error
                messages = build_repair_messages(expected_prompt, str(error), response.get("content", ""))
            else:
                if position != len(attempts) or attempt.get("parse_error") is not None:
                    raise InteractionError(f"Repair success position mismatch for {identity!r}")
                rebuilt = parsed
        if rebuilt is None:
            raise InteractionError(f"Completed branch has no valid final proposal for {identity!r}")
        if rebuilt != raw.get("proposal"):
            raise InteractionError(f"Parsed proposal audit mismatch for {identity!r}")
        rebuilt_proposals.append(rebuilt)
    if read_jsonl(parsed_path) != rebuilt_proposals:
        raise InteractionError("Materialized branch proposals do not rebuild from append-only raw")
    started = manifest.get("last_started_unix_ns")
    completed = manifest.get("completed_unix_ns")
    if type(started) is not int or type(completed) is not int or completed < started:
        raise InteractionError("Branch nanosecond process interval is invalid")
    return {
        "schema_version": "bidirectional-interaction-branch-validation-v1",
        "branch": branch,
        "round": round_number,
        "status": "valid",
        "target_count": len(inputs),
        "input_sha256": sha256_file(input_path),
        "prompt_sha256": sha256_file(prompt_path),
        "raw_sha256": sha256_file(raw_path),
        "parsed_sha256": sha256_file(parsed_path),
        "started_unix_ns": started,
        "completed_unix_ns": completed,
    }
